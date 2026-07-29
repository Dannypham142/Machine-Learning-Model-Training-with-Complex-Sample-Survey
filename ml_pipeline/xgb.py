"""
    python xgb.py --mode train /path/to/train.parquet
    python xgb.py --mode test  /path/to/test.parquet
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.xgboost
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier

TARGET = "Disability (recoded)_With a disability"
TARGET_NEG = "Disability (recoded)_Without a disability"
WEIGHT = "State population"
EXPERIMENT = "xgboost_survey_weights"
VARIANTS = ("standard", "survey_weighted")
BATCH = 50_000  # test-mode parquet row-batch size; keeps peak X to BATCH × n_features
SEED = 42
N_ITER = 20
VAL_FRAC = 0.1
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"file://{Path(__file__).resolve().parent / 'mlruns'}",
)

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("train", "test"), required=True)
p.add_argument("--threshold", type=float, default=None,
               help="test-mode decision threshold; default: each variant's "
                    "learned best_threshold (falls back to 0.5)")
p.add_argument("--n-iter", type=int, default=N_ITER,
               help="random-search trials per variant (default %(default)s)")
p.add_argument("input")
args = p.parse_args()

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)


class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc, unit="tree")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model


def score(y, pred, prob, sw=None):
    return {
        "accuracy": accuracy_score(y, pred, sample_weight=sw),
        "precision": precision_score(y, pred, sample_weight=sw, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=sw, zero_division=0),
        "f1": f1_score(y, pred, sample_weight=sw, zero_division=0),
        "roc_auc": roc_auc_score(y, prob, sample_weight=sw),
    }


def best_threshold(y, prob, sw=None):
    # rank by AUC
    prec, rec, thr = precision_recall_curve(y, prob, sample_weight=sw)
    if len(thr) == 0:
        return 0.5
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    return float(thr[int(np.argmax(f1[:-1]))])


def sample_params(rng, balanced_spw):
    use_class_weight = bool(rng.integers(2))
    return {
        "n_estimators": int(rng.integers(50, 301)),
        "max_depth": int(rng.integers(3, 11)),
        "learning_rate": float(10 ** rng.uniform(-2, -0.5)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "use_class_weight": use_class_weight,
        "scale_pos_weight": balanced_spw if use_class_weight else 1.0,
    }


def make_clf(params, callbacks=None):
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=params["scale_pos_weight"],
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        callbacks=callbacks,
    )


if args.mode == "train":
    df = pl.read_parquet(args.input)
    w = df[WEIGHT].to_numpy().astype(np.float64)
    X = df.drop([TARGET, TARGET_NEG, WEIGHT]).to_numpy()
    y = df[TARGET].to_numpy().astype(np.int8)

    balanced_spw = float((y == 0).sum()) / float(max(int((y == 1).sum()), 1))

    for variant in VARIANTS:
        Xtr, Xval, ytr, yval, wtr, wval = train_test_split(
            X, y, w, test_size=VAL_FRAC, stratify=y, random_state=SEED)
        tr_sw = wtr if variant == "survey_weighted" else None
        val_sw = wval if variant == "survey_weighted" else None

        rng = np.random.default_rng(SEED)
        with mlflow.start_run(run_name=f"{variant}_train"):
            best = None  # (val_auc, params, val_prob)
            for i in tqdm(range(args.n_iter), desc=f"search xgb {variant}", unit="trial"):
                params = sample_params(rng, balanced_spw)
                clf = make_clf(params).fit(Xtr, ytr, sample_weight=tr_sw)
                val_prob = clf.predict_proba(Xval)[:, 1]
                auc = roc_auc_score(yval, val_prob, sample_weight=val_sw)
                with mlflow.start_run(run_name=f"{variant}_trial_{i}", nested=True):
                    mlflow.log_params({
                        "variant": variant,
                        "trial": i,
                        "n_estimators": params["n_estimators"],
                        "max_depth": params["max_depth"],
                        "learning_rate": params["learning_rate"],
                        "subsample": params["subsample"],
                        "colsample_bytree": params["colsample_bytree"],
                        "use_class_weight": params["use_class_weight"],
                        "scale_pos_weight": params["scale_pos_weight"],
                    })
                    mlflow.log_metric("val_roc_auc", float(auc))
                if best is None or auc > best[0]:
                    best = (auc, params, val_prob)
            val_auc, params, val_prob = best
            thr = best_threshold(yval, val_prob, val_sw)

            mlflow.log_params({
                "variant": variant,
                "model": "XGBClassifier",
                "search": "random",
                "n_iter": args.n_iter,
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "use_class_weight": params["use_class_weight"],
                "scale_pos_weight": params["scale_pos_weight"],
                "best_threshold": thr,
                "tree_method": "hist",
                "weight_col": WEIGHT,
                "target": TARGET,
                "n_rows": len(y),
                "n_features": X.shape[1],
                "input": args.input,
            })
            mlflow.log_metric("val_roc_auc", float(val_auc))
            mlflow.log_metric("best_threshold", float(thr))

            sw_full = w if variant == "survey_weighted" else None
            clf = make_clf(params, callbacks=[
                TqdmCallback(total=params["n_estimators"], desc=f"train xgb {variant}")])
            clf = clf.fit(X, y, sample_weight=sw_full)

            prob = clf.predict_proba(X)[:, 1]
            pred = (prob >= thr).astype(np.int8)
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"train_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"train_weighted_{name}", float(val))

            mlflow.xgboost.log_model(clf, name=variant)

elif args.mode == "test":
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)

    loaded = {}
    for variant in VARIANTS:
        models = client.search_logged_models(
            experiment_ids=[exp.experiment_id],
            filter_string=f"name='{variant}'",
            order_by=[{"field_name": "creation_timestamp", "ascending": False}],
            max_results=1,
        )
        if not models:
            raise SystemExit(f"no logged model named {variant!r} in experiment {EXPERIMENT}")
        uri = f"models:/{models[0].model_id}"
        thr = (args.threshold if args.threshold is not None
               else float(models[0].params.get("best_threshold", 0.5)))
        loaded[variant] = (uri, mlflow.xgboost.load_model(uri), thr)

    ys, ws, probs = [], [], {v: [] for v in VARIANTS}
    n_features = 0
    pf = pq.ParquetFile(args.input)
    total_batches = (pf.metadata.num_rows + BATCH - 1) // BATCH
    for batch in tqdm(pf.iter_batches(batch_size=BATCH), total=total_batches,
                      desc="test xgb", unit="batch"):
        df_b = pl.from_arrow(pa.Table.from_batches([batch]))
        ys.append(df_b[TARGET].to_numpy().astype(np.int8))
        ws.append(df_b[WEIGHT].to_numpy().astype(np.float64))
        X_b = df_b.drop([TARGET, TARGET_NEG, WEIGHT]).to_numpy()
        n_features = X_b.shape[1]
        for v, (_, clf, _) in loaded.items():
            probs[v].append(clf.predict_proba(X_b)[:, 1])

    y = np.concatenate(ys)
    w = np.concatenate(ws)

    out_cols = {}
    for variant, (uri, _, thr) in loaded.items():
        prob = np.concatenate(probs[variant])
        pred = (prob >= thr).astype(np.int8)
        out_cols[f"predicted_proba_{variant}"] = prob
        out_cols[f"predicted_disability_{variant}"] = pred

        with mlflow.start_run(run_name=f"{variant}_test"):
            mlflow.log_params({
                "variant": variant,
                "model_uri": uri,
                "threshold": thr,
                "input": args.input,
                "n_rows": len(y),
                "n_features": n_features,
            })
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"test_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"test_weighted_{name}", float(val))

    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(args.input).stem}_labeled_xgboost.parquet"
    pl.DataFrame({TARGET: y, **out_cols}).write_parquet(out)
