"""
    python xgb.py --mode train /path/to/train.parquet
    python xgb.py --mode test  /path/to/test.parquet

Train fits a standard and a survey-weighted XGBoost classifier on input.
Test passes through data to output a labeled parquet file containing features and designed prediction labels for control and treatment group.
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
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier

TARGET = "Disability (recoded)_With a disability"
TARGET_NEG = "Disability (recoded)_Without a disability"
WEIGHT = "State population"
EXPERIMENT = "xgboost_survey_weights"
VARIANTS = ("standard", "survey_weighted")
BATCH = 50_000  # test-mode parquet row-batch size; keeps peak X to BATCH × n_features
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"file://{Path(__file__).resolve().parent / 'mlruns'}",
)

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("train", "test"), required=True)
p.add_argument("--threshold", type=float, default=0.5,
               help="test-mode decision threshold on predict_proba (default 0.5)")
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


if args.mode == "train": # Training a model to be logged into MlFlow
    df = pl.read_parquet(args.input)
    w = df[WEIGHT].to_numpy().astype(np.float64)
    X = df.drop([TARGET, TARGET_NEG, WEIGHT]).to_numpy()
    y = df[TARGET].to_numpy().astype(np.int8)

    scale_pos_weight = float((y == 0).sum()) / float(max(int((y == 1).sum()), 1))
    for variant in VARIANTS:
        with mlflow.start_run(run_name=f"{variant}_train"):
            mlflow.log_params({
                "variant": variant,
                "model": "XGBClassifier",
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "weight_col": WEIGHT,
                "target": TARGET,
                "n_rows": len(y),
                "n_features": X.shape[1],
                "input": args.input,
            })

            sw = w if variant == "survey_weighted" else None
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                callbacks=[TqdmCallback(total=200, desc=f"train xgb {variant}")],
            ).fit(X, y, sample_weight=sw)
            prob = clf.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(np.int8)
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
        loaded[variant] = (uri, mlflow.xgboost.load_model(uri))

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
        for v, (_, clf) in loaded.items():
            probs[v].append(clf.predict_proba(X_b)[:, 1])

    y = np.concatenate(ys)
    w = np.concatenate(ws)

    out_cols = {}
    for variant, (uri, _) in loaded.items():
        prob = np.concatenate(probs[variant])
        pred = (prob >= args.threshold).astype(np.int8)
        out_cols[f"predicted_disability_{variant}"] = pred

        with mlflow.start_run(run_name=f"{variant}_test"):
            mlflow.log_params({
                "variant": variant,
                "model_uri": uri,
                "threshold": args.threshold,
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
