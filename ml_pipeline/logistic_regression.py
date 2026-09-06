"""
    python logistic_regression.py data/test.parquet
    python logistic_regression.py --resample-rounds 10 --search-trials 20 data/test.parquet

Repeated random subsampling (Monte Carlo CV), NOT k-fold: each of --resample-rounds
rounds independently resamples 5000 rows per state (fresh seed), so rounds overlap
rather than partition. Each round splits its sample into
200k train / 50k test, fits the model on train (random search + refit, logging
unweighted + weighted train metrics), logs unweighted + weighted metrics on the
50k test split, then logs unweighted metrics on the rest of the population
(every row not in the sample). data/train.parquet is not used.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TARGET = "Disability (recoded)_With a disability"
TARGET_NEG = "Disability (recoded)_Without a disability"
WEIGHT = "State population"
STATE_PREFIX = "State (FIPS code)_"
EXPERIMENT = "logistic_regression_survey_weights"
VARIANTS = ("standard", "survey_weighted")
PER_STATE = 5_000  # rows sampled per state each iteration
TEST_SIZE = 50_000  # held-out test split carved out of the per-state sample
BATCH = 50_000  # population parquet row-batch size
SEED = 42
SEARCH_TRIALS = 20
RESAMPLE_ROUNDS = 5
VAL_FRAC = 0.1
CLASS_WEIGHTS = ["balanced", None]  # searched hyperparameter
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "file:///Users/dpham/mlruns_ml_pipeline",
)

p = argparse.ArgumentParser()
p.add_argument("--threshold", type=float, default=None,
               help="decision threshold; default: each variant's learned best_threshold")
p.add_argument("--search-trials", type=int, default=SEARCH_TRIALS,
               help="random-search trials per variant (default %(default)s)")
p.add_argument("--resample-rounds", type=int, default=RESAMPLE_ROUNDS,
               help="repeated random subsampling rounds (Monte Carlo CV, NOT "
                    "k-fold); each round independently draws a fresh "
                    "5000-per-state sample split 200k train / 50k test "
                    "(default %(default)s)")
p.add_argument("input")
args = p.parse_args()

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)


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


def sample_params(rng):
    return {
        "C": float(10 ** rng.uniform(-3, 1)),
        "class_weight": CLASS_WEIGHTS[int(rng.integers(len(CLASS_WEIGHTS)))],
    }


def make_clf(params):
    return LogisticRegression(
        C=params["C"], class_weight=params["class_weight"],
        max_iter=1000, solver="lbfgs",
    )


# integer state id per row (one-hot -> code) for per-state sampling
names = pq.ParquetFile(args.input).schema_arrow.names
STATE_COLS = [c for c in names if c.startswith(STATE_PREFIX)]
state_id = (
    pl.scan_parquet(args.input)
    .select(pl.sum_horizontal(
        [pl.col(c).cast(pl.Int32) * i for i, c in enumerate(STATE_COLS)]).alias("s"))
    .collect()["s"].to_numpy()
)
n_rows = state_id.shape[0]
state_indices = [np.where(state_id == s)[0] for s in range(len(STATE_COLS))]


for it in range(args.resample_rounds):
    rng_sample = np.random.default_rng(SEED + it)
    sample_idx = np.concatenate([
        rng_sample.choice(idx, size=min(PER_STATE, len(idx)), replace=False)
        for idx in state_indices
    ])
    sample_idx.sort()
    sample_mask = np.zeros(n_rows, dtype=bool)
    sample_mask[sample_idx] = True

    idx_ser = pl.Series("sample_idx", sample_idx, dtype=pl.UInt32)
    sample_df = (
        pl.scan_parquet(args.input)
        .with_row_index("__ri")
        .filter(pl.col("__ri").is_in(idx_ser))
        .drop("__ri")
        .collect()
    )
    w_all = sample_df[WEIGHT].to_numpy().astype(np.float64)
    X_all = sample_df.drop([TARGET, TARGET_NEG, WEIGHT]).cast(pl.Float32).to_numpy()
    y_all = sample_df[TARGET].to_numpy().astype(np.int8)

    # split the per-state sample into 200k train / 50k held-out test
    X, X_test, y, y_test, w, w_test = train_test_split(
        X_all, y_all, w_all, test_size=TEST_SIZE, stratify=y_all,
        random_state=SEED + it)

    trained = {}  # variant -> (clf, threshold)
    for variant in VARIANTS:
        Xtr, Xval, ytr, yval, wtr, wval = train_test_split(
            X, y, w, test_size=VAL_FRAC, stratify=y, random_state=SEED)
        tr_sw = wtr if variant == "survey_weighted" else None
        val_sw = wval if variant == "survey_weighted" else None

        rng = np.random.default_rng(SEED)
        with mlflow.start_run(run_name=f"{variant}_iter{it}_train"):
            best = None  # (val_auc, params, val_prob)
            for i in tqdm(range(args.search_trials), desc=f"search lr {variant} it{it}", unit="trial"):
                params = sample_params(rng)
                clf = make_clf(params).fit(Xtr, ytr, sample_weight=tr_sw)
                val_prob = clf.predict_proba(Xval)[:, 1]
                auc = roc_auc_score(yval, val_prob, sample_weight=val_sw)
                with mlflow.start_run(run_name=f"{variant}_iter{it}_trial_{i}", nested=True):
                    mlflow.log_params({
                        "variant": variant, "iteration": it, "trial": i,
                        "C": params["C"], "class_weight": str(params["class_weight"]),
                    })
                    mlflow.log_metric("val_roc_auc", float(auc))
                if best is None or auc > best[0]:
                    best = (auc, params, val_prob)
            val_auc, params, val_prob = best
            thr = best_threshold(yval, val_prob, val_sw)

            mlflow.log_params({
                "variant": variant, "iteration": it,
                "model": "LogisticRegression", "search": "random",
                "search_trials": args.search_trials, "C": params["C"],
                "class_weight": str(params["class_weight"]),
                "best_threshold": thr, "max_iter": 1000,
                "weight_col": WEIGHT, "target": TARGET,
                "n_rows": len(y), "n_features": X.shape[1], "input": args.input,
            })
            mlflow.log_metric("val_roc_auc", float(val_auc))
            mlflow.log_metric("best_threshold", float(thr))

            sw_full = w if variant == "survey_weighted" else None
            clf = make_clf(params).fit(X, y, sample_weight=sw_full)
            prob = clf.predict_proba(X)[:, 1]
            pred = (prob >= thr).astype(np.int8)
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"train_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"train_weighted_{name}", float(val))

        use_thr = args.threshold if args.threshold is not None else thr
        trained[variant] = (clf, use_thr)

    # evaluate on the 50k held-out test split (weighted + unweighted)
    out_cols = {}
    for variant, (clf, thr) in trained.items():
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= thr).astype(np.int8)
        out_cols[f"predicted_proba_{variant}"] = prob
        out_cols[f"predicted_disability_{variant}"] = pred

        with mlflow.start_run(run_name=f"{variant}_iter{it}_test"):
            mlflow.log_params({
                "variant": variant, "iteration": it, "threshold": thr,
                "input": args.input, "n_rows": len(y_test),
                "n_features": X_test.shape[1],
            })
            for name, val in score(y_test, pred, prob).items():
                mlflow.log_metric(f"test_unweighted_{name}", float(val))
            for name, val in score(y_test, pred, prob, sw=w_test).items():
                mlflow.log_metric(f"test_weighted_{name}", float(val))

    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(args.input).stem}_labeled_logistic_regression_iter{it}.parquet"
    pl.DataFrame({TARGET: y_test, **out_cols}).write_parquet(out)

    # evaluate on the rest of the population (every row not in the sample; unweighted)
    ys, probs = [], {v: [] for v in VARIANTS}
    pop_features = 0
    offset = 0
    pf = pq.ParquetFile(args.input)
    total_batches = (pf.metadata.num_rows + BATCH - 1) // BATCH
    for batch in tqdm(pf.iter_batches(batch_size=BATCH), total=total_batches,
                      desc=f"population lr it{it}", unit="batch"):
        sel = ~sample_mask[offset:offset + batch.num_rows]
        offset += batch.num_rows
        if not sel.any():
            continue
        df_b = pl.from_arrow(pa.Table.from_batches([batch])).filter(pl.Series(sel))
        ys.append(df_b[TARGET].to_numpy().astype(np.int8))
        X_b = df_b.drop([TARGET, TARGET_NEG, WEIGHT]).cast(pl.Float32).to_numpy()
        pop_features = X_b.shape[1]
        for v, (clf, _) in trained.items():
            probs[v].append(clf.predict_proba(X_b)[:, 1])

    y_pop = np.concatenate(ys)
    for variant, (clf, thr) in trained.items():
        prob = np.concatenate(probs[variant])
        pred = (prob >= thr).astype(np.int8)
        with mlflow.start_run(run_name=f"{variant}_iter{it}_population"):
            mlflow.log_params({
                "variant": variant, "iteration": it, "threshold": thr,
                "input": args.input, "n_rows": len(y_pop),
                "n_features": pop_features,
            })
            for name, val in score(y_pop, pred, prob).items():
                mlflow.log_metric(f"population_unweighted_{name}", float(val))
