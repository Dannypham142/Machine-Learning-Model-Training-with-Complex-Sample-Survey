"""
    python logstic_regression.py --mode train /path/to/train.parquet
    python logstic_regression.py --mode test  /path/to/test.parquet

Train fits a standard and a survey-weighted logistic regression on input.
Test passes through data to output a labeled parquet file containing features and designed prediction labels for control and treatment group.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

TARGET = "Disability (recoded)_With a disability"
TARGET_NEG = "Disability (recoded)_Without a disability"
WEIGHT = "State population"
EXPERIMENT = "logistic_regression_survey_weights"
VARIANTS = ("standard", "survey_weighted")
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

df = pl.read_parquet(args.input)
w = df[WEIGHT].to_numpy().astype(np.float64)
X = df.drop([TARGET, TARGET_NEG, WEIGHT]).to_numpy()
y = df[TARGET].to_numpy().astype(np.int8)


def score(y, pred, prob, sw=None):
    return {
        "accuracy": accuracy_score(y, pred, sample_weight=sw),
        "precision": precision_score(y, pred, sample_weight=sw, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=sw, zero_division=0),
        "f1": f1_score(y, pred, sample_weight=sw, zero_division=0),
        "roc_auc": roc_auc_score(y, prob, sample_weight=sw),
    }


if args.mode == "train": # Training a model to be logged into MlFlow
    for variant in VARIANTS:
        with mlflow.start_run(run_name=f"{variant}_train"):
            mlflow.log_params({
                "variant": variant,
                "model": "LogisticRegression",
                "max_iter": 1000,
                "class_weight": "balanced",
                "weight_col": WEIGHT,
                "target": TARGET,
                "n_rows": len(y),
                "n_features": X.shape[1],
                "input": args.input,
            })

            sw = w if variant == "survey_weighted" else None
            clf = LogisticRegression(
                max_iter=1000, solver="lbfgs", class_weight="balanced",
            ).fit(X, y, sample_weight=sw)
            pred = clf.predict(X)
            prob = clf.predict_proba(X)[:, 1]
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"train_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"train_weighted_{name}", float(val))

            mlflow.sklearn.log_model(clf, name=variant)

elif args.mode == "test": # Load most recent LR model and loging model parameters and metrics into MlFlow
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)

    out_cols = {}
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
        clf = mlflow.sklearn.load_model(uri)
        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= args.threshold).astype(np.int8)
        out_cols[f"predicted_disability_{variant}"] = pred

        with mlflow.start_run(run_name=f"{variant}_test"):
            mlflow.log_params({
                "variant": variant,
                "model_uri": uri,
                "threshold": args.threshold,
                "input": args.input,
                "n_rows": len(y),
                "n_features": X.shape[1],
            })
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"test_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"test_weighted_{name}", float(val))

    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(args.input).stem}_labeled.parquet"
    df.with_columns([pl.Series(n, v) for n, v in out_cols.items()]).write_parquet(out)