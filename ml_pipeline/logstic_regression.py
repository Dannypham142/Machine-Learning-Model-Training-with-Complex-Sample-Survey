"""
    python logstic_regression.py --mode train /path/to/train.parquet
    python logstic_regression.py --mode test  /path/to/test.parquet

Train fits a standard and a survey-weighted logistic regression on the whole input
and logs both via mlflow.sklearn. Test pulls the most recent of each from the
tracking store and writes a labeled parquet to ./data/.
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
p.add_argument("input")
args = p.parse_args()

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

df = pl.read_parquet(args.input)
w = df[WEIGHT].to_numpy().astype(np.float64)
X = df.drop([TARGET, TARGET_NEG, WEIGHT]).to_numpy()


def score(y, pred, prob, sw=None):
    return {
        "accuracy": accuracy_score(y, pred, sample_weight=sw),
        "precision": precision_score(y, pred, sample_weight=sw, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=sw, zero_division=0),
        "f1": f1_score(y, pred, sample_weight=sw, zero_division=0),
        "roc_auc": roc_auc_score(y, prob, sample_weight=sw),
    }


if args.mode == "train":
    y = df[TARGET].to_numpy().astype(np.int8)
    for variant in VARIANTS:
        with mlflow.start_run(run_name=variant):
            mlflow.log_params({
                "variant": variant,
                "model": "LogisticRegression",
                "max_iter": 1000,
                "weight_col": WEIGHT,
                "target": TARGET,
                "n_rows": len(y),
                "n_features": X.shape[1],
                "input": args.input,
            })

            sw = w if variant == "survey_weighted" else None
            clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X, y, sample_weight=sw)
            pred = clf.predict(X)
            prob = clf.predict_proba(X)[:, 1]
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"train_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"train_weighted_{name}", float(val))

            mlflow.sklearn.log_model(clf, name="model")
            print(f"logged {variant}")

else:
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)

    out_cols = {}
    for variant in VARIANTS:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{variant}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        uri = f"runs:/{runs[0].info.run_id}/model"
        clf = mlflow.sklearn.load_model(uri)
        out_cols[f"predicted_disability_{variant}"] = clf.predict(X).astype(np.int8)
        print(f"loaded {variant} <- {uri}")

    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(args.input).stem}_labeled.parquet"
    df.with_columns([pl.Series(n, v) for n, v in out_cols.items()]).write_parquet(out)
    print(f"wrote {out}")
