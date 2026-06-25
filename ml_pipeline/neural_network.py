"""
    python neural_network.py --mode train /path/to/train.parquet
    python neural_network.py --mode test  /path/to/test.parquet

Train fits a standard and a survey-weighted PyTorch MLP.
Test loads the most recent of each variant and writes target + predictions.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.pytorch
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

TARGET = "Disability (recoded)_With a disability"
TARGET_NEG = "Disability (recoded)_Without a disability"
WEIGHT = "State population"
EXPERIMENT = "neural_network_survey_weights"
VARIANTS = ("standard", "survey_weighted")
HIDDEN = (128, 64, 32)
EPOCHS = 200 
BATCH_SIZE = 8192
LR = 1e-3
SEED = 42
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
PREDICT_BATCH = 16_384
TEST_BATCH = 50_000
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"file://{Path(__file__).resolve().parent / 'mlruns'}",
)

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("train", "test"), required=True)
p.add_argument("--threshold", type=float, default=0.5,
               help="test-mode decision threshold on sigmoid output (default 0.5)")
p.add_argument("input")
args = p.parse_args()

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def weighted_bce_loss(logits, targets, sample_weights):
    # Survey-weighted BCE:  L = Σ w_i · BCE(ŷ_i, y_i) / Σ w_i
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (bce * sample_weights).sum() / sample_weights.sum()


def score(y, pred, prob, sw=None):
    return {
        "accuracy": accuracy_score(y, pred, sample_weight=sw),
        "precision": precision_score(y, pred, sample_weight=sw, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=sw, zero_division=0),
        "f1": f1_score(y, pred, sample_weight=sw, zero_division=0),
        "roc_auc": roc_auc_score(y, prob, sample_weight=sw),
    }


@torch.no_grad()
def predict_proba(model, X_np, batch_size=PREDICT_BATCH):
    model.eval()
    X_t = torch.from_numpy(X_np)
    out = []
    for i in range(0, len(X_t), batch_size):
        out.append(torch.sigmoid(model(X_t[i:i + batch_size].to(device))).cpu().numpy())
    return np.concatenate(out)


if args.mode == "train":
    df = pl.read_parquet(args.input)
    w = df[WEIGHT].to_numpy().astype(np.float32)
    X = df.drop([TARGET, TARGET_NEG, WEIGHT]).to_numpy().astype(np.float32)
    y = df[TARGET].to_numpy().astype(np.int8)
    cw = compute_sample_weight("balanced", y).astype(np.float32)

    for variant in VARIANTS:
        with mlflow.start_run(run_name=f"{variant}_train"):
            mlflow.log_params({
                "variant": variant,
                "model": "PyTorchMLP",
                "hidden_layer_sizes": str(HIDDEN),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_col": WEIGHT,
                "target": TARGET,
                "n_rows": len(y),
                "n_features": X.shape[1],
                "input": args.input,
                "random_state": SEED,
            })

            # Class-balanced for both variants; survey_weighted additionally scales by population.
            sw_np = cw * (w if variant == "survey_weighted" else np.ones_like(w))
            X_t = torch.from_numpy(X).to(device)
            y_t = torch.from_numpy(y.astype(np.float32)).to(device)
            sw_t = torch.from_numpy(sw_np).to(device)
            n = X_t.shape[0]

            model = MLP(X.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=LR)

            model.train()
            best_loss = float("inf")
            patience = 0
            trained_epochs = 0
            pbar = tqdm(range(EPOCHS), desc=f"train {variant}", unit="epoch")
            for epoch in pbar:
                perm = torch.randperm(n, device=device)
                loss_sum = torch.zeros((), device=device)
                n_batches = 0
                for i in range(0, n, BATCH_SIZE):
                    idx = perm[i:i + BATCH_SIZE]
                    opt.zero_grad(set_to_none=True)
                    loss = weighted_bce_loss(model(X_t[idx]), y_t[idx], sw_t[idx])
                    loss.backward()
                    opt.step()
                    loss_sum += loss.detach()
                    n_batches += 1
                epoch_loss = (loss_sum / n_batches).item()
                trained_epochs = epoch + 1
                pbar.set_postfix(loss=f"{epoch_loss:.4f}", best=f"{best_loss:.4f}")
                if best_loss - epoch_loss > EARLY_STOP_MIN_DELTA:
                    best_loss = epoch_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= EARLY_STOP_PATIENCE:
                        break
            pbar.close()
            mlflow.log_metric("trained_epochs", trained_epochs)

            prob = predict_proba(model, X)
            pred = (prob >= 0.5).astype(np.int8)
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"train_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"train_weighted_{name}", float(val))

            mlflow.pytorch.log_model(model, name=variant)

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
        loaded[variant] = (uri, mlflow.pytorch.load_model(uri).to(device))

    ys, ws, probs = [], [], {v: [] for v in VARIANTS}
    n_features = 0
    pf = pq.ParquetFile(args.input)
    total_batches = (pf.metadata.num_rows + TEST_BATCH - 1) // TEST_BATCH
    for batch in tqdm(pf.iter_batches(batch_size=TEST_BATCH), total=total_batches,
                      desc="test nn", unit="batch"):
        df_b = pl.from_arrow(pa.Table.from_batches([batch]))
        ys.append(df_b[TARGET].to_numpy().astype(np.int8))
        ws.append(df_b[WEIGHT].to_numpy().astype(np.float32))
        X_b = df_b.drop([TARGET, TARGET_NEG, WEIGHT]).to_numpy().astype(np.float32)
        n_features = X_b.shape[1]
        for v, (_, model) in loaded.items():
            probs[v].append(predict_proba(model, X_b))

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
    out = out_dir / f"{Path(args.input).stem}_labeled_neural_network.parquet"
    pl.DataFrame({TARGET: y, **out_cols}).write_parquet(out)
