"""
    python neural_network.py data/test.parquet
    python neural_network.py --train-test-cycles 10 --search-trials 15 data/test.parquet

Each of --train-test-cycles cycles samples 5000 rows per state, splits that sample into
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
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

TARGET = "Disability (recoded)_With a disability"
TARGET_NEG = "Disability (recoded)_Without a disability"
WEIGHT = "State population"
STATE_PREFIX = "State (FIPS code)_"
EXPERIMENT = "neural_network_survey_weights"
VARIANTS = ("standard", "survey_weighted")
HIDDEN_CHOICES = [(256, 128, 64), (384, 192, 96), (512, 256, 128)] # Note that exists 526 features
PER_STATE = 5_000  # rows sampled per state each iteration
TEST_SIZE = 50_000  # held-out test split carved out of the per-state sample
BATCH_SIZE = 8192
EPOCHS = 15
SEARCH_TRIALS = 15
TRAIN_TEST_CYCLES = 5
VAL_FRAC = 0.1
SEED = 42
LR = 1e-3
PREDICT_BATCH = 16_384
TEST_BATCH = 50_000  # test parquet row-batch size
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "file:///Users/dpham/mlruns_ml_pipeline",
)

p = argparse.ArgumentParser()
p.add_argument("--threshold", type=float, default=None,
               help="decision threshold; default: each variant's learned best_threshold")
p.add_argument("--search-trials", type=int, default=SEARCH_TRIALS,
               help="random-search trials per variant (default %(default)s)")
p.add_argument("--train-test-cycles", type=int, default=TRAIN_TEST_CYCLES,
               help="cycles, each on a fresh 5000-per-state sample split "
                    "200k train / 50k test (default %(default)s)")
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
    def __init__(self, input_size, hidden, dropout):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
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
        "hidden": HIDDEN_CHOICES[int(rng.integers(len(HIDDEN_CHOICES)))],
        "dropout": float(rng.uniform(0.1, 0.5)),
        "weight_decay": float(10 ** rng.uniform(-5, -3)),
        "lr": float(10 ** rng.uniform(-4, -2.5)),
        "use_class_weight": bool(rng.integers(2)),
    }


def train_mlp(X_t, y_t, sw_t, input_size, params, epochs, desc):
    model = MLP(input_size, params["hidden"], params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    model.train()
    n = X_t.shape[0]
    for _ in tqdm(range(epochs), desc=desc, unit="epoch", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = weighted_bce_loss(model(X_t[idx]), y_t[idx], sw_t[idx])
            loss.backward()
            opt.step()
    return model


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


for it in range(args.train_test_cycles):
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
    w_all = sample_df[WEIGHT].to_numpy().astype(np.float32)
    X_all = sample_df.drop([TARGET, TARGET_NEG, WEIGHT]).cast(pl.Float32).to_numpy()
    y_all = sample_df[TARGET].to_numpy().astype(np.int8)

    # split the per-state sample into 200k train / 50k held-out test
    X, X_test, y, y_test, w, w_test = train_test_split(
        X_all, y_all, w_all, test_size=TEST_SIZE, stratify=y_all,
        random_state=SEED + it)
    cw = compute_sample_weight("balanced", y).astype(np.float32)

    trained = {}  # variant -> (model, threshold)
    for variant in VARIANTS:
        Xtr, Xval, ytr, yval, wtr, wval, cwtr, cwval = train_test_split(
            X, y, w, cw, test_size=VAL_FRAC, stratify=y, random_state=SEED)
        pop_tr = wtr if variant == "survey_weighted" else np.ones_like(wtr)
        val_sw = wval if variant == "survey_weighted" else None

        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)

        rng = np.random.default_rng(SEED)
        with mlflow.start_run(run_name=f"{variant}_iter{it}_train"):
            best = None  # (val_auc, params, val_prob)
            for i in tqdm(range(args.search_trials), desc=f"search nn {variant} it{it}", unit="trial"):
                params = sample_params(rng)
                cw_factor = cwtr if params["use_class_weight"] else np.ones_like(cwtr)
                sw_t = torch.from_numpy((cw_factor * pop_tr).astype(np.float32)).to(device)
                model = train_mlp(Xtr_t, ytr_t, sw_t, X.shape[1], params,
                                  EPOCHS, f"nn trial {variant} it{it}")
                val_prob = predict_proba(model, Xval)
                auc = roc_auc_score(yval, val_prob, sample_weight=val_sw)
                with mlflow.start_run(run_name=f"{variant}_iter{it}_trial_{i}", nested=True):
                    mlflow.log_params({
                        "variant": variant, "iteration": it, "trial": i,
                        "hidden_layer_sizes": str(params["hidden"]),
                        "dropout": params["dropout"],
                        "weight_decay": params["weight_decay"],
                        "lr": params["lr"],
                        "use_class_weight": params["use_class_weight"],
                    })
                    mlflow.log_metric("val_roc_auc", float(auc))
                if best is None or auc > best[0]:
                    best = (auc, params, val_prob)
            val_auc, params, val_prob = best
            thr = best_threshold(yval, val_prob, val_sw)
            del Xtr_t, ytr_t

            mlflow.log_params({
                "variant": variant, "iteration": it,
                "model": "PyTorchMLP", "search": "random",
                "search_trials": args.search_trials,
                "hidden_layer_sizes": str(params["hidden"]),
                "dropout": params["dropout"],
                "weight_decay": params["weight_decay"],
                "lr": params["lr"],
                "use_class_weight": params["use_class_weight"],
                "best_threshold": thr, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "weight_col": WEIGHT, "target": TARGET,
                "n_rows": len(y), "n_features": X.shape[1], "input": args.input,
                "random_state": SEED,
            })
            mlflow.log_metric("val_roc_auc", float(val_auc))
            mlflow.log_metric("best_threshold", float(thr))

            cw_factor = cw if params["use_class_weight"] else np.ones_like(cw)
            pop = w if variant == "survey_weighted" else np.ones_like(w)
            X_t = torch.from_numpy(X).to(device)
            y_t = torch.from_numpy(y.astype(np.float32)).to(device)
            sw_t = torch.from_numpy((cw_factor * pop).astype(np.float32)).to(device)
            model = train_mlp(X_t, y_t, sw_t, X.shape[1], params,
                              EPOCHS, f"train nn {variant} it{it}")

            prob = predict_proba(model, X)
            pred = (prob >= thr).astype(np.int8)
            for name, val in score(y, pred, prob).items():
                mlflow.log_metric(f"train_unweighted_{name}", float(val))
            for name, val in score(y, pred, prob, sw=w).items():
                mlflow.log_metric(f"train_weighted_{name}", float(val))
            del X_t, y_t

        use_thr = args.threshold if args.threshold is not None else thr
        trained[variant] = (model, use_thr)

    # evaluate on the 50k held-out test split (weighted + unweighted)
    out_cols = {}
    for variant, (model, thr) in trained.items():
        prob = predict_proba(model, X_test)
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
    out = out_dir / f"{Path(args.input).stem}_labeled_neural_network_iter{it}.parquet"
    pl.DataFrame({TARGET: y_test, **out_cols}).write_parquet(out)

    # evaluate on the rest of the population (every row not in the sample; unweighted)
    ys, probs = [], {v: [] for v in VARIANTS}
    pop_features = 0
    offset = 0
    pf = pq.ParquetFile(args.input)
    total_batches = (pf.metadata.num_rows + TEST_BATCH - 1) // TEST_BATCH
    for batch in tqdm(pf.iter_batches(batch_size=TEST_BATCH), total=total_batches,
                      desc=f"population nn it{it}", unit="batch"):
        sel = ~sample_mask[offset:offset + batch.num_rows]
        offset += batch.num_rows
        if not sel.any():
            continue
        df_b = pl.from_arrow(pa.Table.from_batches([batch])).filter(pl.Series(sel))
        ys.append(df_b[TARGET].to_numpy().astype(np.int8))
        X_b = df_b.drop([TARGET, TARGET_NEG, WEIGHT]).cast(pl.Float32).to_numpy()
        pop_features = X_b.shape[1]
        for v, (model, _) in trained.items():
            probs[v].append(predict_proba(model, X_b))

    y_pop = np.concatenate(ys)
    for variant, (model, thr) in trained.items():
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
