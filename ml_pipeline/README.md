# ml_pipeline
Feature prep and model training/testing for the synthetic PUMS data from `data_pipeline`.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Process the data
Drop the parquet from `data_pipeline` into `data/`, then one-hot encode and scale:
```bash
.venv/bin/python data_processing.py data/_full.parquet data/test.parquet
```

## Train + evaluate
Trains and evaluates two variants (`standard`, `survey_weighted`). Each cycle samples 5,000 rows per state (~250k), splits into 200k train / 50k test, fits on train, then scores the test split and the rest of the population.
```bash
.venv/bin/python logistic_regression.py --train-test-cycles 20 --search-trials 20 data/test.parquet
.venv/bin/python xgb.py --train-test-cycles 20 --search-trials 20 data/test.parquet
.venv/bin/python neural_network.py --train-test-cycles 20 --search-trials 15 data/test.parquet
```

Flags (same on all three):
- `--train-test-cycles` *(default 5)* — sample/split/fit cycles.
- `--search-trials` *(default 20; NN 15)* — random-search trials per variant.
- `--threshold` — decision threshold; omit to use each variant's learned best.
- `input` — the parquet pool to split.

## Browse runs
```bash
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui \
  --backend-store-uri file:///Users/dpham/mlruns_ml_pipeline \
  --port 5001
```
MLflow UI: <http://localhost:5001>

## Output
Per iteration each variant writes three MLflow runs (`_train`, `_test`, `_population`) plus a labeled test-split parquet at `data/<stem>_labeled_<model>_iter<N>.parquet`.
