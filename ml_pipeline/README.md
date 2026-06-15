# ml_pipeline

Feature preperation and model training/testing for the synthetic PUMS data.

## Build

```bash
docker build -t ml_pipeline .
```

## Start

```bash
docker run -it --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -p 5001:5001 \
  ml_pipeline
```

## Run

Drop the parquet from `data_pipeline` into `ml_pipeline/data/`

```bash
# Process data before passing into model
python data_processing.py data/_full.parquet data/train.parquet

# Train both variants (standard + survey_weighted), logged to mlflow
python logstic_regression.py --mode train data/train.parquet

# Run logistic regression model on test set (after running preprocessing on test set)
python logstic_regression.py --mode test --threshold 0.5 data/test.parquet

# Browse runs
mlflow ui --host 0.0.0.0 --port 5001   # then http://localhost:5001
```