# data_pipeline
Airflow & Celery pipeline producing synthetic data from ACS PUMS sampling 5,000 samples per state using Gaussian Copula Synthesizer.

## Quick start
```bash
docker compose build
docker compose up -d
```

Unpause and trigger dag
```bash
docker compose exec airflow-scheduler airflow dags unpause data_pipeline
docker compose exec airflow-scheduler airflow dags trigger data_pipeline
```

View progress
- Airflow UI: <http://localhost:8080> (login: `airflow` / `airflow`)
- Flower (Celery dashboard): <http://localhost:5555>

## Modes
`DATA_PIPELINE_MODE` in `.env` selects the active config:
- `sample` — single state (WY), 5,000 synthetic rows. Smoke test.
- `train` *(default)* — all 50 states, 5,000 synthetic rows per state (250,000 total).
- `test` — all 50 states, synthetic rows per state = sum of PWGTP for that state (true population estimate; hundreds of millions of rows total). The concat output runs into tens of GB — check free space on the data drive before triggering.

```bash
# edit .env → DATA_PIPELINE_MODE=sample|train|test
docker compose up -d --force-recreate airflow-scheduler airflow-worker
docker compose exec airflow-scheduler airflow dags trigger data_pipeline
```

## Null handling
Explicit null casting: categorical nulls become `"NA"`, numeric nulls are mean-imputed (per-column mean of non-null values; integer columns are mean-imputed).

## Output
Each run writes to `data/<run_id>/synth/_full.parquet` — concatenated per-state parquet partitions. In `train` mode this is 250,000 rows × 55 cols (5,000 rows × 50 states). In `sample` mode it is 5,000 rows × 55 cols (WY only). In `test` mode the per-state row count is the state's sum-of-PWGTP population estimate (hundreds of millions of rows total).