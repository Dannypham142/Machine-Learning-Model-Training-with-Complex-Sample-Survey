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
`DATA_PIPELINE_MODE` in `.env` selects the active config: `full` runs all 50 states, `test` runs a single state (WY).

```bash
# edit .env → DATA_PIPELINE_MODE=test|full
docker compose up -d --force-recreate airflow-scheduler airflow-worker
docker compose exec airflow-scheduler airflow dags trigger data_pipeline
```

## Null handling
Explicit null casting: categorical nulls become `"NA"`, numeric nulls become `-1`.

## Output
Each run writes to `data/<run_id>/synth/_full.parquet` — concatenated 5,000 rows × 50 states (250,000 rows × 55 cols) in full mode.