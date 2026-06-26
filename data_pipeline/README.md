# data_pipeline
Synthetic data generation from ACS PUMS using Gaussian Copula Synthesizer.

## Quick start
```bash
docker compose build
docker compose up -d
```

Unpause and trigger dag
```bash
docker compose exec airflow-scheduler airflow dags trigger data_pipeline
```

View progress
- Airflow UI: <http://localhost:8080> (login: `airflow` / `airflow`)
- Flower (Celery dashboard): <http://localhost:5555>

## Modes
`DATA_PIPELINE_MODE` in `.env` selects the active config:
- `sample` — single state (WY), 5,000 synthetic rows.
- `train` *(default)* — all 50 states, 5,000 synthetic rows per state .
- `test` — all 50 states, synthetic rows per state = sum of PWGTP for that state

```bash
# edit .env → DATA_PIPELINE_MODE=sample|train|test
docker compose up -d --force-recreate airflow-scheduler airflow-worker
docker compose exec airflow-scheduler airflow dags trigger data_pipeline
```

## Output
Each run writes to `data/<run_id>/synth/_full.parquet`