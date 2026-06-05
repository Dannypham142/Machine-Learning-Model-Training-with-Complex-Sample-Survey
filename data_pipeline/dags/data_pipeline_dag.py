from pathlib import Path
from airflow.decorators import dag, task
from data_pipeline import config as cfg

DEFAULT_ARGS = {
    "owner": "ml-survey",
    "retries": 0,
    "depends_on_past": False,
}

CONFIG = cfg.load_dag_config()


@dag(
    dag_id="data_pipeline",
    description="Census-like synthetic data pipeline",
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
)
def data_pipeline():

    @task
    def extract_pums(**ctx) -> str:
        from data_pipeline import extract
        return str(extract.extract_pums(CONFIG, run_id=ctx["run_id"]))

    @task
    def select_features(raw_path: str, **ctx) -> str:
        from data_pipeline import features
        return str(features.select_features(Path(raw_path), CONFIG, run_id=ctx["run_id"]))

    @task
    def synth_prepare_training(features_path: str, **ctx) -> str:
        from data_pipeline import synth
        return str(synth.prepare_training_per_state(Path(features_path), CONFIG, run_id=ctx["run_id"]))

    @task
    def state_list(features_path: str) -> list[str]:
        from data_pipeline import synth
        return synth.state_codes_from_features(Path(features_path), CONFIG)

    @task(map_index_template="{{ task.op_kwargs['state_code'] if task.op_kwargs else '' }}")
    def synth_generate(state_code: str, training_dir: str, **ctx) -> str:
        from data_pipeline import synth
        return str(synth.generate_state(
            state_code=state_code,
            training_dir=Path(training_dir),
            c=CONFIG,
            run_id=ctx["run_id"],
        ))

    @task
    def synth_concat(chunk_paths: list[str], **ctx) -> str:
        from data_pipeline import synth
        return str(synth.concat_to_single_file(
            chunk_paths=[Path(p) for p in chunk_paths],
            run_id=ctx["run_id"],
        ))

    raw = extract_pums()
    feats = select_features(raw)

    train = synth_prepare_training(feats)
    states = state_list(feats)
    chunks = synth_generate.partial(training_dir=train).expand(state_code=states)
    synth_concat(chunks)


dag = data_pipeline()
