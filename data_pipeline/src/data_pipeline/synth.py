import json
from pathlib import Path

import numpy as np

from . import config as cfg
from . import io_utils
from .features import STATE_POPULATION_FILE

TRAINING_SUBDIR = "training"


def state_codes_from_features(features_path: Path, c: cfg.DAGConfig) -> list[str]:
    """
    Preparing 50 states for training
    """
    import polars as pl
    from .extract import _USPS_TO_FIPS

    present = (
        pl.scan_parquet(features_path)
        .select("ST")
        .unique()
        .collect(streaming=True)["ST"]
        .cast(pl.Utf8)
        .to_list()
    )
    supported_fips = set(_USPS_TO_FIPS.values())
    present = sorted(s for s in present if s is not None and s in supported_fips)
    if c.acs.states != ["ALL"]:
        wanted_fips = {_USPS_TO_FIPS[s] for s in c.acs.states if s in _USPS_TO_FIPS}
        present = [s for s in present if s in wanted_fips]
    if not present:
        raise ValueError(f"No ST values matched acs.states={c.acs.states}")
    return present


def prepare_training_per_state(features_path: Path, c: cfg.DAGConfig, run_id: str) -> Path:
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    feats = cfg.load_features_config()
    pf = pq.ParquetFile(features_path)
    schema_cols = pf.schema_arrow.names

    st_arr = (
        pl.scan_parquet(features_path)
        .select(["ST"])
        .collect(streaming=True)["ST"]
        .cast(pl.Utf8)
        .to_numpy()
    )

    states = state_codes_from_features(features_path, c)

    per_state_idx: dict[str, np.ndarray] = {}
    for st in states:
        in_state = np.flatnonzero(st_arr == st)
        if in_state.size == 0:
            continue
        per_state_idx[st] = in_state  # already sorted

    # Training files only need the columns that will actually be synthesized,
    synth_cols = [c for c in feats.visit_sequence if c in schema_cols and c != "ST"]

    all_idx = np.concatenate(list(per_state_idx.values()))
    state_tag = np.concatenate(
        [np.full(v.size, st, dtype=object) for st, v in per_state_idx.items()]
    )
    sort_perm = np.argsort(all_idx, kind="stable")
    all_idx_sorted = all_idx[sort_perm]
    state_tag_sorted = state_tag[sort_perm]

    rg_offsets = np.empty(pf.num_row_groups + 1, dtype=np.int64)
    rg_offsets[0] = 0
    for i in range(pf.num_row_groups):
        rg_offsets[i + 1] = rg_offsets[i] + pf.metadata.row_group(i).num_rows

    per_state_tables: dict[str, list[pa.Table]] = {st: [] for st in per_state_idx}
    for rg_i in range(pf.num_row_groups):
        rg_start = rg_offsets[rg_i]
        rg_end = rg_offsets[rg_i + 1]
        lo = int(np.searchsorted(all_idx_sorted, rg_start, side="left"))
        hi = int(np.searchsorted(all_idx_sorted, rg_end, side="left"))
        if lo == hi:
            continue
        rg_table = pf.read_row_group(rg_i, columns=synth_cols)
        local = (all_idx_sorted[lo:hi] - rg_start).astype(np.int64)
        tags = state_tag_sorted[lo:hi]
        for st in np.unique(tags):
            sel = tags == st
            per_state_tables[str(st)].append(rg_table.take(pa.array(local[sel])))

    out_dir = io_utils.synth_dir(run_id)
    training_dir = out_dir / TRAINING_SUBDIR
    training_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for st, tables in per_state_tables.items():
        if not tables:
            continue
        df = pl.from_arrow(pa.concat_tables(tables))
        path = training_dir / f"{st}.parquet"
        df.write_parquet(path, compression="snappy")
        written.append((st, df.height))

    io_utils.write_metadata(training_dir, {
        "n_states": len(written),
        "states": [st for st, _ in written],
        "rows_per_state": dict(written),
        "columns": synth_cols,
    })
    return training_dir


def generate_state(
    state_code: str,
    training_dir: Path,
    c: cfg.DAGConfig,
    run_id: str,
) -> Path:
    """
    Fit an SDV GaussianCopulaSynthesizer on every source row for this state
    and sample `c.synthetic_rows_per_state` for sample/train mode 
    or `state_population.json[state_code] / 10` rows for test mode.
    """
    import pandas as pd
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    out_dir = io_utils.synth_dir(run_id)
    training_path = training_dir / f"{state_code}.parquet"
    part = out_dir / f"part-{state_code}.parquet"

    seed = c.random_seed + (abs(hash(state_code)) % 10_000)
    np.random.seed(seed)

    train = pd.read_parquet(training_path)

    for col in train.columns:
        if isinstance(train[col].dtype, pd.CategoricalDtype):
            train[col] = train[col].astype("object")
    n_train = len(train)

    state_pop_map = json.loads((io_utils.run_dir(run_id) / STATE_POPULATION_FILE).read_text())
    state_population = int(state_pop_map[state_code])
    # test mode samples sum(PWGTP) // 10 per state — full population was OOMing on this host.
    n_synth = c.synthetic_rows_per_state if c.synthetic_rows_per_state is not None else state_population // 10

    feats = cfg.load_features_config()
    spec_by_name = {s.name: s for s in feats.synthesized()}

    metadata = SingleTableMetadata()
    for col in train.columns:
        spec = spec_by_name.get(col)
        if spec is None or spec.dtype == "categorical":
            sdtype = "categorical"
        else:
            sdtype = "numerical"
        metadata.add_column(col, sdtype=sdtype)

    synth = GaussianCopulaSynthesizer(metadata, default_distribution="norm")
    synth.fit(train)
    syn_df = synth.sample(num_rows=n_synth)

    num_cols = syn_df.select_dtypes(include="number").columns
    obj_cols = syn_df.columns.difference(num_cols)
    if len(num_cols):
        syn_df[num_cols] = syn_df[num_cols].fillna(train[num_cols].mean())
    if len(obj_cols):
        syn_df[obj_cols] = syn_df[obj_cols].fillna("NA")

    syn_df["ST"] = state_code
    syn_df["state_population"] = state_population

    syn_df.to_parquet(part, engine="pyarrow", compression="snappy", index=False)

    io_utils.write_metadata(part, {
        "state": state_code,
        "seed": seed,
        "training_rows": n_train,
        "synth_rows": n_synth,
        "training_path": str(training_path),
        "synthesizer": "sdv.GaussianCopulaSynthesizer",
    })
    return part


def concat_to_single_file(chunk_paths: list[Path], run_id: str) -> Path:
    import polars as pl

    out_dir = io_utils.synth_dir(run_id)
    out_path = out_dir / "_full.parquet"
    paths = [str(p) for p in chunk_paths]
    pl.scan_parquet(paths).sink_parquet(str(out_path), compression="snappy")
    return out_path
