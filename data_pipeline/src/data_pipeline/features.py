import json
from pathlib import Path
from typing import Iterable

from . import config as cfg
from . import io_utils

STATE_POPULATION_FILE = "state_population.json"


def select_features(raw_path: Path, c: cfg.DAGConfig, run_id: str) -> Path:
    import polars as pl
    import pyarrow.parquet as pq

    feats = cfg.load_features_config()
    keep = feats.all_columns()

    raw_schema = pl.scan_parquet(raw_path).collect_schema().names()
    missing = [col for col in keep if col not in raw_schema]
    if missing:
        raise ValueError(f"Configured columns missing from raw: {missing}")

    df = pl.read_parquet(raw_path, columns=keep)
    cols = set(df.columns)
    df = _apply_recodes(df, feats, cols)
    df = _apply_adjinc(df, feats, cols)
    df = _cast_dtypes(df, feats, cols)
    df = _fill_nulls(df, feats, cols)

    out = io_utils.features_path(run_id)
    df.write_parquet(out, compression="snappy")

    # Derive state_population (sum of PWGTP per ST)
    state_pop = (
        df.lazy()
        .group_by("ST")
        .agg(pl.col("PWGTP").cast(pl.Int64).sum().alias("state_population"))
        .collect()
    )
    state_pop_map = {
        str(st): int(pop)
        for st, pop in zip(state_pop["ST"].to_list(), state_pop["state_population"].to_list())
    }
    (io_utils.run_dir(run_id) / STATE_POPULATION_FILE).write_text(
        json.dumps(state_pop_map, indent=2, sort_keys=True)
    )

    pf = pq.ParquetFile(out)
    final_cols = pf.schema_arrow.names
    n_rows = pf.metadata.num_rows
    io_utils.write_metadata(out, {
        "rows": int(n_rows),
        "cols": len(final_cols),
        "columns": final_cols,
        "row_groups": pf.num_row_groups,
        "year": c.acs.year, "horizon": c.acs.horizon, "states": c.acs.states,
    })
    return out


def _apply_recodes(frame, feats: cfg.FeaturesConfig, cols: set[str]):
    import polars as pl
    for spec in _all_specs(feats):
        if not spec.recode or spec.name not in cols:
            continue
        mapping = {str(k): v for k, v in spec.recode.items()}
        frame = frame.with_columns(
            pl.col(spec.name).cast(pl.Utf8).replace(mapping, default=pl.col(spec.name).cast(pl.Utf8))
              .alias(spec.name)
        )
    return frame


def _apply_adjinc(frame, feats: cfg.FeaturesConfig, cols: set[str]):
    """
    ADJINC is an integer with 6 implied decimals (e.g. 1010207 → 1.010207).
    """
    import polars as pl
    if "ADJINC" not in cols:
        return frame
    adjinc = (pl.col("ADJINC").cast(pl.Float64) / 1_000_000.0)
    income_cols = [s.name for s in feats.features if s.adjinc and s.name in cols]
    if not income_cols:
        return frame
    return frame.with_columns([
        (pl.col(col).cast(pl.Float64) * adjinc).alias(col) for col in income_cols
    ])


def _cast_dtypes(frame, feats: cfg.FeaturesConfig, cols: set[str]):
    import polars as pl
    exprs = []
    for spec in _all_specs(feats):
        if spec.name not in cols:
            continue
        if spec.dtype == "categorical":
            exprs.append(pl.col(spec.name).cast(pl.Utf8).cast(pl.Categorical))
        elif spec.dtype == "int":
            exprs.append(pl.col(spec.name).cast(pl.Int64, strict=False))
        elif spec.dtype == "float":
            exprs.append(pl.col(spec.name).cast(pl.Float64, strict=False))
    if exprs:
        frame = frame.with_columns(exprs)
    return frame


def _fill_nulls(frame, feats: cfg.FeaturesConfig, cols: set[str]):
    import polars as pl
    exprs = []
    for spec in _all_specs(feats):
        if spec.name not in cols:
            continue
        if spec.dtype == "categorical":
            exprs.append(pl.col(spec.name).fill_null("NA"))
        elif spec.dtype in ("int", "float"):
            exprs.append(pl.col(spec.name).fill_null(-1))
    if exprs:
        frame = frame.with_columns(exprs)
    return frame


def _all_specs(feats: cfg.FeaturesConfig) -> Iterable[cfg.FeatureSpec]:
    yield from feats.keys
    yield from feats.weights
    yield from feats.features
