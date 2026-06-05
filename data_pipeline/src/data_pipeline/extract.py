import shutil
from pathlib import Path

from . import config as cfg
from . import io_utils


def extract_pums(c: cfg.DAGConfig, run_id: str) -> Path:
    cache_key = io_utils.extract_cache_key(c.acs.year, c.acs.horizon, c.acs.states)
    cache_path = io_utils.extract_cache_dir() / f"pums_{cache_key}.parquet"
    raw_path = io_utils.raw_dir(run_id) / "pums.parquet"

    if cache_path.exists():
        shutil.copy(cache_path, raw_path)
        io_utils.write_metadata(raw_path, {
            "source": "cache", "cache_key": cache_key,
            "year": c.acs.year, "horizon": c.acs.horizon, "states": c.acs.states,
        })
        return raw_path

    n_rows, n_cols = _download_via_duckdb(c.acs.year, c.acs.horizon, c.acs.states, cache_path)
    shutil.copy(cache_path, raw_path)

    io_utils.write_metadata(raw_path, {
        "source": "duckdb", "cache_key": cache_key,
        "year": c.acs.year, "horizon": c.acs.horizon, "states": c.acs.states,
        "rows": int(n_rows), "cols": int(n_cols),
    })
    return raw_path


_USPS_TO_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
    "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19",
    "KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27",
    "MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35",
    "NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44",
    "SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53",
    "WV":"54","WI":"55","WY":"56",
}
_US_STATES = list(_USPS_TO_FIPS.keys())


def _fetch_pums_csv(year: int, horizon: str, st: str, base: Path) -> None:
    import urllib.request
    import zipfile

    fips = _USPS_TO_FIPS[st]
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/{horizon}/csv_p{st.lower()}.zip"
    zip_path = base / f"_tmp_{st}.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extract(f"psam_p{fips}.csv", base)
    finally:
        zip_path.unlink(missing_ok=True)


def _download_via_duckdb(year: int, horizon: str, states: list[str], out_parquet: Path) -> tuple[int, int]:
    import duckdb

    targets = _US_STATES if states == ["ALL"] else states

    base = Path(f"/opt/airflow/data/{year}/{horizon}")
    base.mkdir(parents=True, exist_ok=True)

    csv_paths: list[str] = []
    for st in targets:
        _fetch_pums_csv(year, horizon, st, base)
        csv_paths.append(str(base / f"psam_p{_USPS_TO_FIPS[st]}.csv"))

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA threads=4")
    file_list = ", ".join(f"'{p}'" for p in csv_paths)
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (STATE), STATE AS ST FROM read_csv([{file_list}],
                union_by_name=true,
                sample_size=-1,
                all_varchar=false,
                ignore_errors=false)
        )
        TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    n_rows = con.execute(f"SELECT COUNT(*) FROM '{out_parquet}'").fetchone()[0]
    schema = con.execute(f"DESCRIBE SELECT * FROM '{out_parquet}'").fetchall()
    n_cols = len(schema)
    con.close()
    return int(n_rows), int(n_cols)
