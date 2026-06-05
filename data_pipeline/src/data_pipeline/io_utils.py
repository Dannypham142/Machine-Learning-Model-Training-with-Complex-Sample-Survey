import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def output_root() -> Path:
    return Path(os.environ.get("DATA_PIPELINE_OUTPUT_ROOT", "data"))


def run_dir(run_id: str) -> Path:
    p = output_root() / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir(run_id: str) -> Path:
    p = run_dir(run_id) / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def features_path(run_id: str) -> Path:
    return run_dir(run_id) / "features.parquet"


def synth_dir(run_id: str) -> Path:
    p = run_dir(run_id) / "synth"
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_cache_dir() -> Path:
    p = output_root() / "_extract_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_cache_key(year: int, horizon: str, states: list[str]) -> str:
    payload = json.dumps({"year": year, "horizon": horizon, "states": sorted(states)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_metadata(target: Path, payload: dict[str, Any]) -> Path:
    sidecar = target.with_suffix(target.suffix + ".meta.json")
    enriched = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        **{k: _json_safe(v) for k, v in payload.items()},
    }
    sidecar.write_text(json.dumps(enriched, indent=2, default=_json_safe))
    return sidecar


def _json_safe(v: Any) -> Any:
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (set, tuple)):
        return list(v)
    return v
