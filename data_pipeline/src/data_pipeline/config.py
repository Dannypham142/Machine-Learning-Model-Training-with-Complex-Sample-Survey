"""
Three modes, specified via DATA_PIPELINE_MODE env var:
  - "train" (default): 50 states, 5,000 synthetic rows per state (canonical training output)
  - "sample": single-state smoke run (WY), 5,000 synthetic rows
  - "test": 50 states, synthetic rows per state = sum(PWGTP) // 10 for that
           state (full population OOMed on dev host; row count resolved per-state
           at synth time from state_population.json — see synth.generate_state)
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ACSConfig:
    year: int
    horizon: str
    survey: str
    states: list[str]


@dataclass(frozen=True)
class DAGConfig:
    acs: ACSConfig
    random_seed: int
    output_root: str
    # None = match per-state population (sum of PWGTP) read from state_population.json.
    # Concrete int = fixed sample count regardless of training size.
    synthetic_rows_per_state: int | None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    role: str
    recode: dict | None = None
    adjinc: bool = False


@dataclass(frozen=True)
class FeaturesConfig:
    keys: list[FeatureSpec]
    weights: list[FeatureSpec]
    features: list[FeatureSpec]
    visit_sequence: list[str]

    def all_columns(self) -> list[str]:
        return [s.name for s in (*self.keys, *self.weights, *self.features)]

    def synthesized(self) -> list[FeatureSpec]:
        return list(self.features)


CONFIG_TRAIN = DAGConfig(
    acs=ACSConfig(year=2023, horizon="5-Year", survey="person", states=["ALL"]),
    random_seed=42,
    output_root="/opt/airflow/data",
    synthetic_rows_per_state=5000,
)

CONFIG_SAMPLE = DAGConfig(
    acs=ACSConfig(year=2023, horizon="5-Year", survey="person", states=["WY"]),
    random_seed=42,
    output_root="/opt/airflow/data",
    synthetic_rows_per_state=5000,
)

CONFIG_TEST = DAGConfig(
    acs=ACSConfig(year=2023, horizon="5-Year", survey="person", states=["ALL"]),
    random_seed=42,
    output_root="/opt/airflow/data",
    synthetic_rows_per_state=None,
)


def load_dag_config() -> DAGConfig:
    mode = os.environ.get("DATA_PIPELINE_MODE", "train").lower()
    if mode == "sample":
        return CONFIG_SAMPLE
    if mode == "test":
        return CONFIG_TEST
    return CONFIG_TRAIN


VISIT_SEQUENCE = [
    "AGEP", "SEX", "RAC1P", "HISP", "NATIVITY", "CIT", "LANX", "ENG",
    "MAR", "MARHT", "RELSHIPP", "FER", "GCL", "NOP", "PAOC",
    "SCHL", "SCH", "SCHG", "FOD1P",
    "MIL", "VPS",
    "DIS", "DEAR", "DEYE", "DREM", "DPHY", "DDRS", "DOUT",
    "COW", "ESR", "WKHP", "WKWN", "OCCP", "INDP", "JWMNP", "JWTRNS", "POWPUMA", "WRK",
    "MIG", "MIGSP",
    "HICOV", "PRIVCOV", "PUBCOV",
    "WAGP", "SEMP", "INTP", "RETP", "SSP", "SSIP", "PAP",
    "PERNP", "PINCP", "POVPIP",
]

# ST is the partition key in per-state mode (dropped from visit_sequence).
_KEYS = [
    FeatureSpec("SERIALNO", "categorical", "key"),
    FeatureSpec("SPORDER", "int", "key"),
    FeatureSpec("PUMA", "categorical", "key"),
    FeatureSpec("ADJINC", "float", "key"),
]

_WEIGHTS = [FeatureSpec("PWGTP", "float", "weight")]

_FEATURES = [
    FeatureSpec("ST", "categorical", "geography"),
    FeatureSpec("AGEP", "int", "demographic"),
    FeatureSpec("SEX", "categorical", "demographic"),
    FeatureSpec("RAC1P", "categorical", "demographic"),
    FeatureSpec("RACWHT", "categorical", "demographic"),
    FeatureSpec("RACBLK", "categorical", "demographic"),
    FeatureSpec("RACASN", "categorical", "demographic"),
    FeatureSpec("HISP", "categorical", "demographic"),
    FeatureSpec("ANC1P", "categorical", "demographic"),
    FeatureSpec("NATIVITY", "categorical", "demographic"),
    FeatureSpec("CIT", "categorical", "demographic"),
    FeatureSpec("YOEP", "int", "demographic"),
    FeatureSpec("POBP", "categorical", "demographic"),
    FeatureSpec("LANX", "categorical", "demographic"),
    FeatureSpec("ENG", "categorical", "demographic"),
    FeatureSpec("MAR", "categorical", "household"),
    FeatureSpec("MARHT", "categorical", "household"),
    FeatureSpec("RELSHIPP", "categorical", "household"),
    FeatureSpec("FER", "categorical", "household"),
    FeatureSpec("GCL", "categorical", "household"),
    FeatureSpec("NOP", "int", "household"),
    FeatureSpec("PAOC", "categorical", "household"),
    FeatureSpec("SCHL", "categorical", "education"),
    FeatureSpec("SCH", "categorical", "education"),
    FeatureSpec("SCHG", "categorical", "education"),
    FeatureSpec("FOD1P", "categorical", "education"),
    FeatureSpec("COW", "categorical", "labor"),
    FeatureSpec("ESR", "categorical", "labor"),
    FeatureSpec("WKHP", "int", "labor"),
    FeatureSpec("WKWN", "int", "labor"),
    FeatureSpec("OCCP", "categorical", "labor"),
    FeatureSpec("INDP", "categorical", "labor"),
    FeatureSpec("JWMNP", "int", "labor"),
    FeatureSpec("JWTRNS", "categorical", "labor"),
    FeatureSpec("POWPUMA", "categorical", "labor"),
    FeatureSpec("WRK", "categorical", "labor"),
    FeatureSpec("PINCP", "float", "income", adjinc=True),
    FeatureSpec("PERNP", "float", "income", adjinc=True),
    FeatureSpec("WAGP", "float", "income", adjinc=True),
    FeatureSpec("SEMP", "float", "income", adjinc=True),
    FeatureSpec("INTP", "float", "income", adjinc=True),
    FeatureSpec("RETP", "float", "income", adjinc=True),
    FeatureSpec("SSP", "float", "income", adjinc=True),
    FeatureSpec("SSIP", "float", "income", adjinc=True),
    FeatureSpec("PAP", "float", "income", adjinc=True),
    FeatureSpec("POVPIP", "int", "income"),
    FeatureSpec("DIS", "categorical", "disability"),
    FeatureSpec("DEAR", "categorical", "disability"),
    FeatureSpec("DEYE", "categorical", "disability"),
    FeatureSpec("DREM", "categorical", "disability"),
    FeatureSpec("DPHY", "categorical", "disability"),
    FeatureSpec("DDRS", "categorical", "disability"),
    FeatureSpec("DOUT", "categorical", "disability"),
    FeatureSpec("MIL", "categorical", "military"),
    FeatureSpec("VPS", "categorical", "military"),
    FeatureSpec("MIG", "categorical", "migration"),
    FeatureSpec("MIGSP", "categorical", "migration"),
    FeatureSpec("HICOV", "categorical", "insurance"),
    FeatureSpec("PRIVCOV", "categorical", "insurance"),
    FeatureSpec("PUBCOV", "categorical", "insurance"),
]

FEATURES = FeaturesConfig(
    keys=_KEYS,
    weights=_WEIGHTS,
    features=_FEATURES,
    visit_sequence=VISIT_SEQUENCE,
)


def load_features_config() -> FeaturesConfig:
    return FEATURES
