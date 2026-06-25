"""
Usage:
    python data_processing.py <input.parquet> <output.parquet>
"""
import argparse

import polars as pl

from pums_labels import COLUMN_LABELS, VALUE_LABELS

DROP_HIGH_CARD = ["OCCP", "POWPUMA", "FOD1P", "MIGSP"]
WEIGHT_SRC = "state_population"
WEIGHT = "State population"

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input parquet path")
parser.add_argument("output", help="Output parquet path")
args = parser.parse_args()

lf = pl.scan_parquet(args.input).drop(DROP_HIGH_CARD)
schema = lf.collect_schema()


def labeled(c):
    return COLUMN_LABELS.get(c, c)


# One-hot value-coded columns
coded_cols = [c for c in VALUE_LABELS if c in schema]
coded_exprs = [
    (pl.col(c).cast(pl.String) == code).cast(pl.UInt8).alias(f"{labeled(c)}_{label}")
    for c in coded_cols
    for code, label in VALUE_LABELS[c].items()
]

# Any other non-numeric column
extra_cat_cols = [
    c for c, t in schema.items()
    if not t.is_numeric() and c not in coded_cols and c != WEIGHT_SRC
]
extra_exprs = []
if extra_cat_cols:
    uniques = lf.select(
        pl.col(c).unique().drop_nulls().implode().alias(c) for c in extra_cat_cols
    ).collect()
    for c in extra_cat_cols:
        for val in uniques[c].item():
            extra_exprs.append(
                (pl.col(c) == val).cast(pl.UInt8).alias(f"{labeled(c)}_{val}")
            )

# Min-max scale
num_cols = [
    c for c, t in schema.items()
    if t.is_numeric() and c != WEIGHT_SRC and c not in coded_cols
]
scale_exprs = []
if num_cols:
    stats = lf.select(
        *[pl.col(c).min().alias(f"_min_{c}") for c in num_cols],
        *[pl.col(c).max().alias(f"_max_{c}") for c in num_cols],
    ).collect().row(0, named=True)
    scale_exprs = [
        ((pl.col(c) - stats[f"_min_{c}"]) / (stats[f"_max_{c}"] - stats[f"_min_{c}"]))
            .cast(pl.Float32)
            .alias(labeled(c))
        for c in num_cols
    ]

weight_exprs = [pl.col(WEIGHT_SRC).alias(WEIGHT)] if WEIGHT_SRC in schema else []

lf.select(weight_exprs + scale_exprs + coded_exprs + extra_exprs).sink_parquet(
    args.output, compression="lz4"
)
