"""
Usage:
    python data_processing.py <input.parquet> <output.parquet>
"""
import argparse

import polars as pl

from pums_labels import COLUMN_LABELS, VALUE_LABELS

DROP_HIGH_CARD = ["OCCP", "POWPUMA", "FOD1P", "MIGSP"]

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input parquet path")
parser.add_argument("output", help="Output parquet path")
args = parser.parse_args()

df = pl.read_parquet(args.input)
df = df.drop(DROP_HIGH_CARD)

for col, mapping in VALUE_LABELS.items():
    if col in df.columns:
        df = df.with_columns(pl.col(col).replace(mapping))
df = df.rename({k: v for k, v in COLUMN_LABELS.items() if k in df.columns})

# Min-max scaling for numeric featires.
# "State population" is excluded so it can be used as a survey weight downstream.
num_cols = [
    c for c, t in df.schema.items()
    if t.is_numeric() and c != "State population"
]
df = df.with_columns(
    (pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())
    for c in num_cols
)

# One-hot encoding for categorical features
cat_cols = [c for c, t in df.schema.items() if not t.is_numeric()]
df = df.to_dummies(cat_cols)

df.write_parquet(args.output)
