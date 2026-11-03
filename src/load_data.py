"""Phase 1: load the five Polish .arff files into one clean table.

Run from the project root:  .venv/bin/python src/load_data.py
Writes:
  data/processed/polish_all.csv         all horizons stacked, readable column names
  data/processed/column_dictionary.csv  short name -> original Attr -> definition
"""
from pathlib import Path

import numpy as np
import pandas as pd

from columns import COLUMNS, FEATURES, RENAME

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "polish"
OUT = ROOT / "data" / "processed"


def read_arff(path: Path) -> pd.DataFrame:
    """Parse one .arff file (numeric attributes, '?' = missing, last column = class)."""
    names, rows, in_data = [], [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                names.append(line.split()[1])
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                rows.append(line.split(","))
    df = pd.DataFrame(rows, columns=names).replace("?", np.nan)
    return df.apply(pd.to_numeric)


def load_all() -> pd.DataFrame:
    frames = []
    for h in range(1, 6):
        df = read_arff(RAW / f"{h}year.arff").rename(columns=RENAME)
        df = df.rename(columns={"class": "bankrupt"})
        df["bankrupt"] = df["bankrupt"].astype(int)
        df.insert(0, "horizon", h)  # = file number N; UCI: label is "bankrupt after 6-N years" (1 = 5 yrs ahead, 5 = 1 yr ahead)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.insert(0, "row_id", np.arange(len(all_df)))
    return all_df[["row_id", "horizon", *FEATURES, "bankrupt"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    df.to_csv(OUT / "polish_all.csv", index=False)
    pd.DataFrame(
        [(v[0], k, v[1]) for k, v in COLUMNS.items()],
        columns=["name", "original", "definition"],
    ).to_csv(OUT / "column_dictionary.csv", index=False)

    print(f"saved {len(df):,} rows x {df.shape[1]} cols -> {OUT / 'polish_all.csv'}\n")
    g = df.groupby("horizon")["bankrupt"].agg(rows="size", bankrupt="sum")
    g["bankrupt_%"] = (100 * g["bankrupt"] / g["rows"]).round(2)
    print(g.to_string())
    miss = df[FEATURES].isna().mean().mul(100).sort_values(ascending=False)
    print(f"\noverall missing: {df[FEATURES].isna().mean().mean() * 100:.2f}%")
    print("most-missing features (%):")
    print(miss.head(6).round(1).to_string())


if __name__ == "__main__":
    main()
