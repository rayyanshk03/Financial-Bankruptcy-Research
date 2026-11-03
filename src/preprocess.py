"""Phase 2: clean, split, impute and standardise the Polish table.

Per horizon = file number N; per the UCI documentation the label is "bankrupt after 6-N years" (horizon 1 = 5 yrs ahead,
horizon 5 = 1 yr ahead). (The files have no firm ID, so horizons are kept separate):
  1. drop exact duplicate rows
  2. drop features that are mostly missing (> MAX_MISSING)
  3. stratified train/test split
  4. median-impute (fit on train only)
  5. standardise three ways, as in the paper: Log, MinMax, Z-score (fit on train only)

Use from other code:   from preprocess import prepare
Run as a script:       .venv/bin/python src/preprocess.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SEED = 42
TEST_SIZE = 0.2
MAX_MISSING = 0.40   # drop a feature if more than this share is missing
SCALINGS = ("raw", "log", "minmax", "zscore")


def signed_log(x: np.ndarray) -> np.ndarray:
    """log1p that also handles negatives (ratios can be negative): sign(x) * log1p(|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def prepare(horizon: int, seed: int = SEED) -> dict:
    """Return train/test data for one horizon under every scaling.

    result["X_train"][scaling] / result["X_test"][scaling] are DataFrames,
    result["y_train"] / result["y_test"] are Series, result["features"] the kept columns.
    """
    df = pd.read_csv(PROCESSED / "polish_all.csv")
    df = df[df["horizon"] == horizon].drop(columns=["row_id", "horizon"])
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    features = [c for c in df.columns if c != "bankrupt"]
    miss = df[features].isna().mean()
    dropped = miss[miss > MAX_MISSING].index.tolist()
    features = [c for c in features if c not in dropped]

    X, y = df[features], df["bankrupt"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    imp = SimpleImputer(strategy="median").fit(X_tr)
    base = {
        "train": pd.DataFrame(imp.transform(X_tr), columns=features, index=X_tr.index),
        "test": pd.DataFrame(imp.transform(X_te), columns=features, index=X_te.index),
    }

    out = {"X_train": {}, "X_test": {}}
    out["X_train"]["raw"], out["X_test"]["raw"] = base["train"], base["test"]

    out["X_train"]["log"] = signed_log(base["train"])
    out["X_test"]["log"] = signed_log(base["test"])

    for name, scaler in (("minmax", MinMaxScaler()), ("zscore", StandardScaler())):
        scaler.fit(base["train"])
        for part in ("train", "test"):
            out[f"X_{part}"][name] = pd.DataFrame(
                scaler.transform(base[part]), columns=features, index=base[part].index
            )

    out.update(
        y_train=y_tr, y_test=y_te, features=features, dropped=dropped,
        info={"horizon": horizon, "rows_raw": n_before, "rows_after_dedupe": len(df),
              "duplicates_removed": n_before - len(df), "n_features": len(features),
              "train": len(y_tr), "test": len(y_te),
              "bankrupt_train": int(y_tr.sum()), "bankrupt_test": int(y_te.sum())},
    )
    return out


if __name__ == "__main__":
    rows = []
    for h in range(1, 6):
        p = prepare(h)
        rows.append(p["info"])
        d = PROCESSED / f"horizon{h}"
        d.mkdir(exist_ok=True)
        for s in SCALINGS:
            p["X_train"][s].assign(bankrupt=p["y_train"]).to_csv(d / f"train_{s}.csv", index=False)
            p["X_test"][s].assign(bankrupt=p["y_test"]).to_csv(d / f"test_{s}.csv", index=False)
        print(f"horizon {h}: dropped features {p['dropped']}")
    summary = pd.DataFrame(rows).set_index("horizon")
    summary.to_csv(PROCESSED / "preprocess_summary.csv")
    print()
    print(summary.to_string())
