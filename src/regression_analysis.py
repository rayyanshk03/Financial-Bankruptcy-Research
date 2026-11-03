"""Phase 3: empirical (regression) analysis and key-variable selection (paper Sec. 4.1).

Paper: pick a few core indicators, run a regression (Table 4), test sub-groups (Table 5),
then a robustness check (Table 6). Here:
  * key ratios are chosen on the TRAIN set only (L1-logistic path + collinearity filter),
  * baseline = OLS (as in the paper, with R^2 / t-values) and a logit,
  * sub-group test splits firms by size (log_total_assets) instead of industry,
  * robustness re-estimates the same six ratios on the other horizons.
All regressors are log-scaled then z-scored with train statistics, so coefficients are comparable.

Run:  .venv/bin/python src/regression_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.diagnostic import het_breuschpagan

from columns import COLUMNS
from preprocess import prepare

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
N_KEY = 6
MAX_COLLINEAR = 0.7   # skip a candidate if |corr| with an already chosen ratio exceeds this
MAIN_HORIZON = 1
DEFINITION = {v[0]: v[1] for v in COLUMNS.values()}


def standardised_log(p: dict):
    """Log-scaled train/test, z-scored with train mean/std."""
    tr, te = p["X_train"]["log"], p["X_test"]["log"]
    mu, sd = tr.mean(), tr.std().replace(0, 1)
    return (tr - mu) / sd, (te - mu) / sd


def select_key_features(Z: pd.DataFrame, y: pd.Series, n: int = N_KEY) -> list[str]:
    """Order features by when they enter an L1-logistic path; keep the first n that are not collinear."""
    entry = {}
    for C in np.logspace(-3.5, 0, 45):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                               class_weight="balanced", max_iter=500).fit(Z, y)
        for f, w in zip(Z.columns, m.coef_[0]):
            if w != 0 and f not in entry:
                entry[f] = C
    order = sorted(entry, key=entry.get)
    corr = Z.corr().abs()
    chosen = []
    for f in order:
        if all(corr.loc[f, c] <= MAX_COLLINEAR for c in chosen):
            chosen.append(f)
        if len(chosen) == n:
            break
    return chosen


def fit_table(Z: pd.DataFrame, y: pd.Series, cols: list[str], kind: str) -> tuple[pd.DataFrame, dict]:
    X = sm.add_constant(Z[cols])
    if kind == "ols":
        res = sm.OLS(y, X).fit(cov_type="HC1")
        extra = {"N": int(res.nobs), "R2": res.rsquared, "adj_R2": res.rsquared_adj,
                 "BP_pvalue": het_breuschpagan(sm.OLS(y, X).fit().resid, X)[1]}
    else:
        res = sm.Logit(y, X).fit(disp=0, cov_type="HC1")
        extra = {"N": int(res.nobs), "pseudo_R2": res.prsquared}
    t = pd.DataFrame({"coef": res.params, "t": res.tvalues, "p": res.pvalues})
    t["sig"] = np.where(t.p < 0.01, "***", np.where(t.p < 0.05, "**", np.where(t.p < 0.1, "*", "")))
    return t, extra


def main() -> None:
    OUT.mkdir(exist_ok=True)
    p = prepare(MAIN_HORIZON)
    Ztr, Zte = standardised_log(p)
    ytr, yte = p["y_train"], p["y_test"]

    # 1. key ratios (train only)
    key = select_key_features(Ztr, ytr)
    print(f"Key ratios (horizon {MAIN_HORIZON}):")
    for k in key:
        print(f"  {k:34s} {DEFINITION[k]}")
    json.dump({"horizon": MAIN_HORIZON, "key_features": key,
               "definitions": {k: DEFINITION[k] for k in key}},
              open(OUT / "key_features.json", "w"), indent=2)

    # 2. baseline regressions (paper Table 4)
    ols, ols_x = fit_table(Ztr, ytr, key, "ols")
    logit, logit_x = fit_table(Ztr, ytr, key, "logit")
    ols.to_csv(OUT / "regression_baseline_ols.csv")
    logit.to_csv(OUT / "regression_baseline_logit.csv")
    print("\nBaseline OLS (HC1 robust t-values):\n", ols.round(4).to_string())
    print({k: round(v, 4) for k, v in ols_x.items()})
    print("\nBaseline logit:\n", logit.round(4).to_string())
    print({k: round(v, 4) for k, v in logit_x.items()})

    # out-of-sample sanity check of the six-ratio logit
    m = sm.Logit(ytr, sm.add_constant(Ztr[key])).fit(disp=0)
    auc = roc_auc_score(yte, m.predict(sm.add_constant(Zte[key], has_constant="add")))
    print(f"\nSix-ratio logit, test AUC = {auc:.3f}")

    # 3. sub-group test (paper Table 5): small vs large firms by log_total_assets
    size = p["X_train"]["raw"]["log_total_assets"]
    big = size >= size.median()
    rows = []
    for name, mask in (("small firms", ~big), ("large firms", big)):
        t, ex = fit_table(Ztr[mask], ytr[mask], key, "logit")
        rows.append(t.assign(group=name, N=ex["N"], bankrupt=int(ytr[mask].sum())))
    sub = pd.concat(rows).reset_index(names="variable")
    sub.to_csv(OUT / "regression_by_size.csv", index=False)
    print("\nLogit by firm size (coef / t):")
    print(sub.pivot(index="variable", columns="group", values=["coef", "t"]).round(3).to_string())

    # 4. robustness (paper Table 6): same ratios on every horizon
    rob = []
    for h in range(1, 6):
        ph = prepare(h)
        Zh, _ = standardised_log(ph)
        cols = [c for c in key if c in Zh.columns]
        t, ex = fit_table(Zh, ph["y_train"], cols, "logit")
        rob.append(t.assign(horizon=h, N=ex["N"]))
    rob = pd.concat(rob).reset_index(names="variable")
    rob.to_csv(OUT / "regression_robustness_horizons.csv", index=False)
    print("\nLogit t-values by horizon:")
    print(rob.pivot(index="variable", columns="horizon", values="t").round(2).to_string())
    print("\nSigns by horizon (+/-):")
    print(rob.pivot(index="variable", columns="horizon", values="coef").map(lambda v: "+" if v > 0 else "-").to_string())


if __name__ == "__main__":
    main()
