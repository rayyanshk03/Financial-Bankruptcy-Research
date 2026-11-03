"""American dataset: regression analysis and key-ratio selection (paper Sec. 4.1, Tables 4-6).

Same method as regression_analysis.py (L1-logistic path + collinearity filter, OLS with robust errors,
logit), on the train years (<= 2014):
  * Table 4 baseline  : OLS and logit on the six key ratios
  * Table 5 sub-groups: small vs large firms (log total assets)
  * Table 6 robustness: other time window (1999-2006 vs 2007-2014) and the 3-year target `fail_3y`
Writes results/american/key_features.json (used by the LLM prompt) and regression_*.csv.

Run:  .venv/bin/python src/american_regression.py
"""
import json

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from american_models import ROOT, SPLIT_YEAR, load  # noqa: F401
from regression_analysis import fit_table, select_key_features

OUT = ROOT / "results" / "american"
DEFINITION = {
    "roa": "net income / total assets",
    "ebit_to_assets": "EBIT / total assets",
    "ebitda_to_assets": "EBITDA / total assets",
    "retained_earnings_to_assets": "retained earnings / total assets",
    "working_capital_to_assets": "(current assets - current liabilities) / total assets",
    "current_ratio": "current assets / current liabilities",
    "quick_ratio": "(current assets - inventory) / current liabilities",
    "liabilities_to_assets": "total liabilities / total assets",
    "long_term_debt_to_assets": "long-term debt / total assets",
    "debt_to_equity": "total liabilities / (total assets - total liabilities)",
    "current_liabilities_to_assets": "current liabilities / total assets",
    "gross_margin": "gross profit / net sales",
    "net_margin": "net income / net sales",
    "ebit_margin": "EBIT / net sales",
    "opex_to_sales": "operating expenses / net sales",
    "asset_turnover": "net sales / total assets",
    "receivables_to_sales": "receivables / net sales",
    "inventory_to_assets": "inventory / total assets",
    "depreciation_to_assets": "depreciation and amortisation / total assets",
    "market_value_to_liabilities": "market value of equity / total liabilities",
    "market_value_to_assets": "market value of equity / total assets",
    "interest_cover_proxy": "EBIT / long-term debt",
    "log_total_assets": "log(1 + total assets), firm size",
    "sales_growth": "year-on-year growth of net sales",
    "assets_growth": "year-on-year growth of total assets",
    "roa_change": "change in return on assets vs last year",
}


def frame(target: str = "bankrupt", years=None):
    """Imputed, signed-log, z-scored train/test frames (statistics from train)."""
    d = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    feats = [c for c in d.columns if c not in ("company", "year", "bankrupt", "fail_3y")]
    tr = d[d.year <= SPLIT_YEAR]
    te = d[d.year > SPLIT_YEAR]
    if years:
        tr = tr[tr.year.between(*years)]
    med = tr[feats].median()
    from preprocess import signed_log
    prep = lambda x: signed_log(x[feats].fillna(med))
    Ltr, Lte = prep(tr), prep(te)
    mu, sd = Ltr.mean(), Ltr.std().replace(0, 1)
    return (Ltr - mu) / sd, tr[target], (Lte - mu) / sd, te[target], tr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Ztr, ytr, Zte, yte, tr = frame()
    key = select_key_features(Ztr, ytr)
    print("Key ratios (American):")
    for k in key:
        print(f"  {k:32s} {DEFINITION[k]}")
    json.dump({"dataset": "american", "key_features": key, "definitions": {k: DEFINITION[k] for k in key}},
              open(OUT / "key_features.json", "w"), indent=2)

    ols, ols_x = fit_table(Ztr, ytr, key, "ols")
    logit, logit_x = fit_table(Ztr, ytr, key, "logit")
    ols.to_csv(OUT / "regression_baseline_ols.csv"); logit.to_csv(OUT / "regression_baseline_logit.csv")
    print("\nBaseline OLS (robust t):\n", ols.round(4).to_string(), "\n", {k: round(v, 4) for k, v in ols_x.items()})
    print("\nBaseline logit:\n", logit.round(4).to_string(), "\n", {k: round(v, 4) for k, v in logit_x.items()})
    m = sm.Logit(ytr, sm.add_constant(Ztr[key])).fit(disp=0)
    print(f"\nSix-ratio logit, test AUC (2015-2018) = {roc_auc_score(yte, m.predict(sm.add_constant(Zte[key], has_constant='add'))):.3f}")

    size = tr["log_total_assets"]
    big = (size >= size.median()).values
    rows = []
    for name, mask in (("small firms", ~big), ("large firms", big)):
        t, ex = fit_table(Ztr[mask], ytr[mask], key, "logit")
        rows.append(t.assign(group=name, N=ex["N"], bankrupt=int(ytr[mask].sum())))
    sub = pd.concat(rows).reset_index(names="variable")
    sub.to_csv(OUT / "regression_by_size.csv", index=False)
    print("\nLogit by firm size (coef / t):\n", sub.pivot(index="variable", columns="group", values=["coef", "t"]).round(3).to_string())

    rob = []
    for name, target, yrs in (("1999-2006", "bankrupt", (1999, 2006)), ("2007-2014", "bankrupt", (2007, 2014)),
                              ("fail_3y target", "fail_3y", None)):
        Z, y, _, _, _ = frame(target, yrs)
        t, ex = fit_table(Z, y, key, "logit")
        rob.append(t.assign(setting=name, N=ex["N"]))
    rob = pd.concat(rob).reset_index(names="variable")
    rob.to_csv(OUT / "regression_robustness.csv", index=False)
    print("\nLogit t-values, robustness:\n", rob.pivot(index="variable", columns="setting", values="t").round(2).to_string())
    print("\nSigns:\n", rob.pivot(index="variable", columns="setting", values="coef").map(lambda v: "+" if v > 0 else "-").to_string())


if __name__ == "__main__":
    main()
