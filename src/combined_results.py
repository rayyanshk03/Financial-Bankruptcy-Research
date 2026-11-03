"""Step 1 of the wrap-up: Altman Z-score baseline and one table across the three datasets.

Altman Z-score (1968): Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 1.0*Sales/TA; low Z = high risk,
"distress zone" Z < 1.81. American data has every term (market value of equity).
Polish data (file Nyear: label = bankruptcy status after 6-N years) has book equity, so the private-firm version Z' = 0.717, 0.847, 3.107, 0.420, 0.998 is used.
The Taiwan file has no sales/EBIT terms in this form, so no Z-score is computed there.
Baselines are scored on exactly the test sets the ML models use (American 2015-2018; Polish test splits of the 1year (5 yrs ahead) and 5year (1 yr ahead) files).

Run:  .venv/bin/python src/combined_results.py   ->  results/altman_baseline.csv, results/combined_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from preprocess import prepare

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def report(name, y, z, cut):
    pred = (z < cut).astype(int)
    return {"dataset": name, "model": "Altman Z-score", "auc": roc_auc_score(y, -z),
            "recall_bankrupt": recall_score(y, pred), "precision_bankrupt": precision_score(y, pred, zero_division=0),
            "f1_bankrupt": f1_score(y, pred), "cutoff": cut}


def altman_american():
    d = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    d = d[d.year > 2014].dropna(subset=["working_capital_to_assets", "retained_earnings_to_assets", "ebit_to_assets",
                                       "market_value_to_liabilities", "asset_turnover"])
    z = (1.2 * d.working_capital_to_assets + 1.4 * d.retained_earnings_to_assets + 3.3 * d.ebit_to_assets
         + 0.6 * d.market_value_to_liabilities + 1.0 * d.asset_turnover)
    return report("American (2015-2018)", d.bankrupt, z, 1.81)


def altman_polish(h=1, label="Polish (5 yrs ahead)"):
    p = prepare(h)
    X, y = p["X_test"]["raw"], p["y_test"]
    z = (0.717 * X.working_capital_to_assets + 0.847 * X.retained_earnings_to_assets + 3.107 * X.ebit_to_assets
         + 0.420 * X.equity_to_liabilities + 0.998 * X.sales_to_assets)
    return report(label, y, z, 1.23)   # 1.23 = Z' distress cut-off


def main():
    base = pd.DataFrame([altman_american(), altman_polish(1, "Polish (5 yrs ahead)"), altman_polish(5, "Polish (1 yr ahead)")])
    base.round(4).to_csv(RES / "altman_baseline.csv", index=False)
    print(base.round(3).to_string(index=False))

    srcs = {"Polish (5 yrs ahead)": RES / "model_comparison_h1.csv",
            "Polish (1 yr ahead)": RES / "model_comparison_h5.csv",
            "American (2015-2018)": RES / "american" / "model_comparison_bankrupt.csv",
            "Taiwan (5-fold CV)": RES / "taiwan" / "model_comparison.csv"}
    order = ["LightGBM", "RF", "XGBoost", "Logistic", "DT", "SVM", "NN"]
    rows = []
    for ds, f in srcs.items():
        m = pd.read_csv(f).groupby("model")[["auc", "recall_bankrupt", "f1_bankrupt"]].mean()
        rows += [{"dataset": ds, "model": k, **v} for k, v in m.iterrows()]
    rows += base[["dataset", "model", "auc", "recall_bankrupt", "f1_bankrupt"]].to_dict("records")
    long = pd.DataFrame(rows)
    wide = long.pivot(index="model", columns="dataset", values="auc").reindex(order + ["Altman Z-score"])
    wide.round(3).to_csv(RES / "combined_summary.csv")
    long.round(4).to_csv(RES / "combined_summary_long.csv", index=False)
    print("\nAUC by model (mean over the 3 scalings) and dataset:")
    print(wide.round(3).to_string())
    print("\nBankrupt-class F1:")
    print(long.pivot(index="model", columns="dataset", values="f1_bankrupt").reindex(order + ["Altman Z-score"]).round(3).to_string())


if __name__ == "__main__":
    main()
