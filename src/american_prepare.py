"""American dataset: build the 'bankrupt next year' label and financial ratios.

Raw file: data/american/american_bankruptcy.csv (US listed firms 1999-2018, 18 raw items X1..X18).
The raw `status_label` is 'failed' in EVERY year of a failed company, so it is not a prediction target.
Here a firm-year is positive only when it is the LAST year of a company that failed (the statement
filed just before the failure); `fail_3y` also marks the two years before it.

Columns X1..X18 (checked: EBITDA = sales - opex, EBIT = EBITDA - D&A, gross profit = sales - COGS):
X1 current assets, X2 COGS, X3 D&A, X4 EBITDA, X5 inventory, X6 net income, X7 receivables,
X8 market value, X9 net sales, X10 total assets, X11 long-term debt, X12 EBIT, X13 gross profit,
X14 current liabilities, X15 retained earnings, X16 total revenue, X17 total liabilities, X18 operating expenses.

Run:  .venv/bin/python src/american_prepare.py   ->  data/american/american_ratios.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "american"


def ratios_from_items(d: pd.DataFrame) -> pd.DataFrame:
    """The 23 same-year ratios from the 18 raw items (X1..X18); needs no history."""
    r = pd.DataFrame(index=d.index)
    div = lambda a, b: (d[a] / d[b].replace(0, np.nan))
    r["roa"] = div("X6", "X10")
    r["ebit_to_assets"] = div("X12", "X10")
    r["ebitda_to_assets"] = div("X4", "X10")
    r["retained_earnings_to_assets"] = div("X15", "X10")
    r["working_capital_to_assets"] = (d.X1 - d.X14) / d.X10.replace(0, np.nan)
    r["current_ratio"] = div("X1", "X14")
    r["quick_ratio"] = (d.X1 - d.X5) / d.X14.replace(0, np.nan)
    r["liabilities_to_assets"] = div("X17", "X10")
    r["long_term_debt_to_assets"] = div("X11", "X10")
    r["debt_to_equity"] = d.X17 / (d.X10 - d.X17).replace(0, np.nan)
    r["current_liabilities_to_assets"] = div("X14", "X10")
    r["gross_margin"] = div("X13", "X9")
    r["net_margin"] = div("X6", "X9")
    r["ebit_margin"] = div("X12", "X9")
    r["opex_to_sales"] = div("X18", "X9")
    r["asset_turnover"] = div("X9", "X10")
    r["receivables_to_sales"] = div("X7", "X9")
    r["inventory_to_assets"] = div("X5", "X10")
    r["depreciation_to_assets"] = div("X3", "X10")
    r["market_value_to_liabilities"] = div("X8", "X17")
    r["market_value_to_assets"] = div("X8", "X10")
    r["interest_cover_proxy"] = div("X12", "X11")
    r["log_total_assets"] = np.log1p(d.X10.clip(lower=0))
    return r


def build() -> pd.DataFrame:
    d = pd.read_csv(DIR / "american_bankruptcy.csv").sort_values(["company_name", "year"]).reset_index(drop=True)
    failed = d.status_label == "failed"
    last_year = d.groupby("company_name").year.transform("max")
    d["bankrupt"] = (failed & (d.year == last_year)).astype(int)
    d["fail_3y"] = (failed & (d.year >= last_year - 2)).astype(int)

    r = pd.concat([pd.DataFrame({"company": d.company_name, "year": d.year, "bankrupt": d.bankrupt,
                                 "fail_3y": d.fail_3y}), ratios_from_items(d)], axis=1)
    g = d.groupby("company_name")
    r["sales_growth"] = g.X9.pct_change(fill_method=None)
    r["assets_growth"] = g.X10.pct_change(fill_method=None)
    r["roa_change"] = r.roa - r.groupby("company").roa.shift(1)
    feats = [c for c in r.columns if c not in ("company", "year", "bankrupt", "fail_3y")]
    r[feats] = r[feats].replace([np.inf, -np.inf], np.nan).clip(-1e3, 1e3)
    return r


if __name__ == "__main__":
    r = build()
    r.to_csv(DIR / "american_ratios.csv", index=False)
    print(r.shape, "positives:", int(r.bankrupt.sum()), "| fail_3y:", int(r.fail_3y.sum()))
    print("by period  train<=2014:", int(r[r.year <= 2014].bankrupt.sum()), "/", int((r.year <= 2014).sum()),
          "| test>=2015:", int(r[r.year >= 2015].bankrupt.sum()), "/", int((r.year >= 2015).sum()))
    print("missing share:", r.drop(columns=["company"]).isna().mean().round(3).sort_values().tail(4).to_dict())
