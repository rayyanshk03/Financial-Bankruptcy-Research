"""Score a company with the final American model: will it go bankrupt within a year?

Input = accounting items in the company's own currency units (any consistent unit, e.g. USD millions):
  current_assets, depreciation, ebitda, inventory, net_income, receivables, market_value, net_sales,
  total_assets, long_term_debt, ebit, gross_profit, current_liabilities, retained_earnings,
  total_liabilities, operating_expenses.
Ratios are computed from them. Optional `prior` = last year's total_assets, net_sales, net_income gives the three
growth terms; without it they use the training median and the score is noticeably less sharp.

As a library:  from predict import predict_company;  predict_company({...items...})
From the shell: .venv/bin/python src/predict.py '{"total_assets": 500, "total_liabilities": 450, ...}'
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from american_prepare import ratios_from_items
from preprocess import signed_log

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load() -> dict:
    art = joblib.load(ROOT / "models" / "bankruptcy_lgbm_american.joblib")
    art["explainer"] = shap.TreeExplainer(art["model"])
    return art


PRIOR = ("total_assets", "net_sales", "net_income")   # last year's figures, for the three growth terms


def ratios(items: dict, prior: dict | None = None) -> pd.DataFrame:
    art = load()
    d = pd.DataFrame([{art["items"][k]: v for k, v in items.items() if k in art["items"]}])
    for c in art["items"].values():
        if c not in d:
            d[c] = np.nan
    r = ratios_from_items(d)
    if prior and all(prior.get(k) for k in PRIOR[:2]):
        r["assets_growth"] = items["total_assets"] / prior["total_assets"] - 1
        r["sales_growth"] = items["net_sales"] / prior["net_sales"] - 1
        if "net_income" in prior:
            r["roa_change"] = r["roa"] - prior["net_income"] / prior["total_assets"]
    r = r.replace([np.inf, -np.inf], np.nan).clip(-1e3, 1e3)
    return r.reindex(columns=art["features"])


def predict_company(items: dict, prior: dict | None = None, top: int = 4) -> dict:
    art = load()
    r = ratios(items, prior)
    X = signed_log(r.fillna(pd.Series(art["medians"])).values)
    prob = float(art["model"].predict_proba(X)[0, 1])
    sv = art["explainer"].shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)[0] if np.asarray(sv).ndim == 2 else np.asarray(sv)[0, :, 1]
    order = np.argsort(-np.abs(sv))[:top]
    drivers = [{"ratio": art["features"][j], "value": None if np.isnan(r.iloc[0, j]) else float(r.iloc[0, j]),
                "effect": "raises risk" if sv[j] > 0 else "lowers risk", "shap": float(sv[j])} for j in order]
    return {"bankruptcy_score": prob, "high_risk": prob >= art["threshold"], "threshold": art["threshold"],
            "drivers": drivers, "ratios": r.iloc[0].to_dict()}


if __name__ == "__main__":
    out = predict_company(json.loads(sys.argv[1]))
    print(f"score {out['bankruptcy_score']:.3f} (threshold {out['threshold']:.3f}) -> "
          f"{'HIGH RISK' if out['high_risk'] else 'low risk'}")
    for d in out["drivers"]:
        print(f"  {d['ratio']:30s} {d['value']}  {d['effect']}")
