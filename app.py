"""Streamlit demo: enter a company's accounting figures, get a bankruptcy risk score and the reasons.

Run:  .venv/bin/streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import PRIOR, load, predict_company  # noqa: E402

LABELS = {
    "total_assets": "Total assets", "total_liabilities": "Total liabilities", "current_assets": "Current assets",
    "current_liabilities": "Current liabilities", "long_term_debt": "Long-term debt", "inventory": "Inventory",
    "receivables": "Receivables", "retained_earnings": "Retained earnings", "market_value": "Market value of equity",
    "net_sales": "Net sales", "gross_profit": "Gross profit", "operating_expenses": "Operating expenses",
    "ebitda": "EBITDA", "ebit": "EBIT", "depreciation": "Depreciation & amortisation", "net_income": "Net income"}
PLAIN = {"market_value_to_liabilities": "market value vs. debts", "market_value_to_assets": "market value vs. assets",
         "quick_ratio": "quick (liquid) ratio", "roa_change": "change in return on assets", "assets_growth": "asset growth",
         "sales_growth": "sales growth", "ebitda_to_assets": "EBITDA / assets", "liabilities_to_assets": "debt / assets",
         "log_total_assets": "company size", "opex_to_sales": "operating costs / sales", "roa": "return on assets",
         "net_margin": "net margin", "debt_to_equity": "debt / equity"}

st.set_page_config(page_title="Company bankruptcy risk", layout="wide")
art = load()
st.title("Company bankruptcy risk")
st.caption(f"LightGBM trained on US listed firms ({art['train_years']}). On held-out years 2015-2018 its ranking AUC is "
           f"{art['test_auc']:.2f}. It flags roughly 4 in 10 firms that later failed, and most flags are false alarms: "
           "treat the score as a screening signal, not a verdict.")

choice = st.radio("Start from", ["Enter my own figures"] + list(art["examples"]), horizontal=True)
ex = art["examples"].get(choice, {"items": {}, "prior": {}})

st.subheader("This year's figures (any one currency unit, e.g. USD millions)")
cols = st.columns(4)
items = {}
for i, k in enumerate(art["items"]):
    items[k] = cols[i % 4].number_input(LABELS[k], value=float(ex["items"].get(k, 0.0)), format="%.3f", key=f"{choice}-{k}")
st.subheader("Last year's figures (optional, gives the growth terms)")
c2 = st.columns(3)
prior = {k: c2[i].number_input(f"{LABELS[k]} (last year)", value=float(ex["prior"].get(k, 0.0)), format="%.3f",
                               key=f"{choice}-p-{k}") for i, k in enumerate(PRIOR)}

if st.button("Predict", type="primary"):
    if items["total_assets"] <= 0:
        st.warning("Enter total assets (this year) first.")
    else:
        r = predict_company(items, prior if prior["total_assets"] > 0 and prior["net_sales"] > 0 else None)
        a, b = st.columns([1, 2])
        a.metric("Bankruptcy risk score", f"{r['bankruptcy_score']:.0%}")
        (a.error if r["high_risk"] else a.success)(
            "HIGH RISK: above the alert level" if r["high_risk"] else "Below the alert level")
        a.caption(f"Alert level {r['threshold']:.0%}. The score is a model probability, not a calibrated chance of failure.")
        b.markdown("**What drives this score**")
        for d in r["drivers"]:
            arrow = "🔺" if d["shap"] > 0 else "🔻"
            val = "n/a" if d["value"] is None else f"{d['value']:.2f}"
            b.write(f"{arrow} {PLAIN.get(d['ratio'], d['ratio'])} = {val} {d['effect']}")
        b.bar_chart(pd.Series({PLAIN.get(d["ratio"], d["ratio"]): d["shap"] for d in r["drivers"]}))
        if not (prior["total_assets"] > 0 and prior["net_sales"] > 0):
            st.info("No last-year figures given: the growth terms used typical values, so the score is less sharp.")
