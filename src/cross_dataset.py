"""Cross-dataset test: train on one dataset, test on another (American <-> Taiwan).

The two files share eleven ratios (mapping below). Taiwan's values are on a rescaled range, so every
shared ratio is first turned into a within-dataset percentile rank (0-1); the models then learn
"how high is this ratio compared with other firms in its own file".  Train sets are the whole file of
the training dataset (American: years <= 2014); test sets are the whole other file.
For reference, the same shared-ratio models are also scored in-domain (American 2015-2018 test,
Taiwan 5-fold CV).  No company is ever in both train and test.

Run:  .venv/bin/python src/cross_dataset.py   ->  results/cross_dataset.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from confidence_intervals import bootstrap
from run_models import SEED, best_threshold
from taiwan_models import load as load_taiwan

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

# American feature -> Taiwan column
SHARED = {
    "roa": "Net Income to Total Assets",
    "liabilities_to_assets": "Debt ratio %",
    "current_ratio": "Current Ratio",
    "quick_ratio": "Quick Ratio",
    "working_capital_to_assets": "Working Capital to Total Assets",
    "retained_earnings_to_assets": "Retained Earnings to Total Assets",
    "gross_margin": "Gross Profit to Sales",
    "asset_turnover": "Total Asset Turnover",
    "debt_to_equity": "Liability to Equity",
    "current_liabilities_to_assets": "Current Liability to Assets",
    "assets_growth": "Total Asset Growth Rate",
}


def ranks(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: c.rank(pct=True)).fillna(0.5)


def models():
    return {"Logistic": LogisticRegression(class_weight="balanced", max_iter=3000),
            "LightGBM": LGBMClassifier(class_weight="balanced", n_estimators=200, num_leaves=15, learning_rate=0.05,
                                       random_state=SEED, n_jobs=4, verbose=-1),
            "RF": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                         random_state=SEED, n_jobs=4)}


def main() -> None:
    feats = list(SHARED)
    am = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    am_tr, am_te = am[am.year <= 2014], am[am.year > 2014]
    Xa_tr, Xa_te = ranks(am_tr[feats]), ranks(am_te[feats])
    ya_tr, ya_te = am_tr.bankrupt.values, am_te.bankrupt.values
    tw, yt = load_taiwan()
    Xt = ranks(tw[list(SHARED.values())].rename(columns={v: k for k, v in SHARED.items()}))
    yt = np.asarray(yt)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    rows = []
    for name, m in models().items():
        # American -> Taiwan
        thr = best_threshold(ya_tr, cross_val_predict(m, Xa_tr, ya_tr, cv=cv, method="predict_proba")[:, 1])
        s = m.fit(Xa_tr, ya_tr).predict_proba(Xt)[:, 1]
        rows.append({"setting": "train American -> test Taiwan", "model": name, **bootstrap(yt, s, thr)})
        # Taiwan -> American (2015-2018)
        thr = best_threshold(yt, cross_val_predict(m, Xt, yt, cv=cv, method="predict_proba")[:, 1])
        s = m.fit(Xt, yt).predict_proba(Xa_te)[:, 1]
        rows.append({"setting": "train Taiwan -> test American 2015-18", "model": name, **bootstrap(ya_te, s, thr)})
        # in-domain references on the same shared ratios
        thr = best_threshold(ya_tr, cross_val_predict(m, Xa_tr, ya_tr, cv=cv, method="predict_proba")[:, 1])
        s = m.fit(Xa_tr, ya_tr).predict_proba(Xa_te)[:, 1]
        rows.append({"setting": "reference: American -> American 2015-18", "model": name, **bootstrap(ya_te, s, thr)})
        oof = cross_val_predict(m, Xt, yt, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        rows.append({"setting": "reference: Taiwan 5-fold CV", "model": name, **bootstrap(yt, oof, best_threshold(yt, oof))})

    df = pd.DataFrame(rows)
    df.round(4).to_csv(ROOT / "results" / "cross_dataset.csv", index=False)
    df["AUC [95% CI]"] = df.auc.round(3).astype(str) + " [" + df.auc_low.round(3).astype(str) + ", " + df.auc_high.round(3).astype(str) + "]"
    print(df[["setting", "model", "n_bankrupt", "AUC [95% CI]"]].to_string(index=False))


if __name__ == "__main__":
    main()
