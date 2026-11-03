"""Final model for the demo: LightGBM on the 26 American ratios (signed-log), trained on firm-years <= 2014.

Uses the best Phase-B3 hyper-parameters and a threshold from out-of-fold TRAIN scores. Saves everything needed to score
a new company, plus a few real example firms, to models/bankruptcy_lgbm_american.joblib.
(The older Polish demo model is built by train_final_polish.py.)

Run:  .venv/bin/python src/train_final.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from american_models import ROOT, OUT, SEED, load
from run_models import best_threshold

ITEMS = {  # friendly name -> raw column in american_bankruptcy.csv
    "current_assets": "X1", "depreciation": "X3", "ebitda": "X4", "inventory": "X5", "net_income": "X6",
    "receivables": "X7", "market_value": "X8", "net_sales": "X9", "total_assets": "X10", "long_term_debt": "X11",
    "ebit": "X12", "gross_profit": "X13", "current_liabilities": "X14", "retained_earnings": "X15",
    "total_liabilities": "X17", "operating_expenses": "X18"}


def main() -> None:
    a = load()
    feats = a["features"]
    best = json.load(open(OUT / "best_params_bankrupt.json"))["LightGBM|log"]
    model = LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1, **best)
    Xtr = pd.DataFrame(a["X"]["log"]["train"], columns=feats)
    Xte = pd.DataFrame(a["X"]["log"]["test"], columns=feats)
    oof = cross_val_predict(model, Xtr, a["y_train"], cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    thr = best_threshold(a["y_train"], oof)
    model.fit(Xtr, a["y_train"])
    auc = roc_auc_score(a["y_test"], model.predict_proba(Xte)[:, 1])

    d = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    raw = pd.read_csv(ROOT / "data" / "american" / "american_bankruptcy.csv")
    tr = d[d.year <= 2014]
    medians = tr[feats].median().to_dict()
    # two real firms from the held-out years as demo examples (company names are anonymised ids)
    rng = np.random.default_rng(SEED)
    r_sorted = raw.sort_values(["company_name", "year"]).reset_index(drop=True)
    items_of = lambda i: {k: float(r_sorted.loc[i, c]) for k, c in ITEMS.items()}
    prior_of = lambda i: {k: float(r_sorted.loc[i - 1, ITEMS[k]]) for k in ("total_assets", "net_sales", "net_income")}
    has_prior = lambda i: i > 0 and r_sorted.company_name[i] == r_sorted.company_name[i - 1]
    score = lambda i: predict.predict_company(items_of(i), prior_of(i))
    art = {"model": model, "features": feats, "medians": medians, "threshold": thr, "test_auc": auc,
           "items": ITEMS, "train_years": "1999-2014"}
    joblib.dump(art, ROOT / "models" / "bankruptcy_lgbm_american.joblib")

    # demo examples: score candidates through the same pipeline the app uses, keep ones that illustrate each case
    import predict
    predict.load.cache_clear()
    fails = [i for i in rng.permutation(d.index[(d.year > 2014) & (d.bankrupt == 1)]) if has_prior(i)][:60]
    oks = [i for i in rng.permutation(d.index[(d.year > 2014) & (d.bankrupt == 0)]) if has_prior(i)][:60]
    caught = [i for i in fails if score(i)["high_risk"]]
    clear = [i for i in oks if not score(i)["high_risk"]]
    examples = {"Example: a firm that later failed (caught by the model)": {"items": items_of(caught[0]), "prior": prior_of(caught[0])},
                "Example: a healthy firm": {"items": items_of(clear[0]), "prior": prior_of(clear[0])}}
    print(f"demo examples: {len(caught)} of {len(fails)} sampled failures are flagged")

    joblib.dump({**art, "examples": examples}, ROOT / "models" / "bankruptcy_lgbm_american.joblib")
    print(f"saved; threshold {thr:.3f}; test AUC (2015-2018) {auc:.3f}")


if __name__ == "__main__":
    main()
