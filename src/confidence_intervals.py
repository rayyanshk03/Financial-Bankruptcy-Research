"""Bootstrap 95% confidence intervals for the headline results (test sets hold few bankruptcies).

Each test set is resampled 1,000 times (with replacement) and the AUC / bankrupt-class recall recomputed.
Models: the best-params LightGBM (signed-log) on each dataset, plus the American logistic regression.
Taiwan uses pooled out-of-fold predictions from a 5-fold CV.

Run:  .venv/bin/python src/confidence_intervals.py   ->  results/confidence_intervals.csv
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from american_models import load as load_american
from preprocess import prepare
from run_models import SEED, best_threshold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def bootstrap(y, s, thr, n=1000, seed=SEED) -> dict:
    rng, aucs, recs = np.random.default_rng(seed), [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
        recs.append(((s[i] >= thr) & (y[i] == 1)).sum() / y[i].sum())
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return {"n_test": len(y), "n_bankrupt": int(y.sum()), "auc": roc_auc_score(y, s),
            "auc_low": ci(aucs)[0], "auc_high": ci(aucs)[1],
            "recall": ((s >= thr) & (y == 1)).sum() / y.sum(), "recall_low": ci(recs)[0], "recall_high": ci(recs)[1]}


def lgbm(params):
    return LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1, **params)


def main() -> None:
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rows = []

    # American: train <= 2014, test 2015-2018
    a = load_american()
    best = json.load(open(RES / "american" / "best_params_bankrupt.json"))["LightGBM|log"]
    Xtr, Xte = a["X"]["log"]["train"], a["X"]["log"]["test"]
    m = lgbm(best)
    thr = best_threshold(a["y_train"], cross_val_predict(m, Xtr, a["y_train"], cv=cv, method="predict_proba")[:, 1])
    s = m.fit(Xtr, a["y_train"]).predict_proba(Xte)[:, 1]
    rows.append({"dataset": "American (2015-2018)", "model": "LightGBM", **bootstrap(a["y_test"], s, thr)})

    # Polish: file Nyear has the label "bankrupt after 6-N years"; use the 5-years-ahead and 1-year-ahead files
    for h, label in ((1, "Polish (5 yrs ahead)"), (5, "Polish (1 yr ahead)")):
        p = prepare(h)
        best = json.load(open(RES / f"best_params_h{h}.json"))["LightGBM|log"]
        Xtr, Xte, ytr, yte = p["X_train"]["log"], p["X_test"]["log"], p["y_train"].values, p["y_test"].values
        m = lgbm(best)
        thr = best_threshold(ytr, cross_val_predict(m, Xtr, ytr, cv=cv, method="predict_proba")[:, 1])
        s = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        rows.append({"dataset": label, "model": "LightGBM", **bootstrap(yte, s, thr)})

    # Taiwan: pooled out-of-fold predictions (default LightGBM settings, signed-log)
    from taiwan_models import load as load_taiwan
    from preprocess import signed_log
    X, y = load_taiwan()
    X = signed_log(X.values)
    m = lgbm({"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05})
    oof = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    rows.append({"dataset": "Taiwan (5-fold CV, pooled)", "model": "LightGBM", **bootstrap(y, oof, best_threshold(y, oof))})

    df = pd.DataFrame(rows)
    df.round(4).to_csv(RES / "confidence_intervals.csv", index=False)
    show = df.assign(AUC=lambda d: d.auc.round(3).astype(str) + " [" + d.auc_low.round(3).astype(str) + ", " + d.auc_high.round(3).astype(str) + "]",
                     recall=lambda d: d.recall.round(2).astype(str) + " [" + d.recall_low.round(2).astype(str) + ", " + d.recall_high.round(2).astype(str) + "]")
    print(show[["dataset", "n_test", "n_bankrupt", "AUC", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
