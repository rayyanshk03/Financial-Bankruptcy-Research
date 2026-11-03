"""Phase 5 report: LLM prompt results vs the paper's rule, a calibrated cut-off, and Phase 4 models.

Needs results/llm_prompt_h{H}_train_n400.jsonl (calibration) and llm_prompt_h{H}_test_n1000.jsonl.
H = 1 is the 1year file (bankrupt after 5 years), H = 5 the 5year file (after 1 year).
Run:  .venv/bin/python src/report_llm.py [H]      (default 1)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from llm_prompt import KEY, metrics
from preprocess import prepare
from run_models import SEED, best_threshold

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def mean_rating(df: pd.DataFrame) -> pd.Series:
    return df.ratings.apply(lambda r: np.mean([x for x in r if x is not None]) if any(x is not None for x in r) else np.nan)


def main(h: int = 1) -> None:
    cal = pd.read_json(OUT / f"llm_prompt_h{h}_train_n400.jsonl", lines=True)
    test = pd.read_json(OUT / f"llm_prompt_h{h}_test_n1000.jsonl", lines=True)
    cal["score"], test["score"] = mean_rating(cal), mean_rating(test)

    # calibrated cut-off: best balanced accuracy on the TRAIN calibration firms
    grid = np.arange(1, 10.01, 0.25)
    thr = max(grid, key=lambda t: balanced_accuracy_score(cal.bankrupt, (cal.score > t).astype(int)))
    print(f"calibration: paper cut-off is rating > 5; best balanced-accuracy cut-off = rating > {thr:.2f}")

    rows = {}
    rows["LLM (paper rule, majority of 3, >5)"] = {**metrics(test.bankrupt, test.pred),
                                                   "auc": roc_auc_score(test.bankrupt, test.score)}
    rows[f"LLM (calibrated cut-off > {thr:.2f})"] = {**metrics(test.bankrupt, (test.score > thr).astype(int)),
                                                     "auc": roc_auc_score(test.bankrupt, test.score)}

    # Phase 4 models scored on exactly the same test firms
    p = prepare(h)
    y_tr = p["y_train"].values
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    best = json.load(open(OUT / f"best_params_h{h}.json"))
    contenders = {
        "LogReg, same 6 ratios (log)": (LogisticRegression(class_weight="balanced", max_iter=3000),
                                        KEY["key_features"]),
        "LightGBM, 64 ratios (log)": (LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1,
                                                     **best["LightGBM|log"]), None),
    }
    idx = test["index"].values
    for name, (model, cols) in contenders.items():
        cols = cols or p["features"]
        Xtr, Xte = p["X_train"]["log"][cols].values, p["X_test"]["log"].loc[idx, cols].values
        oof = cross_val_predict(model, Xtr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr_m = best_threshold(y_tr, oof)
        s = model.fit(Xtr, y_tr).predict_proba(Xte)[:, 1]
        rows[name] = {**metrics(test.bankrupt, (s >= thr_m).astype(int)), "auc": roc_auc_score(test.bankrupt, s)}

    tab = pd.DataFrame(rows).T[["accuracy", "precision_bankrupt", "recall_bankrupt", "f1_bankrupt", "f1_w", "auc"]]
    tab.round(4).to_csv(OUT / f"table9_llm_h{h}.csv")
    print(f"\ntest firms: {len(test)} ({int(test.bankrupt.sum())} bankrupt), unparsed votes: "
          f"{100 * (1 - test.valid_votes.sum() / (3 * len(test))):.1f}%\n")
    print(tab.round(3).to_string())
    print("\nmean LLM rating: bankrupt firms %.2f vs healthy firms %.2f" % (
        test.score[test.bankrupt == 1].mean(), test.score[test.bankrupt == 0].mean()))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
