"""American dataset: LLM prompt vs ML models on the same test firms (like report_llm.py).

Needs results/american/llm_prompt_train_n300.jsonl (calibration) and llm_prompt_test_n500.jsonl.
Writes results/american/table9_llm.csv.

Run:  .venv/bin/python src/american_report_llm.py
"""
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from american_llm import KEY, OUT
from american_models import ROOT, SEED, SPLIT_YEAR
from llm_prompt import metrics
from preprocess import signed_log
from run_models import best_threshold


def mean_rating(df: pd.DataFrame) -> pd.Series:
    return df.ratings.apply(lambda r: np.mean([x for x in r if x is not None]) if any(x is not None for x in r) else np.nan)


def main() -> None:
    cal = pd.read_json(OUT / "llm_prompt_train_n300.jsonl", lines=True)
    test = pd.read_json(OUT / "llm_prompt_test_n500.jsonl", lines=True)
    cal["score"], test["score"] = mean_rating(cal), mean_rating(test)
    grid = np.arange(1, 10.01, 0.25)
    thr = max(grid, key=lambda t: balanced_accuracy_score(cal.bankrupt, (cal.score > t).astype(int)))
    print(f"paper cut-off: rating > 5; best balanced-accuracy cut-off on train firms: rating > {thr:.2f}")

    rows = {"LLM (paper rule, majority of 3, >5)": {**metrics(test.bankrupt, test.pred), "auc": roc_auc_score(test.bankrupt, test.score)},
            f"LLM (calibrated cut-off > {thr:.2f})": {**metrics(test.bankrupt, (test.score > thr).astype(int)),
                                                      "auc": roc_auc_score(test.bankrupt, test.score)}}

    d = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    feats = [c for c in d.columns if c not in ("company", "year", "bankrupt", "fail_3y")]
    tr = d[d.year <= SPLIT_YEAR]
    imp = SimpleImputer(strategy="median").fit(tr[feats])
    Xtr = pd.DataFrame(signed_log(imp.transform(tr[feats])), columns=feats)
    Xte = pd.DataFrame(signed_log(imp.transform(d.loc[test["index"], feats])), columns=feats)
    ytr = tr.bankrupt.values
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    best = json.load(open(OUT / "best_params_bankrupt.json"))["LightGBM|log"]
    contenders = {
        "LogReg, same 6 ratios (log)": (LogisticRegression(class_weight="balanced", max_iter=3000), KEY["key_features"]),
        f"LightGBM, {len(feats)} ratios (log)": (LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=4,
                                                                 verbose=-1, **best), feats),
    }
    for name, (model, cols) in contenders.items():
        oof = cross_val_predict(model, Xtr[cols], ytr, cv=cv, method="predict_proba")[:, 1]
        thr_m = best_threshold(ytr, oof)
        s = model.fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1]
        rows[name] = {**metrics(test.bankrupt, (s >= thr_m).astype(int)), "auc": roc_auc_score(test.bankrupt, s)}

    tab = pd.DataFrame(rows).T[["accuracy", "precision_bankrupt", "recall_bankrupt", "f1_bankrupt", "f1_w", "auc"]]
    tab.round(4).to_csv(OUT / "table9_llm.csv")
    print(f"\ntest firms: {len(test)} ({int(test.bankrupt.sum())} bankrupt), unparsed votes: "
          f"{100 * (1 - test.valid_votes.sum() / (3 * len(test))):.1f}%\n")
    print(tab.round(3).to_string())
    print("\nmean LLM rating: failed firms %.2f vs healthy firms %.2f" % (
        test.score[test.bankrupt == 1].mean(), test.score[test.bankrupt == 0].mean()))


if __name__ == "__main__":
    main()
