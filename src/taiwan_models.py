"""Taiwan dataset (Taiwan Economic Journal, 1999-2009): 7 models x 3 standardisations, 5-fold CV.

One row per company (6,819 firms, 220 bankrupt, 95 ratios, no missing values), so no company can sit in
both train and test. There is no year column, so the honest check is stratified 5-fold cross-validation:
inside every outer fold the standardiser is fitted, hyper-parameters are grid-searched (3-fold ROC-AUC)
and the threshold is set from out-of-fold scores on that fold's TRAIN part. Predictions of the five test
folds are pooled for the metrics.

Run:  .venv/bin/python src/taiwan_models.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocess import signed_log
from run_models import SCALINGS, SEED, best_threshold, model_zoo, score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "taiwan"


def load() -> tuple[pd.DataFrame, np.ndarray]:
    d = pd.read_csv(ROOT / "data" / "taiwan" / "taiwan_bankruptcy.csv")
    d.columns = [c.strip() for c in d.columns]
    return d.drop(columns="Bankrupt?"), d["Bankrupt?"].values


def scale(name: str, Xtr: np.ndarray, Xte: np.ndarray):
    if name == "log":
        return signed_log(Xtr), signed_log(Xte)
    sc = (MinMaxScaler() if name == "minmax" else StandardScaler()).fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = load()
    X = X.values
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    pooled = {}   # (model, scaling) -> (scores, preds, truth, fold aucs)
    t0 = time.time()
    for f, (tr, te) in enumerate(outer.split(X, y)):
        ytr, yte = y[tr], y[te]
        for sc in SCALINGS:
            Xtr, Xte = scale(sc, X[tr], X[te])
            for name, (est, grid) in model_zoo((ytr == 0).sum() / (ytr == 1).sum()).items():
                gs = GridSearchCV(est, grid, scoring="roc_auc", cv=inner, n_jobs=-1).fit(Xtr, ytr)
                best = gs.best_estimator_
                method = "predict_proba" if hasattr(best, "predict_proba") else "decision_function"
                oof = cross_val_predict(best, Xtr, ytr, cv=inner, method=method, n_jobs=-1)
                thr = best_threshold(ytr, oof[:, 1] if oof.ndim == 2 else oof)
                s = score(best, Xte)
                S, P, T, A = pooled.setdefault((name, sc), ([], [], [], []))
                S.append(s); P.append((s >= thr).astype(int)); T.append(yte); A.append(roc_auc_score(yte, s))
        print(f"fold {f + 1}/5 done  {time.time() - t0:.0f}s", flush=True)

    rows, rocs = [], {}
    for (name, sc), (S, P, T, A) in pooled.items():
        s, p, t = np.concatenate(S), np.concatenate(P), np.concatenate(T)
        rows.append({"model": name, "scaling": sc, "accuracy": accuracy_score(t, p),
                     "precision_w": precision_score(t, p, average="weighted", zero_division=0),
                     "recall_w": recall_score(t, p, average="weighted", zero_division=0),
                     "f1_w": f1_score(t, p, average="weighted", zero_division=0),
                     "precision_bankrupt": precision_score(t, p, zero_division=0),
                     "recall_bankrupt": recall_score(t, p, zero_division=0),
                     "f1_bankrupt": f1_score(t, p, zero_division=0),
                     "auc": float(np.mean(A)), "auc_std_over_folds": float(np.std(A))})
        fpr, tpr, _ = roc_curve(t, s)
        rocs[f"{name}|{sc}"] = (fpr, tpr, rows[-1]["auc"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "model_comparison.csv", index=False)
    np.save(OUT / "roc.npy", rocs, allow_pickle=True)
    cols = ["accuracy", "f1_w", "precision_bankrupt", "recall_bankrupt", "f1_bankrupt", "auc"]
    print(df.groupby("model")[cols].mean().round(3).sort_values("auc", ascending=False).to_string())


if __name__ == "__main__":
    main()
