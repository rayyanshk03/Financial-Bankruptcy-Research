"""American dataset: 7 models x 3 standardisations with a TIME-BASED split.

Train = firm-years up to 2014, test = 2015-2018, so the models are judged on the future.
Same model zoo, grid search (3-fold ROC-AUC, TRAIN only) and out-of-fold threshold as run_models.py.
The SVM is grid-searched and fitted on a 15,000-row stratified train sample (RBF-SVM is too slow on 66,000 rows).

Run:  .venv/bin/python src/american_models.py            # target = bankrupt next year
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocess import signed_log
from run_models import SEED, SCALINGS, best_threshold, model_zoo, score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "american"
SPLIT_YEAR = 2014
SVM_ROWS = 15000


def load(target: str = "bankrupt") -> dict:
    d = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    feats = [c for c in d.columns if c not in ("company", "year", "bankrupt", "fail_3y")]
    tr, te = d[d.year <= SPLIT_YEAR], d[d.year > SPLIT_YEAR]
    imp = SimpleImputer(strategy="median").fit(tr[feats])
    base = {"train": imp.transform(tr[feats]), "test": imp.transform(te[feats])}
    X = {"log": {k: signed_log(v) for k, v in base.items()}}
    for name, sc in (("minmax", MinMaxScaler()), ("zscore", StandardScaler())):
        sc.fit(base["train"])
        X[name] = {k: sc.transform(v) for k, v in base.items()}
    return {"X": X, "y_train": tr[target].values, "y_test": te[target].values, "features": feats}


def main(target: str = "bankrupt") -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(target)
    ytr, yte = data["y_train"], data["y_test"]
    print(f"target {target}: train {len(ytr)} ({ytr.sum()} pos), test {len(yte)} ({yte.sum()} pos)", flush=True)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    sub = train_test_split(np.arange(len(ytr)), train_size=SVM_ROWS, stratify=ytr, random_state=SEED)[0]
    rows, rocs, best_params = [], {}, {}
    for sc in SCALINGS:
        Xtr, Xte = data["X"][sc]["train"], data["X"][sc]["test"]
        for name, (est, grid) in model_zoo((ytr == 0).sum() / (ytr == 1).sum()).items():
            t0 = time.time()
            Xf, yf = (Xtr[sub], ytr[sub]) if name == "SVM" else (Xtr, ytr)
            gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(Xf, yf)
            best = gs.best_estimator_
            method = "predict_proba" if hasattr(best, "predict_proba") else "decision_function"
            oof = cross_val_predict(best, Xf, yf, cv=cv, method=method, n_jobs=-1)
            thr = best_threshold(yf, oof[:, 1] if oof.ndim == 2 else oof)
            s = score(best, Xte)
            pred = (s >= thr).astype(int)
            rows.append({
                "model": name, "scaling": sc, "accuracy": accuracy_score(yte, pred),
                "precision_w": precision_score(yte, pred, average="weighted", zero_division=0),
                "recall_w": recall_score(yte, pred, average="weighted", zero_division=0),
                "f1_w": f1_score(yte, pred, average="weighted", zero_division=0),
                "precision_bankrupt": precision_score(yte, pred, zero_division=0),
                "recall_bankrupt": recall_score(yte, pred, zero_division=0),
                "f1_bankrupt": f1_score(yte, pred, zero_division=0),
                "auc": roc_auc_score(yte, s), "threshold": thr, "cv_auc": gs.best_score_,
                "seconds": round(time.time() - t0, 1)})
            best_params[f"{name}|{sc}"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()}
            fpr, tpr, _ = roc_curve(yte, s)
            rocs[f"{name}|{sc}"] = (fpr, tpr, rows[-1]["auc"])
            r = rows[-1]
            print(f"{sc:7s} {name:9s} auc={r['auc']:.3f} f1_bk={r['f1_bankrupt']:.3f} rec_bk={r['recall_bankrupt']:.3f} ({r['seconds']}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"model_comparison_{target}.csv", index=False)
    json.dump(best_params, open(OUT / f"best_params_{target}.json", "w"), indent=2)
    np.save(OUT / f"roc_{target}.npy", rocs, allow_pickle=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bankrupt")
