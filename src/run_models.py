"""Phase 4: seven models x three standardisations (paper Sec. 4.2, Tables 7-8, Fig. 5).

Models: Logistic, SVM, XGBoost, Decision Tree, Random Forest, Neural Net, LightGBM.
Standardisations: log, minmax, zscore.   Hyper-parameters: grid search (3-fold CV, ROC-AUC) on TRAIN only.

Bankruptcy is rare (~4-7%), so beyond the paper's weighted-average metrics we also report the
bankrupt-class precision / recall / F1. The decision threshold is chosen on out-of-fold TRAIN
predictions (max bankrupt-class F1), never on the test set.

Run:  .venv/bin/python src/run_models.py 1          # one horizon
      .venv/bin/python src/run_models.py 1 2 3 4 5  # several
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocess import prepare

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SEED = 42
SCALINGS = ("log", "minmax", "zscore")


def model_zoo(pos_weight: float) -> dict:
    """name -> (estimator, param grid)."""
    return {
        "Logistic": (LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
                     {"C": [0.01, 0.1, 1, 10]}),
        "SVM": (SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
                {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]}),
        "XGBoost": (XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                                  random_state=SEED, n_jobs=1, verbosity=0),
                    {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
        "DT": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
               {"max_depth": [3, 5, 8], "min_samples_leaf": [5, 20]}),
        "RF": (RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
               {"n_estimators": [300], "max_depth": [10, None], "min_samples_leaf": [1, 5]}),
        "NN": (MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED),
               {"hidden_layer_sizes": [(64,), (64, 32)], "alpha": [1e-4, 1e-2]}),
        "LightGBM": (LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=1, verbose=-1),
                     {"n_estimators": [200, 400], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}),
    }


def score(model, X) -> np.ndarray:
    """Continuous risk score (probability if available, else decision function)."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def best_threshold(y, s) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def run_horizon(h: int) -> pd.DataFrame:
    data = prepare(h)
    ytr, yte = data["y_train"].values, data["y_test"].values
    pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rows, rocs, best_params = [], {}, {}

    for sc in SCALINGS:
        Xtr, Xte = data["X_train"][sc].values, data["X_test"][sc].values
        for name, (est, grid) in model_zoo(pos_weight).items():
            t0 = time.time()
            gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(Xtr, ytr)
            best = gs.best_estimator_
            # threshold from out-of-fold train scores (no test leakage)
            method = "predict_proba" if hasattr(best, "predict_proba") else "decision_function"
            oof = cross_val_predict(best, Xtr, ytr, cv=cv, method=method, n_jobs=-1)
            oof = oof[:, 1] if oof.ndim == 2 else oof
            thr = best_threshold(ytr, oof)
            s = score(best, Xte)
            pred = (s >= thr).astype(int)
            rows.append({
                "horizon": h, "model": name, "scaling": sc,
                "accuracy": accuracy_score(yte, pred),
                "precision_w": precision_score(yte, pred, average="weighted", zero_division=0),
                "recall_w": recall_score(yte, pred, average="weighted", zero_division=0),
                "f1_w": f1_score(yte, pred, average="weighted", zero_division=0),
                "precision_bankrupt": precision_score(yte, pred, zero_division=0),
                "recall_bankrupt": recall_score(yte, pred, zero_division=0),
                "f1_bankrupt": f1_score(yte, pred, zero_division=0),
                "auc": roc_auc_score(yte, s), "threshold": thr,
                "cv_auc": gs.best_score_, "seconds": round(time.time() - t0, 1),
            })
            best_params[f"{name}|{sc}"] = {k: (list(v) if isinstance(v, tuple) else v)
                                           for k, v in gs.best_params_.items()}
            fpr, tpr, _ = roc_curve(yte, s)
            rocs[f"{name}|{sc}"] = (fpr, tpr, rows[-1]["auc"])
            print(f"h{h} {sc:7s} {name:9s} auc={rows[-1]['auc']:.3f} "
                  f"f1_bk={rows[-1]['f1_bankrupt']:.3f} rec_bk={rows[-1]['recall_bankrupt']:.3f} "
                  f"({rows[-1]['seconds']}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"model_comparison_h{h}.csv", index=False)
    json.dump(best_params, open(OUT / f"best_params_h{h}.json", "w"), indent=2)
    np.save(OUT / f"roc_h{h}.npy", rocs, allow_pickle=True)
    return df


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for h in [int(a) for a in sys.argv[1:]] or [1]:
        run_horizon(h)
