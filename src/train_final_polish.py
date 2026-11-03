"""Final model for the demo: LightGBM on all 64 (kept) ratios, signed-log scaling, horizon 1.

Uses the best hyper-parameters from Phase 4 and the threshold chosen on out-of-fold TRAIN scores.
Saves everything needed to score a new company to models/bankruptcy_lgbm_h1.joblib.

Run:  .venv/bin/python src/train_final.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from preprocess import PROCESSED, prepare, signed_log
from run_models import SEED, best_threshold

ROOT = Path(__file__).resolve().parent.parent
HORIZON = 1


def main() -> None:
    p = prepare(HORIZON)
    best = json.load(open(ROOT / "results" / f"best_params_h{HORIZON}.json"))["LightGBM|log"]
    model = LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1, **best)
    Xtr, ytr = p["X_train"]["log"], p["y_train"]
    oof = cross_val_predict(model, Xtr, ytr, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    thr = best_threshold(ytr, oof)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(p["y_test"], model.predict_proba(p["X_test"]["log"])[:, 1])

    # the imputer is refit on the raw train rows so new companies with blanks can be scored
    imputer = SimpleImputer(strategy="median").fit(p["X_train"]["raw"])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "imputer": imputer, "features": p["features"], "threshold": thr,
                 "test_auc": auc, "horizon": HORIZON,
                 "medians": dict(zip(p["features"], imputer.statistics_))},
                ROOT / "models" / f"bankruptcy_lgbm_h{HORIZON}.joblib")
    print(f"saved model; threshold {thr:.3f}; test AUC {auc:.3f}")


if __name__ == "__main__":
    main()
