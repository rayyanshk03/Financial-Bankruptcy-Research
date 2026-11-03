"""Neural-net fix: the plain MLP has no class weights and ignores the rare bankrupt class.

OversampledMLP duplicates bankrupt rows (random oversampling, inside each training fold only) until they
are `ratio` x the number of healthy rows, then trains the same MLP as in run_models.py.
Compared with the original NN on the American data (train <= 2014, test 2015-2018) for all three scalings.

Run:  .venv/bin/python src/nn_balanced.py   ->  results/american/nn_balanced.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from american_models import OUT, SEED, SCALINGS, load
from run_models import best_threshold

warnings.filterwarnings("ignore")


class OversampledMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(64,), alpha=1e-4, ratio=1.0, max_iter=60):
        self.hidden_layer_sizes, self.alpha, self.ratio, self.max_iter = hidden_layer_sizes, alpha, ratio, max_iter

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        pos, neg = X[y == 1], X[y == 0]
        pos = resample(pos, n_samples=max(len(pos), int(self.ratio * len(neg))), replace=True, random_state=SEED)
        Xb, yb = np.vstack([neg, pos]), np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  max_iter=self.max_iter, random_state=SEED).fit(Xb, yb)
        self.classes_ = self.mlp_.classes_
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


def main() -> None:
    a = load()
    ytr, yte = a["y_train"], a["y_test"]
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    old = pd.read_csv(OUT / "model_comparison_bankrupt.csv").query("model == 'NN'").set_index("scaling")
    rows = []
    for sc in SCALINGS:
        Xtr, Xte = a["X"][sc]["train"], a["X"][sc]["test"]
        gs = GridSearchCV(OversampledMLP(), {"hidden_layer_sizes": [(64,), (64, 32)], "alpha": [1e-4, 1e-2],
                                              "ratio": [0.1, 0.5]}, scoring="roc_auc", cv=cv, n_jobs=-1).fit(Xtr, ytr)
        best = gs.best_estimator_
        thr = best_threshold(ytr, cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1])
        s = best.predict_proba(Xte)[:, 1]
        pred = (s >= thr).astype(int)
        rows.append({"scaling": sc, "auc_before": old.loc[sc, "auc"], "auc_after": roc_auc_score(yte, s),
                     "recall_before": old.loc[sc, "recall_bankrupt"], "recall_after": recall_score(yte, pred),
                     "f1_before": old.loc[sc, "f1_bankrupt"], "f1_after": f1_score(yte, pred),
                     "best_params": str(gs.best_params_)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "nn_balanced.csv", index=False)
    print(df.drop(columns="best_params").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
