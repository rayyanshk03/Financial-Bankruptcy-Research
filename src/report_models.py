"""Phase 4 report: paper-style Table 7 / Table 8 and ROC curves from results/model_comparison_h*.csv.

Run:  .venv/bin/python src/report_models.py 1
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
FIG = OUT / "figures"
ORDER = ["Logistic", "SVM", "XGBoost", "DT", "RF", "NN", "LightGBM"]


def report(h: int) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / f"model_comparison_h{h}.csv")
    df["model"] = pd.Categorical(df["model"], ORDER, ordered=True)

    t7 = df.sort_values(["model", "scaling"])[
        ["model", "scaling", "accuracy", "precision_w", "recall_w", "f1_w",
         "precision_bankrupt", "recall_bankrupt", "f1_bankrupt", "auc"]]
    t7.round(4).to_csv(OUT / f"table7_h{h}.csv", index=False)

    t8 = df.groupby("model", observed=True)[
        ["accuracy", "precision_w", "recall_w", "f1_w", "f1_bankrupt", "recall_bankrupt", "auc"]
    ].mean().round(4)
    t8.to_csv(OUT / f"table8_h{h}.csv")

    print(f"=== Horizon {h}: Table 7 (every model x scaling) ===")
    print(t7.round(3).to_string(index=False))
    print(f"\n=== Horizon {h}: Table 8 (mean over the 3 scalings) ===")
    print(t8.round(3).to_string())
    print("\nBest scaling per model by AUC:")
    best = df.loc[df.groupby("model", observed=True)["auc"].idxmax()][["model", "scaling", "auc", "f1_bankrupt"]]
    print(best.round(3).to_string(index=False))

    rocs = np.load(OUT / f"roc_h{h}.npy", allow_pickle=True).item()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, ORDER):
        sc = best[best.model == m]["scaling"].iloc[0]
        fpr, tpr, auc = rocs[f"{m}|{sc}"]
        ax.plot(fpr, tpr, color="darkorange", label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], "--", color="navy", lw=1)
        ax.set_title(f"{m} ({sc})", fontsize=10)
        ax.legend(loc="lower right")
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("True positive rate")
    for ax in axes[1, :]:
        ax.set_xlabel("False positive rate")
    fig.suptitle(f"ROC curves, horizon {h} (best scaling per model)")
    fig.savefig(FIG / f"roc_h{h}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["1"]:
        report(int(a))
