"""Phase 2: Pearson correlation heatmaps before / after log scaling (paper Figs. 2-3).

Run:  .venv/bin/python src/plot_correlations.py
Saves PNGs to results/figures/ and prints how much log scaling raises feature-label correlation.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from preprocess import prepare

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
TOP = 20


def corr_with_label(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    return X.corrwith(y)  # Pearson


def heatmap(ax, X, y, cols, title):
    c = X[cols].assign(bankrupt=y).corr()
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(c)), c.columns, fontsize=6)
    ax.set_title(title, fontsize=10)
    return im


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = []
    for h in range(1, 6):
        p = prepare(h)
        y = p["y_train"]
        raw, log = p["X_train"]["raw"], p["X_train"]["log"]
        c_raw, c_log = corr_with_label(raw, y).abs(), corr_with_label(log, y).abs()
        summary.append({"horizon": h, "mean_abs_corr_raw": c_raw.mean(),
                        "mean_abs_corr_log": c_log.mean(),
                        "max_abs_corr_raw": c_raw.max(), "max_abs_corr_log": c_log.max()})
        if h == 1:  # heatmaps for the main horizon, same columns in both panels
            cols = c_log.sort_values(ascending=False).head(TOP).index.tolist()
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            heatmap(axes[0], raw, y, cols, "Raw (no standardisation)")
            im = heatmap(axes[1], log, y, cols, "After log scaling")
            fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
            fig.suptitle(f"Horizon 1 - top {TOP} features by |corr| with bankruptcy (train set)")
            fig.savefig(FIG / "corr_heatmap_raw_vs_log_h1.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            order = cols[::-1]
            pos = range(len(order))
            ax.barh([i + 0.2 for i in pos], c_raw[order], height=0.4, label="raw")
            ax.barh([i - 0.2 for i in pos], c_log[order], height=0.4, label="log")
            ax.set_yticks(list(pos), order, fontsize=7)
            ax.set_xlabel("|Pearson r| with bankrupt")
            ax.legend()
            ax.set_title("Feature-label correlation: raw vs log (horizon 1)")
            fig.savefig(FIG / "label_corr_raw_vs_log_h1.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    s = pd.DataFrame(summary).set_index("horizon").round(4)
    s.to_csv(ROOT / "results" / "label_correlation_summary.csv")
    print(s.to_string())
    print("\nNote: MinMax and Z-score are linear, so Pearson r is identical to raw; only log changes it.")


if __name__ == "__main__":
    main()
