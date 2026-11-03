"""SHAP explanations for the American LightGBM model (best params, signed-log ratios, train <= 2014).

Shows which ratios push a firm's bankruptcy score up or down:
  results/american/shap_importance.csv       mean |SHAP| per ratio on the 2015-2018 test firms
  results/american/figures/shap_summary.png  beeswarm (red = high ratio value)
  results/american/figures/shap_bar.png      top-12 bar chart
  results/american/shap_example_failed.csv   top drivers for three firms that actually failed

Run:  .venv/bin/python src/explain_shap.py
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from american_models import OUT, SEED, load

warnings.filterwarnings("ignore")


def main() -> None:
    (OUT / "figures").mkdir(exist_ok=True)
    a = load()
    best = json.load(open(OUT / "best_params_bankrupt.json"))["LightGBM|log"]
    model = LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1, **best)
    Xtr = pd.DataFrame(a["X"]["log"]["train"], columns=a["features"])
    Xte = pd.DataFrame(a["X"]["log"]["test"], columns=a["features"])
    model.fit(Xtr, a["y_train"])

    sv = shap.TreeExplainer(model).shap_values(Xte)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = sv[..., 1] if sv.ndim == 3 else sv
    imp = pd.Series(np.abs(sv).mean(0), index=a["features"]).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(OUT / "shap_importance.csv")
    print("Top 12 ratios by mean |SHAP|:\n", imp.head(12).round(3).to_string())

    shap.summary_plot(sv, Xte, max_display=12, show=False)
    plt.title("What drives bankruptcy risk (American, 2015-2018)")
    plt.savefig(OUT / "figures" / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    top = imp.head(12)[::-1]
    plt.figure(figsize=(7, 5)); plt.barh(top.index, top.values, color="steelblue")
    plt.xlabel("mean |SHAP value|"); plt.title("Top ratios by influence")
    plt.savefig(OUT / "figures" / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    rows = []
    for i in np.where(a["y_test"] == 1)[0][:3]:
        order = np.argsort(-np.abs(sv[i]))[:4]
        rows += [{"firm_row": int(i), "ratio": a["features"][j], "shap": round(float(sv[i][j]), 3)} for j in order]
    pd.DataFrame(rows).to_csv(OUT / "shap_example_failed.csv", index=False)
    print("\nExample failed firms, top drivers:\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
