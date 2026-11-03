# One-page summary

**Question.** Can we predict whether a company goes bankrupt within a year, following the method of Huang et al. (2025)
(seven ML models x three standardisations, regression-based key-ratio selection, LLM prompting and fine-tuning)?

**Data.** The paper's data is proprietary, so three public financial-ratio datasets were used: American listed firms
(main result, time-based split 2015-2018 held out), Taiwan (5-fold CV, confirmation) and Polish (benchmark, likely leakage).
The American file marks a failed company "failed" in every year; the label was rebuilt so only the last year before failure is positive.

**Results (AUC on held-out data).**
| | American | Taiwan | Polish, 5 yrs ahead | Polish, 1 yr ahead |
|---|---|---|---|---|
| LightGBM | 0.91 | 0.94 | 0.97 | 0.96 |
| Random Forest / XGBoost | 0.91 / 0.91 | 0.94 / 0.94 | 0.89 / 0.97 | 0.91 / 0.97 |
| Logistic regression | 0.78 | 0.92 | 0.72 | 0.81 |
| Neural net | 0.69 | 0.73 | 0.59 | 0.79 |
| Altman Z-score baseline | 0.79 | n/a | 0.62 | 0.72 |

(Model rows are averages over the three standardisations; see `results/combined_summary.csv`. MinMax / Z-score hurt the weaker models,
so per-model best-scaling numbers are higher for them.)

**LLM (American, same 500 test firms).** Prompting AUC 0.85 (the paper's cut-off of 5 flags 73% of firms and must be calibrated);
fine-tuned Qwen2.5-0.5B with LoRA AUC 0.90 and the best bankrupt-class F1 (0.53 on this oversampled sample): on par with the best ML model (0.91).

**Also done.** Bootstrap confidence intervals, American <-> Taiwan transfer (models generalise, logistic regression best),
SHAP explanations (market value vs. debt is the dominant driver), a balanced neural net, and a Streamlit demo that scores a company and explains why.

**Caveats.** Financial ratios only (not the paper's non-financial features); ~1% base rate gives low precision (F1 ~0.23 on the full American test years);
few failures per test set, so wide intervals on recall; Polish data may leak; LLM results are single runs on oversampled test samples.
