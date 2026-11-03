# Bankruptcy prediction with ML and LLMs

Reproduces the method of *Advanced Default Risk Prediction in Small and Medium-Sized Enterprises Using Large
Language Models* (Appl. Sci. 2025, 15, 2733) for the question **"will this company go bankrupt within a year?"**.
The paper's own data (non-financial SME features) is proprietary, so the method is run on three public datasets
of **financial ratios**. Start with `notebooks/Project_Walkthrough.ipynb` (all results, with outputs) or `SUMMARY.md`.

| Dataset | Evaluation | Role |
|---|---|---|
| **American** listed firms 1999-2018 (78,682 firm-years, 609 failed firms) | train <= 2014, test 2015-2018 | main, honest result |
| **Taiwan** Economic Journal 1999-2009 (6,819 firms, 220 bankrupt) | stratified 5-fold CV | independent confirmation |
| **Polish** companies (UCI, 43,405 firm-years, 5 files) | random 80/20 split | benchmark; likely leakage (no firm IDs) |

**Polish horizons.** File `Nyear` ("horizon N" in the code) has the label "bankrupt after 6-N years": `1year` = 5 years ahead (hardest), `5year` = 1 year ahead. An early assumption in this project had this backwards; tables are labelled with the correct direction.

## Folder structure
```
LLMs Project/
├── README.md, SUMMARY.md, requirements.txt, run_all.sh
├── app.py                         Streamlit demo (enter a company's figures -> risk score + reasons)
├── notebooks/Project_Walkthrough.ipynb
├── data/   polish/ american/ taiwan/ processed/
├── models/ bankruptcy_lgbm_american.joblib (final model for the demo), LoRA adapters
├── colab/  fine-tuning on a free Colab GPU (notebook + script + data files)
├── src/                           ALL the code (see below)
├── results/                       tables, figures, logs; results/american/ and results/taiwan/
└── archive/                       old, unused downloader
```

## Code (`src/`)
| Step | Polish | American | Taiwan |
|---|---|---|---|
| Load / preprocess | `load_data.py`, `preprocess.py`, `columns.py`, `plot_correlations.py` | `american_prepare.py` (label + ratios) | built into `taiwan_models.py` |
| Regression, six key ratios | `regression_analysis.py` | `american_regression.py` | - |
| 7 models x 3 scalings | `run_models.py`, `report_models.py` | `american_models.py` | `taiwan_models.py` |
| LLM prompt learning (Ollama) | `llm_prompt.py`, `report_llm.py` | `american_llm.py`, `american_report_llm.py` | - |
| LLM fine-tuning (LoRA) | `llm_finetune.py` (Mac, small runs only) | `american_finetune.py` (Mac) / `colab/american_finetune_colab.py` (GPU) | - |

Cross-cutting: `combined_results.py` (Altman Z baseline + all-dataset table), `confidence_intervals.py` (bootstrap),
`cross_dataset.py` (train on one dataset, test on another), `explain_shap.py`, `nn_balanced.py` (neural-net fix),
`train_final.py` + `predict.py` (final model and scoring), `train_final_polish.py` (older Polish demo model).

## How to run
```
sh run_all.sh               # everything except fine-tuning (LLM prompting alone is ~1 hour)
sh run_all.sh --no-llm      # all ML phases, no LLM (about 30 minutes)
.venv/bin/streamlit run app.py
```
Fine-tuning needs a GPU: follow `colab/README.txt` (about 10 minutes on a free T4), then put the downloaded
`llm_finetune_metrics.json` into `results/american/`. The Mac version stalled from lack of free memory.
Use `.venv/bin/python -m pip ...` (the venv's `pip` launcher is broken).

## Key results (AUC; see `results/combined_summary.csv`)
Tree models (LightGBM / Random Forest / XGBoost) are best on all three datasets: American about 0.91, Taiwan 0.94,
Polish 0.96-0.97 (least trustworthy: the boosted trees score the same 5 years ahead as 1 year ahead, a sign of leakage). Altman Z-score: American 0.79, Polish 0.62 (5 yrs ahead) / 0.72 (1 yr ahead). American LLM on the same 500 firms:
prompting 0.85, fine-tuned 0.90 (`results/american/table9_llm.csv`). Models transfer between American and Taiwan
(`results/cross_dataset.csv`).

## Honest limitations
- Different data from the paper (financial ratios only): the method is reproduced, not the paper's findings.
- American failures are ~1% of firm-years: ranking is good (AUC ~0.91) but precision is low (bankrupt F1 ~0.23 on the full test years).
- Test sets hold few failures (119 / 220 / 54): recall and F1 have wide intervals (`results/confidence_intervals.csv`).
- Polish data has no firm IDs or dates (likely leakage); its 64 column definitions were checked against the official UCI list and match. The Polish LLM prompt says "within the next year" but that run used the 5-years-ahead file.
- The LLM test sample oversamples failures (10%): compare LLM methods with each other, not with full-test tables.
- LLM results are single runs, not repeated.
