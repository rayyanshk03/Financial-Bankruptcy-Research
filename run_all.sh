#!/bin/sh
# Run the pipeline in order. From the project root:  sh run_all.sh
# The LLM prompt phases need Ollama running with qwen2.5:7b (about 1 hour); skip them with:  sh run_all.sh --no-llm
# Fine-tuning is not run here (needs a GPU): see colab/README.txt.
set -e
PY=.venv/bin/python

# --- Polish data ---
$PY src/load_data.py            # .arff -> data/processed/polish_all.csv
$PY src/preprocess.py           # dedupe, split, impute, log/MinMax/Z-score
$PY src/plot_correlations.py
$PY src/regression_analysis.py  # six key ratios
$PY src/run_models.py 1 2 3 4 5 # 7 models x 3 scalings x 5 horizons
$PY src/report_models.py 1 2 3 4 5

# --- American data (main result) ---
$PY src/american_prepare.py     # next-year label + 26 ratios -> data/american/american_ratios.csv
$PY src/american_regression.py  # six key ratios -> results/american/key_features.json
$PY src/american_models.py      # 7 models x 3 scalings, train <= 2014 / test 2015-2018

# --- Taiwan data ---
$PY src/taiwan_models.py        # 5-fold CV

# --- Extras ---
$PY src/combined_results.py     # Altman baseline + all-dataset table
$PY src/confidence_intervals.py
$PY src/cross_dataset.py
$PY src/explain_shap.py
$PY src/nn_balanced.py
$PY src/train_final.py          # final model for app.py

if [ "$1" != "--no-llm" ]; then
  $PY src/llm_prompt.py --n 400 --split train --pos_frac 0.15       # Polish LLM: calibration
  $PY src/llm_prompt.py --n 1000 --split test                       # Polish LLM: test
  $PY src/report_llm.py
  $PY src/american_llm.py --n 300 --split train --pos_frac 0.15     # American LLM: calibration
  $PY src/american_llm.py --n 500 --split test --pos_frac 0.10      # American LLM: test
  $PY src/american_report_llm.py
fi
