Colab steps (about 10 minutes):
1. Go to https://colab.research.google.com  ->  New notebook.
2. Runtime -> Change runtime type -> T4 GPU -> Save.
3. Left sidebar, folder icon -> upload ALL 4 files from this folder:
   american_finetune_colab.py, american_ratios.csv, key_features.json, test_firms.csv
4. In a cell, run:
   !pip -q install peft transformers
   !pip -q uninstall -y torchao
   !python american_finetune_colab.py
5. When it prints the final numbers, download llm_finetune_metrics.json (folder icon, right-click ->
   Download) and put it in  LLMs Project/results/american/
