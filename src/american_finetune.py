"""American dataset, Phase B6: LoRA fine-tuning of a small LLM (paper Sec. 3.2.4).

Same setup as llm_finetune.py (Qwen2.5-0.5B + LoRA + 2-class head, text prompt of the six key ratios),
trained on a class-balanced sample of TRAIN-year firms (<= 2014) and scored on the same 500 test firms
(2015-2018) used for the prompt-learning run. The cut-off comes from 600 other train firms.

Run:  .venv/bin/python src/american_finetune.py                 # 2,000 firms, 1 epoch
      .venv/bin/python src/american_finetune.py --n_train 160   # smoke test
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import llm_finetune as ft
from american_llm import KEY, OUT, ROOT
from american_models import SPLIT_YEAR
from llm_prompt import SEED, fmt, metrics
from run_models import best_threshold


def to_text(row: pd.Series) -> str:
    lines = [f"- {n} ({KEY['definitions'][n]}): {fmt(row[n])}" for n in KEY["key_features"]]
    return ("Financial indicators of a company:\n" + "\n".join(lines) +
            "\nWill this company go bankrupt within the next year?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--pos_frac", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    torch.manual_seed(SEED)

    d = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    cols = KEY["key_features"]
    tr = d[d.year <= SPLIT_YEAR].copy()
    med = tr[cols].median()
    d[cols] = d[cols].fillna(med)
    tr = d[d.year <= SPLIT_YEAR]
    pos = tr[tr.bankrupt == 1].sample(min(int(tr.bankrupt.sum()), round(a.n_train * a.pos_frac)), random_state=SEED)
    neg = tr[tr.bankrupt == 0].sample(a.n_train - len(pos), random_state=SEED)
    fit = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
    texts, labels = [to_text(r) for _, r in fit.iterrows()], fit.bankrupt.tolist()

    tok = AutoTokenizer.from_pretrained(ft.BASE)
    model = AutoModelForSequenceClassification.from_pretrained(ft.BASE, num_labels=2, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])).to(ft.DEVICE)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    print(f"training on {len(texts)} firms ({sum(labels)} bankrupt), device {ft.DEVICE}", flush=True)

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        model.train()
        order = np.random.RandomState(SEED + ep).permutation(len(texts))
        for enc, y in ft.batches([texts[i] for i in order], [labels[i] for i in order], tok, a.bs):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 5 == 0:
                print(f"epoch {ep + 1} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(ROOT / "models" / "qwen_lora_american")

    rest = tr.drop(fit.index)
    held = pd.concat([rest[rest.bankrupt == 1], rest[rest.bankrupt == 0].sample(600 - int(rest.bankrupt.sum()), random_state=SEED)])
    s_held = ft.predict_proba(model, tok, [to_text(r) for _, r in held.iterrows()])
    thr = best_threshold(held.bankrupt.values, s_held)
    test = pd.read_json(OUT / "llm_prompt_test_n500.jsonl", lines=True)
    Xte = d.loc[test["index"].values]
    s = ft.predict_proba(model, tok, [to_text(r) for _, r in Xte.iterrows()])
    res = {**metrics(test.bankrupt, (s >= thr).astype(int)), "auc": roc_auc_score(test.bankrupt, s),
           "threshold": thr, "n_train": len(texts), "epochs": a.epochs}
    json.dump({k: float(v) for k, v in res.items()}, open(OUT / "llm_finetune_metrics.json", "w"), indent=2)
    print(json.dumps({k: round(float(v), 4) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
