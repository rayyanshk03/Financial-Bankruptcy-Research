"""Self-contained version of src/american_finetune.py for Google Colab (GPU).

Needs in the same folder: american_ratios.csv, key_features.json, test_firms.csv.
Trains Qwen2.5-0.5B + LoRA (2-class head) on 2,000 train-year firms (<= 2014), picks the cut-off on
600 other train firms, scores the same 500 test firms (2015-2018) as the prompt-learning run.
Writes llm_finetune_metrics.json.   Run:  python american_finetune_colab.py
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE, SEED, SPLIT_YEAR = "Qwen/Qwen2.5-0.5B-Instruct", 42, 2014
N_TRAIN, POS_FRAC, EPOCHS, BS, LR = 2000, 0.20, 1, 16, 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEY = json.load(open("key_features.json"))
COLS = KEY["key_features"]
torch.manual_seed(SEED)


def to_text(row):
    lines = [f"- {n} ({KEY['definitions'][n]}): {row[n]:.3g}" for n in COLS]
    return "Financial indicators of a company:\n" + "\n".join(lines) + "\nWill this company go bankrupt within the next year?"


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def metrics(y, pred):
    return {"accuracy": accuracy_score(y, pred),
            "precision_w": precision_score(y, pred, average="weighted", zero_division=0),
            "recall_w": recall_score(y, pred, average="weighted", zero_division=0),
            "f1_w": f1_score(y, pred, average="weighted", zero_division=0),
            "precision_bankrupt": precision_score(y, pred, zero_division=0),
            "recall_bankrupt": recall_score(y, pred, zero_division=0),
            "f1_bankrupt": f1_score(y, pred, zero_division=0)}


def batches(texts, labels, tok, bs):
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256)
        yield {k: v.to(DEVICE) for k, v in enc.items()}, torch.tensor(labels[i:i + bs]).to(DEVICE)


@torch.no_grad()
def predict_proba(model, tok, texts, bs=32):
    model.eval()
    return np.concatenate([torch.softmax(model(**e).logits.float(), -1)[:, 1].cpu().numpy()
                           for e, _ in batches(texts, [0] * len(texts), tok, bs)])


d = pd.read_csv("american_ratios.csv")
tr = d[d.year <= SPLIT_YEAR]
d[COLS] = d[COLS].fillna(tr[COLS].median())
tr = d[d.year <= SPLIT_YEAR]
pos = tr[tr.bankrupt == 1].sample(min(int(tr.bankrupt.sum()), round(N_TRAIN * POS_FRAC)), random_state=SEED)
neg = tr[tr.bankrupt == 0].sample(N_TRAIN - len(pos), random_state=SEED)
fit = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
texts, labels = [to_text(r) for _, r in fit.iterrows()], fit.bankrupt.tolist()

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, torch_dtype=torch.float32)
model.config.pad_token_id = tok.pad_token_id
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])).to(DEVICE)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=LR)
print(f"training on {len(texts)} firms ({sum(labels)} bankrupt), device {DEVICE}", flush=True)

t0, step = time.time(), 0
for ep in range(EPOCHS):
    model.train()
    order = np.random.RandomState(SEED + ep).permutation(len(texts))
    for enc, y in batches([texts[i] for i in order], [labels[i] for i in order], tok, BS):
        loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
        loss.backward(); opt.step(); opt.zero_grad(); step += 1
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)

rest = tr.drop(fit.index)
held = pd.concat([rest[rest.bankrupt == 1], rest[rest.bankrupt == 0].sample(600 - int(rest.bankrupt.sum()), random_state=SEED)])
thr = best_threshold(held.bankrupt.values, predict_proba(model, tok, [to_text(r) for _, r in held.iterrows()]))
test = pd.read_csv("test_firms.csv")
s = predict_proba(model, tok, [to_text(r) for _, r in d.loc[test["index"].values].iterrows()])
res = {**metrics(test.bankrupt, (s >= thr).astype(int)), "auc": roc_auc_score(test.bankrupt, s),
       "threshold": thr, "n_train": len(texts), "epochs": EPOCHS}
json.dump({k: float(v) for k, v in res.items()}, open("llm_finetune_metrics.json", "w"), indent=2)
print(json.dumps({k: round(float(v), 4) for k, v in res.items()}, indent=2))
