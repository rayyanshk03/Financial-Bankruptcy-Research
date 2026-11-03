"""Phase 6 (NOTE: horizon 1 = the 1year file, label "bankrupt after 5 years"): fine-tune a small LLM (LoRA) to classify bankruptcy from the six key ratios (paper Sec. 3.2.4).

Same text prompt as Phase 5 (financial indicators in words), but the model is now TRAINED on labelled
firms instead of just prompted. A LoRA-adapted Qwen2.5-0.5B with a 2-class head is trained on a
class-balanced sample of horizon-1 TRAIN firms and scored on the same 1,000 test firms used in Phase 5.

Run:  .venv/bin/python src/llm_finetune.py                 # default: 2,000 training firms, 1 epoch
      .venv/bin/python src/llm_finetune.py --n_train 200    # quick smoke test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_prompt import KEY, SEED, fmt, metrics
from preprocess import prepare
from run_models import best_threshold

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def to_text(row: pd.Series) -> str:
    lines = [f"- {n} ({KEY['definitions'][n]}): {fmt(row[n])}" for n in KEY["key_features"]]
    return ("Financial indicators of a company:\n" + "\n".join(lines) +
            "\nWill this company go bankrupt within the next year?")


def batches(texts, labels, tok, bs):
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256)
        yield {k: v.to(DEVICE) for k, v in enc.items()}, torch.tensor(labels[i:i + bs]).to(DEVICE)


@torch.no_grad()
def predict_proba(model, tok, texts, bs=16):
    model.eval()
    out = []
    for enc, _ in batches(texts, [0] * len(texts), tok, bs):
        out.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--pos_frac", type=float, default=0.25, help="share of bankrupt firms in the training sample")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    torch.manual_seed(SEED)

    p = prepare(1)
    Xtr, ytr = p["X_train"]["raw"][KEY["key_features"]], p["y_train"]
    pos = ytr[ytr == 1].sample(min(int((ytr == 1).sum()), round(a.n_train * a.pos_frac)), random_state=SEED).index
    neg = ytr[ytr == 0].sample(a.n_train - len(pos), random_state=SEED).index
    tr = Xtr.loc[pos.union(neg)].sample(frac=1, random_state=SEED)
    texts, labels = [to_text(r) for _, r in tr.iterrows()], ytr.loc[tr.index].tolist()

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])).to(DEVICE)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    print(f"training on {len(texts)} firms ({sum(labels)} bankrupt), device {DEVICE}", flush=True)

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        model.train()
        order = np.random.RandomState(SEED + ep).permutation(len(texts))
        for enc, y in batches([texts[i] for i in order], [labels[i] for i in order], tok, a.bs):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                print(f"epoch {ep + 1} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(ROOT / "models" / "qwen_lora_h1")

    # threshold from TRAIN firms (not in the fitting sample), then score the Phase-5 test firms
    held = Xtr.drop(tr.index).sample(600, random_state=SEED)
    s_held = predict_proba(model, tok, [to_text(r) for _, r in held.iterrows()])
    thr = best_threshold(ytr.loc[held.index].values, s_held)
    test = pd.read_json(OUT / "llm_prompt_h1_test_n1000.jsonl", lines=True)
    Xte = p["X_test"]["raw"].loc[test["index"].values, KEY["key_features"]]
    s = predict_proba(model, tok, [to_text(r) for _, r in Xte.iterrows()])
    pred = (s >= thr).astype(int)
    res = {**metrics(test.bankrupt, pred), "auc": roc_auc_score(test.bankrupt, s), "threshold": thr,
           "n_train": len(texts), "epochs": a.epochs}
    json.dump({k: float(v) for k, v in res.items()}, open(OUT / "llm_finetune_metrics_h1.json", "w"), indent=2)
    print(json.dumps({k: round(float(v), 4) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
