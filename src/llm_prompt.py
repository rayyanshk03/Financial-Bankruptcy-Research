"""Phase 5 (NOTE: horizon 1 = the 1year file, label "bankrupt after 5 years"; the template text still says "within the next year"): LLM prompt learning for bankruptcy risk (paper Sec. 3.2.3 / 4.3, Fig. 6).

The paper's template is adapted: the six non-financial indicators become the six key financial
ratios chosen in Phase 3 (results/key_features.json). The LLM rates risk 1-10; >5 = bankrupt.
Each firm is asked three times and the majority answer is the prediction.

Run:  .venv/bin/python src/llm_prompt.py            # full run, horizon 1
      .venv/bin/python src/llm_prompt.py --n 12     # quick smoke test
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from preprocess import prepare

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
SEED = 42
N_VOTES = 3
KEY = json.load(open(OUT / "key_features.json"))

TEMPLATE = """【Task】: As a professional financial analyst, please conduct a comprehensive analysis of the various financial indicators of the following company to estimate whether the company will go bankrupt within the next year. The risk of bankruptcy is graded on a scale of 1-10, with a higher number indicating a higher risk. Please maintain an objective and impartial attitude and make a serious and responsible judgment.

Below are the financial indicators of the company:
{indicators}

【Assessment Process】:
1. Firstly, focus on the key indicators, especially leverage and liquidity, to analyze the company's financial condition comprehensively. It is essential to weigh the pros and cons. Although some indicators have a positive impact on the company, the negative effects of other indicators should also be considered;
2. Based on the analysis results, determine an accurate risk rating for the company;
3. Finally, draw a precise conclusion. If the company's risk rating exceeds 5, it is considered that the company will go bankrupt within the next year; if it is 5 or less, it is considered that it will not.
Answer with only the final line, exactly in the form "【Result】: <rating>" where <rating> is an integer from 1 to 10.
【Result】:"""


def fmt(v: float) -> str:
    return f"{v:.3g}"


def build_prompt(row: pd.Series) -> str:
    lines = [f"- {name} ({KEY['definitions'][name]}): {fmt(row[name])}" for name in KEY["key_features"]]
    return TEMPLATE.format(indicators="\n".join(lines))


def parse_rating(text: str) -> int | None:
    m = re.findall(r"【?Result】?\s*[:：]\s*\**\s*(\d{1,2})", text)
    if not m:
        m = re.findall(r"\b(10|[1-9])\b\s*$", text.strip())
    if not m:
        return None
    r = int(m[-1])
    return r if 1 <= r <= 10 else None


def ask(prompt: str, temperature: float, seed: int, model: str = MODEL) -> tuple[int | None, str]:
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "seed": seed, "num_predict": 12}}
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA, json=body, timeout=300)
            r.raise_for_status()
            text = r.json()["response"]
            return parse_rating(text), text
        except Exception as e:  # network hiccup / model busy
            time.sleep(2 * (attempt + 1))
    return None, ""


def predict_firm(prompt: str, idx: int) -> dict:
    ratings, texts = [], []
    for v in range(N_VOTES):
        rating, text = ask(prompt, temperature=0.7, seed=SEED + 1000 * v + idx)
        ratings.append(rating)
        texts.append(text)
    votes = [("bankrupt" if r > 5 else "healthy") for r in ratings if r is not None]
    if not votes:
        return {"ratings": ratings, "pred": 0, "valid_votes": 0, "texts": texts}
    pred = int(Counter(votes).most_common(1)[0][0] == "bankrupt")
    return {"ratings": ratings, "pred": pred, "valid_votes": len(votes), "texts": texts}


def sample_firms(n: int, horizon: int = 1, split: str = "test", pos_frac: float = 0.054) -> pd.DataFrame:
    """n firms from the given split; pos_frac of them bankrupt (natural rate is ~4%)."""
    p = prepare(horizon)
    X, y = p[f"X_{split}"]["raw"], p[f"y_{split}"]
    pos = y[y == 1].sample(min(int((y == 1).sum()), round(n * pos_frac)), random_state=SEED).index
    neg = y[y == 0].sample(n - len(pos), random_state=SEED).index
    idx = pos.union(neg)
    df = X.loc[idx, KEY["key_features"]].copy()
    df["bankrupt"] = y.loc[idx]
    return df.sample(frac=1, random_state=SEED)  # shuffle


def metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred),
            "precision_w": precision_score(y, pred, average="weighted", zero_division=0),
            "recall_w": recall_score(y, pred, average="weighted", zero_division=0),
            "f1_w": f1_score(y, pred, average="weighted", zero_division=0),
            "precision_bankrupt": precision_score(y, pred, zero_division=0),
            "recall_bankrupt": recall_score(y, pred, zero_division=0),
            "f1_bankrupt": f1_score(y, pred, zero_division=0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--pos_frac", type=float, default=0.054)
    a = ap.parse_args()
    tag = f"h{a.horizon}_{a.split}_n{a.n}"

    firms = sample_firms(a.n, a.horizon, a.split, a.pos_frac)
    print(f"{len(firms)} firms ({a.split}), {int(firms.bankrupt.sum())} bankrupt; model {MODEL}", flush=True)
    print("\n--- example prompt ---\n" + build_prompt(firms.iloc[0]) + "\n", flush=True)

    rows, t0 = [], time.time()
    for i, (idx, row) in enumerate(firms.iterrows()):
        res = predict_firm(build_prompt(row), int(idx))
        rows.append({"index": int(idx), "bankrupt": int(row["bankrupt"]), **res})
        if (i + 1) % 25 == 0 or i + 1 == len(firms):
            print(f"{i + 1}/{len(firms)}  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_json(OUT / f"llm_prompt_{tag}.jsonl", orient="records", lines=True)

    df = pd.DataFrame(rows)
    m = metrics(df.bankrupt, df.pred)
    avg = df.ratings.apply(lambda r: np.mean([x for x in r if x is not None]) if any(x is not None for x in r) else np.nan)
    m["auc_mean_rating"] = roc_auc_score(df.bankrupt[avg.notna()], avg.dropna())
    m["unparsed_votes_%"] = 100 * (1 - df.valid_votes.sum() / (N_VOTES * len(df)))
    print(json.dumps({k: round(float(v), 4) for k, v in m.items()}, indent=2))
    json.dump({k: float(v) for k, v in m.items()}, open(OUT / f"llm_prompt_metrics_{tag}.json", "w"), indent=2)
