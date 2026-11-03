"""American dataset: LLM prompt learning (paper Sec. 3.2.3, Fig. 6) with the six key ratios.

Same template, rating scale (1-10, >5 = bankrupt), 3 votes and Ollama model as llm_prompt.py.
Test firms are drawn from 2015-2018 (the held-out years); calibration firms from the train years.
`pos_frac` oversamples failures so a small run still has enough of them (AUC is not affected).

Run:  .venv/bin/python src/american_llm.py --n 300 --split train --pos_frac 0.15   # calibration
      .venv/bin/python src/american_llm.py --n 500 --split test  --pos_frac 0.10   # test
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import llm_prompt as lp
from american_models import ROOT, SPLIT_YEAR

OUT = ROOT / "results" / "american"
KEY = json.load(open(OUT / "key_features.json"))


def build_prompt(row: pd.Series) -> str:
    lines = [f"- {n} ({KEY['definitions'][n]}): {lp.fmt(row[n])}" for n in KEY["key_features"]]
    return lp.TEMPLATE.format(indicators="\n".join(lines))


def sample_firms(n: int, split: str, pos_frac: float) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "american" / "american_ratios.csv")
    tr = d[d.year <= SPLIT_YEAR]
    med = tr[KEY["key_features"]].median()
    d = (tr if split == "train" else d[d.year > SPLIT_YEAR]).copy()
    d[KEY["key_features"]] = d[KEY["key_features"]].fillna(med)
    pos = d[d.bankrupt == 1].sample(min(int(d.bankrupt.sum()), round(n * pos_frac)), random_state=lp.SEED)
    neg = d[d.bankrupt == 0].sample(n - len(pos), random_state=lp.SEED)
    return pd.concat([pos, neg]).sample(frac=1, random_state=lp.SEED)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--pos_frac", type=float, default=0.10)
    a = ap.parse_args()
    tag = f"{a.split}_n{a.n}"
    firms = sample_firms(a.n, a.split, a.pos_frac)
    print(f"{len(firms)} firms ({a.split}), {int(firms.bankrupt.sum())} bankrupt; model {lp.MODEL}", flush=True)
    print("\n--- example prompt ---\n" + build_prompt(firms.iloc[0]) + "\n", flush=True)
    rows, t0 = [], time.time()
    for i, (idx, row) in enumerate(firms.iterrows()):
        res = lp.predict_firm(build_prompt(row), int(idx))
        rows.append({"index": int(idx), "bankrupt": int(row.bankrupt), **res})
        if (i + 1) % 25 == 0 or i + 1 == len(firms):
            print(f"{i + 1}/{len(firms)}  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_json(OUT / f"llm_prompt_{tag}.jsonl", orient="records", lines=True)
    df = pd.DataFrame(rows)
    m = lp.metrics(df.bankrupt, df.pred)
    avg = df.ratings.apply(lambda r: np.mean([x for x in r if x is not None]) if any(x is not None for x in r) else np.nan)
    m["auc_mean_rating"] = roc_auc_score(df.bankrupt[avg.notna()], avg.dropna())
    m["unparsed_votes_%"] = 100 * (1 - df.valid_votes.sum() / (lp.N_VOTES * len(df)))
    print(json.dumps({k: round(float(v), 4) for k, v in m.items()}, indent=2))
    json.dump({k: float(v) for k, v in m.items()}, open(OUT / f"llm_prompt_metrics_{tag}.json", "w"), indent=2)
