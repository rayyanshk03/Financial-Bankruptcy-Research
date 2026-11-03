"""Build an A-share bankruptcy-risk dataset (paper: SLMNN, CSMAR substitute) with AKShare.

Stages (python download_data.py <stage> ...): labels, financials, lawsuits, holders, news
Output goes to ./data
"""
import sys, time
from pathlib import Path
import akshare as ak
import pandas as pd

OUT = Path("data"); OUT.mkdir(exist_ok=True)
YEARS = range(2015, 2025)


def retry(f, n=4, wait=3, **kw):
    for i in range(n):
        try:
            return f(**kw)
        except Exception as e:
            if i == n - 1:
                print("  failed:", f.__name__, kw, str(e)[:80]); return None
            time.sleep(wait * (i + 1))


def labels():
    """High-risk = delisted / suspended / ST / *ST (paper's definition)."""
    sh = retry(ak.stock_info_sh_delist)
    sz = retry(ak.stock_info_sz_delist, symbol="终止上市公司")
    sh = sh.rename(columns={"公司代码": "code", "公司简称": "name", "暂停上市日期": "event_date"})
    sz = sz.rename(columns={"证券代码": "code", "证券简称": "name", "终止上市日期": "event_date"})
    d = pd.concat([sh[["code", "name", "event_date"]], sz[["code", "name", "event_date"]]])
    d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce")
    d["event"] = "delist"
    st = retry(ak.stock_zh_a_st_em)
    st = st.rename(columns={"代码": "code", "名称": "name"})[["code", "name"]]
    st["event_date"] = pd.Timestamp.today().normalize(); st["event"] = "st_now"
    out = pd.concat([d, st], ignore_index=True)
    out["event_year"] = out["event_date"].dt.year
    out.to_csv(OUT / "labels.csv", index=False); print("labels", out.shape)


def financials():
    """Bulk per-period statements for every listed firm (annual reports, 12-31)."""
    fns = {"income": ak.stock_lrb_em, "balance": ak.stock_zcfz_em,
           "cashflow": ak.stock_xjll_em, "perf": ak.stock_yjbb_em}
    for y in YEARS:
        for k, f in fns.items():
            p = OUT / f"fin_{k}_{y}.csv"
            if p.exists(): continue
            df = retry(f, date=f"{y}1231")
            if df is not None:
                df.to_csv(p, index=False); print(k, y, df.shape)


def lawsuits():
    for y in YEARS:
        p = OUT / f"lawsuit_{y}.csv"
        if p.exists(): continue
        df = retry(ak.stock_cg_lawsuit_cninfo, symbol="全部", start_date=f"{y}0101", end_date=f"{y}1231")
        if df is not None: df.to_csv(p, index=False); print("lawsuit", y, df.shape)


def holders(year=2023):
    """Top-10 shareholders per firm (graph edges). Slow: ~1 request per firm."""
    codes = pd.read_csv(OUT / f"fin_income_{year}.csv", dtype=str)["股票代码"]
    rows = []
    for i, c in enumerate(codes):
        pre = "sh" if c.startswith("6") else "bj" if c[0] in "48" else "sz"
        df = retry(ak.stock_gdfx_top_10_em, symbol=pre + c, date=f"{year}1231", n=2)
        if df is not None and len(df):
            df["code"] = c; df["year"] = year; rows.append(df)
        if i % 200 == 0:
            print("holders", i, len(codes))
            pd.concat(rows).to_csv(OUT / f"holders_{year}.csv", index=False)
    pd.concat(rows).to_csv(OUT / f"holders_{year}.csv", index=False)


def news(year=2023):
    """Recent headlines per firm (East Money only serves the latest ~10)."""
    codes = pd.read_csv(OUT / f"fin_income_{year}.csv", dtype=str)["股票代码"]
    rows = []
    for i, c in enumerate(codes):
        df = retry(ak.stock_news_em, symbol=c, n=2)
        if df is not None and len(df): rows.append(df)
        if i % 200 == 0: print("news", i, len(codes))
    pd.concat(rows).to_csv(OUT / "news.csv", index=False)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["labels", "financials", "lawsuits"]:
        globals()[s]()
