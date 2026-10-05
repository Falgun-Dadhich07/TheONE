"""Download 5-minute Binance spot klines for the most liquid Roostoo coins.

Same sources as research/fetch_data.py (data.binance.vision monthly archives
plus the REST endpoint for the newest bars). Output: data/klines_5m/<COIN>.pkl
(UTC open-time index; open, high, low, close, quote_volume).

Run:  python research/fetch_intraday.py --months 13 --top 25
"""
import argparse
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
from fetch_data import month_list  # noqa: E402

INTERVAL = "5m"
STEP = pd.Timedelta(minutes=5)
ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/{s}/5m/{s}-5m-{ym}.zip"
REST = "https://data-api.binance.vision/api/v3/klines"
OUT_DIR = os.path.join(ROOT, "data", "klines_5m")
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
KEEP = ["open", "high", "low", "close", "quote_volume"]

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_maxsize=8, max_retries=Retry(
    total=5, connect=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))))


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    t = pd.to_numeric(df["open_time"])
    t = t.where(t < 1e14, t // 1000)             # 2025+ archives use microseconds
    df.index = pd.to_datetime(t, unit="ms", utc=True)
    return df[KEEP].astype("float32" if False else float)


def month(symbol: str, ym: str) -> pd.DataFrame:
    r = SESSION.get(ARCHIVE.format(s=symbol, ym=ym), timeout=60)
    if r.status_code != 200:
        return pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(raw.iloc[0, 0]).isdigit():
        raw = raw.iloc[1:]
    return frame(raw.values.tolist())


def recent(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    out, t = [], int(start.timestamp() * 1000)
    now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    for _ in range(40):
        r = SESSION.get(REST, params={"symbol": symbol, "interval": INTERVAL, "startTime": t, "limit": 1000},
                        timeout=20)
        rows = [k for k in (r.json() if r.status_code == 200 else []) if k[6] < now_ms]
        if not rows:
            break
        out.append(frame(rows))
        t = rows[-1][0] + 1
        if len(rows) < 1000:
            break
    return pd.concat(out) if out else pd.DataFrame()


def coin(coin: str, months: list) -> str:
    path = os.path.join(OUT_DIR, f"{coin}.pkl")
    try:
        frames = [f for f in (month(f"{coin}USDT", ym) for ym in months) if not f.empty]
        start = frames[-1].index[-1] + STEP if frames else pd.Timestamp(months[0] + "-01", tz="UTC")
        tail = recent(f"{coin}USDT", start)
        if not tail.empty:
            frames.append(tail)
        if not frames:
            return f"{coin}: no data"
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.to_pickle(path)
        gap = df.index.to_series().diff().max()
        return f"{coin}: {len(df)} bars {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d %H:%M}, max gap {gap}"
    except Exception as exc:
        return f"{coin}: FAILED {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=13)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = pd.read_csv(os.path.join(ROOT, "data", "analysis", "coin_stats.csv"), index_col=0)
    stats = stats[~stats.index.isin(["TON", "OMNI"]) & (stats["history_days"] >= 400)]
    coins = list(stats.sort_values("med_daily_vol_$M", ascending=False).index[: a.top])
    months = month_list(a.months)
    print(f"{len(coins)} coins {coins}\nmonths {months[0]}..{months[-1]} + REST", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for line in ex.map(lambda c: coin(c, months), coins):
            print(line, flush=True)


if __name__ == "__main__":
    main()
