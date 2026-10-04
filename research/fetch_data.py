"""Download hourly Binance spot klines for the Roostoo crypto universe.

Roostoo prices are streamed from Binance (official FAQ), so Binance <COIN>USDT
candles are the closest public proxy for what we trade.

Source: data.binance.vision monthly archives (no API key, no rate limit),
plus the public REST klines endpoint for the current, unfinished month.
Output: data/klines_1h/<COIN>.csv  (UTC open_time index, OHLCV + quote volume)

Run:  python research/fetch_data.py --months 24
"""
import argparse
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{ym}.zip"
REST = "https://data-api.binance.vision/api/v3/klines"
OUT_DIR = os.path.join("data", "klines_1h")
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_maxsize=8, max_retries=Retry(
    total=5, connect=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))))

# Tokenized stocks on Roostoo (suffix B) and gold: not crypto, excluded.
NON_CRYPTO = {"PAXG"}


def roostoo_coins() -> list:
    info = requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=10).json()
    coins = []
    for pair, p in info["TradePairs"].items():
        coin = p.get("Coin") or pair.split("/")[0]
        if coin in NON_CRYPTO or is_tokenized_stock(coin):
            continue
        coins.append(coin)
    return sorted(coins)


def is_tokenized_stock(coin: str) -> bool:
    stocks = {"AMD", "CBRS", "COIN", "CRCL", "GLW", "GOOGL", "INTC", "LITE", "META",
              "MSFT", "MSTR", "MU", "NBIS", "NVDA", "PLTR", "QCOM", "SKHY", "SNDK",
              "SPCX", "TSLA", "WDC"}
    return coin.endswith("B") and coin[:-1] in stocks


def month_list(n: int) -> list:
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append(f"{y:04d}-{m:02d}")
    return sorted(out)


def _to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    t = pd.to_numeric(df["open_time"])
    # Binance spot archives switched to microsecond timestamps in 2025.
    t = t.where(t < 1e14, t // 1000)
    df.index = pd.to_datetime(t, unit="ms", utc=True)
    df.index.name = "open_time"
    df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades"]].astype(float)
    return df


def fetch_month(symbol: str, ym: str) -> pd.DataFrame:
    r = SESSION.get(ARCHIVE.format(s=symbol, ym=ym), timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(raw.iloc[0, 0]).isdigit():  # some files carry a header row
        raw = raw.iloc[1:]
    return _to_frame(raw.values.tolist())


def fetch_recent(symbol: str, start_ms: int) -> pd.DataFrame:
    frames = []
    while True:
        r = SESSION.get(REST, params={"symbol": symbol, "interval": "1h",
                                       "startTime": start_ms, "limit": 1000}, timeout=15)
        rows = r.json() if r.status_code == 200 else []
        if not rows:
            break
        frames.append(_to_frame(rows))
        start_ms = rows[-1][0] + 3600_000
        if len(rows) < 1000:
            break
    return pd.concat(frames) if frames else pd.DataFrame()


def fetch_coin(coin: str, months: list, refresh: bool = False) -> str:
    path = os.path.join(OUT_DIR, f"{coin}.csv")
    if os.path.exists(path) and not refresh:
        return f"{coin}: cached"
    try:
        return _fetch_coin(coin, months, path)
    except Exception as exc:  # one bad coin must not stop the batch
        return f"{coin}: FAILED {exc.__class__.__name__}: {exc}"


def _fetch_coin(coin: str, months: list, path: str) -> str:
    symbol = f"{coin}USDT"
    frames = [fetch_month(symbol, ym) for ym in months]
    frames = [f for f in frames if not f.empty]
    # Monthly archives appear a few days after month end, so fill everything
    # after the last archived bar from the REST endpoint.
    if frames:
        start = frames[-1].index[-1] + pd.Timedelta(hours=1)
    else:
        start = pd.Timestamp(months[0] + "-01", tz="UTC")
    recent = fetch_recent(symbol, int(start.timestamp() * 1000))
    if not recent.empty:
        frames.append(recent)
    if not frames:
        return f"{coin}: no Binance data"
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # drop the still-forming hour
    df = df[df.index < pd.Timestamp.now(tz="UTC").floor("h")]
    df.to_csv(path)
    return f"{coin}: {len(df)} bars {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d %H:%M}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--coins", nargs="*")
    ap.add_argument("--refresh", action="store_true", help="re-download cached coins")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    coins = args.coins or roostoo_coins()
    months = month_list(args.months)
    print(f"{len(coins)} coins, months {months[0]}..{months[-1]} + current month via REST")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for line in ex.map(lambda c: fetch_coin(c, months, args.refresh), coins):
            print(line)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
