"""Download extra market data for factor research (all public, no API key):

  spot    : hourly Binance spot klines incl. taker-buy quote volume and trade count
            (data.binance.vision archives + REST)            -> data/factors/spot_<COIN>.pkl
  funding : USD-M perpetual funding rate history (fapi REST)  -> data/factors/funding.pkl
  metrics : 5-minute futures metrics - open interest, top-trader and global
            long/short ratios, taker buy/sell ratio
            (data.binance.vision daily archives), resampled to hourly
                                                              -> data/factors/metrics_<COIN>.pkl

Run:  python research/fetch_factors.py spot funding metrics
"""
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

OUT = os.path.join(ROOT, "data", "factors")
K1H = os.path.join(ROOT, "data", "klines_1h")
PREFIX_1000 = {"PEPE", "SHIB", "BONK", "FLOKI"}
SPOT_ARCH = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{ym}.zip"
SPOT_REST = "https://data-api.binance.vision/api/v3/klines"
FUND_REST = "https://fapi.binance.com/fapi/v1/fundingRate"
MET_ARCH = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]

S = requests.Session()
S.mount("https://", HTTPAdapter(pool_maxsize=16, max_retries=Retry(
    total=5, connect=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))))


def coins() -> list:
    return sorted(f[:-4] for f in os.listdir(K1H) if f[:-4] not in ("TON", "OMNI"))


def perp(coin: str) -> str:
    return ("1000" + coin if coin in PREFIX_1000 else coin) + "USDT"


# ----------------------------------------------------------------- spot
def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    t = pd.to_numeric(df["open_time"])
    t = t.where(t < 1e14, t // 1000)
    df.index = pd.to_datetime(t, unit="ms", utc=True)
    return df[["close", "quote_volume", "taker_quote", "trades"]].astype(float)


def spot_coin(coin: str) -> str:
    sym, frames = coin + "USDT", []
    try:
        for ym in month_list(24):
            r = S.get(SPOT_ARCH.format(s=sym, ym=ym), timeout=60)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
            if not str(raw.iloc[0, 0]).isdigit():
                raw = raw.iloc[1:]
            frames.append(_frame(raw.values.tolist()))
        t = int((frames[-1].index[-1] + pd.Timedelta(hours=1)).timestamp() * 1000) if frames else \
            int(pd.Timestamp("2024-10-01", tz="UTC").timestamp() * 1000)
        now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
        for _ in range(30):
            r = S.get(SPOT_REST, params={"symbol": sym, "interval": "1h", "startTime": t, "limit": 1000}, timeout=20)
            rows = [k for k in (r.json() if r.status_code == 200 else []) if k[6] < now_ms]
            if not rows:
                break
            frames.append(_frame(rows))
            t = rows[-1][0] + 1
            if len(rows) < 1000:
                break
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.to_pickle(os.path.join(OUT, f"spot_{coin}.pkl"))
        return f"spot {coin}: {len(df)} bars to {df.index[-1]:%Y-%m-%d %H:%M}"
    except Exception as exc:
        return f"spot {coin}: FAILED {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------- funding
def funding_coin(coin: str):
    sym, out = perp(coin), []
    t = int(pd.Timestamp("2024-09-25", tz="UTC").timestamp() * 1000)
    try:
        for _ in range(20):
            r = S.get(FUND_REST, params={"symbol": sym, "startTime": t, "limit": 1000}, timeout=20)
            rows = r.json() if r.status_code == 200 else []
            if not isinstance(rows, list) or not rows:
                break
            out += rows
            t = rows[-1]["fundingTime"] + 1
            if len(rows) < 1000:
                break
    except Exception as exc:
        print(f"funding {coin}: {exc}")
    if not out:
        return coin, None
    s = pd.Series([float(x["fundingRate"]) for x in out],
                  index=pd.to_datetime([x["fundingTime"] for x in out], unit="ms", utc=True).floor("h"))
    return coin, s[~s.index.duplicated(keep="last")]


# ----------------------------------------------------------------- metrics
def metrics_coin(coin: str, days: int = 400) -> str:
    sym, frames = perp(coin), []
    end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    try:
        for d in pd.date_range(end - pd.Timedelta(days=days), end, freq="D"):
            r = S.get(MET_ARCH.format(s=sym, d=d.strftime("%Y-%m-%d")), timeout=30)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                frames.append(pd.read_csv(z.open(z.namelist()[0])))
        if not frames:
            return f"metrics {coin}: none"
        df = pd.concat(frames)
        df.index = pd.to_datetime(df["create_time"], utc=True)
        keep = {"sum_open_interest": "oi", "sum_open_interest_value": "oi_usd",
                "count_toptrader_long_short_ratio": "top_acct_ls", "sum_toptrader_long_short_ratio": "top_pos_ls",
                "count_long_short_ratio": "global_ls", "sum_taker_long_short_vol_ratio": "taker_ls"}
        df = df[list(keep)].rename(columns=keep).apply(pd.to_numeric, errors="coerce").sort_index()
        h = df.resample("1h").last()
        h.to_pickle(os.path.join(OUT, f"metrics_{coin}.pkl"))
        return f"metrics {coin}: {len(h)} hours {h.index[0]:%Y-%m-%d} -> {h.index[-1]:%Y-%m-%d}"
    except Exception as exc:
        return f"metrics {coin}: FAILED {type(exc).__name__}: {exc}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["spot", "funding", "metrics"]
    cs = coins()
    if "funding" in what:
        with ThreadPoolExecutor(4) as ex:
            res = dict(ex.map(funding_coin, cs))
        df = pd.DataFrame({k: v for k, v in res.items() if v is not None}).sort_index()
        df.to_pickle(os.path.join(OUT, "funding.pkl"))
        print(f"funding: {df.shape[1]} coins, {len(df)} rows {df.index[0]} -> {df.index[-1]}; "
              f"missing {[k for k, v in res.items() if v is None]}", flush=True)
    if "spot" in what:
        with ThreadPoolExecutor(4) as ex:
            for line in ex.map(spot_coin, cs):
                print(line, flush=True)
    if "metrics" in what:
        stats = pd.read_csv(os.path.join(ROOT, "data", "analysis", "coin_stats.csv"), index_col=0)
        top = [c for c in stats.sort_values("med_daily_vol_$M", ascending=False).index if c in cs][:30]
        with ThreadPoolExecutor(8) as ex:
            for line in ex.map(metrics_coin, top):
                print(line, flush=True)


if __name__ == "__main__":
    main()
