"""Tick scalping on wide-tick coins: is the leaderboard leader's edge real?

The #1 team (2026-10-05) rests limit buys/sells one tick apart on STO and PEPE,
where one price tick is ~0.25% of the price. This replays that idea on Binance
1-minute bars under two fill models:

  touch   : a resting limit fills as soon as the price trades AT it. This is how
            a simulator with no order queue behaves (and what Roostoo appears to do).
  through : it fills only if the price trades at least one tick THROUGH it.
            On a real exchange you sit behind a long queue at each tick of such
            coins, so this is the honest model.

Rules (long only, one position at a time, per coin):
  flat  -> each minute, rest a buy one tick below the last close, good for 1 bar
  long  -> rest a sell one tick above the entry (from the next bar on)
           stop: if a bar closes STOP ticks below the entry, sell at market on
           the next open, half a tick worse (taker)
Fees: 0.05% per resting fill, 0.10% for the market exit.

Run:  python research/tick_scalp.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "klines_1m")
OUT = os.path.join(ROOT, "data", "analysis")
COINS = ["STO", "PEPE", "BONK", "LISTA", "WLFI", "SHIB", "DOGE", "BTC"]
DAYS = 30
MAKER, TAKER = 0.0005, 0.0010
URL = "https://data-api.binance.vision/api/v3/klines"


def fetch(coin: str) -> pd.DataFrame:
    path = os.path.join(CACHE, f"{coin}.pkl")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 3600:
        return pd.read_pickle(path)
    end = int(time.time() // 60 * 60 * 1000)
    start = end - DAYS * 86400 * 1000
    rows = []
    while start < end:
        for attempt in range(5):
            try:
                r = requests.get(URL, params={"symbol": f"{coin}USDT", "interval": "1m", "startTime": start,
                                              "limit": 1000}, timeout=20)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2 + attempt * 2)
        else:
            raise RuntimeError(f"{coin}: fetch failed")
        k = r.json()
        if not k:
            break
        rows += k
        start = k[-1][0] + 60_000
    df = pd.DataFrame(rows).iloc[:, :6].astype(float)
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    df = df[~df.index.duplicated()]
    os.makedirs(CACHE, exist_ok=True)
    df.to_pickle(path)
    return df


def tick_size(df: pd.DataFrame) -> float:
    p = np.unique(np.round(df[["open", "high", "low", "close"]].to_numpy().ravel(), 12))
    return float(np.round(np.diff(p).min(), 12))


def simulate(df: pd.DataFrame, tick: float, through: bool, stop: int) -> pd.DataFrame:
    """Returns one row per round trip: entry time, net return (fraction of notional), exit kind."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    eps = tick * 1e-6
    extra = tick if through else 0.0
    trades = []
    entry, t_in, stop_next = None, None, False
    for i in range(1, len(c)):
        if entry is None:
            bid = c[i - 1] - tick
            if l[i] <= bid - extra + eps:
                entry, t_in = bid, i
            continue
        if stop_next:
            px = o[i] - tick / 2
            trades.append((df.index[t_in], px * (1 - TAKER) / (entry * (1 + MAKER)) - 1, "stop", i - t_in))
            entry, stop_next = None, False
            continue
        target = entry + tick
        if h[i] >= target + extra - eps:
            trades.append((df.index[t_in], target * (1 - MAKER) / (entry * (1 + MAKER)) - 1, "target", i - t_in))
            entry = None
        elif c[i] <= entry - stop * tick + eps:
            stop_next = True
    return pd.DataFrame(trades, columns=["time", "ret", "kind", "bars"]).set_index("time")


def summarise(tr: pd.DataFrame, days: float) -> dict:
    if tr.empty:
        return {"trades/day": 0.0}
    daily = tr["ret"].groupby(tr.index.floor("D")).sum()
    half = tr.index[0] + (tr.index[-1] - tr.index[0]) / 2
    return {"trades/day": len(tr) / days, "win_%": (tr["ret"] > 0).mean() * 100,
            "net_bp/trade": tr["ret"].mean() * 1e4, "ret/day_%_of_notional": tr["ret"].sum() / days * 100,
            "half1_%/day": tr.loc[:half, "ret"].sum() / (days / 2) * 100,
            "half2_%/day": tr.loc[half:, "ret"].sum() / (days / 2) * 100,
            "worst_day_%": daily.min() * 100, "losing_days_%": (daily < 0).mean() * 100,
            "avg_hold_min": tr["bars"].mean()}


def main() -> None:
    with ThreadPoolExecutor(4) as ex:
        data = dict(zip(COINS, ex.map(fetch, COINS)))
    rows = {}
    for coin, df in data.items():
        tick = tick_size(df)
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        print(f"{coin}: {len(df)} bars, {days:.1f} days, tick {tick:g} = {tick / df['close'].iloc[-1] * 1e4:.1f} bp",
              flush=True)
        for model, through in (("touch", False), ("through", True)):
            for stop in (1, 2, 3):
                rows[(coin, model, stop)] = {"tick_bp": tick / df["close"].iloc[-1] * 1e4,
                                             **summarise(simulate(df, tick, through, stop), days)}
    res = pd.DataFrame(rows).T.astype(float).round(2)
    res.index.names = ["coin", "fill", "stop_ticks"]
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "tick_scalp.csv"))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(res.to_string())


if __name__ == "__main__":
    sys.exit(main())
