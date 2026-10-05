"""Scalping lab: the best 1-minute scalp we can find, tested honestly.

Research only; nothing here is wired into the bot.

Data: 30 days of Binance 1-minute bars (cached by tick_scalp.py).
Protocol: every variant is chosen on the first 20 days (train) and reported on
the last 10 days (test), which the selection never saw.

Family A - maker ladder (the leaderboard leader's pattern), wide-tick coins:
  flat -> each minute, if the filter allows, rest a buy OFFSET ticks below the
          last close, good for one bar
  long -> rest a sell TARGET ticks above the entry (from the next bar on);
          if a bar closes STOP ticks below the entry, or after 120 minutes,
          sell at market on the next open, half a tick worse
  filters: none | up (close above its 60-minute average) | quiet (the last
           30 minutes traded inside a 3-tick range)
  fills:   touch   = filled when the price trades AT the limit (no queue)
           through = filled only when the price trades one tick THROUGH it
  fees:    0.05% per resting fill, 0.10% for a market exit

Family B - taker scalps (fills any real market would give), all coins:
  signal: the M-minute return is more than Z standard deviations (rolling 1 day)
  momentum = trade with the move, reversal = trade against it; hold H minutes
  entry and exit at the next bar's open; cost = 0.10% per side + half a spread
  per side. Reported as the average edge per trade before and after costs.

Run:  python research/scalp_lab.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import tick_scalp as T  # noqa: E402

OUT = os.path.join(ROOT, "data", "analysis")
WIDE = ["PEPE", "BONK", "SHIB", "WLFI", "STO", "LISTA"]
ALL = WIDE + ["DOGE", "BTC"]
MAKER, TAKER = T.MAKER, T.TAKER
TRAIN_DAYS, MAX_HOLD = 20, 120


def load(coin: str) -> pd.DataFrame:
    path = os.path.join(T.CACHE, f"{coin}.pkl")
    return pd.read_pickle(path) if os.path.exists(path) else T.fetch(coin)


def ladder(df, tick, through, offset, target, stop, allow) -> pd.DataFrame:
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    eps, extra = tick * 1e-6, (tick if through else 0.0)
    out, entry, t_in, exit_next = [], None, 0, False
    for i in range(1, len(c)):
        if entry is None:
            if allow[i - 1]:
                bid = c[i - 1] - offset * tick
                if l[i] <= bid - extra + eps:
                    entry, t_in = bid, i
            continue
        if exit_next:
            out.append((t_in, (o[i] - tick / 2) * (1 - TAKER) / (entry * (1 + MAKER)) - 1, 0))
            entry, exit_next = None, False
            continue
        tgt = entry + target * tick
        if h[i] >= tgt + extra - eps:
            out.append((t_in, tgt * (1 - MAKER) / (entry * (1 + MAKER)) - 1, 1))
            entry = None
        elif c[i] <= entry - stop * tick + eps or i - t_in >= MAX_HOLD:
            exit_next = True
    tr = pd.DataFrame(out, columns=["i", "ret", "win"])
    tr.index = df.index[tr.pop("i").to_numpy()] if len(tr) else pd.DatetimeIndex([], tz="UTC")
    return tr


def per_day(tr: pd.DataFrame, a, b) -> dict:
    x = tr.loc[a:b]
    days = (b - a).total_seconds() / 86400
    if x.empty:
        return {"ret/day_%": 0.0, "trades/day": 0.0, "win_%": np.nan, "worst_day_%": 0.0, "losing_days_%": 0.0}
    daily = x["ret"].groupby(x.index.floor("D")).sum()
    return {"ret/day_%": x["ret"].sum() / days * 100, "trades/day": len(x) / days, "win_%": x["win"].mean() * 100,
            "worst_day_%": daily.min() * 100, "losing_days_%": (daily < 0).mean() * 100}


def family_a(data) -> pd.DataFrame:
    rows = []
    for coin in WIDE:
        df = data[coin]
        tick = T.tick_size(df)
        c = df["close"]
        filters = {"none": np.ones(len(df), bool),
                   "up": (c > c.rolling(60).mean()).to_numpy(),
                   "quiet": ((df["high"].rolling(30).max() - df["low"].rolling(30).min()) <= 3 * tick * 1.0001).to_numpy()}
        split = df.index[0] + pd.Timedelta(days=TRAIN_DAYS)
        for through in (False, True):
            best = None
            for offset, target, stop, fname in itertools.product((0, 1, 2), (1, 2), (1, 2, 3), filters):
                if offset == 0 and not through:
                    continue      # buying at the last price with touch fills is just a taker buy
                tr = ladder(df, tick, through, offset, target, stop, filters[fname])
                train = per_day(tr, df.index[0], split)
                if best is None or train["ret/day_%"] > best[0]["ret/day_%"]:
                    best = (train, per_day(tr, split, df.index[-1]), (offset, target, stop, fname))
            train, test, cfg = best
            rows.append({"coin": coin, "tick_bp": round(tick / c.iloc[-1] * 1e4, 1),
                         "fill": "through" if through else "touch",
                         "best (offset,target,stop,filter)": cfg, "train ret/day_%": train["ret/day_%"],
                         **{f"test {k}": v for k, v in test.items()}})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def family_b(data) -> pd.DataFrame:
    rows = []
    for coin in ALL:
        df = data[coin]
        tick = T.tick_size(df)
        o, c = df["open"], df["close"]
        cost = (2 * TAKER + tick / c.iloc[-1]) * 1e4          # round trip, bp
        split = df.index[0] + pd.Timedelta(days=TRAIN_DAYS)
        best = None
        for m, z, hold, kind in itertools.product((5, 15, 60), (2.0, 3.0), (5, 15, 60), ("momentum", "reversal")):
            r = c / c.shift(m) - 1
            zs = r / r.rolling(1440).std()
            side = np.sign(zs).where(zs.abs() > z, 0.0) * (1 if kind == "momentum" else -1)
            fwd = (o.shift(-1 - hold) / o.shift(-1) - 1) * side * 1e4      # bp, entry at the next open
            g = fwd[side != 0].dropna()
            tr, te = g.loc[:split], g.loc[split:]
            if len(tr) < 50:
                continue
            if best is None or tr.mean() > best[0]:
                best = (tr.mean(), te.mean() if len(te) else np.nan, len(te), (m, z, hold, kind))
        rows.append({"coin": coin, "round_trip_cost_bp": round(cost, 1), "best (M,Z,H,kind)": best[3],
                     "train gross_bp": best[0], "test gross_bp": best[1], "test net_bp": best[1] - cost,
                     "test signals": best[2]})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    data = {coin: load(coin) for coin in ALL}
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    a = family_a(data).round(2)
    b = family_b(data).round(2)
    a.to_csv(os.path.join(OUT, "scalp_lab_maker.csv"), index=False)
    b.to_csv(os.path.join(OUT, "scalp_lab_taker.csv"), index=False)
    print("\n=== A. Maker ladder: best variant on 20 train days, result on the 10 unseen days (% of position per day) ===")
    print(a.to_string(index=False))
    print("\n=== B. Taker scalps: best variant on train, edge per trade on the unseen days (bp) ===")
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
