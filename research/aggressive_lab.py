"""Aggressive lab: can v3 earn more per 10 days with faster signals and more risk?

Grid (every combination reported):
  exposure  : base  (vol 25%, 25%/coin, 50% gross)
              x1.5  (vol 37.5%, 35%/coin, 75% gross)
              x2    (vol 50%, 40%/coin, 100% gross)        -- never leverage
  momentum  : 336h lookback, rebalance daily     (current v3)
              168h lookback, rebalance every 12h
              72h  lookback, rebalance every 4h
  kill sw.  : 8% / 12% drawdown

Hawkes sleeve as validated (walk-forward params, hourly). Same data, engine,
spreads and evaluation window as backtest/run_v3.py (Jul 2025 - Oct 2026).

Selection rule fixed before running: highest MEAN 10-day return among configs
whose worst 10-day return is >= -12%. Report first/second half too.

Run:  python research/aggressive_lab.py
"""
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backtest"))
warnings.filterwarnings("ignore")

import regime  # noqa: E402
import run_v3 as R  # noqa: E402
import strategy as S  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

OUT = os.path.join(ROOT, "data", "analysis")
EXPOSURE = {"base": (0.25, 0.25, 0.50), "x1.5": (0.375, 0.35, 0.75), "x2": (0.50, 0.40, 1.00)}
MOMENTUM = {"336h/24h": (336, 24), "168h/12h": (168, 12), "72h/4h": (72, 4)}
KILL = (0.08, 0.12)
WORST_10D_LIMIT = -12.0


def hold_every(w: pd.DataFrame, hours: int) -> pd.DataFrame:
    held = w.copy()
    held.loc[w.index.hour % hours != 0] = np.nan
    return held.ffill().fillna(0.0)


def momentum(px, el, on, off, lookback, every):
    mom = (px / px.shift(lookback) - 1).where(el)
    top = (mom.rank(axis=1, ascending=False) <= 2) & (mom > 0) & on.to_numpy()[:, None]
    bot = (mom.rank(axis=1, ascending=True) <= 2) & (mom < 0) & off.to_numpy()[:, None]
    return hold_every(S.size_long_short(top, bot, px), every)


def window_stats(eq: pd.Series, days: int = 10) -> dict:
    daily = eq.resample("1D").last()
    r = (daily.shift(-days) / daily - 1).dropna() * 100
    return {"mean_10d_%": r.mean(), "median_10d_%": r.median(), "p10_10d_%": r.quantile(0.1),
            "p90_10d_%": r.quantile(0.9), "best_10d_%": r.max(), "worst_10d_%": r.min(),
            "P(>5%)": (r > 5).mean() * 100, "P(<-5%)": (r < -5).mean() * 100}


def main() -> None:
    t0 = time.time()
    d = R.load()
    c = d["close"]
    px = c.ffill()
    dqv = d["quote_volume"].resample("1D").sum(min_count=1)
    avail = c.notna() & d["high"].notna() & d["low"].notna()
    el = S.liquid_universe(avail, dqv, 20)
    sma = S.btc_regime(px, 480)
    print("HMM ensemble ...", flush=True)
    prob = regime.hourly(regime.bull_probability_ensemble(regime.daily_close_from_hourly(px["BTC"])), c.index)
    on = (prob > 0.5).fillna(False) & sma
    off = (prob < 0.5).fillna(False) & ~sma
    folds = json.load(open(os.path.join(OUT, "short_lab_folds.json")))["folds"]["hawkes_ls_wfo"]
    a, b = R.EVAL_START, c.index[-1]
    mid = a + (b - a) / 2
    hs = R.half_spreads(c.columns)
    dvol = S.realized_vol(px) / np.sqrt(365)
    base = (S.TARGET_VOL, S.MAX_PER_COIN, S.MAX_GROSS)
    rows = {}
    try:
        for ex_name, (tv, pc, gr) in EXPOSURE.items():
            S.TARGET_VOL, S.MAX_PER_COIN, S.MAX_GROSS = tv, pc, gr
            hawk = pd.DataFrame(0.0, index=c.index, columns=c.columns)
            for f in folds:
                k, n1, n2, p5, p95 = f["params"]
                fa, fb = [pd.Timestamp(x, tz="UTC") for x in f["test"].split("..")]
                w = S.hawkes_long_short(d["high"].ffill(), d["low"].ffill(), px, el, k, int(n1), int(n2), p5, p95)
                m = (c.index >= fa) & (c.index < fb)
                hawk.loc[m] = w.loc[m]
            for mo_name, (lb, every) in MOMENTUM.items():
                w = S.v3_blend(momentum(px, el, on, off, lb, every), hawk, 0.5)
                gross = w.abs().sum(axis=1)
                w = w.mul((gr / gross).where(gross > gr, 1.0), axis=0)
                used = w.columns[(w.loc[a:b].abs() > 0).any()]
                for kill in KILL:
                    res = engine.run(d["open"].ffill().loc[a:b, used], px.loc[a:b, used], w.loc[a:b, used],
                                     half_spread=hs, risk=RiskManager(max_drawdown=kill), daily_vol=dvol[used])
                    eq = res.equity
                    m = metrics.summary(eq)
                    row = {"ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "composite": m["composite"],
                           "half1_%": metrics.summary(eq.loc[a:mid])["total_ret_%"],
                           "half2_%": metrics.summary(eq.loc[mid:b])["total_ret_%"],
                           "trades/day": len(res.trades) / (b - a).days, "fees_%": res.fees / 1000,
                           "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}
                    row.update(window_stats(eq))
                    key = (ex_name, mo_name, f"kill{int(kill * 100)}")
                    rows[key] = row
                    print(f"  {key}: ret {row['ret_%']:6.1f}% dd {row['max_dd_%']:6.1f}% mean10d "
                          f"{row['mean_10d_%']:5.2f}% worst10d {row['worst_10d_%']:6.1f}% "
                          f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        S.TARGET_VOL, S.MAX_PER_COIN, S.MAX_GROSS = base
    df = pd.DataFrame(rows).T.astype(float)
    df.index.names = ["exposure", "momentum", "kill"]
    df.to_csv(os.path.join(OUT, "aggressive_lab.csv"))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print("\n=== ALL CONFIGS (Jul 2025 - Oct 2026) ===")
    print(df.round(2).to_string())
    ok = df[df["worst_10d_%"] >= WORST_10D_LIMIT]
    pick = ok["mean_10d_%"].idxmax() if len(ok) else df["mean_10d_%"].idxmax()
    print(f"\n>>> Rule pick (max mean 10-day return, worst 10-day >= {WORST_10D_LIMIT}%): {pick}")
    print(df.loc[[pick]].round(2).to_string())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
