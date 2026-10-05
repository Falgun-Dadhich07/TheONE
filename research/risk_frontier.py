"""Risk frontier: what does each risk level buy over a 10-day window?

No strategy tested has an edge much above ~+0.7-1.0% per 10 days at the
deployed risk. A return-ranked leaderboard rewards upside, so this maps the
trade-off honestly: for v3 (deployed) and v3 with the nearness-to-14-day-high
ranking (top 3), at rising exposure and looser kill switches, the full
distribution of 10-day outcomes.

Exposure levels (annual vol target, per-coin cap, gross cap; never leverage):
  x2 (deployed) 50% / 40% / 100%     x3  75% / 50% / 100%
  x4           100% / 60% / 100%     max 200% / 100% / 100%  (caps bind: ~fully invested)

Run:  python research/risk_frontier.py
"""
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
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")

import regime  # noqa: E402
import run_v3 as R  # noqa: E402
import strategy as S  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

OUT = os.path.join(ROOT, "data", "analysis")
EXPOSURE = {"x2 (deployed)": (0.50, 0.40, 1.0), "x3": (0.75, 0.50, 1.0), "x4": (1.00, 0.60, 1.0),
            "max": (2.00, 1.00, 1.0)}
KILLS = {"kill 12%": 0.12, "kill 20%": 0.20, "no kill": None}


def high14_sleeve(px, high, el, on, off, k=3):
    s = (px / high.rolling(336).max() - 1).where(el)
    top = (s.rank(axis=1, ascending=False) <= k) & on.to_numpy()[:, None]
    bot = (s.rank(axis=1, ascending=True) <= k) & off.to_numpy()[:, None]
    return S.hold_until_rebalance(S.size_long_short(top, bot, px))


def stats(eq: pd.Series, res) -> dict:
    m = metrics.summary(eq)
    daily = eq.resample("1D").last()
    r = (daily.shift(-10) / daily - 1).dropna() * 100
    return {"ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "composite": m["composite"],
            "mean_10d": r.mean(), "median_10d": r.median(), "p10": r.quantile(0.1), "p25": r.quantile(0.25),
            "p75": r.quantile(0.75), "p90": r.quantile(0.9), "best": r.max(), "worst": r.min(),
            "P(>5%)": (r > 5).mean() * 100, "P(>10%)": (r > 10).mean() * 100, "P(>15%)": (r > 15).mean() * 100,
            "P(<-5%)": (r < -5).mean() * 100, "P(<-10%)": (r < -10).mean() * 100,
            "avg_gross_%": res.weights.abs().sum(axis=1).mean() * 100, "fees_%": res.fees / 1000,
            "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}


def main() -> None:
    t0 = time.time()
    d = R.load()
    c = d["close"]
    px, high, low = c.ffill(), d["high"].ffill(), d["low"].ffill()
    dqv = d["quote_volume"].resample("1D").sum(min_count=1)
    el = S.liquid_universe(c.notna() & d["high"].notna() & d["low"].notna(), dqv, 20)
    sma = S.btc_regime(px, 480)
    prob = regime.hourly(regime.bull_probability_ensemble(regime.daily_close_from_hourly(px["BTC"])), c.index)
    on, off = (prob > 0.5).fillna(False) & sma, (prob < 0.5).fillna(False) & ~sma
    folds = json.load(open(os.path.join(OUT, "short_lab_folds.json")))["folds"]["hawkes_ls_wfo"]
    a, b = R.EVAL_START, c.index[-1]
    dvol = S.realized_vol(px) / np.sqrt(365)
    hs = R.half_spreads(c.columns)
    base = (S.TARGET_VOL, S.MAX_PER_COIN, S.MAX_GROSS)
    rows = {}
    try:
        for ename, (tv, pc, gr) in EXPOSURE.items():
            S.TARGET_VOL, S.MAX_PER_COIN, S.MAX_GROSS = tv, pc, gr
            hawk = pd.DataFrame(0.0, index=c.index, columns=c.columns)
            for f in folds:
                k, n1, n2, p5, p95 = f["params"]
                fa, fb = [pd.Timestamp(x, tz="UTC") for x in f["test"].split("..")]
                w = S.hawkes_long_short(high, low, px, el, k, int(n1), int(n2), p5, p95)
                m = (c.index >= fa) & (c.index < fb)
                hawk.loc[m] = w.loc[m]
            strategies = {"v3": S.v3_blend(S.momentum_long_short(px, el, on, off), hawk, 0.5),
                          "v3+high14": S.v3_blend(high14_sleeve(px, high, el, on, off), hawk, 0.5),
                          "high14 only": high14_sleeve(px, high, el, on, off)}
            for sname, w in strategies.items():
                used = w.columns[(w.loc[a:b].abs() > 0).any()]
                for kname, kill in KILLS.items():
                    res = engine.run(d["open"].ffill().loc[a:b, used], px.loc[a:b, used], w.loc[a:b, used],
                                     half_spread=hs, risk=RiskManager(max_drawdown=kill), daily_vol=dvol[used])
                    rows[(sname, ename, kname)] = stats(res.equity, res)
                    x = rows[(sname, ename, kname)]
                    print(f"  {sname:<12} {ename:<14} {kname:<9} ret {x['ret_%']:6.1f}% dd {x['max_dd_%']:6.1f}% "
                          f"mean10d {x['mean_10d']:5.2f}% P(>10%) {x['P(>10%)']:4.1f}% P(<-10%) {x['P(<-10%)']:4.1f}% "
                          f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        S.TARGET_VOL, S.MAX_PER_COIN, S.MAX_GROSS = base
    df = pd.DataFrame(rows).T.astype(float).round(2)
    df.index.names = ["strategy", "exposure", "kill"]
    df.to_csv(os.path.join(OUT, "risk_frontier.csv"))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(f"\n=== RISK FRONTIER, {a:%Y-%m-%d} -> {b:%Y-%m-%d}; 10-day windows in % ===")
    print(df.to_string())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
