"""Would a slow trend-breakout sleeve improve v3?

The intraday lab showed fast breakouts lose and only multi-day channels earn.
This tests a slow channel breakout on HOURLY bars (2 years of data, top-20
causal universe, same regime gate as v3: longs only when risk-on, shorts only
when risk-off) as a third sleeve next to v3's momentum and Hawkes sleeves.

Channel length is chosen walk-forward (train 180d / test 60d, same folds as the
Hawkes validation) from a small fixed set. Everything is evaluated with the
real engine, kill switch and per-coin spreads at the deployed risk settings.

Run:  python research/trend_sleeve.py
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
import strategy_lab as L  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

OUT = os.path.join(ROOT, "data", "analysis")
CHANNELS = [48, 72, 96, 144, 192, 240, 336]      # hours


def breakout_sleeve(px, high, low, eligible, on, off, n: int) -> pd.DataFrame:
    """Long after a close above the n-hour high (while risk-on), short after a close below the
    n-hour low (while risk-off); exit on the opposite n/2-hour channel or when the gate closes."""
    up = (px > high.rolling(n).max().shift(1)).to_numpy()
    dn = (px < low.rolling(n).min().shift(1)).to_numpy()
    xl = (px < low.rolling(n // 2).min().shift(1)).to_numpy()
    xs = (px > high.rolling(n // 2).max().shift(1)).to_numpy()
    el = eligible.to_numpy()
    onv, offv = on.to_numpy(), off.to_numpy()
    pos = np.zeros(px.shape)
    cur = np.zeros(px.shape[1])
    for t in range(len(px)):
        cur = np.where(up[t] & el[t] & onv[t], 1.0, np.where(dn[t] & el[t] & offv[t], -1.0, cur))
        cur = np.where((cur > 0) & (xl[t] | ~onv[t] | ~el[t]), 0.0, cur)
        cur = np.where((cur < 0) & (xs[t] | ~offv[t] | ~el[t]), 0.0, cur)
        pos[t] = cur
    sig = pd.DataFrame(pos, index=px.index, columns=px.columns)
    return S.size_long_short(sig > 0, sig < 0, px)


def evaluate(d, w, a, b) -> dict:
    c = d["close"].ffill()
    used = w.columns[(w.loc[a:b].abs() > 0).any()]
    res = engine.run(d["open"].ffill().loc[a:b, used], c.loc[a:b, used], w.loc[a:b, used],
                     half_spread=R.half_spreads(used), risk=RiskManager(),
                     daily_vol=(S.realized_vol(c) / np.sqrt(365))[used])
    eq = res.equity
    m = metrics.summary(eq)
    daily = eq.resample("1D").last()
    r10 = (daily.shift(-10) / daily - 1).dropna() * 100
    mid = a + (b - a) / 2
    return {"ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "calmar": m["calmar"], "composite": m["composite"],
            "half1_%": metrics.summary(eq.loc[a:mid])["total_ret_%"], "half2_%": metrics.summary(eq.loc[mid:b])["total_ret_%"],
            "mean_10d_%": r10.mean(), "median_10d_%": r10.median(), "p10_10d_%": r10.quantile(0.1),
            "p90_10d_%": r10.quantile(0.9), "worst_10d_%": r10.min(), "best_10d_%": r10.max(),
            "P(10d>5%)": (r10 > 5).mean() * 100, "P(10d<-5%)": (r10 < -5).mean() * 100,
            "avg_gross_%": res.weights.abs().sum(axis=1).mean() * 100,
            "trades/day": len(res.trades) / max((b - a).days, 1), "fees_%": res.fees / 1000,
            "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}


def cap_gross(w: pd.DataFrame) -> pd.DataFrame:
    w = w.clip(lower=-S.MAX_PER_COIN, upper=S.MAX_PER_COIN)
    g = w.abs().sum(axis=1)
    return w.mul((S.MAX_GROSS / g).where(g > S.MAX_GROSS, 1.0), axis=0)


def main() -> None:
    t0 = time.time()
    d = R.load()
    c = d["close"]
    px = c.ffill()
    dqv = d["quote_volume"].resample("1D").sum(min_count=1)
    avail = c.notna() & d["high"].notna() & d["low"].notna()
    el = S.liquid_universe(avail, dqv, 20)
    sma = S.btc_regime(px, 480)
    prob = regime.hourly(regime.bull_probability_ensemble(regime.daily_close_from_hourly(px["BTC"])), c.index)
    on = (prob > 0.5).fillna(False) & sma
    off = (prob < 0.5).fillna(False) & ~sma
    high, low = d["high"].ffill(), d["low"].ffill()

    # v3 sleeves exactly as validated
    mom = S.momentum_long_short(px, el, on, off)
    folds_h = json.load(open(os.path.join(OUT, "short_lab_folds.json")))["folds"]["hawkes_ls_wfo"]
    hawk = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for f in folds_h:
        k, n1, n2, p5, p95 = f["params"]
        fa, fb = [pd.Timestamp(x, tz="UTC") for x in f["test"].split("..")]
        w = S.hawkes_long_short(high, low, px, el, k, int(n1), int(n2), p5, p95)
        m = (c.index >= fa) & (c.index < fb)
        hawk.loc[m] = w.loc[m]
    v3 = S.v3_blend(mom, hawk, 0.5)

    # breakout sleeve: walk-forward choice of channel length
    print("building breakout sleeves ...", flush=True)
    sleeves = {n: breakout_sleeve(px, high, low, el, on, off, n) for n in CHANNELS}
    fe = L.FastEval(d["open"].ffill())
    rets = {n: fe.returns(w.to_numpy()) for n, w in sleeves.items()}
    idx = c.index
    bo = pd.DataFrame(0.0, index=idx, columns=c.columns)
    picks = []
    for (a, b, cc) in L.folds(idx):
        ia, ib = idx.searchsorted(a), idx.searchsorted(b)
        sc = {n: L.score(r[ia:ib])["composite"] for n, r in rets.items()}
        n = max(sc, key=sc.get)
        picks.append(n)
        m = (idx >= b) & (idx < cc)
        bo.loc[m] = sleeves[n].loc[m]
    print("channel (hours) picked per fold:", picks, flush=True)

    a, b = R.EVAL_START, idx[-1]
    a2 = pd.Timestamp("2025-12-30", tz="UTC")
    variants = {
        "v3 (deployed: 50% momentum + 50% Hawkes)": v3,
        "breakout sleeve alone (walk-forward channel)": bo,
        "v3 two-thirds + breakout one-third": cap_gross(2 / 3 * v3 + 1 / 3 * bo),
        "v3 half + breakout half": cap_gross(0.5 * v3 + 0.5 * bo),
        "v3 full + breakout half on top (uses idle cash)": cap_gross(v3 + 0.5 * bo),
        "v3 full + breakout full on top": cap_gross(v3 + bo),
    }
    for n in CHANNELS:
        variants[f"  breakout fixed {n}h alone"] = sleeves[n]
    rows, rows2 = {}, {}
    for name, w in variants.items():
        rows[name] = evaluate(d, w, a, b)
        if not name.startswith("  "):
            rows2[name] = evaluate(d, w, a2, b)
        print(f"  {name:<50} ret {rows[name]['ret_%']:6.1f}% dd {rows[name]['max_dd_%']:6.1f}% comp "
              f"{rows[name]['composite']:5.2f} mean10d {rows[name]['mean_10d_%']:5.2f}% ({time.time() - t0:.0f}s)", flush=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    df = pd.DataFrame(rows).T.astype(float).round(2)
    df.to_csv(os.path.join(OUT, "trend_sleeve.csv"))
    print(f"\n=== {a:%Y-%m-%d} -> {b:%Y-%m-%d} (real engine, kill switch, spreads, deployed risk settings) ===")
    print(df.to_string())
    print(f"\n=== same, {a2:%Y-%m-%d} -> {b:%Y-%m-%d} ===")
    print(pd.DataFrame(rows2).T.astype(float).round(2)[["ret_%", "max_dd_%", "composite", "mean_10d_%", "worst_10d_%",
                                                        "P(10d>5%)", "P(10d<-5%)", "avg_gross_%", "fees_%"]].to_string())
    dr = pd.DataFrame({"v3": fe.returns(v3.to_numpy()), "bo": fe.returns(bo.to_numpy())}, index=idx).loc[a:b]
    print("\ndaily return correlation v3 vs breakout sleeve:",
          round(float(dr.resample("1D").sum().corr().iloc[0, 1]), 2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
