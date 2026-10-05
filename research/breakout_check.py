"""Due diligence on the regime-gated 15-minute breakout from intraday_lab.py.

Fixes a lookahead found in the lab (hourly BTC SMA was labeled at the hour's
start, so 15m bars saw up to 45 minutes ahead) and then asks:
  1. Does it survive the fix, in the fast evaluator and in the REAL engine
     (5% rebalance band, kill switch, per-coin spreads)?
  2. Is it one lucky coin? (drop ZEC, drop each of the top contributors)
  3. Is it one lucky parameter? (every channel length, whole OOS period)
  4. Long side vs short side; sensitivity to training-window length.
  5. How does deployed v3 do over exactly the same dates?

Run:  python research/breakout_check.py
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
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "backtest"))
warnings.filterwarnings("ignore")

import intraday_lab as I  # noqa: E402
import regime  # noqa: E402
import settings  # noqa: E402
import strategy as S  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
CHANNELS = [float(v) for v in sorted(set(np.round(np.geomspace(8, 960, 24))))]


def gates(idx, btc_15m: pd.Series):
    """Causal regime gate on a 15m index: HMM ensemble (previous day) AND BTC vs its 480h SMA
    computed from hourly closes that are only used after the hour has ended."""
    daily_btc = pd.read_csv(os.path.join(ROOT, "data", "klines_1h", "BTC.csv"), index_col=0, parse_dates=True)["close"]
    daily = regime.daily_close_from_hourly(daily_btc)
    extra = btc_15m.resample("1D").last()
    daily = pd.concat([daily, extra[extra.index > daily.index[-1]]])
    prob = regime.hourly(regime.bull_probability_ensemble(daily), idx)
    hourly = btc_15m.resample("1h").last()                       # label H = close at H+1h
    above = (hourly > hourly.rolling(settings.V3_REGIME_SMA).mean())
    above.index = above.index + pd.Timedelta(hours=1)            # known only once the hour has closed
    sma = above.reindex(idx, method="ffill").fillna(False).astype(bool)
    on = ((prob > 0.5).fillna(False) & sma).to_numpy()
    off = ((prob < 0.5).fillna(False) & ~sma).to_numpy()
    return on, off


def wf_weights(d, vol, gate, idx, train_days=I.TRAIN_D, ex=None, long_only=False, short_only=False):
    """Walk-forward spliced weights: each fold uses the channel with the best training composite."""
    ex = ex or I.Exec(d, "taker")
    W = {n: I.breakout(d, vol, (n,), gate) for n in CHANNELS}
    if long_only:
        W = {n: np.maximum(w, 0) for n, w in W.items()}
    if short_only:
        W = {n: np.minimum(w, 0) for n, w in W.items()}
    runs = {n: ex.returns(w) for n, w in W.items()}
    out = np.zeros_like(next(iter(W.values())))
    picks = []
    t0 = idx[0] + pd.Timedelta(days=I.WARM_D + I.TRAIN_D - train_days)
    while True:
        b = t0 + pd.Timedelta(days=train_days)
        c = b + pd.Timedelta(days=I.TEST_D)
        if b >= idx[-1]:
            break
        ia, ib, ic = idx.searchsorted(t0), idx.searchsorted(b), idx.searchsorted(min(c, idx[-1] + pd.Timedelta(minutes=15)))
        sc = {n: (I.score(r[ia:ib])["composite"] if tn[ia:ib].sum() >= 5 else -10) for n, (r, tn, _) in runs.items()}
        n = max(sc, key=sc.get)
        out[ib:ic] = W[n][ib:ic]
        picks.append(int(n))
        t0 += pd.Timedelta(days=I.TEST_D)
    return out, picks, runs


def fast(ex, w, idx, start) -> dict:
    r, tn, fe = ex.returns(w)
    return I.report(r, tn, fe, idx, start)


def real_engine(d, w, idx, start, bars_per_year=I.BARS_PER_YEAR) -> dict:
    cols = d["close"].columns
    wdf = pd.DataFrame(w, index=idx, columns=cols)
    hs = dict(zip(cols, I.half_spread_vec(cols)))
    dvol = (np.log(d["close"]).diff().rolling(I.VOL_LB).std() * np.sqrt(I.BARS_PER_DAY))
    res = engine.run(d["open"].loc[start:], d["close"].loc[start:], wdf.loc[start:], half_spread=hs,
                     risk=RiskManager(), daily_vol=dvol)
    eq = res.equity
    r = eq.pct_change().dropna().to_numpy()
    s = I.score(r)
    daily = eq.resample("1D").last()
    r10 = (daily.shift(-10) / daily - 1).dropna() * 100
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    return {"ret_%": s["ret"] * 100, "max_dd_%": s["mdd"] * 100, "sharpe": s["sharpe"], "sortino": s["sortino"],
            "calmar": s["calmar"], "composite": s["composite"], "mean_10d_%": r10.mean(), "median_10d_%": r10.median(),
            "p10_10d_%": r10.quantile(0.1), "p90_10d_%": r10.quantile(0.9), "worst_10d_%": r10.min(),
            "best_10d_%": r10.max(), "P(10d>5%)": (r10 > 5).mean() * 100, "P(10d<-5%)": (r10 < -5).mean() * 100,
            "trades/day": len(res.trades) / days, "fees_%": res.fees / 1000,
            "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}, eq


def main() -> None:
    t0 = time.time()
    d = I.load()
    c = d["close"]
    idx = c.index
    vol = (np.log(c).diff().rolling(I.VOL_LB, min_periods=I.VOL_LB // 2).std() * np.sqrt(I.BARS_PER_YEAR)).to_numpy()
    gate = gates(idx, c["BTC"])
    oos = I.folds(idx)[0][1]
    ex_t, ex_m = I.Exec(d, "taker"), I.Exec(d, "maker")
    print(f"OOS {oos:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d %H:%M}; risk-on {gate[0][idx >= oos].mean():.0%}, "
          f"risk-off {gate[1][idx >= oos].mean():.0%} of bars\n", flush=True)

    w, picks, runs = wf_weights(d, vol, gate, idx)
    print("channel length picked per fold (15m bars):", picks, "\n", flush=True)
    rows = {"gated breakout, fast eval, taker (lookahead fixed)": fast(ex_t, w, idx, oos),
            "gated breakout, fast eval, maker": fast(ex_m, w, idx, oos)}
    eng, eq = real_engine(d, w, idx, oos)
    rows["gated breakout, REAL ENGINE (band 5%, kill switch, taker)"] = eng
    rows["  long side only (real engine)"] = real_engine(d, wf_weights(d, vol, gate, idx, long_only=True)[0], idx, oos)[0]
    rows["  short side only (real engine)"] = real_engine(d, wf_weights(d, vol, gate, idx, short_only=True)[0], idx, oos)[0]
    for td in (60, 120):
        rows[f"  train window {td}d (fast, taker)"] = fast(ex_t, wf_weights(d, vol, gate, idx, train_days=td)[0], idx, oos)
    print(pd.DataFrame(rows).T.astype(float).round(2).to_string(), "\n", flush=True)

    # 3. every channel length over the whole OOS period (no selection at all)
    i0 = idx.searchsorted(oos)
    allc = {int(n): I.score(I.kill_switch(r[i0:])[0]) for n, (r, tn, _) in runs.items()}
    tab = pd.DataFrame({n: {"ret_%": s["ret"] * 100, "max_dd_%": s["mdd"] * 100, "composite": s["composite"]}
                        for n, s in allc.items()}).T
    print("Every channel length, fixed for the whole OOS period (fast eval, taker):")
    print(tab.round(1).T.to_string())
    print(f"profitable: {(tab['ret_%'] > 0).sum()}/{len(tab)}; median return {tab['ret_%'].median():.1f}%\n", flush=True)

    # 2. coin dependence: P&L attribution and leave-one-out (real engine)
    cols = list(c.columns)
    r_next = np.vstack([ex_t.r[1:], np.zeros((1, len(cols)))])
    prev = np.vstack([np.zeros((1, len(cols))), w[:-1]])
    contrib = pd.Series((prev * r_next)[i0:].sum(axis=0) * 100, index=cols).sort_values(ascending=False)
    print("gross P&L contribution by coin (% of equity, before costs):")
    print(contrib.round(1).to_string(), "\n", flush=True)
    loo = {}
    for drop in list(contrib.index[:4]) + ["ZEC"]:
        if drop in loo:
            continue
        keep = [x for x in cols if x != drop]
        dd = {k: v[keep] for k, v in d.items()}
        vv = vol[:, [cols.index(x) for x in keep]]
        loo[f"without {drop}"] = real_engine(dd, wf_weights(dd, vv, gate, idx)[0], idx, oos)[0]
    print("Leave-one-coin-out (real engine):")
    print(pd.DataFrame(loo).T[["ret_%", "max_dd_%", "composite", "mean_10d_%", "worst_10d_%"]].astype(float).round(2).to_string(), "\n")

    eq.to_pickle(os.path.join(ROOT, "data", "analysis", "breakout_equity.pkl"))
    pd.DataFrame(w, index=idx, columns=cols).to_pickle(os.path.join(ROOT, "data", "analysis", "breakout_weights.pkl"))
    with open(os.path.join(ROOT, "data", "analysis", "breakout_picks.json"), "w") as f:
        json.dump({"picks": picks, "oos_start": str(oos)}, f)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
