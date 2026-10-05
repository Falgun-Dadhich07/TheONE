"""Short lab: every strategy family again, now allowed to short.

Roostoo shorts are 1x (sized by USD collateral, 0.1% fee each way, loss capped
at the collateral); backtest/engine.py models that for negative weights.
Caps apply to |weights|: 25% per coin, 50% gross.

Candidates (majors unless noted):
  long-only refs : v1_fixed (live bot), v2_long (HMM AND SMA gate)
  v2_ls          : long top-2 (positive momentum) when HMM bull AND BTC > SMA;
                   short bottom-2 (negative momentum) when HMM bear AND BTC < SMA; else cash
  v2_ls_top20    : same on the causal top-20 liquidity universe
  xs_ls          : always long top-2 / short bottom-2 by 14-day momentum (no gate)
  beta_neutral   : long top-2, short bottom-2, short leg scaled so net BTC beta = 0
  beta_neutral_hmm: same, only while HMM bull (as the research note proposes)
  hawkes_ls_wfo  : Hawkes signal +1 long / -1 short, 5 params, PSO walk-forward
  pairs_ls_wfo   : Kalman pairs, long cheap leg + short rich leg, PSO walk-forward

Evaluation: real engine + kill switch + per-coin measured spreads, on the same
stitched walk-forward out-of-sample window as strategy_lab (May 2025 - Oct 2026),
plus the last 3 months and the last month.

Run:  python research/short_lab.py   (~45 min, mostly the two PSO walk-forwards)
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
warnings.filterwarnings("ignore")

import portfolio_lab as P  # noqa: E402
import strategy as S  # noqa: E402
import strategy_lab as L  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "analysis")
MAJORS = P.MAJORS
LOOKBACK, K = 336, 2

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)


def n_entries_abs(w: np.ndarray) -> int:
    on = np.abs(w) > 1e-9
    return int((on[1:] & ~on[:-1]).sum())


L.n_entries = n_entries_abs   # walk_forward must count short entries too


def size_ls(sel_long: pd.DataFrame, sel_short: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Inverse-vol size each side, combine signed, cap gross |w| at MAX_GROSS."""
    w = S.size_inverse_vol(sel_long, closes) - S.size_inverse_vol(sel_short, closes)
    gross = w.abs().sum(axis=1)
    scale = (S.MAX_GROSS / gross).where(gross > S.MAX_GROSS, 1.0)
    return w.mul(scale, axis=0)


def gate_prob(closes: pd.DataFrame, refit: int = 30) -> pd.Series:
    daily = closes["BTC"].resample("1D").last()
    x = np.log(daily).diff().dropna()
    prob = pd.Series(np.nan, index=x.index)
    for st in range(180, len(x), refit):
        mu, var, A, pi = L.hmm_fit(x.values[:st])
        bull = int(np.argmax(mu / np.sqrt(var)))
        f = L.hmm_filter(x.values[: st + refit], mu, var, A, pi)[:, bull]
        prob.iloc[st: st + refit] = f[st: st + refit]
    prob.index = prob.index + pd.Timedelta(days=1)
    return prob.reindex(closes.index, method="ffill")


def momentum_ls(closes, eligible, on, off, long_only=False) -> pd.DataFrame:
    mom = (closes / closes.shift(LOOKBACK) - 1).where(eligible)
    top = (mom.rank(axis=1, ascending=False) <= K) & (mom > 0) & on.to_numpy()[:, None]
    bot = (mom.rank(axis=1, ascending=True) <= K) & (mom < 0) & off.to_numpy()[:, None]
    if long_only:
        bot &= False
    return S.hold_until_rebalance(size_ls(top, bot, closes))


def beta_neutral(closes, eligible, gate=None) -> pd.DataFrame:
    r = np.log(closes).diff()
    beta = r.rolling(720, min_periods=360).cov(r["BTC"]).div(r["BTC"].rolling(720, min_periods=360).var(), axis=0)
    mom = (closes / closes.shift(LOOKBACK) - 1).where(eligible)
    top = mom.rank(axis=1, ascending=False) <= K
    bot = mom.rank(axis=1, ascending=True) <= K
    if gate is not None:
        top &= gate.to_numpy()[:, None]
        bot &= gate.to_numpy()[:, None]
    wl = S.size_inverse_vol(top, closes)
    ws = S.size_inverse_vol(bot, closes)
    bl = (wl * beta).sum(axis=1)
    bs = (ws * beta).sum(axis=1)
    factor = (bl / bs).where(bs > 1e-9, 0.0).clip(0, 3)
    w = wl - ws.mul(factor, axis=0)
    gross = w.abs().sum(axis=1)
    w = w.mul((S.MAX_GROSS / gross).where(gross > S.MAX_GROSS, 1.0), axis=0)
    return S.hold_until_rebalance(w.fillna(0.0))


def hawkes_ls(ohlc, closes, kappa, nlb, slb, p5, p95) -> pd.DataFrame:
    h, l, c = (ohlc[f].to_numpy() for f in ("high", "low", "close"))
    v = L.hawkes_volatility(h, l, c, kappa, int(nlb))
    sig = L.hawkes_signal(v, c, int(slb), p5, p95)
    sl = pd.DataFrame(sig > 0, index=closes.index, columns=closes.columns)
    ss = pd.DataFrame(sig < 0, index=closes.index, columns=closes.columns)
    return size_ls(sl, ss, closes)


def pairs_ls(z: np.ndarray, closes: pd.DataFrame, entry: float, exit_: float) -> pd.DataFrame:
    """z < -entry: long A / short B until z > -exit; z > entry: long B / short A until z < exit."""
    T, Pn = z.shape
    ia = np.array([a for a, _ in L.PAIRS])
    ib = np.array([b for _, b in L.PAIRS])
    posA = np.zeros(Pn, bool)
    posB = np.zeros(Pn, bool)
    net = np.zeros((T, closes.shape[1]))
    for t in range(T):
        zt = z[t]
        ok = np.isfinite(zt)
        posA = np.where(ok, np.where(posA, ~(zt > -exit_), zt < -entry), False)
        posB = np.where(ok, np.where(posB, ~(zt < exit_), zt > entry), False)
        np.add.at(net[t], ia, posA.astype(float) - posB)
        np.add.at(net[t], ib, posB.astype(float) - posA)
    sl = pd.DataFrame(net > 0, index=closes.index, columns=closes.columns)
    ss = pd.DataFrame(net < 0, index=closes.index, columns=closes.columns)
    return size_ls(sl, ss, closes)


def evaluate(o, c, w, start, end, hs, dvol) -> dict:
    sl = slice(start, end)
    used = w.columns[(w.loc[sl].abs() > 0).any()]
    if len(used) == 0:
        used = w.columns[:1]
    res = engine.run(o.loc[sl, used], c.loc[sl, used], w.loc[sl, used], half_spread=hs,
                     risk=RiskManager(), daily_vol=dvol[used])
    m = metrics.summary(res.equity)
    win = metrics.window_summary(metrics.rolling_windows(res.equity))
    days = max((res.equity.index[-1] - res.equity.index[0]).days, 1)
    wts = res.weights
    sides = res.trades["side"] if len(res.trades) else pd.Series(dtype=object)
    return {"ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "sharpe": m["sharpe"],
            "sortino": m["sortino"], "calmar": m["calmar"], "composite": m["composite"],
            "w14_profit_%": (100 - win["w14_pct_losing"]) if win["w14_pct_losing"] == win["w14_pct_losing"] else np.nan,
            "w14_worst_dd_%": win["w14_worst_dd_%"], "trades/day": len(res.trades) / days,
            "fees_%": res.fees / res.equity.iloc[0] * 100,
            "long_expo_%": wts.clip(lower=0).sum(axis=1).mean() * 100,
            "short_expo_%": -wts.clip(upper=0).sum(axis=1).mean() * 100,
            "shorts_opened": int((sides == "SHORT").sum()), "wiped": int((sides == "SHORT_WIPED").sum()),
            "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}


def main() -> None:
    t0 = time.time()
    raw_all, c_all, o_all, qv_all = P.load_all()
    hs = P.half_spreads(c_all.columns)
    dv_all = S.realized_vol(c_all) / np.sqrt(365)
    ohlc = L.load_ohlc()                       # majors, with high/low for Hawkes
    c = ohlc["close"]
    idx = c.index
    print("HMM gate ...", flush=True)
    prob = gate_prob(c_all)
    sma = S.btc_regime(c_all, 480)
    bull = (prob > 0.5).fillna(False)
    bear = (prob < 0.5).fillna(False)
    on, off = bull & sma, bear & ~sma
    print(f"  risk-on share {on.mean():.0%}, risk-off (short) share {off.mean():.0%}", flush=True)

    elig_maj = c_all.notna() & c_all.columns.isin(MAJORS)
    rank = P.liquidity_rank(qv_all)
    hist = raw_all.notna().cumsum()
    elig_t20 = raw_all.notna() & (hist >= P.MIN_HISTORY_H) & (rank <= 20)

    W = {
        "v1_fixed (live)": S.xs_momentum_regime(c).reindex(index=c_all.index, columns=c_all.columns).fillna(0.0),
        "v2_long": momentum_ls(c_all, elig_maj, on, off, long_only=True),
        "v2_ls": momentum_ls(c_all, elig_maj, on, off),
        "v2_ls_top20": momentum_ls(c_all, elig_t20, on, off),
        "xs_ls (no gate)": momentum_ls(c_all, elig_maj, pd.Series(True, index=c_all.index),
                                       pd.Series(True, index=c_all.index)),
        "beta_neutral": beta_neutral(c_all, elig_maj),
        "beta_neutral_hmm": beta_neutral(c_all, elig_maj, bull),
    }

    # --- PSO walk-forward families with shorts (majors) -----------------------
    fe = L.FastEval(ohlc["open"])
    deltas = [10 ** e for e in np.arange(-9, -4.9, 0.5)]
    print("Kalman filters ...", flush=True)
    zs = [L.kalman_zscores(c, d) for d in deltas]
    fams = {
        "hawkes_ls_wfo": (lambda p: hawkes_ls(ohlc, c, p[0], int(p[1]), int(p[2]), p[3], p[4]),
                          [0.01, 24, 48, 0.01, 0.70], [1.0, 336, 720, 0.30, 0.99], [1, 2]),
        "pairs_ls_wfo": (lambda p: pairs_ls(zs[int(p[0])], c, p[1], min(p[2], p[1] - 0.25)),
                         [0, 1.0, -0.5], [8, 3.5, 1.0], [0]),
    }
    a = idx.searchsorted(idx[0] + pd.Timedelta(days=L.WARMUP_DAYS))
    b = idx.searchsorted(L.HOLDOUT_START)
    pbo, logs = {}, {}
    for name, (fn, lo, hi, ints) in fams.items():
        cfgs = L.random_configs(lo, hi, ints, 40, seed=11)
        R = np.column_stack([fe.returns(fn(cfg).to_numpy())[a:b] for cfg in cfgs])
        pbo[name] = L.pbo_cscv(R)
        print(f"  {name} PBO = {pbo[name]:.2f}", flush=True)
        w, logs[name] = L.walk_forward(name, fn, lo, hi, ints, fe, idx)
        W[name] = w.reindex(index=c_all.index, columns=c_all.columns).fillna(0.0)

    fl = L.folds(idx)
    oos_start, end = fl[0][1], idx[-1]
    periods = {"OOS": (oos_start, end), "3M": (L.HOLDOUT_START, end),
               "1M": ((end - pd.Timedelta(days=30)).floor("D"), end)}
    rows = {}
    for name, w in W.items():
        for pn, (s0, s1) in periods.items():
            for k, v in evaluate(o_all, c_all, w, s0, s1, hs, dv_all).items():
                rows.setdefault(name, {})[(pn, k)] = v
    df = pd.DataFrame(rows).T
    df.to_csv(os.path.join(OUT_DIR, "short_lab.csv"))
    with open(os.path.join(OUT_DIR, "short_lab_folds.json"), "w") as f:
        json.dump({"folds": logs, "pbo": pbo}, f, indent=1, default=str)
    show = [("OOS", k) for k in ("ret_%", "max_dd_%", "sharpe", "sortino", "calmar", "composite",
                                 "w14_profit_%", "w14_worst_dd_%", "long_expo_%", "short_expo_%",
                                 "shorts_opened", "wiped", "kills", "fees_%")]
    show += [("3M", "ret_%"), ("3M", "max_dd_%"), ("3M", "composite"),
             ("1M", "ret_%"), ("1M", "max_dd_%"), ("1M", "composite")]
    out = df[show].astype(float).round(2)
    out.columns = [f"{p}_{k}" for p, k in show]
    out["PBO"] = pd.Series(pbo)
    print(f"\n=== WITH SHORTS: OOS {oos_start:%Y-%m-%d} -> {end:%Y-%m-%d}, 3M from {L.HOLDOUT_START:%Y-%m-%d}, "
          f"1M from {periods['1M'][0]:%Y-%m-%d} (real engine, kill switch, measured spreads) ===")
    print(out.to_string())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
