"""Portfolio lab: which coins, how many, and how to split the money.

Base signal = v2 proposal: 14-day momentum, held only when BTC's HMM bull
probability > 0.5 AND BTC > its 480h SMA; daily rebalance at 00:00 UTC.

Grid (fixed rules, nothing tuned; every combination is reported):
  universe : majors6 (fixed list) | top20 / top40 by trailing 30-day median daily
             Binance volume, recomputed daily (causal), >= 60 days of history
  K        : number of coins held: 2, 5, 10, 15 (top K by 14-day return, return > 0)
  sizing   : invvol_c1  - current: TARGET_VOL / (K * vol_i), assumes correlation 1
             invvol_cov - weights ~ 1/vol_i, book scaled to TARGET_VOL with a real covariance
             erc_cov    - equal risk contribution (risk parity), scaled with covariance
             minvar_cov - long-only minimum variance (capped), scaled with covariance
  Covariance: 720h of hourly returns, Ledoit-Wolf shrinkage of the correlation matrix.
  Caps everywhere: 25% per coin, 50% gross (settings.py).

Costs: 0.1% fee + each coin's measured Roostoo half-spread (floor 2 bp).
Selection rule (fixed before running): best in-sample composite among configs
whose in-sample worst 14-day drawdown is better than -10%.

Run:  python research/portfolio_lab.py
"""
import itertools
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

import strategy as S  # noqa: E402
import strategy_lab as L  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "klines_1h")
OUT_DIR = os.path.join(ROOT, "data", "analysis")
MAJORS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
EXCLUDE = {"TON", "OMNI"}            # no recent Binance data -> Roostoo price source unknown
MIN_HISTORY_H = 60 * 24
LOOKBACK = 336
COV_WINDOW = 720
IS_START = pd.Timestamp("2025-04-01", tz="UTC")   # HMM gate needs 180 days of daily data
HOLDOUT_START = pd.Timestamp("2026-07-05", tz="UTC")
MAX_WORST_14D_DD = -10.0

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)


# ---------------------------------------------------------------------------
def load_all():
    closes, opens, qv = {}, {}, {}
    for f in sorted(os.listdir(DATA_DIR)):
        coin = f[:-4]
        if coin in EXCLUDE:
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, f), index_col=0, parse_dates=True)
        closes[coin], opens[coin], qv[coin] = df["close"], df["open"], df["quote_volume"]
    raw = pd.DataFrame(closes)
    idx = pd.date_range(raw.index[0], raw.index[-1], freq="h", tz="UTC")
    raw = raw.reindex(idx)
    o = pd.DataFrame(opens).reindex(idx)
    v = pd.DataFrame(qv).reindex(idx)
    return raw, raw.ffill(), o.ffill(), v


def liquidity_rank(qv: pd.DataFrame) -> pd.DataFrame:
    """Rank (1 = most liquid) by trailing 30-day median daily volume, known at each hour (causal)."""
    daily = qv.resample("1D").sum(min_count=1)
    med = daily.rolling(30, min_periods=20).median().shift(1)        # uses days strictly before
    rank = med.rank(axis=1, ascending=False)
    return rank.reindex(qv.index, method="ffill")


def half_spreads(coins) -> dict:
    sp = pd.read_csv(os.path.join(OUT_DIR, "roostoo_spreads.csv"), index_col=0)["spread_bps"]
    return {c: max(float(sp.get(c, 10.0)) / 2 / 1e4, engine.HALF_SPREAD) for c in coins}


# ---------------------------------------------------------------------------
# Covariance and optimizers (numpy only)
# ---------------------------------------------------------------------------
def shrunk_cov(R: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage of the correlation matrix toward identity, rescaled by vols."""
    sd = R.std(axis=0, ddof=1)
    sd[sd == 0] = 1e-12
    X = (R - R.mean(axis=0)) / sd
    T, N = X.shape
    Sm = X.T @ X / T
    F = np.eye(N) * np.trace(Sm) / N
    d2 = ((Sm - F) ** 2).sum()
    xx = (X ** 2).sum(axis=1)
    quad = np.einsum("ti,ij,tj->t", X, Sm, X)
    b2 = ((xx ** 2 - 2 * quad).sum() / T + (Sm ** 2).sum()) / T
    shrink = 0.0 if d2 == 0 else min(max(b2 / d2, 0.0), 1.0)
    C = shrink * F + (1 - shrink) * Sm
    return C * np.outer(sd, sd)


def erc(cov: np.ndarray, iters: int = 300) -> np.ndarray:
    w = 1 / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(iters):
        rc = w * (cov @ w)
        w = w * np.sqrt(rc.mean() / np.maximum(rc, 1e-18))
        w /= w.sum()
    return w


def project_capped_simplex(y: np.ndarray, cap: float) -> np.ndarray:
    lo, hi = y.min() - cap, y.max()
    for _ in range(60):
        tau = (lo + hi) / 2
        if np.clip(y - tau, 0, cap).sum() > 1:
            lo = tau
        else:
            hi = tau
    return np.clip(y - (lo + hi) / 2, 0, cap)


def minvar(cov: np.ndarray, cap: float, iters: int = 400) -> np.ndarray:
    n = len(cov)
    w = np.full(n, 1 / n)
    step = 1 / np.linalg.eigvalsh(cov).max()
    for _ in range(iters):
        w = project_capped_simplex(w - step * (cov @ w), cap)
    return w


def scale_and_cap(rel: np.ndarray, cov_ann: np.ndarray) -> np.ndarray:
    port_vol = float(np.sqrt(rel @ cov_ann @ rel))
    w = rel * (S.TARGET_VOL / port_vol) if port_vol > 0 else rel * 0
    w = np.minimum(w, S.MAX_PER_COIN)
    if w.sum() > S.MAX_GROSS:
        w *= S.MAX_GROSS / w.sum()
    return w


# ---------------------------------------------------------------------------
def build_weights(raw, closes, rank, gate, universe: str, K: int, sizing: str) -> pd.DataFrame:
    logret = np.log(closes).diff()
    mom = closes / closes.shift(LOOKBACK) - 1
    hist = raw.notna().cumsum()
    vol168 = S.realized_vol(closes)
    W = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    reb = closes.index[closes.index.hour == S.REBALANCE_HOUR]
    for t in reb:
        if t < IS_START - pd.Timedelta(days=5) or not gate.loc[t]:
            continue
        ok = raw.loc[t].notna() & (hist.loc[t] >= MIN_HISTORY_H) & (mom.loc[t] > 0)
        if universe == "majors6":
            ok &= closes.columns.isin(MAJORS)
        else:
            ok &= rank.loc[t] <= int(universe[3:])
        cand = mom.loc[t][ok].sort_values(ascending=False).index[:K]
        if len(cand) == 0:
            continue
        vols = vol168.loc[t, cand].to_numpy()
        if sizing == "invvol_c1":
            w = np.minimum(S.TARGET_VOL / (len(cand) * vols), S.MAX_PER_COIN)
            if w.sum() > S.MAX_GROSS:
                w *= S.MAX_GROSS / w.sum()
        else:
            R = logret.loc[:t, cand].iloc[-COV_WINDOW:].dropna().to_numpy()
            if len(R) < COV_WINDOW // 2:
                continue
            cov = shrunk_cov(R)
            if sizing == "invvol_cov":
                rel = (1 / vols) / (1 / vols).sum()
            elif sizing == "erc_cov":
                rel = erc(cov)
            else:
                rel = minvar(cov, cap=min(1.0, 3.0 / len(cand)))
            w = scale_and_cap(rel, cov * S.HOURS_PER_YEAR)
        W.loc[t, cand] = np.nan_to_num(w)
    W.loc[~W.index.isin(reb)] = np.nan
    return W.ffill().fillna(0.0)


def run_period(o, c, w, start, end, hs, dvol):
    used = w.columns[(w.loc[start:end] > 0).any()]
    if len(used) == 0:
        used = w.columns[:1]
    sl = slice(start, end)
    res = engine.run(o.loc[sl, used], c.loc[sl, used], w.loc[sl, used], half_spread=hs,
                     risk=RiskManager(), daily_vol=dvol[used])
    m = metrics.summary(res.equity)
    win = metrics.window_summary(metrics.rolling_windows(res.equity))
    days = max((res.equity.index[-1] - res.equity.index[0]).days, 1)
    return {"ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "sharpe": m["sharpe"],
            "composite": m["composite"], "w14_worst_dd_%": win["w14_worst_dd_%"],
            "w14_profit_%": 100 - win["w14_pct_losing"] if win["w14_pct_losing"] == win["w14_pct_losing"] else np.nan,
            "trades/day": len(res.trades) / days, "fees_%": res.fees / res.equity.iloc[0] * 100,
            "avg_expo_%": res.weights.sum(axis=1).mean() * 100,
            "avg_coins": (res.weights > 0.005).sum(axis=1)[res.weights.sum(axis=1) > 0.01].mean(),
            "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}


def main() -> None:
    t0 = time.time()
    raw, closes, opens, qv = load_all()
    rank = liquidity_rank(qv)
    hs = half_spreads(closes.columns)
    dvol = S.realized_vol(closes) / np.sqrt(365)
    print(f"{closes.shape[1]} coins. Building gates ...", flush=True)
    hmm_gate, _ = L.hmm_bull_gate(closes)
    sma = S.btc_regime(closes, 480)
    gate_v2 = hmm_gate & sma
    end = closes.index[-1]
    month = (end - pd.Timedelta(days=30)).floor("D")
    periods = {"IS": (IS_START, HOLDOUT_START - pd.Timedelta(hours=1)),
               "HO": (HOLDOUT_START, end), "1M": (month, end)}

    configs = [("majors6", k, s) for k in (2, 5) for s in ("invvol_c1", "invvol_cov", "erc_cov", "minvar_cov")]
    configs += [(u, k, s) for u in ("top20", "top40") for k in (2, 5, 10, 15)
                for s in ("invvol_c1", "invvol_cov", "erc_cov", "minvar_cov")]
    results = {}
    for u, k, s in configs:
        w = build_weights(raw, closes, rank, gate_v2, u, k, s)
        row = {}
        for pname, (a, b) in periods.items():
            for key, val in run_period(opens, closes, w, a, b, hs, dvol).items():
                row[(pname, key)] = val
        results[(u, k, s)] = row
        print(f"  {u:<8} K={k:<2} {s:<11} IS ret {row[('IS','ret_%')]:6.1f}% dd {row[('IS','max_dd_%')]:6.1f}% "
              f"comp {row[('IS','composite')]:5.2f} | HO ret {row[('HO','ret_%')]:5.1f}% dd {row[('HO','max_dd_%')]:5.1f}% "
              f"| 1M ret {row[('1M','ret_%')]:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)

    # baselines: live v1 (SMA gate, majors6, K=2, invvol_c1) and buy & hold BTC
    w_v1 = S.xs_momentum_regime(closes[MAJORS]).reindex(columns=closes.columns, fill_value=0.0)
    w_bh = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    w_bh["BTC"] = 1.0
    for name, w in (("BASELINE live v1", w_v1), ("BASELINE buy&hold BTC", w_bh)):
        row = {}
        for pname, (a, b) in periods.items():
            for key, val in run_period(opens, closes, w, a, b, hs, dvol).items():
                row[(pname, key)] = val
        results[(name, 0, "")] = row

    df = pd.DataFrame(results).T
    df.index.names = ["universe", "K", "sizing"]
    df.to_csv(os.path.join(OUT_DIR, "portfolio_lab.csv"))
    cols = [("IS", "ret_%"), ("IS", "max_dd_%"), ("IS", "composite"), ("IS", "w14_worst_dd_%"),
            ("IS", "avg_expo_%"), ("IS", "avg_coins"), ("IS", "fees_%"), ("IS", "kills"),
            ("HO", "ret_%"), ("HO", "max_dd_%"), ("HO", "composite"),
            ("1M", "ret_%"), ("1M", "max_dd_%"), ("1M", "composite")]
    out = df[cols].astype(float).round(2)
    out.columns = [f"{a}_{b}" for a, b in cols]
    print(f"\n=== ALL CONFIGS (IS {IS_START:%Y-%m-%d}..{HOLDOUT_START:%Y-%m-%d}, HO to {end:%Y-%m-%d}, "
          f"1M from {month:%Y-%m-%d}) ===")
    print(out.to_string())
    grid = df.drop(index=[i for i in df.index if str(i[0]).startswith("BASELINE")])
    elig = grid[grid[("IS", "w14_worst_dd_%")].astype(float) > MAX_WORST_14D_DD]
    pick = (elig if len(elig) else grid)[("IS", "composite")].astype(float).idxmax()
    print(f"\n>>> Selected on in-sample only: {pick}")
    print(out.loc[[pick]].to_string())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
