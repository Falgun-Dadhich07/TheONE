"""Strategy lab: tests the ideas from the team's research note and the Hawkes
("lunar phase") volatility strategy, under one honest protocol.

Candidates (all long-only, spot; shorts are off and "arbitrage" is banned):
  * v1_fixed      : current live strategy (xs momentum + BTC>SMA gate), fixed params
  * v1_wfo        : same family, params re-optimized walk-forward with PSO
  * v1_hmm        : v1 with the BTC SMA gate replaced by a 2-state Gaussian HMM
                    (bull-state filtered probability > 0.5), refit monthly, causal
  * hawkes_wfo    : Hawkes volatility-breakout signal per coin (signal -1 -> cash),
                    5 params optimized walk-forward with PSO
  * pairs_wfo     : Kalman-filter pairs mean reversion, LONG-ONLY (buy the cheap
                    leg, exit on reversion); 3 params optimized walk-forward with PSO
  * pairs_wfo_gate: same, only while BTC > its 480h SMA
  * benchmarks    : buy & hold BTC, equal weight 50%

Protocol:
  * Walk-forward: optimize on 180 days, trade the next 60 days blind, roll 60 days.
    Stitched out-of-sample periods are evaluated with the real engine + kill switch.
  * PSO objective = competition composite on the training window, net of costs
    (fast vectorized evaluator: next-bar-open execution, fee + half-spread on turnover).
  * Overfitting: CSCV probability of backtest overfitting (Bailey et al. 2015)
    on 40 random configs per family over the pre-holdout period.

Run:  python research/strategy_lab.py      (takes ~20-30 min)
"""
import itertools
import json
import os
import sys
import time
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import strategy as S  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
DATA_DIR = os.path.join(ROOT, "data", "klines_1h")
OUT_DIR = os.path.join(ROOT, "data", "analysis")
HOLDOUT_START = pd.Timestamp("2026-07-05", tz="UTC")
COST = engine.TAKER_FEE + engine.HALF_SPREAD
N_YEAR = 24 * 365
WARMUP_DAYS = 45
TRAIN_DAYS, TEST_DAYS = 180, 60

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


# ---------------------------------------------------------------------------
# Data and fast evaluation
# ---------------------------------------------------------------------------
def load_ohlc() -> Dict[str, pd.DataFrame]:
    fields = {f: {} for f in ("open", "high", "low", "close")}
    for c in UNIVERSE:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{c}.csv"), index_col=0, parse_dates=True)
        for f in fields:
            fields[f][c] = df[f]
    out = {f: pd.DataFrame(v) for f, v in fields.items()}
    full = pd.date_range(out["close"].index[0], out["close"].index[-1], freq="h", tz="UTC")
    return {f: d.reindex(full).ffill() for f, d in out.items()}


class FastEval:
    """Hourly portfolio returns for a weight matrix: weights decided at close t
    are held from open t+1 to open t+2; turnover pays COST."""

    def __init__(self, opens: pd.DataFrame):
        o = opens.to_numpy()
        r = np.zeros_like(o)
        r[:-1] = o[1:] / o[:-1] - 1
        self.r_oo = np.nan_to_num(r)
        self.index = opens.index

    def returns(self, w: np.ndarray) -> np.ndarray:
        held = np.zeros_like(w)
        held[1:] = w[:-1]
        turn = np.zeros(len(w))
        turn[1:] = np.abs(held[1:] - held[:-1]).sum(axis=1)
        return (held * self.r_oo).sum(axis=1) - COST * turn


def score(r: np.ndarray) -> Dict[str, float]:
    r = r[np.isfinite(r)]
    if len(r) < 48 or r.std() == 0:
        return {"composite": -10.0, "sharpe": 0.0, "ret": 0.0, "mdd": 0.0}
    eq = np.cumprod(1 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    total = eq[-1] - 1
    years = len(r) / N_YEAR
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1
    sharpe = r.mean() / r.std() * np.sqrt(N_YEAR)
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    sortino = r.mean() / dd * np.sqrt(N_YEAR) if dd > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {"composite": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar,
            "sharpe": sharpe, "ret": total, "mdd": mdd}


def n_entries(w: np.ndarray) -> int:
    on = w > 1e-9
    return int((on[1:] & ~on[:-1]).sum())


# ---------------------------------------------------------------------------
# Hawkes volatility strategy (team's "lunar phase" strategy, long-only)
# ---------------------------------------------------------------------------
def hawkes_volatility(h, l, c, kappa: float, norm_lb: int) -> np.ndarray:
    """kappa * Hawkes-smoothed (high-low)/ATR of log prices. Arrays T x N."""
    lh, ll, lc = np.log(h), np.log(l), np.log(c)
    prev = np.vstack([np.full((1, lc.shape[1]), np.nan), lc[:-1]])
    tr = np.fmax(lh - ll, np.fmax(np.abs(lh - prev), np.abs(ll - prev)))
    # Wilder ATR (TA-Lib seeds with an SMA; the difference vanishes after warm-up)
    atr = pd.DataFrame(tr).ewm(alpha=1.0 / norm_lb, adjust=False, min_periods=norm_lb).mean().to_numpy()
    nr = (lh - ll) / atr
    alpha = np.exp(-kappa)
    out = np.full_like(nr, np.nan)
    for i in range(1, len(nr)):
        p = out[i - 1]
        out[i] = np.where(np.isnan(p), nr[i], p * alpha + nr[i])
    return out * kappa


def hawkes_signal(v: np.ndarray, c: np.ndarray, sig_lb: int, p5: float, p95: float) -> np.ndarray:
    """Same state machine as the original code, vectorized over coins.
    (Original bug fixed: at i=0 it compared with v[-1], the LAST bar.)"""
    vdf = pd.DataFrame(v)
    q05 = vdf.rolling(sig_lb).quantile(p5).to_numpy()
    q95 = vdf.rolling(sig_lb).quantile(p95).to_numpy()
    T, N = v.shape
    sig = np.zeros((T, N))
    last_below = np.full(N, -1)
    cur = np.zeros(N)
    for i in range(1, T):
        below = v[i] < q05[i]
        last_below = np.where(below, i, last_below)
        cur = np.where(below, 0.0, cur)
        cross = (v[i] > q95[i]) & (v[i - 1] <= q95[i - 1]) & (last_below > 0)
        if cross.any():
            idx = np.where(cross)[0]
            change = c[i, idx] - c[last_below[idx], idx]
            cur[idx] = np.where(change > 0, 1.0, -1.0)
        sig[i] = cur
    return sig


def hawkes_weights(ohlc, closes, kappa, norm_lb, sig_lb, p5, p95) -> pd.DataFrame:
    h, l, c = (ohlc[f].to_numpy() for f in ("high", "low", "close"))
    v = hawkes_volatility(h, l, c, kappa, int(norm_lb))
    sig = hawkes_signal(v, c, int(sig_lb), p5, p95)
    selected = pd.DataFrame(sig > 0, index=closes.index, columns=closes.columns)  # -1 -> cash
    return S.size_inverse_vol(selected, closes)


# ---------------------------------------------------------------------------
# Kalman-filter pairs mean reversion (long-only)
# ---------------------------------------------------------------------------
PAIRS = list(itertools.combinations(range(len(UNIVERSE)), 2))


def kalman_zscores(closes: pd.DataFrame, delta: float, ve: float = 1e-2, z_span: int = 168) -> np.ndarray:
    """Innovation z-scores of log A = beta*log B + alpha with a random-walk
    [beta, alpha] (Kalman filter), vectorized over all pairs. Causal: the
    innovation at t uses the state estimated from data up to t-1."""
    lp = np.log(closes.to_numpy())
    A = lp[:, [a for a, _ in PAIRS]]
    B = lp[:, [b for _, b in PAIRS]]
    T, P = A.shape
    beta = np.zeros((P, 2))
    beta[:, 0] = 1.0
    Pm = np.tile(np.eye(2), (P, 1, 1))
    Vw = delta / (1 - delta) * np.eye(2)
    e = np.full((T, P), np.nan)
    for t in range(T):
        x = np.stack([B[t], np.ones(P)], axis=1)
        R = Pm + Vw
        Rx = np.einsum("pij,pj->pi", R, x)
        Q = (x * Rx).sum(axis=1) + ve
        et = A[t] - (x * beta).sum(axis=1)
        K = Rx / Q[:, None]
        beta = beta + K * et[:, None]
        Pm = R - K[:, :, None] * Rx[:, None, :]
        e[t] = et
    sd = np.sqrt(pd.DataFrame(e ** 2).ewm(span=z_span, min_periods=z_span).mean().shift(1).to_numpy())
    return e / sd


def pairs_weights(z: np.ndarray, closes: pd.DataFrame, entry: float, exit_: float,
                  gate: np.ndarray = None) -> pd.DataFrame:
    """z < -entry: A is cheap vs B -> hold A until z > -exit. z > entry: hold B until z < exit."""
    T, P = z.shape
    posA = np.zeros(P, bool)
    posB = np.zeros(P, bool)
    count = np.zeros((T, len(UNIVERSE)))
    ia = np.array([a for a, _ in PAIRS])
    ib = np.array([b for _, b in PAIRS])
    for t in range(T):
        zt = z[t]
        ok = np.isfinite(zt)
        posA = np.where(ok, np.where(posA, ~(zt > -exit_), zt < -entry), False)
        posB = np.where(ok, np.where(posB, ~(zt < exit_), zt > entry), False)
        np.add.at(count[t], ia, posA)
        np.add.at(count[t], ib, posB)
    sel = count > 0
    if gate is not None:
        sel &= gate[:, None]
    return S.size_inverse_vol(pd.DataFrame(sel, index=closes.index, columns=closes.columns), closes)


# ---------------------------------------------------------------------------
# 2-state Gaussian HMM regime (numpy Baum-Welch, causal forward filtering)
# ---------------------------------------------------------------------------
def _emis(x, mu, var):
    return np.exp(-(x[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var) + 1e-300


def hmm_fit(x: np.ndarray, n_iter: int = 200, tol: float = 1e-6):
    mu = np.quantile(x, [0.25, 0.75])
    var = np.full(2, x.var())
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    prev_ll = -np.inf
    for _ in range(n_iter):
        Bm = _emis(x, mu, var)
        T = len(x)
        al = np.zeros((T, 2))
        cs = np.zeros(T)
        al[0] = pi * Bm[0]
        cs[0] = al[0].sum()
        al[0] /= cs[0]
        for t in range(1, T):
            al[t] = (al[t - 1] @ A) * Bm[t]
            cs[t] = al[t].sum()
            al[t] /= cs[t]
        be = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            be[t] = (A @ (Bm[t + 1] * be[t + 1])) / cs[t + 1]
        g = al * be
        g /= g.sum(axis=1, keepdims=True)
        xi = (al[:-1, :, None] * A[None] * (Bm[1:] * be[1:])[:, None, :]) / cs[1:, None, None]
        pi = g[0]
        A = xi.sum(axis=0) / g[:-1].sum(axis=0)[:, None]
        A /= A.sum(axis=1, keepdims=True)
        mu = (g * x[:, None]).sum(axis=0) / g.sum(axis=0)
        var = (g * (x[:, None] - mu) ** 2).sum(axis=0) / g.sum(axis=0) + 1e-10
        ll = np.log(cs).sum()
        if ll - prev_ll < tol:
            break
        prev_ll = ll
    return mu, var, A, pi


def hmm_filter(x, mu, var, A, pi) -> np.ndarray:
    Bm = _emis(x, mu, var)
    al = np.zeros((len(x), 2))
    a = pi * Bm[0]
    al[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (al[t - 1] @ A) * Bm[t]
        al[t] = a / a.sum()
    return al


def hmm_bull_gate(closes: pd.DataFrame, min_days: int = 180, refit_days: int = 30) -> Tuple[pd.Series, list]:
    """Hourly boolean gate. Day d's filtered bull probability (data <= end of day d)
    gates every hour of day d+1. Model refit every `refit_days` on all data so far."""
    daily = closes["BTC"].resample("1D").last()
    x = np.log(daily).diff().dropna()
    prob = pd.Series(np.nan, index=x.index)
    fits = []
    for start in range(min_days, len(x), refit_days):
        mu, var, A, pi = hmm_fit(x.values[:start])
        bull = int(np.argmax(mu / np.sqrt(var)))   # best return per unit of risk (labels can swap)
        filt = hmm_filter(x.values[: start + refit_days], mu, var, A, pi)[:, bull]
        prob.iloc[start: start + refit_days] = filt[start: start + refit_days]
        fits.append({"fit_until": str(x.index[start - 1].date()), "mu_ann_%": (mu * 365 * 100).round(0).tolist(),
                     "vol_ann_%": (np.sqrt(var * 365) * 100).round(0).tolist(), "bull": bull})
    gate_daily = (prob > 0.5)
    gate_daily.index = gate_daily.index + pd.Timedelta(days=1)       # usable the next day
    hourly = gate_daily.reindex(closes.index, method="ffill").fillna(False).astype(bool)
    return hourly, fits


def momentum_with_gate(closes, gate: pd.Series, lookback=336, top_n=2) -> pd.DataFrame:
    """v1 with an arbitrary regime gate instead of BTC > SMA."""
    mom = closes / closes.shift(lookback) - 1
    rank = mom.rank(axis=1, ascending=False)
    sel = (rank <= top_n) & (mom > 0) & gate.to_numpy()[:, None]
    return S.hold_until_rebalance(S.size_inverse_vol(sel, closes))


# ---------------------------------------------------------------------------
# PSO, walk-forward, CSCV
# ---------------------------------------------------------------------------
def pso(obj: Callable[[tuple], float], lo, hi, ints, n=20, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(lo, float), np.array(hi, float)
    d = len(lo)

    def fix(x):
        x = np.clip(x, lo, hi)
        x[:, ints] = np.round(x[:, ints])
        return x

    cache = {}

    def f(row):
        key = tuple(np.round(row, 3))
        if key not in cache:
            cache[key] = obj(key)
        return cache[key]

    x = fix(rng.uniform(lo, hi, (n, d)))
    v = np.zeros((n, d))
    fx = np.array([f(r) for r in x])
    pb, pf = x.copy(), fx.copy()
    g = pb[np.argmax(pf)].copy()
    for _ in range(iters):
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        v = 0.7 * v + 1.5 * r1 * (pb - x) + 1.5 * r2 * (g - x)
        v = np.clip(v, -(hi - lo) * 0.3, (hi - lo) * 0.3)
        x = fix(x + v)
        fx = np.array([f(r) for r in x])
        better = fx > pf
        pb[better], pf[better] = x[better], fx[better]
        g = pb[np.argmax(pf)].copy()
    return tuple(np.round(g, 3)), float(pf.max()), len(cache)


def folds(index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    out = []
    t0 = index[0] + pd.Timedelta(days=WARMUP_DAYS)
    while True:
        tr_end = t0 + pd.Timedelta(days=TRAIN_DAYS)
        te_end = tr_end + pd.Timedelta(days=TEST_DAYS)
        if tr_end >= index[-1]:
            break
        out.append((t0, tr_end, min(te_end, index[-1] + pd.Timedelta(hours=1))))
        t0 = t0 + pd.Timedelta(days=TEST_DAYS)
    return out


def walk_forward(name, weights_fn, lo, hi, ints, fe: FastEval, idx, min_entries=6, seed=0):
    """Returns stitched OOS weight frame and per-fold chosen params."""
    pieces, log = [], []
    for k, (a, b, c) in enumerate(folds(idx)):
        ia, ib = idx.searchsorted(a), idx.searchsorted(b)

        def obj(params):
            w = weights_fn(params).to_numpy()
            seg = w[ia:ib]
            if n_entries(seg) < min_entries:
                return -10.0
            return score(fe.returns(w)[ia:ib])["composite"]

        t = time.time()
        best, val, n_eval = pso(obj, lo, hi, ints, seed=seed + k)
        w = weights_fn(best)
        pieces.append(w.loc[(w.index >= b) & (w.index < c)])
        oos = score(fe.returns(w.to_numpy())[ib: idx.searchsorted(c)])
        log.append({"fold": k, "train": f"{a:%Y-%m-%d}..{b:%Y-%m-%d}", "test": f"{b:%Y-%m-%d}..{c:%Y-%m-%d}",
                    "params": best, "train_composite": round(val, 2), "test_composite": round(oos["composite"], 2),
                    "test_ret_%": round(oos["ret"] * 100, 2), "evals": n_eval, "secs": round(time.time() - t)})
        print(f"  [{name}] fold {k}: {log[-1]}", flush=True)
    return pd.concat(pieces), log


def pbo_cscv(R: np.ndarray, n_blocks: int = 10) -> float:
    """Probability that the in-sample best config ranks below median out-of-sample."""
    T, N = R.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    bad, total = 0, 0
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        os_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])

        def sh(ix):
            r = R[ix]
            sd = r.std(axis=0)
            return np.where(sd > 0, r.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)

        best = int(np.argmax(sh(is_idx)))
        oos = sh(os_idx)
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1)
        w = (rank + 1) / (N + 1)
        bad += np.log(w / (1 - w)) <= 0
        total += 1
    return bad / total


def random_configs(lo, hi, ints, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, (n, len(lo)))
    x[:, ints] = np.round(x[:, ints])
    return [tuple(np.round(r, 3)) for r in x]


# ---------------------------------------------------------------------------
def evaluate_engine(ohlc, w: pd.DataFrame, start, end, dvol) -> dict:
    sl = slice(start, end)
    res = engine.run(ohlc["open"].loc[sl], ohlc["close"].loc[sl], w.reindex(ohlc["close"].index).fillna(0).loc[sl],
                     risk=RiskManager(), daily_vol=dvol)
    m = metrics.summary(res.equity)
    win = metrics.window_summary(metrics.rolling_windows(res.equity))
    days = max((res.equity.index[-1] - res.equity.index[0]).days, 1)
    return {"ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "calmar": m["calmar"], "composite": m["composite"], "w14_profit_%": 100 - win["w14_pct_losing"],
            "w14_median_%": win["w14_median_ret_%"], "w14_worst_dd_%": win["w14_worst_dd_%"],
            "trades/day": len(res.trades) / days, "fees_%": res.fees / res.equity.iloc[0] * 100,
            "avg_exposure_%": res.weights.sum(axis=1).mean() * 100,
            "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}


def main() -> None:
    t_all = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    ohlc = load_ohlc()
    closes = ohlc["close"]
    idx = closes.index
    fe = FastEval(ohlc["open"])
    dvol = S.realized_vol(closes) / np.sqrt(365)
    fl = folds(idx)
    oos_start, oos_end = fl[0][1], idx[-1]
    print(f"Walk-forward: {len(fl)} folds, train {TRAIN_DAYS}d / test {TEST_DAYS}d; "
          f"stitched OOS {oos_start:%Y-%m-%d} -> {oos_end:%Y-%m-%d}\n", flush=True)

    # --- families: (weights_fn(params), lo, hi, int dims) --------------------
    hawk_cache: Dict[tuple, np.ndarray] = {}

    def hawk_fn(p):
        kappa, nlb, slb, p5, p95 = p
        return hawkes_weights(ohlc, closes, kappa, int(nlb), int(slb), p5, p95)

    # Only delta/ve matters; faster filters make the z-score white noise (checked: acf24 ~ 0).
    deltas = [10 ** e for e in np.arange(-9, -4.9, 0.5)]
    print("Kalman filters for 15 pairs x 9 deltas ...", flush=True)
    zs = [kalman_zscores(closes, d) for d in deltas]
    btc_gate = S.btc_regime(closes, 480).to_numpy()

    def pairs_fn(p, gate=None):
        di, entry, exit_ = p
        return pairs_weights(zs[int(di)], closes, entry, min(exit_, entry - 0.25), gate)

    def mom_fn(p):
        lb, rs, n = p
        return S.xs_momentum_regime(closes, lookback=int(lb), regime_sma=int(rs), top_n=int(n))

    fams = {
        "v1_wfo": (mom_fn, [72, 120, 1], [720, 1200, 3], [0, 1, 2]),
        "hawkes_wfo": (hawk_fn, [0.01, 24, 48, 0.01, 0.70], [1.0, 336, 720, 0.30, 0.99], [1, 2]),
        "pairs_wfo": (pairs_fn, [0, 1.0, -0.5], [8, 3.5, 1.0], [0]),
        "pairs_wfo_gate": (lambda p: pairs_fn(p, btc_gate), [0, 1.0, -0.5], [8, 3.5, 1.0], [0]),
    }

    # --- CSCV / PBO on the pre-holdout period --------------------------------
    print("\nCSCV probability of backtest overfitting (40 random configs each, pre-holdout):", flush=True)
    a = idx.searchsorted(idx[0] + pd.Timedelta(days=WARMUP_DAYS))
    b = idx.searchsorted(HOLDOUT_START)
    pbo = {}
    for name, (fn, lo, hi, ints) in fams.items():
        cfgs = random_configs(lo, hi, ints, 40, seed=11)
        R = np.column_stack([fe.returns(fn(cfg).to_numpy())[a:b] for cfg in cfgs])
        pbo[name] = pbo_cscv(R)
        print(f"  {name:<16} PBO = {pbo[name]:.2f}", flush=True)

    # --- walk-forward ---------------------------------------------------------
    oos_w, logs = {}, {}
    for name, (fn, lo, hi, ints) in fams.items():
        print(f"\nWalk-forward PSO: {name}", flush=True)
        oos_w[name], logs[name] = walk_forward(name, fn, lo, hi, ints, fe, idx)

    print("\nHMM regime gate ...", flush=True)
    gate, fits = hmm_bull_gate(closes)
    for f_ in fits[::3]:
        print("  ", f_)
    oos_w["v1_hmm"] = momentum_with_gate(closes, gate)
    oos_w["v1_fixed"] = S.xs_momentum_regime(closes)
    oos_w["buy_hold_btc"] = S.buy_hold_btc(closes)
    oos_w["equal_weight_half"] = S.equal_weight_half(closes)
    print(f"  HMM bull share of OOS hours: {gate.loc[oos_start:].mean():.0%}; "
          f"BTC>SMA share: {S.btc_regime(closes, 480).loc[oos_start:].mean():.0%}")

    # --- one table, real engine with kill switch, stitched OOS ---------------
    rows = {n: evaluate_engine(ohlc, w, oos_start, oos_end, dvol) for n, w in oos_w.items()}
    table = pd.DataFrame(rows).T
    table["PBO"] = pd.Series(pbo)
    order = ["v1_fixed", "v1_wfo", "v1_hmm", "hawkes_wfo", "pairs_wfo", "pairs_wfo_gate",
             "buy_hold_btc", "equal_weight_half"]
    table = table.loc[order]
    print(f"\n=== OUT-OF-SAMPLE {oos_start:%Y-%m-%d} -> {oos_end:%Y-%m-%d} "
          f"(real engine, fee+spread, 8% kill switch) ===")
    print(table.round(2).to_string())
    table.to_csv(os.path.join(OUT_DIR, "lab_oos_table.csv"))
    with open(os.path.join(OUT_DIR, "lab_wfo_folds.json"), "w") as f:
        json.dump({"folds": logs, "hmm_fits": fits, "pbo": pbo}, f, indent=1, default=str)
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
