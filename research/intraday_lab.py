"""Intraday lab: do faster strategies and limit-order execution beat v3?

Data: 5-minute Binance klines (research/fetch_intraday.py), 25 liquid coins,
Sep 2025 - Oct 2026, resampled to 15-minute bars for signals.

Execution models (per bar; decisions at the close of bar t, traded during bar t+1):
  taker : fill at the next bar's open, pay 0.10% + half-spread.
  maker : rest a limit at the decision close price for one bar.
          BUY fills only if the next bar trades THROUGH it (low < limit - 1bp),
          SELL only if high > limit + 1bp; fee 0.05%. If it does not fill, the
          order is cancelled and executed at the next bar's close as a taker
          (0.10% + half-spread), wherever the price has gone.
          Trade-through (not touch) keeps this honest for a real market with a
          queue: we get filled when price moves against us and chase when it
          runs away. Verified on Roostoo: resting limits pay 0.05%, marketable
          limits pay 0.10% (probe_limits.py).

Strategies (long/short, inverse-vol sized with settings.py caps):
  hawkes15  : Hawkes volatility breakout on 15m bars (5 params)
  breakout  : Donchian channel breakout / half-channel exit (1 param + regime option)
  reversal  : fade an N-bar z-score move, hold until it normalizes (3 params)
  v3_hourly : the deployed v3 momentum sleeve + Hawkes with live params on hourly
              bars from the same data (reference; shows what maker execution saves)

Protocol: walk-forward, train 90 days / test 30 days / step 30 days. Each
family has a fixed pool of random configs evaluated once over the whole
history; each fold trades the config with the best training composite (net of
costs, under the SAME execution model that is evaluated). Kill switch (settings.MAX_DRAWDOWN, 24h) applied to the
stitched out-of-sample returns.

Run:  python research/intraday_lab.py        (~40-60 min)
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

import regime  # noqa: E402
import settings  # noqa: E402
import strategy as S  # noqa: E402
import strategy_lab as L  # noqa: E402

DATA = os.path.join(ROOT, "data", "klines_5m")
OUT = os.path.join(ROOT, "data", "analysis")
BAR_MIN = 15
BARS_PER_DAY = 24 * 60 // BAR_MIN
BARS_PER_YEAR = BARS_PER_DAY * 365
VOL_LB = 7 * BARS_PER_DAY
TAKER, MAKER, THROUGH = 0.001, 0.0005, 0.0001
TRAIN_D, TEST_D, WARM_D = 90, 30, 30

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)


# ---------------------------------------------------------------------------
def load(bar_min: int = BAR_MIN):
    f = {k: {} for k in ("open", "high", "low", "close", "quote_volume")}
    for name in sorted(os.listdir(DATA)):
        df = pd.read_pickle(os.path.join(DATA, name))
        r = df.resample(f"{bar_min}min")
        f["open"][name[:-4]] = r["open"].first()
        f["high"][name[:-4]] = r["high"].max()
        f["low"][name[:-4]] = r["low"].min()
        f["close"][name[:-4]] = r["close"].last()
        f["quote_volume"][name[:-4]] = r["quote_volume"].sum()
    out = {k: pd.DataFrame(v).ffill() for k, v in f.items()}
    last_full = out["close"].index[-2]          # drop the possibly unfinished last bar
    return {k: v.loc[:last_full] for k, v in out.items()}


def half_spread_vec(coins) -> np.ndarray:
    sp = pd.read_csv(os.path.join(OUT, "roostoo_spreads.csv"), index_col=0)["spread_bps"]
    return np.array([max(float(sp.get(c, 10.0)) / 2 / 1e4, 0.0002) for c in coins])


class Exec:
    """Per-bar portfolio returns for a signed weight matrix under an execution model."""

    def __init__(self, d, mode: str):
        self.o, self.h, self.l, self.c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.mode = mode
        self.hs = half_spread_vec(d["close"].columns)
        c = self.c
        self.r = np.zeros_like(c)
        self.r[1:] = c[1:] / c[:-1] - 1
        nxt_o = np.vstack([self.o[1:], c[-1:]])
        nxt_c = np.vstack([c[1:], c[-1:]])
        nxt_l = np.vstack([self.l[1:], c[-1:]])
        nxt_h = np.vstack([self.h[1:], c[-1:]])
        # value at next close of $1 traded during the next bar, per side, net of costs
        if mode == "taker":
            self.buy = nxt_c / (nxt_o * (1 + self.hs)) - 1 - TAKER
            self.sell = 1 - nxt_c / (nxt_o * (1 - self.hs)) - TAKER
            self.fee_buy = np.full_like(c, TAKER) + self.hs
            self.fee_sell = self.fee_buy
        else:
            fb = nxt_l < c * (1 - THROUGH)                      # passive buy filled at c
            fs = nxt_h > c * (1 + THROUGH)                      # passive sell filled at c
            self.buy = np.where(fb, nxt_c / c - 1 - MAKER, -(TAKER + self.hs))
            self.sell = np.where(fs, 1 - nxt_c / c - MAKER, -(TAKER + self.hs))
            self.fee_buy = np.where(fb, MAKER, TAKER + self.hs)
            self.fee_sell = np.where(fs, MAKER, TAKER + self.hs)
            self.fill_rate = (float(fb.mean()), float(fs.mean()))

    def returns(self, w: np.ndarray):
        """w[t] = target decided at close t. Returns (per-bar return, turnover, fees)."""
        prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        dw = w - prev                                            # traded during bar t+1
        hold = prev                                              # carried through bar t+1
        nxt_r = np.vstack([self.r[1:], np.zeros((1, w.shape[1]))])
        pnl = (hold * nxt_r).sum(axis=1) + (np.maximum(dw, 0) * self.buy).sum(axis=1) \
            + (np.maximum(-dw, 0) * self.sell).sum(axis=1)
        fees = (np.maximum(dw, 0) * self.fee_buy).sum(axis=1) + (np.maximum(-dw, 0) * self.fee_sell).sum(axis=1)
        out = np.zeros(len(w))
        out[1:] = pnl[:-1]                                       # realized at close t+1
        f = np.zeros(len(w))
        f[1:] = fees[:-1]
        return out, np.abs(dw).sum(axis=1), f


def score(r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    if len(r) < 100 or r.std() == 0:
        return {"composite": -10.0, "ret": 0.0, "mdd": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}
    eq = np.cumprod(1 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    total = eq[-1] - 1
    yrs = len(r) / BARS_PER_YEAR
    cagr = (1 + total) ** (1 / yrs) - 1 if total > -1 else -1
    sh = r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    so = r.mean() / dd * np.sqrt(BARS_PER_YEAR) if dd > 0 else 0
    ca = cagr / abs(mdd) if mdd < 0 else 0
    return {"composite": 0.4 * so + 0.3 * sh + 0.3 * ca, "ret": total, "mdd": mdd, "sharpe": sh,
            "sortino": so, "calmar": ca}


def kill_switch(r: np.ndarray, dd_limit: float = settings.MAX_DRAWDOWN, cool: int = BARS_PER_DAY) -> np.ndarray:
    out = r.copy()
    eq, peak, halt = 1.0, 1.0, 0
    kills = 0
    for i in range(len(out)):
        if halt > 0:
            out[i] = 0.0
            halt -= 1
            if halt == 0:
                peak = eq
            continue
        eq *= 1 + out[i]
        peak = max(peak, eq)
        if eq <= peak * (1 - dd_limit):
            halt, kills = cool, kills + 1
    return out, kills


# ---------------------------------------------------------------------------
# Sizing and signals on 15m bars
# ---------------------------------------------------------------------------
def size(sel_long: np.ndarray, sel_short: np.ndarray, vol: np.ndarray) -> np.ndarray:
    def side(sel):
        n = sel.sum(axis=1, keepdims=True).astype(float)
        n[n == 0] = np.nan
        w = np.where(sel, S.TARGET_VOL / (n * vol), 0.0)
        w = np.nan_to_num(np.minimum(w, S.MAX_PER_COIN))
        g = w.sum(axis=1, keepdims=True)
        return w * np.where(g > S.MAX_GROSS, S.MAX_GROSS / np.where(g > 0, g, 1), 1.0)
    w = side(sel_long) - side(sel_short)
    g = np.abs(w).sum(axis=1, keepdims=True)
    return w * np.where(g > S.MAX_GROSS, S.MAX_GROSS / np.where(g > 0, g, 1), 1.0)


def hawkes15(d, vol, p):
    kappa, nlb, slb, p5, p95 = p
    c = d["close"].to_numpy()
    v = S.hawkes_volatility(d["high"].to_numpy(), d["low"].to_numpy(), c, kappa, int(nlb))
    sig = S.hawkes_signal(v, c, int(slb), p5, p95)
    return size(sig > 0, sig < 0, vol)


def breakout(d, vol, p, gate=None):
    n = int(p[0])
    c, h, l = d["close"], d["high"], d["low"]
    up = (c > h.rolling(n).max().shift(1)).to_numpy()
    dn = (c < l.rolling(n).min().shift(1)).to_numpy()
    ex_l = (c < l.rolling(max(n // 2, 2)).min().shift(1)).to_numpy()
    ex_s = (c > h.rolling(max(n // 2, 2)).max().shift(1)).to_numpy()
    pos = np.zeros(c.shape)
    cur = np.zeros(c.shape[1])
    for t in range(len(c)):
        cur = np.where(up[t], 1.0, np.where(dn[t], -1.0, cur))
        cur = np.where((cur > 0) & ex_l[t] & ~up[t], 0.0, cur)
        cur = np.where((cur < 0) & ex_s[t] & ~dn[t], 0.0, cur)
        pos[t] = cur
    if gate is not None:                      # longs only when risk-on, shorts only when risk-off
        pos = np.where(pos > 0, pos * gate[0][:, None], pos * gate[1][:, None])
    return size(pos > 0, pos < 0, vol)


def reversal(d, vol, p):
    n, k, exit_k = int(p[0]), p[1], p[2]
    c = d["close"]
    ret = np.log(c).diff(n)
    z = (ret / (np.log(c).diff().rolling(VOL_LB).std() * np.sqrt(n))).to_numpy()
    pos = np.zeros(z.shape)
    cur = np.zeros(z.shape[1])
    for t in range(len(z)):
        zt = np.nan_to_num(z[t])
        cur = np.where(zt < -k, 1.0, np.where(zt > k, -1.0, cur))
        cur = np.where((cur > 0) & (zt > -exit_k), 0.0, cur)
        cur = np.where((cur < 0) & (zt < exit_k), 0.0, cur)
        pos[t] = cur
    return size(pos > 0, pos < 0, vol)


# ---------------------------------------------------------------------------
def folds(idx):
    out, t0 = [], idx[0] + pd.Timedelta(days=WARM_D)
    while True:
        b = t0 + pd.Timedelta(days=TRAIN_D)
        c = b + pd.Timedelta(days=TEST_D)
        if b >= idx[-1]:
            break
        out.append((t0, b, min(c, idx[-1] + pd.Timedelta(minutes=BAR_MIN))))
        t0 += pd.Timedelta(days=TEST_D)
    return out


def evaluate_pool(name, fn, configs, execs) -> dict:
    """Run every config once over the full history under each execution model."""
    out = {m: [] for m in execs}
    t = time.time()
    for i, p in enumerate(configs):
        w = fn(p)
        for m, ex in execs.items():
            out[m].append(ex.returns(w))
        if (i + 1) % 30 == 0:
            print(f"  [{name}] {i + 1}/{len(configs)} configs ({time.time() - t:.0f}s)", flush=True)
    return out


def walk_forward(name, configs, runs, idx):
    """Each fold: pick the config with the best training composite, trade it on the next window."""
    oos, turn, fee, log = np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx)), []
    for k, (a, b, c) in enumerate(folds(idx)):
        ia, ib, ic = idx.searchsorted(a), idx.searchsorted(b), idx.searchsorted(c)
        sc = [score(r[ia:ib])["composite"] if tn[ia:ib].sum() >= 5 else -10.0 for r, tn, _ in runs]
        j = int(np.argmax(sc))
        r, tn, fe = runs[j]
        oos[ib:ic], turn[ib:ic], fee[ib:ic] = r[ib:ic], tn[ib:ic], fe[ib:ic]
        s = score(r[ib:ic])
        log.append({"fold": k, "test": f"{b:%Y-%m-%d}..{c:%Y-%m-%d}", "params": list(configs[j]),
                    "train_comp": round(sc[j], 2), "test_comp": round(s["composite"], 2),
                    "test_ret_%": round(s["ret"] * 100, 2)})
    return oos, turn, fee, log


def report(r, turn, fee, idx, start) -> dict:
    i0 = idx.searchsorted(start)
    r, kills = kill_switch(r[i0:])
    s = score(r)
    eq = pd.Series(np.cumprod(1 + r), index=idx[i0:])
    daily = eq.resample("1D").last()
    r10 = (daily.shift(-10) / daily - 1).dropna() * 100
    days = max((idx[-1] - start).days, 1)
    return {"ret_%": s["ret"] * 100, "max_dd_%": s["mdd"] * 100, "sharpe": s["sharpe"], "sortino": s["sortino"],
            "calmar": s["calmar"], "composite": s["composite"], "mean_10d_%": r10.mean(), "median_10d_%": r10.median(),
            "p10_10d_%": r10.quantile(0.1), "p90_10d_%": r10.quantile(0.9), "worst_10d_%": r10.min(),
            "best_10d_%": r10.max(), "P(10d>5%)": (r10 > 5).mean() * 100, "P(10d<-5%)": (r10 < -5).mean() * 100,
            "turnover/day": turn[i0:].sum() / days, "fees_%": fee[i0:].sum() * 100, "kills": kills}


def main() -> None:
    t0 = time.time()
    d = load()
    c = d["close"]
    idx = c.index
    coins = list(c.columns)
    print(f"{len(coins)} coins, {len(idx)} {BAR_MIN}m bars {idx[0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d %H:%M}", flush=True)
    vol = (np.log(c).diff().rolling(VOL_LB, min_periods=VOL_LB // 2).std() * np.sqrt(BARS_PER_YEAR)).to_numpy()
    fl = folds(idx)
    oos_start = fl[0][1]
    print(f"{len(fl)} folds; OOS {oos_start:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}", flush=True)

    # regime gate (same HMM ensemble + BTC SMA as v3), mapped to 15m bars
    hourly = c["BTC"].resample("1h").last()
    daily_btc = pd.read_csv(os.path.join(ROOT, "data", "klines_1h", "BTC.csv"), index_col=0, parse_dates=True)["close"]
    prob = regime.hourly(regime.bull_probability_ensemble(regime.daily_close_from_hourly(daily_btc)), idx)
    sma = (hourly > hourly.rolling(settings.V3_REGIME_SMA).mean()).reindex(idx, method="ffill").fillna(False)
    on = ((prob > 0.5).fillna(False) & sma).to_numpy()
    off = ((prob < 0.5).fillna(False) & ~sma).to_numpy()

    fams = {
        "hawkes15": (lambda p: hawkes15(d, vol, p), [0.01, 24, 96, 0.01, 0.70], [1.0, 672, 2880, 0.30, 0.99], [1, 2]),
        "breakout": (lambda p: breakout(d, vol, p), [8], [960], [0]),
        "breakout_gated": (lambda p: breakout(d, vol, p, (on, off)), [8], [960], [0]),
        "reversal": (lambda p: reversal(d, vol, p), [2, 1.0, -0.5], [96, 4.0, 1.5], [0]),
    }
    rows, logs = {}, {}
    execs = {"taker": Exec(d, "taker"), "maker": Exec(d, "maker")}
    print(f"maker fill rate within one {BAR_MIN}m bar: buys {execs['maker'].fill_rate[0]:.0%}, "
          f"sells {execs['maker'].fill_rate[1]:.0%}", flush=True)
    for name, (fn, lo, hi, ints) in fams.items():
        if len(lo) == 1:
            configs = [(float(v),) for v in sorted(set(np.round(np.geomspace(lo[0], hi[0], 24))))]
        else:
            configs = L.random_configs(lo, hi, ints, 120, seed=21)
        print(f"--- {name}: {len(configs)} configs", flush=True)
        pool = evaluate_pool(name, fn, configs, execs)
        for mode in execs:
            r, tn, fe, lg = walk_forward(name, configs, pool[mode], idx)
            rows[(name, mode)] = report(r, tn, fe, idx, oos_start)
            logs[f"{name}|{mode}"] = lg
            print(f"  => {name} [{mode}] " + json.dumps({k: round(float(v), 2) for k, v in rows[(name, mode)].items()}),
                  flush=True)
            print("     folds: " + " | ".join(f"{x['test_ret_%']:+.1f}%" for x in lg), flush=True)

    df = pd.DataFrame(rows).T
    df.to_csv(os.path.join(OUT, "intraday_lab.csv"))
    with open(os.path.join(OUT, "intraday_lab_folds.json"), "w") as f:
        json.dump(logs, f, indent=1, default=str)
    print(f"\n=== INTRADAY LAB: OOS {oos_start:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d} "
          f"(kill switch {settings.MAX_DRAWDOWN:.0%}, caps vol {S.TARGET_VOL:.0%} / gross {S.MAX_GROSS:.0%}) ===")
    print(df.astype(float).round(2).to_string())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
