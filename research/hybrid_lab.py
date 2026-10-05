"""Hybrid lab: per-coin choice between the HMM+BTC momentum strategy and the
Hawkes volatility strategy, on the causal top-20 liquidity universe.

Sleeves (both long/short, Roostoo 1x shorts):
  MOM    : v2 long/short - long if the coin is a top-2 14-day performer while
           risk-on (HMM bull AND BTC > SMA), short if bottom-2 while risk-off.
  HAWKES : Hawkes volatility-breakout signal per coin (+1 long, -1 short) with the
           walk-forward PSO parameters from short_lab (each fold's params are only
           used in that fold's out-of-sample window; tuned on majors only).

Hybrid (per coin, decided daily at 00:00 UTC, causal):
  For each coin, compute each sleeve's trailing N-day Sharpe of its signal on
  that coin, net of that coin's costs. Follow the sleeve with the higher Sharpe
  if it is > 0, else no position. Inverse-vol sizing, 25% per coin, 50% gross.

Variants: N = 30 / 60 / 90 days, a 50/50 capital blend of the two sleeves,
each sleeve alone. Evaluation window starts 2025-07-13 so that the first
lookback window is already out-of-sample for the Hawkes parameters.

Run:  python research/hybrid_lab.py
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
import short_lab as X  # noqa: E402
import strategy as S  # noqa: E402
import strategy_lab as L  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "analysis")
EVAL_START = pd.Timestamp("2025-07-13", tz="UTC")
COST_FEE = 0.001

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)


def load_high_low(index, columns):
    hi, lo = {}, {}
    for coin in columns:
        df = pd.read_csv(os.path.join(P.DATA_DIR, f"{coin}.csv"), index_col=0, parse_dates=True)
        hi[coin], lo[coin] = df["high"], df["low"]
    return pd.DataFrame(hi).reindex(index).ffill(), pd.DataFrame(lo).reindex(index).ffill()


def hawkes_signals_walkforward(high, low, close, folds_log) -> pd.DataFrame:
    """Splice per-fold Hawkes signals: fold k's params drive fold k's test window only."""
    out = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    h, l, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    for f in folds_log:
        kappa, nlb, slb, p5, p95 = f["params"]
        a, b = [pd.Timestamp(x, tz="UTC") for x in f["test"].split("..")]
        v = L.hawkes_volatility(h, l, c, kappa, int(nlb))
        sig = L.hawkes_signal(v, c, int(slb), p5, p95)
        m = (close.index >= a) & (close.index < b)
        out.loc[m] = sig[m]
    return out


def trailing_sharpe(sig: pd.DataFrame, closes: pd.DataFrame, costs: pd.Series, days: int) -> pd.DataFrame:
    r = closes.pct_change().fillna(0.0)
    held = sig.shift(1).fillna(0.0)
    turn = held.diff().abs().fillna(0.0)
    pnl = held * r - turn * costs
    n = days * 24
    mu = pnl.rolling(n, min_periods=n // 2).mean()
    sd = pnl.rolling(n, min_periods=n // 2).std()
    return (mu / sd).where(sd > 0) * np.sqrt(S.HOURS_PER_YEAR)


def hybrid_signal(mom_sig, hawk_sig, closes, costs, days) -> pd.DataFrame:
    sh_m = trailing_sharpe(mom_sig, closes, costs, days)
    sh_h = trailing_sharpe(hawk_sig, closes, costs, days)
    pick_m = (sh_m > sh_h.fillna(-np.inf)) & (sh_m > 0)
    pick_h = (sh_h >= sh_m.fillna(-np.inf)) & (sh_h > 0) & ~pick_m
    # decide once a day, hold the choice; the chosen sleeve's live signal is used
    choice = pd.DataFrame(np.where(pick_m, 1, np.where(pick_h, 2, 0)), index=closes.index, columns=closes.columns)
    choice = S.hold_until_rebalance(choice.astype(float))
    return mom_sig.where(choice == 1, 0.0) + hawk_sig.where(choice == 2, 0.0), choice


def to_weights(sig: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    return X.size_ls(sig > 0, sig < 0, closes)


def main() -> None:
    t0 = time.time()
    raw, closes, opens, qv = P.load_all()
    high, low = load_high_low(closes.index, closes.columns)
    hs = P.half_spreads(closes.columns)
    costs = pd.Series({c: COST_FEE + hs[c] for c in closes.columns})
    dvol = S.realized_vol(closes) / np.sqrt(365)

    prob = X.gate_prob(closes)
    sma = S.btc_regime(closes, 480)
    on = (prob > 0.5).fillna(False) & sma
    off = (prob < 0.5).fillna(False) & ~sma
    rank = P.liquidity_rank(qv)
    hist = raw.notna().cumsum()
    e20 = raw.notna() & (hist >= P.MIN_HISTORY_H) & (rank <= 20)

    w_mom = X.momentum_ls(closes, e20, on, off)
    mom_sig = np.sign(w_mom)

    folds_log = json.load(open(os.path.join(OUT_DIR, "short_lab_folds.json")))["folds"]["hawkes_ls_wfo"]
    print("Hawkes signals for all coins, per fold ...", flush=True)
    hawk_all = hawkes_signals_walkforward(high, low, closes, folds_log)
    hawk_sig = hawk_all.where(e20, 0.0)
    w_hawk = to_weights(hawk_sig, closes)

    W = {"MOM sleeve (v2 L/S top20)": w_mom, "HAWKES sleeve (top20)": w_hawk}
    choices = {}
    for d in (30, 60, 90):
        sig, ch = hybrid_signal(mom_sig, hawk_sig, closes, costs, d)
        W[f"HYBRID per-coin pick, {d}d"] = to_weights(sig, closes)
        choices[d] = ch
    blend = 0.5 * w_mom + 0.5 * w_hawk
    W["BLEND 50/50"] = blend

    end = closes.index[-1]
    periods = {"EVAL": (EVAL_START, end), "3M": (pd.Timestamp("2026-07-05", tz="UTC"), end),
               "1M": ((end - pd.Timedelta(days=30)).floor("D"), end)}
    rows = {}
    for name, w in W.items():
        for pn, (a, b) in periods.items():
            for k, v in X.evaluate(opens, closes, w, a, b, hs, dvol).items():
                rows.setdefault(name, {})[f"{pn}_{k}"] = v
        print(f"  {name:<30} EVAL ret {rows[name]['EVAL_ret_%']:6.2f}%  dd {rows[name]['EVAL_max_dd_%']:6.2f}%  "
              f"comp {rows[name]['EVAL_composite']:5.2f}", flush=True)
    keep = ["ret_%", "max_dd_%", "sharpe", "sortino", "calmar", "composite", "w14_profit_%", "w14_worst_dd_%",
            "long_expo_%", "short_expo_%", "trades/day", "fees_%", "kills"]
    cols = [f"EVAL_{k}" for k in keep] + [f"{p}_{k}" for p in ("3M", "1M") for k in ("ret_%", "max_dd_%", "composite")]
    df = pd.DataFrame(rows).T[cols].astype(float).round(2)
    df.to_csv(os.path.join(OUT_DIR, "hybrid_lab.csv"))
    print(f"\n=== HYBRID LAB: EVAL {EVAL_START:%Y-%m-%d} -> {end:%Y-%m-%d} (real engine, kill switch, spreads) ===")
    print(df.to_string())

    # which coins went to which sleeve (60d variant), share of eligible coin-days in EVAL
    ch = choices[60].loc[EVAL_START:][e20.loc[EVAL_START:]]
    daily = ch[ch.index.hour == 0]
    share = pd.DataFrame({"MOM_%": (daily == 1).mean() * 100, "HAWKES_%": (daily == 2).mean() * 100,
                          "none_%": (daily == 0).mean() * 100, "eligible_days": daily.notna().sum()})
    share = share[share["eligible_days"] > 30].sort_values("HAWKES_%", ascending=False).round(0)
    print("\nPer-coin sleeve choice (60d rule, share of eligible days in EVAL):")
    print(share.head(25).to_string())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
