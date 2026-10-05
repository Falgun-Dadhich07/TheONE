"""Turn the factor-lab findings into strategies and test them against v3.

Scores (per coin, daily at 00:00 UTC, top-20 causal liquidity universe):
  high14   : nearness to the 14-day high  (close / 336h high - 1)
  composite: walk-forward model. Every 30 days, each candidate signal's daily
             24h rank IC is measured on all PAST data (expanding, >= 180 days,
             last 3 days dropped so no future return leaks in); signals with
             |t| > 2 get weight = their mean IC, others 0; the score is the
             weighted sum of cross-sectional rank z-scores.

Sleeves built from a score (regime gate = v3's: HMM ensemble AND BTC vs SMA):
  gated K   : long top-K while risk-on, short bottom-K while risk-off
  neutral K : always long top-K and short bottom-K (market-neutral-ish)

Variants: each sleeve alone, and v3 with its 14-day-momentum sleeve replaced.
Real engine, kill switch, per-coin spreads, deployed risk settings.

Run:  python research/factor_strategy.py
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

import factor_lab as F  # noqa: E402
import regime  # noqa: E402
import run_v3 as R  # noqa: E402
import strategy as S  # noqa: E402
from trend_sleeve import evaluate  # noqa: E402

OUT = os.path.join(ROOT, "data", "analysis")
REFIT_DAYS, MIN_DAYS = 30, 180
EXCLUDE = ("oi_", "top_trader", "global_", "futures_taker")      # only 13 months / 30 coins of history


def zrank(df: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    x = df.where(elig)
    r = x.rank(axis=1, pct=True) - 0.5
    return r.div(r.std(axis=1).replace(0, np.nan), axis=0)


def composite_score(feats, px, elig30, days):
    """Walk-forward IC-weighted composite. Returns (score at daily rows, weights log)."""
    fwd = px.shift(-24) / px - 1
    ics = {k: F.rank_ic(v, fwd, elig30, days) for k, v in feats.items()}
    z = {k: zrank(v.loc[days], elig30.loc[days]) for k, v in feats.items()}
    score = pd.DataFrame(np.nan, index=days, columns=px.columns)
    log = []
    refits = days[MIN_DAYS::REFIT_DAYS]
    for j, t in enumerate(refits):
        nxt = refits[j + 1] if j + 1 < len(refits) else days[-1] + pd.Timedelta(days=1)
        w = {}
        for k, ic in ics.items():
            past = ic[ic.index < t - pd.Timedelta(days=3)]
            if len(past) < 120:
                continue
            tstat = past.mean() / past.std() * np.sqrt(len(past))
            if abs(tstat) > 2:
                w[k] = float(past.mean())
        seg = (days >= t) & (days < nxt)
        s = sum(z[k].loc[seg] * wk for k, wk in w.items()) if w else None
        if s is not None:
            score.loc[seg] = s
        log.append({"refit": str(t.date()), "weights": {k: round(v, 3) for k, v in sorted(w.items(), key=lambda kv: -abs(kv[1]))}})
    return score, log


def sleeve(score_daily, px, elig, on, off, k: int, gated: bool) -> pd.DataFrame:
    s = score_daily.reindex(px.index).where(elig)
    top = s.rank(axis=1, ascending=False) <= k
    bot = s.rank(axis=1, ascending=True) <= k
    if gated:
        top &= on.to_numpy()[:, None]
        bot &= off.to_numpy()[:, None]
    return S.hold_until_rebalance(S.size_long_short(top, bot, px))


def main() -> None:
    t0 = time.time()
    d, px, taker, trades, qv, fund, met = F.load_panel()
    idx = px.index
    dqv = d["quote_volume"].resample("1D").sum(min_count=1)
    avail = d["close"].notna() & d["high"].notna() & d["low"].notna()
    el20 = S.liquid_universe(avail, dqv, 20)
    el30 = S.liquid_universe(d["close"].notna(), dqv, 30)
    feats = {k: v for k, v in F.build_features(d, px, taker, trades, qv, fund, {}).items()
             if not k.startswith(EXCLUDE)}
    sma = S.btc_regime(px, 480)
    prob = regime.hourly(regime.bull_probability_ensemble(regime.daily_close_from_hourly(px["BTC"])), idx)
    on = (prob > 0.5).fillna(False) & sma
    off = (prob < 0.5).fillna(False) & ~sma
    days = idx[(idx.hour == 0) & (idx >= idx[0] + pd.Timedelta(days=45))]

    print("walk-forward composite ...", flush=True)
    comp, log = composite_score(feats, px, el30, days)
    for e in log[::3] + log[-1:]:
        print("  ", e["refit"], e["weights"])
    high14 = feats["dist_from_14d_high"].loc[days]

    # v3 pieces exactly as validated
    high, low = d["high"].ffill(), d["low"].ffill()
    mom = S.momentum_long_short(px, el20, on, off)
    hawk = pd.DataFrame(0.0, index=idx, columns=px.columns)
    for f in json.load(open(os.path.join(OUT, "short_lab_folds.json")))["folds"]["hawkes_ls_wfo"]:
        k, n1, n2, p5, p95 = f["params"]
        fa, fb = [pd.Timestamp(x, tz="UTC") for x in f["test"].split("..")]
        w = S.hawkes_long_short(high, low, px, el20, k, int(n1), int(n2), p5, p95)
        m = (idx >= fa) & (idx < fb)
        hawk.loc[m] = w.loc[m]

    V = {"v3 deployed (14d momentum + Hawkes)": S.v3_blend(mom, hawk, 0.5),
         "  v3 momentum sleeve alone": mom}
    for name, sc in (("high14", high14), ("composite", comp)):
        for k in (2, 3, 5):
            g = sleeve(sc, px, el20, on, off, k, gated=True)
            V[f"{name} gated K={k} alone"] = g
            V[f"v3 with {name} gated K={k} + Hawkes"] = S.v3_blend(g, hawk, 0.5)
        for k in (3, 5):
            V[f"{name} neutral K={k} alone"] = sleeve(sc, px, el20, on, off, k, gated=False)
    a = max(R.EVAL_START, days[MIN_DAYS])
    b = idx[-1]
    a2 = pd.Timestamp("2025-12-30", tz="UTC")
    rows, rows2 = {}, {}
    for name, w in V.items():
        rows[name] = evaluate(d, w, a, b)
        rows2[name] = evaluate(d, w, a2, b)
        print(f"  {name:<42} ret {rows[name]['ret_%']:6.1f}% dd {rows[name]['max_dd_%']:6.1f}% comp "
              f"{rows[name]['composite']:5.2f} mean10d {rows[name]['mean_10d_%']:5.2f}% | 2026: ret "
              f"{rows2[name]['ret_%']:6.1f}% dd {rows2[name]['max_dd_%']:6.1f}% ({time.time() - t0:.0f}s)", flush=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    cols = ["ret_%", "max_dd_%", "sharpe", "sortino", "calmar", "composite", "half1_%", "half2_%", "mean_10d_%",
            "median_10d_%", "p10_10d_%", "p90_10d_%", "worst_10d_%", "P(10d>5%)", "P(10d<-5%)", "avg_gross_%",
            "trades/day", "fees_%", "kills"]
    df = pd.DataFrame(rows).T.astype(float).round(2)
    df.to_csv(os.path.join(OUT, "factor_strategy.csv"))
    print(f"\n=== {a:%Y-%m-%d} -> {b:%Y-%m-%d} (real engine, kill switch, spreads, deployed risk) ===")
    print(df[cols].to_string())
    print(f"\n=== {a2:%Y-%m-%d} -> {b:%Y-%m-%d} ===")
    print(pd.DataFrame(rows2).T.astype(float).round(2)[["ret_%", "max_dd_%", "composite", "mean_10d_%", "worst_10d_%",
                                                        "P(10d>5%)", "P(10d<-5%)", "fees_%"]].to_string())
    with open(os.path.join(OUT, "factor_weights_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
