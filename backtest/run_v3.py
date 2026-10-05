"""Backtest the production v3 code (strategy.v3_* + regime.py) exactly as the bot runs it.

1. Validation: Hawkes parameters spliced per walk-forward fold (each fold's
   params only in its own out-of-sample window), Jul 2025 - Oct 2026. This
   should reproduce research/hybrid_lab.py's 50/50 blend.
2. Deployed config: the live Hawkes params from settings.py over the last
   30 days. This is in-sample for those params, shown for information only.

Run:  python backtest/run_v3.py
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import regime  # noqa: E402
import settings  # noqa: E402
import strategy as S  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from risk import RiskManager  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "klines_1h")
OUT_DIR = os.path.join(ROOT, "data", "analysis")
EXCLUDE = {"TON", "OMNI"}
EVAL_START = pd.Timestamp("2025-07-13", tz="UTC")


def load():
    f = {k: {} for k in ("open", "high", "low", "close", "quote_volume")}
    for name in sorted(os.listdir(DATA_DIR)):
        coin = name[:-4]
        if coin in EXCLUDE:
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, name), index_col=0, parse_dates=True)
        for k in f:
            f[k][coin] = df[k]
    frames = {k: pd.DataFrame(v) for k, v in f.items()}
    idx = pd.date_range(frames["close"].index[0], frames["close"].index[-1], freq="h", tz="UTC")
    return {k: v.reindex(idx) for k, v in frames.items()}


def half_spreads(coins):
    sp = pd.read_csv(os.path.join(OUT_DIR, "roostoo_spreads.csv"), index_col=0)["spread_bps"]
    return {c: max(float(sp.get(c, 10.0)) / 2 / 1e4, engine.HALF_SPREAD) for c in coins}


def evaluate(d, w, start, end):
    c, o = d["close"].ffill(), d["open"].ffill()
    used = w.columns[(w.loc[start:end].abs() > 0).any()]
    sl = slice(start, end)
    res = engine.run(o.loc[sl, used], c.loc[sl, used], w.loc[sl, used], half_spread=half_spreads(used),
                     risk=RiskManager(), daily_vol=(S.realized_vol(c) / np.sqrt(365))[used])
    m = metrics.summary(res.equity)
    win = metrics.rolling_windows(res.equity)
    out = {"ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "sharpe": m["sharpe"],
           "sortino": m["sortino"], "calmar": m["calmar"], "composite": m["composite"],
           "fees_%": res.fees / res.equity.iloc[0] * 100, "trades/day": len(res.trades) / max((end - start).days, 1),
           "kills": sum(1 for e in res.extra["risk_events"] if e[1] == "KILL_SWITCH")}
    if len(win):
        out.update({"w14_profit_%": (win["ret_%"] > 0).mean() * 100, "w14_worst_dd_%": win["max_dd_%"].min()})
    return out


def main() -> None:
    d = load()
    c = d["close"]
    dqv = d["quote_volume"].resample("1D").sum(min_count=1)
    prob = regime.hourly(regime.bull_probability(regime.daily_close_from_hourly(c["BTC"].ffill())), c.index)
    available = c.notna() & d["high"].notna() & d["low"].notna()
    px = c.ffill()
    eligible = S.liquid_universe(available, dqv, settings.V3_TOP_N)
    sma = S.btc_regime(px, settings.V3_REGIME_SMA)
    on = (prob > 0.5).fillna(False) & sma
    off = (prob < 0.5).fillna(False) & ~sma
    mom = S.momentum_long_short(px, eligible, on, off)

    folds = json.load(open(os.path.join(OUT_DIR, "short_lab_folds.json")))["folds"]["hawkes_ls_wfo"]
    hawk = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for f in folds:
        kappa, nlb, slb, p5, p95 = f["params"]
        a, b = [pd.Timestamp(x, tz="UTC") for x in f["test"].split("..")]
        w = S.hawkes_long_short(d["high"].ffill(), d["low"].ffill(), px, eligible, kappa, int(nlb), int(slb), p5, p95)
        m = (c.index >= a) & (c.index < b)
        hawk.loc[m] = w.loc[m]
    blend = S.v3_blend(mom, hawk, settings.V3_HAWKES_SHARE)
    end = c.index[-1]
    rows = {"v3 validation (walk-forward Hawkes params)": evaluate(d, blend, EVAL_START, end),
            "  momentum sleeve": evaluate(d, mom, EVAL_START, end),
            "  hawkes sleeve": evaluate(d, hawk, EVAL_START, end),
            "v3 long-only fallback (shorts refused)": evaluate(d, blend.clip(lower=0), EVAL_START, end)}
    live = S.v3_weights(c, d["high"], d["low"], dqv, prob, settings.HAWKES_PARAMS, settings.V3_HAWKES_SHARE,
                        settings.V3_TOP_N, settings.V3_REGIME_SMA)
    month = (end - pd.Timedelta(days=30)).floor("D")
    rows["deployed params, last 30d (in-sample!)"] = evaluate(d, live, month, end)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).T.astype(float).round(2).to_string())
    last = live.iloc[-1]
    print("\nTarget right now:", {k: round(v, 3) for k, v in last.items() if abs(v) > 1e-4})
    print("Bull prob now:", round(float(prob.iloc[-1]), 3), "| BTC > SMA:", bool(sma.iloc[-1]))


if __name__ == "__main__":
    main()
