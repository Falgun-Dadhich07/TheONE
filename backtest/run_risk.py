"""Does the risk overlay help? Same strategy (xs_momentum_regime, default
parameters), with and without each risk rule, at two vol targets.

Risk settings are the brief's defaults (8% kill switch, 24h cooldown, 3x daily
vol trailing stop), not tuned. The holdout was already seen once in
run_compare.py, so treat the holdout columns as a sanity check, not proof.

Also replays the worst BTC crash in the data to show what each rule does.

Run:  python backtest/run_risk.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import strategy  # noqa: E402
from backtest import engine, metrics  # noqa: E402
from backtest.run_compare import HOLDOUT_START, WARMUP, load  # noqa: E402
from risk import RiskManager  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

VARIANTS = {
    "no overlay":         dict(max_drawdown=None, stop_vol_mult=None),
    "kill switch only":   dict(max_drawdown=0.08, stop_vol_mult=None),
    "trailing stop only": dict(max_drawdown=None, stop_vol_mult=3.0),
    "kill + stop":        dict(max_drawdown=0.08, stop_vol_mult=3.0),
}


def run_one(opens, closes, weights, dvol, start, end, rm_kwargs):
    sl = slice(start, end)
    rm = RiskManager(**rm_kwargs) if any(v is not None for v in rm_kwargs.values()) else None
    res = engine.run(opens.loc[sl], closes.loc[sl], weights.loc[sl], risk=rm, daily_vol=dvol)
    m = metrics.summary(res.equity)
    m.update(metrics.window_summary(metrics.rolling_windows(res.equity)))
    ev = pd.Series([e[1] for e in res.extra["risk_events"]], dtype=object)
    days = max((res.equity.index[-1] - res.equity.index[0]).days, 1)
    return res, {
        "ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"], "sharpe": m["sharpe"],
        "sortino": m["sortino"], "calmar": m["calmar"], "composite": m["composite"],
        "w14_losing_%": m["w14_pct_losing"], "w14_worst_dd_%": m["w14_worst_dd_%"],
        "trades/day": len(res.trades) / days, "fees_%": res.fees / res.equity.iloc[0] * 100,
        "kills": int((ev == "KILL_SWITCH").sum()), "stops": int((ev == "STOP").sum()),
    }


def main() -> None:
    opens, closes = load()
    dvol = strategy.realized_vol(closes) / np.sqrt(365)
    periods = {
        "IN-SAMPLE": (closes.index[0] + WARMUP, HOLDOUT_START - pd.Timedelta(hours=1)),
        "HOLDOUT": (HOLDOUT_START, closes.index[-1]),
    }
    base_vol = strategy.TARGET_VOL
    try:
        for tv in (0.25, 0.40):
            strategy.TARGET_VOL = tv
            weights = strategy.xs_momentum_regime(closes)
            for pname, (s, e) in periods.items():
                rows = {v: run_one(opens, closes, weights, dvol, s, e, kw)[1] for v, kw in VARIANTS.items()}
                print(f"=== {pname} | target vol {tv:.0%} ===")
                print(pd.DataFrame(rows).T.round(2).to_string(), "\n")

        # Crash replay: worst 72h BTC drop in the data, at the default vol target.
        strategy.TARGET_VOL = base_vol
        weights = strategy.xs_momentum_regime(closes)
        drop = closes["BTC"] / closes["BTC"].shift(72) - 1
        crash_end = drop.idxmin()
        s, e = crash_end - pd.Timedelta(days=10), crash_end + pd.Timedelta(days=10)
        print(f"=== CRASH REPLAY: BTC {drop.min():.1%} in 72h ending {crash_end:%Y-%m-%d %H:%M} "
              f"(window {s:%m-%d} -> {e:%m-%d}) ===")
        for v, kw in VARIANTS.items():
            res, row = run_one(opens, closes, weights, dvol, s, e, kw)
            evs = [f"{t:%m-%d %H}h {k}{' ' + c if c else ''}" for t, k, c, _ in res.extra["risk_events"]]
            print(f"{v:<20} ret {row['ret_%']:6.2f}%  max_dd {row['max_dd_%']:6.2f}%  "
                  f"events: {', '.join(evs) or 'none'}")
    finally:
        strategy.TARGET_VOL = base_vol


if __name__ == "__main__":
    main()
