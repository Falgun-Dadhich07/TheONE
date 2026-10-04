"""Compare every candidate strategy in one table, losers included.

Protocol (fixed BEFORE looking at holdout results):
  1. In-sample = data before HOLDOUT_START. All candidates run with default
     parameters; a sensitivity grid is shown for the draft strategy.
  2. Pick the winner by in-sample full-period composite score among strategies
     whose worst 14-day drawdown is better than -15%. Benchmarks can't win.
  3. Only then run the holdout (last ~3 months) and report it for all.

Run:  python backtest/run_compare.py
"""
import itertools
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import strategy  # noqa: E402
from backtest import engine, metrics  # noqa: E402

UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
DATA_DIR = os.path.join("data", "klines_1h")
HOLDOUT_START = pd.Timestamp("2026-07-05", tz="UTC")
WARMUP = pd.Timedelta(days=30)        # indicators need <= 720h of history
MAX_WORST_14D_DD = -15.0
BENCHMARKS = {"buy_hold_btc", "equal_weight_half"}

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def load():
    opens, closes = {}, {}
    for c in UNIVERSE:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{c}.csv"), index_col=0, parse_dates=True)
        opens[c], closes[c] = df["open"], df["close"]
    o, c = pd.DataFrame(opens), pd.DataFrame(closes)
    full = pd.date_range(c.index[0], c.index[-1], freq="h", tz="UTC")
    return o.reindex(full).ffill(), c.reindex(full).ffill()


def evaluate(opens, closes, weights, start, end) -> dict:
    sl = slice(start, end)
    res = engine.run(opens.loc[sl], closes.loc[sl], weights.loc[sl])
    row = metrics.summary(res.equity)
    row.update(metrics.window_summary(metrics.rolling_windows(res.equity)))
    days = (res.equity.index[-1] - res.equity.index[0]).days or 1
    gross = res.weights.sum(axis=1)
    row.update({
        "trades": len(res.trades),
        "trades_per_day": len(res.trades) / days,
        "fees_%_of_start": res.fees / res.equity.iloc[0] * 100,
        "avg_exposure_%": gross.mean() * 100,
        "pct_time_invested": (gross > 0.01).mean() * 100,
    })
    return row


def fmt(df: pd.DataFrame) -> str:
    return df.round(2).to_string()


def main() -> None:
    opens, closes = load()
    is_start = closes.index[0] + WARMUP
    is_end = HOLDOUT_START - pd.Timedelta(hours=1)
    print(f"Universe {UNIVERSE}")
    print(f"In-sample {is_start:%Y-%m-%d} -> {is_end:%Y-%m-%d} | holdout {HOLDOUT_START:%Y-%m-%d} -> "
          f"{closes.index[-1]:%Y-%m-%d %H:%M} | fee {engine.TAKER_FEE:.2%} + half-spread "
          f"{engine.HALF_SPREAD:.2%} per side\n")

    # Weights are computed on the FULL history (functions are causal), then sliced.
    weights = {name: fn(closes) for name, fn in strategy.STRATEGIES.items()}

    ins = pd.DataFrame({n: evaluate(opens, closes, w, is_start, is_end) for n, w in weights.items()}).T
    print("=== IN-SAMPLE: all candidates, default parameters ===")
    print(fmt(ins), "\n")

    print("=== IN-SAMPLE sensitivity: xs_momentum_regime (lookback x regime_sma x top_n) ===")
    grid = []
    for lb, rs, n in itertools.product((168, 336, 504), (240, 480, 720), (1, 2, 3)):
        w = strategy.xs_momentum_regime(closes, lookback=lb, regime_sma=rs, top_n=n)
        m = evaluate(opens, closes, w, is_start, is_end)
        grid.append({"lookback": lb, "regime_sma": rs, "top_n": n, **{k: m[k] for k in (
            "total_ret_%", "max_dd_%", "sharpe", "composite", "w14_median_ret_%",
            "w14_pct_losing", "w14_worst_dd_%", "trades_per_day")}})
    grid = pd.DataFrame(grid)
    print(grid.round(2).to_string(index=False))
    print(f"Grid: composite > 0 in {(grid['composite'] > 0).sum()}/{len(grid)} combos; "
          f"median composite {grid['composite'].median():.2f}\n")

    eligible = ins.drop(index=list(BENCHMARKS))
    eligible = eligible[eligible["w14_worst_dd_%"] > MAX_WORST_14D_DD]
    if eligible.empty:
        print("No candidate meets the drawdown rule; picking by composite among all non-benchmarks.")
        eligible = ins.drop(index=list(BENCHMARKS))
    winner = eligible["composite"].idxmax()
    print(f">>> Selection locked on in-sample data: {winner}\n")

    hold = pd.DataFrame({n: evaluate(opens, closes, w, HOLDOUT_START, closes.index[-1])
                         for n, w in weights.items()}).T
    print("=== HOLDOUT (untouched until now): all candidates ===")
    print(fmt(hold))

    os.makedirs(os.path.join("data", "analysis"), exist_ok=True)
    ins.to_csv(os.path.join("data", "analysis", "compare_insample.csv"))
    hold.to_csv(os.path.join("data", "analysis", "compare_holdout.csv"))
    grid.to_csv(os.path.join("data", "analysis", "grid_xs_momentum.csv"), index=False)


if __name__ == "__main__":
    main()
