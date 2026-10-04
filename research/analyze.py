"""Descriptive analysis of the Roostoo crypto universe (hourly Binance data).

Answers, per coin and for the universe:
  - Is it liquid enough and does it have enough history?
  - How volatile is it, how deep are its drawdowns, how tied to BTC is it?
  - Do returns trend (momentum) or revert at the horizons we can trade
    (we pay ~0.2% per round trip, so only multi-hour horizons matter)?
  - What does a 14-day window (competition length) look like for buy & hold?
  - Does cross-sectional momentum (buy the recent winners) predict anything?

Discipline: predictive statistics use only data BEFORE the holdout start, so
the most recent ~3 months stay untouched for final validation.

Run:  python research/analyze.py
Output: printed tables + data/analysis/*.csv
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data", "klines_1h")
OUT_DIR = os.path.join("data", "analysis")
HOLDOUT_START = pd.Timestamp("2026-07-05", tz="UTC")
HOURS_PER_YEAR = 24 * 365
MIN_HISTORY_DAYS = 300          # coins younger than this are excluded from the core set
ROUND_TRIP_COST = 0.002         # 0.1% taker fee per side + spread, measured on Roostoo

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", message=".*Sorting by default.*")
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def load() -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        coin = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        out[coin] = df
    return out


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation without scipy (not in our dependency list)."""
    return float(a.rank().corr(b.rank()))


def max_drawdown(prices: pd.Series) -> float:
    return float((prices / prices.cummax() - 1).min())


def variance_ratio(r: pd.Series, q: int) -> float:
    """Lo-MacKinlay VR(q): >1 means returns trend, <1 means they revert."""
    r = r.dropna()
    rq = r.rolling(q).sum().dropna()
    return float(rq.var() / (q * r.var()))


def coin_stats(coins: dict) -> pd.DataFrame:
    btc_r = np.log(coins["BTC"]["close"]).diff()
    rows = []
    for coin, df in coins.items():
        close = df["close"]
        r = np.log(close).diff()
        ins = r[r.index < HOLDOUT_START]
        last90 = df[df.index >= df.index[-1] - pd.Timedelta(days=90)]
        last365 = close[close.index >= close.index[-1] - pd.Timedelta(days=365)]
        both = pd.concat([r, btc_r], axis=1, keys=["c", "b"]).dropna()
        beta = both["c"].cov(both["b"]) / both["b"].var() if len(both) > 100 else np.nan
        daily_usd = last90["quote_volume"].resample("1D").sum()
        rows.append({
            "coin": coin,
            "history_days": round((df.index[-1] - df.index[0]).days),
            "last_bar": df.index[-1].strftime("%Y-%m-%d"),
            "med_daily_vol_$M": round(daily_usd.median() / 1e6, 1),
            "ann_vol_%": round(r.std() * np.sqrt(HOURS_PER_YEAR) * 100, 1),
            "vol_last90_%": round(np.log(last90["close"]).diff().std() * np.sqrt(HOURS_PER_YEAR) * 100, 1),
            "ret_1y_%": round((last365.iloc[-1] / last365.iloc[0] - 1) * 100, 1),
            "maxdd_1y_%": round(max_drawdown(last365) * 100, 1),
            "corr_btc": round(both["c"].corr(both["b"]), 2),
            "beta_btc": round(beta, 2),
            "acf1_hourly": round(ins.autocorr(1), 3),
            "VR_24h": round(variance_ratio(ins, 24), 2),
            "VR_7d": round(variance_ratio(ins, 168), 2),
            "med_abs_24h_move_%": round(close.pct_change(24).abs().median() * 100, 2),
        })
    return pd.DataFrame(rows).set_index("coin").sort_values("med_daily_vol_$M", ascending=False)


def window_stats(close: pd.Series, days: int = 14) -> dict:
    """Buy & hold over every 14-day window (stepping 1 day)."""
    h = days * 24
    daily = close.resample("1D").last().dropna()
    rets, dds = [], []
    for i in range(0, len(daily) - days):
        start, end = daily.index[i], daily.index[i + days]
        seg = close[(close.index >= start) & (close.index <= end)]
        if len(seg) < h * 0.9:
            continue
        rets.append(seg.iloc[-1] / seg.iloc[0] - 1)
        dds.append(max_drawdown(seg))
    rets, dds = np.array(rets), np.array(dds)
    return {
        "windows": len(rets),
        "median_ret_%": round(np.median(rets) * 100, 1),
        "p10_ret_%": round(np.percentile(rets, 10) * 100, 1),
        "p90_ret_%": round(np.percentile(rets, 90) * 100, 1),
        "pct_losing": round((rets < 0).mean() * 100, 0),
        "median_maxdd_%": round(np.median(dds) * 100, 1),
        "worst_maxdd_%": round(dds.min() * 100, 1),
    }


def tsmom_ic(coins: dict, lookbacks=(6, 24, 72, 168, 336), horizon: int = 24) -> pd.DataFrame:
    """Time-series momentum: corr(past k-hour return, next `horizon`-hour return).

    Sampled every `horizon` hours (non-overlapping forward windows). In-sample only.
    Positive = trending, negative = mean-reverting at that lookback.
    """
    rows = {}
    for coin, df in coins.items():
        c = df["close"][df.index < HOLDOUT_START]
        fwd = c.shift(-horizon) / c - 1
        row = {}
        for k in lookbacks:
            past = c / c.shift(k) - 1
            s = pd.concat([past, fwd], axis=1, keys=["p", "f"]).dropna().iloc[::horizon]
            row[f"lb_{k}h"] = round(spearman(s["p"], s["f"]), 3) if len(s) > 50 else np.nan
        rows[coin] = row
    return pd.DataFrame(rows).T


def trend_filter_edge(coins: dict, fast: int = 24, slow: int = 96, horizon: int = 24) -> pd.DataFrame:
    """Mean forward return when fast EMA > slow EMA vs when below. In-sample only."""
    rows = []
    for coin, df in coins.items():
        c = df["close"][df.index < HOLDOUT_START]
        up = c.ewm(span=fast, adjust=False).mean() > c.ewm(span=slow, adjust=False).mean()
        fwd = (c.shift(-horizon) / c - 1)
        s = pd.concat([up, fwd], axis=1, keys=["up", "f"]).dropna().iloc[::horizon]
        if len(s) < 50:
            continue
        rows.append({
            "coin": coin,
            "pct_time_up": round(s["up"].mean() * 100, 0),
            "fwd24h_when_up_%": round(s.loc[s["up"], "f"].mean() * 100, 3),
            "fwd24h_when_down_%": round(s.loc[~s["up"], "f"].mean() * 100, 3),
            "vol_when_up_%": round(s.loc[s["up"], "f"].std() * 100, 2),
            "vol_when_down_%": round(s.loc[~s["up"], "f"].std() * 100, 2),
        })
    return pd.DataFrame(rows).set_index("coin")


def xsec_momentum(coins: dict, lookback: int, horizon: int, top_n: int = 3) -> dict:
    """Cross-sectional momentum: each `horizon` hours, rank coins by past
    `lookback`-hour return; compare forward return of top_n vs bottom_n vs average.
    In-sample only, gross of costs."""
    closes = pd.DataFrame({k: v["close"] for k, v in coins.items()})
    closes = closes[closes.index < HOLDOUT_START]
    past = closes / closes.shift(lookback) - 1
    fwd = closes.shift(-horizon) / closes - 1
    top, bot, avg, ics = [], [], [], []
    for t in closes.index[lookback::horizon]:
        p, f = past.loc[t].dropna(), fwd.loc[t].dropna()
        idx = p.index.intersection(f.index)
        if len(idx) < 2 * top_n + 2:
            continue
        p, f = p[idx], f[idx]
        ranked = p.sort_values()
        top.append(f[ranked.index[-top_n:]].mean())
        bot.append(f[ranked.index[:top_n]].mean())
        avg.append(f.mean())
        ics.append(spearman(p, f))
    top, bot, avg = np.array(top), np.array(bot), np.array(avg)
    per_year = HOURS_PER_YEAR / horizon
    return {
        "lookback_h": lookback, "hold_h": horizon, "periods": len(top),
        "top_minus_avg_%/period": round((top - avg).mean() * 100, 3),
        "top_minus_bot_%/period": round((top - bot).mean() * 100, 3),
        "hit_rate_top>avg_%": round((top > avg).mean() * 100, 0),
        "mean_IC": round(float(np.nanmean(ics)), 3),
        "top_ann_gross_%": round(((1 + top).prod() ** (per_year / len(top)) - 1) * 100, 0),
        "avg_ann_gross_%": round(((1 + avg).prod() ** (per_year / len(avg)) - 1) * 100, 0),
        "cost_per_period_if_full_turnover_%": round(ROUND_TRIP_COST * 100, 2),
    }


def seasonality(coins: dict, names=("BTC", "ETH", "SOL", "BNB")) -> pd.DataFrame:
    r = pd.concat([np.log(coins[n]["close"]).diff() for n in names], axis=1).mean(axis=1)
    r = r[r.index < HOLDOUT_START].dropna()
    by_hour = r.groupby(r.index.hour).agg(["mean", "std"]) * 100
    by_hour.columns = ["mean_ret_%", "std_%"]
    by_hour["t_stat"] = by_hour["mean_ret_%"] / (by_hour["std_%"] / np.sqrt(r.groupby(r.index.hour).size()))
    return by_hour.round(4)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    coins = load()
    print(f"Loaded {len(coins)} coins. In-sample ends {HOLDOUT_START:%Y-%m-%d}; holdout after that.\n")

    stats = coin_stats(coins)
    stats.to_csv(os.path.join(OUT_DIR, "coin_stats.csv"))
    print("=== 1. Per-coin profile (sorted by Binance liquidity) ===")
    print(stats.to_string(), "\n")

    stale = stats[stats["last_bar"] < coins["BTC"].index[-1].strftime("%Y-%m-%d")]
    young = stats[stats["history_days"] < MIN_HISTORY_DAYS]
    print("Stale on Binance (no recent USDT data):", list(stale.index))
    print(f"Too young (<{MIN_HISTORY_DAYS}d history):", list(young.index), "\n")

    core = [c for c in stats.index if c not in stale.index and c not in young.index]
    core_coins = {c: coins[c] for c in core}

    print("=== 2. Buy & hold, every 14-day window, last 2 years (core coins) ===")
    ws = pd.DataFrame({c: window_stats(coins[c]["close"]) for c in core}).T
    ws.to_csv(os.path.join(OUT_DIR, "window14_buyhold.csv"))
    print(ws.sort_values("median_maxdd_%", ascending=False).to_string(), "\n")

    print("=== 3. Time-series momentum IC (Spearman, past k-h return vs next 24h), in-sample ===")
    ic = tsmom_ic(core_coins)
    ic.to_csv(os.path.join(OUT_DIR, "tsmom_ic.csv"))
    print(ic.to_string())
    print("Universe median:\n", ic.median().round(3).to_string(), "\n")

    print("=== 4. Trend filter (EMA24 > EMA96): next-24h return when up vs down, in-sample ===")
    tf = trend_filter_edge(core_coins)
    tf.to_csv(os.path.join(OUT_DIR, "trend_filter.csv"))
    print(tf.to_string())
    print("Universe mean:\n", tf.mean().round(3).to_string(), "\n")

    print("=== 5. Cross-sectional momentum (top 3 vs universe), in-sample, gross ===")
    xs = pd.DataFrame([xsec_momentum(core_coins, lb, h)
                       for lb in (24, 72, 168, 336) for h in (24, 72, 168)])
    xs.to_csv(os.path.join(OUT_DIR, "xsec_momentum.csv"), index=False)
    print(xs.to_string(index=False), "\n")

    print("=== 6. Hour-of-day mean return, BTC/ETH/SOL/BNB average, in-sample ===")
    print(seasonality(coins).to_string(), "\n")

    closes = pd.DataFrame({c: coins[c]["close"] for c in core})
    corr = np.log(closes).diff().corr()
    corr.to_csv(os.path.join(OUT_DIR, "correlation.csv"))
    print("=== 7. Average pairwise hourly correlation within core:",
          round(corr.values[np.triu_indices(len(corr), 1)].mean(), 2), "===")


if __name__ == "__main__":
    main()
