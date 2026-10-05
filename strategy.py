"""Signal logic shared by the live bot and the backtester.

Every strategy is a function  f(closes, **params) -> weights  where
  closes  : DataFrame of hourly closes, UTC index, one column per coin
  weights : DataFrame, same shape, target fraction of equity per coin

Contract (enforced by tests/test_strategy.py):
  * Causal: the row at time t uses only closes up to and including t.
    The backtester trades it at the open of bar t+1; the live bot computes it
    right after bar t closes and trades immediately. Same function, same row.
  * Long-only, no leverage: 0 <= w, row sum <= MAX_GROSS.

The live bot calls  latest_weights(name, closes)  and uses the last row only.
"""
from typing import Callable, Dict

import numpy as np
import pandas as pd

import settings

HOURS_PER_YEAR = 24 * 365

# Risk settings shared by every candidate. Fixed, not optimized. Values live in settings.py.
TARGET_VOL = settings.TARGET_VOL        # annualized vol target (assumes correlation 1 = conservative)
VOL_LOOKBACK = settings.VOL_LOOKBACK    # hours of returns used for realized vol
MAX_PER_COIN = settings.MAX_PER_COIN    # fraction of equity
MAX_GROSS = settings.MAX_GROSS          # total exposure cap
REBALANCE_HOUR = settings.REBALANCE_HOUR


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def realized_vol(closes: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> pd.DataFrame:
    r = np.log(closes).diff()
    return r.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(HOURS_PER_YEAR)


def size_inverse_vol(selected: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Turn a 0/1 selection mask into vol-targeted weights with caps.

    Each selected coin gets  TARGET_VOL / (n_selected * its_vol), so the book's
    vol is <= TARGET_VOL even if all coins were perfectly correlated.
    """
    vol = realized_vol(closes)
    n = selected.sum(axis=1).replace(0, np.nan)
    w = selected.astype(float).div(vol).mul(TARGET_VOL).div(n, axis=0)
    w = w.clip(upper=MAX_PER_COIN).fillna(0.0)
    gross = w.sum(axis=1)
    scale = (MAX_GROSS / gross).where(gross > MAX_GROSS, 1.0)
    return w.mul(scale, axis=0)


def btc_regime(closes: pd.DataFrame, sma_hours: int) -> pd.Series:
    """True when BTC closes above its simple moving average."""
    btc = closes["BTC"]
    return btc > btc.rolling(sma_hours).mean()


def hold_until_rebalance(w: pd.DataFrame, hour: int = REBALANCE_HOUR) -> pd.DataFrame:
    """Only let target weights change at the daily rebalance hour."""
    held = w.copy()
    held.loc[w.index.hour != hour] = np.nan
    return held.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# Candidates (max 3 tunable parameters each)
# ---------------------------------------------------------------------------
def xs_momentum_regime(closes: pd.DataFrame, lookback: int = 336, regime_sma: int = 480,
                       top_n: int = 2) -> pd.DataFrame:
    """v1. Once a day: if BTC > its SMA, hold the top_n coins by
    `lookback`-hour return (only those with positive return), vol-sized;
    otherwise all cash.

    v0 checked the BTC regime every hour; BTC crossed its SMA 290 times in
    20 months and the whipsaw cost ~18% of capital in fees. Checking daily
    halved trades and fees and improved in-sample composite 0.68 -> 0.77
    (with the kill switch). Intraday crashes are left to the kill switch.
    """
    mom = closes / closes.shift(lookback) - 1
    rank = mom.rank(axis=1, ascending=False)
    on = btc_regime(closes, regime_sma)
    selected = (rank <= top_n) & (mom > 0) & on.to_numpy()[:, None]
    return hold_until_rebalance(size_inverse_vol(selected, closes))


def ema_trend(closes: pd.DataFrame, fast: int = 24, slow: int = 96) -> pd.DataFrame:
    """Phase A baseline idea. Each coin held while EMA(fast) > EMA(slow),
    vol-sized, re-evaluated every hour."""
    up = closes.ewm(span=fast, adjust=False).mean() > closes.ewm(span=slow, adjust=False).mean()
    return size_inverse_vol(up, closes)


def invvol_regime(closes: pd.DataFrame, regime_sma: int = 480) -> pd.DataFrame:
    """Hold every coin, vol-sized and rebalanced daily, while BTC > its SMA."""
    selected = closes.notna()
    w = hold_until_rebalance(size_inverse_vol(selected, closes))
    return w.mul(btc_regime(closes, regime_sma).astype(float), axis=0)


def reversal_regime(closes: pd.DataFrame, lookback: int = 72, regime_sma: int = 480,
                    bottom_n: int = 2) -> pd.DataFrame:
    """Mean reversion: daily, buy the bottom_n coins by `lookback`-hour return
    (the research showed 72h returns slightly revert), only while BTC > SMA."""
    ret = closes / closes.shift(lookback) - 1
    rank = ret.rank(axis=1, ascending=True)
    selected = (rank <= bottom_n) & ret.notna()
    w = hold_until_rebalance(size_inverse_vol(selected, closes))
    return w.mul(btc_regime(closes, regime_sma).astype(float), axis=0)


def buy_hold_btc(closes: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: 100% BTC, ignores the exposure cap on purpose."""
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    w["BTC"] = 1.0
    return w


def equal_weight_half(closes: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: MAX_GROSS spread equally, rebalanced daily, always invested."""
    w = closes.notna().astype(float).div(closes.notna().sum(axis=1), axis=0) * MAX_GROSS
    return hold_until_rebalance(w)


STRATEGIES: Dict[str, Callable[..., pd.DataFrame]] = {
    "xs_momentum_regime": xs_momentum_regime,
    "ema_trend": ema_trend,
    "invvol_regime": invvol_regime,
    "reversal_regime": reversal_regime,
    "buy_hold_btc": buy_hold_btc,
    "equal_weight_half": equal_weight_half,
}


REBALANCE_BAND = settings.REBALANCE_BAND  # min |target - current| weight change worth paying fees for


def rebalance_trades(current: Dict[str, float], target: Dict[str, float],
                     band: float = REBALANCE_BAND) -> Dict[str, float]:
    """Weight changes to execute, shared by live bot and backtester.

    Weights are signed (negative = short). Full exits, fresh entries and side
    flips always trade; otherwise only trade when the weight is off by more
    than `band`. Returns {coin: target_weight}.
    """
    out = {}
    for coin in sorted(set(current) | set(target)):     # sorted: same order every run
        cur, tgt = current.get(coin, 0.0), target.get(coin, 0.0)
        if (cur == 0) != (tgt == 0) or cur * tgt < 0 or abs(tgt - cur) > band:
            out[coin] = tgt
    return out


def latest_weights(name: str, closes: pd.DataFrame, **params) -> Dict[str, float]:
    """What the live bot calls after each hourly close: the last row of weights."""
    row = STRATEGIES[name](closes, **params).iloc[-1]
    return {coin: float(w) for coin, w in row.items() if w > 0}


# ---------------------------------------------------------------------------
# v3: 50/50 blend of regime-gated long/short momentum and Hawkes volatility
# breakouts on the 20 most liquid coins (research: docs/strategy_draft.md s.9-10)
# Weights are signed: negative = Roostoo 1x short.
# ---------------------------------------------------------------------------
def size_long_short(sel_long: pd.DataFrame, sel_short: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Inverse-vol size each side, combine signed, cap gross |w| at MAX_GROSS."""
    w = size_inverse_vol(sel_long, closes) - size_inverse_vol(sel_short, closes)
    gross = w.abs().sum(axis=1)
    return w.mul((MAX_GROSS / gross).where(gross > MAX_GROSS, 1.0), axis=0)


def liquid_universe(available: pd.DataFrame, daily_quote_volume: pd.DataFrame, top_n: int = 20,
                    min_history_h: int = 60 * 24, window_days: int = 30) -> pd.DataFrame:
    """Hourly eligibility: coin has a price now, >= min_history_h hours of history,
    and ranks in the top_n by trailing median daily quote volume, using only
    days that have fully ended (causal)."""
    med = daily_quote_volume.rolling(window_days, min_periods=20).median().shift(1)
    med.index = med.index.normalize()
    rank = med.rank(axis=1, ascending=False).reindex(available.index, method="ffill")
    rank = rank.reindex(columns=available.columns)
    history = available.cumsum()
    return available & (history >= min_history_h) & (rank <= top_n)


def momentum_long_short(closes: pd.DataFrame, eligible: pd.DataFrame, risk_on: pd.Series,
                        risk_off: pd.Series, lookback: int = 336, k: int = 2) -> pd.DataFrame:
    """Daily: long the top-k eligible coins by `lookback`-hour return (if > 0)
    while risk_on; short the bottom-k (if < 0) while risk_off; else cash."""
    mom = (closes / closes.shift(lookback) - 1).where(eligible)
    top = (mom.rank(axis=1, ascending=False) <= k) & (mom > 0) & risk_on.to_numpy()[:, None]
    bot = (mom.rank(axis=1, ascending=True) <= k) & (mom < 0) & risk_off.to_numpy()[:, None]
    return hold_until_rebalance(size_long_short(top, bot, closes))


def high_long_short(closes: pd.DataFrame, high: pd.DataFrame, eligible: pd.DataFrame, risk_on: pd.Series,
                    risk_off: pd.Series, lookback: int = 336, k: int = 3) -> pd.DataFrame:
    """Daily: long the k eligible coins closest to their `lookback`-hour high while
    risk_on; short the k furthest below it while risk_off; else cash.

    Nearness to the recent high predicted next-day cross-sectional returns far
    better than raw 14-day return (rank IC 0.05, t = 4.1, same sign in both
    halves of 2 years, vs ~0 for 14-day return): research/factor_lab.py.
    """
    score = (closes / high.rolling(lookback).max() - 1).where(eligible)
    top = (score.rank(axis=1, ascending=False) <= k) & risk_on.to_numpy()[:, None]
    bot = (score.rank(axis=1, ascending=True) <= k) & risk_off.to_numpy()[:, None]
    return hold_until_rebalance(size_long_short(top, bot, closes))


def hawkes_volatility(high: np.ndarray, low: np.ndarray, close: np.ndarray, kappa: float,
                      norm_lookback: int) -> np.ndarray:
    """kappa * Hawkes-smoothed (high - low) / ATR, all in log prices. Arrays are T x N."""
    lh, ll, lc = np.log(high), np.log(low), np.log(close)
    prev = np.vstack([np.full((1, lc.shape[1]), np.nan), lc[:-1]])
    tr = np.fmax(lh - ll, np.fmax(np.abs(lh - prev), np.abs(ll - prev)))
    atr = pd.DataFrame(tr).ewm(alpha=1.0 / norm_lookback, adjust=False, min_periods=norm_lookback).mean().to_numpy()
    norm_range = (lh - ll) / atr
    alpha = np.exp(-kappa)
    out = np.full_like(norm_range, np.nan)
    for i in range(1, len(norm_range)):
        p = out[i - 1]
        out[i] = np.where(np.isnan(p), norm_range[i], p * alpha + norm_range[i])
    return out * kappa


def hawkes_signal(v: np.ndarray, close: np.ndarray, signal_lookback: int, perc5: float,
                  perc95: float) -> np.ndarray:
    """+1 / -1 / 0 per coin. After volatility was quiet (below its perc5 quantile),
    a breakout above the perc95 quantile goes long if price rose since the quiet
    point, short if it fell; a new quiet spell resets to flat."""
    vdf = pd.DataFrame(v)
    q05 = vdf.rolling(signal_lookback).quantile(perc5).to_numpy()
    q95 = vdf.rolling(signal_lookback).quantile(perc95).to_numpy()
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
            cur[idx] = np.where(close[i, idx] - close[last_below[idx], idx] > 0, 1.0, -1.0)
        sig[i] = cur
    return sig


def hawkes_long_short(high: pd.DataFrame, low: pd.DataFrame, closes: pd.DataFrame, eligible: pd.DataFrame,
                      kappa: float, norm_lookback: int, signal_lookback: int, perc5: float,
                      perc95: float) -> pd.DataFrame:
    v = hawkes_volatility(high.to_numpy(), low.to_numpy(), closes.to_numpy(), kappa, int(norm_lookback))
    sig = pd.DataFrame(hawkes_signal(v, closes.to_numpy(), int(signal_lookback), perc5, perc95),
                       index=closes.index, columns=closes.columns).where(eligible, 0.0)
    return size_long_short(sig > 0, sig < 0, closes)


def v3_blend(momentum_w: pd.DataFrame, hawkes_w: pd.DataFrame, hawkes_share: float = 0.5) -> pd.DataFrame:
    return hawkes_share * hawkes_w + (1 - hawkes_share) * momentum_w


def v3_weights(closes: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, daily_quote_volume: pd.DataFrame,
               bull_prob_hourly: pd.Series, hawkes_params: Dict[str, float], hawkes_share: float = 0.5,
               top_n: int = 20, regime_sma: int = 480, allow_shorts: bool = True,
               rank_by: str = "return", k: int = 2) -> pd.DataFrame:
    """Full v3/v4 target weights (signed). Causal: row t uses data up to bar t only.

    rank_by="return", k=2 : v3 (14-day return momentum sleeve)
    rank_by="high",   k=3 : v4 (nearness to the 14-day high)
    """
    available = closes.notna() & high.notna() & low.notna()
    px = closes.ffill()
    eligible = liquid_universe(available, daily_quote_volume, top_n)
    sma = btc_regime(px, regime_sma)
    on = (bull_prob_hourly > 0.5).fillna(False) & sma
    off = (bull_prob_hourly < 0.5).fillna(False) & ~sma
    if rank_by == "high":
        mom = high_long_short(px, high.ffill(), eligible, on, off, k=k)
    else:
        mom = momentum_long_short(px, eligible, on, off, k=k)
    hawk = hawkes_long_short(high.ffill(), low.ffill(), px, eligible, **hawkes_params)
    w = v3_blend(mom, hawk, hawkes_share)
    if not allow_shorts:
        w = w.clip(lower=0.0)
    return w.fillna(0.0)
