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

    Full exits and fresh entries always trade; otherwise only trade when the
    weight is off by more than `band`. Returns {coin: target_weight}.
    """
    out = {}
    for coin in set(current) | set(target):
        cur, tgt = current.get(coin, 0.0), target.get(coin, 0.0)
        if tgt <= 0 < cur or (cur <= 0 < tgt) or abs(tgt - cur) > band:
            out[coin] = tgt
    return out


def latest_weights(name: str, closes: pd.DataFrame, **params) -> Dict[str, float]:
    """What the live bot calls after each hourly close: the last row of weights."""
    row = STRATEGIES[name](closes, **params).iloc[-1]
    return {coin: float(w) for coin, w in row.items() if w > 0}
