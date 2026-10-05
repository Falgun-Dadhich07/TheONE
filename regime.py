"""Market regime from a 2-state Gaussian hidden Markov model on BTC daily returns.

Shared by the live bot and the backtester (numpy only, no extra dependencies).

* Fit: Baum-Welch (EM) on all daily log returns strictly before the refit day.
* Refit days are fixed on the calendar (every REFIT_DAYS days since EPOCH), so
  the live bot and a backtest refit on exactly the same days.
* Filtering is causal (forward algorithm only): the probability for day d uses
  returns up to and including day d, and is used for trading on day d+1.
* "Bull" is the state with the higher mean/volatility ratio (state labels can
  swap between fits).
"""
from typing import Tuple

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2020-01-01", tz="UTC")
REFIT_DAYS = 30
MIN_DAYS = 180


def _emissions(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return np.exp(-(x[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var) + 1e-300


def fit(x: np.ndarray, n_iter: int = 200, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, var, transition matrix, initial probs) for 2 states."""
    mu = np.quantile(x, [0.25, 0.75])
    var = np.full(2, x.var())
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    prev_ll = -np.inf
    T = len(x)
    for _ in range(n_iter):
        B = _emissions(x, mu, var)
        al = np.zeros((T, 2))
        cs = np.zeros(T)
        al[0] = pi * B[0]
        cs[0] = al[0].sum()
        al[0] /= cs[0]
        for t in range(1, T):
            al[t] = (al[t - 1] @ A) * B[t]
            cs[t] = al[t].sum()
            al[t] /= cs[t]
        be = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / cs[t + 1]
        g = al * be
        g /= g.sum(axis=1, keepdims=True)
        xi = (al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :]) / cs[1:, None, None]
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


def filter_probs(x: np.ndarray, mu, var, A, pi) -> np.ndarray:
    B = _emissions(x, mu, var)
    al = np.zeros((len(x), 2))
    a = pi * B[0]
    al[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (al[t - 1] @ A) * B[t]
        al[t] = a / a.sum()
    return al


def bull_probability(daily_close: pd.Series, refit_days: int = REFIT_DAYS,
                     min_days: int = MIN_DAYS) -> pd.Series:
    """Daily filtered P(bull). Index = day d; value uses returns up to day d."""
    x = np.log(daily_close.astype(float)).diff().dropna()
    prob = pd.Series(np.nan, index=x.index)
    days_since = ((x.index.normalize() - EPOCH).days).to_numpy()
    refits = [i for i in range(min_days, len(x)) if days_since[i] % refit_days == 0]
    for j, st in enumerate(refits):
        nxt = refits[j + 1] if j + 1 < len(refits) else len(x)
        mu, var, A, pi = fit(x.values[:st])
        bull = int(np.argmax(mu / np.sqrt(var)))
        prob.iloc[st:nxt] = filter_probs(x.values[:nxt], mu, var, A, pi)[st:nxt, bull]
    return prob


def _with_epoch(offset_days: int):
    return EPOCH + pd.Timedelta(days=offset_days)


ENSEMBLE_OFFSETS = (0, 5, 10, 15, 20, 25)


def bull_probability_ensemble(daily_close: pd.Series, offsets=ENSEMBLE_OFFSETS) -> pd.Series:
    """Average P(bull) over several refit calendars.

    A single 30-day refit calendar made results depend on luck of timing
    (momentum sleeve +8.6%..+26.1% across offsets). Averaging removes that.
    """
    global EPOCH
    base = EPOCH
    try:
        probs = []
        for o in offsets:
            EPOCH = base + pd.Timedelta(days=o)
            probs.append(bull_probability(daily_close))
    finally:
        EPOCH = base
    return pd.concat(probs, axis=1).mean(axis=1, skipna=False)


def latest_bull_probability(daily_close: pd.Series, offsets=ENSEMBLE_OFFSETS,
                            refit_days: int = REFIT_DAYS, min_days: int = MIN_DAYS) -> float:
    """Fast path for the live bot: the last value of bull_probability_ensemble,
    computed with one fit per refit calendar instead of the whole history."""
    x = np.log(daily_close.astype(float)).diff().dropna()
    days_since = ((x.index.normalize() - EPOCH).days).to_numpy()
    out = []
    for o in offsets:
        refits = [i for i in range(min_days, len(x)) if (days_since[i] - o) % refit_days == 0]
        if not refits:
            return float("nan")
        st = refits[-1]
        mu, var, A, pi = fit(x.values[:st])
        bull = int(np.argmax(mu / np.sqrt(var)))
        out.append(filter_probs(x.values, mu, var, A, pi)[-1, bull])
    return float(np.mean(out))


def hourly(prob_daily: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.Series:
    """Day d's probability applies to every hour of day d+1 (it is only known after d ends)."""
    p = prob_daily.copy()
    p.index = p.index.normalize() + pd.Timedelta(days=1)
    return p.reindex(hourly_index, method="ffill")


def daily_close_from_hourly(closes: pd.Series) -> pd.Series:
    """Daily close = close of the last hourly bar of each UTC day (what Binance 1d klines report)."""
    return closes.resample("1D").last().dropna()
