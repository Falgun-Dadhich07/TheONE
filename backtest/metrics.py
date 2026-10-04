"""Performance metrics on an hourly equity curve (annualized with 24*365).

Sharpe  = mean(r) / std(r) * sqrt(N)                      (risk-free = 0)
Sortino = mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(N)     (downside deviation, target 0)
Calmar  = CAGR / |max drawdown|
Composite (competition) = 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar
The organizers have not published their exact formulas; these are the
standard definitions.
"""
import numpy as np
import pandas as pd

N = 24 * 365


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summary(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    years = len(r) / N
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    sd = r.std()
    dd_dev = np.sqrt((np.minimum(r, 0) ** 2).mean())
    sharpe = r.mean() / sd * np.sqrt(N) if sd > 0 else 0.0
    sortino = r.mean() / dd_dev * np.sqrt(N) if dd_dev > 0 else 0.0
    mdd = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {
        "total_ret_%": total * 100,
        "cagr_%": cagr * 100,
        "max_dd_%": mdd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar,
    }


def rolling_windows(equity: pd.Series, days: int = 14) -> pd.DataFrame:
    """Every `days`-day window, starting at each UTC midnight."""
    rows = []
    starts = equity.index[equity.index.hour == 0]
    span = pd.Timedelta(days=days)
    for s in starts:
        if s + span > equity.index[-1]:
            break
        seg = equity[(equity.index >= s) & (equity.index <= s + span)]
        m = summary(seg)
        rows.append({"start": s, "ret_%": m["total_ret_%"], "max_dd_%": m["max_dd_%"],
                     "composite": m["composite"]})
    if not rows:  # period shorter than one window
        return pd.DataFrame(columns=["ret_%", "max_dd_%", "composite"])
    return pd.DataFrame(rows).set_index("start")


def window_summary(win: pd.DataFrame) -> dict:
    if win.empty:
        return {k: float("nan") for k in ("w14_median_ret_%", "w14_p10_ret_%", "w14_pct_losing",
                                          "w14_median_dd_%", "w14_worst_dd_%", "w14_median_composite")}
    return {
        "w14_median_ret_%": win["ret_%"].median(),
        "w14_p10_ret_%": win["ret_%"].quantile(0.10),
        "w14_pct_losing": (win["ret_%"] < 0).mean() * 100,
        "w14_median_dd_%": win["max_dd_%"].median(),
        "w14_worst_dd_%": win["max_dd_%"].min(),
        "w14_median_composite": win["composite"].median(),
    }
