import numpy as np
import pandas as pd
import pytest

from backtest import engine, metrics


def frame(values, coins=("BTC",)):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({c: values for c in coins}, index=idx, dtype=float)


def test_executes_at_next_open_with_costs():
    opens = frame([100, 110, 120, 130])
    closes = frame([105, 115, 125, 135])
    target = frame([0.5, 0.5, 0.5, 0.5])
    res = engine.run(opens, closes, target, initial=1000, band=0.05)
    # target from bar 0 executes at bar 1 open = 110, not at bar 0 close
    first = res.trades.iloc[0]
    assert first["time"] == opens.index[1] and first["price"] == 110
    cost = engine.TAKER_FEE + engine.HALF_SPREAD
    assert first["fee"] == pytest.approx(500 * cost)
    assert res.equity.iloc[0] == 1000                     # nothing held at bar 0
    qty = 500 / 110
    assert res.equity.iloc[1] == pytest.approx(1000 - 500 * (1 + cost) + qty * 115)


def test_round_trip_costs_and_no_leverage():
    flat = frame([100] * 6)
    target = frame([1.0, 0.0, 1.0, 0.0, 0, 0])
    res = engine.run(flat, flat, target, initial=1000)
    assert res.equity.iloc[-1] < 1000
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all()
    assert len(res.trades) == 4


def test_metrics_known_values():
    idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    eq = pd.Series([100, 110, 99, 120, 108], index=idx, dtype=float)
    assert metrics.max_drawdown(eq) == pytest.approx(-0.1)
    m = metrics.summary(eq)
    r = eq.pct_change().dropna()
    assert m["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(8760))
    assert m["composite"] == pytest.approx(0.4 * m["sortino"] + 0.3 * m["sharpe"] + 0.3 * m["calmar"])
