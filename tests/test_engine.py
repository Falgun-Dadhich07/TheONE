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


def test_engine_applies_risk_overlay():
    from risk import RiskManager
    # price falls 30% steadily; strategy wants 100% in; kill switch must cut losses
    prices = list(np.linspace(100, 70, 60))
    o = frame(prices)
    target = frame([1.0] * 60)
    plain = engine.run(o, o, target, initial=1000)
    rm = RiskManager(max_drawdown=0.08, cooldown_hours=1000, stop_vol_mult=None)
    guarded = engine.run(o, o, target, initial=1000, risk=rm)
    assert metrics.max_drawdown(guarded.equity) > -0.10
    assert metrics.max_drawdown(plain.equity) < -0.25
    assert guarded.extra["risk_events"][0][1] == "KILL_SWITCH"


def test_short_profits_when_price_falls_and_pays_fees():
    prices = frame([100, 100, 90, 80, 80])
    target = frame([-0.5] * 5)
    res = engine.run(prices, prices, target, initial=1000, band=0.5)
    cost = engine.TAKER_FEE + engine.HALF_SPREAD
    assert res.trades.iloc[0]["side"] == "SHORT"
    # 500 collateral at 100 -> 5 units; at 80 worth 500 + 5*20 = 600
    assert res.equity.iloc[-1] == pytest.approx(1000 - 500 * cost + 100)
    assert res.weights.iloc[-1, 0] < 0


def test_short_loss_capped_at_collateral():
    prices = frame([100, 100, 150, 250, 300])
    target = frame([-0.5] * 5)
    # band=1.0: never top up the losing short. Price gaps 150 -> 250 (> 2x entry):
    # the first short loses exactly its 500 collateral, then a fresh short opens at 250.
    res = engine.run(prices, prices, target, initial=1000, band=1.0)
    cost = engine.TAKER_FEE + engine.HALF_SPREAD
    sides = list(res.trades["side"])
    assert sides[:3] == ["SHORT", "SHORT_WIPED", "SHORT"]
    cash_after_wipe = 1000 - 500 - 500 * cost
    new_collateral = 0.5 * cash_after_wipe
    assert res.equity.iloc[3] == pytest.approx(cash_after_wipe - new_collateral * cost)
    assert res.equity.min() > 0


def test_flip_long_to_short_closes_first():
    prices = frame([100] * 4)
    target = frame([0.5, -0.5, -0.5, -0.5])
    res = engine.run(prices, prices, target, initial=1000, band=0.05)
    assert list(res.trades["side"]) == ["BUY", "SELL", "SHORT"]
    assert res.weights.iloc[-1, 0] == pytest.approx(-0.5, rel=0.01)
