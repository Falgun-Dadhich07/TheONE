import numpy as np
import pandas as pd
import pytest

import strategy

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]


def synthetic_closes(hours=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    drift = np.linspace(-0.0004, 0.0006, len(COINS))
    r = rng.normal(drift, 0.01, size=(hours, len(COINS)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=COINS)


@pytest.mark.parametrize("name", list(strategy.STRATEGIES))
def test_strategies_are_causal(name):
    """Weights at time t must not change when future data is removed."""
    closes = synthetic_closes()
    full = strategy.STRATEGIES[name](closes)
    for cut in (800, 1100, 1499):
        part = strategy.STRATEGIES[name](closes.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut], check_freq=False)


@pytest.mark.parametrize("name", [n for n in strategy.STRATEGIES if n != "buy_hold_btc"])
def test_weights_respect_caps(name):
    w = strategy.STRATEGIES[name](synthetic_closes())
    assert (w >= 0).all().all()
    assert (w <= strategy.MAX_PER_COIN + 1e-9).all().all() or name == "equal_weight_half"
    assert (w.sum(axis=1) <= strategy.MAX_GROSS + 1e-9).all()


def test_regime_off_means_cash():
    closes = synthetic_closes()
    closes["BTC"] = np.linspace(200, 100, len(closes))  # BTC always below its SMA
    w = strategy.xs_momentum_regime(closes)
    assert w.iloc[600:].sum().sum() == 0


def test_momentum_picks_the_winners():
    closes = synthetic_closes()
    closes["BTC"] = np.linspace(100, 200, len(closes))  # regime on, BTC strong
    closes["DOGE"] = np.linspace(100, 400, len(closes))  # strongest
    w = strategy.xs_momentum_regime(closes, top_n=2)
    last = w.iloc[-1]
    assert set(last[last > 0].index) == {"DOGE", "BTC"}


def test_rebalance_trades_band():
    cur = {"BTC": 0.20, "ETH": 0.10}
    tgt = {"BTC": 0.22, "SOL": 0.15}
    out = strategy.rebalance_trades(cur, tgt, band=0.05)
    assert out == {"ETH": 0.0, "SOL": 0.15}          # BTC change too small
    assert strategy.rebalance_trades(cur, {"BTC": 0.30, "ETH": 0.10}, 0.05) == {"BTC": 0.30}


def test_rebalance_trades_signed():
    assert strategy.rebalance_trades({}, {"BTC": -0.2}) == {"BTC": -0.2}
    assert strategy.rebalance_trades({"BTC": -0.2}, {}) == {"BTC": 0.0}
    assert strategy.rebalance_trades({"BTC": 0.2}, {"BTC": -0.2}) == {"BTC": -0.2}
    assert strategy.rebalance_trades({"BTC": -0.20}, {"BTC": -0.22}) == {}
