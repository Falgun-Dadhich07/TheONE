from decimal import Decimal

import pytest

from execution import PairRule, cap_buys_to_cash, plan_orders, round_down, value_portfolio

RULES = {
    "BTC/USD": PairRule(2, 5, 1.0),
    "ETH/USD": PairRule(2, 4, 1.0),
    "DOGE/USD": PairRule(5, 0, 1.0),
}
PRICES = {"BTC": 85_000.0, "ETH": 3_000.0, "DOGE": 0.2}


def test_round_down_never_rounds_up_and_has_no_float_artifacts():
    assert round_down(0.1 + 0.2, 1) == Decimal("0.3")          # 0.30000000000000004
    assert str(round_down(0.123456789, 5)) == "0.12345"
    assert str(round_down(0.999999, 3)) == "0.999"
    assert str(round_down(1234.9, 0)) == "1234"
    assert round_down(0.000009, 5) == Decimal("0.00000")
    assert round_down(-1, 3) == Decimal("0.000")


def test_value_portfolio_uses_free_and_lock():
    wallet = {"USD": {"Free": 40_000, "Lock": 0}, "BTC": {"Free": 0.1, "Lock": 0.02},
              "XYZ": {"Free": 0, "Lock": 0}}
    eq, total, free = value_portfolio(wallet, PRICES)
    assert eq == pytest.approx(40_000 + 0.12 * 85_000)
    assert total == {"BTC": pytest.approx(0.12)} and free == {"BTC": pytest.approx(0.1)}


def test_sizing_is_percent_of_equity_for_both_account_sizes():
    for equity in (50_000, 100_000):
        orders, skips = plan_orders({"BTC": 0.20}, {}, {}, PRICES, equity, RULES)
        assert len(orders) == 1 and orders[0].side == "BUY"
        assert orders[0].notional == pytest.approx(0.20 * equity, rel=1e-3)
        assert orders[0].notional <= 0.20 * equity                # rounded down, never up


def test_full_exit_sells_free_quantity_and_sells_come_first():
    total, free = {"BTC": 0.1, "DOGE": 50_000}, {"BTC": 0.1, "DOGE": 50_000}
    orders, _ = plan_orders({"ETH": 0.2}, total, free, PRICES, 100_000, RULES)
    assert [o.side for o in orders] == ["SELL", "SELL", "BUY"]
    assert {o.pair: o.qty for o in orders if o.side == "SELL"} == {"BTC/USD": "0.10000", "DOGE/USD": "50000"}


def test_small_orders_are_skipped_with_reason():
    orders, skips = plan_orders({"BTC": 0.06}, {}, {}, PRICES, 100, RULES)   # $6 < $10 minimum
    assert orders == [] and "below minimum" in skips[0]


def test_dust_is_ignored():
    orders, skips = plan_orders({}, {"BTC": 0.00002}, {"BTC": 0.00002}, PRICES, 50_000, RULES)
    assert orders == [] and skips == []                         # $1.70 of BTC is dust


def test_buys_capped_to_cash():
    orders, _ = plan_orders({"BTC": 0.25, "ETH": 0.25}, {}, {}, PRICES, 100_000, RULES)
    capped, _ = cap_buys_to_cash(orders, usd_free=20_000, rules=RULES)
    assert sum(o.notional for o in capped) * 1.001 <= 20_000
