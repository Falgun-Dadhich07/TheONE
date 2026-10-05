import numpy as np
import pandas as pd
import pytest

import regime
import settings
import strategy
from bot import v3_target
from execution import (Executor, Order, PairRule, ShortsNotAllowed, cap_buys_to_cash, plan_orders,
                       short_book, value_portfolio)
from roostoo_client import RoostooError

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK"]


def synthetic(hours=2600, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    r = rng.normal(0, 0.01, size=(hours, len(COINS))) + np.linspace(-3e-4, 3e-4, len(COINS))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=COINS)
    spread = np.abs(rng.normal(0, 0.006, size=close.shape))
    high, low = close * (1 + spread), close * (1 - spread)
    qv = pd.DataFrame(rng.uniform(1, 2, size=close.shape) * np.arange(len(COINS), 0, -1), index=idx, columns=COINS)
    return close, high, low, qv.resample("1D").sum()


PARAMS = {"kappa": 0.1, "norm_lookback": 48, "signal_lookback": 120, "perc5": 0.1, "perc95": 0.9}


def weights(close, high, low, dqv, prob):
    return strategy.v3_weights(close, high, low, dqv, prob, PARAMS, 0.5, top_n=5, regime_sma=240)


def test_v3_weights_are_causal_and_capped():
    close, high, low, dqv = synthetic()
    prob = pd.Series(np.where(np.arange(len(close)) % 700 < 350, 0.8, 0.2), index=close.index)
    full = weights(close, high, low, dqv, prob)
    assert (full.abs().sum(axis=1) <= settings.MAX_GROSS + 1e-9).all()
    assert (full < 0).any().any() and (full > 0).any().any()      # both sides used
    for cut in (1800, 2300):
        part = weights(close.iloc[:cut], high.iloc[:cut], low.iloc[:cut],
                       dqv.loc[: close.index[cut - 1].normalize()], prob.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut], check_freq=False)


def test_v3_no_shorts_when_disallowed():
    close, high, low, dqv = synthetic()
    prob = pd.Series(0.2, index=close.index)          # bear regime -> momentum wants shorts
    w = strategy.v3_weights(close, high, low, dqv, prob, PARAMS, 0.5, top_n=5, regime_sma=240,
                            allow_shorts=False)
    assert (w >= 0).all().all()


def test_liquid_universe_ranks_by_past_volume_and_needs_history():
    close, high, low, dqv = synthetic(hours=24 * 70)
    avail = close.notna()
    avail.iloc[: 24 * 65, avail.columns.get_loc("ADA")] = False           # ADA listed late
    el = strategy.liquid_universe(avail, dqv, top_n=3, min_history_h=24 * 60)
    last = el.iloc[-1]
    assert set(last[last].index) == {"BTC", "ETH", "SOL"}                 # highest volumes
    assert not el["ADA"].any()


def test_hmm_fast_path_matches_full_and_is_causal():
    rng = np.random.default_rng(3)
    days = pd.date_range("2024-10-01", periods=330, freq="D", tz="UTC")
    regimes = np.where((np.arange(330) // 60) % 2 == 0, 0.004, -0.004)
    daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(regimes, 0.03))), index=days)
    ens = regime.bull_probability_ensemble(daily)
    assert regime.latest_bull_probability(daily) == pytest.approx(ens.iloc[-1])
    cut = daily.iloc[:300]
    assert regime.latest_bull_probability(cut) == pytest.approx(ens.loc[cut.index[-1]])
    assert ens.dropna().between(0, 1).all()


def test_regime_hourly_uses_previous_day():
    p = pd.Series([0.9, 0.1], index=pd.to_datetime(["2026-10-01", "2026-10-02"], utc=True))
    idx = pd.date_range("2026-10-02", periods=48, freq="h", tz="UTC")
    h = regime.hourly(p, idx)
    assert (h.iloc[:24] == 0.9).all() and (h.iloc[24:] == 0.1).all()


def test_bot_v3_target_returns_signed_dict():
    close, high, low, dqv = synthetic()
    days = pd.Series(0.2, index=dqv.index)
    tgt = v3_target(close, high, low, dqv, days.to_dict(), allow_shorts=True)
    assert all(abs(v) <= settings.MAX_PER_COIN for v in tgt.values())
    tgt_long = v3_target(close, high, low, dqv, days.to_dict(), allow_shorts=False)
    assert all(v > 0 for v in tgt_long.values())


# ---------------------------------------------------------------- execution
RULES = {"BTC/USD": PairRule(2, 5, 1.0), "ETH/USD": PairRule(2, 4, 1.0)}
PRICES = {"BTC": 80_000.0, "ETH": 2_500.0}


def test_equity_counts_short_pnl_not_collateral_twice():
    wallet = {"USD": {"Free": 9_000, "Lock": 1_000, "ShortCollateral": 1_000}}
    shorts = [{"Pair": "BTC/USD", "ShortQty": 0.0125, "Collateral": 1000, "UnrealizedPNL": 50, "PositionValue": 1050}]
    eq, total, free = value_portfolio(wallet, PRICES, shorts=shorts)
    assert eq == pytest.approx(10_050)


def test_plan_open_short_and_cash_cap():
    orders, _ = plan_orders({"BTC": -0.2}, {}, {}, PRICES, 10_000, RULES)
    assert [(o.side, o.collateral) for o in orders] == [("SHORT_OPEN", "2000.00")]
    capped, _ = cap_buys_to_cash(orders, usd_free=1_000, rules=RULES)
    assert float(capped[0].collateral) * 1.001 <= 1_000


def test_plan_cover_trim_and_flips():
    book = short_book([{"Pair": "BTC/USD", "ShortQty": 0.025, "PositionValue": 2000}])
    # full cover
    o, _ = plan_orders({}, {}, {}, PRICES, 10_000, RULES, shorts=book)
    assert [(x.side, x.qty) for x in o] == [("SHORT_CLOSE", "")]
    # trim -20% -> -10%: close half of the quantity
    o, _ = plan_orders({"BTC": -0.10}, {}, {}, PRICES, 10_000, RULES, shorts=book)
    assert [(x.side, x.qty) for x in o] == [("SHORT_CLOSE", "0.01250")]
    # flip short -> long: cover first, then buy
    o, _ = plan_orders({"BTC": 0.10}, {}, {}, PRICES, 10_000, RULES, shorts=book)
    assert [x.side for x in o] == ["SHORT_CLOSE", "BUY"]
    # flip long -> short: sell first, then open short
    o, _ = plan_orders({"BTC": -0.10}, {"BTC": 0.025}, {"BTC": 0.025}, PRICES, 10_000, RULES)
    assert [x.side for x in o] == ["SELL", "SHORT_OPEN"]


class _RefusingClient:
    def short_open(self, pair, collateral):
        raise RoostooError(f"SHORT_OPEN {pair} rejected: this competition does not allow short positions")


def test_executor_raises_shorts_not_allowed(tmp_path):
    ex = Executor(_RefusingClient(), "test", dry_run=False, log_dir=str(tmp_path))
    with pytest.raises(ShortsNotAllowed):
        ex.execute(Order("BTC/USD", "SHORT_OPEN", "", 80_000, "test", collateral="100"))
    assert "SHORTS_NOT_ALLOWED" in (tmp_path / "orders.csv").read_text()


def test_v4_high_ranking_is_causal_and_picks_coins_near_their_high():
    close, high, low, dqv = synthetic()
    prob = pd.Series(np.where(np.arange(len(close)) % 700 < 350, 0.8, 0.2), index=close.index)

    def w4(c, h, l, q, p):
        return strategy.v3_weights(c, h, l, q, p, PARAMS, 0.5, top_n=6, regime_sma=240, rank_by="high", k=3)

    full = w4(close, high, low, dqv, prob)
    cut = 2200
    part = w4(close.iloc[:cut], high.iloc[:cut], low.iloc[:cut], dqv.loc[: close.index[cut - 1].normalize()],
              prob.iloc[:cut])
    pd.testing.assert_frame_equal(part, full.iloc[:cut], check_freq=False)

    # sleeve alone: in a risk-on market it holds exactly the 3 coins nearest their 14-day high
    el = pd.DataFrame(True, index=close.index, columns=close.columns)
    on = pd.Series(True, index=close.index)
    sleeve = strategy.high_long_short(close, high, el, on, ~on, k=3)
    t = close.index[close.index.hour == 0][-1]
    score = (close.loc[t] / high.rolling(336).max().loc[t] - 1)
    assert set(sleeve.loc[t][sleeve.loc[t] > 0].index) == set(score.sort_values(ascending=False).index[:3])
    assert (sleeve >= 0).all().all()
