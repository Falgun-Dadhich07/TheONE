import pandas as pd

from risk import RiskManager

T0 = pd.Timestamp("2026-10-04 00:00", tz="UTC")
H = pd.Timedelta(hours=1)


def step(rm, i, equity, target=None, prices=None, held=None, vol=None):
    return rm.apply(T0 + i * H, equity, target or {"BTC": 0.3}, prices or {"BTC": 100.0},
                    held or {}, vol or {"BTC": 0.02})


def test_kill_switch_triggers_and_goes_to_cash():
    rm = RiskManager(max_drawdown=0.08, cooldown_hours=24, stop_vol_mult=None)
    assert step(rm, 0, 100_000) == {"BTC": 0.3}
    assert step(rm, 1, 105_000) == {"BTC": 0.3}          # new peak
    assert step(rm, 2, 97_000) == {"BTC": 0.3}           # -7.6%: still fine
    assert step(rm, 3, 96_500) == {}                     # -8.1%: kill
    assert rm.events[-1][1] == "KILL_SWITCH"


def test_cooldown_then_automatic_resume_with_reset_peak():
    rm = RiskManager(max_drawdown=0.08, cooldown_hours=24, stop_vol_mult=None)
    step(rm, 0, 100_000)
    step(rm, 1, 90_000)                                  # kill at t=1, halted until t=25
    assert step(rm, 10, 95_000) == {}
    assert step(rm, 24, 95_000) == {}
    assert step(rm, 25, 89_000) == {"BTC": 0.3}          # resumes
    assert rm.peak == 89_000                             # peak reset, no instant re-trigger
    assert [e[1] for e in rm.events] == ["KILL_SWITCH", "RESUME"]


def test_trailing_stop_exits_and_blocks_reentry():
    rm = RiskManager(max_drawdown=None, cooldown_hours=24, stop_vol_mult=3.0)
    held = {"BTC": 1.0}
    assert step(rm, 0, 1e5, prices={"BTC": 100}, held=held) == {"BTC": 0.3}
    assert step(rm, 1, 1e5, prices={"BTC": 110}, held=held) == {"BTC": 0.3}   # high = 110
    # stop distance = 3 * 2% = 6% below 110 -> 103.4
    assert step(rm, 2, 1e5, prices={"BTC": 104}, held=held) == {"BTC": 0.3}
    assert step(rm, 3, 1e5, prices={"BTC": 103}, held=held) == {}
    assert rm.events[-1][1:3] == ("STOP", "BTC")
    assert step(rm, 10, 1e5, prices={"BTC": 120}, held={}) == {}              # still blocked
    assert step(rm, 27, 1e5, prices={"BTC": 120}, held={}) == {"BTC": 0.3}    # unblocked


def test_other_coins_unaffected_by_stop():
    rm = RiskManager(max_drawdown=None, stop_vol_mult=3.0)
    tgt = {"BTC": 0.2, "ETH": 0.2}
    held = {"BTC": 1.0, "ETH": 1.0}
    rm.apply(T0, 1e5, tgt, {"BTC": 100, "ETH": 100}, held, {"BTC": 0.02, "ETH": 0.02})
    out = rm.apply(T0 + H, 1e5, tgt, {"BTC": 90, "ETH": 101}, held, {"BTC": 0.02, "ETH": 0.02})
    assert out == {"ETH": 0.2}


def test_state_round_trip():
    rm = RiskManager(stop_vol_mult=3.0)
    step(rm, 0, 100_000, held={"BTC": 1.0})
    step(rm, 1, 80_000)
    rm2 = RiskManager()
    rm2.load_dict(rm.to_dict())
    assert rm2.peak == rm.peak and rm2.halted_until == rm.halted_until
    assert step(rm2, 2, 80_000) == {}
