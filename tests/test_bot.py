from datetime import datetime, timezone

from bot import activity_trades_needed, competition_allowed


def t(hour):
    return datetime(2026, 10, 5, hour, 0, tzinfo=timezone.utc)


def test_activity_waits_until_configured_hour():
    assert activity_trades_needed(t(19), 0, 2, 20) == 0
    assert activity_trades_needed(t(20), 0, 2, 20) == 2
    assert activity_trades_needed(t(23), 1, 2, 20) == 1


def test_activity_not_needed_when_strategy_traded():
    assert activity_trades_needed(t(21), 2, 2, 20) == 0
    assert activity_trades_needed(t(21), 5, 2, 20) == 0


def test_competition_guard():
    ok = "yes-on-ec2"
    assert competition_allowed("test", False, "win32", None)            # test mode anywhere
    assert competition_allowed("competition", True, "win32", None)       # dry run is harmless
    assert not competition_allowed("competition", False, "win32", ok)    # never from Windows laptop
    assert not competition_allowed("competition", False, "linux", None)  # needs explicit confirm
    assert competition_allowed("competition", False, "linux", ok)


def test_cycle_aborts_when_short_positions_unreadable(monkeypatch, tmp_path):
    """A failed /v6/short_positions read must stop the cycle before any order."""
    import pytest

    import bot as botmod
    import settings
    from roostoo_client import RoostooError

    class FakeClient:
        def __init__(self, *a, **k):
            self.time_offset_ms = 0
            self.orders = 0

        def ticker(self):
            return {"BTC/USD": {"LastPrice": 80000}}

        def balance(self):
            return {"USD": {"Free": 1000, "Lock": 0}}

        def short_positions(self):
            raise RoostooError("GET /v6/short_positions failed: timeout")

        def place_order(self, *a, **k):
            self.orders += 1

        def short_open(self, *a, **k):
            self.orders += 1

    class FakeData:
        close = None

        def on_poll(self, *a):
            pass

        def close_completed_bars(self, *a):
            return []

    monkeypatch.setattr(botmod, "RoostooClient", FakeClient)
    monkeypatch.setattr(settings, "STATE_FILE", str(tmp_path / "state.json"))
    b = botmod.Bot()
    b.v3, b.md = True, FakeData()
    with pytest.raises(RoostooError):
        b.cycle()
    assert b.client.orders == 0
