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
