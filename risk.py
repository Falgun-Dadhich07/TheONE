"""Risk overlay shared by the live bot and the backtester.

Sits between the strategy's target weights and execution. Two rules:

1. Kill switch (portfolio): if equity falls more than MAX_DRAWDOWN below its
   peak, go to cash for COOLDOWN_HOURS. Then resume with the peak reset to
   the current equity. This is automatic: the FAQ forbids manually stopping or
   overriding the bot, so it can't wait for a manual reset.
2. Trailing stop (per coin): while a coin is held, track its highest price.
   If it falls more than STOP_VOL_MULT x its daily volatility below that high,
   sell it and block re-entry for COOLDOWN_HOURS.

State is a plain dict so the live bot can persist it across restarts.
"""
from typing import Dict, List, Optional

import pandas as pd

MAX_DRAWDOWN = 0.08
COOLDOWN_HOURS = 24
# Trailing stop is OFF by default: in backtests (backtest/run_risk.py) it fired
# 12 times in 20 months and lowered return and composite in both periods,
# because the momentum picks often dip and recover. Set e.g. 3.0 to enable.
STOP_VOL_MULT = None


class RiskManager:
    def __init__(self, max_drawdown: Optional[float] = MAX_DRAWDOWN,
                 cooldown_hours: int = COOLDOWN_HOURS,
                 stop_vol_mult: Optional[float] = STOP_VOL_MULT):
        self.max_drawdown = max_drawdown          # None disables the kill switch
        self.cooldown = pd.Timedelta(hours=cooldown_hours)
        self.stop_vol_mult = stop_vol_mult        # None disables trailing stops
        self.peak: Optional[float] = None
        self.halted_until: Optional[pd.Timestamp] = None
        self.coin_high: Dict[str, float] = {}
        self.blocked_until: Dict[str, pd.Timestamp] = {}
        self.events: List[tuple] = []

    # ------------------------------------------------------------------
    def apply(self, now: pd.Timestamp, equity: float, target: Dict[str, float],
              prices: Dict[str, float], held: Dict[str, float],
              daily_vol: Dict[str, float]) -> Dict[str, float]:
        """Return the risk-adjusted target weights for the next execution.

        now       : time of the bar close just observed
        equity    : portfolio value at that close
        target    : strategy weights {coin: w}
        prices    : close prices {coin: px}
        held      : current quantities {coin: qty}
        daily_vol : daily vol fraction per coin (annual vol / sqrt(365))
        """
        # --- kill switch -------------------------------------------------
        if self.halted_until is not None:
            if now < self.halted_until:
                return {}
            self.halted_until = None
            self.peak = equity
            self.events.append((now, "RESUME", None, equity))
        self.peak = equity if self.peak is None else max(self.peak, equity)
        if self.max_drawdown is not None and equity <= self.peak * (1 - self.max_drawdown):
            self.halted_until = now + self.cooldown
            self.coin_high.clear()
            self.events.append((now, "KILL_SWITCH", None, equity))
            return {}

        # --- trailing stops ----------------------------------------------
        out = dict(target)
        for coin in list(self.coin_high):
            if held.get(coin, 0) <= 0:
                del self.coin_high[coin]
        for coin, qty in held.items():
            if qty <= 0 or coin not in prices:
                continue
            px = prices[coin]
            self.coin_high[coin] = max(self.coin_high.get(coin, px), px)
            vol = daily_vol.get(coin)
            if self.stop_vol_mult is None or vol is None or vol != vol:  # NaN check
                continue
            if px < self.coin_high[coin] * (1 - self.stop_vol_mult * vol):
                self.blocked_until[coin] = now + self.cooldown
                del self.coin_high[coin]
                self.events.append((now, "STOP", coin, px))
        for coin, until in list(self.blocked_until.items()):
            if now >= until:
                del self.blocked_until[coin]
            else:
                out.pop(coin, None)
        return out

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "peak": self.peak,
            "halted_until": self.halted_until.isoformat() if self.halted_until is not None else None,
            "coin_high": self.coin_high,
            "blocked_until": {k: v.isoformat() for k, v in self.blocked_until.items()},
        }

    def load_dict(self, d: dict) -> None:
        self.peak = d.get("peak")
        h = d.get("halted_until")
        self.halted_until = pd.Timestamp(h) if h else None
        self.coin_high = dict(d.get("coin_high", {}))
        self.blocked_until = {k: pd.Timestamp(v) for k, v in d.get("blocked_until", {}).items()}
