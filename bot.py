"""TheONE: autonomous Roostoo trading bot.

Every ~60s:   ticker (1 call) + balance (1 call) -> equity, logs, kill-switch check
Each new hourly bar close:  strategy target -> risk overlay -> rebalance orders
Daily:        activity guarantee (smallest valid round trip if too few trades)

Run:   python bot.py            (DRY_RUN comes from .env, default true)
Stop:  Ctrl+C (finishes the current step, saves state, exits)
"""
import csv
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config
import settings
import strategy
from execution import Executor, Order, cap_buys_to_cash, plan_orders, round_down, value_portfolio
from market_data import HourlyBars
from risk import RiskManager
from roostoo_client import RoostooClient

log = logging.getLogger("bot")
MIN_BARS = max(settings.STRATEGY_PARAMS.get("regime_sma", 480),
               settings.STRATEGY_PARAMS.get("lookback", 336)) + settings.VOL_LOOKBACK


# ---------------------------------------------------------------------------
# Pure decision helpers (unit-tested)
# ---------------------------------------------------------------------------
def activity_trades_needed(now: datetime, fills_today: int, min_trades: int, hour_utc: int) -> int:
    """How many more fills we need today, once it's late enough to top up."""
    if now.hour < hour_utc:
        return 0
    return max(0, min_trades - fills_today)


def competition_allowed(mode: str, dry_run: bool, platform: str, confirm: Optional[str]) -> bool:
    """Live competition orders only from the EC2 box, never from a Windows laptop."""
    if mode != "competition" or dry_run:
        return True
    return not platform.startswith("win") and confirm == settings.COMPETITION_CONFIRM_VALUE


# ---------------------------------------------------------------------------
class Bot:
    def __init__(self):
        self.stop = False
        self.client = RoostooClient(min_interval=settings.API_MIN_INTERVAL_S)
        self.exec = Executor(self.client, config.MODE, settings.DRY_RUN)
        self.bars = HourlyBars(settings.UNIVERSE)
        self.risk = RiskManager(settings.MAX_DRAWDOWN, settings.KILL_COOLDOWN_HOURS, settings.STOP_VOL_MULT)
        self.target: Dict[str, float] = {}
        self.day = ""
        self.fills_today = 0
        self.strategy_orders_today: Dict[str, int] = {}
        self.needs_rebalance = True           # sync with the exchange right after startup
        self.equity_log = os.path.join(settings.LOG_DIR, "equity.csv")

    # ---------------- state ----------------
    def load_state(self) -> None:
        if not os.path.exists(settings.STATE_FILE):
            return
        with open(settings.STATE_FILE) as f:
            s = json.load(f)
        if s.get("mode") != config.MODE:
            log.warning("state file is for mode %s, ignoring it in mode %s", s.get("mode"), config.MODE)
            return
        self.risk.load_dict(s.get("risk", {}))
        self.day = s.get("day", "")
        self.fills_today = s.get("fills_today", 0)
        self.strategy_orders_today = s.get("strategy_orders_today", {})
        log.info("state loaded: day=%s fills_today=%d risk=%s", self.day, self.fills_today, s.get("risk"))

    def save_state(self) -> None:
        os.makedirs(os.path.dirname(settings.STATE_FILE), exist_ok=True)
        tmp = settings.STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"mode": config.MODE, "risk": self.risk.to_dict(), "day": self.day,
                       "fills_today": self.fills_today,
                       "strategy_orders_today": self.strategy_orders_today}, f, indent=1)
        os.replace(tmp, settings.STATE_FILE)

    # ---------------- startup ----------------
    def startup(self) -> None:
        log.info("=== TheONE starting | mode=%s DRY_RUN=%s strategy=%s %s ===", config.MODE,
                 settings.DRY_RUN, settings.STRATEGY, settings.STRATEGY_PARAMS)
        server_ms = self.client.server_time()
        skew = server_ms - int(time.time() * 1000)
        self.client.time_offset_ms = skew if abs(skew) > 1000 else 0
        log.info("clock skew vs server: %d ms%s", skew, " (correcting)" if self.client.time_offset_ms else "")
        if abs(skew) > settings.MAX_CLOCK_SKEW_MS:
            log.warning("clock skew is large; check NTP on this machine")
        self.exec.load_rules()
        missing = [c for c in settings.UNIVERSE if f"{c}/{settings.QUOTE}" not in self.exec.rules]
        if missing:
            raise RuntimeError(f"pairs not on exchange: {missing}")
        log.info("warmup: %s", self.bars.warmup())
        self.load_state()

    # ---------------- logging ----------------
    def log_equity(self, now: datetime, equity: float, usd: float, exposure: float) -> None:
        new = not os.path.exists(self.equity_log)
        peak = self.risk.peak or equity
        with open(self.equity_log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["utc_time", "mode", "dry_run", "equity", "usd", "exposure_pct", "peak",
                            "drawdown_pct", "halted", "fills_today"])
            w.writerow([now.isoformat(timespec="seconds"), config.MODE, settings.DRY_RUN, round(equity, 2),
                        round(usd, 2), round(exposure * 100, 2), round(peak, 2),
                        round((equity / peak - 1) * 100, 3), self.risk.halted_until is not None,
                        self.fills_today])

    # ---------------- one cycle ----------------
    def cycle(self) -> None:
        now = datetime.now(timezone.utc)
        ts = pd.Timestamp(now)
        today = now.strftime("%Y-%m-%d")
        if today != self.day:
            log.info("new UTC day %s (yesterday: %d fills)", today, self.fills_today)
            self.day, self.fills_today, self.strategy_orders_today = today, 0, {}

        tick = self.client.ticker()
        prices = {p.split("/")[0]: float(t["LastPrice"]) for p, t in tick.items()
                  if p.endswith("/" + settings.QUOTE) and t.get("LastPrice")}
        self.bars.on_poll(ts, prices)
        new_bars = self.bars.close_completed_bars(ts)
        if new_bars:
            log.info("bar closed: %s", ", ".join(f"{b:%m-%d %H}h" for b in new_bars))
            self.needs_rebalance = True

        wallet = self.client.balance()
        equity, total, free = value_portfolio(wallet, prices, settings.QUOTE)
        usd_free = float(wallet.get(settings.QUOTE, {}).get("Free", 0) or 0)
        held_val = sum(total.get(c, 0) * prices.get(c, 0) for c in settings.UNIVERSE)

        # strategy target, recomputed whenever a bar closes
        if self.needs_rebalance:
            if self.bars.ready(MIN_BARS):
                self.target = strategy.latest_weights(settings.STRATEGY, self.bars.closes,
                                                      **settings.STRATEGY_PARAMS)
            else:
                log.warning("not enough bars yet (%d < %d); strategy idle", len(self.bars.closes), MIN_BARS)
                self.target = {}

        # risk overlay every cycle so the kill switch reacts within a minute
        dvol = (strategy.realized_vol(self.bars.closes).iloc[-1] / np.sqrt(365)).to_dict() \
            if len(self.bars.closes) > settings.VOL_LOOKBACK else {}
        n_events = len(self.risk.events)
        held_qty = {c: total[c] for c in settings.UNIVERSE if c in total}
        adjusted = self.risk.apply(ts, equity, self.target, prices, held_qty, dvol)
        for ev in self.risk.events[n_events:]:
            log.warning("RISK EVENT %s %s equity=%.2f peak=%s", ev[1], ev[2] or "", equity, self.risk.peak)
            self.needs_rebalance = True

        self.log_equity(now, equity, usd_free, held_val / equity if equity else 0)

        if self.needs_rebalance:
            log.info("equity $%.2f | target %s | adjusted %s", equity,
                     {k: round(v, 3) for k, v in self.target.items()},
                     {k: round(v, 3) for k, v in adjusted.items()})
            self.rebalance(adjusted, total, free, prices, equity, usd_free)
            self.needs_rebalance = False

        self.activity_guarantee(now, prices)
        self.save_state()

    def rebalance(self, target, total, free, prices, equity, usd_free) -> None:
        universe_total = {c: q for c, q in total.items() if c in settings.UNIVERSE}
        orders, skips = plan_orders(target, universe_total, free, prices, equity, self.exec.rules,
                                    quote=settings.QUOTE)
        orders, more = cap_buys_to_cash(orders, usd_free, self.exec.rules)
        for s in skips + more:
            self.exec.log_skip(s)
        if not orders:
            log.info("rebalance: nothing to do")
        for o in orders:
            if self.stop:
                break
            is_exit = o.side == "SELL" and target.get(o.pair.split("/")[0], 0) <= 0
            n = self.strategy_orders_today.get(o.pair, 0)
            if not is_exit and n >= settings.MAX_STRATEGY_ORDERS_PER_PAIR_PER_DAY:
                self.exec.log_skip(f"{o.pair}: daily order limit reached ({n}) [{o.reason}]")
                continue
            filled = self.exec.execute(o)
            self.strategy_orders_today[o.pair] = n + 1
            if filled:
                self.fills_today += 1

    def activity_guarantee(self, now: datetime, prices: Dict[str, float]) -> None:
        need = activity_trades_needed(now, self.fills_today, settings.MIN_TRADES_PER_DAY,
                                      settings.ACTIVITY_HOUR_UTC)
        if need <= 0:
            return
        pair = settings.ACTIVITY_PAIR
        coin = pair.split("/")[0]
        rule, px = self.exec.rules[pair], prices.get(coin)
        if not px:
            return
        log.info("activity guarantee: %d fill(s) short today; round trip on %s", need, pair)
        rounds = (need + 1) // 2
        for _ in range(min(rounds, 5)):
            qty = round_down(settings.ACTIVITY_NOTIONAL_USD / px, rule.amount_precision)
            buy = self.exec.execute(Order(pair, "BUY", str(qty), px, "activity guarantee"))
            if not buy:
                return          # don't sell what we may not have; retry next cycle
            self.fills_today += 1
            got = float(buy.get("FilledQuantity") or qty)
            sell_qty = round_down(got, rule.amount_precision)
            if float(sell_qty) * px > rule.mini_order:
                if self.exec.execute(Order(pair, "SELL", str(sell_qty), px, "activity guarantee")):
                    self.fills_today += 1

    # ---------------- main loop ----------------
    def run(self) -> None:
        self.startup()
        failures = 0
        while not self.stop:
            try:
                self.cycle()
                failures = 0
                delay = settings.POLL_SECONDS
            except Exception:
                failures += 1
                delay = min(settings.POLL_SECONDS * 2 ** failures, settings.MAX_BACKOFF_SECONDS)
                log.exception("cycle failed (%d in a row); next try in %ds", failures, delay)
            end = time.time() + delay
            while not self.stop and time.time() < end:
                time.sleep(1)
        self.save_state()
        log.info("=== TheONE stopped cleanly ===")


def setup_logging() -> None:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fmt.converter = time.gmtime   # log in UTC
    fh = logging.handlers.RotatingFileHandler(os.path.join(settings.LOG_DIR, "bot.log"),
                                              maxBytes=5_000_000, backupCount=10)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [fh, sh]
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    if not competition_allowed(config.MODE, settings.DRY_RUN, sys.platform, os.getenv("CONFIRM_COMPETITION")):
        log.error("Refusing to trade the competition account here. Live competition orders need "
                  "CONFIRM_COMPETITION=%s in .env and must run on the EC2 instance.",
                  settings.COMPETITION_CONFIRM_VALUE)
        sys.exit(2)
    bot = Bot()

    def handle_stop(signum, _frame):
        log.info("signal %s received: finishing current step, then exiting", signum)
        bot.stop = True

    signal.signal(signal.SIGINT, handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop)
    bot.run()


if __name__ == "__main__":
    main()
