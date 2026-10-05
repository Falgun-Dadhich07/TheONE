"""TheONE: autonomous Roostoo trading bot.

Every ~60s:   ticker + balance (+ short positions) -> equity, logs, kill-switch check
Each new hourly bar close:  strategy target -> risk overlay -> rebalance orders
Daily:        activity guarantee (smallest valid round trip if too few trades)

Strategy (settings.STRATEGY):
  v3 (default): 50/50 blend of HMM+SMA-gated long/short momentum and Hawkes
                long/short on the 20 most liquid coins. Shorts are Roostoo 1x
                collateral shorts; if the exchange refuses them, the bot
                switches to long-only automatically and remembers it.
  v1          : original long-only momentum on 6 majors (rollback option).

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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
import regime
import settings
import strategy
from execution import (Executor, Order, ShortsNotAllowed, cap_buys_to_cash, plan_orders, round_down,
                       short_book, value_portfolio)
from market_data import HourlyBars
from market_data_v3 import MarketData
from risk import RiskManager
from roostoo_client import RoostooClient, RoostooError

log = logging.getLogger("bot")
MIN_BARS_V1 = max(settings.STRATEGY_PARAMS.get("regime_sma", 480),
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


def crypto_coins(rules: Dict, quote: str = "USD") -> List[str]:
    return sorted(p.split("/")[0] for p in rules if p.endswith("/" + quote)
                  and p.split("/")[0] not in settings.NON_CRYPTO)


def v3_target(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, daily_qv: pd.DataFrame,
              bull_by_day: Dict[pd.Timestamp, float], allow_shorts: bool) -> Dict[str, float]:
    """Last row of strategy.v3_weights on live data.

    bull_by_day maps a completed UTC day d -> P(bull) using returns up to d;
    it applies to the hours of day d+1, exactly like the backtest. An empty
    row is added for today so the liquidity ranking sees yesterday's volume
    on the same schedule as the backtest (which resamples partial days).
    """
    today = close.index[-1].normalize()
    dqv = daily_qv.reindex(daily_qv.index.union(pd.DatetimeIndex([today])))
    prob = regime.hourly(pd.Series(bull_by_day, dtype=float).sort_index(), close.index)
    w = strategy.v3_weights(close, high, low, dqv, prob, settings.HAWKES_PARAMS, settings.V3_HAWKES_SHARE,
                            settings.V3_TOP_N, settings.V3_REGIME_SMA, allow_shorts=allow_shorts)
    row = w.iloc[-1]
    return {c: float(v) for c, v in row.items() if abs(v) > 1e-6}


# ---------------------------------------------------------------------------
class Bot:
    def __init__(self):
        self.stop = False
        self.client = RoostooClient(min_interval=settings.API_MIN_INTERVAL_S)
        self.exec = Executor(self.client, config.MODE, settings.DRY_RUN)
        self.v3 = settings.STRATEGY == "v3"
        self.bars: Optional[HourlyBars] = None
        self.md: Optional[MarketData] = None
        self.risk = RiskManager(settings.MAX_DRAWDOWN, settings.KILL_COOLDOWN_HOURS, settings.STOP_VOL_MULT)
        self.target: Dict[str, float] = {}
        self.day = ""
        self.fills_today = 0
        self.strategy_orders_today: Dict[str, int] = {}
        self.shorts_disabled = not settings.ALLOW_SHORTS
        self.bull_by_day: Dict[str, float] = {}
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
        self.shorts_disabled = self.shorts_disabled or bool(s.get("shorts_disabled", False))
        self.bull_by_day = s.get("bull_by_day", {})
        log.info("state loaded: day=%s fills_today=%d shorts_disabled=%s risk=%s", self.day, self.fills_today,
                 self.shorts_disabled, s.get("risk"))

    def save_state(self) -> None:
        os.makedirs(os.path.dirname(settings.STATE_FILE), exist_ok=True)
        tmp = settings.STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"mode": config.MODE, "strategy": settings.STRATEGY, "risk": self.risk.to_dict(),
                       "day": self.day, "fills_today": self.fills_today,
                       "strategy_orders_today": self.strategy_orders_today,
                       "shorts_disabled": self.shorts_disabled, "bull_by_day": self.bull_by_day}, f, indent=1)
        os.replace(tmp, settings.STATE_FILE)

    # ---------------- startup ----------------
    def startup(self) -> None:
        log.info("=== TheONE starting | mode=%s DRY_RUN=%s strategy=%s shorts=%s ===", config.MODE,
                 settings.DRY_RUN, settings.STRATEGY, settings.ALLOW_SHORTS)
        server_ms = self.client.server_time()
        skew = server_ms - int(time.time() * 1000)
        self.client.time_offset_ms = skew if abs(skew) > 1000 else 0
        log.info("clock skew vs server: %d ms%s", skew, " (correcting)" if self.client.time_offset_ms else "")
        if abs(skew) > settings.MAX_CLOCK_SKEW_MS:
            log.warning("clock skew is large; check NTP on this machine")
        self.exec.load_rules()
        self.load_state()
        if self.v3:
            coins = crypto_coins(self.exec.rules, settings.QUOTE)
            self.md = MarketData(coins)
            log.info("warmup (%d coins): %s", len(coins), self.md.warmup())
            self.refresh_regime()
        else:
            missing = [c for c in settings.UNIVERSE if f"{c}/{settings.QUOTE}" not in self.exec.rules]
            if missing:
                raise RuntimeError(f"pairs not on exchange: {missing}")
            self.bars = HourlyBars(settings.UNIVERSE)
            log.info("warmup: %s", self.bars.warmup())

    def refresh_regime(self) -> None:
        """P(bull) for the last 3 completed days (only days not computed yet; ~7s each)."""
        btc = self.md.daily_close["BTC"].dropna()
        for d in btc.index[-3:]:
            key = d.strftime("%Y-%m-%d")
            if key not in self.bull_by_day:
                self.bull_by_day[key] = regime.latest_bull_probability(btc.loc[:d])
                log.info("HMM P(bull) for %s = %.3f", key, self.bull_by_day[key])
        for k in sorted(self.bull_by_day)[:-10]:
            del self.bull_by_day[k]

    # ---------------- logging ----------------
    def log_equity(self, now: datetime, equity: float, usd: float, long_x: float, short_x: float) -> None:
        new = not os.path.exists(self.equity_log)
        peak = self.risk.peak or equity
        with open(self.equity_log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["utc_time", "mode", "dry_run", "equity", "usd", "exposure_pct", "peak",
                            "drawdown_pct", "halted", "fills_today", "short_exposure_pct"])
            w.writerow([now.isoformat(timespec="seconds"), config.MODE, settings.DRY_RUN, round(equity, 2),
                        round(usd, 2), round(long_x * 100, 2), round(peak, 2),
                        round((equity / peak - 1) * 100, 3), self.risk.halted_until is not None,
                        self.fills_today, round(short_x * 100, 2)])

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
        data = self.md if self.v3 else self.bars
        data.on_poll(ts, prices)
        new_bars = data.close_completed_bars(ts)
        if new_bars:
            log.info("bar closed: %s", ", ".join(f"{b:%m-%d %H}h" for b in new_bars[-3:]))
            if self.v3:
                self.refresh_regime()
            self.needs_rebalance = True

        wallet = self.client.balance()
        shorts: List[dict] = []
        if self.v3:
            try:
                shorts = self.client.short_positions()
            except RoostooError as exc:
                log.warning("short_positions failed: %s", exc)
        equity, total, free = value_portfolio(wallet, prices, settings.QUOTE, shorts=shorts)
        book = short_book(shorts, settings.QUOTE)
        usd_free = float(wallet.get(settings.QUOTE, {}).get("Free", 0) or 0)
        long_val = sum(q * prices.get(c, 0) for c, q in total.items())
        short_val = sum(s["value"] for s in book.values())

        if self.needs_rebalance:
            self.target = self.compute_target()

        dvol = {}
        closes = self.md.close if self.v3 else self.bars.closes
        if len(closes) > settings.VOL_LOOKBACK:
            dvol = (strategy.realized_vol(closes.ffill()).iloc[-1] / np.sqrt(365)).to_dict()
        n_events = len(self.risk.events)
        adjusted = self.risk.apply(ts, equity, self.target, prices, dict(total), dvol)
        for ev in self.risk.events[n_events:]:
            log.warning("RISK EVENT %s %s equity=%.2f peak=%s", ev[1], ev[2] or "", equity, self.risk.peak)
            self.needs_rebalance = True

        self.log_equity(now, equity, usd_free, long_val / equity if equity else 0,
                        short_val / equity if equity else 0)

        if self.needs_rebalance:
            log.info("equity $%.2f | target %s | adjusted %s", equity,
                     {k: round(v, 3) for k, v in self.target.items()},
                     {k: round(v, 3) for k, v in adjusted.items()})
            self.rebalance(adjusted, total, free, prices, equity, usd_free, book)
            self.needs_rebalance = False

        self.activity_guarantee(now, prices)
        self.save_state()

    def compute_target(self) -> Dict[str, float]:
        if self.v3:
            if not self.md.ready():
                log.warning("market data not ready; strategy idle")
                return {}
            bull = {pd.Timestamp(k, tz="UTC"): v for k, v in self.bull_by_day.items()}
            return v3_target(self.md.close, self.md.high, self.md.low, self.md.daily_qv, bull,
                             allow_shorts=not self.shorts_disabled)
        if not self.bars.ready(MIN_BARS_V1):
            log.warning("not enough bars yet (%d < %d); strategy idle", len(self.bars.closes), MIN_BARS_V1)
            return {}
        return strategy.latest_weights("xs_momentum_regime", self.bars.closes, **settings.STRATEGY_PARAMS)

    def rebalance(self, target, total, free, prices, equity, usd_free, book) -> None:
        if not self.v3:
            total = {c: q for c, q in total.items() if c in settings.UNIVERSE}
        orders, skips = plan_orders(target, total, free, prices, equity, self.exec.rules,
                                    quote=settings.QUOTE, shorts=book)
        orders, more = cap_buys_to_cash(orders, usd_free, self.exec.rules)
        for s in skips + more:
            self.exec.log_skip(s)
        if not orders:
            log.info("rebalance: nothing to do")
        for o in orders:
            if self.stop:
                break
            coin = o.pair.split("/")[0]
            is_exit = (o.side == "SELL" and target.get(coin, 0) <= 0) or \
                      (o.side == "SHORT_CLOSE" and target.get(coin, 0) >= 0)
            n = self.strategy_orders_today.get(o.pair, 0)
            if not is_exit and n >= settings.MAX_STRATEGY_ORDERS_PER_PAIR_PER_DAY:
                self.exec.log_skip(f"{o.pair}: daily order limit reached ({n}) [{o.reason}]")
                continue
            if o.side == "SHORT_OPEN" and self.shorts_disabled:
                self.exec.log_skip(f"{o.pair}: shorts disabled [{o.reason}]")
                continue
            try:
                filled = self.exec.execute(o)
            except ShortsNotAllowed:
                self.shorts_disabled = True
                self.needs_rebalance = True     # recompute a long-only target next cycle
                self.save_state()
                continue
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
