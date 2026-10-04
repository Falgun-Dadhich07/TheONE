"""Turning target weights into safe Roostoo orders.

Pure helpers (unit-tested): portfolio valuation, quantity rounding, order
planning. The Executor sends orders and logs every request and response to
logs/orders.csv.

Order safety rules:
  * place_order is never retried automatically (it could duplicate an order).
  * After a successful placement, query the order to confirm its status.
  * If placement fails ambiguously (network error/timeout), look at recent
    orders for that pair before concluding anything; never resend in the
    same cycle. The next cycle re-plans from real exchange balances.
"""
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Dict, List, Optional, Tuple

import settings
import strategy
from roostoo_client import RoostooError

log = logging.getLogger(__name__)


@dataclass
class PairRule:
    price_precision: int
    amount_precision: int
    mini_order: float


@dataclass
class Order:
    pair: str
    side: str            # BUY / SELL
    qty: str             # already rounded, as sent to the API
    price: float         # reference price used for sizing
    reason: str

    @property
    def notional(self) -> float:
        return float(self.qty) * self.price


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def round_down(qty: float, precision: int) -> Decimal:
    """Round a quantity DOWN to `precision` decimals, via str() to avoid float artifacts."""
    q = Decimal(str(qty))
    step = Decimal(1).scaleb(-precision)        # 10**-precision
    return q.quantize(step, rounding=ROUND_DOWN) if q > 0 else Decimal(0).quantize(step)


def value_portfolio(wallet: dict, prices: Dict[str, float], quote: str = "USD"
                    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Return (equity, total qty per coin, free qty per coin) from a balance wallet."""
    equity = 0.0
    total, free = {}, {}
    for asset, a in wallet.items():
        f, lock = float(a.get("Free", 0) or 0), float(a.get("Lock", 0) or 0)
        if asset == quote:
            equity += f + lock
            continue
        qty = f + lock
        if qty <= 0:
            continue
        total[asset], free[asset] = qty, f
        equity += qty * prices.get(asset, 0.0)
    return equity, total, free


def current_weights(total: Dict[str, float], prices: Dict[str, float], equity: float,
                    dust_usd: float = settings.DUST_USD) -> Dict[str, float]:
    if equity <= 0:
        return {}
    out = {}
    for coin, qty in total.items():
        v = qty * prices.get(coin, 0.0)
        if v >= dust_usd:
            out[coin] = v / equity
    return out


def plan_orders(target: Dict[str, float], total: Dict[str, float], free: Dict[str, float],
                prices: Dict[str, float], equity: float, rules: Dict[str, PairRule],
                band: float = settings.REBALANCE_BAND, quote: str = "USD",
                min_order_usd: float = settings.MIN_ORDER_USD
                ) -> Tuple[List[Order], List[str]]:
    """Orders that move the portfolio toward `target`, sells first. Also returns skip reasons."""
    cur = current_weights(total, prices, equity)
    changes = strategy.rebalance_trades(cur, target, band)
    orders: List[Order] = []
    skips: List[str] = []
    for coin, tgt in changes.items():
        pair = f"{coin}/{quote}"
        px = prices.get(coin)
        rule = rules.get(pair)
        if not px or rule is None:
            skips.append(f"{pair}: no price or exchange rule")
            continue
        delta_usd = tgt * equity - total.get(coin, 0.0) * px
        if tgt <= 0:      # full exit: sell everything free
            qty = round_down(free.get(coin, 0.0), rule.amount_precision)
            side, reason = "SELL", f"exit (target 0, was {cur.get(coin, 0):.1%})"
        elif delta_usd < 0:
            qty = round_down(min(-delta_usd / px, free.get(coin, 0.0)), rule.amount_precision)
            side, reason = "SELL", f"trim {cur.get(coin, 0):.1%} -> {tgt:.1%}"
        else:
            qty = round_down(delta_usd / px, rule.amount_precision)
            side, reason = "BUY", f"{'enter' if cur.get(coin, 0) == 0 else 'add'} {cur.get(coin, 0):.1%} -> {tgt:.1%}"
        notional = float(qty) * px
        floor = max(min_order_usd, rule.mini_order)
        if qty <= 0 or notional <= floor:
            skips.append(f"{pair}: {side} {qty} (${notional:,.2f}) below minimum ${floor:,.2f} [{reason}]")
            continue
        orders.append(Order(pair, side, str(qty), px, reason))
    orders.sort(key=lambda o: 0 if o.side == "SELL" else 1)
    return orders, skips


def cap_buys_to_cash(orders: List[Order], usd_free: float, rules: Dict[str, PairRule],
                     fee: float = 0.001, min_order_usd: float = settings.MIN_ORDER_USD
                     ) -> Tuple[List[Order], List[str]]:
    """Scale buys down so they fit the cash expected after sells (no leverage)."""
    cash = usd_free + sum(o.notional * (1 - fee) for o in orders if o.side == "SELL")
    buys = [o for o in orders if o.side == "BUY"]
    need = sum(o.notional * (1 + fee) for o in buys)
    if need <= cash * 0.995 or not buys:
        return orders, []
    scale = cash * 0.995 / need
    out, skips = [o for o in orders if o.side == "SELL"], []
    for o in buys:
        rule = rules[o.pair]
        qty = round_down(float(o.qty) * scale, rule.amount_precision)
        if float(qty) * o.price <= max(min_order_usd, rule.mini_order):
            skips.append(f"{o.pair}: BUY too small after cash cap")
            continue
        out.append(Order(o.pair, o.side, str(qty), o.price, o.reason + f" (scaled x{scale:.2f} to cash)"))
    return out, skips


# ---------------------------------------------------------------------------
# Order sending
# ---------------------------------------------------------------------------
ORDER_LOG_FIELDS = ["utc_time", "mode", "dry_run", "event", "pair", "side", "qty", "ref_price",
                    "notional", "reason", "order_id", "status", "filled_qty", "avg_price", "fee", "detail"]


class Executor:
    def __init__(self, client, mode: str, dry_run: bool, log_dir: str = settings.LOG_DIR):
        self.client = client
        self.mode = mode
        self.dry_run = dry_run
        self.rules: Dict[str, PairRule] = {}
        os.makedirs(log_dir, exist_ok=True)
        self.order_log = os.path.join(log_dir, "orders.csv")

    def load_rules(self) -> None:
        info = self.client.exchange_info()
        for pair, p in info.get("TradePairs", {}).items():
            self.rules[pair] = PairRule(int(p["PricePrecision"]), int(p["AmountPrecision"]),
                                        float(p.get("MiniOrder", 1)))

    def _log(self, event: str, order: Optional[Order] = None, detail=None, **kw) -> None:
        new = not os.path.exists(self.order_log)
        row = {
            "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": self.mode, "dry_run": self.dry_run, "event": event,
            "pair": order.pair if order else kw.get("pair", ""),
            "side": order.side if order else "", "qty": order.qty if order else "",
            "ref_price": order.price if order else "",
            "notional": round(order.notional, 2) if order else "",
            "reason": order.reason if order else kw.get("reason", ""),
            "order_id": kw.get("order_id", ""), "status": kw.get("status", ""),
            "filled_qty": kw.get("filled_qty", ""), "avg_price": kw.get("avg_price", ""),
            "fee": kw.get("fee", ""),
            "detail": json.dumps(detail, default=str)[:2000] if detail is not None else "",
        }
        with open(self.order_log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ORDER_LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    def log_skip(self, reason: str) -> None:
        log.info("SKIP %s", reason)
        self._log("SKIP", reason=reason)

    def execute(self, order: Order) -> Optional[dict]:
        """Send one order. Returns the confirmed order dict if it filled, else None."""
        self._log("REQUEST", order)
        if self.dry_run:
            log.info("DRY_RUN %s %s %s (~$%.2f) %s", order.side, order.qty, order.pair,
                     order.notional, order.reason)
            fake = {"Status": "FILLED", "FilledQuantity": float(order.qty), "FilledAverPrice": order.price}
            self._log("DRY_RUN", order, status="FILLED")
            return fake
        sent_ms = int(time.time() * 1000) + getattr(self.client, "time_offset_ms", 0)
        try:
            detail = self.client.place_order(order.pair, order.side, order.qty)
        except RoostooError as exc:
            msg = str(exc)
            if "rejected" in msg:          # exchange answered Success=false: not placed
                log.warning("REJECTED %s %s %s: %s", order.side, order.qty, order.pair, msg)
                self._log("REJECTED", order, detail=msg)
                return None
            log.error("AMBIGUOUS %s %s %s: %s; checking recent orders", order.side, order.qty,
                      order.pair, msg)
            self._log("AMBIGUOUS", order, detail=msg)
            return self._reconcile(order, sent_ms)
        self._log("RESPONSE", order, detail=detail, order_id=detail.get("OrderID"),
                  status=detail.get("Status"), filled_qty=detail.get("FilledQuantity"),
                  avg_price=detail.get("FilledAverPrice"), fee=detail.get("CommissionChargeValue"))
        return self._confirm(order, detail)

    def _confirm(self, order: Order, detail: dict) -> Optional[dict]:
        oid = detail.get("OrderID")
        confirmed = detail
        try:
            found = self.client.query_order(order_id=oid) if oid is not None else []
            if found:
                confirmed = found[0]
        except RoostooError as exc:
            log.warning("confirm query for order %s failed: %s (using placement response)", oid, exc)
        status = confirmed.get("Status")
        self._log("CONFIRM", order, detail=confirmed, order_id=oid, status=status,
                  filled_qty=confirmed.get("FilledQuantity"), avg_price=confirmed.get("FilledAverPrice"),
                  fee=confirmed.get("CommissionChargeValue"))
        log.info("FILLED? %s %s %s -> %s @ %s (order %s)", order.side, order.qty, order.pair, status,
                 confirmed.get("FilledAverPrice"), oid)
        if status == "PENDING":
            # Market orders should fill instantly; a pending one is unexpected. Cancel it so it
            # can't fill later behind our back; the next cycle re-plans from balances.
            try:
                self.client.cancel_order(order_id=oid)
                self._log("CANCELED_PENDING", order, order_id=oid)
            except RoostooError as exc:
                log.error("cancel of pending order %s failed: %s", oid, exc)
            return None
        return confirmed if status == "FILLED" or float(confirmed.get("FilledQuantity") or 0) > 0 else None

    def _reconcile(self, order: Order, sent_ms: int) -> Optional[dict]:
        try:
            recent = self.client.query_order(pair=order.pair, limit=20)
        except RoostooError as exc:
            log.error("reconcile query failed: %s; assuming unknown, will re-plan from balances", exc)
            self._log("RECONCILE_FAILED", order, detail=str(exc))
            return None
        for o in recent:
            if o.get("Side") == order.side and int(o.get("CreateTimestamp", 0)) >= sent_ms - 5000:
                self._log("RECONCILED_FOUND", order, detail=o, order_id=o.get("OrderID"),
                          status=o.get("Status"), filled_qty=o.get("FilledQuantity"),
                          avg_price=o.get("FilledAverPrice"))
                log.warning("reconcile: order was placed after all (id %s, %s)", o.get("OrderID"),
                            o.get("Status"))
                return o if o.get("Status") == "FILLED" else None
        self._log("RECONCILED_NOT_FOUND", order)
        log.warning("reconcile: no matching order; it was not placed")
        return None
