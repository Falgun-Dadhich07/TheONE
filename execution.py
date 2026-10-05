"""Turning target weights into safe Roostoo orders.

Pure helpers (unit-tested): portfolio valuation, quantity rounding, order
planning. The Executor sends orders and logs every request and response to
logs/orders.csv.

Order safety rules:
  * place_order / short_open / short_close are never retried automatically
    (a blind retry could duplicate an order).
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


REDUCE_SIDES = ("SELL", "SHORT_CLOSE")


class ShortsNotAllowed(Exception):
    """The exchange refused short selling for this account/competition."""


@dataclass
class Order:
    pair: str
    side: str            # BUY / SELL / SHORT_OPEN / SHORT_CLOSE
    qty: str             # base quantity as sent ("" = close the whole short)
    price: float         # reference price used for sizing
    reason: str
    collateral: str = ""     # SHORT_OPEN only: USD collateral as sent
    value: float = 0.0       # SHORT_CLOSE: USD value being closed (for cash planning)

    @property
    def notional(self) -> float:
        if self.side == "SHORT_OPEN":
            return float(self.collateral)
        if self.side == "SHORT_CLOSE" and not self.qty:
            return self.value
        return float(self.qty) * self.price


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def round_down(qty: float, precision: int) -> Decimal:
    """Round a quantity DOWN to `precision` decimals, via str() to avoid float artifacts."""
    q = Decimal(str(qty))
    step = Decimal(1).scaleb(-precision)        # 10**-precision
    return q.quantize(step, rounding=ROUND_DOWN) if q > 0 else Decimal(0).quantize(step)


def value_portfolio(wallet: dict, prices: Dict[str, float], quote: str = "USD",
                    shorts: Optional[List[dict]] = None) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Return (equity, total qty per coin, free qty per coin) from a balance wallet.

    Verified on the test account (probe_shorts.py): short collateral sits in
    USD Lock (and is mirrored in ShortCollateral), so a short adds only its
    UnrealizedPNL here; adding PositionValue would count the collateral twice.
    """
    equity = sum(float(p.get("UnrealizedPNL", 0) or 0) for p in (shorts or []))
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


def short_book(shorts: Optional[List[dict]], quote: str = "USD") -> Dict[str, dict]:
    """{coin: {"qty", "value"}} from /v6/short_positions (value = collateral + PnL)."""
    out = {}
    for p in shorts or []:
        coin = p["Pair"].split("/")[0] if p.get("Pair", "").endswith("/" + quote) else None
        if coin:
            out[coin] = {"qty": float(p.get("ShortQty", 0) or 0), "value": float(p.get("PositionValue", 0) or 0)}
    return out


def current_weights(total: Dict[str, float], prices: Dict[str, float], equity: float,
                    dust_usd: float = settings.DUST_USD, shorts: Optional[Dict[str, dict]] = None
                    ) -> Dict[str, float]:
    """Signed weights: longs positive, shorts negative (by PositionValue)."""
    if equity <= 0:
        return {}
    out = {}
    for coin, qty in total.items():
        v = qty * prices.get(coin, 0.0)
        if v >= dust_usd:
            out[coin] = v / equity
    for coin, sh in (shorts or {}).items():
        if sh["value"] >= dust_usd:
            out[coin] = out.get(coin, 0.0) - sh["value"] / equity
    return out


def plan_orders(target: Dict[str, float], total: Dict[str, float], free: Dict[str, float],
                prices: Dict[str, float], equity: float, rules: Dict[str, PairRule],
                band: float = settings.REBALANCE_BAND, quote: str = "USD",
                min_order_usd: float = settings.MIN_ORDER_USD,
                shorts: Optional[Dict[str, dict]] = None) -> Tuple[List[Order], List[str]]:
    """Orders that move the portfolio toward signed `target` weights.

    Exposure-reducing orders (SELL, SHORT_CLOSE) come first, then BUY and
    SHORT_OPEN. Also returns human-readable skip reasons.
    """
    shorts = shorts or {}
    cur = current_weights(total, prices, equity, shorts=shorts)
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
        c = cur.get(coin, 0.0)
        floor = max(min_order_usd, rule.mini_order)
        long_val = total.get(coin, 0.0) * px
        sh = shorts.get(coin, {"qty": 0.0, "value": 0.0})
        tag = f"{c:+.1%} -> {tgt:+.1%}"

        # 1) reduce
        if c > 0 and tgt < c:
            if tgt <= 0:
                qty = round_down(free.get(coin, 0.0), rule.amount_precision)
                reason = f"exit long {tag}"
            else:
                qty = round_down(min((long_val - tgt * equity) / px, free.get(coin, 0.0)), rule.amount_precision)
                reason = f"trim long {tag}"
            if qty > 0 and float(qty) * px > floor:
                orders.append(Order(pair, "SELL", str(qty), px, reason))
            else:
                skips.append(f"{pair}: SELL {qty} (${float(qty) * px:,.2f}) below minimum ${floor:,.2f} [{reason}]")
        elif c < 0 and tgt > c:
            if tgt >= 0:
                orders.append(Order(pair, "SHORT_CLOSE", "", px, f"cover short {tag}", value=sh["value"]))
            else:
                keep = -tgt * equity
                frac = 1 - keep / sh["value"] if sh["value"] > 0 else 1.0
                qty = round_down(sh["qty"] * frac, rule.amount_precision)
                if qty > 0 and float(qty) * px > floor:
                    orders.append(Order(pair, "SHORT_CLOSE", str(qty), px, f"trim short {tag}",
                                        value=sh["value"] * frac))
                else:
                    skips.append(f"{pair}: SHORT_CLOSE {qty} below minimum ${floor:,.2f} [trim short {tag}]")

        # 2) add
        if tgt > 0 and tgt > max(c, 0.0):
            qty = round_down((tgt * equity - (long_val if c > 0 else 0.0)) / px, rule.amount_precision)
            reason = f"{'enter' if c <= 0 else 'add'} long {tag}"
            if qty > 0 and float(qty) * px > floor:
                orders.append(Order(pair, "BUY", str(qty), px, reason))
            else:
                skips.append(f"{pair}: BUY {qty} (${float(qty) * px:,.2f}) below minimum ${floor:,.2f} [{reason}]")
        elif tgt < 0 and tgt < min(c, 0.0):
            collat = round_down(-tgt * equity - (sh["value"] if c < 0 else 0.0), 2)
            reason = f"{'open' if c >= 0 else 'add'} short {tag}"
            if float(collat) > floor and float(collat) / px >= 10 ** -rule.amount_precision:
                orders.append(Order(pair, "SHORT_OPEN", "", px, reason, collateral=str(collat)))
            else:
                skips.append(f"{pair}: SHORT_OPEN ${collat} below minimum ${floor:,.2f} [{reason}]")
    orders.sort(key=lambda o: 0 if o.side in REDUCE_SIDES else 1)
    return orders, skips


def cap_buys_to_cash(orders: List[Order], usd_free: float, rules: Dict[str, PairRule],
                     fee: float = 0.001, min_order_usd: float = settings.MIN_ORDER_USD
                     ) -> Tuple[List[Order], List[str]]:
    """Scale BUY and SHORT_OPEN down so they fit the cash expected after reductions (no leverage)."""
    cash = usd_free + sum(o.notional * (1 - fee) for o in orders if o.side in REDUCE_SIDES)
    adds = [o for o in orders if o.side not in REDUCE_SIDES]
    need = sum(o.notional * (1 + fee) for o in adds)
    if need <= cash * 0.995 or not adds:
        return orders, []
    scale = cash * 0.995 / need
    out, skips = [o for o in orders if o.side in REDUCE_SIDES], []
    for o in adds:
        rule = rules[o.pair]
        floor = max(min_order_usd, rule.mini_order)
        if o.side == "BUY":
            qty = round_down(float(o.qty) * scale, rule.amount_precision)
            if float(qty) * o.price <= floor:
                skips.append(f"{o.pair}: BUY too small after cash cap")
                continue
            out.append(Order(o.pair, o.side, str(qty), o.price, o.reason + f" (scaled x{scale:.2f} to cash)"))
        else:
            collat = round_down(float(o.collateral) * scale, 2)
            if float(collat) <= floor:
                skips.append(f"{o.pair}: SHORT_OPEN too small after cash cap")
                continue
            out.append(Order(o.pair, o.side, "", o.price, o.reason + f" (scaled x{scale:.2f} to cash)",
                             collateral=str(collat)))
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
        if order.side in ("SHORT_OPEN", "SHORT_CLOSE"):
            return self._execute_short(order)
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

    # ------------------------------------------------------------------
    def _execute_short(self, order: Order) -> Optional[dict]:
        """Open or close a 1x short. Never retried; an ambiguous failure is
        resolved by the next cycle, which re-plans from /v6/short_positions."""
        self._log("REQUEST", order, detail={"collateral": order.collateral} if order.collateral else None)
        if self.dry_run:
            log.info("DRY_RUN %s %s %s (~$%.2f) %s", order.side, order.collateral or order.qty or "all",
                     order.pair, order.notional, order.reason)
            self._log("DRY_RUN", order, status="FILLED")
            return {"Status": "FILLED"}
        try:
            if order.side == "SHORT_OPEN":
                resp = self.client.short_open(order.pair, order.collateral)
                self._log("RESPONSE", order, detail=resp, order_id=resp.get("ID"), status=resp.get("Status"),
                          filled_qty=resp.get("ShortQty"), avg_price=resp.get("EntryPrice"), fee=resp.get("OpenFee"))
                ok = resp.get("Status") == "OPEN"
            else:
                resp = self.client.short_close(order.pair, order.qty or None)
                self._log("RESPONSE", order, detail=resp, status="FILLED", filled_qty=resp.get("ClosedQty"),
                          avg_price=resp.get("ClosePrice"), fee=resp.get("CloseFee"))
                ok = float(resp.get("ClosedQty", 0) or 0) > 0
        except RoostooError as exc:
            msg = str(exc)
            if "not allow short" in msg:
                self._log("SHORTS_NOT_ALLOWED", order, detail=msg)
                log.error("Exchange refuses shorts: %s. Switching to long-only.", msg)
                raise ShortsNotAllowed(msg) from exc
            event = "REJECTED" if "rejected" in msg else "AMBIGUOUS"
            self._log(event, order, detail=msg)
            log.error("%s %s %s: %s (not retried; next cycle re-plans from real positions)", event,
                      order.side, order.pair, msg)
            return None
        log.info("%s %s %s -> %s", order.side, order.pair, order.collateral or order.qty or "all",
                 "OK" if ok else "NOT FILLED")
        return {"Status": "FILLED"} if ok else None
