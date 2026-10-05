"""TEST ACCOUNT ONLY: how does Roostoo fill LIMIT orders, and what do they cost?

1. Passive limit buy a few bps below the bid: when does it fill (touch of
   last price? of the ask?), at what price, with what fee/role?
2. Marketable limit buy above the ask: taker or maker? price improvement?
3. Flatten whatever was bought (passive limit sell, then market for the rest).

Run:  python probe_limits.py      (refuses competition mode)
"""
import json
import sys
import time
from decimal import ROUND_DOWN, Decimal

import config
from roostoo_client import RoostooClient, RoostooError

PAIR, QTY = "BTC/USD", "0.0002"
WAIT_S = 240


def px(x: float) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_DOWN))


def tick(c):
    t = c.ticker(PAIR)[PAIR]
    return float(t["MaxBid"]), float(t["MinAsk"]), float(t["LastPrice"])


def brief(o):
    keys = ("OrderID", "Status", "Role", "Side", "Type", "Price", "Quantity", "FilledQuantity",
            "FilledAverPrice", "CommissionChargeValue", "CommissionPercent", "CommissionCoin")
    return {k: o.get(k) for k in keys}


def wait_fill(c, oid, limit, side):
    t0 = time.time()
    lo, hi = 1e18, 0.0
    while time.time() - t0 < WAIT_S:
        bid, ask, last = tick(c)
        lo, hi = min(lo, bid, last), max(hi, ask, last)
        o = c.query_order(order_id=oid)[0]
        if o["Status"] != "PENDING":
            print(f"  -> {o['Status']} after {time.time() - t0:.0f}s | bid {bid} ask {ask} last {last} "
                  f"| limit {limit} | range seen: low {lo} high {hi}")
            print("  ", json.dumps(brief(o)))
            return o
    print(f"  -> still PENDING after {WAIT_S}s (limit {limit}; range seen low {lo} high {hi}); cancelling")
    print("   cancel:", c.cancel_order(order_id=oid))
    return None


def main() -> None:
    if config.MODE != "test":
        sys.exit("Refusing: probe_limits.py only runs with ROOSTOO_MODE=test.")
    c = RoostooClient(min_interval=2.1)
    held = 0.0

    bid, ask, last = tick(c)
    print(f"market: bid {bid} ask {ask} last {last} spread {(ask - bid) / bid * 1e4:.2f} bps")

    print("\n[1] passive LIMIT BUY 2 bps below the bid")
    limit = px(bid * (1 - 0.0002))
    o = c.place_order(PAIR, "BUY", QTY, "LIMIT", limit)
    print("   placed:", json.dumps(brief(o)))
    f = wait_fill(c, o["OrderID"], limit, "BUY") if o["Status"] == "PENDING" else o
    if f and f.get("Status") == "FILLED":
        held += float(f["FilledQuantity"])

    print("\n[2] marketable LIMIT BUY 5 bps above the ask")
    bid, ask, last = tick(c)
    limit = px(ask * 1.0005)
    o = c.place_order(PAIR, "BUY", QTY, "LIMIT", limit)
    print(f"   market at placement: bid {bid} ask {ask} | limit {limit}")
    print("   placed:", json.dumps(brief(o)))
    f = wait_fill(c, o["OrderID"], limit, "BUY") if o["Status"] == "PENDING" else o
    if f and f.get("Status") == "FILLED":
        held += float(f["FilledQuantity"])

    if held > 0:
        print(f"\n[3] passive LIMIT SELL of {held} BTC 2 bps above the ask")
        bid, ask, last = tick(c)
        limit = px(ask * 1.0002)
        o = c.place_order(PAIR, "SELL", f"{held:.5f}", "LIMIT", limit)
        print("   placed:", json.dumps(brief(o)))
        f = wait_fill(c, o["OrderID"], limit, "SELL") if o["Status"] == "PENDING" else o
        if not (f and f.get("Status") == "FILLED"):
            try:
                m = c.place_order(PAIR, "SELL", f"{held:.5f}")
                print("   flattened at market:", json.dumps(brief(m)))
            except RoostooError as exc:
                print("   market flatten failed:", exc)
    w = c.balance()
    print("\nfinal BTC:", w.get("BTC"), "| USD:", w.get("USD"))
    print("pending:", c.pending_count())


if __name__ == "__main__":
    main()
