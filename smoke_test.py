"""Brick 1 check: confirms signing, connectivity and market data all work.

Run:   python smoke_test.py
Trade: python smoke_test.py --trade     (test mode only: tiny BUY then SELL)
"""
import argparse
import logging

import config
from roostoo_client import RoostooClient, RoostooError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade", action="store_true", help="place a tiny BUY then SELL (test mode only)")
    parser.add_argument("--pair", default="BNB/USD")
    parser.add_argument("--qty", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Mode: {config.MODE} | Base URL: {config.BASE_URL}")
    client = RoostooClient()

    print("\n1) Server time:", client.server_time())

    info = client.exchange_info()
    pairs = info.get("TradePairs", {})
    print(f"\n2) Exchange running: {info.get('IsRunning')} | {len(pairs)} pairs")
    print("   Sample:", list(pairs)[:10])
    if args.pair in pairs:
        p = pairs[args.pair]
        print(f"   {args.pair}: price precision={p['PricePrecision']}, "
              f"amount precision={p['AmountPrecision']}, min order={p['MiniOrder']}")

    tick = client.ticker(args.pair)
    print(f"\n3) Ticker {args.pair}:", tick.get(args.pair))

    wallet = client.balance()
    nonzero = {k: v for k, v in wallet.items() if v.get("Free") or v.get("Lock")}
    print("\n4) Balance (non-zero):", nonzero)

    if args.trade:
        if config.MODE != "test":
            raise SystemExit("Refusing to trade from the smoke test in competition mode.")
        print(f"\n5) Round trip {args.qty} {args.pair} (test mode)")
        try:
            buy = client.place_order(args.pair, "BUY", args.qty)
            print("   BUY :", buy["Status"], buy["Role"], "avg", buy["FilledAverPrice"],
                  "fee", buy["CommissionChargeValue"])
            sell = client.place_order(args.pair, "SELL", args.qty)
            print("   SELL:", sell["Status"], sell["Role"], "avg", sell["FilledAverPrice"],
                  "fee", sell["CommissionChargeValue"])
        except RoostooError as exc:
            print("   Order failed:", exc)

    print("\nAll checks finished.")


if __name__ == "__main__":
    main()
