"""TEST ACCOUNT ONLY: verify Roostoo short mechanics before the bot relies on them.

Opens a tiny short, prints the wallet and short positions, closes it, prints
again. Answers: does collateral leave USD Free? does it show in USD Lock or
ShortCollateral? what fields do the responses carry?

Run:  python probe_shorts.py            (refuses competition mode)
"""
import json
import sys

import config
from roostoo_client import RoostooClient

PAIR, COLLATERAL = "BTC/USD", "20"


def show(client, label):
    wallet = client.balance()
    pos = client.short_positions()
    print(f"\n--- {label} ---")
    print("USD wallet:", wallet.get("USD"))
    print("BTC wallet:", wallet.get("BTC"))
    print("short positions:", json.dumps(pos, indent=1))


def main() -> None:
    if config.MODE != "test":
        sys.exit("Refusing: probe_shorts.py only runs with ROOSTOO_MODE=test.")
    c = RoostooClient()
    show(c, "before")
    opened = c.short_open(PAIR, COLLATERAL)
    print("\nshort_open response:", json.dumps(opened, indent=1))
    show(c, "after open")
    closed = c.short_close(PAIR)
    print("\nshort_close response:", json.dumps(closed, indent=1))
    show(c, "after close")
    print("\norder history (last 3):", json.dumps(c.query_order(pair=PAIR, limit=3), indent=1))


if __name__ == "__main__":
    main()
