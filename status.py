"""Read-only P&L check: how much is the account worth right now?

Run:  python status.py
It never places orders. Each run appends one row to logs/status.csv,
so over time that file becomes your own equity curve.
The bot writes its own logs/equity.csv; this script uses logs/status.csv.
Do not run it against the competition account by hand (no manual API calls).
"""
import csv
import os
from datetime import datetime, timezone

import config
from roostoo_client import RoostooClient


def main() -> None:
    client = RoostooClient()

    info = client.exchange_info()
    initial = float(os.getenv("ROOSTOO_INITIAL_USD")
                    or info.get("InitialWallet", {}).get("USD", 0))

    wallet = client.balance()
    tickers = client.ticker()  # all pairs in one call

    total = 0.0
    rows = []
    for asset, amounts in wallet.items():
        qty = float(amounts.get("Free", 0)) + float(amounts.get("Lock", 0))
        if asset == "USD":
            qty += float(amounts.get("ShortCollateral", 0))
            value = qty
        else:
            price = tickers.get(f"{asset}/USD", {}).get("LastPrice", 0)
            value = qty * float(price)
        if qty:
            rows.append((asset, qty, value))
            total += value

    print(f"Mode: {config.MODE}")
    for asset, qty, value in sorted(rows, key=lambda r: -r[2]):
        print(f"  {asset:<8} qty={qty:<16g} value=${value:,.2f}")
    print(f"\nPortfolio value: ${total:,.2f}")
    if initial:
        pnl = total - initial
        print(f"Started with:    ${initial:,.2f}")
        print(f"P&L:             ${pnl:,.2f} ({pnl / initial:+.2%})")

    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "status.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["utc_time", "mode", "portfolio_value", "initial"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    config.MODE, round(total, 2), initial])


if __name__ == "__main__":
    main()