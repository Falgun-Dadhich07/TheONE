"""Multi-asset, long-only, hourly backtester.

Replaces the legacy single-asset backtester (see backtest/legacy_backtest.py
and the audit in docs/strategy_draft.md). Rules:

  * Weights computed from data up to the close of bar t are executed at the
    OPEN of bar t+1 (no lookahead).
  * Every fill pays the taker fee plus half the spread on its notional.
  * Sells execute before buys; buys are limited to available cash
    (no leverage, no shorting).
  * Trade decisions use strategy.rebalance_trades, the same rule the live bot uses.
  * Equity is marked at each bar's close.
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

import strategy

TAKER_FEE = 0.001      # 0.1% per side (verified on Roostoo)
HALF_SPREAD = 0.0002   # 2 bp per side; measured Roostoo spreads are <= 1 bp total


@dataclass
class Result:
    equity: pd.Series                 # marked at each bar close
    weights: pd.DataFrame             # actual held weights at each close
    trades: pd.DataFrame              # one row per fill
    fees: float
    extra: Dict = field(default_factory=dict)


def run(opens: pd.DataFrame, closes: pd.DataFrame, target: pd.DataFrame,
        initial: float = 100_000.0, fee: float = TAKER_FEE, half_spread: float = HALF_SPREAD,
        band: float = strategy.REBALANCE_BAND) -> Result:
    coins = list(closes.columns)
    o = opens[coins].to_numpy()
    c = closes[coins].to_numpy()
    tw = target[coins].reindex(closes.index).fillna(0.0).to_numpy()
    idx = closes.index
    n = len(idx)

    cash = initial
    qty = np.zeros(len(coins))
    equity = np.empty(n)
    held_w = np.zeros((n, len(coins)))
    fees_paid = 0.0
    trades = []
    cost = fee + half_spread

    for t in range(n):
        if t > 0:
            # Execute the previous bar's target at this bar's open.
            px = o[t]
            valid = ~np.isnan(px)
            eq_open = cash + np.nansum(qty * np.where(valid, px, 0.0))
            cur = {coins[i]: qty[i] * px[i] / eq_open for i in range(len(coins)) if valid[i] and qty[i] > 0}
            tgt = {coins[i]: tw[t - 1, i] for i in range(len(coins)) if valid[i] and tw[t - 1, i] > 0}
            orders = strategy.rebalance_trades(cur, tgt, band)
            # sells first
            for coin, w in sorted(orders.items(), key=lambda kv: kv[1] - cur.get(kv[0], 0.0)):
                i = coins.index(coin)
                if not valid[i]:
                    continue
                delta_notional = w * eq_open - qty[i] * px[i]
                if delta_notional < 0:
                    sell_qty = min(qty[i], -delta_notional / px[i])
                    gross = sell_qty * px[i]
                    f = gross * cost
                    cash += gross - f
                    qty[i] -= sell_qty
                    fees_paid += f
                    trades.append((idx[t], coin, "SELL", sell_qty, px[i], f))
                elif delta_notional > 0:
                    spend = min(delta_notional, cash / (1 + cost))
                    if spend <= 0:
                        continue
                    f = spend * cost
                    cash -= spend + f
                    qty[i] += spend / px[i]
                    fees_paid += f
                    trades.append((idx[t], coin, "BUY", spend / px[i], px[i], f))
        mark = np.where(np.isnan(c[t]), 0.0, c[t])
        equity[t] = cash + float(np.sum(qty * mark))
        held_w[t] = qty * mark / equity[t]

    return Result(
        equity=pd.Series(equity, index=idx, name="equity"),
        weights=pd.DataFrame(held_w, index=idx, columns=coins),
        trades=pd.DataFrame(trades, columns=["time", "coin", "side", "qty", "price", "fee"]),
        fees=fees_paid,
    )
