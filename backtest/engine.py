"""Multi-asset, long-only, hourly backtester.

Replaces the legacy single-asset backtester (see backtest/legacy_backtest.py
and the audit in docs/strategy_draft.md). Rules:

  * Weights computed from data up to the close of bar t are executed at the
    OPEN of bar t+1 (no lookahead).
  * Every fill pays the taker fee plus half the spread on its notional
    (half_spread can be one number or a per-coin mapping).
  * Negative weights are Roostoo-style 1x shorts: USD collateral is locked,
    the position is worth collateral + qty*(entry - price), floored at 0.
  * Exposure-reducing trades (sells, covers) execute before exposure-adding
    ones (buys, new shorts); both are limited to available cash (no leverage).
  * Trade decisions use strategy.rebalance_trades, the same rule the live bot uses.
  * Equity is marked at each bar's close.
  * Optional risk overlay (risk.RiskManager, the same class the live bot uses)
    adjusts each bar's target after the close, before the next open.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

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
        initial: float = 100_000.0, fee: float = TAKER_FEE, half_spread=HALF_SPREAD,
        band: float = strategy.REBALANCE_BAND, risk=None,
        daily_vol: Optional[pd.DataFrame] = None) -> Result:
    coins = list(closes.columns)
    o = opens[coins].to_numpy()
    c = closes[coins].to_numpy()
    tw = target[coins].reindex(closes.index).fillna(0.0).to_numpy()
    idx = closes.index
    n = len(idx)

    cash = initial
    qty = np.zeros(len(coins))      # long quantity
    sq = np.zeros(len(coins))       # short quantity (Roostoo 1x short: collateral = qty * entry)
    se = np.zeros(len(coins))       # short average entry price
    equity = np.empty(n)
    held_w = np.zeros((n, len(coins)))
    fees_paid = 0.0
    trades = []
    if isinstance(half_spread, (int, float)):
        costs = np.full(len(coins), fee + half_spread)
    else:  # per-coin mapping; unknown coins get the default
        costs = np.array([fee + float(half_spread.get(c, HALF_SPREAD)) for c in coins])
    adj = tw.copy()  # target actually executed at the next open (after risk overlay)
    dv = daily_vol[coins].reindex(idx).to_numpy() if daily_vol is not None else None

    def short_values(p: np.ndarray) -> np.ndarray:
        """Collateral + PnL, floored at 0 (a short never loses more than its collateral)."""
        return np.where(sq > 0, np.maximum(sq * (2 * se - p), 0.0), 0.0)

    for t in range(n):
        if t > 0:
            # Execute the previous bar's target at this bar's open.
            px = o[t]
            valid = ~np.isnan(px)
            pxv = np.where(valid, px, 0.0)
            sval = short_values(np.where(valid, px, se))
            gone = (sq > 0) & (sval <= 0)
            if gone.any():   # gapped past 2x entry since the last close: collateral lost
                for i in np.where(gone)[0]:
                    trades.append((idx[t], coins[i], "SHORT_WIPED", sq[i], px[i], 0.0))
                sq[gone], se[gone] = 0.0, 0.0
            eq_open = cash + float(np.sum(qty * pxv)) + float(sval.sum())
            cur = {}
            for i in range(len(coins)):
                if valid[i] and qty[i] > 0:
                    cur[coins[i]] = qty[i] * px[i] / eq_open
                elif valid[i] and sq[i] > 0:
                    cur[coins[i]] = -sval[i] / eq_open
            tgt = {coins[i]: adj[t - 1, i] for i in range(len(coins)) if valid[i] and adj[t - 1, i] != 0}
            orders = strategy.rebalance_trades(cur, tgt, band)
            # Phase 1: reduce exposure (sell longs, cover shorts) to free cash.
            for coin, w in orders.items():
                i = coins.index(coin)
                if not valid[i]:
                    continue
                cw = cur.get(coin, 0.0)
                if cw > 0 and w < cw:
                    sell_qty = min(qty[i], (qty[i] * px[i] - max(w, 0.0) * eq_open) / px[i])
                    gross = sell_qty * px[i]
                    f = gross * costs[i]
                    cash += gross - f
                    qty[i] -= sell_qty
                    fees_paid += f
                    trades.append((idx[t], coin, "SELL", sell_qty, px[i], f))
                elif cw < 0 and w > cw:
                    keep = max(-w, 0.0) * eq_open
                    frac = 1.0 if sval[i] <= 0 else min(1.0, max(0.0, 1 - keep / sval[i]))
                    cq = sq[i] * frac
                    f = cq * px[i] * costs[i]
                    cash += max(cq * (2 * se[i] - px[i]), 0.0) - f
                    sq[i] -= cq
                    if sq[i] <= 1e-12:
                        sq[i], se[i] = 0.0, 0.0
                    fees_paid += f
                    trades.append((idx[t], coin, "COVER", cq, px[i], f))
            # Phase 2: add exposure (buy longs, open shorts), limited to available cash.
            for coin, w in orders.items():
                i = coins.index(coin)
                if not valid[i]:
                    continue
                if w > 0:
                    delta = w * eq_open - qty[i] * px[i]
                    spend = min(delta, cash / (1 + costs[i]))
                    if spend <= 0:
                        continue
                    f = spend * costs[i]
                    cash -= spend + f
                    qty[i] += spend / px[i]
                    fees_paid += f
                    trades.append((idx[t], coin, "BUY", spend / px[i], px[i], f))
                elif w < 0:
                    now = max(sq[i] * (2 * se[i] - px[i]), 0.0) if sq[i] > 0 else 0.0
                    collat = min(-w * eq_open - now, cash / (1 + costs[i]))
                    if collat <= 0:
                        continue
                    q = collat / px[i]
                    f = collat * costs[i]
                    se[i] = (sq[i] * se[i] + q * px[i]) / (sq[i] + q)
                    sq[i] += q
                    cash -= collat + f
                    fees_paid += f
                    trades.append((idx[t], coin, "SHORT", q, px[i], f))
        mark = np.where(np.isnan(c[t]), 0.0, c[t])
        sval = short_values(np.where(np.isnan(c[t]), se, c[t]))
        wiped = (sq > 0) & (sval <= 0)
        if wiped.any():          # price doubled: collateral fully lost, position gone
            for i in np.where(wiped)[0]:
                trades.append((idx[t], coins[i], "SHORT_WIPED", sq[i], c[t, i], 0.0))
            sq[wiped], se[wiped] = 0.0, 0.0
        equity[t] = cash + float(np.sum(qty * mark)) + float(sval.sum())
        held_w[t] = (qty * mark - sval) / equity[t]
        if risk is not None:
            new = risk.apply(
                idx[t], equity[t],
                {coins[i]: tw[t, i] for i in range(len(coins)) if tw[t, i] != 0},
                {coins[i]: c[t, i] for i in range(len(coins)) if not np.isnan(c[t, i])},
                {coins[i]: qty[i] for i in range(len(coins)) if qty[i] > 0},
                {coins[i]: dv[t, i] for i in range(len(coins))} if dv is not None else {},
            )
            adj[t] = [new.get(coin, 0.0) for coin in coins]

    return Result(
        equity=pd.Series(equity, index=idx, name="equity"),
        weights=pd.DataFrame(held_w, index=idx, columns=coins),
        trades=pd.DataFrame(trades, columns=["time", "coin", "side", "qty", "price", "fee"]),
        fees=fees_paid,
        extra={"risk_events": list(risk.events) if risk is not None else []},
    )
