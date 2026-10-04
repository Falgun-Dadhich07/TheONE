# Strategy draft v0: crypto analysis and first strategy

Status: **draft, not backtested properly yet.** Every number below is gross or roughly costed in-sample research, not a validated backtest.

## 1. Data and method

- 66 Roostoo crypto pairs (21 tokenized stocks and PAXG excluded). Hourly Binance `<COIN>USDT` klines, Oct 2024 to 4 Oct 2026, downloaded by `research/fetch_data.py`. Roostoo prices are streamed from Binance (official FAQ), so this is the closest proxy for our fills.
- TON and OMNI have no recent Binance spot data, so they are excluded: we can't model what Roostoo prices them at.
- Predictive statistics use data **before 2026-07-05 only**. The last 3 months are kept untouched for final validation.
- Reproduce: `python research/fetch_data.py` then `python research/analyze.py`. Tables land in `data/analysis/`.

## 2. What the data says

| Finding | Evidence | Consequence |
|---|---|---|
| The last year was a bear market | BTC −30%, ETH −40%, SOL −46%; most alts −50 to −90%. An equal-weight basket of liquid coins lost 19%/yr with a −75% max drawdown | Long-only must spend a lot of time in cash. Cash is a position. |
| 14 days of buy & hold is a coin flip with big drawdowns | BTC: 45% of 14-day windows lose money, median drawdown −7.6%, worst −31%. Alts: median drawdown −15% to −27% | Fully invested, an 8% kill switch would fire constantly. Exposure must be well below 100%. |
| Hourly signals are noise | Hourly autocorrelation ≈ 0, variance ratios ≈ 1. Hour-of-day effects: no \|t\| > 2.1 across 24 tests | No intraday timing edge to pay a 0.2% round trip for. Trade slowly. |
| Short-horizon momentum doesn't work | Past 6h and 72h returns slightly *predict reversals* (median IC −0.03 and −0.05) | Avoid short lookbacks. |
| The EMA24/96 trend filter (the Phase A baseline idea) has no edge on average | Next-24h return when EMA24 > EMA96: −0.08%; when below: −0.15%. ETH and BNB do *worse* when "up" | Its only benefit is fewer hours exposed. Not worth keeping as the core signal. |
| Cross-sectional momentum over 14 days is the strongest signal found | Top-3 coins by 14-day return beat the universe by +0.45%/day gross (24h hold) | It is fragile and tail-driven: hit rate under 50%, mean IC ≈ 0. Returns come from a few big runs. |
| …and much of it is one coin | Liquid set: 266%/yr. **Without ZEC (+832% in a year): 124%/yr.** Drawdowns −53% to −70% fully invested | Picking the universe by *current* volume is a selection bias. Use a fixed list of large coins chosen up front. |
| A BTC trend filter cuts drawdowns | Large coins, 14-day lookback, BTC-above-20-day-average filter: max drawdown −46% vs −69% unfiltered | Keep the regime filter. |
| Coins move together | Average pairwise hourly correlation 0.49 | Diversification is limited: in a crash, everything falls together. |

Lowest-risk coins: TRX, BTC, BNB (median 14-day drawdown −6% to −9%). TRX is low volatility but barely correlated with the rest (0.37), and sometimes jumps on its own news.

## 3. Draft strategy: regime-gated momentum rotation

Long-only. Decisions on hourly bar close; the portfolio is rebalanced once a day.

- **Universe (fixed up front, not tuned):** BTC, ETH, SOL, BNB, XRP, DOGE. All are large and liquid with 2+ years of history.
- **Regime gate:** risk-on only when BTC's close is above its 20-day (480h) simple moving average. Risk-off means 100% cash.
- **Selection:** rank the six by 14-day (336h) return. Hold the top 2, but only those with a positive 14-day return.
- **Sizing:** inverse volatility (7-day realized vol), scaled to about 25% annualized target portfolio vol. Caps: 25% of equity per coin, 50% total.
- **Rebalance:** once a day at 00:00 UTC. Skip a change unless the weight moves by more than 5 percentage points, to save fees. The regime gate is checked every hour, and a flip to risk-off exits at the next hour.
- **Safety rails:** kill switch at 8% drawdown from peak, with an automatic cooldown instead of a manual reset (the FAQ bans manual intervention). Plus the daily activity-guarantee trade.
- **Tunable parameters (3):** momentum lookback (336h), regime SMA (480h), top N (2). The vol target and caps are risk settings, fixed rather than optimized.

**What it would do today (4 Oct 16:00 UTC):** BTC at 85,374 is above its 20-day average of 82,569, so risk is on. The top two by 14-day return are SOL (+10.4%) and DOGE (+8.0%). It would buy both, sized by volatility.

Rough in-sample result (large coins, 14-day lookback, 24h hold, both filters, 0.2% cost per swap, **fully invested**): about 56%/yr, Sharpe 1.1, max drawdown −46%. At our ~30–50% exposure, drawdown should scale down to roughly −15 to −23% over two years. That's still a lot for a 14-day contest, which is why the kill switch and vol target matter.

## 4. Known weaknesses

1. **It's fragile and tail-driven.** Under half of rebalances beat the universe; profit comes from catching a few strong runs. A 14-day contest may contain none.
2. **It can sit in cash.** BTC was risk-on only 51% of the last two years (74% in the last 90 days). Cash is good for drawdown, but ranking starts with **return** (top 20 per region), so too much cash risks missing the finals.
3. **Whipsaw near the 20-day average.** BTC chopping around it causes repeated entries and exits. Needs a buffer band (e.g. ±1%) or a minimum hold time.
4. **Small sample.** Two years is mostly one regime. The 14-day/20-day settings were picked from a handful of combinations, not optimized, but still in-sample.
5. **The research shortcuts were optimistic.** The quick test entered at the signal bar's close; the real backtester must use the next bar's open.

## 5. Backtest results (proper engine, 4 Oct 2026)

Engine: `backtest/engine.py`. Fills at the next bar's open, 0.10% fee plus 0.02% half-spread per side (measured Roostoo spreads are ≤1 bp), long-only, no leverage. The same `strategy.py` functions and `rebalance_trades()` band are used by the live bot. The tests prove the strategies never look ahead (`tests/test_strategy.py`).

Universe: BTC, ETH, SOL, BNB, XRP, DOGE. In-sample is 31 Oct 2024 to 4 Jul 2026; the holdout is 5 Jul to 4 Oct 2026. The selection rule was fixed before the holdout ran: best in-sample composite among non-benchmarks with a worst 14-day drawdown better than −15%. Reproduce with `python backtest/run_compare.py`.

**In-sample (20 months), default parameters**

| Strategy | Return | Max DD | Sharpe | Sortino | Calmar | Composite | 14d windows losing | Worst 14d DD | Trades/day | Fees (% of start) | Avg exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **xs_momentum_regime (draft)** | **+9.1%** | **−22.1%** | 0.40 | 0.57 | 0.24 | **0.42** | 55% | −8.9% | 1.2 | 18.4% | 17% |
| invvol_regime | −0.1% | −22.2% | 0.08 | 0.10 | 0.00 | 0.06 | 57% | −8.1% | 2.9 | 17.0% | 21% |
| ema_trend (Phase A idea) | −6.8% | −29.9% | −0.13 | −0.18 | −0.14 | −0.15 | 63% | −7.8% | 3.1 | 26.1% | 30% |
| reversal_regime | −24.4% | −31.2% | −1.06 | −1.47 | −0.49 | −1.05 | 62% | −8.8% | 1.8 | 23.1% | 20% |
| *buy & hold BTC (benchmark)* | −12.6% | −53.7% | 0.06 | 0.08 | −0.14 | 0.01 | 49% | −30.6% | 0 | 0.1% | 100% |
| *equal weight at 50% (benchmark)* | +5.7% | −34.8% | 0.26 | 0.35 | 0.10 | 0.25 | 53% | −15.3% | 0 | 0.1% | 42% |

**Holdout (last 3 months, a rising market), run once after the selection was locked**

| Strategy | Return | Max DD | Composite | 14d windows losing | Worst 14d DD |
|---|---|---|---|---|---|
| **xs_momentum_regime (selected)** | **+8.3%** | **−5.8%** | 3.55 | 41% | −4.9% |
| invvol_regime | +8.3% | −7.7% | 2.98 | 49% | −4.8% |
| ema_trend | +1.0% | −10.4% | 0.39 | 65% | −8.0% |
| reversal_regime | −1.0% | −11.1% | −0.24 | 62% | −5.8% |
| *buy & hold BTC* | +35.4% | −7.7% | 12.52 | 32% | −7.7% |
| *equal weight at 50%* | +19.0% | −5.5% | 8.13 | 29% | −5.2% |

**Reading it honestly**

- The draft is the only active strategy that made money in both periods, and it had the smallest holdout drawdown. Its in-sample parameter grid was positive in 27 of 27 combinations, so it isn't a lucky setting.
- It **badly lags passive holding in a rising market** (+8% vs +19% to +35%), because it is only ~31% invested on average and pays fees to rotate.
- **Fees are the biggest drag:** 18% of starting capital over 20 months against a +9% net return.
- **Overfitting check:** the grid's in-sample favourite (504h lookback: +28% in-sample) made just +1.6% on the holdout. The pre-selected settings were kept.
- The EMA trend idea planned for the Phase A baseline lost money in-sample and is dropped. Mean reversion lost in both periods.
- The holdout has now been looked at, so it is no longer clean. Future changes need forward (live) evidence.

## 6. Risk management: what's in place and whether it works

Layers, from always-on to emergency:

1. **Sizing by volatility:** each coin gets `25% target vol / (n coins × its 7-day vol)`. This assumes the coins move perfectly together, which is conservative.
2. **Caps:** 25% of equity per coin, 50% in total. No leverage, no shorts.
3. **BTC filter:** all cash while BTC is below its 20-day average, checked every hour.
4. **Kill switch** (`risk.py`, shared by the bot and the backtest): if equity falls 8% below its peak, go to cash for 24h. It then resumes on its own with the peak reset, because the FAQ forbids manual resets.
5. **Trailing stop per coin** (`risk.py`): sell a coin that falls 3× its daily vol below its high since entry. **Built but turned off by default**, see below.

Tested with `python backtest/run_risk.py`, using the brief's default settings without tuning:

| Same strategy, in-sample (20 months) | Return | Max DD | Composite | Worst 14d DD | Kill switches | Stops |
|---|---|---|---|---|---|---|
| No overlay | +9.1% | −22.1% | 0.42 | −8.9% | — | — |
| **Kill switch only** | **+15.5%** | **−19.0%** | **0.68** | **−8.1%** | 7 | — |
| Trailing stop only | +8.3% | −21.6% | 0.40 | −8.9% | — | 12 |
| Kill + stop | +14.7% | −18.5% | 0.65 | −8.1% | 7 | 10 |
| Kill only, at 40% vol target | +13.2% | −29.9% | 0.47 | −11.8% | 10 | — |

On the holdout, neither rule fired at 25% vol (no 8% drawdown happened), so the results there are unchanged.

What we learned:

- **The kill switch helps:** +6 percentage points of return, 3 points less drawdown, and the composite up 60%. It caps a bad 14-day stretch at roughly 8%.
- **It doesn't stop a slow bleed.** The worst 20-month drawdown (−22%, Jul 2025 to mid-2026) was a long grind of small losses, mostly fees and whipsaws. After each 24h cooldown the peak resets, so another 8% is allowed. It fired 4 times in that stretch and saved only about 2 points.
- **The trailing stop hurts slightly.** Momentum picks often dip and then recover. It stays off.
- **The 40% volatility target isn't worth it.** Drawdowns grow (−30%) with almost no extra return, because the 50% total cap and the extra fees eat the gain.
- **The BTC filter did the heavy lifting in the worst crash.** BTC fell 20% in 72 hours (5 Feb 2026), and the strategy was already in cash.

Hourly bars mean a fast crash can overshoot the 8% before the next check. The live bot checks equity every 60 seconds, not just on the hourly close, which narrows that gap.

## 7. Next steps (pending approval)

1. Build the proper backtest engine (`backtest/engine.py`): next-bar-open fills, 0.1% fee plus half-spread, multi-asset target weights, Sharpe/Sortino/Calmar/composite, and rolling 14-day windows.
2. Backtest the draft against the alternatives in one table, losers included:
   - the Phase A EMA baseline;
   - buy & hold BTC;
   - inverse-vol majors with a regime filter;
   - mean reversion on 72h returns with the regime filter, since 72h lookbacks showed reversal.
3. Validate the winner once on the untouched holdout (5 Jul to 4 Oct 2026).
4. Put the chosen `target_weights()` into `strategy.py` and get the bot running. **Every day without trades uses up the 8-active-day buffer**, so steps 1–3 should take hours, not days.
