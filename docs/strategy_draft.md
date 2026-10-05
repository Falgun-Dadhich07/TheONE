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

## 7. Strategy lab: research-note ideas and the Hawkes strategy (`research/strategy_lab.py`)

Protocol:
- **Walk-forward:** tune with particle swarm optimisation (PSO) on 180 days, trade the next 60 days blind, roll forward 60 days. That gives 9 folds, and the out-of-sample (OOS) stretches are stitched into one run from 14 May 2025 to 4 Oct 2026.
- **Evaluation:** the real engine, with fees, spread and the 8% kill switch.
- **Overfitting check:** CSCV probability of backtest overfitting (PBO), using 40 random configurations per strategy family on pre-holdout data.

All candidates are long-only: shorts are off, and "arbitrage" is banned.

| Strategy (OOS 17 months) | Return | Max DD | Composite | Worst 14d DD | PBO |
|---|---|---|---|---|---|
| v1 fixed (current bot) | +4.8% | −18.8% | 0.31 | −8.1% | — |
| v1 re-tuned by PSO each fold | −3.4% | −19.8% | −0.13 | −7.4% | 0.26 |
| **v1 with HMM regime gate (fixed rules)** | **+6.2%** | **−15.8%** | **0.41** | **−6.3%** | — |
| Hawkes volatility (5 params, PSO each fold) | −1.2% | −24.3% | 0.00 | −7.8% | 0.21 |
| Kalman pairs, long-only (PSO) | −14.9% | −28.3% | −0.79 | −12.3% | 0.61 |
| Kalman pairs, long-only + BTC gate (PSO) | −7.2% | −19.9% | −0.49 | −6.2% | 0.62 |
| *Buy & hold BTC* | −30.0% | −59.0% | −0.52 | −26.9% | — |

Findings:
- **PSO re-tuning doesn't help.** The chosen parameters jumped every fold, often to the edge of their bounds, and a high training score said nothing about the next 60 days. Hawkes OOS folds ranged from −10.6% to +9.1%.
- **Long-only pairs mean reversion loses money.** PBO above 0.5 means it is likely overfit. A faster Kalman filter turns the z-score into white noise (lag-24 autocorrelation ≈ 0). A slower one simply gets run over by trends.
- **The HMM gate is the one research-note idea that helped.** It's a 2-state Gaussian HMM on BTC daily returns: refit monthly, filtered (causal) probabilities only, with the bull state labelled by its mean/vol ratio.

Robustness of the HMM gate on the same OOS stretch. These are variants, not tuning; the pre-registered setting was p>0.5, refit every 30 days.

| Gate | Return | Max DD | Composite | Worst 14d DD | Kill switches |
|---|---|---|---|---|---|
| BTC > SMA (current) | +4.8% | −18.8% | 0.31 | −8.1% | 4 |
| HMM p>0.4 / 0.5 / 0.6, refit 30d | −0.4 / +6.2 / +18.5% | −20.8 / −15.8 / −11.3% | 0.05 / 0.41 / 1.19 | −8.3 / −6.3 / −5.8% | 6 / 3 / 2 |
| HMM p>0.4 / 0.5 / 0.6, refit 60d | +6.1 / +11.1 / +19.0% | −17.3 / −12.2 / −11.3% | 0.38 / 0.70 / 1.21 | −8.3 / −7.5 / −5.4% | 6 / 4 / 2 |
| **HMM p>0.5 AND BTC > SMA**, refit 30d / 60d | +7.2 / +8.6% | **−9.6 / −8.5%** | 0.58 / 0.69 | −7.7 / −6.2% | 1 / 1 |

How to read this:
- Results get monotonically better as the threshold tightens, so the HMM probability seems to carry information.
- Picking p>0.6 would be choosing on this same OOS data, so don't.
- Requiring **both** the HMM and the SMA to be risk-on roughly halves max drawdown in both refit settings.
- Caveat: this is one 17-month stretch, and v1's own lookback/top-N were chosen on data overlapping it.

## 8. Portfolio construction: which coins, how many, how to size (`research/portfolio_lab.py`)

Setup: the v2 signal (14-day momentum, risk-on only if the BTC HMM bull probability is above 0.5 **and** BTC is above its 480h SMA), with a daily rebalance.

- **Universe:** either the 6 majors, or the top 20 / top 40 coins by trailing 30-day median daily volume. The liquid list is recomputed daily using past data only, and a coin needs at least 60 days of history.
- **Costs:** 0.1% fee plus each coin's **measured Roostoo half-spread**, with a 2 bp floor. Spreads range from 0 bp (BTC) to 17–25 bp (PEPE, SHIB, BONK, LISTA), because of tick size.
- **Selection rule, fixed before the run:** best in-sample composite among configurations whose worst 14-day drawdown is better than −10%.
- **Periods:** in-sample (IS) is Apr 2025 to Jul 2026, holdout (HO) is Jul to Oct 2026, and 1M is the last month.

Results from 40 configurations (full table in `data/analysis/portfolio_report.txt`):

- **Holding more coins is consistently worse.** In-sample composite, K = 2 / 5 / 10 / 15:
  - majors: 0.75 / 0.38
  - top 20: 0.99 / 0.11 / 0.26 / 0.78
  - top 40: 0.95 / 0.52 / −0.07 / −0.05

  Momentum's edge sits in the top 1–2 coins; extra names add weak signals and fees.
- **Covariance-based sizing doesn't beat simple inverse volatility.** Equal risk contribution, minimum variance and covariance-scaled inverse vol were below `invvol_c1` in nearly every row, and minimum variance was the worst. Correlations between coins are too unstable over a 30-day window.
- **A wider universe is a mixed bag.**

  | Universe (K=2, invvol_c1) | IS | HO | 1M |
  |---|---|---|---|
  | top 20 | +10.3% / −8.3% DD | +5.6% | +4.9% |
  | majors | +8.2% / −9.6% DD | +7.1% | +1.3% |
  | top 40 | — | −3.6% | — |

  The top 40 universe lost money in the holdout; small coins hurt.
- **Exposure is the real lever, but it's a risk choice.** The book is in cash most of the time, so average exposure is only about 6–10% in-sample. Raising the vol target, per-coin cap and gross cap from 25/25/50% to 40/40/80% (majors):

  | Metric | 25/25/50% | 40/40/80% |
  |---|---|---|
  | IS return | +8.2% | +16.0% |
  | IS max DD | −9.6% | −14.6% |
  | IS composite | 0.75 | 0.93 |
  | HO return | +7.1% | +10.9% |
  | Worst 14-day DD | — | −12% (worse than the 8% kill switch) |

  The results don't change in a consistent direction (vol 60 was worse), so there's no free improvement here.

## 9. With short positions (`research/short_lab.py`)

Shorts were approved for research on 2026-10-05. The engine models Roostoo's 1x collateral shorts: a 0.1% fee to open and to close, and a loss capped at the collateral.

- **Window:** the same stitched walk-forward out-of-sample (OOS) window as section 7, 14 May 2025 to 4 Oct 2026.
- **Engine:** the real one, with the kill switch and measured per-coin spreads.
- **Gate:** "risk-on" means HMM bull and BTC > SMA (23% of hours). "Risk-off" means HMM bear and BTC < SMA (13% of hours).

| Strategy | OOS return | Max DD | Composite | Worst 14d DD | 3M return | 1M return | PBO |
|---|---|---|---|---|---|---|---|
| v1 fixed (live bot) | +4.8% | −18.8% | 0.31 | −8.1% | +10.9% | −0.2% | — |
| v2 long-only, majors | +7.2% | −9.6% | 0.58 | −7.7% | +7.1% | +1.3% | — |
| v2 long/short, majors | +9.9% | −12.3% | 0.64 | −7.9% | +7.1% | +1.3% | — |
| v2 long-only, top 20 | +13.0% | −10.2% | 0.98 | −5.3% | +5.6% | +4.9% | — |
| **v2 long/short, top 20** | **+18.4%** | **−11.7%** | **1.09** | −6.7% | +5.6% | +4.9% | — |
| v2 short leg only, top 20 | +6.3% | −9.2% | 0.58 | −5.0% | 0.0% | 0.0% | — |
| Momentum long/short, no gate | −2.9% | −19.5% | −0.06 | −10.3% | +0.6% | −6.2% | — |
| Beta-neutral momentum (research note) | −13.2% | −18.5% | −1.35 | −5.2% | −2.2% | −2.1% | — |
| Beta-neutral momentum, HMM-gated | −12.6% | −13.0% | −1.72 | −3.7% | −0.5% | −1.9% | — |
| Hawkes long/short (PSO walk-forward) | +11.6% | −17.7% | 0.63 | −8.2% | +11.2% | −1.3% | 0.07 |
| Kalman pairs long/short (PSO walk-forward) | −8.0% | −15.0% | −0.81 | −5.7% | +5.7% | −0.2% | 0.02 |

Findings:

- **Shorting helps only behind the regime gate.** The v2 short leg made +6.3% on its own, active 86 of 509 days, with no shorts wiped out. Always-on long/short momentum lost money.
- **Market-neutral ideas from the research note lose money,** even with real shorts: beta-neutral momentum at −13% and pairs at −8%. Fees and trend persistence eat the edge.
- **The Hawkes strategy with real shorts is much better than long-only** (+11.6% vs −1.2%) and has a low PBO (0.07). But it is volatile: −17.7% max DD, five kill switches, and fees of 17% of capital.
- **No shorting happened in the last 3 months** (bull regime), so the 3M and 1M columns equal long-only.
- **Best risk-adjusted result: v2 long/short on top 20.** Caveats:
  - top 20 vs majors was mixed in section 8;
  - shorting altcoins carries squeeze risk;
  - the competition account may refuse shorts.

## 10. Hybrid: momentum (HMM+BTC) and Hawkes on top-20 coins (`research/hybrid_lab.py`)

Two long/short sleeves run on the causal top-20 liquidity universe:

- **MOM:** v2 long/short momentum.
- **HAWKES:** per-coin signal, using the walk-forward PSO parameters where each fold's parameters are used only in that fold's out-of-sample window.

The evaluation runs from 13 Jul 2025 to 4 Oct 2026, so the trailing look-back windows are already out-of-sample. It uses the real engine, the kill switch and measured spreads.

| Strategy | Return | Max DD | Composite | Worst 14d DD | Trades/day | Fees | 3M return | 1M return |
|---|---|---|---|---|---|---|---|---|
| MOM sleeve alone | +8.7% | −11.5% | 0.64 | −6.8% | 1.0 | 7.5% | +5.6% | +4.9% |
| HAWKES sleeve alone | +24.8% | −18.5% | 1.22 | −10.9% | 6.5 | 22.7% | +13.7% | +6.6% |
| Per-coin pick, 30-day look-back | −7.0% | −30.0% | −0.23 | −9.4% | 4.1 | 17.4% | +4.4% | −0.4% |
| Per-coin pick, 60-day look-back | +2.0% | −24.9% | 0.18 | −8.4% | 3.6 | 16.8% | +8.4% | +4.8% |
| Per-coin pick, 90-day look-back | +8.9% | −19.9% | 0.51 | −8.9% | 3.5 | 16.2% | +13.3% | +4.6% |
| **Blend 50/50 (both sleeves, half capital each)** | **+16.2%** | **−12.2%** | **1.13** | **−7.4%** | 6.3 | 12.9% | +10.0% | +4.2% |

The per-coin pick follows, for each coin, the sleeve with the better trailing Sharpe on that coin, and holds no position if neither is positive.

Findings:

- **Picking a strategy per coin from recent performance loses** against both sleeves, and gets worse with shorter look-backs. Which sleeve "works" on a coin doesn't persist: it is performance-chasing.
- **Hawkes on 20 coins is the strongest single sleeve,** but it is volatile: −18.5% max DD, a worst 14-day DD of −10.9% (beyond the 8% kill switch), and fees of 22.7% of capital.
- **The 50/50 blend gives most of Hawkes' return with momentum's drawdown.** The two sleeves make money at different times. Its composite (1.13) is close to Hawkes alone (1.22) with two-thirds of the drawdown, and its worst 14-day DD (−7.4%) stays inside the kill switch.

## 11. Higher risk for more profit (`research/aggressive_lab.py`)

The team lead asked for more profit over about 10 days, with more trades and more risk. The test grid:

- **Exposure:** base (25% vol, 25% per coin, 50% gross), ×1.5, or ×2 (50% vol, 40% per coin, 100% gross, never leverage).
- **Momentum speed:** 336h / daily, 168h / 12h, or 72h / 4h.
- **Kill switch:** 8% or 12%.

The selection rule, fixed before the run: the highest **mean 10-day return** among configs whose worst 10-day return is at least −12%.

Results (Jul 2025 to Oct 2026, production v3 code, real engine; full table in `data/analysis/aggressive_report.txt`):

- **Faster momentum is worse at every exposure level.** 72h / 4h cut return to +4% to +10%, and fees rose to 22–43% of capital. 168h / 12h was only slightly better than daily at base exposure, and worse at higher exposure. Short-horizon crypto moves are mostly noise (see section 2).
- **Exposure is what raises profit.**

| Config | Return | Max DD | Composite | Mean 10d | 10d top-decile | Best 10d | Worst 10d | P(10d > +5%) | P(10d < −5%) |
|---|---|---|---|---|---|---|---|---|---|
| Base, daily, 8% kill switch (previous v3) | +18.4% | −12.2% | 1.27 | +0.37% | +3.4% | +9.2% | −5.5% | 4% | 0.5% |
| ×1.5, daily, 8% kill switch | +30.9% | −16.9% | 1.43 | +0.62% | +5.4% | +14.9% | −7.8% | 10% | 3.2% |
| **×2, daily, 12% kill switch (rule pick, deployed)** | **+37.4%** | **−21.1%** | **1.38** | **+0.73%** | **+6.8%** | **+17.9%** | **−10.4%** | **16%** | **6.8%** |

- **The pick is profitable in both halves** of the period (+11.8%, +22.9%).
- **Its risks:**
  - a worst 14-day drawdown of −14%, because fast moves overshoot the 12% kill switch within the hour;
  - fees of about 29% of capital over 15 months;
  - Hawkes alone at ×2 is poor (+10.8%, −31.5% DD), so the 50/50 blend matters even more;
  - the long-only fallback is weak (+9.6%, −26.8% DD).

## 12. Faster trading and limit orders (after the organizers' note on scalping, 2026-10-05)

The organizers said tick scalping is not prohibited, but that finalists are judged on whether the strategy "would actually hold up in real markets", and that an edge from simulator mechanics is not high value. So faster strategies were tested only with fills a real market would give.

**Roostoo limit-order mechanics** (`probe_limits.py`, test account):
- A resting limit pays a **0.05%** maker fee and fills at the limit price when the market reaches it.
- A limit priced through the market fills at once at the market price as a taker (0.10%).
- A resting buy below a rising market simply never fills.

**Intraday backtests** (`research/intraday_lab.py`):
- Data: 5-minute Binance data for 25 liquid coins, Sep 2025 to Oct 2026, on 15-minute bars.
- Protocol: walk-forward, train 90 days / test 30 days, at the deployed risk settings.
- Two execution models: taker, and "maker". The maker model rests a limit for one bar. It fills only if the price trades *through* the limit by 1 bp (90% of the time), otherwise it chases as a taker at the bar's close.

| Strategy (out-of-sample, 30 Dec 2025 to 5 Oct 2026) | Taker | Maker | Max DD | Note |
|---|---|---|---|---|
| Hawkes on 15m bars | −27.1% | +2.4% | −18% to −38% | Fees 22–60% of capital |
| Channel breakout, no gate | +30.6% | +36.5% | −19% | Only slow channels profit |
| Reversal (fade N-bar z-score) | −44.9% | −43.1% | −54% | |
| Channel breakout, regime-gated, real engine | **+9.9%** | +15.7% (fast eval) | −21.9% | See the bug note below |
| **Deployed v3, same dates, real engine** | **+28.9%** | — | −21.1% | Composite 1.71 |

**Bug found and fixed.** The first run reported the gated breakout at +77% (taker) and +92% (maker). The BTC > SMA gate had been built from hourly closes labeled at the hour's start and forward-filled onto 15-minute bars, so the gate saw up to 45 minutes ahead. `research/breakout_check.py` rebuilds it causally: the return falls to +5% in the fast evaluator and +9.9% in the real engine. The v3 production code never had this issue, because its hourly gate is used at bar close and is covered by causality tests.

Further checks in `breakout_check.py`:
- Only 9 of 24 channel lengths were profitable, and all of those were 2 days or longer.
- Every channel under a day lost 20–95%.
- Leave-one-coin-out results range from +4% (without WLD) to +28% (without ZEC).

**Slow trend sleeve next to v3** (`research/trend_sleeve.py`, hourly, 2 years, real engine):
- A regime-gated channel breakout with a walk-forward channel of 48–336h made +2.9% with a −26% max drawdown on its own.
- Every blend with v3 was worse than v3 alone (v3 +37.1%; blends +1.9% to +23.5%).
- Its daily correlation with v3 is 0.64, so it is not a new return source.

**Limit-order execution of v3's own trades.** 2,738 trades were replayed on 5-minute data, with trade-through fills:

| Execution | Fill rate | All-in cost per trade |
|---|---|---|
| Market order (current) | 100% | 12.5 bp |
| Limit for 15 min, then market | 90% | 9.7 bp |
| Limit for 30 min, then market | 93% | 9.8 bp |
| Limit for 60 min, then market | 95% | 10.0 bp |

Unfilled orders had moved 40–83 bp against us by the time we chased. The saving is about 5.7% of capital over 13 months, or about +0.15% per 10 days. It is real but small, and it needs new live order-management code.

**Conclusion.** Trading faster loses money after costs in every family tested. v3 as deployed remains the best strategy found. The only validated improvement is passive execution, worth about +0.15% per 10 days.

## 13. Factor research: new signal types (`research/factor_lab.py`, `factor_strategy.py`)

Extra public data came from `research/fetch_factors.py`:
- hourly taker-buy volume and trade counts (64 coins, 2 years);
- perpetual funding rates (64 coins, 2 years);
- futures open interest and long/short ratios (30 coins, 13 months).

30 signals were scored by daily cross-sectional rank IC on the causal top-30 universe. To count as stable, a signal needed the same sign in both halves of the sample and |t| > 2.

| Signal group | 24h rank IC (t) | Verdict |
|---|---|---|
| Funding rate (last, 3d, 7d, change) | 0.00 to 0.01 (t at most 1.1) | No cross-sectional edge |
| Taker-buy imbalance (24h, 7d) | -0.01 to 0.00 (t at most 1.0) | None |
| Open interest change, long/short ratios | abs(t) at most 2.3; one half only, few coins | Not usable |
| 14-day return (the v3 momentum ranking) | 0.00 (t = -0.2) | Weak at 1 day; 0.04 at 7 days |
| Short-term returns (1h, 4h) | -0.035 (t = -3.4) | Real reversal, but lost 32-45% a year after costs |
| Low volatility, low beta, liquidity, negative skew | -0.05 to -0.08 (t -3.6 to -5.8) | Strong IC, but a bear-market bet: long/short lost in the second half |
| **Nearness to the 14-day high** | **+0.050 (t = 4.1)**; 0.093 at 72h, 0.118 at 7d | **Stable in both halves; quintile long/short +31% a year net** |

Strategies built from these (real engine, x2 risk):
- **Walk-forward IC-weighted composite of all signals:** it lost 38-44% as a market-neutral long/short, and was +12% to +19% when regime-gated. It mostly learned "risky coins fall", which reversed.
- **v3 with the momentum sleeve ranked by nearness to the 14-day high:**

  | Variant | Return | Max DD | Composite |
  |---|---|---|---|
  | v3 (deployed) | +35.8% | -21.1% | 1.34 |
  | Top/bottom 2 | +36.3% | - | 1.38 |
  | Top/bottom 3 | +40.8% | -19.8% | 1.54 |
  | Top/bottom 5 | +37.1% | - | 1.45 |

  A small, consistent improvement.

## 14. Risk frontier and strategy v4 (`research/risk_frontier.py`)

No signal tested has an edge much above about +0.7% to +1.0% per 10 days at x2 risk. A return-ranked leaderboard rewards upside, so the remaining lever is exposure. The frontier covers Jul 2025 to Oct 2026, with the real engine and 10-day windows:

| Strategy | Exposure | Return | Max DD | Mean 10d | P(10d > +10%) | P(10d < -10%) |
|---|---|---|---|---|---|---|
| v3 | x2 (deployed), kill 12% | +36.7% | -21.1% | +0.72% | 3.9% | 0.5% |
| v3 | max | +20% to +54% (unstable across kill settings) | -26% to -30% | +0.5% to +1.1% | 8-11% | 3-5% |
| **v3 + high14 (top 3)** | **max** | **+62% to +71% across kill 12% / 20% / none** | **-23% to -26%** | **+1.15% to +1.25%** | **6.6-7.1%** | **2.5-3.0%** |

"max" means vol target 200%, 100% per coin per sleeve and 100% gross, so the caps decide and the book is fully invested whenever signals are active. There is never any leverage.

Robustness of the pick at max exposure with a 12% kill switch:

| Variant | Return | Max DD | First half | Second half | 2026 |
|---|---|---|---|---|---|
| K=2 | +55.4% | -28.7% | +36.2% | +15.5% | +12.7% |
| **K=3 (v4)** | **+72.7%** | **-22.0%** | +26.3% | +26.3% | +32.0% |
| K=4 | +71.7% | -22.4% | +36.6% | +24.4% | +29.8% |
| K=5 | +51.0% | -25.3% | +37.2% | +11.4% | +18.0% |
| 10-day high, K=3 | +46.0% | -24.3% | +34.5% | +7.7% | +12.5% |
| 21-day high, K=3 | +63.0% | -22.8% | +27.2% | +25.3% | +27.9% |

All variants are positive in both halves. K=3 with a 14-day high is the best cell of the grid, so expect less live: about +50% to +60% a year in backtest terms.

**v4 = HMM+SMA-gated long/short on nearness to the 14-day high (top/bottom 3) + Hawkes, 50/50, max exposure, 12% kill switch.** Production backtest (`backtest/run_v3.py`): **+64.5%, -22.2% max DD, composite 1.83**. Before order processing was made deterministic, the same code gave +66.9% and +69.7% on two runs, so treat about 3 points either way as noise.

Costs and risks of v4:
- Fees are about 52% of capital over 15 months, at about 10 trades a day.
- The worst 10-day window is -12.6% (-17% for neighbouring variants), and the worst 14-day drawdown is -18%: gaps overshoot the 12% kill switch.
- One coin can be 50% of equity, when the Hawkes sleeve has a single active signal.
- The long-only fallback, if shorts are refused, is +9% with a -28% max DD.

## 15. Tick scalping on wide-tick coins (`research/tick_scalp.py`, 2026-10-05)

The leaderboard leader (+6.0% on 2026-10-05) rests limit buys and sells one tick apart on STO and PEPE, where one tick is about 0.25% of the price. The idea was replayed on 30 days of Binance 1-minute bars: rest a buy one tick below the last close, sell one tick higher, stop out 2 ticks lower. Fees are 0.05% per resting fill and 0.10% for the market exit.

| Coin (tick size) | Fill at touch (simulator, no queue) | Fill only on trade-through (real market) |
|---|---|---|
| PEPE (23 bp) | +53% of notional per day | -19% per day |
| BONK (26 bp) | +33% per day | -18% per day |
| SHIB (17 bp) | +12% per day | -15% per day |
| WLFI (18 bp) | +1.5% per day | -9% per day |
| STO (25 bp) | -5.6% per day | -6.4% per day |
| DOGE, BTC (1 bp or less) | -76% to -83% per day | -69% to -80% per day |

The whole edge is the fill assumption. With a real order queue the same orders lose on every coin and on almost every day. This is also market making, which the brief bans, and the organizers said an edge from simulator mechanics is not valued when finalists are judged. **Rejected.**

v4's signal was also run at v3's exposure (vol 0.50, 40% per coin): +37.5%, -20.5% max DD, composite 1.43, against v3's +36.7%, -21.1%, 1.36. So nearly all of v4's extra return comes from exposure, not from the ranking.

## 16. Next steps


1. Build the proper backtest engine (`backtest/engine.py`): next-bar-open fills, 0.1% fee plus half-spread, multi-asset target weights, Sharpe/Sortino/Calmar/composite, and rolling 14-day windows.
2. Backtest the draft against the alternatives in one table, losers included:
   - the Phase A EMA baseline;
   - buy & hold BTC;
   - inverse-vol majors with a regime filter;
   - mean reversion on 72h returns with the regime filter, since 72h lookbacks showed reversal.
3. Validate the winner once on the untouched holdout (5 Jul to 4 Oct 2026).
4. Put the chosen `target_weights()` into `strategy.py` and get the bot running. **Every day without trades uses up the 8-active-day buffer**, so steps 1–3 should take hours, not days.
