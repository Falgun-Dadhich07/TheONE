# TheONE: autonomous crypto trading bot for the Roostoo Quant Trading Hackathon

A long/short crypto bot for the Roostoo mock exchange. It blends two uncorrelated signals and aims for **risk-adjusted** return (the final score is 0.4·Sortino + 0.3·Sharpe + 0.3·Calmar), so it keeps drawdowns small and holds cash when neither signal is active.

## 1. Strategy (v4)

**Universe:** the 20 most liquid crypto pairs on Roostoo. The list is recomputed daily from the trailing 30-day median Binance volume, using past data only. A coin needs at least 60 days of history; tokenized stocks are excluded.

Capital is split **50/50** between two sleeves:

| Sleeve | Signal | When long | When short |
|---|---|---|---|
| **Trend** (daily, 00:00 UTC) | nearness to the 14-day high, gated by the market regime. It predicted next-day returns far better than raw 14-day return (rank IC 0.05, t = 4.1, stable in both halves of 2 years) | the 3 coins closest to their 14-day high, while risk-on | the 3 coins furthest below it, while risk-off |
| **Hawkes volatility** (hourly) | per coin: volatility goes quiet (below its 1st percentile), then breaks out (above its 99th percentile) | the price rose since the quiet point | the price fell since the quiet point |

**Market regime:** both of these must agree.
- **HMM (hidden Markov model):** a 2-state model on BTC daily returns. It's refit every 30 days and averaged over 6 refit calendars, so results don't depend on when the refit happens. Only causal (forward-filtered) probabilities are used.
- **BTC trend:** BTC above or below its 20-day (480h) moving average.

"Risk-on" means P(bull) > 0.5 and BTC is above its average. "Risk-off" means P(bull) < 0.5 and BTC is below its average. Otherwise the momentum sleeve is in cash.

**Position sizing (maximum-exposure setting, chosen for a return-ranked leaderboard; see `docs/strategy_draft.md` s.14):**
- inverse volatility per side; the caps decide the size (the volatility target is set too high to bind);
- up to 100% gross (|long| + |short|), never leverage;
- one coin can reach 50% of equity when the Hawkes sleeve has a single active signal;
- sized as a % of live total equity.

**Shorts:** Roostoo 1x collateral shorts (`/v6/short_open`), with the loss capped at the collateral. If the exchange refuses shorts, the bot switches to long-only automatically.

**Risk management:**
- **kill switch:** a 12% drawdown from peak sends the bot to cash for 24h, then it resumes on its own. It's checked every minute.
- **rebalance band:** weight changes under 5 percentage points aren't traded.
- **order limit:** at most 4 strategy orders per pair per day.
- **no leverage.**

**Activity guarantee:** if there are fewer than `MIN_TRADES_PER_DAY` fills by 20:00 UTC, the bot buys about $15 of BTC and sells it straight back.

All tunables are in [`settings.py`](settings.py). `STRATEGY=v3` rolls back to the 14-day-return momentum version, and `STRATEGY=v1` to the original long-only bot. The research behind every choice, including the strategies that lost, is in [`docs/strategy_draft.md`](docs/strategy_draft.md).

### Backtest of the production code (`python backtest/run_v3.py`)

Setup:
- hourly Binance data, 13 Jul 2025 to 4 Oct 2026;
- execution at the next bar's open, a 0.1% fee plus each coin's measured Roostoo half-spread (at least 2 bp);
- the kill switch on;
- Hawkes parameters re-tuned by walk-forward PSO, with each fold's parameters used only out-of-sample.

| Strategy (deployed settings) | Return | Max DD | Sharpe | Sortino | Calmar | Composite |
|---|---|---|---|---|---|---|
| **v4 blend** | **+66.9%** | **-23.4%** | 1.34 | 1.96 | 2.21 | **1.85** |
| Trend sleeve alone | +55.7% | -33.3% | 1.06 | 1.53 | 1.30 | 1.32 |
| Hawkes sleeve alone | +21.9% | -41.0% | 0.58 | 0.84 | 0.43 | 0.64 |
| v4 if shorts are refused (long-only) | +10.0% | -26.2% | 0.41 | 0.58 | 0.31 | 0.45 |
| *v3 (14-day-return momentum) at x2 risk* | +36.7% | -21.1% | 1.09 | 1.58 | 1.39 | 1.36 |
| *Buy & hold BTC, May 2025 to Oct 2026* | -30.0% | -59.0% | - | - | - | -0.52 |

**Per 10 days (research run, 439 windows):**

| | Return |
|---|---|
| Mean | +1.27% |
| Median | +0.90% |
| Top decile | +8.0% |
| Bottom decile | -5.3% |
| Worst | -12.6% |

| Chance over 10 days | |
|---|---|
| Gain above +10% | 6.6% |
| Loss worse than -5% | 11.9% |
| Loss worse than -10% | 2.5% |

Neighbouring variants (2 to 5 coins, 10- to 21-day highs) made +46% to +73% and were positive in both halves of the period.

Faster trading (3- to 7-day momentum, 4- to 12-hour rebalancing) was tested and earned **less**: fees outweighed any edge.

At lower risk settings, v3 made +18.4% with a -12.2% max drawdown (25% volatility, 50% gross) and +36.7% with a -21.1% max drawdown (50% volatility, 100% gross cap).

Honest caveats:
- It's one 15-month sample.
- With a single refit calendar, the momentum sleeve's return ranged from +8.6% to +26.1%. Averaging over 6 calendars removes that luck, but it shows the uncertainty.
- The Hawkes parameters changed a lot between walk-forward folds.
- Fees are about 52% of capital over 15 months, at about 10 trades a day.
- Fast moves can overshoot the kill switch within the hour; the worst 14-day drawdown was -18%.
- K=3 with a 14-day high is the best cell of the grid tested, so live results should be lower.
- The fallback without shorts is much weaker.

## 2. Architecture

```
bot.py            main loop: poll -> bars -> strategy -> risk -> orders; logging, state, shutdown
strategy.py       signal functions shared by the bot AND the backtester (no drift possible)
regime.py         2-state Gaussian HMM regime (numpy), causal, ensemble of refit calendars
market_data_v3.py hourly OHLC + daily volume for all coins (Binance), fallback to own Roostoo polls
risk.py           kill switch / trailing stop, shared by bot and backtester
execution.py      valuation, Decimal rounding, order planning, place -> confirm -> reconcile
market_data.py    hourly bars: Binance warm-up + per-hour closes, fallback to own Roostoo polls
roostoo_client.py signed REST client (HMAC-SHA256); reads retry, orders and shorts never auto-retry
settings.py       every tunable; config.py + .env hold mode and keys
backtest/         engine.py, metrics.py, run_compare.py, run_risk.py (+ legacy_backtest.py, audited)
research/         fetch_data.py (Binance archive), analyze.py (universe study)
tests/            pytest: rounding, sizing, kill switch, activity, causality, engine costs
```

- **Data:** Roostoo prices are streamed from Binance (official FAQ) and Roostoo has no candles. So the bot warms up on Binance `<COIN>USDT` 1h klines and reads Roostoo's ticker every 60s.
- **API budget:** about 2 calls/minute normally, plus 2 per order, spaced at least 2.1s apart (the limit is 30/minute).
- **Tech stack:** Python 3.9+, requests, python-dotenv, pandas, numpy, pytest.

### Logs

- `logs/bot.log`: rotating, UTC.
- `logs/orders.csv`: every request, response, confirmation, skip and reconciliation.
- `logs/equity.csv`: equity, exposure and drawdown every cycle.

## 3. Setup and run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys; ROOSTOO_MODE=test; DRY_RUN=true
python -m pytest -q
python bot.py               # DRY_RUN: logs orders, sends nothing
```

- **Test account:** set `DRY_RUN=false` with `ROOSTOO_MODE=test`.
- **Competition:** follow [`DEPLOY.md`](DEPLOY.md). The bot refuses live competition orders unless `CONFIRM_COMPETITION=yes-on-ec2` is set, and never on Windows.
- **Reproduce the research:**
  1. `python research/fetch_data.py`
  2. `python research/analyze.py`
  3. `python backtest/run_compare.py`
  4. `python backtest/run_risk.py`
