# TheONE: autonomous crypto trading bot for the Roostoo Quant Trading Hackathon

A long-only, low-turnover momentum bot for the Roostoo mock exchange. It aims for **risk-adjusted** return (the final score is 0.4·Sortino + 0.3·Sharpe + 0.3·Calmar), so it keeps drawdowns small and is happy to sit in cash.

## 1. Strategy

**Idea:** in crypto, coins that have been strongest over the last two weeks tend to keep leading, but only while the whole market is healthy. Buy the leaders in a healthy market; hold cash otherwise.

- **Universe:** BTC, ETH, SOL, BNB, XRP, DOGE. A fixed list of large, liquid coins, chosen up front rather than by recent performance.
- **Entry / selection (once a day, after the 00:00 UTC bar):**
  - risk-on only if BTC is above its 20-day (480h) moving average;
  - rank the six by 14-day (336h) return;
  - hold the top 2 that have a positive 14-day return.
- **Exit:**
  - a coin drops out of the top 2 or its momentum turns negative;
  - BTC falls below its 20-day average (all cash);
  - or the kill switch fires.
- **Position sizing:**
  - inverse volatility, aiming for 25% annualized portfolio volatility;
  - each coin gets `0.25 / (n × its 7-day vol)`;
  - caps: 25% per coin, 50% total;
  - sized as a % of live total equity, so it works for any account size.
- **Risk management:**
  - **kill switch:** an 8% drawdown from peak equity sends the bot to cash for 24h, then it resumes on its own. Checked every minute.
  - **rebalance band:** don't trade weight changes under 5 percentage points.
  - **order limit:** at most 2 strategy round trips per pair per day.
  - **no leverage, no shorts.**
- **Activity guarantee:** if the bot has made fewer than `MIN_TRADES_PER_DAY` fills by 20:00 UTC, it buys about $15 of BTC and sells it straight back. This guarantees an active day for about $0.03 in fees.

All tunables are in [`settings.py`](settings.py). Research, results and failure modes are in [`docs/strategy_draft.md`](docs/strategy_draft.md).

### Backtest (hourly Binance data, next-bar-open fills, 0.1% fee + 2 bp half-spread per side)

| Period | Return | Max DD | Sharpe | Sortino | Calmar | Composite |
|---|---|---|---|---|---|---|
| In-sample, 31 Oct 2024 to 4 Jul 2026 (BTC −13%) | +18.4% | −18.8% | 0.70 | 0.98 | 0.56 | 0.77 |
| Holdout, 5 Jul to 4 Oct 2026 (BTC +35%) | +10.9% | −5.0% | 2.29 | 3.53 | 10.17 | 5.15 |

Honest caveats:
- It lags buy & hold in strong rallies, because it is only ~30% invested on average.
- The holdout was looked at during development, so it is not a pristine test.
- Two years of data is a small sample.

## 2. Architecture

```
bot.py            main loop: poll -> bars -> strategy -> risk -> orders; logging, state, shutdown
strategy.py       signal functions shared by the bot AND the backtester (no drift possible)
risk.py           kill switch / trailing stop, shared by bot and backtester
execution.py      valuation, Decimal rounding, order planning, place -> confirm -> reconcile
market_data.py    hourly bars: Binance warm-up + per-hour closes, fallback to own Roostoo polls
roostoo_client.py signed REST client (HMAC-SHA256); reads retry, orders never auto-retry
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
