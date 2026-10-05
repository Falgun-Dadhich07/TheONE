# TheONE — Roostoo Quant Trading Hackathon bot

> Project brief (verbatim from the team). Every session must follow it.
> An addendum with facts gathered from the organizers' docs is at the bottom.

You are building an autonomous crypto trading bot for the Roostoo Quant Trading Hackathon, in this repo (TheONE). Read this entire brief first. Step 0: save this brief as CLAUDE.md in the repo root so every future session has it. Then propose a plan and WAIT for my approval before writing code.

## COMPETITION RULES (hard constraints)
- Roostoo is a mock exchange: real-time market prices, simulated fills, fake money. Base URL: https://mock-api.roostoo.com. API docs: https://github.com/roostoo/Roostoo-API-Documents
- Live trading: Oct 4 - Oct 17, 2026 (14 days). It has ALREADY STARTED. Competition account starts with $100,000 USD. Test account has $50,000.
- The bot must run autonomously 24/7 on our AWS EC2 instance (Amazon Linux, Python venv, started inside tmux). Assume Python 3.9 compatibility.
- We need at least 8 active trading days with trades on each day. The exact minimum trades/day is unknown: make it a config value.
- Banned: high-frequency trading, market making, arbitrage. Spot only. Do NOT rely on short selling (unclear if enabled; keep shorts off).
- Disqualification risk: manual API calls on the competition account. Every competition order must come from the bot. Commit history must be consistent and traceable: commit each meaningful change with an honest message.
- Repo must be open-source and submitted before Oct 14.
- Ranking: first by return (top 20 per region), then finalists scored 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar. Optimize risk-adjusted return and LOW DRAWDOWN, not maximum return.

## FACTS VERIFIED BY OUR OWN TESTS
- HMAC signing verified against the official docs example. Test keys work; signed GET and POST both work.
- Market BUY then SELL of 1 BNB/USD: both filled instantly as TAKER at the ask/bid. Fee is 0.1% per side; round trip costs ~0.2% including spread. Limit (maker) fee is 0.05%.
- Exchange info: 88 pairs, including tokenized stocks (e.g. NVDAB/USD, NBISB/USD). Exclude those. Example: BNB/USD price precision 2, amount precision 3, MiniOrder 1 (USD notional).
- Balance response wallet looks like {"USD": {"Free", "Lock", "PendingOrders", "ShortCollateral"}, ...}.
- The API returns HTTP 200 even on failure: always check "Success".
- Quantities must be rounded DOWN to AmountPrecision using Decimal (float artifacts like 0.30000000000000004 get rejected), and price*qty must exceed MiniOrder.
- Request timestamps must be within 60s of server time.

## EXISTING CODE (read first; don't rewrite what works)
- roostoo_client.py: signed client. Read calls retry on network errors; order calls NEVER auto-retry (prevents duplicate orders). Keep that property.
- config.py + .env.example: ROOSTOO_MODE switches between test and competition keys.
- smoke_test.py (connectivity + test-mode round trip), status.py (read-only P&L, appends logs/equity.csv).
- backtest/: MY backtesting code. Audit it before using it: lookahead bias, fee and spread modeling, fill assumptions, bar alignment, metric formulas. Reuse it if sound; report every flaw you find.
- docs/data_sources.md: the organizers' Data Sources Pack. Use the historical sources it lists. Flag anything in it that conflicts with this brief.

## PHASE A: BASELINE BOT (do this first, today)
Goal: a simple, conservative, correct bot running on EC2 within hours. A bad-but-safe bot beats no bot, because every day without trades burns our 8-active-days buffer.
Strategy for the baseline: long-only, BTC/USD, ETH/USD, SOL/USD, BNB/USD only, decisions on hourly bars, a simple trend filter, volatility-scaled position sizing, total exposure capped (e.g. 50-60% of equity), at most ~2 round trips per pair per day. Two or three parameters maximum.
bot.py requirements:
1. Poll the ticker for all pairs in one call about every 60s; build hourly bars; make decisions only on bar close.
2. Warm up indicators at startup from public historical data (from docs/data_sources.md, or Binance public klines, e.g. data-api.binance.vision) so we don't wait days. Verify reachability from EC2; fall back to our own persisted polls in data/.
3. Read real balances from the exchange at startup and every cycle. Never trust local state alone.
4. Size positions as a % of current total equity (works for both the $50k and $100k accounts).
5. Round quantities down to AmountPrecision; skip orders below MiniOrder; log every skip with its reason.
6. After placing an order, query it to confirm. Never auto-retry place_order. On timeout or ambiguity, query open/recent orders before deciding anything.
7. Catch exceptions per cycle; the loop must never die. Back off on repeated failures.
8. Logs: logs/bot.log (rotating), logs/orders.csv (every request + response), logs/equity.csv (portfolio value every cycle).
9. Kill switch: if drawdown from peak equity exceeds a configurable % (e.g. 8%), move to cash and stop opening positions until manually reset.
10. Activity guarantee: if no strategy trade happened by a configurable UTC hour, place the smallest valid trade to count as an active day (configurable count; minimal fee cost).
11. DRY_RUN flag: compute and log orders without sending them.
12. All tunables in one config place. Graceful shutdown on Ctrl+C.
13. The signal logic is ONE function that Phase B's backtester also imports, so live and backtest can never drift apart.
Testing: pytest unit tests for rounding, sizing, kill switch, activity guarantee, and the signal on synthetic data. Then run locally in DRY_RUN, then ~30-60 minutes in test mode with test keys, and show me the logs.
Also write DEPLOY.md: exact EC2 commands, one per line (sudo su - ec2-user, cd TheONE, source .venv/bin/activate, git pull, pip install, tmux new -s bot, python bot.py, how to detach and reattach, how to restart after an update).
Commit when Phase A passes. STOP and report.

## PHASE B: STRATEGY RESEARCH WITH BACKTESTING
Only after Phase A is deployed.
- Data: hourly OHLCV, at least 1-2 years, for liquid candidate pairs, from docs/data_sources.md or the Binance public archive (data.binance.vision). Cache in data/ (gitignored).
- Backtester must: execute at the NEXT bar's open (no lookahead); charge 0.1% per side for taker (0.05% for maker) plus half-spread; annualize with 24*365 hourly periods; report total return, Sharpe, Sortino, Calmar, max drawdown, number of trades, fees paid, and the competition composite (0.4*Sortino + 0.3*Sharpe + 0.3*Calmar).
- Because the competition lasts 14 days, ALSO evaluate every strategy on rolling 14-day windows: the distribution of return, max drawdown, and composite score, plus the % of windows with a loss.
- Out-of-sample discipline: tune only on older data; keep the most recent ~3 months untouched for final validation; use walk-forward where feasible.
- Candidates: 3-5 simple families (e.g. trend/momentum with volatility targeting; cross-sectional momentum rotation among top liquid coins; mean reversion with a regime filter). Max 3 tunable parameters each. Show one comparison table with all of them, including the losers. No cherry-picking.
- Recommend one strategy, explain why, and list its failure modes. I approve before anything is deployed. Deployment = swap the signal function and config, commit, git pull on EC2, restart in tmux.

## WORKING RULES
- Plan first; stop for my approval at the end of each phase.
- Never write real keys into tracked files. .env, logs/, data/ must be in .gitignore. Check git status before every commit.
- NEVER run anything that places orders with ROOSTOO_MODE=competition from my laptop. Locally, use test mode only.
- Minimal dependencies: requests, python-dotenv, pandas, numpy, pytest.
- If unsure how the API behaves, write a small read-only probe using test keys instead of guessing.
- Clear commit messages at each milestone.

---

## Addendum: facts from the organizers' docs (gathered 2026-10-04)

Sources: Official FAQ, AWS guide, and Data Sources Pack (Notion pages linked from the hackathon materials).

- **Rate limit: 30 API calls per minute, queries included.** Over the limit, calls fail. Budget every call.
- **Roostoo prices are streamed from Binance.** Roostoo has no OHLCV endpoint (ticker snapshots only), so Binance klines are the right source for warm-up and backtests.
- **No manual intervention.** FAQ Q28: "You may not manually stop the bot, override its decisions, or place discretionary trades through the API." Code updates during the competition are allowed if committed with a clear history.
- Shorts are allowed per the FAQ, but this brief keeps them off.
- Holdings are liquidated automatically at the end.
- FAQ dates (Main Round starts Sept 30, first trade by Oct 1 8pm) differ from this brief (Oct 4-17). The brief is treated as authoritative.
- AWS: region ap-southeast-2 (Sydney) only; connect via Session Manager (no SSH key); one instance; tmux via `sudo dnf install -y tmux`.
- The Data Sources Pack lists Binance Vision (data.binance.vision, free, recommended), CryptoDataDownload (free CSVs), and CoinAPI (paid). It states no usage rules.
- The FAQ mentions Binance blocking US regions. EC2 is in Sydney, but reachability must still be verified from the instance.
- Repo state on 2026-10-04: there was no `backtest/` or `docs/` folder. The legacy backtester was pasted into the chat (audit in the Phase A plan).

## Decision log

- 2026-10-05: The team lead approved **considering short positions** (overriding "keep shorts off" above) because Roostoo supports them (`/v6/short_open`, `/v6/short_close`, `/v6/short_positions`).
  - Shorts are 1x: sized by USD collateral, 0.1% fee to open and to close, loss capped at the collateral.
  - The API can return `this competition does not allow short positions`, so the bot must handle that and fall back to long-only.
  - Deploying any short-selling strategy still needs explicit approval.
