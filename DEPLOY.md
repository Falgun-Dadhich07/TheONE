# Deploying TheONE on the hackathon EC2 instance

Connect through **AWS Console → EC2 → Instances → select the instance → Connect → Session Manager** (region ap-southeast-2, Sydney). There is no SSH key.

Run every command below exactly as written, one per line.

## First-time setup

```bash
sudo su - ec2-user
sudo dnf install -y git tmux python3-pip
python3 --version
git clone https://github.com/Falgun-Dadhich07/TheONE.git
cd TheONE
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
nano .env
```

In `nano`, set the following, then save with `Ctrl+O`, `Enter`, `Ctrl+X`:

```
ROOSTOO_MODE=competition
ROOSTOO_COMP_KEY=<competition key>
ROOSTOO_COMP_SECRET=<competition secret>
DRY_RUN=false
CONFIRM_COMPETITION=yes-on-ec2
STRATEGY=v3
ALLOW_SHORTS=true
```

`STRATEGY=v3` is the deployed strategy: the 50/50 blend of regime-gated 14-day momentum and Hawkes, long/short, on the 20 most liquid coins. `STRATEGY=v4` (a research variant, never deployed) and `STRATEGY=v1` (the original long-only bot) are kept as options.

If the repo is still private, `git clone` asks for a GitHub username and a personal access token (not your password).

## Check before going live (read-only, no orders)

```bash
cd ~/TheONE
source .venv/bin/activate
curl -s "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1"
python -m pytest -q
DRY_RUN=true timeout 240 python bot.py
```

- The `curl` must print one JSON kline. If Binance is unreachable, the bot still works from its saved data (`data/live_md.pkl`) and its own Roostoo polls, but the first warm-up needs Binance.
- v3's first warm-up downloads about 1500 hourly bars and 2 years of daily bars for about 64 coins. It takes about 1 minute; computing the HMM regime takes about 20 seconds more. Use `timeout 240` for the dry run on the first start.
- The dry run must log:
  - `warmup (64 coins): 1500 hourly bars x 64 coins ... failed: []`;
  - three `HMM P(bull)` lines;
  - an `equity $...` line;
  - `DRY_RUN BUY/SELL/SHORT_OPEN ...` lines;
  - no `ERROR`.

## Start the bot (24/7, inside tmux)

```bash
cd ~/TheONE
source .venv/bin/activate
tmux new -s bot
python bot.py
```

Detach and leave it running: press `Ctrl+B`, release, then press `D`.

## Check on it later

```bash
sudo su - ec2-user
tmux attach -t bot
```

Detach again with `Ctrl+B` then `D`. **Do not type `exit` or press Ctrl+C inside tmux unless you are restarting for an update.**

Read the logs without attaching:

```bash
tail -n 50 ~/TheONE/logs/bot.log
tail -n 5 ~/TheONE/logs/equity.csv
tail -n 20 ~/TheONE/logs/orders.csv
```

## Restart after a code update (committed and pushed first)

```bash
sudo su - ec2-user
cd ~/TheONE
git pull
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
tmux attach -t bot
```

Then, inside tmux:

1. Press `Ctrl+C` once and wait for `TheONE stopped cleanly`.
2. Start it again with `python bot.py`.
3. Detach with `Ctrl+B` then `D`.

The bot reloads its state (`data/bot_state.json`: kill-switch peak, today's trade count) and re-syncs with real exchange balances on startup.

## If the tmux session is gone (e.g. instance rebooted)

```bash
sudo su - ec2-user
cd ~/TheONE
source .venv/bin/activate
tmux new -s bot
python bot.py
```

## Shorts

- v3 opens Roostoo 1x shorts (`/v6/short_open`), sized by USD collateral, only through the bot.
- If the competition refuses shorts (`this competition does not allow short positions`), the bot logs `SHORTS_NOT_ALLOWED`, sets `shorts_disabled` in `data/bot_state.json` and keeps trading long-only. Nothing needs to be done by hand.
- Check open shorts: `grep SHORT ~/TheONE/logs/orders.csv | tail`.

## Re-tuning the Hawkes parameters (about every 60 days; not needed during the 14-day contest)

On the laptop, not on EC2:

1. Run `python research/fetch_data.py --refresh`.
2. Run `python research/tune_hawkes.py`.
3. Copy the result into `HAWKES_PARAMS` in `settings.py`.
4. Commit, push, then pull and restart on EC2.

## Rules to remember

- Only the bot may trade the competition account. No manual orders, and no manual API scripts against the competition keys (`smoke_test.py --trade` refuses competition mode; don't run `status.py` with competition keys either).
- Stop the bot only to deploy a committed update. The FAQ prohibits manually stopping it or overriding its decisions otherwise.
- Every strategy or config change: commit with a clear message → `git push` → `git pull` on EC2 → restart as above.
