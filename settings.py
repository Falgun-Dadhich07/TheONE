"""Every tunable of the bot in one place.

Secrets and the mode (test/competition) live in .env and config.py. A few
switches can be overridden from .env too (DRY_RUN, MIN_TRADES_PER_DAY, ...).
strategy.py and risk.py read their defaults from here, so the backtester and
the live bot always use the same numbers.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v in (None, "") else int(v)


# --- Safety -----------------------------------------------------------------
DRY_RUN = _env_bool("DRY_RUN", True)          # default SAFE: log orders, send nothing
# Competition orders are only allowed when this env var is set to this exact
# value (put it in the EC2 .env only) and never on Windows (our laptops).
COMPETITION_CONFIRM_VALUE = "yes-on-ec2"

# --- Universe & strategy ----------------------------------------------------
# STRATEGY = "v4"  : v3 with the momentum sleeve ranked by nearness to the 14-day high,
#                    top/bottom 3 (docs/strategy_draft.md s.13-14)
# STRATEGY = "v3"  : 50/50 blend of HMM+SMA-gated long/short 14-day-return momentum (top/bottom 2)
#                    and Hawkes long/short on the 20 most liquid coins (s.9-10)
# STRATEGY = "v1"  : original long-only momentum on the 6 majors (rollback option)
STRATEGY = os.getenv("STRATEGY", "v4").strip().lower()
UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]       # v1 universe
QUOTE = "USD"
STRATEGY_PARAMS = {"lookback": 336, "regime_sma": 480, "top_n": 2}   # v1

# v3
V3_TOP_N = 20                 # liquid universe size (trailing 30-day median Binance volume)
V3_REGIME_SMA = 480           # BTC SMA gate, hours
V3_HAWKES_SHARE = 0.5         # capital split: 0.5 Hawkes / 0.5 momentum
V3_RANK_BY = "high" if STRATEGY == "v4" else "return"   # momentum sleeve ranking
V3_K = 3 if STRATEGY == "v4" else 2                      # coins per side in the momentum sleeve
ALLOW_SHORTS = _env_bool("ALLOW_SHORTS", True)
# Tuned by research/tune_hawkes.py on 2026-04-07..2026-10-04 (majors, PSO, composite objective).
# Re-tune about every 60 days (the tested walk-forward step).
HAWKES_PARAMS = {"kappa": 0.016, "norm_lookback": 244, "signal_lookback": 479,
                 "perc5": 0.01, "perc95": 0.99}
# Not crypto or no Binance price source: never traded by v3.
NON_CRYPTO = {"PAXG", "AMDB", "CBRSB", "COINB", "CRCLB", "GLWB", "GOOGLB", "INTCB", "LITEB", "METAB",
              "MSFTB", "MSTRB", "MUB", "NBISB", "NVDAB", "PLTRB", "QCOMB", "SKHYB", "SNDKB", "SPCXB",
              "TSLAB", "WDCB", "TON", "OMNI"}

# --- Position sizing (shared with backtest) --------------------------------
# Risk history (each step requested by the team lead; research in docs/strategy_draft.md):
#   start        vol 0.25, 25%/coin, 50% gross, kill 8%
#   2026-10-05   vol 0.50, 40%/coin, 100% gross, kill 12%   (s.11)
#   2026-10-05   vol 2.00, 100%/coin, 100% gross, kill 12%  (s.14: fully invested whenever
#                signals are active; the caps bind, not the vol target; never leverage)
# The risk level is tied to the strategy so a rollback also rolls back the risk.
if STRATEGY == "v4":
    TARGET_VOL = 2.00      # annualized vol target per sleeve side (so high the caps decide)
    MAX_PER_COIN = 1.00    # fraction of equity per sleeve (a coin in one sleeve is <= 50% of equity)
else:
    TARGET_VOL = 0.50
    MAX_PER_COIN = 0.40
VOL_LOOKBACK = 168         # hours
MAX_GROSS = 1.00           # total |long| + |short| exposure cap (never leverage)
REBALANCE_HOUR = 0         # UTC hour of the bar whose close triggers the daily rebalance
REBALANCE_BAND = 0.05      # min weight change worth trading

# --- Risk overlay (shared with backtest) -----------------------------------
MAX_DRAWDOWN = 0.12        # kill switch: drawdown from peak equity
KILL_COOLDOWN_HOURS = 24   # cash period after the kill switch; then automatic resume
STOP_VOL_MULT = None       # per-coin trailing stop (None = off; backtests showed it hurts)

# --- Activity guarantee -----------------------------------------------------
MIN_TRADES_PER_DAY = _env_int("MIN_TRADES_PER_DAY", 2)   # exact rule unknown; filled orders per UTC day
ACTIVITY_HOUR_UTC = _env_int("ACTIVITY_HOUR_UTC", 20)    # if short of trades by this hour, top up
ACTIVITY_PAIR = "BTC/USD"
ACTIVITY_NOTIONAL_USD = 15.0                             # buy this much, then sell it straight back

# --- Execution --------------------------------------------------------------
MIN_ORDER_USD = 10.0                 # never send orders smaller than this (also > exchange MiniOrder)
MAX_STRATEGY_ORDERS_PER_PAIR_PER_DAY = 4   # = 2 round trips; risk exits are exempt
DUST_USD = 5.0                       # holdings below this are ignored when computing weights

# --- Loop & API -------------------------------------------------------------
POLL_SECONDS = 60
API_MIN_INTERVAL_S = 2.1             # Roostoo limit is 30 calls/min incl. queries
MAX_BACKOFF_SECONDS = 900
MAX_CLOCK_SKEW_MS = 10_000

# --- Market data ------------------------------------------------------------
BINANCE_KLINES_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
WARMUP_HOURS = 1500                  # v3 Hawkes needs ~1000h (ATR 244h + quantiles 479h + vol 168h) to settle

# --- Files ------------------------------------------------------------------
LOG_DIR = "logs"
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
BARS_FILE = os.path.join(DATA_DIR, "live_bars.csv")
