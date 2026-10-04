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
UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
QUOTE = "USD"
STRATEGY = "xs_momentum_regime"
STRATEGY_PARAMS = {"lookback": 336, "regime_sma": 480, "top_n": 2}

# --- Position sizing (shared with backtest) --------------------------------
TARGET_VOL = 0.25          # annualized portfolio vol target
VOL_LOOKBACK = 168         # hours
MAX_PER_COIN = 0.25        # fraction of equity
MAX_GROSS = 0.50           # total exposure cap
REBALANCE_HOUR = 0         # UTC hour of the bar whose close triggers the daily rebalance
REBALANCE_BAND = 0.05      # min weight change worth trading

# --- Risk overlay (shared with backtest) -----------------------------------
MAX_DRAWDOWN = 0.08        # kill switch: drawdown from peak equity
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
WARMUP_HOURS = 1000                  # strategy needs ~816h (480h SMA + 336h momentum)

# --- Files ------------------------------------------------------------------
LOG_DIR = "logs"
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
BARS_FILE = os.path.join(DATA_DIR, "live_bars.csv")
