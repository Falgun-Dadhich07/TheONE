"""Central configuration. Secrets come from environment variables / .env only."""
import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")
MODE = os.getenv("ROOSTOO_MODE", "test").strip().lower()  # "test" or "competition"


def get_credentials() -> tuple[str, str]:
    """Return (api_key, secret) for the active mode."""
    if MODE == "competition":
        key, secret = os.getenv("ROOSTOO_COMP_KEY"), os.getenv("ROOSTOO_COMP_SECRET")
    elif MODE == "test":
        key, secret = os.getenv("ROOSTOO_TEST_KEY"), os.getenv("ROOSTOO_TEST_SECRET")
    else:
        raise RuntimeError(f"ROOSTOO_MODE must be 'test' or 'competition', got {MODE!r}")

    if not key or not secret:
        raise RuntimeError(
            f"Missing API credentials for mode '{MODE}'. Fill them in your .env file."
        )
    return key, secret
