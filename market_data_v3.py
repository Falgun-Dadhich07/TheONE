"""Market data for strategy v3: every crypto pair on Roostoo.

Roostoo has no candles and streams Binance prices (FAQ), so:
  * hourly OHLC: Binance <COIN>USDT 1h klines, warm-up WARMUP_HOURS, then each
    closed hour is fetched; if Binance fails for a coin, that hour is built
    from our own Roostoo ticker polls (high/low/last seen in the hour);
  * daily close + quote volume: Binance 1d klines since HMM_HISTORY_START
    (same start as the validated backtest), for the HMM regime and the
    top-20 liquidity ranking;
  * everything is persisted to data/live_md.pkl so restarts don't depend on Binance.

Index convention matches the backtest: row H = the bar opening at H.
"""
import logging
import os
import pickle
from typing import Dict, List, Optional

import pandas as pd
import requests

import settings

log = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
HMM_HISTORY_START = pd.Timestamp("2024-10-01", tz="UTC")
PATH = os.path.join(settings.DATA_DIR, "live_md.pkl")


def fetch_klines(coin: str, interval: str, start: pd.Timestamp, session: requests.Session,
                 max_pages: int = 3) -> pd.DataFrame:
    """Closed Binance klines from `start` (paginated). Columns: high, low, close, quote_volume."""
    rows, t = [], int(start.timestamp() * 1000)
    now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    for _ in range(max_pages):
        page, last_exc = None, None
        for url in settings.BINANCE_KLINES_URLS:
            try:
                r = session.get(url, params={"symbol": f"{coin}USDT", "interval": interval,
                                             "startTime": t, "limit": 1000}, timeout=10)
                if r.status_code == 400:            # unknown symbol on Binance
                    raise LookupError(f"{coin}USDT not on Binance")
                r.raise_for_status()
                page = r.json()
                break
            except LookupError:
                raise
            except Exception as exc:
                last_exc = exc
        if page is None:
            raise ConnectionError(f"Binance {interval} klines for {coin} unavailable: {last_exc}")
        page = [k for k in page if k[6] < now_ms]            # closed bars only
        rows += page
        if len(page) < 1000:
            break
        t = page[-1][0] + 1
    if not rows:
        return pd.DataFrame(columns=["high", "low", "close", "quote_volume"], dtype=float)
    idx = pd.to_datetime([k[0] for k in rows], unit="ms", utc=True)
    df = pd.DataFrame({"high": [float(k[2]) for k in rows], "low": [float(k[3]) for k in rows],
                       "close": [float(k[4]) for k in rows], "quote_volume": [float(k[7]) for k in rows]},
                      index=idx)
    return df[~df.index.duplicated(keep="last")]


class MarketData:
    def __init__(self, coins: List[str], path: str = PATH):
        self.coins = sorted(coins)
        self.path = path
        self.high = pd.DataFrame(columns=self.coins, dtype=float)
        self.low = pd.DataFrame(columns=self.coins, dtype=float)
        self.close = pd.DataFrame(columns=self.coins, dtype=float)
        self.daily_close = pd.DataFrame(columns=self.coins, dtype=float)
        self.daily_qv = pd.DataFrame(columns=self.coins, dtype=float)
        self.unavailable: set = set()          # coins Binance doesn't list
        self.polls: Dict[str, Dict[pd.Timestamp, list]] = {}   # coin -> hour -> [high, low, last]
        self.session = requests.Session()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as f:
            d = pickle.load(f)
        for k in ("high", "low", "close", "daily_close", "daily_qv"):
            setattr(self, k, d[k].reindex(columns=self.coins))
        self.unavailable = set(d.get("unavailable", set()))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        keep = settings.WARMUP_HOURS + 200
        d = {"high": self.high.tail(keep), "low": self.low.tail(keep), "close": self.close.tail(keep),
             "daily_close": self.daily_close, "daily_qv": self.daily_qv, "unavailable": self.unavailable}
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(d, f)
        os.replace(tmp, self.path)

    def _merge(self, attr: str, new: pd.DataFrame) -> None:
        old = getattr(self, attr)
        comb = new.combine_first(old) if not old.empty else new
        setattr(self, attr, comb.reindex(columns=self.coins).sort_index())

    def warmup(self) -> str:
        self._load()
        now = pd.Timestamp.now(tz="UTC")
        h_start = now.floor("h") - settings.WARMUP_HOURS * HOUR
        fails = []
        for coin in self.coins:
            if coin in self.unavailable:
                continue
            try:
                last = self.close[coin].last_valid_index() if coin in self.close and not self.close.empty else None
                start = h_start if last is None or last < h_start else last + HOUR
                hk = fetch_klines(coin, "1h", start, self.session)
                dlast = self.daily_close[coin].last_valid_index() if not self.daily_close.empty else None
                dstart = HMM_HISTORY_START if dlast is None else dlast + DAY
                dk = fetch_klines(coin, "1d", dstart, self.session)
            except LookupError:
                self.unavailable.add(coin)
                continue
            except ConnectionError as exc:
                fails.append(coin)
                log.warning("warmup %s: %s", coin, exc)
                continue
            for attr, col in (("high", "high"), ("low", "low"), ("close", "close")):
                self._merge(attr, hk[[col]].rename(columns={col: coin}))
            self._merge("daily_close", dk[["close"]].rename(columns={"close": coin}))
            self._merge("daily_qv", dk[["quote_volume"]].rename(columns={"quote_volume": coin}))
        self.save()
        return (f"{len(self.close)} hourly bars x {self.close.notna().any().sum()} coins, "
                f"{len(self.daily_close)} daily bars; last hour {self.close.index[-1] if len(self.close) else None}; "
                f"not on Binance: {sorted(self.unavailable)}; failed: {fails}")

    # ------------------------------------------------------------------
    def on_poll(self, now: pd.Timestamp, prices: Dict[str, float]) -> None:
        hour = now.floor("h")
        for coin, px in prices.items():
            if coin not in self.coins or not px:
                continue
            slot = self.polls.setdefault(coin, {}).setdefault(hour, [px, px, px])
            slot[0], slot[1], slot[2] = max(slot[0], px), min(slot[1], px), px
        for coin in self.polls:                       # keep a few hours only
            for h in [h for h in self.polls[coin] if h < hour - 6 * HOUR]:
                del self.polls[coin][h]

    def close_completed_bars(self, now: pd.Timestamp) -> List[pd.Timestamp]:
        """Fetch every hourly bar closed since the last one; daily bars when a day ends."""
        current = now.floor("h")
        last = self.close.index[-1] if len(self.close) else current - 2 * HOUR
        if last + HOUR >= current:
            return []
        missing = pd.date_range(last + HOUR, current - HOUR, freq="h", tz="UTC")
        new = {k: pd.DataFrame(index=missing, columns=self.coins, dtype=float) for k in ("high", "low", "close")}
        for coin in self.coins:
            if coin in self.unavailable:
                continue
            try:
                hk = fetch_klines(coin, "1h", missing[0], self.session, max_pages=1).reindex(missing)
                for k in new:
                    new[k][coin] = hk[k]
            except (LookupError, ConnectionError) as exc:
                log.warning("bar close %s: %s; using Roostoo polls", coin, exc)
            for h in missing:                          # fallback from our own polls
                p = self.polls.get(coin, {}).get(h)
                if p is not None and pd.isna(new["close"].at[h, coin]):
                    new["high"].at[h, coin], new["low"].at[h, coin], new["close"].at[h, coin] = p
        for k, df in new.items():
            comb = pd.concat([getattr(self, k), df])
            setattr(self, k, comb[~comb.index.duplicated(keep="last")].sort_index())
        if missing[-1].normalize() > (self.daily_close.index[-1] if len(self.daily_close) else HMM_HISTORY_START):
            self.update_daily()
        self.save()
        return list(missing)

    def update_daily(self) -> None:
        for coin in self.coins:
            if coin in self.unavailable:
                continue
            dlast = self.daily_close[coin].last_valid_index() if len(self.daily_close) else None
            try:
                dk = fetch_klines(coin, "1d", HMM_HISTORY_START if dlast is None else dlast + DAY, self.session)
            except (LookupError, ConnectionError) as exc:
                log.warning("daily %s: %s", coin, exc)
                continue
            if len(dk):
                self._merge("daily_close", dk[["close"]].rename(columns={"close": coin}))
                self._merge("daily_qv", dk[["quote_volume"]].rename(columns={"quote_volume": coin}))

    def ready(self) -> bool:
        return len(self.close) >= 1000 and "BTC" in self.close and self.close["BTC"].notna().sum() >= 1000 \
            and self.daily_close["BTC"].notna().sum() >= 200
