"""Hourly close prices for the live bot.

Roostoo has no candle endpoint (FAQ), but its prices are streamed from
Binance, so:
  * at startup we warm up with ~1000 hourly Binance klines (<COIN>USDT);
  * after each hour closes we fetch that hour's Binance kline;
  * if Binance is unreachable we use the last Roostoo price we polled in that
    hour, so the bot keeps working on its own data;
  * bars are persisted to data/live_bars.csv so a restart doesn't depend on Binance.

Index convention matches the backtest: a row labeled H is the bar that opens
at H and closes at H+1h, holding its close price.
"""
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import requests

import settings

log = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)


def fetch_binance_closes(coin: str, start: Optional[pd.Timestamp] = None,
                         limit: int = 1000, session: Optional[requests.Session] = None) -> pd.Series:
    """Closed hourly bars from Binance as a Series indexed by open time (UTC)."""
    s = session or requests
    params = {"symbol": f"{coin}USDT", "interval": "1h", "limit": limit}
    if start is not None:
        params["startTime"] = int(start.timestamp() * 1000)
    last_exc: Optional[Exception] = None
    for url in settings.BINANCE_KLINES_URLS:
        try:
            r = s.get(url, params=params, timeout=10)
            r.raise_for_status()
            rows = r.json()
            now_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
            rows = [k for k in rows if k[6] < now_ms]          # drop the still-open bar
            idx = pd.to_datetime([k[0] for k in rows], unit="ms", utc=True)
            return pd.Series([float(k[4]) for k in rows], index=idx, name=coin)
        except Exception as exc:  # try the next mirror
            last_exc = exc
    raise ConnectionError(f"Binance klines for {coin} unavailable: {last_exc}")


class HourlyBars:
    def __init__(self, coins: List[str], path: str = settings.BARS_FILE):
        self.coins = list(coins)
        self.path = path
        self.closes = pd.DataFrame(columns=self.coins, dtype=float)
        self.last_poll: Dict[str, tuple] = {}     # coin -> (time, price) of latest Roostoo poll
        self.session = requests.Session()

    # ------------------------------------------------------------------
    def warmup(self) -> str:
        """Load persisted bars, then top up from Binance. Returns a source note."""
        if os.path.exists(self.path):
            saved = pd.read_csv(self.path, index_col=0, parse_dates=True)
            saved.index = pd.to_datetime(saved.index, utc=True)
            self.closes = saved.reindex(columns=self.coins).astype(float)
        fetched = {}
        for coin in self.coins:
            try:
                fetched[coin] = fetch_binance_closes(coin, limit=settings.WARMUP_HOURS, session=self.session)
            except ConnectionError as exc:
                log.warning("warmup: %s", exc)
        if fetched:
            fresh = pd.DataFrame(fetched)
            self.closes = fresh.combine_first(self.closes) if not self.closes.empty else fresh
            self.closes = self.closes.reindex(columns=self.coins).sort_index()
        self._save()
        src = "binance" if len(fetched) == len(self.coins) else (
            f"binance partial ({len(fetched)}/{len(self.coins)}) + saved" if fetched else "saved bars only")
        return f"{len(self.closes)} bars, last {self.last_bar_time()} ({src})"

    def last_bar_time(self) -> Optional[pd.Timestamp]:
        return None if self.closes.empty else self.closes.index[-1]

    def ready(self, min_bars: int) -> bool:
        return len(self.closes.dropna()) >= min_bars

    # ------------------------------------------------------------------
    def on_poll(self, now: pd.Timestamp, prices: Dict[str, float]) -> None:
        for coin in self.coins:
            if prices.get(coin):
                self.last_poll[coin] = (now, float(prices[coin]))

    def close_completed_bars(self, now: pd.Timestamp) -> List[pd.Timestamp]:
        """Append every bar that has closed since the last one. Returns new labels."""
        current_open = now.floor("h")
        last = self.last_bar_time()
        start = current_open - HOUR if last is None else last + HOUR
        if start >= current_open:
            return []
        missing = pd.date_range(start, current_open - HOUR, freq="h", tz="UTC")
        got = {}
        for coin in self.coins:
            try:
                s = fetch_binance_closes(coin, start=missing[0], limit=len(missing) + 1, session=self.session)
                got[coin] = s.reindex(missing)
            except ConnectionError as exc:
                log.warning("bar close: %s; falling back to Roostoo polls", exc)
        new = pd.DataFrame(got, index=missing).reindex(columns=self.coins)
        # Fallback: last Roostoo poll seen before the bar closed.
        for coin in self.coins:
            poll = self.last_poll.get(coin)
            if poll is not None and poll[0] < current_open + pd.Timedelta(minutes=5):
                new[coin] = new[coin].fillna(poll[1])
        combined = pd.concat([self.closes, new])
        self.closes = combined[~combined.index.duplicated(keep="last")].ffill()
        self._save()
        return list(missing)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.closes.tail(settings.WARMUP_HOURS + 200).to_csv(self.path)
