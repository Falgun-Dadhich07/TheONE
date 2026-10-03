"""Thin, careful wrapper around the Roostoo REST API.

Design notes
- Every request returns HTTP 200 even on failure, so we ALWAYS check `Success`.
- Read-only GET calls are retried on network errors. Order-placing calls are
  NOT retried automatically, because a blind retry could duplicate an order.
- Signed endpoints: HMAC-SHA256 over the sorted "k=v&k=v" string (query string
  for GET, request body for POST), sent in the MSG-SIGNATURE header.
"""
import hashlib
import hmac
import logging
import time
from typing import Any, Optional

import requests

import config

log = logging.getLogger(__name__)


class RoostooError(Exception):
    """Raised when the API answers Success=false or the request fails."""


class RoostooClient:
    def __init__(self, timeout: float = 10.0, max_retries: int = 3):
        self.base_url = config.BASE_URL.rstrip("/")
        self.api_key, self.secret = config.get_credentials()
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _timestamp() -> str:
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> tuple[dict, str]:
        """Add timestamp, build the sorted param string, return (headers, total_params)."""
        params = dict(params)
        params["timestamp"] = self._timestamp()
        total_params = "&".join(f"{k}={params[k]}" for k in sorted(params))
        signature = hmac.new(
            self.secret.encode("utf-8"), total_params.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": signature}
        return headers, total_params

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        signed: bool = False,
        retry: bool = False,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        params = params or {}
        attempts = self.max_retries if retry else 1
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                if signed:
                    headers, total_params = self._sign(params)
                    if method == "GET":
                        res = self.session.get(
                            f"{url}?{total_params}", headers=headers, timeout=self.timeout
                        )
                    else:
                        headers["Content-Type"] = "application/x-www-form-urlencoded"
                        res = self.session.post(
                            url, headers=headers, data=total_params, timeout=self.timeout
                        )
                else:
                    # timestamp-only endpoints (e.g. ticker) still need a timestamp
                    q = dict(params)
                    q["timestamp"] = self._timestamp()
                    res = self.session.get(url, params=q, timeout=self.timeout)

                res.raise_for_status()
                return res.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                log.warning("%s %s failed (attempt %d/%d): %s", method, path, attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 8))

        raise RoostooError(f"{method} {path} failed: {last_exc}")

    @staticmethod
    def _check(data: dict, what: str) -> dict:
        if not data.get("Success", True):
            raise RoostooError(f"{what} rejected: {data.get('ErrMsg', 'unknown error')}")
        return data

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------
    def server_time(self) -> int:
        data = self._request("GET", "/v3/serverTime", retry=True)
        return data["ServerTime"]

    def exchange_info(self) -> dict:
        return self._request("GET", "/v3/exchangeInfo", retry=True)

    def ticker(self, pair: Optional[str] = None) -> dict:
        """Return {pair: {MaxBid, MinAsk, LastPrice, Change, ...}}."""
        params = {"pair": pair} if pair else {}
        data = self._request("GET", "/v3/ticker", params=params, retry=True)
        return self._check(data, "ticker")["Data"]

    # ------------------------------------------------------------------
    # Signed endpoints
    # ------------------------------------------------------------------
    def balance(self) -> dict:
        data = self._request("GET", "/v3/balance", signed=True, retry=True)
        data = self._check(data, "balance")
        for key in ("Wallet", "SpotWallet"):
            if key in data:
                return data[key]
        raise RoostooError(f"balance: unexpected response {data}")

    def pending_count(self) -> dict:
        # Success=false here just means "no pending orders", so no _check.
        return self._request("GET", "/v3/pending_count", signed=True, retry=True)

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> dict:
        """Place a MARKET or LIMIT order. Not auto-retried (could duplicate)."""
        order_type = order_type.upper()
        params = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type,
            "quantity": str(quantity),
        }
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require a price")
            params["price"] = str(price)
        data = self._request("POST", "/v3/place_order", params=params, signed=True)
        return self._check(data, f"{side} {pair}")["OrderDetail"]

    def query_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
    ) -> list:
        params: dict = {}
        if order_id is not None:
            params["order_id"] = str(order_id)  # no other params allowed with order_id
        else:
            if pair:
                params["pair"] = pair
            if pending_only is not None:
                params["pending_only"] = "TRUE" if pending_only else "FALSE"
        data = self._request("POST", "/v3/query_order", params=params, signed=True, retry=True)
        if not data.get("Success") and data.get("ErrMsg") == "no order matched":
            return []
        return self._check(data, "query_order")["OrderMatched"]

    def cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None) -> list:
        params: dict = {}
        if order_id is not None:
            params["order_id"] = str(order_id)
        elif pair:
            params["pair"] = pair
        data = self._request("POST", "/v3/cancel_order", params=params, signed=True)
        return self._check(data, "cancel_order").get("CanceledList", [])
