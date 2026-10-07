"""OANDA v20 price vendor for spot forex and metals.

Broker-grade candles and bid/ask quotes for the forex scanner. Yahoo has no spot
gold or silver (it maps them to COMEX futures) and its forex bars are composites,
so a limit order placed off them can sit a few pips away from where a broker
fills. OANDA quotes the spot instruments a retail forex account actually trades.

Configuration comes from the environment:

    OANDA_API_TOKEN     personal access token (Account → Manage API Access)
    OANDA_ENVIRONMENT   "practice" (default) or "live"

A practice (demo) account is free and its token reads the same live prices, so
no money needs to be in an account to use the scanner. The token is sent in a
header, never in the URL, so request errors cannot carry it into a log.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)

BASE_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

REQUEST_TIMEOUT = 30

# OANDA allows up to 5000 candles per request; the scanner needs a few hundred.
MAX_COUNT = 5000

GRANULARITIES = frozenset({"S5", "M1", "M5", "M15", "M30", "H1", "H4", "D"})


class OandaNotConfiguredError(VendorNotConfiguredError):
    """OANDA was asked for data without a token, or the token was refused."""


@dataclass(frozen=True)
class Quote:
    """The latest bid and ask for an instrument."""

    symbol: str
    bid: float
    ask: float
    time: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


def to_oanda_instrument(symbol: str) -> str:
    """``EURUSD`` / ``EUR/USD`` / ``XAUUSD+`` → ``EUR_USD`` / ``XAU_USD``.

    Six letters split three and three, which covers every forex pair and the
    metals OANDA lists against a currency (XAU, XAG, XPT, XPD). A symbol that
    already carries OANDA's underscore is passed through.
    """
    s = symbol.strip().upper().rstrip("+")
    if "_" in s:
        return s
    s = s.replace("/", "").replace("-", "")
    if len(s) != 6 or not s.isalpha():
        raise ValueError(f"Not a forex or metal pair: {symbol!r}")
    return f"{s[:3]}_{s[3:]}"


def _settings() -> tuple[str, str]:
    token = os.getenv("OANDA_API_TOKEN", "").strip()
    if not token:
        raise OandaNotConfiguredError(
            "OANDA_API_TOKEN is not set. Create a free practice account at oanda.com, "
            "generate a token under Manage API Access, and add it to your .env file."
        )
    env = (os.getenv("OANDA_ENVIRONMENT") or "practice").strip().lower()
    if env not in BASE_URLS:
        raise OandaNotConfiguredError(
            f"OANDA_ENVIRONMENT must be 'practice' or 'live', not {env!r}."
        )
    return token, BASE_URLS[env]


def _get(path: str, params: dict) -> dict:
    token, base = _settings()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }
    try:
        response = requests.get(f"{base}{path}", params=params, headers=headers,
                                timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        # The token lives in a header, which requests does not quote in errors.
        raise VendorUnavailableError(f"OANDA request failed: {exc}") from None

    if response.status_code == 401:
        raise OandaNotConfiguredError(
            "OANDA refused the token (401). Check OANDA_API_TOKEN, and that "
            "OANDA_ENVIRONMENT matches the account it came from (practice or live)."
        )
    if response.status_code == 429 or response.status_code >= 500:
        raise VendorUnavailableError(
            f"OANDA is unavailable right now (HTTP {response.status_code})."
        )
    if response.status_code >= 400:
        message = ""
        with contextlib.suppress(ValueError, AttributeError):
            message = response.json().get("errorMessage", "")
        raise VendorUnavailableError(
            f"OANDA rejected the request (HTTP {response.status_code}): {message}".rstrip(": ")
        )
    try:
        return response.json()
    except ValueError:
        raise VendorUnavailableError("OANDA returned a response that is not JSON.") from None


def _candles_json(instrument: str, granularity: str, count: int, price: str) -> list[dict]:
    if granularity not in GRANULARITIES:
        raise ValueError(f"Unsupported granularity {granularity!r}")
    count = max(1, min(int(count), MAX_COUNT))
    data = _get(f"/v3/instruments/{instrument}/candles",
                {"granularity": granularity, "count": count, "price": price})
    return data.get("candles") or []


def _frame(candles: list[dict], include_incomplete: bool = False) -> pd.DataFrame:
    rows = []
    for candle in candles:
        if not include_incomplete and not candle.get("complete", False):
            continue
        mid = candle.get("mid") or {}
        try:
            rows.append({
                "time": candle["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(candle.get("volume", 0)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time").sort_index()


def get_candles(symbol: str, granularity: str = "H1", count: int = 300,
                include_incomplete: bool = False) -> pd.DataFrame:
    """Mid-price OHLCV bars, oldest first, indexed by UTC bar open time.

    The still-forming bar is dropped unless ``include_incomplete``: a level or
    an indicator read off a bar that has not closed changes until it does.
    """
    instrument = to_oanda_instrument(symbol)
    frame = _frame(_candles_json(instrument, granularity, count, "M"), include_incomplete)
    if frame.empty:
        raise NoMarketDataError(symbol, instrument, f"no {granularity} candles from OANDA")
    return frame


def get_candle_history(symbol: str, granularity: str, start: datetime, end: datetime,
                       progress=None) -> pd.DataFrame:
    """Every complete mid-price bar from ``start`` to ``end`` (UTC), fetched 5,000 at a time.

    For backtests: OANDA serves years of history to any account, including a
    free practice one. ``progress`` is called with each page's last bar time.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"Unsupported granularity {granularity!r}")
    instrument = to_oanda_instrument(symbol)
    frames, cursor = [], start
    while cursor < end:
        data = _get(f"/v3/instruments/{instrument}/candles",
                    {"granularity": granularity, "price": "M", "count": MAX_COUNT,
                     "from": cursor.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                     "includeFirst": "true"})
        page = _frame(data.get("candles") or [])
        page = page[page.index < pd.Timestamp(end)]
        if page.empty:
            break
        frames.append(page)
        last = page.index[-1].to_pydatetime()
        if progress:
            progress(last)
        if last <= cursor:
            break
        cursor = last + timedelta(seconds=1)
    if not frames:
        return _frame([])
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index()


def get_quote(symbol: str) -> Quote:
    """The latest bid and ask, from the newest five-second bar."""
    instrument = to_oanda_instrument(symbol)
    candles = _candles_json(instrument, "S5", 1, "BA")
    if not candles:
        raise NoMarketDataError(symbol, instrument, "no current price from OANDA")
    candle = candles[-1]
    try:
        bid = float(candle["bid"]["c"])
        ask = float(candle["ask"]["c"])
    except (KeyError, TypeError, ValueError):
        raise NoMarketDataError(symbol, instrument, "OANDA price has no bid/ask") from None
    return Quote(symbol=symbol.upper().rstrip("+"), bid=bid, ask=ask,
                 time=pd.Timestamp(candle["time"]).to_pydatetime())
