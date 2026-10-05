"""This week's economic calendar, for the forex agents' event-risk check.

A limit order that fills an hour before a central-bank decision or a jobs
report is a different trade from the one its chart describes: the release can
move price through the stop in seconds. The scanner sees only price, so the
calendar is what lets the agents see those moments coming.

Source: the Forex Factory weekly calendar feed (no key). It lists each event's
currency, time, impact (High / Medium / Low / Holiday), forecast and previous
value. The feed limits how often it may be fetched, so a copy is cached on disk
for an hour and reused by every scan in that hour.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorUnavailableError

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
REQUEST_TIMEOUT = 20
CACHE_SECONDS = 3600
IMPACTS = ("High", "Medium", "Low", "Holiday")

# Gold and silver trade against the dollar, so US releases move them; the
# calendar has no XAU/XAG entries of its own.
_CURRENCY_FOR = {"XAU": "USD", "XAG": "USD", "XPT": "USD", "XPD": "USD"}


@dataclass(frozen=True)
class Event:
    title: str
    currency: str            # USD, EUR, ... or "All"
    time: datetime           # UTC
    impact: str              # High, Medium, Low or Holiday
    forecast: str = ""
    previous: str = ""

    def describe(self) -> str:
        figures = ", ".join(part for part in (
            f"forecast {self.forecast}" if self.forecast else "",
            f"previous {self.previous}" if self.previous else "",
        ) if part)
        tail = f" ({figures})" if figures else ""
        return f"{self.time:%a %H:%M} UTC · {self.currency} · {self.impact}: {self.title}{tail}"


def _cache_path() -> Path:
    return Path(get_config()["data_cache_dir"]) / "forex_calendar" / "thisweek.json"


def _download() -> list[dict]:
    try:
        response = requests.get(FEED_URL, timeout=REQUEST_TIMEOUT,
                                headers={"User-Agent": "TradingAgents fx-scan"})
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"economic calendar unreachable: {exc}") from None
    if response.status_code != 200:
        raise VendorUnavailableError(f"economic calendar returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise VendorUnavailableError("economic calendar returned a response that is not JSON") from None
    if not isinstance(data, list):
        raise VendorUnavailableError("economic calendar returned an unexpected shape")
    return data


def _raw_events() -> list[dict]:
    """The week's feed, from the cache when it is under an hour old.

    A stale cache is still used when the feed cannot be reached: last hour's
    calendar is far better than none, and events do not move often.
    """
    path = _cache_path()
    fresh = path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS
    if fresh:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        data = _download()
    except VendorUnavailableError:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
    return data


def parse(raw: list[dict]) -> list[Event]:
    """Feed rows → events in UTC, oldest first; unreadable rows are skipped."""
    events = []
    for row in raw:
        try:
            when = datetime.fromisoformat(str(row["date"])).astimezone(UTC)
        except (KeyError, ValueError):
            continue
        impact = str(row.get("impact", "")).strip().title()
        events.append(Event(
            title=str(row.get("title", "")).strip(),
            currency=str(row.get("country", "")).strip().upper(),
            time=when,
            impact=impact if impact in IMPACTS else "Low",
            forecast=str(row.get("forecast") or "").strip(),
            previous=str(row.get("previous") or "").strip(),
        ))
    return sorted(events, key=lambda e: e.time)


def currencies_for(symbol: str) -> set[str]:
    """The calendar currencies that move a pair: both legs, metals as USD."""
    s = symbol.upper()
    return {_CURRENCY_FOR.get(s[:3], s[:3]), _CURRENCY_FOR.get(s[3:], s[3:])}


def events_between(start: datetime, end: datetime, currencies: set[str] | None = None,
                   impacts: tuple[str, ...] = ("High", "Medium"),
                   raw: list[dict] | None = None) -> list[Event]:
    """Events in ``[start, end]`` for ``currencies`` (plus "All"), at the given impacts.

    Raises ``VendorUnavailableError`` when there is neither a live feed nor a
    cached copy; callers must then say the calendar is unknown, never empty.
    """
    events = parse(raw if raw is not None else _raw_events())
    wanted = None if currencies is None else {c.upper() for c in currencies} | {"ALL"}
    return [
        e for e in events
        if start <= e.time <= end
        and e.impact in impacts
        and (wanted is None or e.currency in wanted)
    ]
