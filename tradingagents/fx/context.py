"""What the forex agents read besides price: the calendar and the news.

Gathered once per scan, in code, before any model is called, so every agent
argues from the same evidence and a missing source is stated as missing rather
than left for a model to guess about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.dataflows.vendors import forex_calendar
from tradingagents.dataflows.vendors.forex_calendar import Event
from tradingagents.fx.scanner import Setup

# How far back the agents see released data: a surprise from this morning
# still drives the session.
RECENT_EVENTS = timedelta(hours=12)
NEWS_DAYS = 2


@dataclass
class ReviewContext:
    now: datetime
    upcoming: dict[str, list[Event]] = field(default_factory=dict)   # per symbol, until expiry
    recent: list[Event] = field(default_factory=list)                # released in the last 12h
    calendar_note: str = ""                                          # set when the calendar is unknown
    news: dict[str, str] = field(default_factory=dict)               # per symbol
    global_news: str = ""


def _default_news(symbol: str, start: str, end: str) -> str:
    from tradingagents.dataflows.router import route_to_vendor
    return route_to_vendor("get_news", symbol, start, end)


def _default_global_news(as_of: str) -> str:
    from tradingagents.dataflows.router import route_to_vendor
    return route_to_vendor("get_global_news", as_of)


def gather(
    candidates: list[Setup],
    now: datetime,
    *,
    calendar: Callable[..., list[Event]] = forex_calendar.events_between,
    news: Callable[[str, str, str], str] | None = _default_news,
    global_news: Callable[[str], str] | None = _default_global_news,
) -> ReviewContext:
    """Calendar events and headlines for the candidates; failures are noted, not fatal."""
    ctx = ReviewContext(now=now)
    try:
        for s in candidates:
            ctx.upcoming[s.symbol] = calendar(now, s.expires_at, forex_calendar.currencies_for(s.symbol))
        currencies = set().union(*(forex_calendar.currencies_for(s.symbol) for s in candidates)) \
            if candidates else set()
        ctx.recent = calendar(now - RECENT_EVENTS, now, currencies)
    except Exception as exc:
        ctx.upcoming = {}
        ctx.recent = []
        ctx.calendar_note = (f"The economic calendar could not be loaded ({exc}). Event risk is "
                             "UNKNOWN, not absent: treat every setup as possibly exposed to news.")

    start = (now - timedelta(days=NEWS_DAYS)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    if news is not None:
        for s in candidates:
            try:
                ctx.news[s.symbol] = str(news(s.symbol, start, end))[:4000]
            except Exception as exc:
                ctx.news[s.symbol] = f"(news unavailable: {exc})"
    if global_news is not None:
        try:
            ctx.global_news = str(global_news(end))[:6000]
        except Exception as exc:
            ctx.global_news = f"(macro headlines unavailable: {exc})"
    return ctx
