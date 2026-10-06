"""The watcher: one cycle every few minutes, all morning, without supervision.

Each cycle:

1. **Settles** the journal on the latest one-minute prices and reports every
   change (filled, won, lost, expired, missed).
2. Inside the scan window, runs the **free scanner**. Only when it finds a
   setup not seen earlier in the trading day does it call the **agents**, so a
   quiet morning costs nothing and a busy one at most ``max_agent_runs`` reviews.
3. **Records** the agents' final orders in the journal and announces each one.
4. When the window closes, sends the morning's **summary** once.

The cycle takes every dependency as an argument (scanner, reviewer, notifier),
so it runs the same way in tests with stand-ins as on live prices.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from tradingagents.fx import smc
from tradingagents.fx.journal import Change, Entry, Journal, stats
from tradingagents.fx.scanner import ScanResult, Setup
from tradingagents.fx.smc_scanner import ScanWindow
from tradingagents.fx.telegram import order_message, result_message, summary_message

logger = logging.getLogger(__name__)

Notify = Callable[[str], None]
ScanFn = Callable[[datetime], ScanResult]
ReviewFn = Callable[[ScanResult, datetime], tuple[object | None, str]]   # (Review, report path)


@dataclass
class WatchState:
    day: date | None = None
    seen: set[str] = field(default_factory=set)
    agent_runs: int = 0
    cap_warned: bool = False
    in_window: bool = False


@dataclass
class Cycle:
    at: datetime
    in_window: bool
    changes: list[Change] = field(default_factory=list)
    candidates: int = 0
    new_setups: int = 0
    reviewed: bool = False
    new_orders: list[Entry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def setup_key(s: Setup) -> str:
    """One trade idea: the same symbol, direction and zone (or levels) seen again is not new."""
    if s.zone_low is not None:
        return f"{s.symbol}|{s.direction}|{s.zone_low}|{s.zone_high}"
    return f"{s.symbol}|{s.direction}|{s.entry}|{s.stop}"


def cycle(
    now: datetime,
    *,
    window: ScanWindow,
    journal: Journal,
    candles,
    scan_fn: ScanFn,
    review_fn: ReviewFn,
    notify: Notify,
    state: WatchState,
    max_agent_runs: int = 12,
) -> Cycle:
    """Run one watcher cycle at ``now``; never raises for a data or model problem."""
    day = smc.trading_day(pd.Timestamp(now))
    if state.day != day:
        state.day, state.seen, state.agent_runs, state.cap_warned = day, set(), 0, False

    inside = window.current_end(now) is not None
    report = Cycle(at=now, in_window=inside)

    try:
        report.changes = journal.settle(candles, now)
    except Exception as exc:
        report.notes.append(f"settling failed: {exc}")
    for change in report.changes:
        notify(result_message(change.entry))

    if state.in_window and not inside:
        notify(summary_message(stats(journal.entries()), f"{now.astimezone(smc.NEW_YORK):%a %d %b}"))
    state.in_window = inside
    if not inside:
        return report

    try:
        result = scan_fn(now)
    except Exception as exc:
        report.notes.append(f"scan failed: {exc}")
        return report
    keys = {setup_key(s) for s in result.setups}
    fresh = keys - state.seen
    report.candidates, report.new_setups = len(result.setups), len(fresh)
    if not fresh:
        return report
    if state.agent_runs >= max_agent_runs:
        if not state.cap_warned:
            notify(f"<b>FX desk</b>\nDaily agent-review limit ({max_agent_runs}) reached; "
                   "new setups are logged but not reviewed until tomorrow.")
            state.cap_warned = True
        state.seen |= keys
        report.notes.append("agent-review limit reached")
        return report

    try:
        review, report_path = review_fn(result, now)
    except Exception as exc:
        report.notes.append(f"agent review failed: {exc}")
        return report
    state.agent_runs += 1
    state.seen |= keys
    report.reviewed = True
    if review is None or not getattr(review, "orders", None):
        return report
    report.new_orders = journal.record(review.orders, now=now, report=report_path)
    for entry in report.new_orders:
        notify(order_message(entry))
    return report
