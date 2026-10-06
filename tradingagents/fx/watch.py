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

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from tradingagents.fx import smc
from tradingagents.fx.journal import ACTIVE, Change, Entry, Journal, stats
from tradingagents.fx.scanner import ScanResult, Setup
from tradingagents.fx.smc_scanner import ScanWindow
from tradingagents.fx.telegram import order_message, result_message, summary_message

logger = logging.getLogger(__name__)

Notify = Callable[[str], None]
ScanFn = Callable[[datetime], ScanResult]
ReviewFn = Callable[[ScanResult, datetime], tuple[object | None, str]]   # (Review, report path)
WardFn = Callable[[datetime], object | None]                             # live-book check -> Review


@dataclass
class WatchState:
    day: date | None = None
    seen: set[str] = field(default_factory=set)
    agent_runs: int = 0
    cap_warned: bool = False
    in_window: bool = False
    last_ward: datetime | None = None      # when the trade manager last looked at the live book


@dataclass
class Cycle:
    at: datetime
    in_window: bool
    changes: list[Change] = field(default_factory=list)
    candidates: int = 0
    new_setups: int = 0
    reviewed: bool = False
    ward_checked: bool = False
    new_orders: list[Entry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def file_review(review, journal: Journal, *, now: datetime, report: str = "",
                notify: Notify | None = None, candles=None) -> tuple[list[Entry], list]:
    """After a review: apply the trade manager's actions, then save the chat and the new orders.

    Returns the new journal entries and the trade-manager results. Ward's
    results, applied or refused, are added to the conversation by the Desk
    before it is saved, so the chat shows what actually happened.
    """
    from tradingagents.fx.manage import apply
    from tradingagents.fx.team import team

    applied = []
    actions = getattr(review, "management", None) or []
    if actions and candles is not None:
        # Replay prices up to the decision before changing anything: a change
        # applies from now on, so the minutes the agents spent thinking must be
        # settled under the old stop and target first.
        try:
            for change in journal.settle(candles, now):
                if notify:
                    notify(result_message(change.entry))
        except Exception:
            pass
    if actions:
        applied = apply(actions, getattr(review, "positions", []) or [], journal, now=now,
                        by=team()["trade_manager"].name)
        if applied and hasattr(review, "messages"):
            desk = team()["desk"]
            review.messages.append({"agent": "desk", "name": desk.name, "role": desk.role,
                                    "title": "Trade changes", "kind": "system",
                                    "text": "\n".join(("✓ " if a.ok else "✕ ") + a.text for a in applied)})
        if notify:
            for a in applied:
                if a.ok:
                    notify(f"<b>⚙ {a.text.split(':', 1)[0]}</b>:{a.text.split(':', 1)[1]}")
    review_id = journal.record_review(review, now=now)
    added = []
    if getattr(review, "orders", None):
        added = journal.record(review.orders, now=now, report=report, review_id=review_id,
                               tickets=getattr(review, "tickets", None))
    return added, applied


def setup_key(s: Setup) -> str:
    """One trade idea: the same symbol, direction and zone (or levels) seen again is not new."""
    if s.zone_low is not None:
        return f"{s.symbol}|{s.direction}|{s.zone_low}|{s.zone_high}"
    return f"{s.symbol}|{s.direction}|{s.entry}|{s.stop}"


def _load_progress(journal: Journal, day) -> tuple[set[str], int]:
    """What was already reviewed on ``day``, kept in the journal so a restart doesn't review it again."""
    try:
        seen = set(json.loads(journal.get_meta(f"seen:{day}", "[]")))
        runs = int(journal.get_meta(f"runs:{day}", "0"))
    except Exception:
        return set(), 0
    return seen, runs


def _save_progress(journal: Journal, state: WatchState) -> None:
    try:
        journal.set_meta(f"seen:{state.day}", json.dumps(sorted(state.seen)))
        journal.set_meta(f"runs:{state.day}", str(state.agent_runs))
    except Exception:
        pass                                    # memory still holds it for this run


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
    ward_fn: WardFn | None = None,
    ward_every: timedelta = timedelta(minutes=30),
) -> Cycle:
    """Run one watcher cycle at ``now``; never raises for a data or model problem.

    After the scan, when trades are pending or open and the trade manager has
    not looked at them for ``ward_every`` (in a full review or on his own),
    ``ward_fn`` runs his check. That happens inside and outside the scan
    window, so open trades are watched until they finish.
    """
    day = smc.trading_day(pd.Timestamp(now))
    if state.day != day:
        state.day, state.cap_warned = day, False
        state.seen, state.agent_runs = _load_progress(journal, day)

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
    if inside:
        _scan_and_review(now, report, journal=journal, scan_fn=scan_fn, review_fn=review_fn,
                         notify=notify, state=state, max_agent_runs=max_agent_runs, candles=candles)
    _ward_check(now, report, journal=journal, ward_fn=ward_fn, ward_every=ward_every,
                notify=notify, state=state, candles=candles)
    return report


def _ward_check(now, report: Cycle, *, journal: Journal, ward_fn, ward_every, notify, state,
                candles=None) -> None:
    if ward_fn is None or not journal.entries(ACTIVE):
        return
    if state.last_ward is not None and now - state.last_ward < ward_every:
        return
    state.last_ward = now                       # also on failure: retry next interval, not next cycle
    try:
        review = ward_fn(now)
    except Exception as exc:
        report.notes.append(f"trade manager check failed: {exc}")
        return
    if review is None:
        return
    report.ward_checked = True
    _, applied = file_review(review, journal, now=now, notify=notify, candles=candles)
    report.notes += [a.text for a in applied if not a.ok]


def _scan_and_review(now, report: Cycle, *, journal, scan_fn, review_fn, notify, state, max_agent_runs,
                     candles=None) -> None:
    try:
        result = scan_fn(now)
    except Exception as exc:
        report.notes.append(f"scan failed: {exc}")
        return
    keys = {setup_key(s) for s in result.setups}
    fresh = keys - state.seen
    report.candidates, report.new_setups = len(result.setups), len(fresh)
    if not fresh:
        return
    if state.agent_runs >= max_agent_runs:
        if not state.cap_warned:
            notify(f"<b>FX desk</b>\nDaily agent-review limit ({max_agent_runs}) reached; "
                   "new setups are logged but not reviewed until tomorrow.")
            state.cap_warned = True
        state.seen |= keys
        _save_progress(journal, state)
        report.notes.append("agent-review limit reached")
        return

    try:
        review, report_path = review_fn(result, now)
    except Exception as exc:
        report.notes.append(f"agent review failed: {exc}")
        return
    state.agent_runs += 1
    state.seen |= keys
    _save_progress(journal, state)
    report.reviewed = True
    if review is None:
        return
    if getattr(review, "positions", None):      # the trade manager took part in this review
        state.last_ward = now
    report.new_orders, applied = file_review(review, journal, now=now, report=report_path, notify=notify,
                                             candles=candles)
    report.notes += [a.text for a in applied if not a.ok]
    for entry in report.new_orders:
        notify(order_message(entry))
