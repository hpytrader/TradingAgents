"""Ward's desk: the state of every live trade, and checked changes to it.

Each time the agents meet, Ward (the trade manager) reviews the pending and
open trades from a snapshot built here in code: the price now, the result in R
so far, the best and worst it has been, and the time left. His decisions come
back as actions, and :func:`apply` checks each one before it touches the
journal:

- **pending** orders can only be held or cancelled;
- **open** trades can be held, closed at the current price, have their target
  moved (still beyond price), or have their stop **tightened**: a stop is never
  moved further away, and never to the wrong side of price.

Anything that breaks a rule is refused with the reason, and the trade is left
as it was.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from tradingagents.fx.instruments import spec_for
from tradingagents.fx.journal import OPEN, PENDING, Entry, Journal, day_close

CandleFetcher = Callable[[str, str, int], pd.DataFrame]
QuoteFetcher = Callable[[str], object]


@dataclass
class Position:
    entry: Entry
    mid: float
    spread: float
    r_now: float | None          # open trades: result so far in R (before the spread)
    best_r: float | None         # open trades: best and worst since the fill
    worst_r: float | None
    minutes: int                 # since the fill (open) or the order (pending)
    minutes_left: int            # to the cancel time (pending) or the New York close (open)

    @property
    def ticket(self) -> str:
        return self.entry.ticket


def snapshot(entries: list[Entry], candles: CandleFetcher, quote: QuoteFetcher,
             now: datetime) -> list[Position]:
    """Live state of each pending and open trade; one that cannot be priced is left out."""
    out = []
    for e in entries:
        if e.status not in (PENDING, OPEN):
            continue
        try:
            q = quote(e.symbol)
            bid, ask = float(q.bid), float(q.ask)
        except Exception:
            continue
        mid, sign, risk = (bid + ask) / 2, (1 if e.long else -1), e.risk or 1e-9
        r_now = best = worst = None
        if e.status == OPEN:
            r_now = round((mid - e.entry) * sign / risk, 2)
            try:
                minutes = int((now - e.filled_at).total_seconds() // 60) + 2
                bars = candles(e.symbol, "M1", max(min(minutes, 4900), 5))
                bars = bars[bars.index >= pd.Timestamp(e.filled_at)]
            except Exception:
                bars = None
            if bars is not None and len(bars):
                hi, lo = float(bars["high"].max()), float(bars["low"].min())
                best = round(((hi if e.long else lo) - e.entry) * sign / risk, 2)
                worst = round(((lo if e.long else hi) - e.entry) * sign / risk, 2)
            since, until = e.filled_at, day_close(e.filled_at)
        else:
            since, until = e.created_at, e.expires_at
        out.append(Position(entry=e, mid=mid, spread=max(ask - bid, 0.0), r_now=r_now, best_r=best,
                            worst_r=worst, minutes=int((now - since).total_seconds() // 60),
                            minutes_left=max(int((until - now).total_seconds() // 60), 0)))
    return out


def describe(positions: list[Position]) -> str:
    """The open book as the agents read it."""
    if not positions:
        return "(no pending or open trades)"
    lines = []
    for p in positions:
        e, spec = p.entry, spec_for(p.entry.symbol)
        price = spec.round_price(p.mid)
        head = (f"- {e.ticket} {e.symbol} {e.order_type} entry {e.entry}, stop {e.stop}, target {e.target}; "
                f"price now {price}")
        if e.status == OPEN:
            lines.append(f"{head}. OPEN {p.minutes} min, now {p.r_now:+.2f}R"
                         + (f" (best {p.best_r:+.2f}R, worst {p.worst_r:+.2f}R)" if p.best_r is not None else "")
                         + f"; {p.minutes_left} min to the New York close.")
        else:
            lines.append(f"{head}. PENDING {p.minutes} min, "
                         f"{spec.pips(abs(p.mid - e.entry)):.1f} pips from the entry; "
                         f"{p.minutes_left} min to its cancel time.")
        if e.changes:
            lines.append("  earlier changes: " + "; ".join(
                f"{c['action']} {c.get('old', '')}→{c.get('new', c.get('price', ''))}" for c in e.changes))
        if e.watch_for:
            lines.append(f"  watch for (from the original plan): {e.watch_for}")
    return "\n".join(lines)


@dataclass
class Applied:
    entry: Entry
    action: str
    text: str            # what happened, for the chat and Telegram
    ok: bool


def apply(actions: list[dict], positions: list[Position], journal: Journal, *, now: datetime,
          by: str = "Ward") -> list[Applied]:
    """Check and apply the trade manager's actions; refused ones are reported, not applied.

    Each trade is re-read from the journal first: the decision may have taken a
    minute or two, and a trade that has since filled, hit its target or its stop
    is acted on as it is now, or not at all.
    """
    current = {e.id: e for e in journal.entries()}
    by_ticket = {p.ticket: p for p in positions}
    results: list[Applied] = []
    for a in actions:
        ticket = str(a.get("ticket", "")).strip()
        if ticket and not ticket.startswith("#"):
            ticket = f"#{ticket}"
        action = str(a.get("action", "hold")).lower()
        reason = str(a.get("reason", "")).strip() or "no reason given"
        p = by_ticket.get(ticket)
        if p is None:
            continue                                  # not a live trade: nothing to do
        e = current.get(p.entry.id, p.entry)
        p.entry = e
        spec = spec_for(e.symbol)
        if action == "hold":
            continue
        if e.status not in (PENDING, OPEN):
            results.append(Applied(e, action, f"{e.label}: {action} not applied, the trade "
                                              f"already finished ({e.status})", False))
            continue

        def refuse(why: str, e=e, action=action):
            results.append(Applied(e, action, f"{e.label}: {action} refused, {why}", False))

        if e.status == PENDING:
            if action != "cancel":
                refuse("a pending order can only be held or cancelled")
                continue
            journal.cancel(e, now=now, by=by, reason=reason)
            results.append(Applied(e, action, f"{e.label}: pending order cancelled. {reason}", True))
            continue

        if action == "cancel":
            refuse("the order has already filled; close it instead")
            continue
        sign = 1 if e.long else -1
        if action == "close":
            price = spec.round_price(p.mid)
            journal.close_now(e, price=price, now=now, by=by, reason=reason)
            results.append(Applied(e, action, f"{e.label}: closed early at {price} "
                                              f"({e.result_r:+.2f}R). {reason}", True))
        elif action == "move_stop":
            try:
                new = spec.round_price(float(a.get("new_stop")))
            except (TypeError, ValueError):
                refuse("no new stop given")
                continue
            if (new - e.stop) * sign <= 0:
                refuse(f"{new} would widen the stop from {e.stop}; stops are only tightened")
                continue
            if (p.mid - new) * sign <= p.spread:
                refuse(f"{new} is at or beyond the current price {spec.round_price(p.mid)}")
                continue
            old = e.stop
            journal.modify(e, now=now, by=by, reason=reason, stop=new)
            results.append(Applied(e, action, f"{e.label}: stop moved {old} → {new}. {reason}", True))
        elif action == "move_target":
            try:
                new = spec.round_price(float(a.get("new_target")))
            except (TypeError, ValueError):
                refuse("no new target given")
                continue
            if (new - p.mid) * sign <= p.spread:
                refuse(f"{new} is not beyond the current price {spec.round_price(p.mid)}")
                continue
            old = e.target
            journal.modify(e, now=now, by=by, reason=reason, target=new)
            results.append(Applied(e, action, f"{e.label}: target moved {old} → {new}. {reason}", True))
        else:
            refuse(f"unknown action {action!r}")
    return results
