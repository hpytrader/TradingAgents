"""The trade journal: every final order the agents give, and what then happened.

Orders are paper trades. Nothing here places an order at a broker: the
settler replays OANDA's one-minute prices after each order was suggested and
records what a limit order with that entry, stop and target would have done.

Settlement rules, applied bar by bar on mid prices:

- **Pending** until price touches the entry. If the target trades first, the
  order is ``missed``: the move happened without it, so it is cancelled. If the
  cancel time passes first, it is ``expired``.
- **Open** once filled. The stop and target are then checked on every bar; a
  bar that reaches both counts as the stop (the conservative reading). A trade
  still open at the New York 17:00 close is ``closed`` at that bar's price.
- A bar that touches the entry and the target together is ``missed``: within
  one minute the order of the two cannot be known, and a fill is not assumed.

Results are in R, net of the spread recorded at the scan: a win pays
``(reward - spread) / (risk + spread)``, a loss costs ``-1``.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.fx.instruments import spec_for
from tradingagents.fx.smc import NEW_YORK

CandleFetcher = Callable[[str, str, int], pd.DataFrame]

PENDING, OPEN = "pending", "open"
WON, LOST, CLOSED, EXPIRED, MISSED = "won", "lost", "closed", "expired", "missed"
CANCELLED = "cancelled"               # a pending order the trade manager withdrew
ACTIVE = (PENDING, OPEN)
FINISHED = (WON, LOST, CLOSED)        # filled and done: these carry an R result
UNFILLED = (EXPIRED, MISSED, CANCELLED)
FIRST_TICKET = 1001

COLUMNS = (
    "id", "created_at", "symbol", "direction", "entry", "stop", "target", "planned_rr",
    "spread", "expires_at", "conviction", "rationale", "watch_for", "strategy", "scanner_score",
    "report", "status", "filled_at", "exit_at", "exit_price", "result_r", "note", "notified",
    "review_id", "ticket", "initial_stop", "settled_to", "changes",
)

# Columns added after the first release, with how an older journal fills them.
ADDED_COLUMNS = {"review_id": "TEXT", "ticket": "TEXT", "initial_stop": "REAL",
                 "settled_to": "TEXT", "changes": "TEXT"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry REAL NOT NULL,
    stop REAL NOT NULL,
    target REAL NOT NULL,
    planned_rr REAL,
    spread REAL DEFAULT 0,
    expires_at TEXT NOT NULL,
    conviction TEXT,
    rationale TEXT,
    watch_for TEXT,
    strategy TEXT,
    scanner_score REAL,
    report TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    filled_at TEXT,
    exit_at TEXT,
    exit_price REAL,
    result_r REAL,
    note TEXT,
    notified TEXT,
    review_id TEXT,
    ticket TEXT,
    initial_stop REAL,
    settled_to TEXT,
    changes TEXT
)
"""

META = "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"

REVIEWS = """
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    at TEXT NOT NULL,
    summary TEXT,
    orders INTEGER DEFAULT 0,
    messages TEXT NOT NULL
)
"""


@dataclass
class Entry:
    """One journal row."""

    id: str
    created_at: datetime
    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    planned_rr: float | None
    spread: float
    expires_at: datetime
    conviction: str = "low"
    rationale: str = ""
    watch_for: str = ""
    strategy: str = "smc"
    scanner_score: float | None = None
    report: str = ""
    status: str = PENDING
    filled_at: datetime | None = None
    exit_at: datetime | None = None
    exit_price: float | None = None
    result_r: float | None = None
    note: str = ""
    notified: str = ""
    review_id: str = ""
    ticket: str = ""
    initial_stop: float | None = None     # the stop at entry: R is always measured against it
    settled_to: datetime | None = None    # prices replayed up to here; later changes apply after it
    changes: list = field(default_factory=list)   # trade-manager actions, oldest first

    @property
    def long(self) -> bool:
        return self.direction == "long"

    @property
    def order_type(self) -> str:
        return "BUY LIMIT" if self.long else "SELL LIMIT"

    @property
    def risk(self) -> float:
        """One R: the distance to the stop the trade was planned with."""
        return abs(self.entry - (self.initial_stop if self.initial_stop is not None else self.stop))

    @property
    def label(self) -> str:
        return f"{self.ticket} {self.symbol}".strip()


@dataclass
class Change:
    entry: Entry
    before: str
    after: str


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Journal:
    """The orders table in a SQLite file, ``<results_dir>/fx_journal.db`` by default."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(SCHEMA)
            db.execute(REVIEWS)
            db.execute(META)
            columns = {row[1] for row in db.execute("PRAGMA table_info(orders)")}
            for name, kind in ADDED_COLUMNS.items():   # journals from earlier versions
                if name not in columns:
                    db.execute(f"ALTER TABLE orders ADD COLUMN {name} {kind}")
            db.execute("UPDATE orders SET initial_stop = stop WHERE initial_stop IS NULL")
            untagged = db.execute("SELECT id FROM orders WHERE ticket IS NULL OR ticket = '' "
                                  "ORDER BY created_at").fetchall()
            for (oid,) in untagged:
                db.execute("UPDATE orders SET ticket = ? WHERE id = ?", (self._next(db), oid))

    @staticmethod
    def _next(db: sqlite3.Connection) -> str:
        row = db.execute("SELECT value FROM meta WHERE key = 'ticket'").fetchone()
        number = int(row[0]) + 1 if row else FIRST_TICKET
        db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('ticket', ?)", (str(number),))
        return f"#{number}"

    def next_tickets(self, symbols: Iterable[str]) -> dict[str, str]:
        """A new ticket for each symbol, e.g. ``{"AUDUSD": "#1043"}``: never reused."""
        with closing(self._connect()) as db, db:
            return {symbol: self._next(db) for symbol in symbols}

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """A stored setting, e.g. what the watcher reviewed today; ``default`` if unset."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # -- reading --------------------------------------------------------------

    def entries(self, statuses: Iterable[str] | None = None) -> list[Entry]:
        query, args = "SELECT * FROM orders", ()
        if statuses is not None:
            statuses = tuple(statuses)
            query += f" WHERE status IN ({','.join('?' * len(statuses))})"
            args = statuses
        with closing(self._connect()) as db:
            rows = db.execute(query + " ORDER BY created_at", args).fetchall()
        return [self._entry(r) for r in rows]

    @staticmethod
    def _entry(row: sqlite3.Row) -> Entry:
        data = dict(row)
        import json

        for key in ("created_at", "expires_at", "filled_at", "exit_at", "settled_to"):
            data[key] = _dt(data[key])
        data["changes"] = json.loads(data["changes"]) if data.get("changes") else []
        data["ticket"] = data.get("ticket") or ""
        data["spread"] = data["spread"] or 0.0
        data["notified"] = data["notified"] or ""
        data["note"] = data["note"] or ""
        data["review_id"] = data.get("review_id") or ""
        return Entry(**data)

    # -- writing --------------------------------------------------------------

    def record_review(self, review, *, now: datetime) -> str:
        """Save a review's conversation; returns its id (empty when it has no messages)."""
        import json

        messages = getattr(review, "messages", None) or []
        if not messages:
            return ""
        rid = f"{now:%Y%m%d-%H%M%S}"
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO reviews (id, at, summary, orders, messages) VALUES (?,?,?,?,?)",
                       (rid, _iso(now), getattr(review, "summary", ""), len(getattr(review, "orders", []) or []),
                        json.dumps(messages)))
        return rid

    def reviews(self, limit: int = 30) -> list[dict]:
        """The latest saved conversations, newest first."""
        import json

        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM reviews ORDER BY at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r["id"], "at": r["at"], "summary": r["summary"] or "", "orders": r["orders"],
                 "messages": json.loads(r["messages"])} for r in rows]

    def record(self, orders: Iterable, *, now: datetime, report: str = "", review_id: str = "",
               tickets: dict[str, str] | None = None) -> list[Entry]:
        """Add the review's final orders; returns the ones that are new.

        A suggestion repeating an order already pending or open for the same
        symbol and direction is the same idea seen again, and is not added.
        """
        active = {(e.symbol, e.direction) for e in self.entries(ACTIVE)}
        added = []
        with closing(self._connect()) as db, db:
            for o in orders:
                if (o.symbol, o.direction) in active:
                    continue
                spec = spec_for(o.symbol)
                entry = Entry(
                    id=f"{now:%Y%m%d-%H%M}-{o.symbol}-{o.direction}",
                    created_at=now, symbol=o.symbol, direction=o.direction,
                    entry=o.entry, stop=o.stop, target=o.target, planned_rr=o.rr,
                    spread=o.setup.spread_pips * spec.pip, expires_at=o.expires_at,
                    conviction=o.conviction, rationale=o.rationale, watch_for=o.watch_for,
                    strategy=o.setup.strategy, scanner_score=o.setup.score, report=report,
                    review_id=review_id, ticket=(tickets or {}).get(o.symbol) or self._next(db),
                    initial_stop=o.stop,
                )
                db.execute(
                    f"INSERT OR IGNORE INTO orders ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                    self._values(entry),
                )
                active.add((o.symbol, o.direction))
                added.append(entry)
        return added

    def import_report(self, path: str | Path) -> list[Entry]:
        """Add the final orders from a saved ``*_agents.json`` report not already journaled."""
        import json
        from types import SimpleNamespace

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        review = data.get("review") or {}
        if not review.get("orders"):
            return []
        created = datetime.fromisoformat(data["scanned_at"])
        spreads = {s["symbol"]: s.get("spread_pips", 0.0) for s in data.get("setups", [])}
        strategies = {s["symbol"]: s.get("strategy", "trend") for s in data.get("setups", [])}
        known = {e.id for e in self.entries()}
        orders = []
        for o in review["orders"]:
            if f"{created:%Y%m%d-%H%M}-{o['symbol']}-{o['direction']}" in known:
                continue
            setup = SimpleNamespace(spread_pips=spreads.get(o["symbol"], 0.0),
                                    strategy=strategies.get(o["symbol"], "trend"),
                                    score=o.get("scanner_score"))
            orders.append(SimpleNamespace(
                symbol=o["symbol"], direction=o["direction"], entry=o["entry"], stop=o["stop"],
                target=o["target"], rr=o.get("rr"), expires_at=datetime.fromisoformat(o["expires_at"]),
                conviction=o.get("conviction", "low"), rationale=o.get("rationale", ""),
                watch_for=o.get("watch_for", ""), setup=setup))
        report = str(Path(path).with_suffix(".md"))
        return self.record(orders, now=created, report=report)

    @staticmethod
    def _values(e: Entry) -> tuple:
        return (e.id, _iso(e.created_at), e.symbol, e.direction, e.entry, e.stop, e.target,
                e.planned_rr, e.spread, _iso(e.expires_at), e.conviction, e.rationale,
                e.watch_for, e.strategy, e.scanner_score, e.report, e.status, _iso(e.filled_at),
                _iso(e.exit_at), e.exit_price, e.result_r, e.note, e.notified, e.review_id,
                e.ticket, e.initial_stop, _iso(e.settled_to), Journal._changes_json(e))

    @staticmethod
    def _changes_json(e: Entry) -> str:
        import json

        return json.dumps(e.changes) if e.changes else ""

    def save(self, e: Entry) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE orders SET status=?, filled_at=?, exit_at=?, exit_price=?, result_r=?, "
                "note=?, notified=?, stop=?, target=?, settled_to=?, changes=? WHERE id=?",
                (e.status, _iso(e.filled_at), _iso(e.exit_at), e.exit_price, e.result_r,
                 e.note, e.notified, e.stop, e.target, _iso(e.settled_to), self._changes_json(e), e.id),
            )

    # -- trade management ---------------------------------------------------

    def _log(self, e: Entry, now: datetime, by: str, action: str, reason: str, **values) -> None:
        e.changes.append({"at": _iso(now), "by": by, "action": action, "reason": reason, **values})

    def modify(self, e: Entry, *, now: datetime, by: str, reason: str,
               stop: float | None = None, target: float | None = None) -> None:
        """Move an open trade's stop and/or target from ``now`` on."""
        if stop is not None and stop != e.stop:
            self._log(e, now, by, "move_stop", reason, old=e.stop, new=stop)
            e.stop = stop
        if target is not None and target != e.target:
            self._log(e, now, by, "move_target", reason, old=e.target, new=target)
            e.target = target
        e.settled_to = max(e.settled_to or now, now)
        self.save(e)

    def close_now(self, e: Entry, *, price: float, now: datetime, by: str, reason: str) -> None:
        """Close an open trade early at ``price`` (paper)."""
        self._log(e, now, by, "close", reason, price=price)
        _exit(e, CLOSED, now, price, f"closed early by {by}: {reason}")
        self.save(e)

    def cancel(self, e: Entry, *, now: datetime, by: str, reason: str) -> None:
        """Withdraw a pending order."""
        self._log(e, now, by, "cancel", reason)
        e.status, e.exit_at, e.note = CANCELLED, now, f"cancelled by {by}: {reason}"
        self.save(e)

    # -- settling -------------------------------------------------------------

    def settle(self, candles: CandleFetcher, now: datetime) -> list[Change]:
        """Replay prices for every pending and open order; returns what changed."""
        active = self.entries(ACTIVE)
        by_symbol: dict[str, list[Entry]] = defaultdict(list)
        for e in active:
            by_symbol[e.symbol].append(e)
        changes = []
        for symbol, entries in by_symbol.items():
            since = min(e.created_at for e in entries)
            bars = _bars(candles, symbol, since, now)
            for e in entries:
                before = e.status
                simulate(e, bars, now)
                if e.status != before or e.filled_at is not None:
                    self.save(e)
                if e.status != before:
                    changes.append(Change(e, before, e.status))
        return changes


def _bars(candles: CandleFetcher, symbol: str, since: datetime, now: datetime) -> pd.DataFrame:
    """One-minute bars from ``since`` (five-minute ones for a span past OANDA's limit)."""
    minutes = int((now - since).total_seconds() // 60) + 5
    if minutes <= 4900:
        frame = candles(symbol, "M1", max(minutes, 10))
    else:
        frame = candles(symbol, "M5", min(minutes // 5 + 5, 5000))
    return frame[frame.index >= pd.Timestamp(since)]


def day_close(t: datetime) -> datetime:
    """The New York 17:00 close that ends the trading day ``t`` falls in (UTC)."""
    local = t.astimezone(NEW_YORK)
    close = NEW_YORK.localize(datetime.combine(local.date(), time(17, 0)))
    if local >= close:
        close = NEW_YORK.localize(datetime.combine(local.date() + timedelta(days=1), time(17, 0)))
    return close.astimezone(UTC)


def net_r(e: Entry, exit_price: float) -> float:
    """Result in R after the spread, the way the scanner priced the plan."""
    sign = 1 if e.long else -1
    move = (exit_price - e.entry) * sign
    if move <= -e.risk + 1e-12:
        return -1.0
    return round((move - e.spread) / (e.risk + e.spread), 2)


def simulate(e: Entry, bars: pd.DataFrame, now: datetime) -> Entry:
    """Advance ``e`` through ``bars`` (mid OHLC, oldest first); updates and returns it."""
    if e.status not in ACTIVE:
        return e
    long = e.long
    if e.status == PENDING:
        bars = bars[bars.index >= pd.Timestamp(e.created_at)]
    else:                                   # resuming an open trade: after what was replayed
        # settled_to is the first moment not yet replayed: just after the last bar
        # seen, or the time of a change, whichever is later.
        if e.settled_to:
            bars = bars[bars.index >= pd.Timestamp(max(e.settled_to, e.filled_at + timedelta(seconds=1)))]
        else:
            bars = bars[bars.index > pd.Timestamp(e.filled_at)]
    rows = list(bars.itertuples())
    k = 0

    if e.status == PENDING:
        while k < len(rows):
            bar = rows[k]
            t = bar.Index.to_pydatetime()
            if t >= e.expires_at:
                e.status, e.note = EXPIRED, "not filled before the cancel time"
                return e
            touched = bar.low <= e.entry if long else bar.high >= e.entry
            reached = bar.high >= e.target if long else bar.low <= e.target
            if reached:
                e.status = MISSED
                e.note = ("target and entry traded in the same minute; no fill assumed" if touched
                          else "target reached before the entry filled; order cancelled")
                e.exit_at = t
                return e
            if touched:
                e.status, e.filled_at = OPEN, t
                stopped = bar.low <= e.stop if long else bar.high >= e.stop
                if stopped:
                    return _exit(e, LOST, t, e.stop, "stopped out in the minute it filled")
                k += 1
                break
            k += 1
        else:
            if now >= e.expires_at:
                e.status, e.note = EXPIRED, "not filled before the cancel time"
            return e

    deadline = day_close(e.filled_at)
    last_close = None
    while k < len(rows):
        bar = rows[k]
        t = bar.Index.to_pydatetime()
        if t >= deadline:
            price = last_close if last_close is not None else bar.open
            return _exit(e, CLOSED, deadline, price, "still open at the New York 17:00 close")
        stopped = bar.low <= e.stop if long else bar.high >= e.stop
        hit = bar.high >= e.target if long else bar.low <= e.target
        if stopped:
            note = "stop and target in the same minute; counted as the stop" if hit else "stop hit"
            _exit(e, LOST, t, e.stop, note)
            if any(c.get("action") == "move_stop" for c in e.changes):   # the manager had tightened it
                profit = (e.result_r or 0) > 0
                e.status = WON if profit else CLOSED
                e.note = "moved stop hit in profit" if profit else "moved stop hit"
            return e
        if hit:
            return _exit(e, WON, t, e.target, "target hit")
        last_close = bar.close
        e.settled_to = t + timedelta(seconds=1)
        k += 1
    if now >= deadline and last_close is not None:
        return _exit(e, CLOSED, deadline, last_close, "still open at the New York 17:00 close")
    return e


def _exit(e: Entry, status: str, t: datetime, price: float, note: str) -> Entry:
    e.status, e.exit_at, e.exit_price, e.note = status, t, price, note
    e.result_r = net_r(e, price)
    return e


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    total: int = 0
    pending: int = 0
    open: int = 0
    won: int = 0
    lost: int = 0
    closed: int = 0
    expired: int = 0
    missed: int = 0
    cancelled: int = 0
    total_r: float = 0.0
    avg_r: float | None = None            # expectancy per filled trade
    win_rate: float | None = None         # share of filled trades with R > 0
    fill_rate: float | None = None        # filled ÷ (filled + expired + missed)
    profit_factor: float | None = None
    max_drawdown_r: float = 0.0
    best_r: float | None = None
    worst_r: float | None = None
    curve: list[tuple[datetime, float]] = field(default_factory=list)   # cumulative R
    by_symbol: dict[str, dict] = field(default_factory=dict)
    by_conviction: dict[str, dict] = field(default_factory=dict)

    @property
    def finished(self) -> int:
        return self.won + self.lost + self.closed


def stats(entries: Iterable[Entry]) -> Stats:
    s = Stats()
    finished = []
    for e in entries:
        s.total += 1
        setattr(s, e.status, getattr(s, e.status) + 1)
        if e.status in FINISHED and e.result_r is not None:
            finished.append(e)
    finished.sort(key=lambda e: e.exit_at or e.created_at)
    if finished:
        rs = [e.result_r for e in finished]
        s.total_r = round(sum(rs), 2)
        s.avg_r = round(s.total_r / len(rs), 2)
        s.win_rate = sum(r > 0 for r in rs) / len(rs)
        gains, losses = sum(r for r in rs if r > 0), -sum(r for r in rs if r < 0)
        s.profit_factor = round(gains / losses, 2) if losses else None
        s.best_r, s.worst_r = max(rs), min(rs)
        running = peak = 0.0
        for e in finished:
            running = round(running + e.result_r, 2)
            peak = max(peak, running)
            s.max_drawdown_r = round(max(s.max_drawdown_r, peak - running), 2)
            s.curve.append((e.exit_at or e.created_at, running))
    filled = s.finished + s.open
    if filled + s.expired + s.missed:
        s.fill_rate = filled / (filled + s.expired + s.missed)
    for key, attr in (("by_symbol", "symbol"), ("by_conviction", "conviction")):
        groups: dict[str, list[float]] = defaultdict(list)
        for e in finished:
            groups[getattr(e, attr)].append(e.result_r)
        setattr(s, key, {
            name: {"n": len(rs), "total_r": round(sum(rs), 2),
                   "win_rate": sum(r > 0 for r in rs) / len(rs)}
            for name, rs in sorted(groups.items(), key=lambda kv: -sum(kv[1]))
        })
    return s
