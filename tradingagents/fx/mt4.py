"""Send the desk's orders to a MetaTrader 4 account (CMC Markets or any MT4 broker).

MT4 has no API a Mac can call, so the two sides talk through files in MT4's
``MQL4/Files`` folder. The Expert Advisor ``TradingAgentsBridge.mq4`` runs on
one chart; it reads instructions, acts, and answers:

- the watcher writes ``ta_cmd_<id>.txt`` (place, modify, cancel, close, flatten);
- the EA answers in ``ta_res_<id>.txt`` and keeps ``ta_state.txt`` current
  (account, this desk's orders, recent history, prices) and ``ta_symbols.txt``.

The journal stays the desk's record, priced on OANDA mid. :meth:`Bridge.sync`
makes the MT4 book follow it, one step per order:

- a new pending order in the journal is placed as a limit order, ``lots`` each,
  with its levels shifted half of CMC's spread (see :func:`broker_levels`) so
  it fills and exits when the mid price reaches them, as the journal assumes;
- a stop or target the trade manager moved is moved in MT4;
- an order the journal no longer waits on (cancelled, expired, missed, or a
  paper trade that already finished) is deleted if it has not filled at CMC;
- a trade the manager closed early is closed at CMC.

A trade that filled at CMC otherwise runs on its own stop and target: CMC's
prices, not the journal's, decide it. The EA cancels unfilled orders at their
cancel time and closes trades at 16:55 New York itself, so both happen even if
the watcher stops.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tradingagents.fx.instruments import spec_for
from tradingagents.fx.journal import ACTIVE, CANCELLED, CLOSED, PENDING, Entry, Journal, flat_by

EA_FILE = Path(__file__).with_name("TradingAgentsBridge.mq4")
STALE_AFTER = timedelta(seconds=90)         # no state file this fresh: the EA is not running
ANSWER_WITHIN = timedelta(minutes=2)        # the EA ignores an instruction older than this
MIN_LIFE = timedelta(minutes=5)             # don't place an order about to be cancelled
PRICE_SANITY = 0.03                          # entry this far from CMC's price: wrong symbol or scale
MAX_RETRIES = 3                              # re-sends of an order that timed out, never one MT4 refused
RETRYABLE = ("stale instruction", "MT4 did not answer")
ALIASES = {"XAUUSD": ("GOLD",), "XAGUSD": ("SILVER",)}

# Broker-side states of one order.
SENDING, B_PENDING, B_OPEN, B_CLOSED, B_CANCELLED, FAILED = (
    "sending", "pending", "open", "closed", "cancelled", "failed")
LIVE = (SENDING, B_PENDING, B_OPEN)

Notify = Callable[[str], None]

TABLE = """
CREATE TABLE IF NOT EXISTS mt4_orders (
    ticket TEXT PRIMARY KEY,
    journal_id TEXT,
    symbol TEXT,
    broker_symbol TEXT,
    side TEXT,
    lots REAL,
    state TEXT,
    mt4 INTEGER,
    sent_sl REAL,
    sent_tp REAL,
    inflight TEXT,
    inflight_action TEXT,
    inflight_at TEXT,
    want_sl REAL,
    want_tp REAL,
    failed_action TEXT,
    fill_price REAL,
    close_price REAL,
    profit REAL,
    note TEXT,
    updated_at TEXT,
    half_spread REAL
)
"""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def config_path() -> Path:
    override = os.environ.get("TRADINGAGENTS_FX_MT4")
    return Path(override) if override else Path(os.path.expanduser("~")) / ".tradingagents" / "fx_mt4.json"


@dataclass
class Mt4Config:
    files_dir: str = ""
    lots: float = 0.01
    symbols: dict[str, str] = field(default_factory=dict)    # e.g. {"XAUUSD": "GOLD"}

    @classmethod
    def load(cls, path: Path | None = None) -> Mt4Config | None:
        path = path or config_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(files_dir=data.get("files_dir", ""), lots=float(data.get("lots", 0.01)),
                   symbols=dict(data.get("symbols", {})))

    def save(self, path: Path | None = None) -> Path:
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"files_dir": self.files_dir, "lots": self.lots,
                                    "symbols": self.symbols}, indent=2) + "\n", encoding="utf-8")
        return path


def find_files_folders(roots: list[Path] | None = None, max_depth: int = 14) -> list[Path]:
    """Every ``MQL4/Files`` folder under ``roots``, most recently used first.

    On Windows it is ``%APPDATA%\\MetaQuotes\\Terminal\\<id>\\MQL4\\Files`` (or under
    Program Files for a portable install). On a Mac, MT4 runs inside Wine, so
    it sits deep in ``~/Library/Application Support/<app>/drive_c/...``.
    """
    home = Path(os.path.expanduser("~"))
    if roots is None:
        roots = [home / "Library" / "Application Support", home / ".wine", home / "Library" / "Containers"]
        for var in ("APPDATA", "ProgramFiles", "ProgramFiles(x86)"):
            if os.environ.get(var):
                roots.append(Path(os.environ[var]))
    skip = {"Caches", "Google", "Code", "Slack", "CrashReporter", "MobileSync", "node_modules", "Logs"}
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        base = len(root.parts)
        for here, dirs, _ in os.walk(root):
            depth = len(Path(here).parts) - base
            if Path(here).name == "MQL4":
                if "Files" in dirs:
                    found.append(Path(here) / "Files")
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in skip and depth < max_depth]

    def used(p: Path) -> float:
        times = [p.stat().st_mtime]
        times += [(p.parent.parent / name).stat().st_mtime for name in ("logs", "MQL4")
                  if (p.parent.parent / name).exists()]
        return max(times)

    return sorted(set(found), key=used, reverse=True)


def install_ea(files_dir: Path) -> Path:
    """Copy the EA into ``MQL4/Experts`` next to ``files_dir``; returns where it went."""
    experts = files_dir.parent / "Experts"
    experts.mkdir(exist_ok=True)
    target = experts / EA_FILE.name
    target.write_bytes(EA_FILE.read_bytes())
    return target


# ---------------------------------------------------------------------------
# What the EA reports
# ---------------------------------------------------------------------------

@dataclass
class BrokerOrder:
    mt4: int
    comment: str
    symbol: str
    type: int                      # 0 buy, 1 sell, 2 buy limit, 3 sell limit, 4/5 stops
    lots: float
    open_price: float
    sl: float
    tp: float
    profit: float
    close_price: float | None = None
    close_time: datetime | None = None

    @property
    def pending(self) -> bool:
        return self.type >= 2


@dataclass
class State:
    time: datetime
    account: str = ""
    server: str = ""
    company: str = ""
    currency: str = ""
    balance: float = 0.0
    equity: float = 0.0
    demo: bool = False
    connected: bool = False
    trade_allowed: bool = False
    max_lots: float = 0.0
    orders: dict[int, BrokerOrder] = field(default_factory=dict)
    history: dict[int, BrokerOrder] = field(default_factory=dict)
    quotes: dict[str, tuple[float, float]] = field(default_factory=dict)


def _order(value: str) -> BrokerOrder:
    p = value.split("|")
    o = BrokerOrder(mt4=int(p[0]), comment=p[1], symbol=p[2], type=int(p[3]), lots=float(p[4]),
                    open_price=float(p[5]), sl=float(p[6]), tp=float(p[7]), profit=float(p[8]))
    if len(p) >= 11:
        o.close_price = float(p[9])
        o.close_time = datetime.fromtimestamp(int(p[10]), UTC)
    return o


def parse_state(text: str) -> State:
    state = State(time=datetime.fromtimestamp(0, UTC))
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key == "time":
            state.time = datetime.fromtimestamp(int(value), UTC)
        elif key in ("account", "server", "company", "currency"):
            setattr(state, key, value)
        elif key in ("balance", "equity", "max_lots"):
            setattr(state, key, float(value))
        elif key in ("demo", "connected", "trade_allowed"):
            setattr(state, key, value == "1")
        elif key == "order":
            o = _order(value)
            state.orders[o.mt4] = o
        elif key == "hist":
            o = _order(value)
            state.history[o.mt4] = o
        elif key == "quote":
            sym, bid, ask = value.split("|")
            state.quotes[sym] = (float(bid), float(ask))
    return state


def _kv(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def map_symbol(symbol: str, broker_symbols: list[str], overrides: dict[str, str] | None = None) -> str | None:
    """The broker's name for ``symbol``: ``EURUSD`` may be ``EURUSD``, ``EURUSD.`` or ``EURUSDm``."""
    if overrides and symbol in overrides:
        return overrides[symbol]
    if symbol in broker_symbols:
        return symbol
    letters = {b: re.sub(r"[^A-Z]", "", b.upper()) for b in broker_symbols}
    for name in (symbol, *ALIASES.get(symbol, ())):
        matches = [b for b, s in letters.items() if s == name] or \
                  [b for b, s in letters.items() if s.startswith(name)]
        if matches:
            return min(matches, key=len)
    return None


def symbol_choices(symbol: str, broker_symbols: list[str]) -> list[str]:
    """Every broker symbol that could be ``symbol`` (e.g. ``USDJPY`` and ``USDJPY.r``)."""
    names = (symbol, *ALIASES.get(symbol, ()))
    return [b for b in broker_symbols if any(re.sub(r"[^A-Z]", "", b.upper()).startswith(n) for n in names)]


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------

@dataclass
class Row:
    ticket: str
    journal_id: str
    symbol: str
    broker_symbol: str
    side: str
    lots: float
    state: str
    mt4: int | None = None
    sent_sl: float | None = None
    sent_tp: float | None = None
    inflight: str | None = None
    inflight_action: str | None = None
    inflight_at: datetime | None = None
    want_sl: float | None = None
    want_tp: float | None = None
    failed_action: str | None = None
    fill_price: float | None = None
    close_price: float | None = None
    profit: float | None = None
    note: str = ""
    updated_at: datetime | None = None
    half_spread: float | None = None       # levels shifted by this so CMC acts on the mid price


_ROW_FIELDS = list(Row.__dataclass_fields__)


class Bridge:
    """The MT4 side of the desk: one instance per watcher."""

    def __init__(self, files_dir: str | Path, journal: Journal, *, lots: float = 0.01,
                 symbols: dict[str, str] | None = None):
        self.dir = Path(files_dir)
        self.journal = journal
        self.lots = lots
        self.overrides = symbols or {}
        self._warned: set[str] = set()
        self._retries: dict[str, int] = {}
        with closing(self._db()) as db, db:
            db.execute(TABLE)
            columns = {r[1] for r in db.execute("PRAGMA table_info(mt4_orders)")}
            if "half_spread" not in columns:              # tables from before spread adjustment
                db.execute("ALTER TABLE mt4_orders ADD COLUMN half_spread REAL")

    # -- storage --------------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.journal.path)
        db.row_factory = sqlite3.Row
        return db

    def rows(self) -> dict[str, Row]:
        with closing(self._db()) as db:
            out = {}
            for r in db.execute("SELECT * FROM mt4_orders"):
                d = dict(r)
                for k in ("inflight_at", "updated_at"):
                    d[k] = datetime.fromisoformat(d[k]) if d[k] else None
                d["note"] = d["note"] or ""
                out[d["ticket"]] = Row(**{k: d[k] for k in _ROW_FIELDS})
            return out

    def _save(self, row: Row, now: datetime) -> None:
        row.updated_at = now
        values = [getattr(row, k) for k in _ROW_FIELDS]
        values = [v.isoformat() if isinstance(v, datetime) else v for v in values]
        with closing(self._db()) as db, db:
            db.execute(f"INSERT OR REPLACE INTO mt4_orders ({','.join(_ROW_FIELDS)}) "
                       f"VALUES ({','.join('?' * len(_ROW_FIELDS))})", values)

    # -- files ----------------------------------------------------------------

    def state(self) -> State | None:
        path = self.dir / "ta_state.txt"
        try:
            return parse_state(path.read_text(encoding="latin-1"))
        except (OSError, ValueError, IndexError):
            return None

    def broker_symbols(self) -> list[str]:
        try:
            text = (self.dir / "ta_symbols.txt").read_text(encoding="latin-1")
        except OSError:
            return []
        return [s.strip() for s in text.splitlines() if s.strip()]

    def send(self, action: str, now: datetime, **fields) -> str:
        """Write one instruction for the EA; returns its id."""
        number = int(self.journal.get_meta("mt4_cmd", "0")) + 1
        self.journal.set_meta("mt4_cmd", str(number))
        cid = f"{number:08d}"
        lines = {"id": cid, "action": action,
                 "valid_until": int((now + ANSWER_WITHIN).timestamp()), **fields}
        text = "".join(f"{k}={v}\r\n" for k, v in lines.items())
        tmp = self.dir / f"ta_tmp_{cid}.part"
        tmp.write_text(text, encoding="latin-1")
        os.replace(tmp, self.dir / f"ta_cmd_{cid}.txt")
        return cid

    def answers(self) -> dict[str, dict[str, str]]:
        """Every answer the EA left, by instruction id (read and removed)."""
        out = {}
        for path in sorted(self.dir.glob("ta_res_*.txt")):
            try:
                data = _kv(path.read_text(encoding="latin-1"))
            except OSError:
                continue
            if "id" in data:
                out[data["id"]] = data
            path.unlink(missing_ok=True)
        return out

    def healthy(self, now: datetime, state: State | None = None) -> tuple[bool, str]:
        state = state if state is not None else self.state()
        if state is None:
            return False, "MT4 bridge not running: no ta_state.txt (is the EA on a chart?)"
        if now - state.time > STALE_AFTER:
            age = int((now - state.time).total_seconds() // 60)
            return False, f"MT4 bridge silent for {age} min (MT4 closed, or the EA removed?)"
        if not state.connected:
            return False, "MT4 is not connected to the broker"
        if not state.trade_allowed:
            return False, "AutoTrading is off in MT4 (or the EA may not trade)"
        return True, ""

    def flatten(self, now: datetime) -> str:
        return self.send("flatten", now)

    def ping(self, now: datetime) -> str:
        return self.send("ping", now)

    def wait(self, cid: str, seconds: float = 10.0, sleep=None) -> dict[str, str] | None:
        """The EA's answer to instruction ``cid``, waiting up to ``seconds``."""
        import time
        sleep = sleep or time.sleep
        for _ in range(int(seconds * 2)):
            answer = self.answers().get(cid)
            if answer:
                return answer
            sleep(0.5)
        return None

    def test_order(self, broker_symbol: str, now: datetime, sleep=None) -> list[str]:
        """Place a buy limit 1% under the price at ``lots``, then cancel it: proves the whole route."""
        state = self.state()
        quote = state.quotes.get(broker_symbol) if state else None
        if quote is None:
            return [f"No price for {broker_symbol}: add it to MT4's Market Watch first (Ctrl+M → right-click → Symbols)."]
        price = quote[0] * 0.99
        lines = [f"Placing a test BUY LIMIT {broker_symbol} {self.lots:g} lot at {price:.5g} "
                 f"(1% under the price {quote[0]:g}), then cancelling it…"]
        cid = self.send("place", now, ticket="TEST", symbol=broker_symbol, side="buy", lots=f"{self.lots:.2f}",
                        price=_price(price), sl=_price(price * 0.995), tp=_price(price * 1.01),
                        comment=f"TA-TEST {now:%H%M%S}", expires=int((now + timedelta(minutes=30)).timestamp()),
                        close_by=int((now + timedelta(minutes=30)).timestamp()))
        placed = self.wait(cid, sleep=sleep)
        if placed is None:
            return [*lines, "No answer from MT4 within 10 s: is the EA on a chart with a smiley face?"]
        if placed.get("ok") != "1":
            return [*lines, f"REFUSED by MT4: {placed.get('message', '')} (error {placed.get('error', '?')})"]
        mt4_ticket = placed.get("mt4", "")
        lines.append(f"Placed: MT4 order #{mt4_ticket}")
        done = self.wait(self.send("cancel", datetime.now(UTC), ticket="TEST", mt4=mt4_ticket), sleep=sleep)
        if done is None or done.get("ok") != "1":
            why = done.get("message", "") if done else "no answer"
            return [*lines, f"Could NOT cancel #{mt4_ticket} ({why}): delete it by hand in MT4's Trade tab."]
        return [*lines, f"Cancelled #{mt4_ticket}. {broker_symbol} works end to end."]

    # -- the sync -------------------------------------------------------------

    def _warn(self, key: str, text: str, notify: Notify) -> None:
        if key not in self._warned:
            self._warned.add(key)
            notify(text)

    def sync(self, now: datetime, notify: Notify) -> list[str]:
        """Bring MT4 in line with the journal; returns notes for the console."""
        notes: list[str] = []
        rows = self.rows()
        by_id = {r.inflight: r for r in rows.values() if r.inflight}
        for cid, ans in self.answers().items():
            row = by_id.get(cid)
            if row is not None:
                self._answered(row, ans, now, notify)

        for row in rows.values():
            if row.inflight and row.inflight_at and now - row.inflight_at > ANSWER_WITHIN + timedelta(minutes=1):
                self._no_answer(row, now, notify)

        state = self.state()
        ok, why = self.healthy(now, state)
        if not ok:
            notes.append(why)
            self._warn("down", f"⚠️ <b>CMC MT4</b>\n{why}. Orders wait until it is back.", notify)
            return notes
        if "down" in self._warned:
            self._warned.discard("down")
            notify("✅ <b>CMC MT4</b>\nBridge is back.")

        for row in rows.values():
            self._follow_broker(row, state, now, notify)
        self._follow_journal(rows, state, now, notify, notes)
        return notes

    def _answered(self, row: Row, ans: dict[str, str], now: datetime, notify: Notify) -> None:
        ok, action = ans.get("ok") == "1", row.inflight_action
        message = ans.get("message", "")
        row.inflight = row.inflight_action = row.inflight_at = None
        if not ok and message.startswith(RETRYABLE) and action != "place":
            row.note = message                 # arrived late: the next sync sends it again
            self._save(row, now)
            return
        if action == "place":
            if ok:
                row.state, row.mt4 = B_PENDING, int(ans.get("mt4") or 0) or None
                row.sent_sl, row.sent_tp = row.want_sl, row.want_tp
                at = float(ans.get("price") or 0)
                levels = (f" at {at:.7g} · SL {row.sent_sl:.7g} · TP {row.sent_tp:.7g}"
                          if at and row.sent_sl is not None and row.sent_tp is not None else "")
                spread = (f" (levels shifted {row.half_spread / spec_for(row.symbol).pip:.1f} pips for CMC's spread)"
                          if row.half_spread else "")
                notify(f"📤 <b>CMC</b> {row.ticket} {row.side.upper()} LIMIT {row.broker_symbol} "
                       f"{row.lots:g} lot placed{levels}{spread} (MT4 #{row.mt4})")
            else:
                row.state, row.note = FAILED, message
                if message.startswith(RETRYABLE):
                    notify(f"⏳ <b>CMC</b> {row.ticket} {row.symbol}: the instruction reached MT4 late; re-sending")
                else:
                    notify(f"⚠️ <b>CMC refused</b> {row.ticket} {row.symbol}: {message}")
        elif action == "modify":
            row.sent_sl, row.sent_tp = row.want_sl, row.want_tp   # tried: don't retry the same values
            if ok:
                notify(f"✏️ <b>CMC</b> {row.ticket} {row.broker_symbol}: stop {row.want_sl:g}, target {row.want_tp:g}")
            else:
                row.note = message
                notify(f"⚠️ <b>CMC</b> could not move {row.ticket} {row.symbol}'s stop/target: {message}")
        elif action in ("cancel", "close") and not ok:
            row.failed_action, row.note = action, message
            notify(f"⚠️ <b>CMC</b> could not {action} {row.ticket} {row.symbol}: {message}. "
                   "Please check it in MT4.")
        self._save(row, now)

    def _no_answer(self, row: Row, now: datetime, notify: Notify) -> None:
        action = row.inflight_action
        row.inflight = row.inflight_action = row.inflight_at = None
        if action == "place":
            row.state, row.note = FAILED, "MT4 did not answer in time"
            notify(f"⚠️ <b>CMC</b> {row.ticket} {row.symbol}: MT4 did not answer in time.")
        elif action == "modify":
            row.sent_sl, row.sent_tp = row.want_sl, row.want_tp
        else:
            row.failed_action = action
        self._save(row, now)

    def _follow_broker(self, row: Row, state: State, now: datetime, notify: Notify) -> None:
        if row.state not in (B_PENDING, B_OPEN) or not row.mt4:
            return
        live = state.orders.get(row.mt4)
        if live is not None:
            if row.state == B_PENDING and not live.pending:
                row.state, row.fill_price = B_OPEN, live.open_price
                notify(f"✅ <b>CMC filled</b> {row.ticket} {row.broker_symbol} at {live.open_price:g}")
                self._save(row, now)
            return
        done = state.history.get(row.mt4)
        if done is None:
            row.state, row.note = B_CLOSED, "no longer in MT4; see its Account History"
            notify(f"ℹ️ <b>CMC</b> {row.ticket} {row.symbol} is no longer in MT4 "
                   "(set Account History to 'Last 3 days' so the bridge can read results).")
        elif done.pending:
            row.state = B_CANCELLED
            notify(f"🗑 <b>CMC</b> {row.ticket} {row.broker_symbol} cancelled, unfilled")
        else:
            row.state, row.close_price, row.profit = B_CLOSED, done.close_price, done.profit
            row.fill_price = row.fill_price or done.open_price
            sign = "+" if done.profit >= 0 else "−"
            notify(f"🏁 <b>CMC closed</b> {row.ticket} {row.broker_symbol} at {done.close_price:g} · "
                   f"P/L {sign}{abs(done.profit):.2f} {state.currency}")
        self._save(row, now)

    def _follow_journal(self, rows: dict[str, Row], state: State, now: datetime, notify: Notify,
                        notes: list[str]) -> None:
        for e in self.journal.entries():
            if not e.ticket:
                continue
            row = rows.get(e.ticket)
            if row is None or (row.state == FAILED and row.note.startswith(RETRYABLE)):
                if e.status == PENDING and e.expires_at - now >= MIN_LIFE:
                    if row is not None:                # timed out, not refused: try again
                        tries = self._retries.get(e.ticket, 0)
                        if tries >= MAX_RETRIES:
                            self._warn(f"gave-up {e.ticket}", f"⚠️ <b>CMC</b> {e.ticket} {e.symbol} not placed: "
                                       f"gave up after {MAX_RETRIES} retries ({row.note}).", notify)
                            continue
                        self._retries[e.ticket] = tries + 1
                        notes.append(f"re-sending {e.ticket} {e.symbol} to MT4 ({row.note})")
                    self._place(e, state, now, notify, notes)
                continue
            if row.inflight:
                continue
            finished = e.status not in ACTIVE
            if row.state == B_PENDING:
                if finished and row.failed_action != "cancel":
                    self._act(row, "cancel", now)
                elif not finished and self._moved(row, e):
                    self._modify(row, e, now)
            elif row.state == B_OPEN:
                if e.status in (CLOSED, CANCELLED) and _by_manager(e) and row.failed_action != "close":
                    self._act(row, "close", now)
                elif not finished and self._moved(row, e):
                    self._modify(row, e, now)

    @staticmethod
    def _moved(row: Row, e: Entry) -> bool:
        tick = spec_for(e.symbol).pip / 100
        _, sl, tp = broker_levels(e.long, e.entry, e.stop, e.target, row.half_spread or 0.0)
        return (row.sent_sl is None or abs(sl - row.sent_sl) > tick
                or row.sent_tp is None or abs(tp - row.sent_tp) > tick)

    def _act(self, row: Row, action: str, now: datetime) -> None:
        row.inflight = self.send(action, now, ticket=row.ticket, mt4=row.mt4)
        row.inflight_action, row.inflight_at = action, now
        self._save(row, now)

    def _modify(self, row: Row, e: Entry, now: datetime) -> None:
        _, sl, tp = broker_levels(e.long, e.entry, e.stop, e.target, row.half_spread or 0.0)
        row.want_sl, row.want_tp = sl, tp
        row.inflight = self.send("modify", now, ticket=row.ticket, mt4=row.mt4,
                                 sl=_price(sl), tp=_price(tp))
        row.inflight_action, row.inflight_at = "modify", now
        self._save(row, now)

    def _place(self, e: Entry, state: State, now: datetime, notify: Notify, notes: list[str]) -> None:
        side = "buy" if e.long else "sell"
        row = Row(ticket=e.ticket, journal_id=e.id, symbol=e.symbol, broker_symbol="", side=side,
                  lots=self.lots, state=FAILED)
        broker = map_symbol(e.symbol, self.broker_symbols() or list(state.quotes), self.overrides)
        if broker is None:
            row.note = f"no MT4 symbol matches {e.symbol}"
            self._save(row, now)
            notify(f"⚠️ <b>CMC</b> {e.ticket} {e.symbol} not placed: no matching MT4 symbol. "
                   f"Map it with `tradingagents fx-mt4 --symbol {e.symbol}=<MT4 name>`.")
            return
        row.broker_symbol = broker
        quote = state.quotes.get(broker)
        if quote:
            mid = (quote[0] + quote[1]) / 2
            if mid <= 0 or abs(e.entry / mid - 1) > PRICE_SANITY:
                row.note = f"entry {e.entry:g} is far from CMC's {broker} price {mid:g}"
                self._save(row, now)
                notify(f"⚠️ <b>CMC</b> {e.ticket} {e.symbol} not placed: {row.note} (wrong symbol?).")
                return
        row.half_spread = half_spread(quote, e.spread)
        entry, sl, tp = broker_levels(e.long, e.entry, e.stop, e.target, row.half_spread)
        row.state, row.want_sl, row.want_tp = SENDING, sl, tp
        row.inflight = self.send(
            "place", now, ticket=e.ticket, symbol=broker, side=side, lots=f"{self.lots:.2f}",
            price=_price(entry), sl=_price(sl), tp=_price(tp),
            comment=f"TA{e.ticket}", expires=int(e.expires_at.timestamp()),
            close_by=int(flat_by(e.created_at).timestamp()))
        row.inflight_action, row.inflight_at = "place", now
        self._save(row, now)
        notes.append(f"sent {e.ticket} {e.symbol} to MT4")


def half_spread(quote: tuple[float, float] | None, planned_spread: float) -> float:
    """Half of CMC's spread now, capped at three times the spread the desk planned with.

    The cap keeps a momentary blow-out (rollover, a news spike) from shifting
    the levels of an order that will wait for hours.
    """
    if quote is None:
        return max(planned_spread, 0.0) / 2
    spread = max(quote[1] - quote[0], 0.0)
    if planned_spread > 0:
        spread = min(spread, 3 * planned_spread)
    return spread / 2


def broker_levels(long: bool, entry: float, stop: float, target: float, half: float) -> tuple[float, float, float]:
    """The desk's mid-price levels as MT4 levels that trigger when the mid price gets there.

    MT4 fills a buy limit and exits a sell on the ask, and fills a sell limit
    and exits a buy on the bid; the ask is half a spread above the mid and the
    bid half a spread below. So a buy's entry goes up half a spread and its
    stop and target down; a sell's the other way. Risk grows and reward
    shrinks by one spread, the same net figures the journal scores.
    """
    sign = 1 if long else -1
    return entry + sign * half, stop - sign * half, target - sign * half


def _price(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _by_manager(e: Entry) -> bool:
    """Closed or cancelled by the trade manager (not by the 16:55 close or the paper prices)."""
    return bool(e.changes) and e.changes[-1].get("action") in ("close", "cancel")
