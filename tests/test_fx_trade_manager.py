"""Tickets, modifiable trades, and Ward the trade manager."""

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_fx_agents import FakeLLM, _ctx, _setup, _team  # noqa: E402

from tradingagents.fx import (  # noqa: E402
    agents as fx_agents,  # noqa: E402
    dashboard,
    journal as jr,  # noqa: E402
    telegram,
)
from tradingagents.fx.journal import Journal, simulate, stats  # noqa: E402
from tradingagents.fx.manage import apply, describe, snapshot  # noqa: E402
from tradingagents.fx.schemas import ManagementPlan, TradeAction  # noqa: E402
from tradingagents.fx.watch import file_review  # noqa: E402

T0 = datetime(2026, 10, 6, 11, 0, tzinfo=UTC)


def _order(symbol="EURUSD", direction="long", entry=1.1700, stop=1.1680, target=1.1760):
    setup = SimpleNamespace(spread_pips=0.0, strategy="smc", score=70.0)
    return SimpleNamespace(symbol=symbol, direction=direction, entry=entry, stop=stop, target=target, rr=3.0,
                           expires_at=T0 + timedelta(hours=4), conviction="medium", rationale="r",
                           watch_for="w", setup=setup)


def _bars(*ohlc, start=T0):
    idx = pd.date_range(start, periods=len(ohlc), freq="1min", tz="UTC")
    return pd.DataFrame(ohlc, columns=["open", "high", "low", "close"], index=idx)


def _quote(mid, spread=0.0001):
    return lambda symbol: SimpleNamespace(bid=mid - spread / 2, ask=mid + spread / 2)


def _open_trade(tmp_path, **kw):
    book = Journal(tmp_path / "j.db")
    book.record([_order(**kw)], now=T0)
    fill = (1.1705, 1.1706, 1.1699, 1.1702)
    book.settle(lambda s, g, n: _bars(fill), T0 + timedelta(minutes=1))
    return book, book.entries()[0]


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tickets_are_sequential_and_never_reused(tmp_path):
    book = Journal(tmp_path / "j.db")
    assert book.next_tickets(["EURUSD", "GBPUSD"]) == {"EURUSD": "#1001", "GBPUSD": "#1002"}
    given = book.record([_order()], now=T0, tickets={"EURUSD": "#1001"})
    unassigned = book.record([_order("USDJPY", "short", 150.4, 150.6, 149.8)], now=T0)
    assert given[0].ticket == "#1001" and unassigned[0].ticket == "#1003"
    assert Journal(tmp_path / "j.db").next_tickets(["X"]) == {"X": "#1004"}


@pytest.mark.unit
def test_an_older_journal_gets_tickets_and_its_original_stops(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE orders (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, symbol TEXT NOT NULL, "
               "direction TEXT NOT NULL, entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL, "
               "planned_rr REAL, spread REAL DEFAULT 0, expires_at TEXT NOT NULL, conviction TEXT, rationale TEXT, "
               "watch_for TEXT, strategy TEXT, scanner_score REAL, report TEXT, status TEXT NOT NULL DEFAULT "
               "'pending', filled_at TEXT, exit_at TEXT, exit_price REAL, result_r REAL, note TEXT, notified TEXT)")
    for i, sym in enumerate(("EURUSD", "GBPUSD")):
        db.execute("INSERT INTO orders (id, created_at, symbol, direction, entry, stop, target, expires_at) "
                   "VALUES (?, ?, ?, 'long', 1.17, 1.168, 1.175, ?)",
                   (sym, (T0 + timedelta(minutes=i)).isoformat(), sym, (T0 + timedelta(hours=4)).isoformat()))
    db.commit()
    db.close()
    entries = Journal(path).entries()
    assert [(e.symbol, e.ticket, e.initial_stop) for e in entries] == [
        ("EURUSD", "#1001", 1.168), ("GBPUSD", "#1002", 1.168)]


@pytest.mark.unit
def test_tickets_run_through_the_review_and_the_messages():
    candidates = [_setup("EURUSD"), _setup("USDJPY", "short")]
    quick, deep = _team(candidates)
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=T0,
                              tickets={"EURUSD": "#1043", "USDJPY": "#1044"})
    assert result.tickets == {"EURUSD": "#1043", "USDJPY": "#1044"}
    assert "#1043 EURUSD" in result.messages[0]["text"] and "#1044 USDJPY" in result.messages[-1]["text"]
    assert any("### #1043 EURUSD" in p for p in quick.prompts)
    assert all('"#1043 AUDUSD"' in p for p in quick.prompts)        # the desk rule


# ---------------------------------------------------------------------------
# Modifiable trades
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_a_moved_stop_applies_from_the_move_on_and_r_uses_the_original_stop(tmp_path):
    book, e = _open_trade(tmp_path)
    assert e.status == jr.OPEN
    later = T0 + timedelta(minutes=10)
    book.settle(lambda s, g, n: _bars((1.1705, 1.1720, 1.1704, 1.1718), start=T0 + timedelta(minutes=1)),
                later)
    e = book.entries()[0]
    book.modify(e, now=later, by="Ward", reason="lock in", stop=1.1710)

    # A dip to 1.1708 before the move must not count; after it, it hits the new stop.
    bars = _bars((1.1705, 1.1720, 1.1704, 1.1718), (1.1712, 1.1713, 1.1708, 1.1709),
                 start=later - timedelta(minutes=1))
    book.settle(lambda s, g, n: bars, later + timedelta(minutes=1))
    e = book.entries()[0]
    assert e.status == jr.WON and e.note == "moved stop hit in profit"
    assert e.result_r == pytest.approx(0.5)              # +10 pips on the original 20-pip risk
    assert e.changes[0]["action"] == "move_stop" and e.changes[0]["old"] == 1.168


@pytest.mark.unit
def test_breakeven_stop_hit_is_closed_not_lost(tmp_path):
    book, e = _open_trade(tmp_path)
    book.modify(e, now=T0 + timedelta(minutes=2), by="Ward", reason="breakeven", stop=1.1700)
    e = book.entries()[0]
    simulate(e, _bars((1.1702, 1.1703, 1.1699, 1.1700), start=T0 + timedelta(minutes=3)), T0 + timedelta(minutes=5))
    assert e.status == jr.CLOSED and e.result_r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Ward's actions, checked in code
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ward_can_tighten_but_never_widen_or_cross_price(tmp_path):
    book, e = _open_trade(tmp_path)
    positions = snapshot(book.entries(jr.ACTIVE), lambda s, g, n: _bars((1.1705, 1.1725, 1.1699, 1.1720)),
                         _quote(1.1720), T0 + timedelta(minutes=30))
    assert positions[0].r_now == pytest.approx(1.0) and positions[0].best_r == pytest.approx(1.25)

    results = apply([
        {"ticket": e.ticket, "action": "move_stop", "new_stop": 1.1670, "reason": "wider"},
        {"ticket": e.ticket, "action": "move_stop", "new_stop": 1.1721, "reason": "past price"},
        {"ticket": e.ticket, "action": "cancel", "reason": "x"},
        {"ticket": e.ticket.lstrip("#"), "action": "move_stop", "new_stop": 1.1700, "reason": "breakeven"},
        {"ticket": e.ticket, "action": "move_target", "new_target": 1.1715, "reason": "behind price"},
        {"ticket": e.ticket, "action": "move_target", "new_target": 1.1740, "reason": "closer"},
        {"ticket": "#9999", "action": "close", "reason": "not ours"},
    ], positions, book, now=T0 + timedelta(minutes=30))

    assert [r.ok for r in results] == [False, False, False, True, False, True]
    assert "only tightened" in results[0].text and "beyond the current price" in results[1].text
    assert "close it instead" in results[2].text
    e = book.entries()[0]
    assert (e.stop, e.target, e.initial_stop) == (1.17, 1.174, 1.168)


@pytest.mark.unit
def test_ward_closes_open_trades_and_cancels_pending_ones(tmp_path):
    book, e = _open_trade(tmp_path)
    pending = book.record([_order("GBPUSD", entry=1.3400, stop=1.3380, target=1.3460)], now=T0)[0]
    now = T0 + timedelta(minutes=20)
    positions = snapshot(book.entries(jr.ACTIVE), lambda s, g, n: _bars((1.1705, 1.1712, 1.1699, 1.1710)),
                         _quote(1.1710), now)
    assert "PENDING" in describe(positions) and "OPEN" in describe(positions)
    results = apply([{"ticket": e.ticket, "action": "close", "reason": "CPI in 10 minutes"},
                     {"ticket": pending.ticket, "action": "move_target", "new_target": 1.35, "reason": "x"},
                     {"ticket": pending.ticket, "action": "cancel", "reason": "setup failed"}],
                    positions, book, now=now)
    assert [r.ok for r in results] == [True, False, True]
    by = {x.symbol: x for x in book.entries()}
    assert by["EURUSD"].status == jr.CLOSED and by["EURUSD"].result_r == pytest.approx(0.5)
    assert "closed early by Ward" in by["EURUSD"].note
    assert by["GBPUSD"].status == jr.CANCELLED
    assert stats(book.entries()).cancelled == 1
    assert "CLOSED #1001 EURUSD +0.50R" in telegram.result_message(by["EURUSD"])


@pytest.mark.unit
def test_ward_joins_the_review_when_trades_are_live(tmp_path):
    book, e = _open_trade(tmp_path)
    positions = snapshot(book.entries(jr.ACTIVE), lambda s, g, n: _bars((1.1705, 1.1722, 1.1699, 1.1720)),
                         _quote(1.1720), T0 + timedelta(minutes=30))
    candidates = [_setup("USDJPY", "short")]
    quick, deep = _team(candidates)
    deep.structured[ManagementPlan] = ManagementPlan(summary="Lock in.", actions=[
        TradeAction(ticket=e.ticket, action="move_stop", new_stop=1.1700, reason="breakeven at +1R")])

    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=T0 + timedelta(minutes=30),
                              positions=positions, tickets={"USDJPY": "#1050"})

    agents = [m["agent"] for m in result.messages]
    assert agents[:3] == ["desk", "macro", "trade_manager"]
    assert "Live book: 1 open" in result.messages[0]["text"]
    assert result.management[0]["action"] == "move_stop"
    assert len(deep.prompts) == 3                                   # Sage, Ward, Donna
    risk = [p for p in quick.prompts if "RISK ANALYST" in p]
    assert all("#1001 EURUSD BUY LIMIT" in p for p in risk)

    sent = []
    added, applied = file_review(result, book, now=T0 + timedelta(minutes=30), notify=sent.append)
    assert [a.ok for a in applied] == [True] and added[0].ticket == "#1050"
    assert book.entries()[0].stop == 1.17
    assert result.messages[-1]["title"] == "Trade changes" and "stop moved 1.168 → 1.17" in result.messages[-1]["text"]
    assert any("#1001 EURUSD" in m for m in sent)
    assert [m["agent"] for m in book.reviews()[0]["messages"]][-1] == "desk"


@pytest.mark.unit
def test_no_live_trades_means_no_trade_manager():
    candidates = [_setup("EURUSD")]
    quick, deep = _team(candidates)
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=T0)
    assert "trade_manager" not in [m["agent"] for m in result.messages]
    assert len(deep.prompts) == 2


@pytest.mark.unit
def test_dashboard_shows_tickets_and_the_change_history(tmp_path):
    book, e = _open_trade(tmp_path)
    book.modify(e, now=T0 + timedelta(minutes=5), by="Ward", reason="breakeven <now>", stop=1.1700)
    page = dashboard.render(book.entries(), stats(book.entries()), now=T0 + timedelta(minutes=6))
    assert ">#1001</td>" in page and "<th>Ticket</th>" in page
    assert "move stop 1.168 → 1.17" in page and "breakeven &lt;now&gt;" in page
    assert "#1001 BUY LIMIT EURUSD" in telegram.order_message(e)


@pytest.mark.unit
def test_fake_llm_helper_still_answers():
    assert FakeLLM("x").invoke("hi").content.startswith("x says")
