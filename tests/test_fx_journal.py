"""The trade journal: recording orders, settling them on one-minute bars, the stats."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.fx import journal as jr
from tradingagents.fx.journal import Entry, Journal, day_close, simulate, stats

T0 = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)          # 06:00 New York


def _entry(direction="long", **kw):
    base = {"long": {"entry": 1.1700, "stop": 1.1680, "target": 1.1750},
            "short": {"entry": 1.1700, "stop": 1.1720, "target": 1.1650}}[direction]
    base = {"planned_rr": 2.5, "spread": 0.0, "expires_at": T0 + timedelta(hours=4), **base, **kw}
    return Entry(id="x", created_at=T0, symbol="EURUSD", direction=direction, **base)


def _bars(*ohlc, start=T0, minutes=1):
    """Bars from ``start``, one per (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(ohlc), freq=f"{minutes}min", tz="UTC")
    return pd.DataFrame(ohlc, columns=["open", "high", "low", "close"], index=idx)


FLAT = (1.1710, 1.1712, 1.1708, 1.1710)


@pytest.mark.unit
def test_fill_then_target_is_a_win():
    e = simulate(_entry(), _bars(FLAT, (1.1710, 1.1711, 1.1699, 1.1702), FLAT,
                                 (1.1740, 1.1751, 1.1739, 1.1750)), T0 + timedelta(hours=1))
    assert e.status == jr.WON and e.filled_at == T0 + timedelta(minutes=1)
    assert e.result_r == pytest.approx(2.5)
    assert e.exit_price == 1.1750


@pytest.mark.unit
def test_fill_then_stop_is_minus_one_r():
    e = simulate(_entry(), _bars((1.1710, 1.1711, 1.1699, 1.1702), (1.1700, 1.1701, 1.1679, 1.1680)),
                 T0 + timedelta(hours=1))
    assert e.status == jr.LOST and e.result_r == -1.0


@pytest.mark.unit
def test_target_before_fill_is_missed():
    e = simulate(_entry(), _bars(FLAT, (1.1720, 1.1752, 1.1719, 1.1750), (1.1740, 1.1741, 1.1698, 1.1700)),
                 T0 + timedelta(hours=1))
    assert e.status == jr.MISSED and e.filled_at is None
    assert "before the entry filled" in e.note


@pytest.mark.unit
def test_entry_and_target_in_one_minute_assume_no_fill():
    e = simulate(_entry(), _bars((1.1710, 1.1755, 1.1695, 1.1720)), T0 + timedelta(hours=1))
    assert e.status == jr.MISSED and "same minute" in e.note


@pytest.mark.unit
def test_stop_and_target_in_one_minute_count_as_the_stop():
    e = simulate(_entry(), _bars((1.1710, 1.1711, 1.1699, 1.1702), (1.1700, 1.1760, 1.1670, 1.1700)),
                 T0 + timedelta(hours=1))
    assert e.status == jr.LOST and "counted as the stop" in e.note


@pytest.mark.unit
def test_unfilled_order_expires_at_its_cancel_time():
    bars = _bars(*[FLAT] * 300)                               # five hours, never near the entry
    e = simulate(_entry(), bars, T0 + timedelta(hours=5))
    assert e.status == jr.EXPIRED
    still = simulate(_entry(), bars.iloc[:60], T0 + timedelta(hours=1))
    assert still.status == jr.PENDING


@pytest.mark.unit
def test_an_open_trade_closes_at_the_new_york_close():
    fill = (1.1710, 1.1711, 1.1699, 1.1702)
    drift = (1.1720, 1.1722, 1.1718, 1.1720)
    bars = _bars(fill, *[drift] * 2000)                       # past 17:00 New York (21:00 UTC)
    e = simulate(_entry(expires_at=T0 + timedelta(hours=12)), bars, T0 + timedelta(hours=34))
    assert e.status == jr.CLOSED and e.exit_at == datetime(2026, 10, 6, 20, 55, tzinfo=UTC)   # 16:55 New York
    assert e.result_r == pytest.approx(1.0)                   # +20 pips on 20 risk


@pytest.mark.unit
def test_shorts_mirror_longs_and_the_spread_is_charged():
    e = _entry("short", spread=0.0002)
    simulate(e, _bars((1.1690, 1.1701, 1.1689, 1.1698), (1.1660, 1.1662, 1.1649, 1.1650)),
             T0 + timedelta(hours=1))
    assert e.status == jr.WON
    assert e.result_r == pytest.approx((0.0050 - 0.0002) / (0.0020 + 0.0002), abs=0.01)


@pytest.mark.unit
def test_trades_are_flat_five_minutes_before_the_close():
    from tradingagents.fx.journal import flat_by
    assert flat_by(datetime(2026, 10, 6, 14, 0, tzinfo=UTC)) == datetime(2026, 10, 6, 20, 55, tzinfo=UTC)
    assert flat_by(datetime(2026, 11, 3, 14, 0, tzinfo=UTC)) == datetime(2026, 11, 3, 21, 55, tzinfo=UTC)   # winter


@pytest.mark.unit
def test_day_close_follows_new_york_time():
    assert day_close(datetime(2026, 10, 6, 14, 0, tzinfo=UTC)) == datetime(2026, 10, 6, 21, 0, tzinfo=UTC)
    assert day_close(datetime(2026, 11, 3, 14, 0, tzinfo=UTC)) == datetime(2026, 11, 3, 22, 0, tzinfo=UTC)
    assert day_close(datetime(2026, 10, 6, 21, 30, tzinfo=UTC)) == datetime(2026, 10, 7, 21, 0, tzinfo=UTC)


def _order(symbol="EURUSD", direction="long", entry=1.1700, stop=1.1680, target=1.1750):
    setup = SimpleNamespace(spread_pips=1.0, strategy="smc", score=70.0)
    return SimpleNamespace(symbol=symbol, direction=direction, entry=entry, stop=stop, target=target,
                           rr=2.4, expires_at=T0 + timedelta(hours=4), conviction="medium",
                           rationale="r", watch_for="w", setup=setup)


@pytest.mark.unit
def test_journal_records_settles_and_resumes(tmp_path):
    book = Journal(tmp_path / "j.db")
    added = book.record([_order(), _order("GBPUSD", "short", 1.3400, 1.3420, 1.3350)], now=T0, report="r.md")
    assert [e.symbol for e in added] == ["EURUSD", "GBPUSD"]
    assert book.record([_order()], now=T0 + timedelta(minutes=10)) == []   # same idea again

    fill = (1.1710, 1.1711, 1.1699, 1.1702)
    data = {"EURUSD": _bars(FLAT, fill, FLAT), "GBPUSD": _bars((1.3390, 1.3392, 1.3388, 1.3390))}
    candles = lambda symbol, gran, count: data[symbol]  # noqa: E731
    changes = book.settle(candles, T0 + timedelta(minutes=3))
    assert [(c.entry.symbol, c.before, c.after) for c in changes] == [("EURUSD", "pending", "open")]

    # Later bars: EURUSD reaches its target; the reopened trade resumes after its fill.
    data["EURUSD"] = _bars(FLAT, fill, FLAT, (1.1745, 1.1751, 1.1744, 1.1750))
    changes = book.settle(candles, T0 + timedelta(minutes=4))
    won = [c for c in changes if c.after == jr.WON]
    assert won and won[0].entry.result_r > 2
    assert {e.symbol: e.status for e in Journal(tmp_path / "j.db").entries()} == \
        {"EURUSD": "won", "GBPUSD": "pending"}


def _done(r, symbol="EURUSD", conviction="low", minutes=0):
    e = _entry(conviction=conviction)
    e.symbol = symbol
    e.status = jr.WON if r > 0 else jr.LOST
    e.result_r, e.exit_at = r, T0 + timedelta(minutes=minutes)
    return e


@pytest.mark.unit
def test_stats_measure_the_record():
    rows = [_done(2.5, minutes=1), _done(-1, minutes=2), _done(-1, "XAUUSD", "high", 3),
            _done(3.0, "XAUUSD", "high", 4)]
    expired = _entry()
    expired.status = jr.EXPIRED
    s = stats(rows + [expired, _entry()])
    assert (s.won, s.lost, s.expired, s.pending) == (2, 2, 1, 1)
    assert s.total_r == pytest.approx(3.5) and s.avg_r == pytest.approx(0.88, abs=0.01)
    assert s.win_rate == 0.5 and s.profit_factor == pytest.approx(2.75)
    assert s.max_drawdown_r == pytest.approx(2.0)                  # 2.5 → 0.5
    assert [r for _, r in s.curve] == [2.5, 1.5, 0.5, 3.5]
    assert s.fill_rate == pytest.approx(4 / 5)
    assert list(s.by_symbol) == ["XAUUSD", "EURUSD"]               # best first
    assert s.by_conviction["high"]["n"] == 2


@pytest.mark.unit
def test_stats_of_an_empty_journal():
    s = stats([])
    assert s.total == 0 and s.win_rate is None and s.curve == []


@pytest.mark.unit
def test_orders_import_from_a_saved_agent_report_once(tmp_path):
    import json

    report = tmp_path / "2026-10-06_1017_agents.json"
    report.write_text(json.dumps({
        "scanned_at": "2026-10-06T10:17:00+00:00",
        "setups": [{"symbol": "XAUUSD", "spread_pips": 3.0, "strategy": "smc"}],
        "review": {"orders": [{
            "symbol": "XAUUSD", "direction": "long", "entry": 4125.08, "stop": 4119.46, "target": 4151.95,
            "rr": 4.48, "expires_at": "2026-10-06T13:18:00+00:00", "conviction": "low",
            "rationale": "FVG entry", "watch_for": "dollar", "scanner_score": 77}]},
    }), encoding="utf-8")
    book = Journal(tmp_path / "j.db")
    added = book.import_report(report)
    assert [(e.symbol, e.entry, e.spread) for e in added] == [("XAUUSD", 4125.08, pytest.approx(0.3))]
    assert added[0].created_at == datetime(2026, 10, 6, 10, 17, tzinfo=UTC)
    assert book.import_report(report) == []
