"""Telegram alerts, the watcher cycle, the dashboard and the journal commands."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.fx import dashboard, journal as jr, telegram
from tradingagents.fx.journal import Entry, Journal, stats
from tradingagents.fx.scanner import ScanResult, Setup
from tradingagents.fx.smc_scanner import ScanWindow
from tradingagents.fx.watch import WatchState, cycle, setup_key

NOW = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)          # Tuesday 06:00 New York


def _setup(symbol="EURUSD", zone=(1.1695, 1.1705)):
    return Setup(symbol=symbol, direction="long", entry=1.1700, stop=1.1680, target=1.1750, rr=2.4,
                 score=70, risk_pips=20, reward_pips=50, spread_pips=1, atr_pips=15, price=1.1720,
                 target_kind="equal highs", expires_at=NOW + timedelta(hours=4), strategy="smc",
                 zone_low=zone[0], zone_high=zone[1], invalidation=1.1688)


def _order(setup):
    return SimpleNamespace(symbol=setup.symbol, direction=setup.direction, entry=setup.entry,
                           stop=setup.stop, target=setup.target, rr=setup.rr,
                           expires_at=setup.expires_at, conviction="medium",
                           rationale="Swept the Asian low <b>cleanly</b>.", watch_for="US data",
                           setup=setup)


def _flat(symbol, gran, count):
    idx = pd.date_range(NOW - timedelta(minutes=5), periods=5, freq="1min", tz="UTC")
    return pd.DataFrame({"open": 1.1720, "high": 1.1722, "low": 1.1718, "close": 1.1720}, index=idx)


class Harness:
    def __init__(self, tmp_path, setups):
        self.book = Journal(tmp_path / "j.db")
        self.setups = setups
        self.sent, self.reviews = [], 0
        self.state = WatchState()

    def scan(self, now):
        return ScanResult(scanned_at=now, setups=list(self.setups))

    def review(self, result, now):
        self.reviews += 1
        return SimpleNamespace(orders=[_order(s) for s in result.setups]), "report.md"

    def run(self, now, **kw):
        return cycle(now, window=ScanWindow(), journal=self.book, candles=_flat, scan_fn=self.scan,
                     review_fn=self.review, notify=self.sent.append, state=self.state, **kw)


@pytest.mark.unit
def test_new_setups_are_reviewed_once_and_announced(tmp_path):
    h = Harness(tmp_path, [_setup()])
    first = h.run(NOW)
    assert first.reviewed and [e.symbol for e in first.new_orders] == ["EURUSD"]
    assert any("BUY LIMIT EURUSD" in m for m in h.sent)

    again = h.run(NOW + timedelta(minutes=10))          # same idea: no second review
    assert not again.reviewed and h.reviews == 1

    h.setups.append(_setup("GBPUSD"))
    third = h.run(NOW + timedelta(minutes=20))
    assert third.reviewed and h.reviews == 2
    assert [e.symbol for e in third.new_orders] == ["GBPUSD"]   # EURUSD already open in the journal


@pytest.mark.unit
def test_the_daily_agent_cap_is_respected_and_resets(tmp_path):
    h = Harness(tmp_path, [_setup()])
    h.run(NOW, max_agent_runs=1)
    h.setups.append(_setup("GBPUSD"))
    capped = h.run(NOW + timedelta(minutes=10), max_agent_runs=1)
    assert not capped.reviewed and "limit reached" in capped.notes[0]
    assert sum("limit" in m for m in h.sent) == 1
    h.run(NOW + timedelta(minutes=20), max_agent_runs=1)
    assert sum("limit" in m for m in h.sent) == 1                # warned once
    tomorrow = h.run(NOW + timedelta(days=1), max_agent_runs=1)
    assert tomorrow.reviewed                                      # a new trading day


@pytest.mark.unit
def test_a_restart_remembers_what_was_reviewed_today(tmp_path):
    h = Harness(tmp_path, [_setup()])
    assert h.run(NOW, max_agent_runs=2).reviewed
    h.state = WatchState()                                        # the watcher was restarted
    again = h.run(NOW + timedelta(minutes=10), max_agent_runs=2)
    assert not again.reviewed and h.reviews == 1                  # same setup, no new ticket

    h.setups.append(_setup("GBPUSD"))
    h.state = WatchState()
    assert h.run(NOW + timedelta(minutes=20), max_agent_runs=2).reviewed
    h.setups.append(_setup("USDJPY"))
    h.state = WatchState()
    capped = h.run(NOW + timedelta(minutes=30), max_agent_runs=2)
    assert not capped.reviewed and "limit reached" in capped.notes[0]   # the count survived too
    h.state = WatchState()
    assert h.run(NOW + timedelta(days=1), max_agent_runs=2).reviewed     # a new day starts fresh


@pytest.mark.unit
def test_outside_the_window_it_only_settles_and_sums_up_once(tmp_path):
    h = Harness(tmp_path, [_setup()])
    h.run(NOW)
    h.sent.clear()
    after = h.run(NOW + timedelta(hours=7))                      # 13:00 New York
    assert not after.in_window and not after.reviewed
    assert sum("FX desk" in m for m in h.sent) == 1               # the morning summary
    h.run(NOW + timedelta(hours=8))
    assert sum("FX desk" in m for m in h.sent) == 1


@pytest.mark.unit
def test_a_failing_scan_or_review_does_not_stop_the_watcher(tmp_path):
    h = Harness(tmp_path, [_setup()])
    h.scan = lambda now: (_ for _ in ()).throw(RuntimeError("OANDA down"))
    assert "scan failed" in h.run(NOW).notes[0]
    h.scan = lambda now: ScanResult(scanned_at=now, setups=[_setup()])
    h.review = lambda result, now: (_ for _ in ()).throw(RuntimeError("quota"))
    assert "agent review failed" in h.run(NOW + timedelta(minutes=10)).notes[0]


@pytest.mark.unit
def test_setup_keys_separate_zones():
    assert setup_key(_setup()) != setup_key(_setup(zone=(1.1690, 1.1699)))


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _entry(**kw):
    base = {"id": "1", "created_at": NOW, "symbol": "XAUUSD", "direction": "long", "entry": 4125.08,
            "stop": 4119.46, "target": 4151.95, "planned_rr": 4.48, "spread": 0.3,
            "expires_at": NOW + timedelta(hours=3), "conviction": "low",
            "rationale": "Entry in an <unmitigated> FVG & stop beyond the sweep.",
            "watch_for": "a firming dollar"}
    base.update(kw)
    return Entry(**base)


@pytest.mark.unit
def test_messages_escape_agent_text_and_show_local_times():
    text = telegram.order_message(_entry())
    assert "BUY LIMIT XAUUSD" in text and "4125.08" in text and "4.48R" in text
    assert "&lt;unmitigated&gt;" in text and "&amp;" in text
    assert "Cancel 09:00 New York" in text
    won = _entry(status=jr.WON, result_r=4.2, note="target hit")
    assert "WON XAUUSD +4.20R" in telegram.result_message(won)
    assert "MISSED" in telegram.result_message(_entry(status=jr.MISSED))


@pytest.mark.unit
def test_telegram_errors_never_show_the_token(monkeypatch):
    import requests

    token = "123456:SECRET-TOKEN"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def boom(url, json, timeout):
        raise requests.ConnectionError(f"failed to reach {url}")
    monkeypatch.setattr(telegram.requests, "post", boom)
    with pytest.raises(telegram.TelegramError) as info:
        telegram.send("hi")
    assert token not in str(info.value) and "***" in str(info.value)


@pytest.mark.unit
def test_find_chats_lists_who_messaged_the_bot(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    payload = {"ok": True, "result": [{"message": {"chat": {"id": 777, "first_name": "Hpy", "last_name": "T"}}}]}
    monkeypatch.setattr(telegram.requests, "post",
                        lambda url, json, timeout: SimpleNamespace(json=lambda: payload, status_code=200))
    assert telegram.find_chats() == [("777", "Hpy T")]


@pytest.mark.unit
def test_configured_needs_both_settings(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert not telegram.configured()
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    assert telegram.configured()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dashboard_shows_the_record_and_escapes_agent_text(tmp_path):
    won = _entry(id="a", status=jr.WON, result_r=4.2, filled_at=NOW, exit_at=NOW + timedelta(hours=1),
                 rationale="<script>alert(1)</script>")
    lost = _entry(id="b", symbol="EURUSD", entry=1.12, stop=1.118, target=1.125, status=jr.LOST,
                  result_r=-1.0, filled_at=NOW, exit_at=NOW + timedelta(hours=2))
    pending = _entry(id="c", symbol="GBPUSD", entry=1.32, stop=1.318, target=1.326)
    rows = [won, lost, pending]
    page = dashboard.render(rows, stats(rows), now=NOW, live=True)

    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "+3.20R" in page and "50%" in page                      # total and win rate
    assert "✓" in page and "✕" in page                            # status never by colour alone
    assert 'http-equiv="refresh"' in page and "Window <b>open</b>" in page
    assert "GBPUSD" in page and "Pending" in page
    assert "http://" not in page and "https://" not in page        # loads nothing


@pytest.mark.unit
def test_an_empty_dashboard_explains_itself(tmp_path):
    path = dashboard.write(tmp_path / "d.html", [], stats([]), now=NOW + timedelta(hours=8))
    page = path.read_text(encoding="utf-8")
    assert "journal is empty" in page and "window closed" in page and "refresh" not in page


@pytest.mark.unit
def test_fx_journal_command_lists_and_writes_the_dashboard(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import cli.main as main

    book = Journal(tmp_path / "j.db")
    book.record([_order(_setup())], now=NOW)
    monkeypatch.setattr(main, "_fx_journal", lambda: book)
    monkeypatch.setattr(main, "_fx_dashboard_path", lambda: tmp_path / "dash.html")
    out = CliRunner().invoke(main.app, ["fx-journal", "--no-settle"], env={"COLUMNS": "160"}).output
    assert "EURUSD" in out and "pending" in out and "Dashboard:" in out
    assert (tmp_path / "dash.html").exists()


@pytest.mark.unit
def test_fx_telegram_command_guides_setup(monkeypatch):
    from typer.testing import CliRunner

    from cli.main import app

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    out = CliRunner().invoke(app, ["fx-telegram"]).output
    assert "@BotFather" in out and "TELEGRAM_BOT_TOKEN" in out
