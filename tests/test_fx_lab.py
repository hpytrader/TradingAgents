"""Quinn's lab: the baseline backtest replays the scanner over past prices without peeking."""

from datetime import UTC, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from tradingagents.fx import lab
from tradingagents.fx.scanner import ScanResult, Setup

MON = datetime(2026, 9, 7, 6, 0, tzinfo=UTC)            # Monday 02:00 New York: the window opens


def _bars(start, n, price=1.1700, freq="5min"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": price, "high": price + 0.0002, "low": price - 0.0002, "close": price,
                         "volume": 1}, index=idx)


def _setup(t, symbol="EURUSD", entry=1.1690, stop=1.1680, target=1.1720, direction="long"):
    return Setup(symbol=symbol, direction=direction, entry=entry, stop=stop, target=target, rr=2.8, score=70,
                 risk_pips=10, reward_pips=30, spread_pips=1, atr_pips=15, price=1.1700,
                 target_kind="equal highs", expires_at=t + timedelta(hours=4), strategy="smc",
                 zone_low=entry - 0.0003, zone_high=entry + 0.0003, invalidation=stop + 0.0003)


def _feed(m5):
    h1 = m5.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return lab.HistoricalFeed({("EURUSD", "M5"): m5, ("EURUSD", "H1"): h1})


@pytest.mark.unit
def test_the_feed_never_shows_a_bar_that_had_not_closed():
    m5 = _bars(MON - timedelta(days=3), 2000)
    feed = _feed(m5)
    for now in (MON, MON + timedelta(minutes=7), MON + timedelta(hours=3, minutes=1)):
        feed.now = now
        five, hour = feed.candles("EURUSD", "M5", 600), feed.candles("EURUSD", "H1", 300)
        assert (five.index + timedelta(minutes=5) <= now).all() and len(five) == 600
        assert (hour.index + timedelta(hours=1) <= now).all()
        assert five.index[-1] + timedelta(minutes=10) > now          # but the latest closed bar is there
    feed.now = MON
    q = feed.quote("EURUSD")
    assert q.ask - q.bid == pytest.approx(lab.TYPICAL_SPREAD_PIPS["EURUSD"] * 0.0001)


@pytest.mark.unit
def test_scans_run_only_in_the_window_on_trading_days():
    times = list(lab.scan_times(MON - timedelta(days=2), MON + timedelta(days=2), lab.Rules()))
    ny = [t.astimezone(lab.NEW_YORK) for t in times]
    assert all(time(2, 0) <= t.time() < time(12, 0) for t in ny)
    assert {t.weekday() for t in ny} == {0, 1}                     # not the weekend
    assert times[0] == MON and times[1] - times[0] == timedelta(minutes=10)
    assert len([t for t in ny if t.weekday() == 0]) == 60          # ten hours, every ten minutes


def _price_path(start, n, events):
    """Flat bars with the given (minutes after start, high, low) overrides."""
    m5 = _bars(start, n)
    for minutes, hi, lo in events:
        t = start + timedelta(minutes=minutes)
        m5.loc[t, "high"], m5.loc[t, "low"] = hi, lo
    return m5


@pytest.mark.unit
def test_an_order_is_placed_once_filled_after_the_scan_and_settled_like_the_journal(monkeypatch):
    start = MON - timedelta(days=2)
    offset = int((MON - start).total_seconds() // 60)
    # the entry trades at 02:20, the target at 03:00
    m5 = _price_path(start, 1200, [(offset + 20, 1.1702, 1.1689), (offset + 60, 1.1721, 1.1700)])
    feed = _feed(m5)
    calls = []

    def fake_scan(candles, quote, symbols, *, now, **kw):
        calls.append(now)
        return ScanResult(scanned_at=now, setups=[_setup(MON)])     # the same idea every scan

    monkeypatch.setattr(lab, "scan_smc", fake_scan)
    entries = lab.replay(feed, ["EURUSD"], MON, MON + timedelta(hours=10))
    assert len(entries) == 1                                         # seen again: not handed over again
    e = entries[0]
    assert e.created_at == MON and e.status == "won"
    assert e.filled_at == MON + timedelta(minutes=20)
    assert e.result_r == pytest.approx((0.0030 - 0.0001) / (0.0010 + 0.0001), abs=0.01)   # net of the spread
    assert len(calls) == 60


@pytest.mark.unit
def test_one_order_per_symbol_and_side_and_at_most_final_per_scan(monkeypatch):
    m5 = _bars(MON - timedelta(days=2), 1200)
    feed = _feed(m5)
    feed.frames[("GBPUSD", "M5")] = feed.frames[("EURUSD", "M5")]
    feed.frames[("GBPUSD", "H1")] = feed.frames[("EURUSD", "H1")]
    feed._ends[("GBPUSD", "M5")] = feed._ends[("EURUSD", "M5")]
    feed._ends[("GBPUSD", "H1")] = feed._ends[("EURUSD", "H1")]
    shifts = iter(range(1000))

    def fake_scan(candles, quote, symbols, *, now, **kw):
        k = next(shifts) * 0.00001                                  # a "new" zone every scan
        return ScanResult(scanned_at=now, setups=[_setup(now, entry=1.1650 - k, stop=1.1640 - k),
                                                  _setup(now, symbol="GBPUSD", entry=1.1650, stop=1.1640)])

    monkeypatch.setattr(lab, "scan_smc", fake_scan)
    entries = lab.replay(feed, ["EURUSD", "GBPUSD"], MON, MON + timedelta(hours=1),
                         lab.Rules(final=1))
    assert [e.symbol for e in entries] == ["EURUSD"]                 # final=1; EURUSD long stays pending


@pytest.mark.unit
def test_history_downloads_once_then_only_tops_up(tmp_path):
    calls = []

    def fetch(symbol, gran, start, end):
        calls.append((start, end))
        idx = pd.date_range(pd.Timestamp(start).ceil("1h"), end, freq="1h", tz="UTC", inclusive="left")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx)

    h = lab.History(tmp_path)
    a = datetime(2026, 1, 1, tzinfo=UTC)
    first = h.ensure("EURUSD", "H1", a, a + timedelta(days=10), fetch)
    assert len(first) == 240 and h.path("EURUSD", "H1").exists()
    again = h.ensure("EURUSD", "H1", a, a + timedelta(days=12), fetch)
    assert len(again) == 288 and calls[1][0] > a + timedelta(days=9)      # only the two new days


@pytest.mark.unit
def test_oanda_history_is_fetched_page_by_page(monkeypatch):
    from tradingagents.dataflows.vendors import oanda
    start = datetime(2026, 1, 5, tzinfo=UTC)
    pages = []

    def fake_get(path, params):
        t0 = pd.Timestamp(params["from"]).ceil("1h").to_pydatetime()     # OANDA's bars sit on the hour
        pages.append(t0)
        times = [t0 + timedelta(hours=i) for i in range(5000)]
        return {"candles": [{"time": t.isoformat(), "complete": True, "volume": 1,
                             "mid": {"o": "1", "h": "1", "l": "1", "c": "1"}} for t in times]}

    monkeypatch.setattr(oanda, "_get", fake_get)
    frame = oanda.get_candle_history("EURUSD", "H1", start, start + timedelta(hours=12000))
    assert len(frame) == 12000 and len(pages) <= 4 and frame.index.is_unique


def _finished(results, start=MON):
    out = []
    for i, r in enumerate(results):
        e = lab._entry(_setup(start), start + timedelta(days=i), i + 1)
        e.status = "won" if r > 0 else "lost"
        e.result_r, e.filled_at, e.exit_at = r, e.created_at, e.created_at + timedelta(hours=1)
        out.append(e)
    return out


@pytest.mark.unit
def test_the_report_grades_against_the_benchmarks_and_needs_enough_trades():
    few = _finished([2.5, -1, -1])
    r = lab.summarize(few, MON, MON + timedelta(days=5), ["EURUSD"], lab.Rules())
    assert r.verdict.startswith("Not enough trades")
    rng = np.random.default_rng(3)
    many = _finished([2.5 if x < 0.4 else -1.0 for x in rng.random(240)])
    r = lab.summarize(many, MON, MON + timedelta(days=300), ["EURUSD"], lab.Rules())
    assert r.verdict.startswith("Evidence of an edge") and r.grades["expectancy_r"] in ("on target", "strong")
    assert lab.grade("max_drawdown_r", 14.0) == "past the red line" and lab.grade("expectancy_r", 0.3) == "on target"
    text = lab.to_markdown(r)
    assert "| Win rate |" in text and "By quarter" in text
    losing = _finished([-1.0] * 150 + [2.0] * 30)
    assert lab.summarize(losing, MON, MON + timedelta(days=200), ["EURUSD"], lab.Rules()).verdict.startswith(
        "No edge")


@pytest.mark.unit
def test_the_higher_timeframe_model_reads_four_hour_and_fifteen_minute_bars(monkeypatch):
    from tradingagents.fx import smc_scanner
    asked = []

    def candles(symbol, gran, count):
        asked.append((gran, count))
        return _bars(MON - timedelta(days=60), 10, freq="15min")       # too short: the scan stops early

    q = lambda s: lab._Quote(s, 1.1699, 1.1701, MON)                     # noqa: E731
    smc_scanner.scan_smc(candles, q, ["EURUSD"], now=MON, timeframes=smc_scanner.profile("m15"))
    assert asked == [("H4", 300), ("M15", 600)]
    asked.clear()
    smc_scanner.scan_smc(candles, q, ["EURUSD"], now=MON)
    assert asked == [("H1", 300), ("M5", 600)]                          # the desk's model is unchanged
    assert lab.granularities("m15") == ("M5", "H4", "M15") and lab.granularities("m5") == ("M5", "H1")
    k5 = lab.log_key(["EURUSD"], MON, MON + timedelta(days=1), lab.Rules())
    k15 = lab.log_key(["EURUSD"], MON, MON + timedelta(days=1), lab.Rules(timeframe="m15"))
    assert k5 != k15
    with pytest.raises(ValueError):
        smc_scanner.profile("m1")
