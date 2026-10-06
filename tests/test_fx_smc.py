"""SMC structure detection and the SMC scanner, on hand-built candle sequences.

The main fixture is a textbook London-session long on EURUSD: the 1-hour chart
trends up; at 07:35 UTC a 5-minute bar sweeps the Asian low and closes back
above it; a bearish candle (the order block) is followed by displacement that
leaves a fair value gap and breaks the last swing high (the CHoCH); price then
holds above the gap. A mirrored copy gives the textbook short.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.fx import smc
from tradingagents.fx.smc_scanner import ScanWindow, market_open, scan_smc
from tradingagents.fx.verify import check_levels, verify

NOW = datetime(2026, 10, 6, 8, 30, tzinfo=UTC)          # 09:30 in London
PIVOT = 1.1750


def _bar(t, o, h, low, c):
    return {"time": t, "open": o, "high": h, "low": low, "close": c, "volume": 100}


def _wave(start_t, n, start, end, amp=0.0003):
    """``n`` M5 bars drifting from ``start`` to ``end`` with a small zig-zag."""
    rows = []
    for k in range(n):
        mid = start + (end - start) * k / max(n - 1, 1) + (amp if k % 4 in (1, 2) else -amp) / 2
        rows.append(_bar(start_t + timedelta(minutes=5 * k), mid, mid + amp, mid - amp, mid))
    return rows


def _m5(*, sweep=True, mitigate=False, mitigate_low=1.1729, spent=False):
    t = datetime(2026, 10, 4, 21, 0, tzinfo=UTC)          # previous trading day starts
    rows = []
    rows += _wave(t, 144, 1.1720, 1.1790)                   # Oct 5 to 09:00: up to the day high
    rows += _wave(t + timedelta(hours=12), 144, 1.1790, 1.1710)  # down to the day low
    rows[100]["high"] = 1.1800                               # previous day high
    rows[200]["low"] = 1.1700                                # previous day low
    t = datetime(2026, 10, 5, 21, 0, tzinfo=UTC)
    rows += _wave(t, 36, 1.1730, 1.1735)                     # evening drift
    t = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)
    asia = _wave(t, 72, 1.1735, 1.1735, amp=0.0010)          # Asian range
    asia[20]["low"] = 1.1720                                 # Asian low
    asia[50]["high"] = 1.1750                                # Asian high
    rows += asia
    t = datetime(2026, 10, 6, 6, 0, tzinfo=UTC)
    for k in range(10):                                      # 06:00-06:45 rise to a swing high
        p = 1.1730 + 0.0001 * k
        rows.append(_bar(t + timedelta(minutes=5 * k), p, p + 0.0002, p - 0.0002, p + 0.0001))
    rows[-1]["high"] = 1.1741
    t = datetime(2026, 10, 6, 6, 50, tzinfo=UTC)
    for k in range(9):                                       # 06:50-07:30 clean decline
        p = 1.1738 - 0.0001 * k
        rows.append(_bar(t + timedelta(minutes=5 * k), p, p + 0.00005, p - 0.00015, p - 0.0001))
    t = datetime(2026, 10, 6, 7, 35, tzinfo=UTC)
    low, ob_low = (1.1712, 1.1719) if sweep else (1.1722, 1.1721)   # no bar under 1.1720
    rows.append(_bar(t, 1.1728, 1.1729, low, 1.1724))        # the sweep of the Asian low
    rows.append(_bar(t + timedelta(minutes=5), 1.1726, 1.1727, ob_low, 1.1722))   # order block
    rows.append(_bar(t + timedelta(minutes=10), 1.1722, 1.1735, 1.1721, 1.1734))  # displacement
    rows.append(_bar(t + timedelta(minutes=15), 1.1734, 1.1746, 1.1733, 1.1745))  # CHoCH
    t = datetime(2026, 10, 6, 7, 55, tzinfo=UTC)
    for k in range(7):                                       # 07:55-08:25 holds above the gap
        p = 1.1743 + (0.0002 if k % 2 else -0.0001)
        low_k = mitigate_low if (mitigate and k == 3) else p - 0.0003
        high_k = 1.1810 if (spent and k == 2) else p + 0.0003   # a run through every target, then back
        rows.append(_bar(t + timedelta(minutes=5 * k), p, high_k, low_k, p))
    frame = pd.DataFrame(rows).set_index("time")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _h1():
    """A clean 1-hour uptrend: six bars up, four back, twelve times over."""
    rows, price = [], 1.1529
    t = NOW.replace(minute=0) - timedelta(hours=119)
    for _wave in range(12):
        for k in range(10):
            step = 0.0005 if k < 6 else -0.0003
            o, c = price, price + step
            peak = 0.0004 if k == 5 else 0.0          # each wave's high and low stand out
            trough = 0.0004 if k == 9 else 0.0
            rows.append(_bar(t, o, max(o, c) + 0.0010 + peak, min(o, c) - 0.0010 - trough, c))
            price, t = c, t + timedelta(hours=1)
    frame = pd.DataFrame(rows).set_index("time")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _mirror(frame):
    """Reflect prices about PIVOT: an uptrend becomes a downtrend, lows become highs."""
    out = pd.DataFrame(index=frame.index)
    out["open"] = 2 * PIVOT - frame["open"]
    out["close"] = 2 * PIVOT - frame["close"]
    out["high"] = 2 * PIVOT - frame["low"]
    out["low"] = 2 * PIVOT - frame["high"]
    out["volume"] = frame["volume"]
    return out


def _fetchers(m5, h1, spread=0.0001):
    def candles(symbol, granularity, count):
        return {"M5": m5, "H1": h1}[granularity]

    def quote(symbol):
        mid = float(m5["close"].iloc[-1])
        return SimpleNamespace(bid=mid - spread / 2, ask=mid + spread / 2)

    return candles, quote


# ---------------------------------------------------------------------------
# Structure primitives
# ---------------------------------------------------------------------------

def _frame(closes, spread=0.5):
    rows = [{"open": c, "high": c + spread, "low": c - spread, "close": c} for c in closes]
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_breaks_label_continuation_and_reversal():
    # Up to a swing high at 5, dip, break above it (BOS up), roll over and
    # close below the last swing low (CHoCH down).
    closes = [1, 2, 3, 5, 3, 2, 3, 4, 6, 7, 6, 4, 3, 1, 0]
    found = smc.breaks(_frame(closes), wing=2)
    assert [(b.kind, b.direction) for b in found][:2] == [("BOS", "up"), ("CHoCH", "down")]


@pytest.mark.unit
def test_a_swing_is_not_breakable_before_it_is_confirmed():
    closes = [1, 2, 3, 5, 6]          # 5 is not a confirmed swing high yet
    assert smc.breaks(_frame(closes), wing=2) == []


@pytest.mark.unit
def test_fair_value_gap_and_order_block_are_found():
    rows = [
        {"open": 10, "high": 10.5, "low": 9.0, "close": 9.2},    # down candle: order block
        {"open": 9.3, "high": 11.0, "low": 9.2, "close": 10.9},  # displacement
        {"open": 10.9, "high": 12.0, "low": 10.8, "close": 11.9},
    ]
    frame = pd.DataFrame(rows)
    ob = smc.order_block(frame, 0, 2, "up")
    gaps = smc.fair_value_gaps(frame, 0, 2, "up")
    assert (ob.low, ob.high) == (9.0, 10.5)
    assert [(g.low, g.high) for g in gaps] == [(10.5, 10.8)]
    assert gaps[0].overlaps(ob)


@pytest.mark.unit
def test_pools_include_previous_day_and_asian_range():
    m5 = _m5()
    names = {p.name: p.level for p in smc.pools(m5, NOW, atr=smc.atr(m5))}
    assert names["previous day high"] == pytest.approx(1.1800)
    assert names["previous day low"] == pytest.approx(1.1700)
    assert names["Asian low"] == pytest.approx(1.1720)
    assert names["Asian high"] == pytest.approx(1.1750)


@pytest.mark.unit
def test_trading_days_roll_at_new_york_five_pm():
    before = pd.Timestamp("2026-10-05 20:55", tz="UTC")     # 16:55 New York
    after = pd.Timestamp("2026-10-05 21:00", tz="UTC")      # 17:00 New York
    assert smc.trading_day(after) == smc.trading_day(before) + timedelta(days=1)


def _ny(y, m, d, hh, mm=0):
    from tradingagents.fx.smc import NEW_YORK
    return NEW_YORK.localize(datetime(y, m, d, hh, mm)).astimezone(UTC)


@pytest.mark.unit
def test_the_default_window_is_two_am_to_noon_new_york():
    w = ScanWindow()
    assert w.current_end(_ny(2026, 10, 6, 6, 11)) == _ny(2026, 10, 6, 12)   # Tuesday 06:11: inside
    assert w.current_end(_ny(2026, 10, 6, 2, 0)) is not None                 # opens at 02:00
    assert w.current_end(_ny(2026, 10, 6, 1, 59)) is None
    assert w.current_end(_ny(2026, 10, 6, 12, 0)) is None                    # closes at 12:00
    assert w.next_open(_ny(2026, 10, 6, 12, 30)) == _ny(2026, 10, 7, 2)


@pytest.mark.unit
def test_the_window_follows_new_york_clocks_through_daylight_saving():
    w = ScanWindow()
    assert w.next_open(_ny(2026, 10, 6, 13)) == datetime(2026, 10, 7, 6, 0, tzinfo=UTC)   # EDT
    assert w.next_open(_ny(2026, 11, 3, 13)) == datetime(2026, 11, 4, 7, 0, tzinfo=UTC)   # EST


@pytest.mark.unit
def test_weekends_are_never_in_the_window():
    w = ScanWindow()
    assert not market_open(_ny(2026, 10, 9, 17, 0))          # Friday 17:00
    assert not market_open(_ny(2026, 10, 10, 6, 0))          # Saturday
    assert not market_open(_ny(2026, 10, 11, 16, 59))        # Sunday before the open
    assert market_open(_ny(2026, 10, 11, 17, 0))
    assert w.current_end(_ny(2026, 10, 10, 6, 0)) is None
    assert w.next_open(_ny(2026, 10, 9, 13, 0)) == _ny(2026, 10, 12, 2)    # Friday → Monday


@pytest.mark.unit
def test_a_window_may_cross_midnight():
    w = ScanWindow.parse("19:00-12:00")
    assert w.current_end(_ny(2026, 10, 6, 23, 0)) == _ny(2026, 10, 7, 12)
    assert w.current_end(_ny(2026, 10, 7, 3, 0)) == _ny(2026, 10, 7, 12)
    assert w.current_end(_ny(2026, 10, 7, 15, 0)) is None
    assert w.current_end(_ny(2026, 10, 11, 19, 0)) is not None   # Sunday evening, market open
    assert w.current_end(_ny(2026, 10, 9, 19, 0)) is None        # Friday evening, market shut


@pytest.mark.unit
@pytest.mark.parametrize("text", ["2-12", "02:00", "12:00-12:00", "25:00-12:00"])
def test_bad_windows_are_refused(text):
    with pytest.raises(ValueError):
        ScanWindow.parse(text)


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_textbook_long_is_found():
    candles, quote = _fetchers(_m5(), _h1())
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)

    assert len(result.setups) == 1, result.skipped
    s = result.setups[0]
    assert s.strategy == "smc" and s.order_type == "BUY LIMIT"
    assert s.zone_low <= s.entry <= s.zone_high
    assert s.entry == pytest.approx(1.1730, abs=0.00011)            # middle of the FVG
    assert s.invalidation == pytest.approx(1.1712)
    assert s.stop < s.invalidation < s.entry < s.price < s.target
    assert s.rr >= 2.0
    text = " ".join(s.reasons)
    assert "Asian low" in text and "structure shift" in text and "inside the order block" in text
    assert s.expires_at == NOW + timedelta(hours=4)                   # before the 12:00 close
    assert "London session" in text


@pytest.mark.unit
def test_mirrored_chart_gives_the_textbook_short():
    candles, quote = _fetchers(_mirror(_m5()), _mirror(_h1()))
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)

    assert len(result.setups) == 1, result.skipped
    s = result.setups[0]
    assert s.order_type == "SELL LIMIT"
    assert s.target < s.price < s.entry < s.invalidation < s.stop
    assert "Asian high" in " ".join(s.reasons)


@pytest.mark.unit
def test_no_sweep_means_no_setup():
    candles, quote = _fetchers(_m5(sweep=False), _h1())
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)
    assert result.setups == []
    # Without the 07:35 sweep, the only sweep left (of overnight equal lows at
    # 02:35) is never followed by a shift, so there is no trade.
    assert "liquidity swept" in result.skipped[0][1]


@pytest.mark.unit
def test_a_pool_cannot_be_swept_before_it_exists():
    m5 = _m5()
    found = smc.sweeps(m5, smc.pools(m5, NOW, smc.atr(m5)), "low", 0)
    assert found and all(s.i > s.pool.formed_i for s in found)


@pytest.mark.unit
def test_a_filled_gap_falls_back_to_the_order_block():
    # Price dipped to 1.1729: through the gap's middle (1.1730), above the
    # order block's (1.1723). The gap would have filled; the OB is still open.
    candles, quote = _fetchers(_m5(mitigate=True, mitigate_low=1.1729), _h1())
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)
    s = result.setups[0]
    assert (s.zone_low, s.zone_high) == (1.1719, 1.1727)
    assert s.entry == pytest.approx(1.1723)
    assert "the order block 1.1719–1.1727" in " ".join(s.reasons)


@pytest.mark.unit
def test_when_every_zone_is_filled_there_is_no_entry():
    candles, quote = _fetchers(_m5(mitigate=True, mitigate_low=1.1721), _h1())
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)
    assert result.setups == []
    assert "already been traded back into" in result.skipped[0][1]


@pytest.mark.unit
def test_a_stop_is_never_closer_than_the_noise(monkeypatch):
    import tradingagents.fx.smc_scanner as smc_scanner

    candles, quote = _fetchers(_m5(), _h1())
    normal = scan_smc(candles, quote, ["EURUSD"], now=NOW).setups[0]
    monkeypatch.setattr(smc_scanner, "MIN_RISK_H1_ATR", 1.0)    # demand a wider stop
    wide = scan_smc(candles, quote, ["EURUSD"], now=NOW).setups[0]

    assert wide.risk_pips > normal.risk_pips
    assert wide.risk_pips == pytest.approx(wide.atr_pips, rel=0.05)
    assert wide.stop < wide.invalidation                          # still beyond the sweep
    assert "stop widened" in " ".join(wide.reasons)
    assert "stop widened" not in " ".join(normal.reasons)


@pytest.mark.unit
def test_outside_the_window_nothing_is_built_unless_asked():
    candles, quote = _fetchers(_m5(), _h1())
    later = NOW + timedelta(hours=8)                         # 12:30 New York
    result = scan_smc(candles, quote, ["EURUSD"], now=later)
    assert result.setups == [] and not result.ran
    assert "outside the scan window (02:00–12:00 New York time)" in result.skipped[0][1]
    assert "Next window opens Wed 02:00 (in 13h 30m)" in result.skipped[0][1]


@pytest.mark.unit
def test_an_order_is_cancelled_when_the_window_closes():
    candles, quote = _fetchers(_m5(), _h1())
    late = scan_smc(candles, quote, ["EURUSD"], window=ScanWindow.parse("02:00-05:00"), now=NOW)
    assert late.setups[0].expires_at == _ny(2026, 10, 6, 5)


# ---------------------------------------------------------------------------
# Verification of SMC orders
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_smc_orders_must_stay_in_the_zone_and_behind_the_sweep():
    candles, quote = _fetchers(_m5(), _h1())
    s = scan_smc(candles, quote, ["EURUSD"], now=NOW).setups[0]

    assert check_levels(s, s.entry, s.stop, s.target, 2.0) is None
    assert "inside the" in check_levels(s, s.zone_high + 0.0005, s.stop, s.target, 2.0)
    assert "beyond the sweep" in check_levels(s, s.entry, s.invalidation + 0.0001, s.target, 2.0)

    accepted, _ = verify([{"symbol": "EURUSD", "entry": s.zone_high + 0.0005, "stop": s.stop,
                           "target": s.target}], [s], now=NOW)
    assert accepted[0].entry == s.entry and "scanner levels used" in accepted[0].notes[0]


@pytest.mark.unit
def test_fx_scan_defaults_to_smc_and_explains_an_off_window_run(monkeypatch):
    from typer.testing import CliRunner

    import tradingagents.dataflows.vendors.oanda as oanda
    import tradingagents.fx.smc_scanner as smc_scanner
    from cli.main import app

    candles, quote = _fetchers(_m5(), _h1())
    monkeypatch.setattr(oanda, "get_candles", lambda s, g="H1", n=300, **k: candles(s, g, n))
    monkeypatch.setattr(oanda, "get_quote", quote)
    real = smc_scanner.scan_smc
    monkeypatch.setattr(smc_scanner, "scan_smc",
                        lambda *a, **k: real(*a, now=NOW + timedelta(hours=8), **k))

    out = CliRunner().invoke(app, ["fx-scan", "--symbols", "EURUSD", "--no-save"],
                             env={"OANDA_API_TOKEN": "x", "COLUMNS": "200"}).output
    assert "outside the scan window" in out and "--any-session" in out
    assert "Saved:" not in out and "win probabilities" not in out

    monkeypatch.setattr(smc_scanner, "scan_smc", lambda *a, **k: real(*a, now=NOW, **k))
    out = CliRunner().invoke(app, ["fx-scan", "--symbols", "EURUSD", "--no-save"],
                             env={"OANDA_API_TOKEN": "x", "COLUMNS": "200"}).output
    assert "BUY LIMIT" in out and "EURUSD" in out


@pytest.mark.unit
def test_fxstreet_headlines_match_pairs_by_currency():
    from tradingagents.dataflows.vendors import fxstreet

    rss = """<?xml version="1.0"?><rss><channel>
      <item><title>US ISM Services PMI rises to 54.9 in September</title>
            <pubDate>Mon, 05 Oct 2026 14:00:00 GMT</pubDate></item>
      <item><title>ECB's Lagarde: inflation risks are balanced</title>
            <pubDate>Mon, 05 Oct 2026 13:00:00 GMT</pubDate></item>
      <item><title>Gold climbs as yields slip</title>
            <pubDate>Mon, 05 Oct 2026 12:00:00 GMT</pubDate></item>
      <item><title>Decade-high copper demand</title>
            <pubDate>Mon, 05 Oct 2026 11:00:00 GMT</pubDate></item>
      <item><title>Old story</title><pubDate>Sat, 03 Oct 2026 11:00:00 GMT</pubDate></item>
    </channel></rss>"""
    items = fxstreet.parse(rss)
    now = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)

    eur = [h.title for h in fxstreet.headlines_for("EURUSD", now, items=items)]
    assert eur == ["US ISM Services PMI rises to 54.9 in September",
                   "ECB's Lagarde: inflation risks are balanced"]
    assert [h.title for h in fxstreet.headlines_for("XAUUSD", now, items=items)][-1] == "Gold climbs as yields slip"
    assert fxstreet.headlines_for("USDCAD", now, items=items)[0].title.startswith("US ISM")
    assert all("Decade" not in h.title for h in fxstreet.headlines_for("USDCAD", now, items=items))
    assert len(fxstreet.market_headlines(now, items=items)) == 4     # the old story is outside 12h


@pytest.mark.unit
def test_the_target_skips_a_pool_sitting_right_at_price():
    from tradingagents.fx import smc_scanner
    from tradingagents.fx.smc import Pool

    m5, h1 = _m5(), _h1()
    mid, atr5 = float(m5["close"].iloc[-1]), smc.atr(m5)
    hair = Pool("equal highs", mid + 0.05 * atr5, "high", 1)          # just above price
    far = Pool("previous day high", mid + 6 * atr5, "high", 3)
    target, name = smc_scanner._target(m5, h1, [hair, far], True, mid, mid - 2 * atr5,
                                       1.5 * atr5, 0.00005, 2.0, smc.atr(h1), atr5)
    assert name != "equal highs"
    assert target > mid


@pytest.mark.unit
def test_a_move_that_already_reached_its_targets_is_not_offered():
    # The XAUUSD case from 2026-10-06: after the shift price ran through the
    # nearby liquidity, then pulled back under it. The entry zone was never
    # revisited, but the trade has already played out.
    candles, quote = _fetchers(_m5(spent=True), _h1())
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)
    assert result.setups == []
    assert "the move has happened" in result.skipped[0][1]


@pytest.mark.unit
def test_an_old_structure_shift_is_stale(monkeypatch):
    import tradingagents.fx.smc_scanner as smc_scanner

    monkeypatch.setattr(smc_scanner, "MAX_SHIFT_AGE", timedelta(minutes=20))   # the shift is 40 min old
    candles, quote = _fetchers(_m5(), _h1())
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)
    assert result.setups == [] and "old; the move has most likely played out" in result.skipped[0][1]


@pytest.mark.unit
def test_an_entry_far_from_price_is_not_offered(monkeypatch):
    import tradingagents.fx.smc_scanner as smc_scanner

    monkeypatch.setattr(smc_scanner, "MAX_ENTRY_H1_ATR", 0.05)
    candles, quote = _fetchers(_m5(), _h1())
    result = scan_smc(candles, quote, ["EURUSD"], now=NOW)
    assert result.setups == [] and "unlikely to fill this session" in result.skipped[0][1]


@pytest.mark.unit
def test_swing_points_are_liquidity_for_a_swing_failure():
    m5 = _m5()
    names = {p.name for p in smc.pools(m5, NOW, atr=smc.atr(m5))}
    assert {"swing low", "swing high"} <= names


@pytest.mark.unit
def test_a_level_closed_through_cannot_be_swept_later():
    from tradingagents.fx.smc import Pool

    frame = pd.DataFrame({"open": [1.2, 1.1, 1.0], "high": [1.25, 1.15, 1.1],
                          "low": [1.15, 0.9, 0.95], "close": [1.2, 0.92, 1.05]})
    pool = Pool("swing low", 1.0, "low", 0, formed_i=0)
    assert smc.sweeps(frame, [pool], "low", 0) == []                 # bar 1 closed below: broken
    intact = Pool("swing low", 0.97, "low", 0, formed_i=1)
    assert [s.i for s in smc.sweeps(frame, [intact], "low", 0)] == [2]
