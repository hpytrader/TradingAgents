"""The Asian-range breakout: the first close outside the night's range, traded on the retest."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.fx import asian
from tradingagents.fx.lab import _Quote

DAY = datetime(2026, 9, 8, tzinfo=UTC)                   # a Tuesday


def _m5(moves, until, high=1.1710, low=1.1690):
    """Five-minute bars from the day before: quiet inside 1.1690-1.1710 overnight, then ``moves``.

    ``moves`` maps a UTC time to the (high, low, close) of that bar.
    """
    idx = pd.date_range(DAY - timedelta(days=2), until - timedelta(minutes=5), freq="5min", tz="UTC")
    mid = (high + low) / 2
    df = pd.DataFrame({"open": mid, "high": mid + 0.0003, "low": mid - 0.0003, "close": mid, "volume": 1},
                      index=idx)
    df.loc[DAY + timedelta(hours=1), ["high", "low"]] = [high, mid]          # the night's extremes
    df.loc[DAY + timedelta(hours=4), ["high", "low"]] = [mid, low]
    for t, (h, lo, c) in moves.items():
        df.loc[t, ["high", "low", "close"]] = [h, lo, c]
        df.loc[t:, "close"] = c                                              # price stays where it went
        later = df.index > t
        df.loc[later, "high"] = c + 0.0003
        df.loc[later, "low"] = c - 0.0003
    return df


def _feed(m5):
    h1 = m5.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})

    def candles(symbol, gran, count):
        return (m5 if gran == "M5" else h1).iloc[-count:]

    def quote(symbol):
        c = float(m5["close"].iloc[-1])
        return _Quote(symbol, c - 0.00005, c + 0.00005, m5.index[-1])
    return candles, quote


def _scan(m5, now):
    candles, quote = _feed(m5)
    return asian.scan_asian(candles, quote, ["EURUSD"], now=now, any_session=True)


@pytest.mark.unit
def test_an_upside_break_becomes_a_buy_limit_on_the_retest():
    now = DAY + timedelta(hours=7)
    m5 = _m5({DAY + timedelta(hours=6, minutes=30): (1.1718, 1.1705, 1.1716)}, now)
    [s] = _scan(m5, now).setups
    assert s.direction == "long" and s.entry == 1.1710 and s.stop == 1.1700
    assert s.zone_low == 1.1690 and s.zone_high == 1.1710 and s.strategy == "asian_breakout"
    assert s.rr == pytest.approx(2.0, abs=0.02) and s.target > s.entry
    assert "first five-minute close above the Asian high at 06:30 UTC" in s.reasons[1]


@pytest.mark.unit
def test_the_first_side_broken_decides_the_day():
    now = DAY + timedelta(hours=8)
    m5 = _m5({DAY + timedelta(hours=6, minutes=10): (1.1700, 1.1683, 1.1685),
              DAY + timedelta(hours=7): (1.1720, 1.1700, 1.1716)}, now)
    result = _scan(m5, now)
    assert not result.setups                                   # down first, so the later upside break is not traded
    assert "the downside break failed" in result.skipped[0][1]


@pytest.mark.unit
def test_a_failed_break_back_inside_the_range_is_not_traded():
    now = DAY + timedelta(hours=7, minutes=30)
    m5 = _m5({DAY + timedelta(hours=6, minutes=30): (1.1718, 1.1705, 1.1716),
              DAY + timedelta(hours=7): (1.1716, 1.1698, 1.1702)}, now)
    [(_, why)] = _scan(m5, now).skipped
    assert "back through the broken Asian high" in why


@pytest.mark.unit
def test_no_trade_before_the_range_closes_or_on_a_range_too_wide():
    early = DAY + timedelta(hours=5)
    assert "not complete" in _scan(_m5({}, early), early).skipped[0][1]
    now = DAY + timedelta(hours=7)
    wide = _m5({DAY + timedelta(hours=6, minutes=30): (1.1830, 1.1790, 1.1825)}, now, high=1.1800, low=1.1600)
    assert "outside 1-5x" in _scan(wide, now).skipped[0][1]


@pytest.mark.unit
def test_an_old_breakout_is_not_offered():
    now = DAY + timedelta(hours=10)
    m5 = _m5({DAY + timedelta(hours=6, minutes=10): (1.1718, 1.1705, 1.1716)}, now)
    assert "more than 3h old" in _scan(m5, now).skipped[0][1]


@pytest.mark.unit
def test_the_lab_can_replay_the_breakout_model():
    from tradingagents.fx import lab
    assert lab.granularities(model="asian") == ("M5", "H1")
    a = lab.log_key(["EURUSD"], DAY, DAY + timedelta(days=1), lab.Rules())
    b = lab.log_key(["EURUSD"], DAY, DAY + timedelta(days=1), lab.Rules(model="asian"))
    assert a != b


@pytest.mark.unit
def test_the_market_version_joins_a_fresh_break_now():
    now = DAY + timedelta(hours=6, minutes=40)
    m5 = _m5({DAY + timedelta(hours=6, minutes=30): (1.1718, 1.1705, 1.1716)}, now)
    candles, quote = _feed(m5)
    [s] = asian.scan_asian(candles, quote, ["EURUSD"], now=now, any_session=True, entry_mode="market").setups
    assert s.entry == pytest.approx(1.1716) and s.stop == 1.1700 and s.features["order"] == "market"
    later = DAY + timedelta(hours=7)
    m5 = _m5({DAY + timedelta(hours=6, minutes=30): (1.1718, 1.1705, 1.1716)}, later)
    candles, quote = _feed(m5)
    result = asian.scan_asian(candles, quote, ["EURUSD"], now=later, any_session=True, entry_mode="market")
    assert not result.setups and "too old to join at market" in result.skipped[0][1]


@pytest.mark.unit
def test_a_market_order_fills_on_the_first_bar_and_judges_the_stop_first():
    from tradingagents.fx import journal as jr
    from tradingagents.fx.journal import Entry
    idx = pd.date_range(DAY, periods=4, freq="5min", tz="UTC")
    bars = pd.DataFrame({"open": 1.1716, "high": [1.1720, 1.1760, 1.1717, 1.1717],
                         "low": [1.1712, 1.1714, 1.1715, 1.1715], "close": 1.1716}, index=idx)
    e = Entry(id="x", created_at=DAY, symbol="EURUSD", direction="long", entry=1.1716, stop=1.1700,
              target=1.1750, planned_rr=2.0, spread=0.0001, expires_at=DAY + timedelta(hours=4), order="market")
    jr.simulate(e, bars, DAY + timedelta(hours=1))
    assert e.filled_at == DAY and e.status == jr.WON and e.exit_at == DAY + timedelta(minutes=5)
    both = bars.copy()
    both.loc[DAY, ["high", "low"]] = [1.1760, 1.1690]                          # stop and target in the first bar
    e2 = Entry(id="y", created_at=DAY, symbol="EURUSD", direction="long", entry=1.1716, stop=1.1700,
               target=1.1750, planned_rr=2.0, spread=0.0001, expires_at=DAY + timedelta(hours=4), order="market")
    jr.simulate(e2, both, DAY + timedelta(hours=1))
    assert e2.status == jr.LOST
