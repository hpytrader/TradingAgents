"""The stage-1 forex scanner, run on synthetic price paths with known shapes."""

import json
import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.fx import indicators as ind, scan, spec_for
from tradingagents.fx.report import save, to_markdown
from tradingagents.fx.scanner import Setup, _rank

NOW = datetime(2026, 10, 6, 7, 0, tzinfo=UTC)


def _path(n, hours, start, drift, amp, period, end=NOW):
    """A trend (``drift`` per bar) with a sine-wave swing on top of it."""
    idx = pd.date_range(end=end - timedelta(hours=hours), periods=n, freq=f"{hours}h", tz="UTC")
    mids = [start + drift * i + amp * math.sin(2 * math.pi * i / period) for i in range(n)]
    return pd.DataFrame({
        "open": mids,
        "high": [m + amp * 0.3 for m in mids],
        "low": [m - amp * 0.3 for m in mids],
        "close": mids,
        "volume": 100,
    }, index=idx)


def _market(direction=1, scale=1.0, end=NOW, spread=0.0001):
    """Candle and quote fetchers for a pair trending up (1), down (-1) or flat (0)."""
    def candles(symbol, granularity, count):
        if granularity == "H4":
            return _path(count, 4, 1.10 * scale, 0.0003 * scale * direction, 0.002 * scale, 20, end)
        last = 1.10 * scale + 0.0003 * scale * direction * 259
        return _path(count, 1, last - 0.00008 * scale * direction * count,
                     0.00008 * scale * direction, 0.0025 * scale, 24, end)

    def quote(symbol):
        mid = float(candles(symbol, "H1", 300)["close"].iloc[-1])
        return SimpleNamespace(bid=mid - spread / 2, ask=mid + spread / 2)

    return candles, quote


@pytest.mark.unit
def test_pip_sizes_follow_retail_conventions():
    assert spec_for("EURUSD").pip == 0.0001
    assert spec_for("usdjpy").pip == 0.01
    assert spec_for("XAUUSD+").pip == 0.1
    assert spec_for("XAG/USD").decimals == 3
    with pytest.raises(ValueError):
        spec_for("NVDA")


@pytest.mark.unit
def test_uptrend_gives_a_buy_limit_below_price_with_at_least_2r():
    candles, quote = _market(direction=1)
    result = scan(candles, quote, ["EURUSD"], now=NOW)

    assert len(result.setups) == 1
    s = result.setups[0]
    assert s.direction == "long" and s.order_type == "BUY LIMIT"
    assert s.stop < s.entry < s.target
    assert s.entry < s.price
    assert s.rr >= 2.0
    assert 0 <= s.score <= 100
    assert s.expires_at == NOW + timedelta(hours=8)
    assert s.reasons


@pytest.mark.unit
def test_downtrend_gives_a_sell_limit_above_price():
    candles, quote = _market(direction=-1)
    s = scan(candles, quote, ["EURUSD"], now=NOW).setups[0]

    assert s.direction == "short" and s.order_type == "SELL LIMIT"
    assert s.target < s.entry < s.stop
    assert s.entry > s.price
    assert s.rr >= 2.0


@pytest.mark.unit
def test_rr_accounts_for_the_spread():
    candles, quote = _market(direction=1, spread=0.0001)
    s = scan(candles, quote, ["EURUSD"], now=NOW).setups[0]
    spread = 0.0001
    reward, risk = abs(s.target - s.entry), abs(s.entry - s.stop)
    assert s.rr == pytest.approx((reward - spread) / (risk + spread), abs=0.02)


@pytest.mark.unit
def test_a_higher_minimum_rr_filters_setups_out():
    candles, quote = _market(direction=1)
    result = scan(candles, quote, ["EURUSD"], min_rr=20, now=NOW)
    assert result.setups == []
    assert "20R room" in result.skipped[0][1]


@pytest.mark.unit
def test_no_trend_means_no_setup():
    candles, quote = _market(direction=0)
    result = scan(candles, quote, ["EURUSD"], now=NOW)
    assert result.setups == []
    assert result.skipped == [("EURUSD", "no clear 4-hour trend")]


@pytest.mark.unit
def test_stale_prices_are_treated_as_a_closed_market():
    candles, quote = _market(direction=1, end=NOW - timedelta(days=2))
    result = scan(candles, quote, ["EURUSD"], now=NOW)
    assert result.setups == []
    assert "market closed" in result.skipped[0][1]


@pytest.mark.unit
def test_one_failing_pair_does_not_stop_the_scan():
    candles, quote = _market(direction=1)

    def flaky(symbol, granularity, count):
        if symbol == "GBPUSD":
            raise RuntimeError("timeout")
        return candles(symbol, granularity, count)

    result = scan(flaky, quote, ["GBPUSD", "EURUSD", "NOTAPAIR"], now=NOW)
    assert [s.symbol for s in result.setups] == ["EURUSD"]
    reasons = dict(result.skipped)
    assert "timeout" in reasons["GBPUSD"]
    assert "NOTAPAIR" in reasons


@pytest.mark.unit
def test_metals_are_priced_in_their_own_units():
    candles, quote = _market(direction=1, scale=2000, spread=0.3)
    s = scan(candles, quote, ["XAUUSD"], now=NOW).setups[0]
    assert s.symbol == "XAUUSD"
    assert s.spread_pips == pytest.approx(3.0)       # 0.30 dollars = 3 gold pips
    assert s.entry == round(s.entry, 2)


def _setup(symbol, direction, score):
    return Setup(symbol=symbol, direction=direction, entry=1, stop=0.9, target=1.2, rr=2,
                 score=score, risk_pips=10, reward_pips=20, spread_pips=1, atr_pips=10,
                 price=1.05, target_kind="structure", expires_at=NOW)


@pytest.mark.unit
def test_ranking_caps_the_same_currency_bet():
    # Three ways to sell the dollar, then an unrelated cross.
    setups = [_setup("EURUSD", "long", 90), _setup("GBPUSD", "long", 80),
              _setup("USDJPY", "short", 70), _setup("EURGBP", "short", 60)]
    ranked = _rank(setups, top=10, max_per_currency=2)
    assert [s.symbol for s in ranked] == ["EURUSD", "GBPUSD", "EURGBP"]


@pytest.mark.unit
def test_ranking_keeps_only_the_top_n():
    setups = [_setup(sym, "long", score) for sym, score in
              [("EURUSD", 50), ("GBPJPY", 90), ("AUDCAD", 70)]]
    assert [s.symbol for s in _rank(setups, top=2, max_per_currency=5)] == ["GBPJPY", "AUDCAD"]


@pytest.mark.unit
def test_broken_swing_lows_are_not_support():
    lows = [5, 4, 3, 1, 3, 4, 5, 4, 3, 2.5, 3, 4, 5, 6]
    frame = pd.DataFrame({"low": lows, "high": [x + 1 for x in lows],
                          "close": [x + 0.5 for x in lows]})
    assert ind.swing_lows(frame, wing=3) == [1, 2.5]
    frame.loc[13, "close"] = 2.0          # a later close below 2.5 breaks it
    assert ind.swing_lows(frame, wing=3) == [1]


@pytest.mark.unit
def test_touches_counts_separate_visits_not_bars():
    frame = pd.DataFrame({"low": [1.0, 1.0, 1.0, 2.0, 1.0, 2.0],
                          "high": [1.5, 1.5, 1.5, 2.5, 1.5, 2.5]})
    assert ind.touches(frame, level=1.0, tolerance=0.1) == 2


@pytest.mark.unit
def test_report_is_saved_as_markdown_and_json(tmp_path):
    candles, quote = _market(direction=1)
    result = scan(candles, quote, ["EURUSD", "USDJPY"], now=NOW)
    md_path, json_path = save(result, tmp_path)

    text = md_path.read_text(encoding="utf-8")
    assert "BUY LIMIT" in text and "not win probabilities" in text
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["setups"][0]["order_type"] == "BUY LIMIT"
    assert md_path.parent.name == "fx_scans"


@pytest.mark.unit
def test_an_empty_scan_says_so():
    candles, quote = _market(direction=0)
    text = to_markdown(scan(candles, quote, ["EURUSD"], now=NOW))
    assert "No setups met the rules" in text


@pytest.mark.unit
def test_stop_sits_a_full_atr_beyond_the_level_by_default():
    candles, quote = _market(direction=1)
    s = scan(candles, quote, ["EURUSD"], now=NOW).setups[0]
    # entry is 0.1 ATR in front of the level, the stop 1 ATR behind it
    assert s.risk_pips == pytest.approx(1.1 * s.atr_pips, rel=0.05)


@pytest.mark.unit
def test_a_tighter_stop_only_flatters_the_rr():
    candles, quote = _market(direction=1)
    wide = scan(candles, quote, ["EURUSD"], now=NOW).setups[0]
    tight = scan(candles, quote, ["EURUSD"], stop_atr=0.5, now=NOW).setups[0]
    assert tight.risk_pips < wide.risk_pips
    assert tight.rr > wide.rr


@pytest.mark.unit
def test_the_currency_cap_is_adjustable():
    setups = [_setup("EURUSD", "long", 90), _setup("GBPUSD", "long", 80),
              _setup("AUDUSD", "long", 70)]
    assert len(_rank(setups, top=10, max_per_currency=2)) == 2
    assert len(_rank(setups, top=10, max_per_currency=3)) == 3


@pytest.mark.unit
def test_an_ema_entry_does_not_count_as_confluence_with_itself():
    from tradingagents.fx.scanner import _Context, _setups

    # A steady climb has no swing lows, so the 1h 50 EMA is the only level.
    n, step = 300, 0.0001
    closes = [1.10 + step * i for i in range(n)]
    idx = pd.date_range(end=NOW, periods=n, freq="1h", tz="UTC")
    h1 = pd.DataFrame({"open": closes, "close": closes,
                       "high": [c + 6 * step for c in closes],
                       "low": [c - 6 * step for c in closes]}, index=idx)
    mid = closes[-1]
    context = _Context(spec=spec_for("EURUSD"), h1=h1, bias="long", trend_strength=3.0,
                       atr1=float(ind.atr(h1).iloc[-1]),
                       ema50_h1=float(ind.ema(h1["close"], 50).iloc[-1]),
                       rsi_h1=55.0, mid=mid, spread=0.00005)

    setups = _setups(context, 2.0, NOW + timedelta(hours=8))

    assert setups, "the EMA pullback should still qualify as a setup"
    assert all("coincide" not in " ".join(s.reasons) for s in setups)
