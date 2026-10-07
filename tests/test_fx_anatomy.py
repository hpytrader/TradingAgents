"""Trade anatomy: each trade's path read in R, with what-ifs that take the worse case."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.fx import anatomy, journal as jr
from tradingagents.fx.journal import Entry

T = datetime(2026, 9, 8, 7, 0, tzinfo=UTC)               # Tuesday 03:00 New York


def _path(points, start=T, flat=1.1700):
    """Five-minute bars from (high, low) pairs, then flat to the close."""
    rows = [(hi, lo) for hi, lo in points] + [(flat + 0.0001, flat - 0.0001)] * 200
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame({"open": [flat] * len(rows), "high": [h for h, _ in rows], "low": [lo for _, lo in rows],
                         "close": [(h + lo) / 2 for h, lo in rows], "volume": 1}, index=idx)


def _trade(bars, start=T, **kw):
    e = Entry(id="x", created_at=start, symbol="EURUSD", direction=kw.get("direction", "long"), entry=1.1700,
              stop=kw.get("stop", 1.1680), target=kw.get("target", 1.1740), planned_rr=2.0, spread=0.0002,
              expires_at=start + timedelta(hours=4), ticket="B1", initial_stop=kw.get("stop", 1.1680))
    jr.simulate(e, bars, start + timedelta(hours=20))
    return e


@pytest.mark.unit
def test_a_winner_that_dipped_first():
    bars = _path([(1.1702, 1.1699), (1.1701, 1.1690), (1.1741, 1.1700)])
    e = _trade(bars)
    assert e.status == jr.WON
    p = anatomy.measure(e, bars)
    assert p.gross_r == pytest.approx(2.0) and p.mae == pytest.approx(0.5) and p.spread_share == pytest.approx(0.1)


@pytest.mark.unit
def test_a_loser_that_was_up_then_stopped_then_ran_to_target():
    bars = _path([(1.1701, 1.1699), (1.1725, 1.1705), (1.1710, 1.1679), (1.1745, 1.1700)])
    e = _trade(bars)
    assert e.status == jr.LOST
    p = anatomy.measure(e, bars)
    assert p.mfe == pytest.approx(1.25) and p.target_after_stop and p.minutes == 10
    scratch = -0.0002 / 0.0022
    assert p.breakeven_r == pytest.approx(scratch, abs=1e-3)              # the stop at the entry saved it
    banked = 0.5 * (0.0020 - 0.0002) / 0.0022
    assert p.half_off_r == pytest.approx(banked + 0.5 * scratch, abs=1e-3)


@pytest.mark.unit
def test_what_ifs_take_the_stop_first_when_a_bar_touches_both():
    bars = _path([(1.1701, 1.1699), (1.1721, 1.1679)])                   # +1R and the stop in one bar
    e = _trade(bars)
    p = anatomy.measure(e, bars)
    assert p.breakeven_r == -1.0 and p.half_off_r == -1.0


@pytest.mark.unit
def test_shorts_mirror_longs():
    bars = _path([(1.1701, 1.1699), (1.1705, 1.1698), (1.1700, 1.1659)])
    e = _trade(bars, direction="short", stop=1.1720, target=1.1660)
    assert e.status == jr.WON
    p = anatomy.measure(e, bars)
    assert p.gross_r == pytest.approx(2.0) and p.mae == pytest.approx(0.25)


@pytest.mark.unit
def test_the_study_names_the_leak():
    paths, entries = [], []
    for i in range(30):                                                    # every loser was +1.25R first
        start = T + timedelta(days=i)
        bars = _path([(1.1701, 1.1699), (1.1725, 1.1705), (1.1710, 1.1679)], start=start)
        entries.append(_trade(bars, start=start))
        paths.append(bars)
    frames = {("EURUSD", "M5"): pd.concat(paths).sort_index()}
    a, _ = anatomy.study(entries, frames)
    assert a.trades == 30 and a.losers_mfe["1R"] == 1.0 and a.net_r == -1.0
    assert a.breakeven_r > a.net_r
    assert any(f.startswith("EXITS:") and "worth testing" in f for f in a.findings)
    assert any(f.startswith("COSTS:") and "cannot fix them" in f for f in a.findings)
    text = anatomy.to_markdown(a, "test")
    assert "What if: break-even at +1R" in text and "+1R | 100%" in text


@pytest.mark.unit
def test_no_single_leak_says_so():
    a = anatomy.Anatomy(trades=500, net_r=-0.1, gross_r=-0.02, spread_share=0.1, win_rate=0.3,
                        winners_mae={"0.5R": 0.2}, losers_mfe={"1R": 0.1}, stopped_within_15=0.1,
                        stopped_within_60=0.4, target_after_stop=0.05, closed_at_17=0.05, breakeven_r=-0.11,
                        half_off_r=-0.12, same_bar_losses=0.0)
    assert anatomy.findings(a)[-1].startswith("VERDICT: no single leak")


@pytest.mark.unit
def test_a_thin_edge_eaten_by_the_spread_is_called_out():
    a = anatomy.Anatomy(trades=2207, net_r=-0.10, gross_r=0.03, spread_share=0.14, win_rate=0.31,
                        winners_mae={"0.5R": 0.39}, losers_mfe={"1R": 0.32}, stopped_within_15=0.20,
                        stopped_within_60=0.52, target_after_stop=0.22, closed_at_17=0.14, breakeven_r=-0.08,
                        half_off_r=-0.07, same_bar_losses=0.0)
    costs = anatomy.findings(a)[0]
    assert costs.startswith("COSTS: before the spread the rules make +0.03R") and "fix to try" in costs
