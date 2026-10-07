"""Quinn's experiments: ideas tried on a design year, then a locked year, with every try on record."""

from datetime import UTC, datetime, timedelta

import pytest

from tests.test_fx_lab import MON, _feed, _finished, _price_path, _setup
from tradingagents.fx import lab, quinn
from tradingagents.fx.quinn import Experiment, Ledger


def _feat(s, **f):
    s.features = {"pool": "Asian low", "pool_quality": 2, "shift": "CHoCH", "h1_break": "BOS",
                  "zone": "fair value gap", "displacement": 2.5, "depth": 0.5, "wide_spread": False, **f}
    return s


@pytest.mark.unit
def test_an_experiment_admits_only_setups_with_its_rule():
    s = _feat(_setup(MON))
    at = lambda h: datetime(2026, 9, 7, h + 4, 0, tzinfo=UTC)          # noqa: E731  New York hour h
    assert Experiment("x", "", skip_hours=(2, 3)).admits(s, at(5))
    assert not Experiment("x", "", skip_hours=(2, 3)).admits(s, at(2))
    assert not Experiment("x", "", last_hour=10).admits(s, at(11))
    assert not Experiment("x", "", min_pool_quality=3).admits(s, at(5))
    assert Experiment("x", "", min_pool_quality=2, h1_break="BOS", min_displacement=2).admits(s, at(5))
    assert not Experiment("x", "", zones=("order block",)).admits(s, at(5))
    assert not Experiment("x", "", symbols=("GBPUSD",)).admits(s, at(5))
    assert not Experiment("x", "", skip_wide_spread=True).admits(_feat(_setup(MON), wide_spread=True), at(5))


@pytest.mark.unit
def test_a_target_cap_brings_far_targets_in_and_leaves_near_ones():
    far = _setup(MON, entry=1.1690, stop=1.1680, target=1.1740)            # 5R away
    near = _setup(MON, entry=1.1690, stop=1.1680, target=1.1712)           # 2.2R away
    exp = Experiment("x", "", target_r=2.5)
    moved = exp.adjust(far)
    assert moved.target == pytest.approx(1.1715) and moved.target_kind == "2.5R"
    assert moved.rr == pytest.approx((0.0025 - 0.0001) / (0.0010 + 0.0001), abs=0.01)
    assert exp.adjust(near) is near
    short = _setup(MON, entry=1.1690, stop=1.1700, target=1.1640, direction="short")
    assert exp.adjust(short).target == pytest.approx(1.1665)


@pytest.mark.unit
def test_measure_counts_quarters_and_a_t_score():
    r = quinn.measure(_finished([2.5, -1.0, -1.0] * 40, start=MON - timedelta(days=200)))
    assert r.trades == 120 and r.expectancy_r == pytest.approx(0.167, abs=0.01)
    assert r.t_score == pytest.approx(0.167 / (1.65 / 120 ** 0.5), abs=0.1)
    assert r.quarters_rated >= 1


@pytest.mark.unit
def test_the_design_year_needs_a_clear_profitable_gain():
    base = quinn.measure(_finished([2.0, -1.0, -1.0, -1.0] * 100))                # -0.25R
    worse = quinn.measure(_finished([2.0, -1.0, -1.0, -1.0] * 100))
    better_but_losing = quinn.measure(_finished([2.0, -1.0, -1.0] * 120))          # 0R
    good = quinn.measure(_finished([2.5, -1.0, -1.0] * 120))
    few = quinn.measure(_finished([2.5, -1.0] * 50))
    assert not quinn.design_verdict(worse, base)[0]
    assert "still losing" in quinn.design_verdict(better_but_losing, base)[1]
    assert quinn.design_verdict(good, base)[0]
    assert "too few" in quinn.design_verdict(few, base)[1]


@pytest.mark.unit
def test_the_locked_year_bar_rises_with_every_ten_ideas_tried():
    assert quinn.locked_bar(0) == 2.0 and quinn.locked_bar(9) == 2.0 and quinn.locked_bar(10) == 2.1
    steady = quinn.measure(_finished([2.5, -1.0] * 150, start=MON - timedelta(days=330)))
    assert quinn.locked_verdict(steady, 2.0)[0]
    assert "below the bar" in quinn.locked_verdict(steady, 99.0)[1]
    luck = quinn.measure(_finished([2.0, -1.0, -1.0, -1.0] * 60))
    assert "luck" in quinn.locked_verdict(luck, 2.0)[1]


@pytest.mark.unit
def test_an_idea_only_sees_the_locked_year_after_passing_the_design_year(tmp_path):
    base = quinn.measure(_finished([2.0, -1.0, -1.0, -1.0] * 100))
    good = _finished([2.5, -1.0] * 200, start=MON - timedelta(days=330))
    bad = _finished([2.0, -1.0, -1.0, -1.0] * 100)
    locked_calls = []
    ledger = Ledger(tmp_path / "ledger.json")

    def locked(e):
        locked_calls.append(e.name)
        return good

    rejected = quinn.run(Experiment("weak", "why"), lambda e: bad, locked, base, ledger)
    assert rejected["status"] == "rejected on the design year" and locked_calls == []
    passed = quinn.run(Experiment("strong", "why"), lambda e: good, locked, base, ledger)
    assert passed["status"].startswith("candidate") and locked_calls == ["strong"]
    again = Ledger(tmp_path / "ledger.json")
    assert [r["name"] for r in again.rows] == ["weak", "strong"] and again.tried_on_locked == 1
    text = quinn.report([rejected, passed], base, None, again)
    assert "strong" in text and "paper-traded next" in text


@pytest.mark.unit
def test_trading_a_scan_log_applies_the_experiment_before_orders_exist():
    start = MON - timedelta(days=2)
    feed = _feed(_price_path(start, 1200, []))
    a = _feat(_setup(MON), pool_quality=0)
    b = _feat(_setup(MON, symbol="EURUSD", entry=1.1650, stop=1.1640, direction="long"), pool_quality=3)
    log = [(MON, [a]), (MON + timedelta(minutes=10), [b])]
    everything = lab.trade(log, feed, MON + timedelta(hours=10))
    assert len(everything) == 1                                   # b waits: EURUSD long is on the book
    picky = lab.trade(log, feed, MON + timedelta(hours=10), experiment=Experiment("x", "", min_pool_quality=2))
    assert len(picky) == 1 and picky[0].entry == 1.1650 and picky[0].features["pool_quality"] == 3


@pytest.mark.unit
def test_a_scan_log_is_cached_and_its_name_tracks_the_rules(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(lab, "scan_log", lambda *a, **k: calls.append(1) or [(MON, [])])
    feed = _feed(_price_path(MON - timedelta(days=2), 600, []))
    for _ in range(2):
        lab.cached_scan_log(tmp_path, feed, ["EURUSD"], MON, MON + timedelta(days=1), lab.Rules())
    assert calls == [1]
    k1 = lab.log_key(["EURUSD"], MON, MON + timedelta(days=1), lab.Rules())
    k2 = lab.log_key(["EURUSD"], MON, MON + timedelta(days=1), lab.Rules(min_rr=2.5))
    assert k1 != k2
