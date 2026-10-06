"""The forex agent review, run with scripted fake models (no key, no network)."""

import json
import os
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors import forex_calendar
from tradingagents.fx import agents as fx_agents
from tradingagents.fx.context import ReviewContext, gather
from tradingagents.fx.report import save
from tradingagents.fx.scanner import ScanResult, Setup
from tradingagents.fx.schemas import (
    FinalBook,
    FinalOrder,
    ResearchVerdict,
    SetupVerdict,
    TraderOrder,
    TraderPlan,
)
from tradingagents.fx.verify import check_levels, verify

NOW = datetime(2026, 10, 6, 11, 0, tzinfo=UTC)


def _setup(symbol="EURUSD", direction="long", score=70.0, **levels):
    long = direction == "long"
    base = {
        "EURUSD": 1.1720, "GBPUSD": 1.3400, "USDJPY": 150.20, "AUDUSD": 0.6975,
    }[symbol]
    pip = 0.01 if symbol.endswith("JPY") else 0.0001
    sign = 1 if long else -1
    defaults = {
        "entry": round(base - sign * 20 * pip, 5),
        "stop": round(base - sign * 37 * pip, 5),
        "target": round(base + sign * 25 * pip, 5),
        "price": base,
    }
    defaults.update(levels)
    return Setup(symbol=symbol, direction=direction, rr=2.5, score=score,
                 risk_pips=17.0, reward_pips=45.0, spread_pips=1.0, atr_pips=15.0,
                 target_kind="structure", expires_at=NOW + timedelta(hours=8),
                 reasons=["4h uptrend" if long else "4h downtrend"], **defaults)


def _ctx(candidates, **kw):
    ctx = ReviewContext(now=NOW, **kw)
    for s in candidates:
        ctx.upcoming.setdefault(s.symbol, [])
    return ctx


class FakeLLM:
    """Answers free-text prompts with a label and structured prompts from a script."""

    def __init__(self, name, structured=None):
        self.name = name
        self.structured = structured or {}
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=f"{self.name} says: noted.")

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                answer = outer.structured.get(schema)
                if isinstance(answer, Exception):
                    raise answer
                return answer(prompt) if callable(answer) else answer

        return _Bound()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_scanner_levels_pass_their_own_check():
    s = _setup()
    assert check_levels(s, s.entry, s.stop, s.target, 2.0) is None


@pytest.mark.unit
@pytest.mark.parametrize("change, problem", [
    ({"entry": 1.1730}, "below the price"),          # buy limit above market
    ({"stop": 1.1710}, "stop must be below"),       # stop above entry
    ({"stop": 1.16955}, "tighter than"),            # 4.5-pip stop on a 15-pip ATR
    ({"target": 1.1730}, "RR after the spread"),    # 1.6R: 2R needs more room
    ({"entry": 1.1660, "stop": 1.1640}, "moved"),   # 40 pips from the level
])
def test_bad_levels_are_named(change, problem):
    s = _setup()
    levels = {"entry": s.entry, "stop": s.stop, "target": s.target, **change}
    assert problem in check_levels(s, levels["entry"], levels["stop"], levels["target"], 2.0)


@pytest.mark.unit
def test_a_broken_agent_order_reverts_to_the_scanner_levels():
    s = _setup()
    accepted, dropped = verify([{"symbol": "EUR/USD", "entry": 1.1735, "stop": 1.1700,
                                 "target": 1.1800, "valid_hours": 6, "conviction": "high",
                                 "rationale": "r", "watch_for": "w"}], [s], now=NOW)
    assert dropped == []
    order = accepted[0]
    assert (order.entry, order.stop, order.target) == (s.entry, s.stop, s.target)
    assert "scanner levels used" in order.notes[0]
    assert order.expires_at == NOW + timedelta(hours=6)


@pytest.mark.unit
def test_verification_drops_what_it_cannot_repair():
    candidates = [_setup("EURUSD"), _setup("GBPUSD"), _setup("AUDUSD")]
    proposals = [
        {"symbol": "NZDUSD", "entry": 0.6, "stop": 0.59, "target": 0.62},   # invented
        {"symbol": "EURUSD", "entry": candidates[0].entry, "stop": candidates[0].stop,
         "target": candidates[0].target},
        {"symbol": "EURUSD", "entry": candidates[0].entry, "stop": candidates[0].stop,
         "target": candidates[0].target},                                     # twice
        {"symbol": "GBPUSD", "entry": candidates[1].entry, "stop": candidates[1].stop,
         "target": candidates[1].target},
        {"symbol": "AUDUSD", "entry": candidates[2].entry, "stop": candidates[2].stop,
         "target": candidates[2].target},                                     # third short USD
    ]
    accepted, dropped = verify(proposals, candidates, now=NOW, max_per_currency=2)
    assert [o.symbol for o in accepted] == ["EURUSD", "GBPUSD"]
    reasons = dict(dropped)
    assert "not one of the scanner's candidates" in reasons["NZDUSD"]
    assert "listed twice" in reasons["EURUSD"]
    assert "short USD" in reasons["AUDUSD"]


@pytest.mark.unit
def test_verification_caps_the_book_and_the_validity():
    candidates = [_setup("EURUSD"), _setup("USDJPY", "short")]
    proposals = [{"symbol": s.symbol, "entry": s.entry, "stop": s.stop, "target": s.target,
                  "valid_hours": 48, "conviction": "certain"} for s in candidates]
    accepted, dropped = verify(proposals, candidates, now=NOW, max_orders=1, max_per_currency=5)
    assert len(accepted) == 1 and "limit" in dropped[0][1]
    assert accepted[0].expires_at == NOW + timedelta(hours=12)
    assert accepted[0].conviction == "low"


@pytest.mark.unit
def test_an_order_whose_scanner_levels_also_fail_is_dropped():
    s = _setup(price=1.1690)          # price fell through the entry since the scan
    accepted, dropped = verify([{"symbol": "EURUSD", "entry": 1.2, "stop": 1.1, "target": 1.3}],
                               [s], now=NOW)
    assert accepted == []
    assert "scanner's levels fail too" in dropped[0][1]


# ---------------------------------------------------------------------------
# The review
# ---------------------------------------------------------------------------

def _team(candidates, *, keep=None, trader_orders=None, final=None):
    keep = set(keep) if keep is not None else {s.symbol for s in candidates}
    verdict = ResearchVerdict(summary="Dollar strength dominates.", decisions=[
        SetupVerdict(symbol=s.symbol, keep=s.symbol in keep, reason=f"{s.symbol} reason")
        for s in candidates])
    plan = TraderPlan(orders=trader_orders if trader_orders is not None else [
        TraderOrder(symbol=s.symbol, entry=s.entry, stop=s.stop, target=s.target,
                    note="scanner levels kept") for s in candidates if s.symbol in keep])
    book = final if final is not None else FinalBook(summary="Two clean trades.", orders=[
        FinalOrder(symbol=o.symbol, entry=o.entry, stop=o.stop, target=o.target,
                   valid_hours=5, conviction="medium", rationale=f"{o.symbol} fits",
                   watch_for="US data") for o in plan.orders])
    quick = FakeLLM("quick", {TraderPlan: plan})
    deep = FakeLLM("deep", {ResearchVerdict: verdict, FinalBook: book})
    return quick, deep


@pytest.mark.unit
def test_full_review_runs_every_role_once_and_verifies_the_book():
    candidates = [_setup("EURUSD", score=80), _setup("USDJPY", "short", score=70),
                  _setup("GBPUSD", score=60)]
    eur = candidates[0]
    bad = TraderOrder(symbol="EURUSD", entry=eur.price + 0.001, stop=eur.stop,
                      target=eur.target, note="moved up")                  # wrong side
    jpy = candidates[1]
    good = TraderOrder(symbol="USDJPY", entry=jpy.entry, stop=jpy.stop, target=jpy.target,
                       note="kept")
    quick, deep = _team(candidates, keep=["EURUSD", "USDJPY"], trader_orders=[bad, good])

    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)

    assert len(quick.prompts) == 8        # macro, price action, bull, bear, trader, three risk
    assert len(deep.prompts) == 2         # research manager, portfolio manager
    assert [o.symbol for o in result.orders] == ["EURUSD", "USDJPY"]
    assert result.orders[0].entry == eur.entry            # reverted by verification
    assert "scanner levels used" in result.orders[0].notes[0]
    assert dict(result.dropped)["GBPUSD"].startswith("research manager")
    assert set(result.transcript) >= {"Macro analyst", "Price-action analyst",
                                      "Bull researcher", "Bear researcher",
                                      "Research manager", "Trader", "Portfolio manager",
                                      "Aggressive risk analyst", "Neutral risk analyst",
                                      "Conservative risk analyst"}
    assert result.fallbacks == []


@pytest.mark.unit
def test_dropping_everything_ends_the_review_early():
    candidates = [_setup("EURUSD")]
    quick, deep = _team(candidates, keep=[])
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    assert result.orders == []
    assert len(deep.prompts) == 1 and len(quick.prompts) == 4


@pytest.mark.unit
def test_a_failed_portfolio_manager_falls_back_to_the_scanner_ranking():
    candidates = [_setup("EURUSD", score=50), _setup("USDJPY", "short", score=90)]
    quick, deep = _team(candidates)
    deep.structured[FinalBook] = RuntimeError("bad JSON")
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    assert [o.symbol for o in result.orders] == ["USDJPY", "EURUSD"]
    assert all(o.conviction == "low" for o in result.orders)
    assert any("Portfolio manager" in f for f in result.fallbacks)


@pytest.mark.unit
def test_a_failed_research_manager_keeps_every_candidate():
    candidates = [_setup("EURUSD"), _setup("USDJPY", "short")]
    quick, deep = _team(candidates)
    deep.structured[ResearchVerdict] = RuntimeError("timeout")
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    assert len(result.orders) == 2
    assert any("Research manager" in f for f in result.fallbacks)


@pytest.mark.unit
def test_the_agents_see_upcoming_releases_and_an_unknown_calendar():
    s = _setup()
    event = forex_calendar.Event("Non-Farm Payrolls", "USD", NOW + timedelta(hours=1, minutes=30), "High",
                                 forecast="150K")
    text = fx_agents.evidence([s], ReviewContext(now=NOW, upcoming={"EURUSD": [event]}))
    assert "Non-Farm Payrolls" in text and "forecast 150K" in text

    unknown = ReviewContext(now=NOW, calendar_note="calendar could not be loaded")
    assert "unknown" in fx_agents.evidence([s], unknown).lower()
    assert "could not be loaded" in fx_agents.macro_evidence([s], unknown)


# ---------------------------------------------------------------------------
# Calendar and context
# ---------------------------------------------------------------------------

RAW = [
    {"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-10-06T08:30:00-04:00",
     "impact": "High", "forecast": "150K", "previous": "142K"},
    {"title": "German Factory Orders", "country": "EUR", "date": "2026-10-06T02:00:00-04:00",
     "impact": "Low", "forecast": "", "previous": ""},
    {"title": "OPEC Meeting", "country": "All", "date": "2026-10-06T09:00:00-04:00",
     "impact": "Medium"},
    {"title": "BoJ Minutes", "country": "JPY", "date": "not a date", "impact": "High"},
]


@pytest.mark.unit
def test_calendar_times_are_utc_and_filtered():
    events = forex_calendar.events_between(NOW, NOW + timedelta(hours=8), {"EUR", "USD"}, raw=RAW)
    assert [e.title for e in events] == ["Non-Farm Employment Change", "OPEC Meeting"]
    assert events[0].time == datetime(2026, 10, 6, 12, 30, tzinfo=UTC)


@pytest.mark.unit
def test_metals_take_us_releases():
    assert forex_calendar.currencies_for("XAUUSD") == {"USD"}
    assert forex_calendar.currencies_for("EURJPY") == {"EUR", "JPY"}


@pytest.mark.unit
def test_calendar_uses_a_stale_copy_when_the_feed_is_down(monkeypatch, tmp_path):
    monkeypatch.setattr(forex_calendar, "_cache_path", lambda: tmp_path / "week.json")
    (tmp_path / "week.json").write_text(json.dumps(RAW), encoding="utf-8")
    old = time.time() - 2 * forex_calendar.CACHE_SECONDS
    os.utime(tmp_path / "week.json", (old, old))

    def down():
        raise VendorUnavailableError("HTTP 429")
    monkeypatch.setattr(forex_calendar, "_download", down)

    assert len(forex_calendar.events_between(NOW, NOW + timedelta(hours=8))) == 2


@pytest.mark.unit
def test_calendar_with_no_feed_and_no_copy_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(forex_calendar, "_cache_path", lambda: tmp_path / "missing.json")

    def down():
        raise VendorUnavailableError("unreachable")
    monkeypatch.setattr(forex_calendar, "_download", down)

    with pytest.raises(VendorUnavailableError):
        forex_calendar.events_between(NOW, NOW + timedelta(hours=8))


@pytest.mark.unit
def test_gather_states_missing_sources_instead_of_hiding_them():
    def no_calendar(*_a, **_k):
        raise VendorUnavailableError("feed down")

    def no_news(*_a):
        raise VendorUnavailableError("Yahoo down")

    ctx = gather([_setup()], NOW, calendar=no_calendar, news=no_news, global_news=no_news)
    assert "UNKNOWN, not absent" in ctx.calendar_note
    assert "news unavailable" in ctx.news["EURUSD"]
    assert "macro headlines unavailable" in ctx.global_news


@pytest.mark.unit
def test_agent_report_leads_with_the_final_orders(tmp_path):
    candidates = [_setup("EURUSD"), _setup("USDJPY", "short")]
    quick, deep = _team(candidates)
    review = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    scan = ScanResult(scanned_at=NOW, setups=candidates)

    md, js = save(scan, tmp_path, review)

    text = md.read_text(encoding="utf-8")
    assert md.name.endswith("_agents.md")
    assert text.index("## Final orders") < text.index("## How the agents got there") \
        < text.index("## Stage 1: scanner candidates")
    assert "no track record" in text
    data = json.loads(js.read_text(encoding="utf-8"))
    assert [o["symbol"] for o in data["review"]["orders"]] == ["EURUSD", "USDJPY"]


QUOTA = RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota "
                     "{'error': {'code': 429, ... 'quotaValue': '20'}}")


@pytest.mark.unit
def test_the_fast_model_decides_when_the_strong_one_is_out_of_quota():
    candidates = [_setup("EURUSD"), _setup("USDJPY", "short")]
    quick, deep = _team(candidates)
    quick.structured[ResearchVerdict] = deep.structured[ResearchVerdict]
    quick.structured[FinalBook] = deep.structured[FinalBook]
    deep.structured[ResearchVerdict] = QUOTA
    deep.structured[FinalBook] = QUOTA

    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)

    assert [o.symbol for o in result.orders] == ["EURUSD", "USDJPY"]
    assert all(o.conviction == "medium" for o in result.orders)       # a real decision
    assert result.problems == ["Research manager: daily quota used up",
                               "Portfolio manager: daily quota used up"]
    assert all("decided by the fast model (strong model: daily quota used up)" in f
               for f in result.fallbacks)
    assert not any("no usable answer" in f for f in result.fallbacks)


@pytest.mark.unit
def test_a_failed_text_agent_does_not_stop_the_review():
    candidates = [_setup("EURUSD")]
    quick, deep = _team(candidates)
    calls = {"n": 0}
    original = quick.invoke

    def flaky(prompt):
        calls["n"] += 1
        if "MACRO ANALYST" in prompt:
            raise TimeoutError("read timed out")
        return original(prompt)
    quick.invoke = flaky

    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)

    assert len(result.orders) == 1
    assert result.problems == ["Macro analyst: the model timed out"]
    assert "unavailable" in result.transcript["Macro analyst"]


@pytest.mark.unit
@pytest.mark.parametrize("message, reason", [
    ("429 RESOURCE_EXHAUSTED quota exceeded", "daily quota used up"),
    ("Error code: 429 - rate limit reached", "rate-limited by the provider"),
    ("Error code: 401 - Incorrect API key provided", "API key refused"),
    ("Request timed out.", "the model timed out"),
    ("something odd\nwith detail", "something odd"),
])
def test_provider_errors_become_one_line(message, reason):
    assert fx_agents.short_reason(RuntimeError(message)) == reason


@pytest.mark.unit
def test_the_desk_rules_name_the_close_and_the_missing_actuals():
    text = fx_agents.GROUND_RULES
    assert "17:00 close" in text and "never the actual result" in text


@pytest.mark.unit
def test_smc_candidates_show_their_zone_and_invalidation():
    s = _setup()
    s.strategy, s.zone_low, s.zone_high, s.invalidation = "smc", 1.1695, 1.1705, 1.1688
    text = fx_agents.evidence([s], _ctx([s]))
    assert "entry zone 1.1695–1.1705" in text and "sweep extreme 1.1688" in text


@pytest.mark.unit
def test_an_smc_order_cannot_outlive_its_session():
    s = _setup(entry=1.1700, stop=1.1683, target=1.1745)
    s.strategy, s.zone_low, s.zone_high, s.invalidation = "smc", 1.1695, 1.1705, 1.1688
    s.expires_at = NOW + timedelta(hours=1)
    accepted, _ = verify([{"symbol": "EURUSD", "entry": 1.1700, "stop": 1.1683, "target": 1.1745,
                           "valid_hours": 6}], [s], now=NOW)
    assert accepted[0].expires_at == NOW + timedelta(hours=1)
    assert "scanner's limit" in accepted[0].notes[0]


@pytest.mark.unit
def test_a_target_already_behind_price_is_refused():
    s = _setup()                                   # price 1.1720, a buy below it
    assert "already reached" in check_levels(s, s.entry, s.stop, 1.1718, 2.0)
    short = _setup("USDJPY", "short")
    assert "already reached" in check_levels(short, short.entry, short.stop, 150.25, 2.0)
