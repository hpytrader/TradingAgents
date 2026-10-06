"""The desk team, the recorded conversation, and how it is stored and shown."""

import json
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_fx_agents import NOW, _ctx, _setup, _team  # noqa: E402

from tradingagents.fx import (
    agents as fx_agents,  # noqa: E402
    dashboard,  # noqa: E402
)
from tradingagents.fx.journal import Journal, stats  # noqa: E402
from tradingagents.fx.team import DEFAULTS, team  # noqa: E402

ORDER = ["desk", "macro", "price_action", "bull", "bear", "research_manager", "trader",
         "risk_aggressive", "risk_conservative", "risk_neutral", "portfolio_manager", "desk"]


@pytest.mark.unit
def test_every_agent_has_a_distinct_name_and_a_bio():
    names = [p.name for p in DEFAULTS]
    assert len(names) == len(set(names)) == 12
    assert all(p.bio and p.expertise and p.role for p in DEFAULTS)


@pytest.mark.unit
def test_names_can_be_changed_in_a_json_file(tmp_path):
    file = tmp_path / "fx_team.json"
    file.write_text(json.dumps({"macro": {"name": "Jarvis", "expertise": ["Rates"]},
                                "nobody": {"name": "x"}, "bull": {"tier": "deep"}}), encoding="utf-8")
    people = team(file)
    assert people["macro"].name == "Jarvis" and people["macro"].expertise == ("Rates",)
    assert people["bull"].tier == "quick"                  # only name, role, bio, expertise change
    file.write_text("{not json", encoding="utf-8")
    assert team(file)["macro"].name == "Atlas"


@pytest.mark.unit
def test_the_conversation_follows_the_decision_in_order():
    candidates = [_setup("EURUSD"), _setup("USDJPY", "short")]
    quick, deep = _team(candidates)
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)

    assert [m["agent"] for m in result.messages] == ORDER
    assert result.messages[0]["kind"] == "system" and "EURUSD BUY LIMIT" in result.messages[0]["text"]
    assert result.messages[-1]["title"] == "Verification" and "✓ #1 EURUSD" in result.messages[-1]["text"]
    assert {m["name"] for m in result.messages} >= {"Atlas", "Vega", "Leo", "Ursa", "Sage", "Donna"}
    assert result.to_dict()["messages"] == result.messages


@pytest.mark.unit
def test_agents_answer_each_other_by_name():
    candidates = [_setup("EURUSD")]
    quick, deep = _team(candidates)
    fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    prompts = quick.prompts
    bear = next(p for p in prompts if "BEAR RESEARCHER" in p)
    assert "You are Ursa" in bear and "Leo's case" in bear and "quick says" in bear   # sees the bull's text
    conservative = next(p for p in prompts if "CONSERVATIVE RISK" in p)
    assert "Blaze (aggressive risk analyst) said" in conservative
    neutral = next(p for p in prompts if "NEUTRAL RISK" in p)
    assert "Blaze" in neutral and "Haven (conservative risk analyst) said" in neutral
    assert any("You are Atlas, the MACRO ANALYST" in p for p in prompts)


@pytest.mark.unit
def test_a_rejected_round_is_still_a_conversation():
    candidates = [_setup("EURUSD")]
    quick, deep = _team(candidates, keep=[])
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    assert result.orders == []
    assert [m["agent"] for m in result.messages][-2:] == ["research_manager", "desk"]
    assert "sits out" in result.messages[-1]["text"]


@pytest.mark.unit
def test_conversations_are_stored_and_orders_link_to_them(tmp_path):
    candidates = [_setup("EURUSD")]
    quick, deep = _team(candidates)
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    book = Journal(tmp_path / "j.db")
    rid = book.record_review(result, now=NOW)
    book.record(result.orders, now=NOW, review_id=rid)
    saved = book.reviews()
    assert saved[0]["id"] == rid and saved[0]["orders"] == 1
    assert [m["agent"] for m in saved[0]["messages"]] == ORDER
    assert book.entries()[0].review_id == rid


@pytest.mark.unit
def test_an_older_journal_gains_the_link_column(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE orders (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, symbol TEXT NOT NULL, "
               "direction TEXT NOT NULL, entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL, "
               "planned_rr REAL, spread REAL DEFAULT 0, expires_at TEXT NOT NULL, conviction TEXT, rationale TEXT, "
               "watch_for TEXT, strategy TEXT, scanner_score REAL, report TEXT, status TEXT NOT NULL DEFAULT "
               "'pending', filled_at TEXT, exit_at TEXT, exit_price REAL, result_r REAL, note TEXT, notified TEXT)")
    db.execute("INSERT INTO orders (id, created_at, symbol, direction, entry, stop, target, expires_at) "
               "VALUES ('a', ?, 'EURUSD', 'long', 1.17, 1.168, 1.175, ?)",
               (NOW.isoformat(), (NOW + timedelta(hours=4)).isoformat()))
    db.commit()
    db.close()
    book = Journal(path)
    assert book.entries()[0].review_id == ""
    assert book.reviews() == []


@pytest.mark.unit
def test_dashboard_embeds_the_chat_safely_and_ends_with_the_team(tmp_path):
    candidates = [_setup("EURUSD")]
    quick, deep = _team(candidates)
    result = fx_agents.review(candidates, _ctx(candidates), quick, deep, now=NOW)
    result.messages[1]["text"] = "headline </script><script>alert(1)</script>"
    book = Journal(tmp_path / "j.db")
    rid = book.record_review(result, now=NOW)
    book.record(result.orders, now=NOW, review_id=rid)
    entries = book.entries()
    page = dashboard.render(entries, stats(entries), now=NOW, reviews=book.reviews())

    payload = page.split('id="chat-data">', 1)[1].split("</script>", 1)[0]
    assert "</script" not in payload and "<\\/script>" in payload       # cannot close the tag
    assert json.loads(payload)["reviews"][0]["id"] == rid
    assert f'data-review="{rid}"' in page and "desk chat ▸" in page
    assert page.index('id="chat"') < page.index("The desk") < page.index('class="foot"')
    for name in ("Atlas", "Vega", "Leo", "Ursa", "Sage", "Nova", "Blaze", "Haven", "Pivot", "Ward", "Donna"):
        assert f"<h3>{name}</h3>" in page


@pytest.mark.unit
def test_an_empty_chat_explains_itself():
    page = dashboard.render([], stats([]), now=NOW)
    assert "No conversations yet" in page and "<h3>Donna</h3>" in page


@pytest.mark.unit
def test_the_watcher_saves_each_review_and_links_its_orders(tmp_path):
    from types import SimpleNamespace

    from test_fx_watch import NOW as WATCH_NOW, Harness, _order, _setup as watch_setup

    h = Harness(tmp_path, [watch_setup()])
    chat = [{"agent": "desk", "name": "Desk", "role": "Desk system", "title": "Scan", "kind": "system", "text": "x"}]
    h.review = lambda result, now: (SimpleNamespace(orders=[_order(s) for s in result.setups],
                                                    messages=chat, summary="s"), "r.md")
    h.run(WATCH_NOW)
    saved = h.book.reviews()
    assert len(saved) == 1 and h.book.entries()[0].review_id == saved[0]["id"]
