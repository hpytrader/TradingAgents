"""Stage 2 of the morning forex scan: the agent team reviews the candidates.

The roles follow the TradingAgents framework, adapted to a book of intraday
limit orders rather than one stock:

    Macro analyst, Price-action analyst ─► Bull researcher ┐
                                           Bear researcher ┴► Research manager
        ─► Trader ─► Aggressive / Neutral / Conservative risk analysts
        ─► Portfolio manager

Each agent sees every candidate at once, so a scan costs ten model calls in
all, whatever the number of candidates. The fast ("quick") model argues; the
strong ("deep") model makes the two judgements that decide the book. The
portfolio manager's orders are then verified in code (``fx.verify``) before
they are reported.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS
from tradingagents.fx.context import ReviewContext
from tradingagents.fx.scanner import Setup
from tradingagents.fx.schemas import FinalBook, ResearchVerdict, TraderPlan
from tradingagents.fx.verify import VerifiedOrder, verify

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]

GROUND_RULES = f"""You work on an intraday forex and metals desk. {NO_EXTERNAL_TOOLS}
Desk rules: every order is a limit order. An unfilled order is cancelled at its listed cancel \
time (no later than the end of the morning scan window, 12:00 New York by default) or \
earlier for a release. A filled trade is \
managed only by its stop and target and is closed at the New York 17:00 close (21:00 UTC in \
summer, 22:00 UTC in winter) at the latest; nothing is held overnight.
Never invent prices, news or data releases. The economic calendar lists times, forecasts and \
previous values but never the actual result; an actual figure is known only if a headline \
reports it. A rule-based scanner found the candidates; its levels and figures are facts. Your \
job is judgement the scanner cannot make: news and event risk, whether a level or zone is \
likely to hold, and how the trades interact."""


@dataclass
class Review:
    orders: list[VerifiedOrder]
    dropped: list[tuple[str, str]]          # symbol, why it is not in the book
    summary: str
    transcript: dict[str, str] = field(default_factory=dict)   # agent → what it said
    fallbacks: list[str] = field(default_factory=list)         # agents whose output was replaced
    problems: list[str] = field(default_factory=list)          # one-line reason per failed call

    def to_dict(self) -> dict:
        return {
            "orders": [o.to_dict() for o in self.orders],
            "dropped": [{"symbol": s, "reason": r} for s, r in self.dropped],
            "summary": self.summary,
            "transcript": self.transcript,
            "fallbacks": self.fallbacks,
            "problems": self.problems,
        }


# ---------------------------------------------------------------------------
# Evidence blocks
# ---------------------------------------------------------------------------

def describe_candidate(s: Setup, ctx: ReviewContext) -> str:
    lines = [
        f"### {s.symbol}: {s.order_type} @ {s.entry}",
        f"- Stop {s.stop} ({s.risk_pips} pips), target {s.target} ({s.reward_pips} pips, "
        f"{s.target_kind}), {s.rr:.2f}R after a {s.spread_pips}-pip spread",
        f"- Price at scan {s.price}; 1h ATR {s.atr_pips} pips; scanner score {s.score:.0f}/100",
        *([f"- SMC setup: entry zone {s.zone_low}–{s.zone_high}; idea invalid beyond the "
           f"sweep extreme {s.invalidation}"] if s.strategy == "smc" else []),
        f"- Order expires {s.expires_at:%H:%M} UTC unless cancelled sooner",
    ]
    lines += [f"- {reason}" for reason in s.reasons]
    if ctx.calendar_note:
        lines.append("- Calendar: unknown (see note)")
    else:
        events = ctx.upcoming.get(s.symbol, [])
        if events:
            lines.append("- Releases before expiry:")
            lines += [f"  - {e.describe()}" for e in events]
        else:
            lines.append("- Releases before expiry: no high- or medium-impact events listed")
    return "\n".join(lines)


def evidence(candidates: list[Setup], ctx: ReviewContext) -> str:
    parts = [f"Time now: {ctx.now:%A %Y-%m-%d %H:%M} UTC", "", "## Candidates", ""]
    parts += [describe_candidate(s, ctx) + "\n" for s in candidates]
    return "\n".join(parts)


def macro_evidence(candidates: list[Setup], ctx: ReviewContext) -> str:
    parts = [f"Time now: {ctx.now:%A %Y-%m-%d %H:%M} UTC", ""]
    if ctx.calendar_note:
        parts += ["## Economic calendar", ctx.calendar_note, ""]
    else:
        parts += ["## Released in the last 12 hours"]
        parts += [f"- {e.describe()}" for e in ctx.recent] or ["- nothing high or medium impact"]
        parts.append("")
        parts.append("## Coming up before the candidates expire")
        upcoming = sorted({e for evs in ctx.upcoming.values() for e in evs}, key=lambda e: e.time)
        parts += [f"- {e.describe()}" for e in upcoming] or ["- nothing high or medium impact"]
        parts.append("")
    parts += ["## Candidates", ", ".join(f"{s.symbol} {s.direction}" for s in candidates), ""]
    for s in candidates:
        if ctx.news.get(s.symbol):
            parts += [f"## Headlines: {s.symbol}", ctx.news[s.symbol], ""]
    if ctx.global_news:
        parts += ["## Global macro headlines", ctx.global_news, ""]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def _text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        ).strip()
    return str(content).strip()


def _say(llm: Any, prompt: str) -> str:
    return _text(llm.invoke(prompt))


def _heard(agent: str, call: Callable[[], str], problems: list[str]) -> str:
    """Run a free-text agent; on failure note it and let the review carry on."""
    try:
        return call()
    except Exception as exc:
        reason = short_reason(exc)
        logger.info("%s failed: %s", agent, exc)
        problems.append(f"{agent}: {reason}")
        return f"({agent} unavailable: {reason})"


def short_reason(exc: BaseException) -> str:
    """One line a trader can act on, instead of a provider's full error payload."""
    text = str(exc)
    lowered = text.lower()
    if "resource_exhausted" in lowered or "quota" in lowered:
        return "daily quota used up"
    if "429" in text or "rate limit" in lowered or "rate_limit" in lowered:
        return "rate-limited by the provider"
    if "401" in text or "api key" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        return "API key refused"
    if "timeout" in lowered or "timed out" in lowered:
        return "the model timed out"
    first = text.strip().splitlines()[0] if text.strip() else type(exc).__name__
    return first[:120]


def _structured(llm: Any, schema: type, prompt: str, agent: str,
                errors: list[str] | None) -> Any | None:
    """A typed answer from ``llm``, or ``None`` with a one-line reason in ``errors``."""
    try:
        answer = llm.with_structured_output(schema).invoke(prompt)
    except Exception as exc:
        reason = short_reason(exc)
        logger.info("%s: structured answer failed: %s", agent, exc)
    else:
        if isinstance(answer, schema):
            return answer
        reason = "the model returned no usable answer"
    if errors is not None:
        errors.append(f"{agent}: {reason}")
    return None


def macro_analyst(llm, candidates, ctx) -> str:
    return _say(llm, f"""{GROUND_RULES}

You are the MACRO ANALYST. From the calendar and headlines below, brief the desk:
1. For each currency involved (and gold/silver if present): the current driver in one or two \
sentences: central-bank stance, the latest data surprise, risk sentiment.
2. Event risk: list every release before the candidates expire, its time, the currencies it \
moves and how dangerous it is for a limit order waiting to fill.
3. Anything in the headlines that cuts against a candidate's direction.
Say plainly when evidence is missing. Under 400 words.

{macro_evidence(candidates, ctx)}""")


def price_action_analyst(llm, candidates, ctx) -> str:
    return _say(llm, f"""{GROUND_RULES}

You are the PRICE-ACTION ANALYST, reading each setup as a smart-money trader would. For \
every candidate, judge from the facts given: how significant the swept liquidity is \
(previous day and session extremes outrank equal highs/lows), whether the displacement and \
structure shift look decisive or marginal, the quality of the order block / fair value gap \
and whether the entry is in discount (longs) or premium (shorts), whether the 1-hour bias \
supports it, and whether the target liquidity is realistic for the time left in the session. \
Give each candidate a grade (A, B or C) with two or three sentences. Under 350 words.

{evidence(candidates, ctx)}""")


def researcher(llm, side: str, candidates, ctx, macro: str, structure: str = "") -> str:
    stance = {
        "bull": ("BULL RESEARCHER", "make the strongest honest case FOR taking each setup",
                 "the liquidity taken and the strength of the shift, order block / fair value "
                 "gap confluence, entry in discount or premium, the higher-timeframe bias, room "
                 "to the opposing liquidity, and news that supports the direction"),
        "bear": ("BEAR RESEARCHER", "make the strongest honest case AGAINST each setup",
                 "a sweep that could be the start of a real breakdown rather than a stop hunt, "
                 "weak displacement, a zone price may slice through, opposing liquidity too "
                 "close or too far for the session, event risk before the order would fill and "
                 "play out, and news against the direction"),
    }[side]
    return _say(llm, f"""{GROUND_RULES}

You are the {stance[0]}. For every candidate, {stance[1]}. Focus on {stance[2]}. \
Use only the evidence given. Two to four sentences per candidate, then one line ranking them.

## Macro brief
{macro}

## Price-action read
{structure or "(not available)"}

{evidence(candidates, ctx)}""")


def research_manager(llm, candidates, ctx, macro, bull, bear,
                     errors: list[str] | None = None, structure: str = "") -> ResearchVerdict | None:
    return _structured(llm, ResearchVerdict, f"""{GROUND_RULES}

You are the RESEARCH MANAGER. Judge the bull and bear cases for each candidate and decide \
which go to the trader. Drop a setup when the bear case is stronger, above all when a \
high-impact release for either currency is due before the order would likely fill and play \
out. Keeping none is acceptable on a bad day. Return a verdict for every candidate.

## Macro brief
{macro}

## Price-action read
{structure or "(not available)"}

## Bull case
{bull}

## Bear case
{bear}

{evidence(candidates, ctx)}""", "Research manager", errors)


def trader(llm, kept, ctx, verdict_text: str, errors: list[str] | None = None) -> TraderPlan | None:
    return _structured(llm, TraderPlan, f"""{GROUND_RULES}

You are the TRADER. For each setup the research manager kept, write the order. Keep the \
scanner's entry, stop and target unless the debate gives a concrete reason to change them, \
and say what you changed. Rules the desk enforces in code: a buy limit stays below the price \
at scan time and a sell limit above it; the reward-to-risk after the spread stays at or above \
the minimum; for an SMC setup the entry stays inside its order block / fair value gap and the \
stop stays beyond the sweep extreme; for any other setup the entry moves at most 1.5 ATR from \
the scanner's level and the stop stays at least 0.5 ATR from the entry. An order breaking a \
rule reverts to the scanner's levels.

## Research manager's verdict
{verdict_text}

{evidence(kept, ctx)}""", "Trader", errors)


def risk_analyst(llm, stance: str, plan_text: str, macro: str, ctx) -> str:
    view = {
        "aggressive": "Argue for taking the strongest setups at full size and for which "
                      "lower-conviction ones still deserve a place.",
        "neutral": "Weigh the trades' combined exposure: shared currencies, correlated "
                   "pairs, and how many orders could fill into the same release.",
        "conservative": "Argue for cutting anything exposed to a release before expiry, "
                        "anything stretched, and any stacked bet on one currency.",
    }[stance]
    return _say(llm, f"""{GROUND_RULES}

You are the {stance.upper()} RISK ANALYST reviewing the trader's book as a whole. {view} \
Name specific symbols. Under 250 words.

## Macro brief
{macro}

## Trader's book
{plan_text}

Time now: {ctx.now:%A %H:%M} UTC""")


def portfolio_manager(llm, plan_text, risk_views, macro, ctx, max_orders, max_per_currency,
                      min_rr, errors: list[str] | None = None) -> FinalBook | None:
    return _structured(llm, FinalBook, f"""{GROUND_RULES}

You are the PORTFOLIO MANAGER and make the final call. From the trader's book, choose at \
most {max_orders} limit orders. At most {max_per_currency} may be long, or short, the same \
currency (gold and silver count as trades against USD). Every order keeps at least \
{min_rr:g}R after the spread. Set each order's validity so it is cancelled before any \
high-impact release for either currency. Prefer fewer, better orders: an empty book is the \
right answer on a day with no clean setups. Orders are re-checked in code; one that breaks \
a rule reverts to the scanner's levels or is dropped.

## Macro brief
{macro}

## Trader's book
{plan_text}

## Risk team
### Aggressive
{risk_views['aggressive']}

### Neutral
{risk_views['neutral']}

### Conservative
{risk_views['conservative']}

Time now: {ctx.now:%A %H:%M} UTC""", "Portfolio manager", errors)


# ---------------------------------------------------------------------------
# Rendering between steps
# ---------------------------------------------------------------------------

def _render_verdict(v: ResearchVerdict) -> str:
    lines = [v.summary, ""]
    lines += [f"- {d.symbol}: {'KEEP' if d.keep else 'DROP'}. {d.reason}" for d in v.decisions]
    return "\n".join(lines)


def _render_plan(orders: list[dict], by_symbol: dict[str, Setup]) -> str:
    lines = []
    for o in orders:
        s = by_symbol[o["symbol"]]
        lines.append(f"- {o['symbol']} {s.order_type} entry {o['entry']}, stop {o['stop']}, "
                     f"target {o['target']} (scanner: {s.entry}/{s.stop}/{s.target}, "
                     f"{s.rr:.2f}R, score {s.score:.0f}). {o.get('note', '')}")
    return "\n".join(lines) or "(no orders)"


def _render_book(book: FinalBook) -> str:
    lines = [book.summary, ""]
    for o in book.orders:
        lines.append(f"- {o.symbol} entry {o.entry} stop {o.stop} target {o.target}, "
                     f"{o.valid_hours:g}h, {o.conviction}: {o.rationale} Watch: {o.watch_for}")
    return "\n".join(lines)


def _normal(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("_", "").rstrip("+")


# ---------------------------------------------------------------------------
# The review
# ---------------------------------------------------------------------------

def review(
    candidates: list[Setup],
    ctx: ReviewContext,
    quick_llm: Any,
    deep_llm: Any,
    *,
    max_orders: int = 6,
    max_per_currency: int = 2,
    min_rr: float = 2.0,
    progress: Progress | None = None,
    now: datetime | None = None,
) -> Review:
    """Run the agent team over ``candidates`` and return the verified book."""
    step = progress or (lambda _msg: None)
    now = now or ctx.now
    by_symbol = {s.symbol: s for s in candidates}
    transcript: dict[str, str] = {}
    fallbacks: list[str] = []
    dropped: list[tuple[str, str]] = []
    problems: list[str] = []          # one-line reasons, collected as agents fail

    def decide(make, label):
        """Strong model first, then the fast one: they often have separate quotas."""
        answer = make(deep_llm)
        if answer is None and quick_llm is not deep_llm:
            answer = make(quick_llm)
            if answer is not None:
                fallbacks.append(f"{label}: decided by the fast model (strong model: "
                                 f"{problems[-1].split(': ', 1)[-1]})")
        return answer

    if not candidates:
        return Review(orders=[], dropped=[], summary="The scanner found no candidates to review.")

    step("Macro analyst reading the calendar and headlines")
    macro = _heard("Macro analyst", lambda: macro_analyst(quick_llm, candidates, ctx), problems)
    transcript["Macro analyst"] = macro

    step("Price-action analyst reading the structure")
    structure = _heard("Price-action analyst",
                       lambda: price_action_analyst(quick_llm, candidates, ctx), problems)
    transcript["Price-action analyst"] = structure

    step("Bull and bear researchers debating the setups")
    bull = _heard("Bull researcher",
                  lambda: researcher(quick_llm, "bull", candidates, ctx, macro, structure), problems)
    bear = _heard("Bear researcher",
                  lambda: researcher(quick_llm, "bear", candidates, ctx, macro, structure), problems)
    transcript["Bull researcher"] = bull
    transcript["Bear researcher"] = bear

    step("Research manager judging the debate")
    verdict = decide(lambda llm: research_manager(llm, candidates, ctx, macro, bull, bear, problems,
                                                  structure),
                     "Research manager")
    if verdict is None:
        fallbacks.append("Research manager: no usable answer; every candidate passed to the trader")
        kept = list(candidates)
        verdict_text = "(research manager unavailable: all candidates kept)"
    else:
        keep = {_normal(d.symbol) for d in verdict.decisions if d.keep}
        reasons = {_normal(d.symbol): d.reason for d in verdict.decisions}
        kept = [s for s in candidates if s.symbol in keep]
        for s in candidates:
            if s.symbol not in keep:
                dropped.append((s.symbol, f"research manager: {reasons.get(s.symbol, 'no verdict given')}"))
        verdict_text = _render_verdict(verdict)
    transcript["Research manager"] = verdict_text

    if not kept:
        return Review(orders=[], dropped=dropped, transcript=transcript, fallbacks=fallbacks,
                      problems=problems,
                      summary=verdict.summary if verdict else "No setups survived the debate.")

    step("Trader writing the orders")
    plan = trader(quick_llm, kept, ctx, verdict_text, problems)
    kept_symbols = {s.symbol for s in kept}
    if plan is None:
        fallbacks.append("Trader: no usable answer; scanner levels used")
        orders = []
    else:
        orders = [{"symbol": _normal(o.symbol), "entry": o.entry, "stop": o.stop,
                   "target": o.target, "note": o.note}
                  for o in plan.orders if _normal(o.symbol) in kept_symbols]
    written = {o["symbol"] for o in orders}
    orders += [{"symbol": s.symbol, "entry": s.entry, "stop": s.stop, "target": s.target,
                "note": "scanner levels"} for s in kept if s.symbol not in written]
    plan_text = _render_plan(orders, by_symbol)
    transcript["Trader"] = plan_text

    step("Risk team reviewing the book")
    risk_views = {stance: _heard(f"{stance.title()} risk analyst",
                                 lambda stance=stance: risk_analyst(quick_llm, stance, plan_text, macro, ctx),
                                 problems)
                  for stance in ("aggressive", "neutral", "conservative")}
    for stance, said in risk_views.items():
        transcript[f"{stance.title()} risk analyst"] = said

    step("Portfolio manager choosing the final orders")
    book = decide(lambda llm: portfolio_manager(llm, plan_text, risk_views, macro, ctx,
                                                max_orders, max_per_currency, min_rr, problems),
                  "Portfolio manager")
    if book is None:
        fallbacks.append("Portfolio manager: no usable answer; the trader's book ranked by "
                         "scanner score was used, all at low conviction")
        ranked = sorted(orders, key=lambda o: by_symbol[o["symbol"]].score, reverse=True)
        proposals = [{**o, "valid_hours": (by_symbol[o["symbol"]].expires_at - now).total_seconds() / 3600,
                      "conviction": "low", "rationale": "Agent review incomplete; scanner ranking.",
                      "watch_for": "See the macro brief."} for o in ranked]
        summary = "The portfolio manager gave no usable answer, so the trader's book is shown."
    else:
        transcript["Portfolio manager"] = _render_book(book)
        proposals = [o.model_dump() for o in book.orders]
        summary = book.summary
        chosen = {_normal(o.symbol) for o in book.orders}
        for o in orders:
            if o["symbol"] not in chosen:
                dropped.append((o["symbol"], "portfolio manager left it out of the final book"))

    step("Verifying the orders")
    accepted, rejected = verify(proposals, candidates, now=now, min_rr=min_rr,
                                max_orders=max_orders, max_per_currency=max_per_currency)
    dropped += [(sym, f"failed verification: {why}") for sym, why in rejected]
    return Review(orders=accepted, dropped=dropped, summary=summary,
                  transcript=transcript, fallbacks=fallbacks, problems=problems)
