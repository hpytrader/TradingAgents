"""The FX desk's agents: who they are, what they do, and what they know.

Each agent in the review has a name, a role, a model tier and a short bio, so
the desk chat reads as a conversation between colleagues and the dashboard
can introduce the team. Rename anyone without touching the code by writing
``~/.tradingagents/fx_team.json`` (or the file ``TRADINGAGENTS_FX_TEAM`` names),
for example::

    {"macro": {"name": "Jarvis"}, "portfolio_manager": {"name": "Friday"}}

Only the keys given are replaced; the rest keep their defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    role: str
    tier: str                 # "quick", "deep" or "code"
    expertise: tuple[str, ...]
    bio: str
    accent: str               # avatar ring colour


DEFAULTS: tuple[Profile, ...] = (
    Profile("desk", "Desk", "Desk system", "code",
            ("Live OANDA prices", "SMC scanner", "Order verification"),
            "The rule-based core. Scans every pair and metal for sweep → structure-shift → zone setups, "
            "gathers the calendar and headlines, and checks every final order against the facts before "
            "anyone sees it. It is code, not a model: it never guesses.", "#5f7782"),
    Profile("macro", "Atlas", "Macro analyst", "quick",
            ("Central banks", "Economic calendar", "Headline flow"),
            "Reads the week's calendar and the latest FXStreet headlines, and briefs the desk on what is "
            "driving each currency and which releases could hit an order before it plays out.", "#3987e5"),
    Profile("price_action", "Vega", "Price-action analyst", "quick",
            ("Liquidity sweeps", "BOS / CHoCH", "Order blocks & FVGs"),
            "Grades each setup the way a smart-money trader reads a chart: the liquidity taken, how "
            "decisive the shift was, the quality of the zone, discount or premium, and whether the "
            "target is realistic for the time left.", "#9085e9"),
    Profile("bull", "Leo", "Bull researcher", "quick",
            ("Trend continuation", "Confluence", "Upside cases"),
            "Makes the strongest honest case for taking each setup, from the structure and the news "
            "that supports it.", "#199e70"),
    Profile("bear", "Ursa", "Bear researcher", "quick",
            ("Stop hunts vs breakdowns", "Event risk", "Failed zones"),
            "Answers Leo point by point with the strongest honest case against each setup: what could "
            "make the sweep a real breakdown, the zone fail, or a release blow through the stop.", "#d95926"),
    Profile("research_manager", "Sage", "Research manager", "deep",
            ("Weighing evidence", "Debate judgement"),
            "Judges the debate between Leo and Ursa and decides which setups go to the trader. Drops "
            "anything whose bear case is stronger.", "#c98500"),
    Profile("trader", "Nova", "Execution trader", "quick",
            ("Limit-order placement", "Stops & targets"),
            "Turns the surviving setups into orders, keeping the scanner's levels unless the debate "
            "gives a concrete reason to move them.", "#d55181"),
    Profile("risk_aggressive", "Blaze", "Risk analyst, aggressive", "quick",
            ("Opportunity cost", "Sizing for strength"),
            "Argues for taking the strongest trades in full and for the lower-conviction ones that "
            "still earn a place.", "#e66767"),
    Profile("risk_conservative", "Haven", "Risk analyst, conservative", "quick",
            ("Capital preservation", "News exposure", "Correlation"),
            "Answers Blaze: argues for cutting anything exposed to a release, stretched, or stacked on "
            "one currency.", "#199e70"),
    Profile("risk_neutral", "Pivot", "Risk analyst, neutral", "quick",
            ("Portfolio balance", "Combined exposure"),
            "Hears Blaze and Haven, then weighs the book as a whole: shared currencies, correlated pairs, "
            "orders that could fill into the same release.", "#9db6c1"),
    Profile("portfolio_manager", "Orion", "Portfolio manager", "deep",
            ("Final allocation", "Cancel timing"),
            "Makes the final call: up to six limit orders, or none, each with a cancel time set ahead "
            "of any high-impact release.", "#5ce1f0"),
)


def _overrides_path() -> Path:
    custom = os.getenv("TRADINGAGENTS_FX_TEAM")
    if custom:
        return Path(custom)
    return Path(os.path.expanduser("~")) / ".tradingagents" / "fx_team.json"


def team(path: str | Path | None = None) -> dict[str, Profile]:
    """The profiles by key, with any renames from ``fx_team.json`` applied."""
    profiles = {p.key: p for p in DEFAULTS}
    file = Path(path) if path else _overrides_path()
    try:
        data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    except (OSError, ValueError):
        data = {}
    for key, changes in (data or {}).items():
        if key in profiles and isinstance(changes, dict):
            allowed = {k: v for k, v in changes.items() if k in ("name", "role", "bio") and isinstance(v, str)}
            if "expertise" in changes and isinstance(changes["expertise"], list):
                allowed["expertise"] = tuple(str(x) for x in changes["expertise"])
            profiles[key] = replace(profiles[key], **allowed)
    return profiles


def name(key: str) -> str:
    return team()[key].name
