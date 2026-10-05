"""Write a scan to disk as Markdown (to read) and JSON (for later tracking)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tradingagents.fx.scanner import ScanResult

if TYPE_CHECKING:
    from tradingagents.fx.agents import Review

DISCLAIMER = (
    "Scores rank setups against each other; they are not win probabilities. "
    "Check each level on your own chart and your broker's prices before placing an order."
)

AGENT_DISCLAIMER = (
    "These orders come from untested rules reviewed by AI agents. They have no track record "
    "yet: paper-trade them and check every level on your own chart before risking money."
)


def _scan_markdown(result: ScanResult, heading: str) -> list[str]:
    lines = [heading, ""]
    if result.setups:
        lines += [
            "| # | Symbol | Order | Entry | Stop | Target | RR | Risk (pips) | Score |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for i, s in enumerate(result.setups, 1):
            lines.append(f"| {i} | {s.symbol} | {s.order_type} | {s.entry} | {s.stop} | "
                         f"{s.target} | {s.rr:.2f} | {s.risk_pips} | {s.score:.0f} |")
        lines.append("")
        for i, s in enumerate(result.setups, 1):
            expiry = s.expires_at.strftime("%H:%M UTC")
            lines += [
                f"### {i}. {s.symbol} {s.order_type} @ {s.entry}",
                "",
                f"- Stop {s.stop} ({s.risk_pips} pips) · target {s.target} "
                f"({s.reward_pips} pips, {s.target_kind}) · {s.rr:.2f}R after a "
                f"{s.spread_pips}-pip spread",
                f"- Price at scan {s.price} · 1h ATR {s.atr_pips} pips · cancel if unfilled by {expiry}",
            ]
            lines += [f"- {reason}" for reason in s.reasons]
            lines.append("")
    else:
        lines += ["No setups met the rules. Sitting out is a valid result.", ""]
    if result.skipped:
        lines += ["### Not taken", ""]
        lines += [f"- {symbol}: {reason}" for symbol, reason in result.skipped]
        lines.append("")
    return lines


def to_markdown(result: ScanResult, review: Review | None = None) -> str:
    when = result.scanned_at.strftime("%Y-%m-%d %H:%M UTC")
    if review is None:
        lines = [f"# Forex & metals scan — {when}", "", f"_{DISCLAIMER}_", ""]
        lines += _scan_markdown(result, "## Setups")[2:]
        return "\n".join(lines)

    lines = [f"# Forex & metals: agent-reviewed orders — {when}", "", f"_{AGENT_DISCLAIMER}_", ""]
    if review.fallbacks:
        lines += ["> **Partial review.** " + " · ".join(review.fallbacks), ""]
    lines += ["## Final orders", ""]
    if review.orders:
        lines += [
            "| # | Symbol | Order | Entry | Stop | Target | RR | Cancel by (UTC) | Conviction |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for i, o in enumerate(review.orders, 1):
            lines.append(f"| {i} | {o.symbol} | {o.order_type} | {o.entry} | {o.stop} | {o.target} "
                         f"| {o.rr:.2f} | {o.expires_at:%H:%M} | {o.conviction} |")
        lines.append("")
        for i, o in enumerate(review.orders, 1):
            lines += [f"### {i}. {o.symbol} {o.order_type} @ {o.entry}", "", o.rationale, "",
                      f"**Watch for:** {o.watch_for}", ""]
            lines += [f"- _Verification: {note}_" for note in o.notes]
            if o.notes:
                lines.append("")
    else:
        lines += ["No orders today. The agents found nothing worth the risk; sitting out is a "
                  "valid result.", ""]
    lines += ["## The day", "", review.summary, ""]
    if review.dropped:
        lines += ["## Left out", ""]
        lines += [f"- {symbol}: {reason}" for symbol, reason in review.dropped]
        lines.append("")
    lines += ["## How the agents got there", ""]
    for agent, said in review.transcript.items():
        lines += [f"### {agent}", "", said, ""]
    lines += _scan_markdown(result, "## Stage 1: scanner candidates")
    return "\n".join(lines)


def save(result: ScanResult, results_dir: str | Path,
         review: Review | None = None) -> tuple[Path, Path]:
    """Save ``<results_dir>/fx_scans/<timestamp>.md`` and ``.json``."""
    folder = Path(results_dir) / "fx_scans"
    folder.mkdir(parents=True, exist_ok=True)
    stem = result.scanned_at.strftime("%Y-%m-%d_%H%M") + ("_agents" if review else "")
    md, js = folder / f"{stem}.md", folder / f"{stem}.json"
    md.write_text(to_markdown(result, review), encoding="utf-8")
    data = result.to_dict()
    if review is not None:
        data["review"] = review.to_dict()
    js.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return md, js
