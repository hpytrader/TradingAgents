"""Write a scan to disk as Markdown (to read) and JSON (for later tracking)."""

from __future__ import annotations

import json
from pathlib import Path

from tradingagents.fx.scanner import ScanResult

DISCLAIMER = (
    "Scores rank setups against each other; they are not win probabilities. "
    "Check each level on your own chart and your broker's prices before placing an order."
)


def to_markdown(result: ScanResult) -> str:
    when = result.scanned_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Forex & metals scan — {when}", "", f"_{DISCLAIMER}_", ""]
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
                f"## {i}. {s.symbol} {s.order_type} @ {s.entry}",
                "",
                f"- Stop {s.stop} ({s.risk_pips} pips) · target {s.target} "
                f"({s.reward_pips} pips, {s.target_kind}) · {s.rr:.2f}R after a "
                f"{s.spread_pips}-pip spread",
                f"- Price at scan {s.price} · 1h ATR {s.atr_pips} pips · cancel if unfilled by {expiry}",
            ]
            lines += [f"- {reason}" for reason in s.reasons]
            lines.append("")
    else:
        lines += ["No setups met the rules this morning. Sitting out is a valid result.", ""]
    if result.skipped:
        lines += ["## Not taken", ""]
        lines += [f"- {symbol}: {reason}" for symbol, reason in result.skipped]
        lines.append("")
    return "\n".join(lines)


def save(result: ScanResult, results_dir: str | Path) -> tuple[Path, Path]:
    """Save ``<results_dir>/fx_scans/<timestamp>.md`` and ``.json``."""
    folder = Path(results_dir) / "fx_scans"
    folder.mkdir(parents=True, exist_ok=True)
    stem = result.scanned_at.strftime("%Y-%m-%d_%H%M")
    md, js = folder / f"{stem}.md", folder / f"{stem}.json"
    md.write_text(to_markdown(result), encoding="utf-8")
    js.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return md, js
