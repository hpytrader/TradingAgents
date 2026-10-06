"""Structured outputs for the forex agents that make decisions.

Free-text agents (the macro analyst, the researchers, the risk debaters) argue;
these three decide, so their answers are typed and then checked in code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SetupVerdict(BaseModel):
    symbol: str = Field(description="The candidate's symbol exactly as listed, e.g. EURUSD.")
    keep: bool = Field(description="True to pass the setup to the trader, False to drop it.")
    reason: str = Field(description="One or two sentences: the deciding argument from the debate.")


class ResearchVerdict(BaseModel):
    decisions: list[SetupVerdict] = Field(
        description="One verdict for every candidate, in any order."
    )
    summary: str = Field(description="Two or three sentences on today's overall picture.")


class TraderOrder(BaseModel):
    symbol: str = Field(description="A symbol the research manager kept.")
    entry: float = Field(description="Limit price. Keep the scanner's level unless there is a reason to move it.")
    stop: float = Field(description="Stop-loss price, beyond the entry level.")
    target: float = Field(description="Take-profit price.")
    note: str = Field(description="What you changed and why, or 'scanner levels kept'.")


class TraderPlan(BaseModel):
    orders: list[TraderOrder]


class FinalOrder(BaseModel):
    symbol: str = Field(description="A symbol from the trader's plan.")
    entry: float = Field(description="Limit price.")
    stop: float = Field(description="Stop-loss price.")
    target: float = Field(description="Take-profit price.")
    valid_hours: float = Field(
        description="Hours until an unfilled order should be cancelled, 1 to 12. "
                    "Cancel before a high-impact release for either currency."
    )
    conviction: Literal["low", "medium", "high"]
    rationale: str = Field(description="Why this order made the final list, in two or three sentences.")
    watch_for: str = Field(description="The event or price action that would invalidate it.")


class FinalBook(BaseModel):
    orders: list[FinalOrder] = Field(
        description="The final limit orders, best first. Fewer than the maximum, or none, is allowed."
    )
    summary: str = Field(description="Three or four sentences: the day's theme and how the book is positioned.")


class TradeAction(BaseModel):
    ticket: str = Field(description="The trade's ticket exactly as listed, e.g. #1043.")
    action: Literal["hold", "close", "move_stop", "move_target", "cancel"] = Field(
        description="hold; close (open trades, at the current price); move_stop (open trades, tighten "
                    "only); move_target (open trades, still beyond price); cancel (pending orders only)."
    )
    new_stop: float | None = Field(default=None, description="For move_stop: the new stop price.")
    new_target: float | None = Field(default=None, description="For move_target: the new target price.")
    reason: str = Field(description="One or two sentences: why.")


class ManagementPlan(BaseModel):
    actions: list[TradeAction] = Field(description="One action for every listed trade.")
    summary: str = Field(description="One or two sentences on the open book.")
