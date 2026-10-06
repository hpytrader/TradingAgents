"""Check every order the agents produce against the scanner's facts.

A language model can write a confident order with a price on the wrong side of
the market, a stop inside the spread or an RR it did not compute. Nothing an
agent writes reaches the report until it passes these checks. An adjusted
order that fails falls back to the scanner's own levels when those pass; an
order that cannot be repaired is dropped, with the reason.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.fx.instruments import spec_for
from tradingagents.fx.scanner import Setup

MAX_SHIFT_ATR = 1.5      # an agent may move the entry this far from the scanner's level
MIN_STOP_ATR = 0.5       # and may not tighten the stop below this
MIN_HOURS, MAX_HOURS = 1.0, 12.0


@dataclass
class VerifiedOrder:
    setup: Setup                    # the scanner's candidate it came from
    entry: float
    stop: float
    target: float
    rr: float
    expires_at: datetime
    conviction: str
    rationale: str
    watch_for: str
    notes: list[str] = field(default_factory=list)   # what verification changed

    @property
    def symbol(self) -> str:
        return self.setup.symbol

    @property
    def direction(self) -> str:
        return self.setup.direction

    @property
    def order_type(self) -> str:
        return self.setup.order_type

    def to_dict(self) -> dict:
        spec = spec_for(self.symbol)
        return {
            "symbol": self.symbol,
            "order_type": self.order_type,
            "direction": self.direction,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "rr": self.rr,
            "risk_pips": round(spec.pips(abs(self.entry - self.stop)), 1),
            "reward_pips": round(spec.pips(abs(self.target - self.entry)), 1),
            "expires_at": self.expires_at.isoformat(),
            "conviction": self.conviction,
            "rationale": self.rationale,
            "watch_for": self.watch_for,
            "notes": self.notes,
            "scanner_score": self.setup.score,
        }


def rr_after_spread(setup: Setup, entry: float, stop: float, target: float) -> float:
    spread = spec_for(setup.symbol).pip * setup.spread_pips
    risk = abs(entry - stop)
    return (abs(target - entry) - spread) / (risk + spread) if risk + spread > 0 else 0.0


def check_levels(setup: Setup, entry: float, stop: float, target: float,
                 min_rr: float) -> str | None:
    """Why these levels are not a valid order for ``setup``, or ``None`` if they are."""
    spec = spec_for(setup.symbol)
    atr = setup.atr_pips * spec.pip
    long = setup.direction == "long"
    if not all(isinstance(x, (int, float)) and x > 0 for x in (entry, stop, target)):
        return "a price is missing or not positive"
    if long and not stop < entry < target:
        return "for a buy the stop must be below the entry and the target above it"
    if not long and not target < entry < stop:
        return "for a sell the stop must be above the entry and the target below it"
    if long and entry >= setup.price:
        return f"a buy limit must sit below the price at scan time ({setup.price})"
    if not long and entry <= setup.price:
        return f"a sell limit must sit above the price at scan time ({setup.price})"
    if setup.strategy == "smc" and setup.zone_low is not None:
        # The trade is defined by its zone and its sweep: the entry stays in the
        # order block / FVG, and the stop stays beyond the swept extreme, where
        # the idea is proven wrong.
        slack = 0.1 * (setup.zone_high - setup.zone_low) + spec.pip / 10
        if not setup.zone_low - slack <= entry <= setup.zone_high + slack:
            return (f"entry {entry} is not inside the {setup.zone_low}–{setup.zone_high} "
                    "order block / fair value gap")
        if long and stop >= setup.invalidation:
            return f"stop {stop} is not beyond the sweep low ({setup.invalidation})"
        if not long and stop <= setup.invalidation:
            return f"stop {stop} is not beyond the sweep high ({setup.invalidation})"
    else:
        if abs(entry - setup.entry) > MAX_SHIFT_ATR * atr:
            return (f"entry moved {spec.pips(abs(entry - setup.entry)):.0f} pips from the scanner's "
                    f"level, more than {MAX_SHIFT_ATR} ATR")
        if abs(entry - stop) < MIN_STOP_ATR * atr:
            return f"stop is tighter than {MIN_STOP_ATR} ATR ({MIN_STOP_ATR * setup.atr_pips:.1f} pips)"
    rr = rr_after_spread(setup, entry, stop, target)
    if rr < min_rr:
        return f"RR after the spread is {rr:.2f}, below {min_rr:g}"
    return None


def verify(
    proposals: list[dict],
    candidates: list[Setup],
    *,
    now: datetime,
    min_rr: float = 2.0,
    max_orders: int = 6,
    max_per_currency: int = 2,
) -> tuple[list[VerifiedOrder], list[tuple[str, str]]]:
    """Turn the portfolio manager's proposals into orders that pass every check.

    ``proposals`` are dicts with symbol, entry, stop, target, valid_hours,
    conviction, rationale and watch_for. Returns the accepted orders and the
    ``(symbol, reason)`` of each proposal that was dropped.
    """
    by_symbol = {s.symbol: s for s in candidates}
    accepted: list[VerifiedOrder] = []
    dropped: list[tuple[str, str]] = []
    exposure: Counter = Counter()
    seen: set[str] = set()

    for p in proposals:
        symbol = str(p.get("symbol", "")).upper().replace("/", "").replace("_", "").rstrip("+")
        setup = by_symbol.get(symbol)
        if setup is None:
            dropped.append((symbol or "?", "not one of the scanner's candidates"))
            continue
        if symbol in seen:
            dropped.append((symbol, "listed twice; the first order was kept"))
            continue
        if len(accepted) >= max_orders:
            dropped.append((symbol, f"over the {max_orders}-order limit"))
            continue

        notes: list[str] = []
        spec = spec_for(symbol)
        try:
            entry, stop, target = (spec.round_price(float(p[k])) for k in ("entry", "stop", "target"))
        except (KeyError, TypeError, ValueError):
            entry = stop = target = 0.0
        problem = check_levels(setup, entry, stop, target, min_rr)
        if problem:
            fallback = check_levels(setup, setup.entry, setup.stop, setup.target, min_rr)
            if fallback:
                dropped.append((symbol, f"{problem}; the scanner's levels fail too ({fallback})"))
                continue
            notes.append(f"agents' levels rejected ({problem}); scanner levels used")
            entry, stop, target = setup.entry, setup.stop, setup.target

        sign = 1 if setup.direction == "long" else -1
        legs = [(symbol[:3], sign), (symbol[3:], -sign)]
        crowded = [cur for cur, s in legs if exposure[(cur, s)] >= max_per_currency]
        if crowded:
            dropped.append((symbol, f"the book already has {max_per_currency} trades "
                                    f"{'long' if dict(legs)[crowded[0]] > 0 else 'short'} {crowded[0]}"))
            continue

        try:
            hours = float(p.get("valid_hours", 8))
        except (TypeError, ValueError):
            hours = 8.0
        clamped = min(max(hours, MIN_HOURS), MAX_HOURS)
        if clamped != hours:
            notes.append(f"validity {hours:g}h clamped to {clamped:g}h")
        expires = now + timedelta(hours=clamped)
        if setup.strategy == "smc" and expires > setup.expires_at:
            expires = setup.expires_at
            notes.append(f"cancel time capped at the session end, {expires:%H:%M} UTC")
        conviction = str(p.get("conviction", "low")).lower()
        if conviction not in ("low", "medium", "high"):
            conviction = "low"

        accepted.append(VerifiedOrder(
            setup=setup,
            entry=entry, stop=stop, target=target,
            rr=round(rr_after_spread(setup, entry, stop, target), 2),
            expires_at=expires,
            conviction=conviction,
            rationale=str(p.get("rationale", "")).strip(),
            watch_for=str(p.get("watch_for", "")).strip(),
            notes=notes,
        ))
        exposure.update(legs)
        seen.add(symbol)
    return accepted, dropped
