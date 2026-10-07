"""Smart-money structure, detected from candles: swings, BOS/CHoCH, liquidity,
sweeps, order blocks and fair value gaps.

Everything here is computed, not judged, so two runs on the same candles give
the same answer and every level in a setup can be pointed to on a chart. Bars
are positional (``frame.iloc[i]``) and a swing only exists once the ``wing``
bars after it have closed, so no function reads a bar from its own future.

Definitions used throughout (bullish side; bearish mirrors it):

- **Swing high / low**: a bar whose high (low) is beyond the ``wing`` bars on
  each side of it.
- **BOS / CHoCH**: a close beyond the last confirmed swing. It is a break of
  structure when it continues the current direction and a change of character
  when it reverses it.
- **Liquidity pool**: a level where stops gather: the previous trading day's
  high or low, a session's high or low, or two or more swing points at nearly
  the same price ("equal highs / lows").
- **Sweep**: a bar trades through a pool and closes back on the other side.
- **Order block**: the last opposite-colour candle before the move that broke
  structure: the last down-close candle before a bullish CHoCH.
- **Fair value gap**: in a bullish move, a bar whose low stays above the high
  two bars earlier, leaving a gap price has not traded through.
- **Mitigated**: price has come back into a zone since it formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytz
from numpy.lib.stride_tricks import sliding_window_view

from tradingagents.fx import indicators as ind

NEW_YORK = pytz.timezone("America/New_York")
LONDON = pytz.timezone("Europe/London")


@dataclass(frozen=True)
class Swing:
    i: int                  # bar position
    price: float
    kind: str               # "high" or "low"


@dataclass(frozen=True)
class Break:
    i: int                  # bar that closed beyond the swing
    kind: str               # "BOS" or "CHoCH"
    direction: str          # "up" or "down"
    level: float            # the swing that was broken
    swing_i: int


@dataclass(frozen=True)
class Pool:
    name: str               # "previous day low", "Asian high", "equal lows", ...
    level: float
    side: str               # "low" (sell-side liquidity) or "high" (buy-side)
    quality: int            # 3 previous day, 2 session range, 1 equal highs/lows, 0 swing point
    formed_i: int = 0       # first bar at which the pool exists; only later bars can sweep it


@dataclass(frozen=True)
class Sweep:
    pool: Pool
    i: int                  # bar that traded through and closed back
    extreme: float          # that bar's low (for a low sweep) or high


@dataclass(frozen=True)
class Zone:
    kind: str               # "order block" or "fair value gap"
    low: float
    high: float
    formed_i: int           # last bar of its formation

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    def overlaps(self, other: Zone) -> bool:
        return self.low <= other.high and other.low <= self.high


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def swings(frame: pd.DataFrame, wing: int = 2) -> list[Swing]:
    """Confirmed swing highs and lows, oldest first."""
    highs, lows = frame["high"].to_numpy(dtype=float), frame["low"].to_numpy(dtype=float)
    size = 2 * wing + 1
    if len(frame) < size:
        return []
    h_win, l_win = sliding_window_view(highs, size), sliding_window_view(lows, size)
    h_mid, l_mid = h_win[:, wing], l_win[:, wing]
    # the strict extreme of its window: the highest (lowest) bar, and the only one at that price
    is_high = (h_mid == h_win.max(axis=1)) & ((h_win == h_mid[:, None]).sum(axis=1) == 1)
    is_low = (l_mid == l_win.min(axis=1)) & ((l_win == l_mid[:, None]).sum(axis=1) == 1)
    out = []
    for j in np.flatnonzero(is_high | is_low):
        i = int(j) + wing
        if is_high[j]:
            out.append(Swing(i, float(highs[i]), "high"))
        if is_low[j]:
            out.append(Swing(i, float(lows[i]), "low"))
    return out


def breaks(frame: pd.DataFrame, wing: int = 2) -> list[Break]:
    """Every BOS and CHoCH in ``frame``, oldest first.

    A swing becomes breakable only once it is confirmed (``wing`` bars later),
    and each swing can be broken once.
    """
    closes = frame["close"].to_numpy()
    pending = sorted(swings(frame, wing), key=lambda s: s.i + wing)
    last_high: Swing | None = None
    last_low: Swing | None = None
    trend: str | None = None
    out: list[Break] = []
    k = 0
    for i in range(len(frame)):
        while k < len(pending) and pending[k].i + wing <= i:
            s = pending[k]
            if s.kind == "high":
                last_high = s
            else:
                last_low = s
            k += 1
        c = closes[i]
        if last_high is not None and c > last_high.price:
            out.append(Break(i, "CHoCH" if trend == "down" else "BOS", "up",
                             last_high.price, last_high.i))
            trend, last_high = "up", None
        elif last_low is not None and c < last_low.price:
            out.append(Break(i, "CHoCH" if trend == "up" else "BOS", "down",
                             last_low.price, last_low.i))
            trend, last_low = "down", None
    return out


def bias(frame: pd.DataFrame, wing: int = 2) -> tuple[str | None, Break | None]:
    """``"long"``/``"short"`` from the latest structure break, with that break."""
    found = breaks(frame, wing)
    if not found:
        return None, None
    last = found[-1]
    return ("long" if last.direction == "up" else "short"), last


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------

def trading_day(ts: pd.Timestamp) -> date:
    """The forex trading day a bar belongs to: days roll at 17:00 New York."""
    return (ts.tz_convert(NEW_YORK) + timedelta(hours=7)).date()


def _session_range(frame: pd.DataFrame, tz, start: time, end: time,
                   day) -> tuple[float, float, int] | None:
    """Low, high and the position of the session's last bar."""
    local = frame.index.tz_convert(tz)
    mask = (local.date == day) & (local.time >= start) & (local.time < end)
    if not mask.any():
        return None
    part = frame[mask]
    last = int(mask.nonzero()[0][-1])
    return float(part["low"].min()), float(part["high"].max()), last


def pools(frame: pd.DataFrame, now: datetime, atr: float, wing: int = 2) -> list[Pool]:
    """Liquidity pools visible at ``now`` on an intraday frame (M5)."""
    out: list[Pool] = []
    if frame.empty:
        return out
    days = pd.Series((frame.index.tz_convert(NEW_YORK) + pd.Timedelta(hours=7)).date, index=frame.index)
    today = trading_day(pd.Timestamp(now))
    earlier = sorted(d for d in days.unique() if d < today)
    if earlier:
        mask = (days == earlier[-1]).to_numpy()
        prev = frame[mask]
        end = int(mask.nonzero()[0][-1])
        out.append(Pool("previous day low", float(prev["low"].min()), "low", 3, end))
        out.append(Pool("previous day high", float(prev["high"].max()), "high", 3, end))

    utc_day = pd.Timestamp(now).tz_convert("UTC").date()
    asia = _session_range(frame, pytz.UTC, time(0, 0), time(6, 0), utc_day)
    if asia and pd.Timestamp(now).tz_convert("UTC").time() >= time(6, 0):
        out.append(Pool("Asian low", asia[0], "low", 2, asia[2]))
        out.append(Pool("Asian high", asia[1], "high", 2, asia[2]))
    london_day = pd.Timestamp(now).tz_convert(LONDON).date()
    london = _session_range(frame, LONDON, time(8, 0), time(12, 0), london_day)
    if london and pd.Timestamp(now).tz_convert(LONDON).time() >= time(12, 0):
        out.append(Pool("London low", london[0], "low", 2, london[2]))
        out.append(Pool("London high", london[1], "high", 2, london[2]))

    tolerance = 0.15 * atr
    offset = max(len(frame) - 288, 0)
    recent = swings(frame.iloc[offset:], wing)
    for kind, side, name in (("low", "low", "equal lows"), ("high", "high", "equal highs")):
        points = [s for s in recent if s.kind == kind]
        for anchor in points:
            # The pool exists once its second swing is confirmed; use the
            # earliest pair so a later sweep of it can be recognised.
            group = [p for p in points if abs(p.price - anchor.price) <= tolerance and p.i >= anchor.i]
            if len(group) < 2:
                continue
            pair = group[:2]
            level = min(p.price for p in pair) if side == "low" else max(p.price for p in pair)
            formed = offset + pair[1].i + wing
            if all(abs(level - p.level) > tolerance for p in out if p.side == side):
                out.append(Pool(name, level, side, 1, formed))

    # Single swing points of the last twelve hours: the stops a swing failure
    # pattern (SFP) runs. Only those not already covered by a stronger pool.
    near = max(len(frame) - 144, 0)
    for s in swings(frame.iloc[near:], wing):
        side = s.kind
        if all(abs(s.price - p.level) > tolerance for p in out if p.side == side):
            out.append(Pool(f"swing {side}", s.price, side, 0, near + s.i + wing))
    return out


def sweeps(frame: pd.DataFrame, liquidity: list[Pool], side: str, since_i: int) -> list[Sweep]:
    """Bars from ``since_i`` on that traded through a ``side`` pool and closed back."""
    lows, highs, closes = (frame[c].to_numpy() for c in ("low", "high", "close"))
    out = []
    for pool in (p for p in liquidity if p.side == side):
        for i in range(max(pool.formed_i + 1, 0), len(frame)):
            if side == "low" and lows[i] < pool.level < closes[i]:
                if i >= since_i:
                    out.append(Sweep(pool, i, float(lows[i])))
                break
            if side == "high" and highs[i] > pool.level > closes[i]:
                if i >= since_i:
                    out.append(Sweep(pool, i, float(highs[i])))
                break
            # A close beyond the level broke it: there is no liquidity left to sweep.
            if (side == "low" and closes[i] < pool.level) or (side == "high" and closes[i] > pool.level):
                break
    return out


# ---------------------------------------------------------------------------
# Points of interest
# ---------------------------------------------------------------------------

def order_block(frame: pd.DataFrame, start: int, end: int, direction: str) -> Zone | None:
    """The last opposite-colour candle in ``[start, end)``: down-close for a bullish move."""
    opens, closes = frame["open"].to_numpy(), frame["close"].to_numpy()
    for i in range(end - 1, start - 1, -1):
        bearish = closes[i] < opens[i]
        if (direction == "up" and bearish) or (direction == "down" and not bearish and closes[i] != opens[i]):
            return Zone("order block", float(frame["low"].iat[i]), float(frame["high"].iat[i]), i)
    return None


def fair_value_gaps(frame: pd.DataFrame, start: int, end: int, direction: str) -> list[Zone]:
    """Gaps left by the move in ``[start, end]`` (three-bar imbalances)."""
    highs, lows = frame["high"].to_numpy(), frame["low"].to_numpy()
    out = []
    for i in range(max(start, 1), min(end, len(frame) - 2) + 1):
        if direction == "up" and highs[i - 1] < lows[i + 1]:
            out.append(Zone("fair value gap", float(highs[i - 1]), float(lows[i + 1]), i + 1))
        if direction == "down" and lows[i - 1] > highs[i + 1]:
            out.append(Zone("fair value gap", float(highs[i + 1]), float(lows[i - 1]), i + 1))
    return out


def mitigated(frame: pd.DataFrame, zone: Zone, price: float, direction: str,
              after: int = -1) -> bool:
    """Whether price came back to ``price`` inside the zone after leaving it.

    Only bars after both the zone and ``after`` (the structure-shift bar)
    count: the displacement candles that leave an order block often wick back
    into it on the way out, and that is the move itself, not a return.
    """
    after = frame.iloc[max(zone.formed_i, after) + 1:]
    if after.empty:
        return False
    if direction == "up":
        return bool((after["low"] <= price).any())
    return bool((after["high"] >= price).any())


def atr(frame: pd.DataFrame, period: int = 14) -> float:
    return float(ind.atr(frame, period).iloc[-1])
