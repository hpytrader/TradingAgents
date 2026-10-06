"""Stage 1 of the morning forex scan: find intraday limit-order setups, no AI.

For each instrument the scanner asks three questions, in order:

1. **Is there a trend to trade with?** On the 4-hour chart, price above a rising
   50 EMA above the 200 EMA is an uptrend (mirror for down). No clear trend, no
   setup: pullback entries against an unclear backdrop are coin flips.
2. **Where would a pullback stop?** On the 1-hour chart, recent swing lows (for a
   long) a reachable distance below price, and the 1-hour 50 EMA, are candidate
   entry levels. A buy limit sits just above the level, the stop just below it.
3. **Is there room to the target?** The take-profit is the nearest 1-hour swing
   high above price that pays at least ``min_rr`` times the risk after the
   spread. With no swing high above (price at new highs) a measured target of
   2.5R is used instead.

Each surviving setup gets a score out of 100 from trend strength, how often the
level has held, EMA confluence, reward-to-risk, distance and RSI. **The score
ranks setups against each other; it is not a probability of winning.** Win rates
can only come from tracking the setups over time.

Everything here is deterministic and free to run. The agents (stage 2) read the
top candidates this produces and decide which, if any, to take.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from tradingagents.fx import indicators as ind
from tradingagents.fx.instruments import DEFAULT_UNIVERSE, InstrumentSpec, spec_for

CandleFetcher = Callable[[str, str, int], pd.DataFrame]   # symbol, granularity, count
QuoteFetcher = Callable[[str], object]                     # symbol -> object with bid/ask

# Bars requested per timeframe: enough for a 200 EMA to settle on H4 and for a
# week of H1 structure.
H4_BARS = 260
H1_BARS = 300
STRUCTURE_BARS = 120      # H1 bars searched for swing levels (~5 trading days)

MIN_ENTRY_ATR = 0.3       # closer than this and the limit is effectively market
MAX_ENTRY_ATR = 2.5       # farther than this and it is unlikely to fill in a session
MAX_TARGET_ATR = 5.0      # an intraday move larger than this is a stretch
MEASURED_R = 2.5          # target multiple when no swing level lies in the way
ENTRY_BUFFER_ATR = 0.1    # front-run the level: orders cluster exactly on it
STOP_BUFFER_ATR = 1.0     # beyond the level: a full hour's range, past the wicks that tag it
MIN_STOP_SPREADS = 3      # a stop must be at least this many spreads beyond the level
STALE_AFTER = timedelta(hours=3)   # newest H1 bar older than this: market closed
MIN_TREND_ATR = 0.5       # EMAs closer than this (in H4 ATRs) are a range, not a trend
FULL_TREND_ATR = 6.0      # EMA spread (H4 ATRs) that earns the full trend points
FULL_LEVEL_VISITS = 6     # visits to a level that earn the full level points


@dataclass
class Setup:
    symbol: str
    direction: str                 # "long" or "short"
    entry: float                   # limit price
    stop: float
    target: float
    rr: float                      # reward ÷ risk after the spread
    score: float                   # 0–100, a ranking, not a probability
    risk_pips: float
    reward_pips: float
    spread_pips: float
    atr_pips: float                # 1-hour ATR
    price: float                   # mid price at scan time
    target_kind: str               # "structure" or "measured"
    expires_at: datetime
    reasons: list[str] = field(default_factory=list)
    strategy: str = "trend"            # "trend" or "smc"
    zone_low: float | None = None      # smc: the order block / FVG the entry sits in
    zone_high: float | None = None
    invalidation: float | None = None  # smc: the sweep extreme the stop must stay beyond

    @property
    def order_type(self) -> str:
        return "BUY LIMIT" if self.direction == "long" else "SELL LIMIT"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["order_type"] = self.order_type
        data["expires_at"] = self.expires_at.isoformat()
        return data


@dataclass
class ScanResult:
    scanned_at: datetime
    setups: list[Setup]                                   # ranked, best first
    skipped: list[tuple[str, str]] = field(default_factory=list)   # symbol, why
    ran: bool = True                                      # False when the scan was outside its window

    def to_dict(self) -> dict:
        return {
            "scanned_at": self.scanned_at.isoformat(),
            "setups": [s.to_dict() for s in self.setups],
            "skipped": [{"symbol": s, "reason": r} for s, r in self.skipped],
        }


@dataclass
class _Context:
    spec: InstrumentSpec
    h1: pd.DataFrame
    bias: str
    trend_strength: float          # (EMA50 − EMA200) ÷ H4 ATR, absolute
    atr1: float
    ema50_h1: float
    rsi_h1: float
    mid: float
    spread: float


def scan(
    candles: CandleFetcher,
    quote: QuoteFetcher,
    symbols: Iterable[str] = DEFAULT_UNIVERSE,
    *,
    min_rr: float = 2.0,
    top: int = 10,
    max_per_currency: int = 2,
    valid_hours: float = 8.0,
    stop_atr: float = STOP_BUFFER_ATR,
    now: datetime | None = None,
) -> ScanResult:
    """Scan ``symbols`` and return the best setup per instrument, ranked.

    ``stop_atr`` places the stop that many 1-hour ATRs beyond the entry level.
    A tighter stop shows a higher reward-to-risk on paper but is hit by
    ordinary noise more often, so it does not make a setup better.

    ``candles`` and ``quote`` fetch the data (the OANDA vendor in normal use, a
    stub in tests), so the scanner itself never touches the network. A symbol
    whose data cannot be fetched is listed in ``skipped`` with the reason and
    the scan carries on.
    """
    now = now or datetime.now(UTC)
    expires = now + timedelta(hours=valid_hours)
    best: list[Setup] = []
    skipped: list[tuple[str, str]] = []

    for symbol in symbols:
        try:
            spec = spec_for(symbol)
        except ValueError as exc:
            skipped.append((symbol, str(exc)))
            continue
        try:
            context = _context(spec, candles, quote, now)
        except Exception as exc:  # a vendor error for one pair must not end the scan
            skipped.append((spec.symbol, f"data unavailable: {exc}"))
            continue
        if isinstance(context, str):
            skipped.append((spec.symbol, context))
            continue
        setups = _setups(context, min_rr, expires, stop_atr)
        if not setups:
            skipped.append((spec.symbol, f"{context.bias} trend, but no level with "
                            f"{min_rr:g}R room after the spread"))
            continue
        best.append(max(setups, key=lambda s: s.score))

    return finish(best, skipped, now, top, max_per_currency)


def finish(best: list[Setup], skipped: list[tuple[str, str]], now: datetime,
           top: int, max_per_currency: int) -> ScanResult:
    """Rank the best setup per instrument and say why each one left out was."""
    ranked = _rank(best, top, max_per_currency)
    taken = Counter(e for s in ranked for e in _exposure(s))
    for setup in best:
        if setup in ranked:
            continue
        crowded = [cur for cur, sign in _exposure(setup) if taken[(cur, sign)] >= max_per_currency]
        if crowded:
            side = "long" if dict(_exposure(setup))[crowded[0]] > 0 else "short"
            why = (f"scored {setup.score:.0f}, but the list already has {max_per_currency} "
                   f"trades {side} {crowded[0]}")
        else:
            why = f"scored {setup.score:.0f}, below the top {top}"
        skipped.append((setup.symbol, why))
    return ScanResult(scanned_at=now, setups=ranked, skipped=skipped)


def _context(spec: InstrumentSpec, candles: CandleFetcher, quote: QuoteFetcher,
             now: datetime) -> _Context | str:
    """Read trend and volatility, or say why the instrument is not tradeable now."""
    h4 = candles(spec.symbol, "H4", H4_BARS)
    h1 = candles(spec.symbol, "H1", H1_BARS)
    if len(h4) < 210 or len(h1) < STRUCTURE_BARS:
        return "not enough price history"
    if now - h1.index[-1].to_pydatetime() > STALE_AFTER:
        return "no recent prices (market closed?)"

    close4 = h4["close"]
    e50, e200 = ind.ema(close4, 50), ind.ema(close4, 200)
    atr4 = float(ind.atr(h4).iloc[-1])
    last, f50, f200 = float(close4.iloc[-1]), float(e50.iloc[-1]), float(e200.iloc[-1])
    atr1 = float(ind.atr(h1).iloc[-1])
    if not atr4 > 0 or not atr1 > 0:
        return "no price movement to measure (flat data)"

    rising = f50 > float(e50.iloc[-6])
    separated = abs(f50 - f200) >= MIN_TREND_ATR * atr4
    if separated and last > f50 > f200 and rising:
        bias = "long"
    elif separated and last < f50 < f200 and not rising:
        bias = "short"
    else:
        return "no clear 4-hour trend"

    q = quote(spec.symbol)
    bid, ask = float(q.bid), float(q.ask)
    return _Context(
        spec=spec,
        h1=h1,
        bias=bias,
        trend_strength=abs(f50 - f200) / atr4,
        atr1=atr1,
        ema50_h1=float(ind.ema(h1["close"], 50).iloc[-1]),
        rsi_h1=float(ind.rsi(h1["close"]).iloc[-1]),
        mid=(bid + ask) / 2,
        spread=max(ask - bid, 0.0),
    )


def _setups(c: _Context, min_rr: float, expires: datetime,
            stop_atr: float = STOP_BUFFER_ATR) -> list[Setup]:
    """Every qualifying limit entry for the context's trend direction."""
    long = c.bias == "long"
    sign = 1 if long else -1
    recent = c.h1.iloc[-STRUCTURE_BARS:]
    levels = ind.swing_lows(recent) if long else ind.swing_highs(recent)
    opposing = ind.swing_highs(recent) if long else ind.swing_lows(recent)
    tolerance = 0.25 * c.atr1

    candidates = [(lvl, "swing") for lvl in levels]
    candidates.append((c.ema50_h1, "ema"))

    out = []
    for level, kind in candidates:
        distance = (c.mid - level) * sign            # how far price must pull back
        if not (MIN_ENTRY_ATR * c.atr1 <= distance <= MAX_ENTRY_ATR * c.atr1):
            continue
        entry = level + sign * ENTRY_BUFFER_ATR * c.atr1
        stop = level - sign * max(stop_atr * c.atr1, MIN_STOP_SPREADS * c.spread)
        risk = abs(entry - stop)

        target, target_kind = _target(c, entry, risk, opposing, min_rr, sign)
        if target is None:
            continue
        reward = abs(target - entry)
        rr = (reward - c.spread) / (risk + c.spread)
        if rr < min_rr:
            continue

        visits = ind.touches(recent, level, tolerance)
        # Two independent reasons for price to stop at the same place. An EMA
        # entry lines up with itself, so it needs a swing level nearby.
        if kind == "ema":
            confluent = any(abs(lvl - level) <= 0.5 * c.atr1 for lvl in levels)
        else:
            confluent = abs(level - c.ema50_h1) <= 0.5 * c.atr1
        score, reasons = _score(c, rr, visits, confluent, distance, target_kind, kind)
        spec = c.spec
        out.append(Setup(
            symbol=spec.symbol,
            direction=c.bias,
            entry=spec.round_price(entry),
            stop=spec.round_price(stop),
            target=spec.round_price(target),
            rr=round(rr, 2),
            score=round(score, 1),
            risk_pips=round(spec.pips(risk), 1),
            reward_pips=round(spec.pips(reward), 1),
            spread_pips=round(spec.pips(c.spread), 1),
            atr_pips=round(spec.pips(c.atr1), 1),
            price=spec.round_price(c.mid),
            target_kind=target_kind,
            expires_at=expires,
            reasons=reasons,
        ))
    return out


def _target(c: _Context, entry: float, risk: float, opposing: list[float],
            min_rr: float, sign: int) -> tuple[float | None, str]:
    """Nearest opposing swing beyond price that pays ``min_rr``, else a measured move."""
    beyond = sorted((lvl for lvl in opposing if (lvl - c.mid) * sign > 0),
                    key=lambda lvl: (lvl - c.mid) * sign)
    limit = MAX_TARGET_ATR * c.atr1
    for level in beyond:
        target = level - sign * ENTRY_BUFFER_ATR * c.atr1   # exit before the crowd
        reward = (target - entry) * sign
        if reward > limit:
            break
        if (reward - c.spread) / (risk + c.spread) >= min_rr:
            return target, "structure"
    if not beyond:
        reward = MEASURED_R * risk
        if reward <= limit:
            return entry + sign * reward, "measured"
    return None, ""


def _score(c: _Context, rr: float, visits: int, confluent: bool, distance: float,
           target_kind: str, level_kind: str) -> tuple[float, list[str]]:
    """Points out of 100, with the reason for each so a trader can check the work."""
    reasons = []
    # Full marks are meant to be rare: a 6-ATR EMA spread is a strong, mature
    # trend, and six separate visits make a well-tested level. A score
    # that most setups max out cannot rank them.
    trend = min(c.trend_strength / FULL_TREND_ATR, 1.0) * 25
    reasons.append(f"4h {'up' if c.bias == 'long' else 'down'}trend, EMA spread "
                   f"{c.trend_strength:.1f}× ATR")

    level = min(visits / FULL_LEVEL_VISITS, 1.0) * 20
    what = "1h 50 EMA" if level_kind == "ema" else ("swing low" if c.bias == "long" else "swing high")
    reasons.append(f"entry at {what}, price returned to it {visits}× this week")

    ema_pts = 15 if confluent else 0
    if confluent:
        reasons.append("a swing level and the 1h 50 EMA coincide here")

    rr_pts = min(max(rr - 2, 0) / 2, 1.0) * 15
    reasons.append(f"{rr:.1f}R to a {target_kind} target after the spread")

    in_atr = distance / c.atr1
    dist_pts = 10 if 0.5 <= in_atr <= 1.5 else 5

    overextended = c.rsi_h1 > 70 if c.bias == "long" else c.rsi_h1 < 30
    balanced = 35 <= c.rsi_h1 <= 65
    rsi_pts = 0 if overextended else (10 if balanced else 5)
    if overextended:
        reasons.append(f"1h RSI {c.rsi_h1:.0f}: stretched, a deeper pullback is possible")

    target_pts = 5 if target_kind == "structure" else 0

    spread_ratio = c.spread / c.atr1 if c.atr1 else 0
    penalty = 10 if spread_ratio > 0.15 else 0
    if penalty:
        reasons.append(f"wide spread: {spread_ratio:.0%} of the hourly range")

    total = trend + level + ema_pts + rr_pts + dist_pts + rsi_pts + target_pts - penalty
    return max(total, 0.0), reasons


def _exposure(setup: Setup) -> list[tuple[str, int]]:
    """Currencies a trade is long (+1) and short (−1)."""
    sign = 1 if setup.direction == "long" else -1
    return [(setup.symbol[:3], sign), (setup.symbol[3:], -sign)]


def _rank(setups: list[Setup], top: int, max_per_currency: int) -> list[Setup]:
    """Best first, without stacking the same currency bet more than the cap.

    Long EURUSD, long GBPUSD and short USDJPY are three ways of selling the
    dollar; taking all three triples one idea's risk.
    """
    chosen: list[Setup] = []
    used: Counter = Counter()
    for setup in sorted(setups, key=lambda s: s.score, reverse=True):
        exposure = _exposure(setup)
        if any(used[e] >= max_per_currency for e in exposure):
            continue
        chosen.append(setup)
        used.update(exposure)
        if len(chosen) >= top:
            break
    return chosen
