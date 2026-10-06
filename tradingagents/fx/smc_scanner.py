"""The SMC strategy: H1 bias, M5 sweep → CHoCH → order block / FVG entry.

For a long (shorts mirror it):

1. **Bias.** The latest break of structure on the 1-hour chart is up.
2. **Sweep.** Within the last few hours a 5-minute bar traded below a pool of
   sell-side liquidity (previous day low, the Asian or London low, equal lows)
   and closed back above it.
3. **Change of character.** After the sweep, a 5-minute close breaks above the
   last swing high: the market has turned with the bias.
4. **Entry.** A buy limit at the middle of the unmitigated fair value gap that
   displacement left (preferring one inside the order block), or of the order
   block when there is no gap.
5. **Stop** beyond the sweep's wick: if price goes back below it, the idea was
   wrong. **Target** at the nearest buy-side liquidity above price (previous day
   high, session high, equal highs, an intact 1-hour swing high) that pays at
   least ``min_rr`` after the spread.
6. **Sessions.** Setups are only built during the London and New York sessions,
   and an order is cancelled when its session ends.

The score ranks setups by the quality of the swept pool, displacement, OB/FVG
confluence, entry depth, freshness and reward-to-risk. It is not a probability.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from tradingagents.fx import smc
from tradingagents.fx.instruments import DEFAULT_UNIVERSE, InstrumentSpec, spec_for
from tradingagents.fx.scanner import CandleFetcher, QuoteFetcher, ScanResult, Setup, finish

H1_BARS = 300
M5_BARS = 600
SWEEP_LOOKBACK = 72        # M5 bars (6 hours) searched for a sweep
CHOCH_WITHIN = 36          # M5 bars (3 hours) after the sweep for the shift to come
MAX_TARGET_H1_ATR = 3.0    # beyond this an intraday target is a stretch
STALE_AFTER = timedelta(minutes=30)


@dataclass(frozen=True)
class Session:
    name: str
    tz: object
    start: time
    end: time

    def window(self, now: datetime) -> tuple[datetime, datetime] | None:
        """This session's start and end today if ``now`` is inside it."""
        local = now.astimezone(self.tz)
        start = self.tz.localize(datetime.combine(local.date(), self.start))
        end = self.tz.localize(datetime.combine(local.date(), self.end))
        if start <= local < end:
            return start.astimezone(UTC), end.astimezone(UTC)
        return None


SESSIONS = {
    "london": Session("London", smc.LONDON, time(7, 0), time(11, 0)),
    "newyork": Session("New York", smc.NEW_YORK, time(8, 0), time(12, 0)),
}


def active_session(now: datetime, names: Iterable[str] = ("london", "newyork")) -> tuple[Session, datetime] | None:
    """The session ``now`` falls in and when it ends, or ``None``."""
    for name in names:
        window = SESSIONS[name].window(now)
        if window:
            return SESSIONS[name], window[1]
    return None


def scan_smc(
    candles: CandleFetcher,
    quote: QuoteFetcher,
    symbols: Iterable[str] = DEFAULT_UNIVERSE,
    *,
    min_rr: float = 2.0,
    top: int = 10,
    max_per_currency: int = 2,
    valid_hours: float = 4.0,
    any_session: bool = False,
    now: datetime | None = None,
) -> ScanResult:
    """Scan ``symbols`` for SMC setups and return the best per instrument, ranked."""
    now = now or datetime.now(UTC)
    current = active_session(now)
    if current is None and not any_session:
        local = now.astimezone(smc.NEW_YORK)
        return ScanResult(scanned_at=now, setups=[], skipped=[(
            "ALL", f"outside the London (07:00–11:00 London) and New York (08:00–12:00 New York) "
                   f"sessions; it is {local:%H:%M} in New York. Use --any-session to scan anyway.")])
    session_end = current[1] if current else now + timedelta(hours=valid_hours)
    expires = min(session_end, now + timedelta(hours=valid_hours))
    session_name = current[0].name if current else "off-session"

    best: list[Setup] = []
    skipped: list[tuple[str, str]] = []
    for symbol in symbols:
        try:
            spec = spec_for(symbol)
        except ValueError as exc:
            skipped.append((symbol, str(exc)))
            continue
        try:
            found = _setup(spec, candles, quote, now, min_rr, expires, session_name)
        except Exception as exc:  # one pair's data problem must not end the scan
            skipped.append((spec.symbol, f"data unavailable: {exc}"))
            continue
        if isinstance(found, str):
            skipped.append((spec.symbol, found))
        else:
            best.append(found)
    return finish(best, skipped, now, top, max_per_currency)


def _setup(spec: InstrumentSpec, candles: CandleFetcher, quote: QuoteFetcher, now: datetime,
           min_rr: float, expires: datetime, session_name: str) -> Setup | str:
    h1 = candles(spec.symbol, "H1", H1_BARS)
    m5 = candles(spec.symbol, "M5", M5_BARS)
    if len(h1) < 60 or len(m5) < 120:
        return "not enough price history"
    if now - m5.index[-1].to_pydatetime() > STALE_AFTER:
        return "no recent prices (market closed?)"

    direction_bias, h1_break = smc.bias(h1)
    if direction_bias is None:
        return "no clear 1-hour structure"
    long = direction_bias == "long"
    up = "up" if long else "down"
    atr5, atr1 = smc.atr(m5), smc.atr(h1)
    if not atr5 > 0 or not atr1 > 0:
        return "no price movement to measure (flat data)"

    q = quote(spec.symbol)
    bid, ask = float(q.bid), float(q.ask)
    mid, spread = (bid + ask) / 2, max(ask - bid, 0.0)

    liquidity = smc.pools(m5, now, atr5)
    since = len(m5) - SWEEP_LOOKBACK
    taken = smc.sweeps(m5, liquidity, "low" if long else "high", since)
    if not taken:
        side = "sell-side (lows)" if long else "buy-side (highs)"
        return f"1h bias {direction_bias}, but no {side} liquidity swept in the last 6 hours"

    m5_breaks = smc.breaks(m5)
    candidates: list[Setup] = []
    reason = f"liquidity swept, but no 5-minute change of character {up} followed"
    for sweep in sorted(taken, key=lambda s: s.i, reverse=True):
        shift = next((b for b in m5_breaks
                      if b.direction == up and sweep.i < b.i <= sweep.i + CHOCH_WITHIN), None)
        if shift is None:
            continue
        built = _from_sweep(spec, m5, h1, sweep, shift, liquidity, long, mid, spread,
                            atr5, atr1, min_rr, h1_break, now, expires, session_name)
        if isinstance(built, Setup):
            candidates.append(built)
        else:
            reason = built
    if candidates:
        return max(candidates, key=lambda s: s.score)
    return reason


def _from_sweep(spec, m5, h1, sweep, shift, liquidity, long, mid, spread,
                atr5, atr1, min_rr, h1_break, now, expires, session_name) -> Setup | str:
    up = "up" if long else "down"
    sign = 1 if long else -1
    ob = smc.order_block(m5, sweep.i, shift.i, up)
    gaps = smc.fair_value_gaps(m5, sweep.i, shift.i, up)
    zone, confluence = None, ""
    if gaps:
        inside = [g for g in gaps if ob and g.overlaps(ob)]
        if inside:
            zone, confluence = inside[-1], "fair value gap inside the order block"
        else:
            zone, confluence = gaps[-1], "fair value gap"
    elif ob:
        zone, confluence = ob, "order block"
    if zone is None:
        return "change of character without an order block or fair value gap to enter at"

    entry = zone.mid
    if smc.mitigated(m5, zone, entry, up):
        return f"the {zone.kind} has already been traded back into"
    if (mid - entry) * sign <= spread:
        return f"price is already at the {zone.kind}; a limit would fill at market"

    buffer = max(3 * spread, 0.3 * atr5)
    stop = sweep.extreme - sign * buffer
    risk = abs(entry - stop)

    target, target_name = _target(m5, h1, liquidity, long, mid, entry, risk, spread, min_rr, atr1, atr5)
    if target is None:
        return target_name
    reward = abs(target - entry)
    rr = (reward - spread) / (risk + spread)

    shift_close = float(m5["close"].iat[shift.i])
    leg = abs(shift_close - sweep.extreme)
    displacement = leg / atr5
    depth = abs(shift_close - entry) / leg if leg else 0.0          # 0 = top of leg, 1 = sweep
    minutes = (now - m5.index[shift.i].to_pydatetime()).total_seconds() / 60

    score, reasons = _score(spec, sweep, shift, h1_break, displacement, confluence, zone, depth,
                            rr, minutes, spread, atr5, target_name, long, m5)
    return Setup(
        symbol=spec.symbol,
        direction="long" if long else "short",
        entry=spec.round_price(entry),
        stop=spec.round_price(stop),
        target=spec.round_price(target),
        rr=round(rr, 2),
        score=round(score, 1),
        risk_pips=round(spec.pips(risk), 1),
        reward_pips=round(spec.pips(reward), 1),
        spread_pips=round(spec.pips(spread), 1),
        atr_pips=round(spec.pips(atr1), 1),
        price=spec.round_price(mid),
        target_kind=target_name,
        expires_at=expires,
        reasons=reasons + [f"{session_name} session; cancel at {expires:%H:%M} UTC if unfilled"],
        strategy="smc",
        zone_low=spec.round_price(zone.low),
        zone_high=spec.round_price(zone.high),
        invalidation=spec.round_price(sweep.extreme),
    )


def _target(m5, h1, liquidity, long, mid, entry, risk, spread, min_rr, atr1, atr5):
    """Nearest opposing liquidity beyond price paying ``min_rr``: (price, name) or (None, why)."""
    sign = 1 if long else -1
    side = "high" if long else "low"
    options = [(p.level, p.name) for p in liquidity if p.side == side]
    h1_points = [s for s in smc.swings(h1, 3) if s.kind == side]
    closes = h1["close"].to_numpy()
    for s in h1_points:
        broken = (closes[s.i + 1:] > s.price).any() if long else (closes[s.i + 1:] < s.price).any()
        if not broken:
            options.append((s.price, "1h swing high" if long else "1h swing low"))
    beyond = sorted(((lvl, name) for lvl, name in options if (lvl - mid) * sign > 0),
                    key=lambda x: (x[0] - mid) * sign)
    if not beyond:
        return None, "no opposing liquidity left beyond price to target"
    limit = MAX_TARGET_H1_ATR * atr1
    for level, name in beyond:
        target = level - sign * 0.1 * atr5           # exit just before the pool
        reward = (target - entry) * sign
        if reward > limit:
            break
        if (reward - spread) / (risk + spread) >= min_rr:
            return target, name
    return None, f"nearest opposing liquidity ({beyond[0][1]}) is under {min_rr:g}R away"


def _score(spec, sweep, shift, h1_break, displacement, confluence, zone, depth, rr, minutes,
           spread, atr5, target_name, long, m5):
    reasons = []
    when = m5.index[sweep.i].strftime("%H:%M")
    pool_pts = {3: 25, 2: 20, 1: 15}[sweep.pool.quality]
    reasons.append(f"swept the {sweep.pool.name} ({spec.round_price(sweep.pool.level)}) at {when} UTC "
                   f"and closed back {'above' if long else 'below'}")

    disp_pts = min(displacement / 3, 1.0) * 20
    reasons.append(f"5m structure shift ({shift.kind}) {'up' if long else 'down'} at "
                   f"{m5.index[shift.i]:%H:%M} UTC through {spec.round_price(shift.level)}, "
                   f"displacement {displacement:.1f}× ATR")

    conf_pts = {"fair value gap inside the order block": 15, "fair value gap": 10, "order block": 5}[confluence]
    reasons.append(f"entry at the middle of the {confluence} "
                   f"{spec.round_price(zone.low)}–{spec.round_price(zone.high)}, unmitigated")

    depth_pts = 10 if depth >= 0.5 else 5 if depth >= 0.3 else 0
    reasons.append(f"entry {depth:.0%} of the way back down the displacement leg"
                   + (" (discount)" if long and depth >= 0.5 else " (premium)" if not long and depth >= 0.5 else ""))

    h1_pts = 10 if h1_break.kind == "BOS" else 5
    reasons.append(f"1h {h1_break.kind} {h1_break.direction} at {spec.round_price(h1_break.level)} sets the bias")

    rr_pts = min(max(rr - 2, 0) / 2, 1.0) * 10
    reasons.append(f"{rr:.1f}R to the {target_name} after the spread")

    fresh_pts = 10 if minutes <= 60 else 5 if minutes <= 180 else 0

    penalty = 10 if spread > 0.3 * atr5 else 0
    if penalty:
        reasons.append(f"wide spread: {spread / atr5:.0%} of the 5-minute range")
    total = pool_pts + disp_pts + conf_pts + depth_pts + h1_pts + rr_pts + fresh_pts - penalty
    return max(total, 0.0), reasons
