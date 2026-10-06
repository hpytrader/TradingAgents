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
6. **Window.** Setups are only built inside the scan window, 02:00–12:00 New
   York (and Toronto) time by default: the London session and the London–New
   York overlap. An unfilled order is cancelled after four hours or when the
   window closes, whichever is first. Weekends are always outside.

The score ranks setups by the quality of the swept pool, displacement, OB/FVG
confluence, entry depth, freshness and reward-to-risk. It is not a probability.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from tradingagents.fx import smc
from tradingagents.fx.instruments import DEFAULT_UNIVERSE, InstrumentSpec, spec_for
from tradingagents.fx.scanner import CandleFetcher, QuoteFetcher, ScanResult, Setup, finish

H1_BARS = 300
M5_BARS = 600
SWEEP_LOOKBACK = 72        # M5 bars (6 hours) searched for a sweep
CHOCH_WITHIN = 36          # M5 bars (3 hours) after the sweep for the shift to come
MAX_TARGET_H1_ATR = 3.0    # beyond this an intraday target is a stretch
MIN_RISK_H1_ATR = 0.35     # a stop closer than this (in 1-hour ATRs) is inside normal noise
MAX_SHIFT_AGE = timedelta(hours=2)   # an older structure shift has usually played out
MAX_ENTRY_H1_ATR = 2.0     # a limit farther than this from price is unlikely to fill this session
STALE_AFTER = timedelta(minutes=30)


@dataclass(frozen=True)
class ScanWindow:
    """The hours setups are built in, on New York time (also Toronto time).

    The default, 02:00–12:00, runs from the London open to New York's midday,
    covering the London session and the London–New York overlap. A window may
    cross midnight (``19:00-12:00``). Weekends, when the forex market is shut
    from Friday 17:00 to Sunday 17:00 New York, are outside every window.
    """

    start: time = time(2, 0)
    end: time = time(12, 0)

    @classmethod
    def parse(cls, text: str) -> ScanWindow:
        """``"02:00-12:00"`` → ScanWindow; ``ValueError`` for anything else."""
        try:
            a, b = (part.strip() for part in text.replace("–", "-").split("-"))
            start, end = time.fromisoformat(a), time.fromisoformat(b)
        except ValueError:
            raise ValueError(f"window must look like 02:00-12:00, not {text!r}") from None
        if start == end:
            raise ValueError("the window's start and end must differ")
        return cls(start, end)

    def label(self) -> str:
        return f"{self.start:%H:%M}–{self.end:%H:%M} New York time"

    def _bounds(self, day: date) -> tuple[datetime, datetime]:
        tz = smc.NEW_YORK
        opens = tz.localize(datetime.combine(day, self.start))
        closes_day = day if self.end > self.start else day + timedelta(days=1)
        closes = tz.localize(datetime.combine(closes_day, self.end))
        return opens, closes

    def current_end(self, now: datetime) -> datetime | None:
        """When the window ``now`` is in closes (UTC), or ``None`` if outside it."""
        if not market_open(now):
            return None
        local = now.astimezone(smc.NEW_YORK)
        for day in (local.date() - timedelta(days=1), local.date()):
            opens, closes = self._bounds(day)
            if opens <= local < closes:
                return min(closes, _weekend_starts(local)).astimezone(UTC)
        return None

    def next_open(self, now: datetime) -> datetime:
        """The next time a window opens with the market open (UTC)."""
        local = now.astimezone(smc.NEW_YORK)
        for offset in range(0, 9):
            opens, _ = self._bounds(local.date() + timedelta(days=offset))
            if opens > local and market_open(opens):
                return opens.astimezone(UTC)
        raise RuntimeError("no window opens in the next week")  # pragma: no cover


LATEST_CANCEL = time(14, 0)


def cancel_deadline(now: datetime, latest: time = LATEST_CANCEL) -> datetime:
    """The next ``latest`` on New York clocks, and never past the Friday 17:00 close (UTC)."""
    local = now.astimezone(smc.NEW_YORK)
    deadline = smc.NEW_YORK.localize(datetime.combine(local.date(), latest))
    if deadline <= local:
        deadline = smc.NEW_YORK.localize(datetime.combine(local.date() + timedelta(days=1), latest))
    return min(deadline, _weekend_starts(local)).astimezone(UTC)


def market_open(now: datetime) -> bool:
    """Whether spot forex trades at ``now``: shut Friday 17:00 to Sunday 17:00 New York."""
    local = now.astimezone(smc.NEW_YORK)
    weekday, t = local.weekday(), local.time()
    if weekday == 5:
        return False
    if weekday == 4 and t >= time(17, 0):
        return False
    return not (weekday == 6 and t < time(17, 0))


def _weekend_starts(local: datetime) -> datetime:
    friday = local.date() + timedelta(days=(4 - local.weekday()) % 7)
    return smc.NEW_YORK.localize(datetime.combine(friday, time(17, 0)))


def phase(now: datetime) -> str:
    """Which session's hours ``now`` falls in, for the setup's reasons."""
    t = now.astimezone(smc.NEW_YORK).time()
    if time(2, 0) <= t < time(8, 0):
        return "London"
    if time(8, 0) <= t < time(12, 0):
        return "New York"
    if time(12, 0) <= t < time(17, 0):
        return "New York afternoon"
    return "Asian"


def _until(later: datetime, now: datetime) -> str:
    minutes = int((later - now).total_seconds() // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def outside_window_reason(window: ScanWindow, now: datetime) -> str:
    local = now.astimezone(smc.NEW_YORK)
    opens = window.next_open(now)
    when = f"{opens.astimezone(smc.NEW_YORK):%a %H:%M} (in {_until(opens, now)})"
    if not market_open(now):
        return (f"the forex market is closed for the weekend (Friday 17:00 to Sunday 17:00 New York). "
                f"The scan window next opens {when}.")
    return (f"outside the scan window ({window.label()}); it is {local:%H:%M} in New York. "
            f"Next window opens {when}. Use --any-session to scan anyway.")


def scan_smc(
    candles: CandleFetcher,
    quote: QuoteFetcher,
    symbols: Iterable[str] = DEFAULT_UNIVERSE,
    *,
    min_rr: float = 2.0,
    top: int = 10,
    max_per_currency: int = 2,
    valid_hours: float = 4.0,
    window: ScanWindow | None = None,
    any_session: bool = False,
    latest_cancel: time = LATEST_CANCEL,
    now: datetime | None = None,
) -> ScanResult:
    """Scan ``symbols`` for SMC setups and return the best per instrument, ranked.

    The window decides when to look; an order found in it keeps ``valid_hours``
    to fill, but is cancelled by ``latest_cancel`` New York time at the latest
    (14:00 by default, leaving a fill three hours before the 17:00 close) and
    never past the weekend close. A setup found at 11:40 therefore gets until
    14:00, not 20 minutes.
    """
    now = now or datetime.now(UTC)
    window = window or ScanWindow()
    closes = window.current_end(now)
    if closes is None and not any_session:
        return ScanResult(scanned_at=now, setups=[], skipped=[("ALL", outside_window_reason(window, now))],
                          ran=False)
    expires = min(now + timedelta(hours=valid_hours), cancel_deadline(now, latest_cancel))
    session_name = phase(now)

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
    age = now - m5.index[shift.i].to_pydatetime()
    if age > MAX_SHIFT_AGE:
        hours = age.total_seconds() / 3600
        return (f"the structure shift at {m5.index[shift.i]:%H:%M} UTC is {hours:.1f}h old; "
                "the move has most likely played out")
    # The idea is wrong if price goes beyond the extreme of the whole move from
    # the sweep to the shift, not only the sweeping candle's wick: price can
    # keep running a few candles after taking the liquidity before it turns.
    move = m5.iloc[sweep.i:shift.i + 1]
    extreme = float(move["low"].min()) if long else float(move["high"].max())
    shift_close = float(m5["close"].iat[shift.i])
    ob = smc.order_block(m5, sweep.i, shift.i, up)
    gaps = smc.fair_value_gaps(m5, sweep.i, shift.i, up)
    # Zones in order of preference: a gap inside the order block, other gaps
    # (newest first), then the order block itself. The first one price has
    # not come back to is the entry; a zone already traded into would have
    # filled the limit already.
    options = [(g, "fair value gap inside the order block") for g in reversed(gaps) if ob and g.overlaps(ob)]
    options += [(g, "fair value gap") for g in reversed(gaps) if not (ob and g.overlaps(ob))]
    if ob:
        options.append((ob, "order block"))
    if not options:
        return "structure shift without an order block or fair value gap to enter at"

    zone = confluence = None
    beyond_extreme = False
    for candidate, label in options:
        if (candidate.mid - extreme) * sign <= 0:      # at or past the invalidation: not an entry
            beyond_extreme = True
            continue
        if smc.mitigated(m5, candidate, candidate.mid, up, after=shift.i):
            continue
        if (mid - candidate.mid) * sign <= spread:
            continue
        zone, confluence = candidate, label
        break
    if zone is None:
        if beyond_extreme and all((z.mid - extreme) * sign <= 0 for z, _ in options):
            return "the only order block / fair value gap sits beyond the low of the move, past invalidation"
        kinds = " and ".join(sorted({z.kind for z, _ in options}))
        return f"every {kinds} from the move has already been traded back into"

    entry = zone.mid
    if abs(mid - entry) > MAX_ENTRY_H1_ATR * atr1:
        return (f"the {zone.kind} is {abs(mid - entry) / atr1:.1f}× the hourly range from price; "
                "a limit there is unlikely to fill this session")
    buffer = max(3 * spread, 0.3 * atr5)
    stop = extreme - sign * buffer
    widened = False
    if abs(entry - stop) < MIN_RISK_H1_ATR * atr1:
        # In a quiet market the sweep wick can sit almost on the entry; a stop
        # a pip away is noise, not invalidation. Keep it beyond the sweep, but
        # never closer than a fraction of the hourly range.
        stop = entry - sign * MIN_RISK_H1_ATR * atr1
        widened = True
    risk = abs(entry - stop)

    target, target_name = _target(m5, h1, liquidity, long, mid, entry, risk, spread, min_rr, atr1, atr5,
                                  after_i=shift.i)
    if target is None:
        return target_name
    reward = abs(target - entry)
    rr = (reward - spread) / (risk + spread)

    leg = abs(shift_close - extreme)
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
        reasons=reasons
        + ([f"stop widened to the minimum {MIN_RISK_H1_ATR}× 1h ATR "
            f"({spec.pips(MIN_RISK_H1_ATR * atr1):.1f} pips); the sweep wick sat too close to the entry"]
           if widened else [])
        + [f"{session_name} session; cancel at {expires:%H:%M} UTC if unfilled"],
        strategy="smc",
        zone_low=spec.round_price(zone.low),
        zone_high=spec.round_price(zone.high),
        invalidation=spec.round_price(extreme),
    )


def _target(m5, h1, liquidity, long, mid, entry, risk, spread, min_rr, atr1, atr5, after_i=None):
    """Nearest opposing liquidity beyond price paying ``min_rr``: (price, name) or (None, why).

    Liquidity price has already traded through since the structure shift
    (``after_i``) is spent: the move took it, so it is no target for an order
    placed now, even when price has since pulled back below it.
    """
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
    if after_i is not None and beyond:
        moved = m5.iloc[after_i:]
        reach = float(moved["high"].max()) if long else float(moved["low"].min())
        fresh = [(lvl, name) for lvl, name in beyond if (lvl - reach) * sign > 0]
        if not fresh:
            return None, (f"price already ran to {reach:g} after the shift, taking the "
                          f"{beyond[0][1]}; the move has happened")
        beyond = fresh
    if not beyond:
        return None, "no opposing liquidity left beyond price to target"
    limit = MAX_TARGET_H1_ATR * atr1
    for level, name in beyond:
        target = level - sign * 0.1 * atr5           # exit just before the pool
        if (target - mid) * sign <= spread:
            continue        # the pool is so close that the exit would sit behind price
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
    pool_pts = {3: 25, 2: 20, 1: 15, 0: 10}[sweep.pool.quality]
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
