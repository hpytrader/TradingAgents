"""The Asian-range breakout: trade the London open's break of the night's range, with it.

The SMC model fades a sweep of liquidity and bets on the reversal. Quinn's lab
found that this loses on both the 1h/5m and 4h/15m charts, and that many of
its "stop hunts" were the start of the day's real move. This model takes the
other side of that idea. For a long (shorts mirror it):

1. **Range.** The high and low of 00:00-06:00 UTC, the Asian session, which
   ends about when London opens (02:00 New York).
2. **A range worth trading.** Between ``MIN_WIDTH_ATR`` and ``MAX_WIDTH_ATR``
   hourly ATRs: a tighter one is noise, a wider one has spent the day's move.
3. **Breakout.** The day's first five-minute close outside the range, from
   06:00 UTC, above the high. A wick is not enough, and only the first side
   broken is traded that day.
4. **Entry.** A buy limit at the broken high: the retest of the level, so the
   order joins the move on a pullback rather than chasing the breakout bar.
   If price is already back through the broken level, the break failed: no order.
5. **Stop** at the middle of the range. **Target** at ``min_rr`` after the
   spread (2R by default).

The breakout must be recent (``MAX_BREAKOUT_AGE``). The desk's rules hold as
for every model: setups only in the scan window, cancelled at the cancel time,
flat by 16:55 New York.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, time, timedelta

import pandas as pd

from tradingagents.fx import smc
from tradingagents.fx.instruments import DEFAULT_UNIVERSE, spec_for
from tradingagents.fx.scanner import CandleFetcher, QuoteFetcher, ScanResult, Setup, finish
from tradingagents.fx.smc_scanner import (
    LATEST_CANCEL,
    ScanWindow,
    cancel_deadline,
    outside_window_reason,
    phase,
)

RANGE_START, RANGE_END = time(0, 0), time(6, 0)      # UTC: the Asian session
MIN_WIDTH_ATR = 1.0
MAX_WIDTH_ATR = 5.0
MAX_BREAKOUT_AGE = timedelta(hours=3)
MAX_MARKET_AGE = timedelta(minutes=15)                # a market entry joins only a fresh break
STALE_AFTER = timedelta(minutes=30)
M5_BARS, H1_BARS = 600, 300


def asian_range(m5: pd.DataFrame, now: datetime) -> tuple[float, float, pd.Timestamp] | None:
    """Today's (UTC) Asian high, low and the range's end, once the range has closed."""
    day = pd.Timestamp(now).tz_convert(UTC).normalize()
    start, end = day + pd.Timedelta(hours=RANGE_START.hour), day + pd.Timedelta(hours=RANGE_END.hour)
    if pd.Timestamp(now) < end:
        return None
    bars = m5[(m5.index >= start) & (m5.index < end)]
    if len(bars) < 36:                                # most of the six hours must be there
        return None
    return float(bars["high"].max()), float(bars["low"].min()), end


def scan_asian(
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
    timeframes=None,                                  # accepted for the lab's sake; this model is M5 + H1
    entry_mode: str = "retest",                       # "retest": limit at the broken level; "market": join now
) -> ScanResult:
    """Every instrument whose Asian range broke this morning, as a retest limit order."""
    now = now or datetime.now(UTC)
    window = window or ScanWindow()
    if window.current_end(now) is None and not any_session:
        return ScanResult(scanned_at=now, setups=[], skipped=[("ALL", outside_window_reason(window, now))],
                          ran=False)
    expires = min(now + timedelta(hours=valid_hours), cancel_deadline(now, latest_cancel))
    best, skipped = [], []
    for symbol in symbols:
        try:
            found = _setup(symbol, candles, quote, now, min_rr, expires, entry_mode)
        except Exception as exc:  # one pair's data problem must not end the scan
            skipped.append((symbol, f"data unavailable: {exc}"))
            continue
        if isinstance(found, str):
            skipped.append((symbol, found))
        else:
            best.append(found)
    return finish(best, skipped, now, top, max_per_currency)


def _setup(symbol: str, candles: CandleFetcher, quote: QuoteFetcher, now: datetime, min_rr: float,
           expires: datetime, entry_mode: str = "retest") -> Setup | str:
    spec = spec_for(symbol)
    m5 = candles(symbol, "M5", M5_BARS)
    h1 = candles(symbol, "H1", H1_BARS)
    if len(m5) < 120 or len(h1) < 30:
        return "not enough price history"
    if now - m5.index[-1].to_pydatetime() > STALE_AFTER:
        return "no recent prices (market closed?)"
    rng = asian_range(m5, now)
    if rng is None:
        return "the Asian range is not complete yet"
    high, low, ended = rng
    width = high - low
    atr1 = smc.atr(h1)
    if not atr1 > 0:
        return "no price movement to measure (flat data)"
    if not MIN_WIDTH_ATR <= width / atr1 <= MAX_WIDTH_ATR:
        return f"Asian range {spec.pips(width):.1f} pips is {width / atr1:.1f}x the hourly ATR, outside " \
               f"{MIN_WIDTH_ATR:g}-{MAX_WIDTH_ATR:g}x"

    after = m5[m5.index >= ended]
    above, below = after["close"] > high, after["close"] < low
    broke = above | below
    if not broke.any():
        return "no five-minute close outside the Asian range yet"
    first = broke.idxmax()
    long = bool(above.loc[first])
    bar_end = first.to_pydatetime() + timedelta(minutes=5)
    if now - bar_end > MAX_BREAKOUT_AGE:
        return f"the breakout at {first:%H:%M} UTC is more than {MAX_BREAKOUT_AGE.seconds // 3600}h old"
    market = entry_mode == "market"
    if market and now - bar_end > MAX_MARKET_AGE:
        return f"the breakout at {first:%H:%M} UTC is too old to join at market"

    q = quote(symbol)
    bid, ask = float(q.bid), float(q.ask)
    mid, spread = (bid + ask) / 2, max(ask - bid, 0.0)
    sign = 1 if long else -1
    edge = high if long else low
    entry = mid if market else edge
    if (mid - edge) * sign <= 0:
        return (f"price is back through the broken Asian {'high' if long else 'low'}: "
                f"the {'upside' if long else 'downside'} break failed")
    stop = (high + low) / 2
    risk = abs(entry - stop)
    target = entry + sign * (min_rr * (risk + spread) + spread)
    since = m5[m5.index >= first]
    best = since["high"].max() if long else since["low"].min()
    if (best - target) * sign >= 0:
        return "the breakout already reached the target"
    reward = abs(target - entry)
    rr = (reward - spread) / (risk + spread)

    strength = abs(float(m5.loc[first, "close"]) - edge) / atr1
    compression = width / atr1
    breakout_ny = first.tz_convert(smc.NEW_YORK)
    score = (40 + min(strength / 0.5, 1.0) * 20 + max(0.0, (5 - compression) / 4) * 20
             + (20 if breakout_ny.hour < 5 else 10 if breakout_ny.hour < 9 else 0)
             - (10 if spread > 0.15 * width else 0))
    direction = "above the Asian high" if long else "below the Asian low"
    reasons = [
        f"Asian range {spec.round_price(low)}-{spec.round_price(high)} ({spec.pips(width):.1f} pips, "
        f"{compression:.1f}x the hourly ATR)",
        f"first five-minute close {direction} at {first:%H:%M} UTC ({strength:.2f}x ATR beyond it)",
        (f"{'buy' if long else 'sell'} at market {spec.round_price(entry)}, joining the break"
         if market else f"{'buy' if long else 'sell'} limit on the retest of {spec.round_price(entry)}")
        + f"; stop at the range middle {spec.round_price(stop)}",
        f"{rr:.1f}R target after the spread",
        f"{phase(now)} session; cancel at {expires:%H:%M} UTC if unfilled",
    ]
    return Setup(
        symbol=spec.symbol, direction="long" if long else "short",
        entry=spec.round_price(entry), stop=spec.round_price(stop), target=spec.round_price(target),
        rr=round(rr, 2), score=round(score, 1), risk_pips=round(spec.pips(risk), 1),
        reward_pips=round(spec.pips(reward), 1), spread_pips=round(spec.pips(spread), 1),
        atr_pips=round(spec.pips(atr1), 1), price=spec.round_price(mid), target_kind="range projection",
        expires_at=expires, reasons=reasons, strategy="asian_breakout_market" if market else "asian_breakout",
        zone_low=spec.round_price(low), zone_high=spec.round_price(high), invalidation=spec.round_price(stop),
        features={"range_atr": round(compression, 2), "breakout_strength": round(strength, 2),
                  "breakout_hour": breakout_ny.hour, "wide_spread": spread > 0.15 * width,
                  "pool": "Asian range", "pool_quality": 2, "shift": "breakout", "h1_break": "none",
                  "zone": "range edge", "displacement": round(strength, 2), "depth": 0.0,
                  "order": "market" if market else "limit"},
    )
