"""Quinn's lab: replay the desk's rules over past prices to measure their edge.

The baseline replays today's SMC scanner, unchanged, over months of OANDA
history and trades every setup it would have handed the desk, under the same
rules as the live journal:

- scans every ``step`` minutes inside the 02:00-12:00 New York window;
- one order per symbol and side at a time; a setup already handed over that
  day is not handed over again; at most ``final`` new orders per scan;
- each order is a limit at the scanner's levels, cancelled at its cancel time
  (14:00 New York at the latest), settled bar by bar by the journal's own
  simulator: the same fill, stop, target, missed and 16:55-close rules, with a
  typical spread charged on every win.

There are no agents in the baseline: it measures what the rules alone are
worth, which is the bar every new idea and the agents' filtering must beat.

No look-ahead: at each scan the feed hands the scanner only bars that had
closed by then, and the simulator starts each order at the bar after it was
placed. Settling runs on five-minute bars (the live journal uses one-minute
ones), so a bar that touches the stop and the target counts as the stop more
often: if anything, the baseline is pessimistic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.fx import journal as jr
from tradingagents.fx.instruments import spec_for
from tradingagents.fx.journal import Entry
from tradingagents.fx.scanner import Setup
from tradingagents.fx.smc import NEW_YORK
from tradingagents.fx.smc_scanner import LATEST_CANCEL, ScanWindow, market_open, scan_smc
from tradingagents.fx.watch import setup_key

# Typical OANDA spreads in pips during London and New York hours, a little on
# the wide side. Every win is charged its spread, as in the live journal.
TYPICAL_SPREAD_PIPS = {
    "EURUSD": 1.2, "GBPUSD": 1.6, "USDJPY": 1.4, "USDCHF": 1.8, "AUDUSD": 1.4, "USDCAD": 2.0,
    "NZDUSD": 2.0, "EURJPY": 2.0, "GBPJPY": 3.0, "EURGBP": 1.6, "AUDJPY": 2.2, "EURAUD": 2.5,
    "XAUUSD": 3.5, "XAGUSD": 2.5,
}
BAR_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D": 86400}
GRANULARITIES = ("M5", "H1")
WARMUP = timedelta(days=30)          # the scanner reads 300 hourly bars back from its first scan

Fetch = Callable[[str, str, datetime, datetime], pd.DataFrame]


def lab_dir() -> Path:
    from tradingagents.default_config import DEFAULT_CONFIG
    return Path(DEFAULT_CONFIG["results_dir"]).parent / "lab"


# ---------------------------------------------------------------------------
# Price history, cached on disk
# ---------------------------------------------------------------------------

class History:
    """Mid-price bars per symbol and granularity, kept in ``folder`` and topped up on demand."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str, granularity: str) -> Path:
        return self.folder / f"{symbol}_{granularity}.csv.gz"

    def load(self, symbol: str, granularity: str) -> pd.DataFrame | None:
        path = self.path(symbol, granularity)
        if not path.exists():
            return None
        frame = pd.read_csv(path, index_col="time")
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame.astype({"open": float, "high": float, "low": float, "close": float})

    def ensure(self, symbol: str, granularity: str, start: datetime, end: datetime, fetch: Fetch,
               refresh: bool = False) -> pd.DataFrame:
        """Bars covering ``start``..``end``, downloading only what the cache lacks."""
        have = None if refresh else self.load(symbol, granularity)
        if have is None or have.empty or have.index[0] > pd.Timestamp(start) + timedelta(days=3):
            frame = fetch(symbol, granularity, start, end)
        else:
            last = have.index[-1].to_pydatetime()
            newer = fetch(symbol, granularity, last + timedelta(seconds=1), end) if last < end else None
            frame = have if newer is None or newer.empty else pd.concat([have, newer])
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame.to_csv(self.path(symbol, granularity), index_label="time", compression="gzip")
        return frame[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# A feed that only shows the past
# ---------------------------------------------------------------------------

def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Bar times as nanoseconds, whatever unit pandas stored them in."""
    return index.as_unit("ns").asi8


@dataclass(frozen=True)
class _Quote:
    symbol: str
    bid: float
    ask: float
    time: datetime


class HistoricalFeed:
    """Stands in for OANDA at a moment in the past: only bars closed by ``now`` exist."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame], spreads: dict[str, float] | None = None):
        self.frames = frames
        self.spreads = spreads or TYPICAL_SPREAD_PIPS
        self.now: datetime | None = None
        self._ends = {key: _ns(frame.index) + BAR_SECONDS[key[1]] * 1_000_000_000
                      for key, frame in frames.items()}

    def _upto(self, symbol: str, granularity: str) -> int:
        """How many bars of ``symbol`` had closed by ``now``."""
        ends = self._ends[(symbol, granularity)]
        return int(np.searchsorted(ends, pd.Timestamp(self.now).value, side="right"))

    def candles(self, symbol: str, granularity: str, count: int) -> pd.DataFrame:
        if (symbol, granularity) not in self.frames:
            raise KeyError(f"no {granularity} history for {symbol}")
        k = self._upto(symbol, granularity)
        return self.frames[(symbol, granularity)].iloc[max(0, k - count):k]

    def quote(self, symbol: str) -> _Quote:
        bars = self.candles(symbol, "M5", 1)
        if bars.empty:
            raise KeyError(f"no price for {symbol} at {self.now}")
        mid = float(bars["close"].iloc[-1])
        half = self.spreads.get(symbol, 2.0) * spec_for(symbol).pip / 2
        return _Quote(symbol, mid - half, mid + half, self.now)


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------

@dataclass
class Rules:
    """The desk's order rules: the knobs an experiment may turn."""

    step_minutes: int = 10
    min_rr: float = 2.0
    final: int = 6
    max_per_currency: int = 2
    window: ScanWindow = field(default_factory=ScanWindow)
    latest_cancel: time = LATEST_CANCEL


def scan_times(start: datetime, end: datetime, rules: Rules) -> Iterable[datetime]:
    """Every scan the watcher would run between ``start`` and ``end``."""
    day = start.astimezone(NEW_YORK).date()
    last = end.astimezone(NEW_YORK).date()
    step = timedelta(minutes=rules.step_minutes)
    while day <= last:
        t = NEW_YORK.localize(datetime.combine(day, rules.window.start)).astimezone(UTC)
        stop = t + timedelta(days=1)
        while t < stop:
            if start <= t < end and rules.window.current_end(t) is not None:
                yield t
            t += step
        day += timedelta(days=1)


def _entry(s: Setup, t: datetime, n: int) -> Entry:
    spec = spec_for(s.symbol)
    return Entry(id=f"B{n}", created_at=t, symbol=s.symbol, direction=s.direction, entry=s.entry,
                 stop=s.stop, target=s.target, planned_rr=s.rr, spread=s.spread_pips * spec.pip,
                 expires_at=s.expires_at, conviction="scanner", strategy=s.strategy,
                 scanner_score=s.score, ticket=f"B{n}", initial_stop=s.stop,
                 rationale="; ".join(s.reasons[:3]))


ScanLog = list[tuple[datetime, list[Setup]]]


def scan_log(feed: HistoricalFeed, symbols: Iterable[str], start: datetime, end: datetime,
             rules: Rules | None = None, progress: Callable[[datetime, int], None] | None = None) -> ScanLog:
    """What the scanner offered at every scan from ``start`` to ``end``: the slow part, done once."""
    rules = rules or Rules()
    symbols = [s for s in symbols if (s, "M5") in feed.frames and (s, "H1") in feed.frames]
    log: ScanLog = []
    found = 0
    for t in scan_times(start, end, rules):
        if not market_open(t):
            continue
        feed.now = t
        result = scan_smc(feed.candles, feed.quote, symbols, min_rr=rules.min_rr, top=rules.final + 4,
                          max_per_currency=rules.max_per_currency + 1, window=rules.window, now=t,
                          latest_cancel=rules.latest_cancel)
        log.append((t, result.setups))
        found += len(result.setups)
        if progress:
            progress(t, found)
    return log


def trade(log: ScanLog, feed: HistoricalFeed, end: datetime, rules: Rules | None = None,
          experiment=None) -> list[Entry]:
    """The orders the desk would have placed from ``log``, settled: fast, so ideas can be tried by the dozen.

    ``experiment`` (see :mod:`tradingagents.fx.quinn`) may refuse setups or move
    their targets before they become orders, exactly as if the scanner itself
    had that rule.
    """
    rules = rules or Rules()
    entries: list[Entry] = []
    live: list[tuple[datetime, Entry]] = []          # (when its symbol and side are free again, the order)
    seen: set[str] = set()
    seen_day: date | None = None

    for t, offered in log:
        day = t.astimezone(NEW_YORK).date()
        if day != seen_day:
            seen, seen_day = set(), day
        live = [(free, e) for free, e in live if t < free]
        setups = offered
        if experiment is not None:
            setups = [experiment.adjust(s) for s in offered if experiment.admits(s, t)]
        on_book = {(e.symbol, e.direction) for _, e in live}
        fresh = [s for s in setups if (s.symbol, s.direction) not in on_book and setup_key(s) not in seen]
        seen |= {setup_key(s) for s in setups}
        for s in fresh[:rules.final]:
            e = _entry(s, t, len(entries) + 1)
            e.features = {**s.features, "hour": t.astimezone(NEW_YORK).hour, "target_kind": s.target_kind}
            _settle(feed, e)
            if experiment is not None and getattr(experiment, "manage", None):
                _manage(feed, e, experiment.manage)
            entries.append(e)
            live.append((_free_from(e), e))
            on_book.add((s.symbol, s.direction))
    return entries


def _settle(feed: HistoricalFeed, e: Entry) -> None:
    """Play an order out to its end at once, on its own bars only.

    The outcome uses bars after the order was placed, as it would live; the
    replay only asks *when* the order stops occupying its symbol and side
    (:func:`_free_from`), which it could have seen by then.
    """
    bars = feed.frames[(e.symbol, "M5")]
    first = int(np.searchsorted(_ns(bars.index), pd.Timestamp(e.created_at).value, side="left"))
    horizon = jr.day_close(e.created_at) + timedelta(minutes=5)
    last = int(np.searchsorted(_ns(bars.index), pd.Timestamp(horizon).value, side="right"))
    jr.simulate(e, bars.iloc[first:last], horizon)


def _manage(feed: HistoricalFeed, e: Entry, how: str) -> None:
    """Replace a filled order's result with what managing it at +1R would have made."""
    from tradingagents.fx.anatomy import _bars, _what_if
    if e.filled_at is None or e.status not in jr.FINISHED:
        return
    path = _bars(feed.frames[(e.symbol, "M5")], e.filled_at, jr.flat_by(e.filled_at))
    r = round(_what_if(e, path, half=how == "half_off"), 2)
    e.result_r = r
    e.status = jr.WON if r > 0 else jr.LOST if r <= -1 else jr.CLOSED
    e.note = f"managed ({how} at +1R)"


def _free_from(e: Entry) -> datetime:
    """When the desk would have seen the order finished: its last bar closed, or its cancel time passed."""
    if e.status in jr.ACTIVE:                        # the history ends first
        return datetime.max.replace(tzinfo=UTC)
    if e.status == jr.EXPIRED:
        return e.expires_at
    return (e.exit_at or e.expires_at) + timedelta(minutes=5)


def replay(feed: HistoricalFeed, symbols: Iterable[str], start: datetime, end: datetime,
           rules: Rules | None = None, progress: Callable[[datetime, int], None] | None = None,
           experiment=None) -> list[Entry]:
    """Every order the scanner would have handed the desk from ``start`` to ``end``, settled."""
    rules = rules or Rules()
    return trade(scan_log(feed, symbols, start, end, rules, progress), feed, end, rules, experiment)


def log_key(symbols: Iterable[str], start: datetime, end: datetime, rules: Rules) -> str:
    """A name for a cached scan log: changes when the period, rules or scanner code change."""
    import hashlib

    from tradingagents.fx import smc, smc_scanner
    code = "".join(Path(m.__file__).read_text(encoding="utf-8") for m in (smc, smc_scanner))
    raw = f"{sorted(symbols)}|{start:%Y%m%d}|{end:%Y%m%d}|{rules.step_minutes}|{rules.min_rr}|{rules.final}|" \
          f"{rules.max_per_currency}|{rules.window.label()}|{rules.latest_cancel}|{code}"
    return f"{start:%Y%m%d}-{end:%Y%m%d}-{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def cached_scan_log(folder: Path, feed: HistoricalFeed, symbols: list[str], start: datetime, end: datetime,
                    rules: Rules, progress=None) -> ScanLog:
    """:func:`scan_log`, kept on disk so each later experiment skips the slow scanning."""
    import pickle
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"scans-{log_key(symbols, start, end, rules)}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    log = scan_log(feed, symbols, start, end, rules, progress)
    with path.open("wb") as fh:
        pickle.dump(log, fh)
    return log


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

# The benchmarks set before the desk went live: (red line, target, strong).
BENCHMARKS = {
    "win_rate": ("Win rate", 0.29, 0.38, 0.45, True, "{:.0%}"),
    "expectancy_r": ("Expectancy per filled trade", 0.0, 0.25, 0.5, True, "{:+.2f}R"),
    "profit_factor": ("Profit factor", 1.0, 1.35, 1.8, True, "{:.2f}"),
    "fill_rate": ("Fill rate", 0.25, 0.40, 0.55, True, "{:.0%}"),
    "missed_rate": ("Target before fill", 0.40, 0.25, 0.15, False, "{:.0%}"),
    "max_drawdown_r": ("Max drawdown", 13.0, 8.0, 5.0, False, "{:.1f}R"),
    "losing_streak": ("Longest losing streak", 11, 8, 5, False, "{:.0f}"),
}


def grade(key: str, value: float | None) -> str:
    if value is None:
        return "no data"
    _, red, target, strong, higher, _ = BENCHMARKS[key]
    better = (lambda a, b: a >= b) if higher else (lambda a, b: a <= b)
    if better(value, strong):
        return "strong"
    if better(value, target):
        return "on target"
    if better(value, red):
        return "short of target"
    return "past the red line"


@dataclass
class Report:
    start: str
    end: str
    symbols: list[str]
    rules: dict
    orders: int
    filled: int
    trading_days: int
    metrics: dict[str, float | None]
    grades: dict[str, str]
    by_symbol: dict[str, dict]
    by_hour: dict[str, dict]
    by_direction: dict[str, dict]
    by_quarter: dict[str, dict]
    verdict: str
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _group(entries: list[Entry], key: Callable[[Entry], str]) -> dict[str, dict]:
    groups: dict[str, list[Entry]] = defaultdict(list)
    for e in entries:
        groups[key(e)].append(e)
    out = {}
    for name in sorted(groups):
        es = groups[name]
        filled = [e for e in es if e.status in jr.FINISHED and e.result_r is not None]
        total = sum(e.result_r for e in filled)
        out[name] = {"orders": len(es), "filled": len(filled), "total_r": round(total, 2),
                     "expectancy_r": round(total / len(filled), 3) if filled else None,
                     "win_rate": round(sum(e.result_r > 0 for e in filled) / len(filled), 3) if filled else None}
    return out


def _streak(results: list[float]) -> int:
    worst = run = 0
    for r in results:
        run = run + 1 if r <= 0 else 0
        worst = max(worst, run)
    return worst


def summarize(entries: list[Entry], start: datetime, end: datetime, symbols: list[str], rules: Rules) -> Report:
    s = jr.stats(entries)
    finished = sorted((e for e in entries if e.status in jr.FINISHED and e.result_r is not None),
                      key=lambda e: e.exit_at or e.created_at)
    unfilled = s.expired + s.missed
    days = len({e.created_at.astimezone(NEW_YORK).date() for e in entries}) or 0
    span_days = sum(1 for d in pd.bdate_range(start.date(), end.date()))
    metrics = {
        "orders_per_day": round(len(entries) / span_days, 2) if span_days else None,
        "filled_per_day": round(len(finished) / span_days, 2) if span_days else None,
        "win_rate": s.win_rate,
        "expectancy_r": s.avg_r,
        "profit_factor": s.profit_factor,
        "fill_rate": s.fill_rate,
        "missed_rate": round(s.missed / (len(finished) + unfilled), 3) if finished or unfilled else None,
        "total_r": round(s.total_r, 2),
        "max_drawdown_r": round(s.max_drawdown_r, 2),
        "losing_streak": _streak([e.result_r for e in finished]),
        "avg_win_r": round(float(np.mean([e.result_r for e in finished if e.result_r > 0])), 2)
        if any(e.result_r > 0 for e in finished) else None,
    }
    grades = {k: grade(k, metrics.get(k)) for k in BENCHMARKS}

    def quarter(e: Entry) -> str:
        d = e.created_at.astimezone(NEW_YORK)
        return f"{d.year} Q{(d.month - 1) // 3 + 1}"

    by_quarter = _group(entries, quarter)
    positive = sum(1 for q in by_quarter.values() if (q["expectancy_r"] or 0) > 0)
    rated = sum(1 for q in by_quarter.values() if q["filled"] >= 10)
    n = len(finished)
    if n < 100:
        verdict = (f"Not enough trades to judge: {n} filled (at least 100 are needed). "
                   "Run a longer history before reading anything into these numbers.")
    elif (s.avg_r or 0) > 0 and (s.profit_factor or 0) > 1 and rated and positive / max(rated, 1) >= 2 / 3:
        verdict = (f"Evidence of an edge: {n} filled trades at {s.avg_r:+.2f}R each, profitable in "
                   f"{positive} of {len(by_quarter)} quarters. This is the bar new ideas must beat.")
    elif (s.avg_r or 0) > 0:
        verdict = (f"A thin, unsteady edge: {s.avg_r:+.2f}R per trade over {n} filled trades, but positive "
                   f"in only {positive} of {len(by_quarter)} quarters. The rules alone are not yet reliable.")
    else:
        verdict = (f"No edge in the rules alone: {s.avg_r:+.2f}R per trade over {n} filled trades. "
                   "The desk's filtering, or new rules from the lab, must find one.")
    return Report(
        start=start.date().isoformat(), end=end.date().isoformat(), symbols=symbols,
        rules={"step_minutes": rules.step_minutes, "min_rr": rules.min_rr, "final": rules.final,
               "max_per_currency": rules.max_per_currency, "window": rules.window.label(),
               "latest_cancel": rules.latest_cancel.strftime("%H:%M")},
        orders=len(entries), filled=n, trading_days=days, metrics=metrics, grades=grades,
        by_symbol=_group(entries, lambda e: e.symbol),
        by_hour=_group(entries, lambda e: f"{e.created_at.astimezone(NEW_YORK):%H}:00"),
        by_direction=_group(entries, lambda e: e.direction),
        by_quarter=by_quarter, verdict=verdict,
        notes=["Scanner only: no agents. Settled on 5-minute bars with typical spreads; "
               "a bar touching stop and target counts as the stop."],
    )


def to_markdown(r: Report) -> str:
    lines = [f"# Baseline backtest, {r.start} to {r.end}", "", f"**{r.verdict}**", "",
             f"{r.orders} orders on {len(r.symbols)} instruments over {r.trading_days} trading days "
             f"({r.metrics['orders_per_day']} a day); {r.filled} filled ({r.metrics['filled_per_day']} a day).", "",
             "## Against the benchmarks", "", "| Measure | Result | Red line | Target | Strong | Grade |",
             "|---|---|---|---|---|---|"]
    for key, (label, red, target, strong, _, fmt) in BENCHMARKS.items():
        value = r.metrics.get(key)
        shown = "—" if value is None else fmt.format(value)
        lines.append(f"| {label} | {shown} | {fmt.format(red)} | {fmt.format(target)} | {fmt.format(strong)} "
                     f"| {r.grades[key]} |")
    lines += ["", f"Total: {r.metrics['total_r']:+.2f}R · average win {r.metrics['avg_win_r'] or 0:+.2f}R", ""]
    for title, groups in (("By quarter (does it hold up over time?)", r.by_quarter),
                          ("By instrument", r.by_symbol), ("By hour placed (New York)", r.by_hour),
                          ("By direction", r.by_direction)):
        lines += [f"## {title}", "", "| | Orders | Filled | Win rate | Per trade | Total |", "|---|---|---|---|---|---|"]
        for name, g in groups.items():
            wr = "—" if g["win_rate"] is None else f"{g['win_rate']:.0%}"
            ex = "—" if g["expectancy_r"] is None else f"{g['expectancy_r']:+.2f}R"
            lines.append(f"| {name} | {g['orders']} | {g['filled']} | {wr} | {ex} | {g['total_r']:+.2f}R |")
        lines.append("")
    lines += ["## Rules replayed", "", *(f"- {k}: {v}" for k, v in r.rules.items()), "", *r.notes, ""]
    return "\n".join(lines)


def save(r: Report, entries: list[Entry], folder: Path, name: str = "baseline") -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    md = folder / f"{name}-{stamp}.md"
    md.write_text(to_markdown(r), encoding="utf-8")
    js = folder / f"{name}-{stamp}.json"
    trades = [{"ticket": e.ticket, "created_at": e.created_at.isoformat(), "symbol": e.symbol,
               "direction": e.direction, "entry": e.entry, "stop": e.stop, "target": e.target,
               "rr": e.planned_rr, "score": e.scanner_score, "status": e.status, "result_r": e.result_r,
               "filled_at": e.filled_at.isoformat() if e.filled_at else None,
               "exit_at": e.exit_at.isoformat() if e.exit_at else None, "note": e.note,
               "features": getattr(e, "features", {}), "why": e.rationale} for e in entries]
    js.write_text(json.dumps({"report": json.loads(r.to_json()), "trades": trades}, indent=1, default=str),
                  encoding="utf-8")
    return md, js


def telegram_summary(r: Report) -> str:
    m = r.metrics
    def show(key: str) -> str:
        value = m.get(key)
        return "—" if value is None else BENCHMARKS[key][5].format(value)
    return (f"🧪 <b>Quinn's lab · baseline {r.start} → {r.end}</b>\n{r.verdict}\n\n"
            f"Filled {r.filled} of {r.orders} orders ({m['filled_per_day']} a day)\n"
            f"Win rate {show('win_rate')} · per trade {show('expectancy_r')} · PF {show('profit_factor')}\n"
            f"Max drawdown {show('max_drawdown_r')} · total {m['total_r']:+.2f}R")


