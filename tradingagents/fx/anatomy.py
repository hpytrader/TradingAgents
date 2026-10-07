"""Trade anatomy: where a losing strategy loses, read off every trade's price path.

Filtering which setups to take did not rescue the baseline, so the question is
how the trades play out. For every filled trade of a replay this measures, in
R against the initial stop:

- **before costs**: the result with no spread. If the rules make money before
  costs and lose after, the spread is the leak, and bigger trades (a higher
  timeframe, wider stops) are the fix;
- **how far it went against us** before winning (MAE), how soon losers were
  stopped, and how many losers then ran to the target the same day: signs
  that the stop sits where the next stop hunt goes;
- **how far it went for us** before losing (MFE): losers that reached +1R
  first point at the exit, and two what-ifs replay every path with a
  break-even stop, and with half taken off, at +1R.

It describes the design year; it proves nothing. A fix it points to is still
tried by Quinn on the locked year before it counts. Paths are read on
five-minute bars: within a bar the order of high and low is unknown, so every
what-if takes the worse case (the stop first).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from tradingagents.fx import journal as jr
from tradingagents.fx.journal import Entry

LEVELS = (0.5, 1.0, 1.5, 2.0)


@dataclass
class Path:
    """One filled trade, measured in R against its initial stop."""

    ticket: str
    symbol: str
    status: str
    net_r: float
    gross_r: float
    spread_share: float          # spread as a share of the risk
    mfe: float                   # best point reached before the exit
    mae: float                   # worst point reached before the exit (a positive number)
    minutes: float               # fill to exit
    target_after_stop: bool      # a loser whose target traded later the same day
    breakeven_r: float           # the result with the stop moved to the entry at +1R
    half_off_r: float            # half taken at +1R, the rest to break-even or the target


def _bars(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    idx = frame.index.as_unit("ns").asi8
    a = int(np.searchsorted(idx, pd.Timestamp(start).value, side="left"))
    b = int(np.searchsorted(idx, pd.Timestamp(end).value, side="right"))
    return frame.iloc[a:b]


def _gross(e: Entry) -> float:
    sign = 1 if e.long else -1
    if e.status == jr.LOST:
        return -1.0
    return (e.exit_price - e.entry) * sign / e.risk


def _what_if(e: Entry, bars: pd.DataFrame, half: bool) -> float:
    """Replay the path with the stop at the entry once +1R trades; ``half`` banks half there.

    Results are net of the spread like the journal's: a scratch costs the spread.
    """
    sign = 1 if e.long else -1
    risk, spread = e.risk, e.spread
    stop = e.initial_stop if e.initial_stop is not None else e.stop
    one_r = e.entry + sign * risk
    moved = False
    banked = 0.0
    deadline = jr.flat_by(e.filled_at)

    def net(price: float) -> float:
        move = (price - e.entry) * sign
        return -1.0 if move <= -risk + 1e-12 else (move - spread) / (risk + spread)

    last = e.entry
    for t, bar in zip(bars.index, bars.itertuples(index=False), strict=False):
        if t.to_pydatetime() >= deadline:
            break
        hi, lo = bar.high, bar.low
        worst, best = (lo, hi) if e.long else (hi, lo)
        hit_stop = (worst - stop) * sign <= 0
        hit_target = (best - e.target) * sign >= 0
        if hit_stop:                                   # the worse case first, inside a bar
            rest = net(stop)
            return banked + (0.5 if half and moved else 1.0) * rest
        if hit_target:
            return banked + (0.5 if half and moved else 1.0) * net(e.target)
        if not moved and (best - one_r) * sign >= 0:
            moved, stop = True, e.entry
            if half:
                banked = 0.5 * net(one_r)
        last = bar.close
    return banked + (0.5 if half and moved else 1.0) * net(last)


def measure(e: Entry, m5: pd.DataFrame) -> Path:
    sign = 1 if e.long else -1
    end = e.exit_at or jr.flat_by(e.filled_at)
    held = _bars(m5, e.filled_at, end)
    # a loser's last bar holds the stop: its high (low) may have come after it, so leave it out
    before = held.iloc[:-1] if e.status == jr.LOST and len(held) > 1 else held
    if before.empty:
        mfe = mae = 0.0
    elif e.long:
        mfe = float((before["high"].max() - e.entry) / e.risk)
        mae = float((e.entry - before["low"].min()) / e.risk)
    else:
        mfe = float((e.entry - before["low"].min()) / e.risk)
        mae = float((before["high"].max() - e.entry) / e.risk)
    after_stop = False
    if e.status == jr.LOST and e.exit_at is not None:
        later = _bars(m5, e.exit_at + timedelta(minutes=5), jr.flat_by(e.filled_at))
        if not later.empty:
            best = later["high"].max() if e.long else later["low"].min()
            after_stop = bool((best - e.target) * sign >= 0)
    path = _bars(m5, e.filled_at, jr.flat_by(e.filled_at))
    return Path(ticket=e.ticket, symbol=e.symbol, status=e.status, net_r=float(e.result_r),
                gross_r=round(_gross(e), 3), spread_share=round(e.spread / e.risk, 3),
                mfe=round(max(mfe, 0.0), 2), mae=round(max(mae, 0.0), 2),
                minutes=round(((e.exit_at or end) - e.filled_at).total_seconds() / 60, 1),
                target_after_stop=after_stop,
                breakeven_r=round(_what_if(e, path, half=False), 3),
                half_off_r=round(_what_if(e, path, half=True), 3))


@dataclass
class Anatomy:
    trades: int
    net_r: float
    gross_r: float
    spread_share: float
    win_rate: float
    winners_mae: dict[str, float]          # share of winners that went this far against us first
    losers_mfe: dict[str, float]           # share of losers that went this far for us first
    stopped_within_15: float               # share of losers stopped within 15 minutes of the fill
    stopped_within_60: float
    target_after_stop: float               # share of losers whose target traded later the same day
    closed_at_17: float                    # share of trades closed at the New York close
    breakeven_r: float
    half_off_r: float
    same_bar_losses: float                 # share of losses where one bar touched the stop and the target
    findings: list[str] = field(default_factory=list)


def _mean(xs) -> float:
    xs = list(xs)
    return round(float(np.mean(xs)), 3) if xs else 0.0


def _share(xs, cond) -> float:
    xs = list(xs)
    return round(sum(1 for x in xs if cond(x)) / len(xs), 3) if xs else 0.0


def study(entries: list[Entry], frames: dict[tuple[str, str], pd.DataFrame]) -> tuple[Anatomy, list[Path]]:
    filled = [e for e in entries if e.status in jr.FINISHED and e.result_r is not None and e.filled_at]
    paths = [measure(e, frames[(e.symbol, "M5")]) for e in filled]
    wins = [p for p in paths if p.status == jr.WON]
    losses = [p for p in paths if p.status == jr.LOST]
    lost_entries = [e for e in filled if e.status == jr.LOST]
    a = Anatomy(
        trades=len(paths), net_r=_mean(p.net_r for p in paths), gross_r=_mean(p.gross_r for p in paths),
        spread_share=_mean(p.spread_share for p in paths),
        win_rate=_share(paths, lambda p: p.net_r > 0),
        winners_mae={f"{x:g}R": _share(wins, lambda p, x=x: p.mae >= x) for x in (0.25, 0.5, 0.75)},
        losers_mfe={f"{x:g}R": _share(losses, lambda p, x=x: p.mfe >= x) for x in LEVELS},
        stopped_within_15=_share(losses, lambda p: p.minutes <= 15),
        stopped_within_60=_share(losses, lambda p: p.minutes <= 60),
        target_after_stop=_share(losses, lambda p: p.target_after_stop),
        closed_at_17=_share(paths, lambda p: p.status == jr.CLOSED),
        breakeven_r=_mean(p.breakeven_r for p in paths), half_off_r=_mean(p.half_off_r for p in paths),
        same_bar_losses=_share(lost_entries, lambda e: "same minute" in (e.note or "")),
    )
    a.findings = findings(a)
    return a, paths


def findings(a: Anatomy) -> list[str]:
    """The plain-language reading, one line per suspect."""
    out = []
    cost = a.gross_r - a.net_r
    if a.gross_r > 0 and a.net_r <= 0:
        out.append(f"COSTS: before the spread the rules make {a.gross_r:+.2f}R a trade; the spread "
                   f"({a.spread_share:.0%} of the risk on average) costs {cost:.2f}R and turns it into "
                   f"{a.net_r:+.2f}R. The raw edge is thin and the spread is too big a share of each trade: "
                   "bigger trades (a higher timeframe, wider stops) are a fix to try.")
    elif a.gross_r > 0:
        out.append(f"COSTS: {a.gross_r:+.2f}R a trade before the spread, {a.net_r:+.2f}R after "
                   f"({a.spread_share:.0%} of the risk). Costs are affordable.")
    else:
        out.append(f"COSTS: the rules lose even before the spread ({a.gross_r:+.2f}R a trade); "
                   "cutting costs alone cannot fix them.")
    hunt = a.target_after_stop
    fast = a.stopped_within_15
    if hunt >= 0.20 or fast >= 0.35:
        out.append(f"STOPS: {hunt:.0%} of losers were stopped and then reached the target the same day, and "
                   f"{fast:.0%} were stopped within 15 minutes of the fill: the stop sits where the next sweep "
                   "goes, or the entry is early. Test a confirmation entry, or a stop beyond the whole zone.")
    else:
        out.append(f"STOPS: only {hunt:.0%} of losers reached the target after being stopped and {fast:.0%} "
                   "were stopped within 15 minutes: stop placement is not the main leak.")
    gave_back = a.losers_mfe.get("1R", 0.0)
    best = max(a.breakeven_r, a.half_off_r)
    if gave_back >= 0.25 or best - a.net_r >= 0.05:
        which = "break-even at +1R" if a.breakeven_r >= a.half_off_r else "half off at +1R"
        out.append(f"EXITS: {gave_back:.0%} of losers were +1R up before turning into a full loss. With "
                   f"{which} the same trades make {best:+.2f}R a trade instead of {a.net_r:+.2f}R. "
                   "Trade management is worth testing.")
    else:
        out.append(f"EXITS: only {gave_back:.0%} of losers were ever +1R up; a break-even stop gives "
                   f"{a.breakeven_r:+.2f}R and half off at +1R {a.half_off_r:+.2f}R a trade, against "
                   f"{a.net_r:+.2f}R now. Exits are not the main leak.")
    if not any(" is a fix to try" in f or "Test a confirmation" in f or "worth testing" in f for f in out):
        out.append("VERDICT: no single leak. The setup itself shows no edge on these timeframes: the next step "
                   "is a different version of it (a higher timeframe) or a different strategy, not a filter.")
    if a.same_bar_losses >= 0.05:
        out.append(f"MEASUREMENT: {a.same_bar_losses:.0%} of losses had the stop and the target in one "
                   "five-minute bar and were counted as losses; on one-minute bars some would be wins, so "
                   "the true result is a little better than shown.")
    return out


def to_markdown(a: Anatomy, period: str) -> str:
    rows = [
        ("Filled trades", f"{a.trades}"),
        ("Result per trade, after the spread", f"{a.net_r:+.2f}R"),
        ("Result per trade, before the spread", f"{a.gross_r:+.2f}R"),
        ("Spread as a share of the risk", f"{a.spread_share:.0%}"),
        ("Win rate", f"{a.win_rate:.0%}"),
        ("Closed at 16:55", f"{a.closed_at_17:.0%}"),
        ("Losers stopped within 15 / 60 minutes", f"{a.stopped_within_15:.0%} / {a.stopped_within_60:.0%}"),
        ("Losers that reached the target after the stop", f"{a.target_after_stop:.0%}"),
        ("What if: break-even at +1R", f"{a.breakeven_r:+.2f}R a trade"),
        ("What if: half off at +1R, rest to break-even", f"{a.half_off_r:+.2f}R a trade"),
        ("Losses with stop and target in one bar", f"{a.same_bar_losses:.0%}"),
    ]
    lines = [f"# Trade anatomy, {period}", "", *(f"- **{f}**" for f in a.findings), "",
             "| Measure | Value |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows), "",
             "## How far losers went our way first", "", "| Reached | Share of losers |", "|---|---|",
             *(f"| +{k} | {v:.0%} |" for k, v in a.losers_mfe.items()), "",
             "## How far winners went against us first", "", "| Against us | Share of winners |", "|---|---|",
             *(f"| −{k} | {v:.0%} |" for k, v in a.winners_mae.items()), "",
             "Read on five-minute bars; every what-if takes the stop first when a bar touches both. "
             "This describes the design year: a fix it suggests must still pass the locked year.", ""]
    return "\n".join(lines)


def telegram_summary(a: Anatomy) -> str:
    return "🔬 <b>Quinn's lab · trade anatomy</b>\n" + "\n\n".join(a.findings)


def as_dict(a: Anatomy) -> dict:
    return asdict(a)
