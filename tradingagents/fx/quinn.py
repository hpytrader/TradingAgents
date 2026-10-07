"""Quinn, the desk's R&D analyst: tests trading ideas against history without fooling himself.

An idea is an :class:`Experiment`: a rule the scanner does not have yet, such
as "no new orders in the London open's first two hours" or "only sweeps of
the previous day's or a session's high or low", with the reason it should
work written down first.

Every experiment is tried in two stages, on two different years:

1. **Design year** (the most recent twelve months, the year the baseline
   report has already shown): the idea must beat the unchanged rules by a
   clear margin, on enough trades, and make money on its own.
2. **Locked year** (the twelve months before that, which no report shows):
   only an idea that passed the design year is tried here. It must make money
   there too, steadily across quarters, with a t-score that rises with every
   idea already tried on the locked year, so that trying many ideas cannot
   produce a lucky winner.

An idea that passes both is a **candidate**. Nothing reaches the live desk
from here: a candidate goes to paper trading next, and live only with the
trader's approval.

Every result goes into a ledger (``~/.tradingagents/lab/ledger.json``) so the
number of ideas tried, and the bar each one faced, is always on record.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.fx import journal as jr
from tradingagents.fx.instruments import spec_for
from tradingagents.fx.journal import Entry
from tradingagents.fx.scanner import Setup
from tradingagents.fx.smc import NEW_YORK

# Pool quality, as the scanner rates it.
POOLS = {3: "previous day high or low", 2: "Asian or London session high or low",
         1: "equal highs or lows", 0: "single swing point"}

# What an idea must show on the design year before it may touch the locked year.
DESIGN_MIN_TRADES = 300
DESIGN_MIN_GAIN_R = 0.05          # better than the unchanged rules, per trade

# What it must show on the locked year to become a candidate.
LOCKED_MIN_TRADES = 150
LOCKED_MIN_PF = 1.10
LOCKED_MIN_QUARTERS = 0.75        # share of quarters (with 10+ trades) that made money
BASE_T = 2.0                      # t-score bar for the first idea; +0.1 for every ten already tried


@dataclass
class Experiment:
    """One idea: a rule to add, and why it should work."""

    name: str
    hypothesis: str
    skip_hours: tuple[int, ...] = ()        # New York hours with no new orders
    last_hour: int | None = None            # no new orders from this New York hour on
    min_score: float = 0.0
    min_pool_quality: int = 0               # 3 previous day, 2 session, 1 equal highs/lows, 0 swing
    shift: str | None = None                # "BOS" or "CHoCH" on the 5-minute chart
    h1_break: str | None = None             # "BOS" or "CHoCH" on the hourly chart
    zones: tuple[str, ...] = ()             # e.g. ("fair value gap", "fair value gap inside the order block")
    min_displacement: float = 0.0           # the move out of the sweep, in 5-minute ATRs
    min_depth: float = 0.0                  # how far back into the move the entry sits, 0-1
    target_r: float | None = None           # bring a farther target in to this many R
    symbols: tuple[str, ...] = ()           # empty: every instrument
    skip_wide_spread: bool = False
    stop_mult: float = 1.0                  # widen the stop to this many times its distance (target kept)
    retarget_r: float | None = None         # then set the target this many R away, after the spread
    max_spread_share: float | None = None   # skip a setup whose spread is more than this share of its risk
    manage: str | None = None               # "breakeven" or "half_off" once the trade is +1R

    def admits(self, s: Setup, t: datetime) -> bool:
        f = s.features or {}
        hour = t.astimezone(NEW_YORK).hour
        if hour in self.skip_hours or (self.last_hour is not None and hour >= self.last_hour):
            return False
        if self.symbols and s.symbol not in self.symbols:
            return False
        if s.score < self.min_score or f.get("pool_quality", 0) < self.min_pool_quality:
            return False
        if self.shift and f.get("shift") != self.shift:
            return False
        if self.h1_break and f.get("h1_break") != self.h1_break:
            return False
        if self.zones and f.get("zone") not in self.zones:
            return False
        if f.get("displacement", 0.0) < self.min_displacement or f.get("depth", 0.0) < self.min_depth:
            return False
        if self.max_spread_share is not None and s.risk_pips > 0 and \
                s.spread_pips / s.risk_pips > self.max_spread_share:
            return False
        return not (self.skip_wide_spread and f.get("wide_spread"))

    def adjust(self, s: Setup) -> Setup:
        """Widen the stop (``stop_mult``), then bring a far target in to ``target_r`` R."""
        sign = 1 if s.direction == "long" else -1
        spec = spec_for(s.symbol)
        spread = s.spread_pips * spec.pip
        if self.stop_mult != 1.0:
            wider = abs(s.entry - s.stop) * self.stop_mult
            reward = abs(s.target - s.entry)
            s = replace(s, stop=spec.round_price(s.entry - sign * wider), risk_pips=round(spec.pips(wider), 1),
                        rr=round((reward - spread) / (wider + spread), 2))
        if self.retarget_r is not None:
            risk = abs(s.entry - s.stop)
            target = s.entry + sign * (self.retarget_r * (risk + spread) + spread)
            reward = abs(target - s.entry)
            s = replace(s, target=spec.round_price(target), reward_pips=round(spec.pips(reward), 1),
                        rr=round((reward - spread) / (risk + spread), 2), target_kind=f"{self.retarget_r:g}R")
        if self.target_r is None:
            return s
        risk = abs(s.entry - s.stop)
        nearer = s.entry + sign * self.target_r * risk
        if (nearer - s.target) * sign >= 0:          # the scanner's target is already nearer
            return s
        reward = abs(nearer - s.entry)
        return replace(s, target=spec.round_price(nearer), reward_pips=round(spec.pips(reward), 1),
                       rr=round((reward - spread) / (risk + spread), 2), target_kind=f"{self.target_r:g}R")

    def describe(self) -> str:
        rules = []
        if self.skip_hours:
            rules.append("no orders placed " + ", ".join(f"{h:02d}:00" for h in self.skip_hours) + " NY")
        if self.last_hour is not None:
            rules.append(f"no orders from {self.last_hour:02d}:00 NY")
        if self.min_score:
            rules.append(f"score {self.min_score:g}+")
        if self.min_pool_quality:
            rules.append("sweeps of " + " or ".join(POOLS[q] for q in range(3, self.min_pool_quality - 1, -1)))
        if self.shift:
            rules.append(f"5m {self.shift} only")
        if self.h1_break:
            rules.append(f"1h {self.h1_break} only")
        if self.zones:
            rules.append("entries in a " + " or ".join(self.zones))
        if self.min_displacement:
            rules.append(f"displacement {self.min_displacement:g}x ATR+")
        if self.min_depth:
            rules.append(f"entry {self.min_depth:.0%}+ back into the move")
        if self.target_r is not None:
            rules.append(f"targets no farther than {self.target_r:g}R")
        if self.symbols:
            rules.append("only " + ", ".join(self.symbols))
        if self.skip_wide_spread:
            rules.append("no wide-spread setups")
        if self.stop_mult != 1.0:
            rules.append(f"stops {self.stop_mult:g}x as far" + ("" if self.retarget_r else ", same targets"))
        if self.retarget_r is not None:
            rules.append(f"targets {self.retarget_r:g}R after the spread")
        if self.max_spread_share is not None:
            rules.append(f"spread at most {self.max_spread_share:.0%} of the risk")
        if self.manage == "breakeven":
            rules.append("stop to break-even at +1R")
        elif self.manage == "half_off":
            rules.append("half off at +1R, the rest to break-even")
        return "; ".join(rules) or "the unchanged rules"


# ---------------------------------------------------------------------------
# Scoring an idea
# ---------------------------------------------------------------------------

@dataclass
class Result:
    trades: int
    expectancy_r: float | None
    profit_factor: float | None
    win_rate: float | None
    total_r: float
    t_score: float | None
    quarters_positive: int
    quarters_rated: int
    max_drawdown_r: float
    stop_target_same_bar: int = 0          # losses where one bar touched both: the pessimistic call

    @property
    def quarter_share(self) -> float:
        return self.quarters_positive / self.quarters_rated if self.quarters_rated else 0.0


def measure(entries: list[Entry]) -> Result:
    filled = [e for e in entries if e.status in jr.FINISHED and e.result_r is not None]
    rs = [e.result_r for e in filled]
    s = jr.stats(entries)
    n = len(rs)
    t = None
    if n >= 2:
        mean = sum(rs) / n
        sd = math.sqrt(sum((r - mean) ** 2 for r in rs) / (n - 1))
        t = round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else None
    quarters: dict[str, list[float]] = {}
    for e in filled:
        d = e.created_at.astimezone(NEW_YORK)
        quarters.setdefault(f"{d.year}Q{(d.month - 1) // 3 + 1}", []).append(e.result_r)
    rated = [q for q in quarters.values() if len(q) >= 10]
    return Result(trades=n, expectancy_r=None if s.avg_r is None else round(s.avg_r, 3),
                  profit_factor=None if s.profit_factor is None else round(s.profit_factor, 2),
                  win_rate=None if s.win_rate is None else round(s.win_rate, 3), total_r=round(s.total_r, 2),
                  t_score=t, quarters_positive=sum(1 for q in rated if sum(q) > 0), quarters_rated=len(rated),
                  max_drawdown_r=round(s.max_drawdown_r, 2),
                  stop_target_same_bar=sum(1 for e in filled if "same minute" in (e.note or "")
                                           and e.status == jr.LOST))


def locked_bar(tried_on_locked: int) -> float:
    """The t-score an idea must reach on the locked year, given how many were tried there before."""
    return round(BASE_T + 0.1 * (tried_on_locked // 10), 2)


def design_verdict(r: Result, base: Result) -> tuple[bool, str]:
    if r.trades < DESIGN_MIN_TRADES:
        return False, f"too few trades on the design year ({r.trades}, needs {DESIGN_MIN_TRADES})"
    gain = (r.expectancy_r or 0) - (base.expectancy_r or 0)
    if gain < DESIGN_MIN_GAIN_R:
        return False, f"no clear gain over the unchanged rules ({gain:+.2f}R per trade, needs +{DESIGN_MIN_GAIN_R}R)"
    if (r.expectancy_r or 0) <= 0:
        return False, f"better than the rules ({gain:+.2f}R per trade) but still losing ({r.expectancy_r:+.2f}R)"
    return True, f"{gain:+.2f}R per trade better than the unchanged rules, and profitable"


def locked_verdict(r: Result, bar: float) -> tuple[bool, str]:
    if r.trades < LOCKED_MIN_TRADES:
        return False, f"too few trades on the locked year ({r.trades}, needs {LOCKED_MIN_TRADES})"
    if (r.expectancy_r or 0) <= 0:
        return False, f"lost money on the locked year ({r.expectancy_r:+.2f}R per trade): the design-year gain was luck"
    if (r.profit_factor or 0) < LOCKED_MIN_PF:
        return False, f"profit factor {r.profit_factor} on the locked year, needs {LOCKED_MIN_PF}"
    if r.quarter_share < LOCKED_MIN_QUARTERS:
        return False, (f"profitable in only {r.quarters_positive} of {r.quarters_rated} quarters of the "
                       "locked year: not steady enough")
    if (r.t_score or 0) < bar:
        return False, f"t-score {r.t_score} on the locked year, below the bar of {bar} (could still be luck)"
    return True, (f"held up on the locked year: {r.expectancy_r:+.2f}R per trade over {r.trades} trades, "
                  f"t-score {r.t_score} (bar {bar})")


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

class Ledger:
    """Every idea tried, its results on each year, and the verdict: an honest count of attempts."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    @property
    def tried_on_locked(self) -> int:
        return sum(1 for r in self.rows if r.get("locked") is not None)

    def tried(self, name: str) -> dict | None:
        return next((r for r in self.rows if r["name"] == name), None)

    def add(self, row: dict) -> None:
        self.rows = [r for r in self.rows if r["name"] != row["name"]] + [row]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows, indent=2, default=str), encoding="utf-8")


def run(experiment: Experiment, trade_design, trade_locked, base: Result, ledger: Ledger) -> dict:
    """Try ``experiment`` on the design year, then (if it earns it) on the locked year; record it.

    ``trade_design`` and ``trade_locked`` take an experiment and return its
    settled orders for that year; the locked one is only called on a pass.
    """
    r = measure(trade_design(experiment))
    passed, why = design_verdict(r, base)
    row = {"name": experiment.name, "hypothesis": experiment.hypothesis, "rules": experiment.describe(),
           "spec": asdict(experiment), "at": datetime.now(UTC).isoformat(timespec="minutes"),
           "design": asdict(r), "design_verdict": why, "locked": None, "locked_bar": None,
           "status": "rejected on the design year"}
    if passed:
        bar = locked_bar(ledger.tried_on_locked)
        locked = measure(trade_locked(experiment))
        ok, why_locked = locked_verdict(locked, bar)
        row.update(locked=asdict(locked), locked_bar=bar, locked_verdict=why_locked,
                   status="candidate: paper-trade it next" if ok else "failed on the locked year")
    ledger.add(row)
    return row


# ---------------------------------------------------------------------------
# Quinn's first ideas, each from a reason before a number
# ---------------------------------------------------------------------------

FIRST_BATCH: tuple[Experiment, ...] = (
    Experiment(
        "no-london-open",
        "The first sweep of the London open is often the start of the day's real move, not a stop hunt: "
        "early orders fade a breakout. Skip orders placed 02:00-03:59 New York.",
        skip_hours=(2, 3)),
    Experiment(
        "major-liquidity-only",
        "Stops cluster at the previous day's and the sessions' highs and lows; a sweep of a lone swing point "
        "takes few of them, so the reversal after it has less fuel.",
        min_pool_quality=2),
    Experiment(
        "reachable-targets",
        "The average win is +1.86R although every setup plans 2R or more: far targets are often still open "
        "at the 17:00 close. A target no farther than 2.5R is reached more often before then.",
        target_r=2.5),
    Experiment(
        "strong-displacement",
        "A strong move out of the sweep shows real orders behind the reversal; a weak one is noise that "
        "often re-tests the sweep and runs the stop.",
        min_displacement=2.0),
    Experiment(
        "fvg-entries",
        "A fair value gap marks where the move left imbalance that price tends to refill and respect; an "
        "order block alone is a weaker magnet.",
        zones=("fair value gap", "fair value gap inside the order block")),
    Experiment(
        "hourly-trend-continuation",
        "Trading with a confirmed hourly break of structure (BOS) joins an established trend; after a "
        "change of character (CHoCH) the trend is still unproven.",
        h1_break="BOS"),
    Experiment(
        "skip-the-new-york-open",
        "Orders placed in the 08:00 hour meet the New York open and its data releases, which run stops "
        "both ways; the sweep before it is rarely the final one.",
        skip_hours=(8,)),
)


SECOND_BATCH: tuple[Experiment, ...] = (
    Experiment(
        "wider-stops-1.5x",
        "The anatomy found 22% of losers stopped and then reaching the target the same day, half of them "
        "within an hour: the stop sits where the next sweep goes. A stop half as far again survives that "
        "sweep; each win pays less R, but more trades should reach the target.",
        stop_mult=1.5),
    Experiment(
        "wider-stops-2x",
        "The same reason as wider-stops-1.5x, further: 39% of winners went 0.5R against us first, so price "
        "routinely trades through the first stop distance before the move.",
        stop_mult=2.0),
    Experiment(
        "half-off-at-1r",
        "32% of losers were +1R up before turning into a full loss. Banking half there and moving the rest "
        "to break-even keeps part of those moves.",
        manage="half_off"),
    Experiment(
        "breakeven-at-1r",
        "The same evidence as half-off-at-1r, keeping the whole position for the target but never letting "
        "a +1R trade become a full loss.",
        manage="breakeven"),
    Experiment(
        "wider-stop-and-half-off",
        "Both leaks at once: a stop that survives the next sweep, and half banked at +1R so the wider "
        "stop's smaller R per win is not given back.",
        stop_mult=1.5, manage="half_off"),
)

THIRD_BATCH: tuple[Experiment, ...] = (
    Experiment(
        "far-side-stop",
        "The breakout makes +0.06R before the spread but the spread is 16% of each trade's risk. A stop "
        "at the far side of the Asian range doubles the risk, halves the spread's share, and is where the "
        "break is truly wrong; the target stays 2R of the new risk.",
        stop_mult=2.0, retarget_r=2.0),
    Experiment(
        "spread-under-10pct",
        "Costs, not direction, sink the breakout: skip any setup whose spread is more than a tenth of its "
        "risk. A rule about costs, not about which pairs did well last year.",
        max_spread_share=0.10),
    Experiment(
        "far-stop-low-cost",
        "Both cost cuts at once: the far-side stop, then only setups where the spread is under a tenth of "
        "that larger risk.",
        stop_mult=2.0, retarget_r=2.0, max_spread_share=0.10),
    Experiment(
        "far-stop-nearer-target",
        "The far-side stop with a 1.5R target: the wider risk makes 2R a long way for one session (18% of "
        "breakout trades were still open at 16:55), so a nearer target is reached more often.",
        stop_mult=2.0, retarget_r=1.5),
)

BATCHES = {1: FIRST_BATCH, 2: SECOND_BATCH, 3: THIRD_BATCH}
BATCH_MODELS = {1: ("smc",), 2: ("smc",), 3: ("asian", "asian-market")}   # the model each batch was written for


def report(rows: list[dict], base_design: Result, base_locked: Result | None, ledger: Ledger) -> str:
    lines = ["# Quinn's lab · experiments", "",
             f"Unchanged rules on the design year: {base_design.expectancy_r:+.2f}R per trade over "
             f"{base_design.trades} trades (PF {base_design.profit_factor}).",
             f"Ideas tried on the locked year so far: {ledger.tried_on_locked} "
             f"(next bar: t-score {locked_bar(ledger.tried_on_locked)}).", "",
             "| Idea | Rule | Design: trades | Per trade | PF | Locked: per trade | t | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d, lk = r["design"], r["locked"]
        locked = "—" if lk is None else f"{lk['expectancy_r']:+.2f}R ({lk['trades']})"
        t = "—" if lk is None else f"{lk['t_score']} / {r['locked_bar']}"
        exp = "—" if d["expectancy_r"] is None else f"{d['expectancy_r']:+.2f}R"
        lines.append(f"| {r['name']} | {r['rules']} | {d['trades']} | {exp} | {d['profit_factor']} | {locked} | {t} "
                     f"| {r['status']} |")
    lines += ["", "## Why each idea was tried, and what happened", ""]
    for r in rows:
        lines += [f"### {r['name']}", "", r["hypothesis"], "", f"- Design year: {r['design_verdict']}"]
        if r["locked"] is not None:
            lines.append(f"- Locked year: {r['locked_verdict']}")
        lines.append("")
    if base_locked is not None:
        lines += [f"Unchanged rules on the locked year, for reference: {base_locked.expectancy_r:+.2f}R per "
                  f"trade over {base_locked.trades} trades.", ""]
    lines += ["A candidate is not a live rule: it is paper-traded next, and goes live only with your approval.", ""]
    return "\n".join(lines)


def telegram_summary(rows: list[dict]) -> str:
    out = ["🧪 <b>Quinn's lab · experiments</b>"]
    for r in rows:
        d = r["design"]
        exp = "—" if d["expectancy_r"] is None else f"{d['expectancy_r']:+.2f}R"
        mark = "✅" if r["status"].startswith("candidate") else "✕"
        out.append(f"{mark} {r['name']}: {exp} on {d['trades']} trades · {r['status']}")
    return "\n".join(out)
