import sys

import typer

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to analyze, e.g. NVDA or 0700.HK; skips the prompt"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
    html: bool | None = typer.Option(
        None, "--html/--no-html",
        help="Save the report as one HTML page too, complete_report.html, without asking (default: yes)",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models, depth and language also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show, "html": html}
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run: market, sentiment, news, fundamentals; omit for all the asset type allows"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {"asset_type": kind.value, "portfolio": book, "run_id": run_id, "progress": show_progress,
              "selected_analysts": [a.value for a in chosen]}

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


@app.command("fx-scan")
def fx_scan(
    symbols: str = typer.Option(
        None, "--symbols", help="Comma-separated pairs, e.g. EURUSD,XAUUSD; omit for the default majors, crosses and metals"
    ),
    top: int = typer.Option(10, "--top", help="How many setups the scanner keeps"),
    min_rr: float = typer.Option(2.0, "--min-rr", help="Minimum reward-to-risk after the spread"),
    strategy: str = typer.Option(
        "smc", "--strategy", help="smc: H1 bias, M5 sweep → CHoCH → order block / FVG (London and NY sessions). "
                                  "trend: H4 trend, H1 pullback to a swing level or the 50 EMA"
    ),
    any_session: bool = typer.Option(
        False, "--any-session", help="smc: scan outside the London and New York sessions too"
    ),
    valid_hours: float = typer.Option(
        None, "--valid-hours", help="Hours before an unfilled limit order is cancelled (smc: 4, capped at the session end; trend: 8)"
    ),
    max_per_currency: int = typer.Option(
        2, "--max-per-currency", help="Most setups long, or short, the same currency (e.g. 3 on a strong-dollar day)"
    ),
    stop_atr: float = typer.Option(
        1.0, "--stop-atr", help="Stop distance beyond the entry level, in 1-hour ATRs"
    ),
    agents: bool = typer.Option(
        False, "--agents", help="Have the AI agent team review the candidates and pick the final orders (uses your AI key)"
    ),
    final: int = typer.Option(6, "--final", help="With --agents: the most orders in the final book"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save the scan as Markdown and JSON"),
):
    """Scan forex and metals for intraday limit-order setups (OANDA prices; --agents adds the AI review)."""
    strategy = strategy.strip().lower()
    if strategy not in ("smc", "trend"):
        console.print("[red]--strategy must be smc or trend.[/red]")
        raise typer.Exit(code=1)
    if valid_hours is None:
        valid_hours = 4.0 if strategy == "smc" else 8.0
    from tradingagents.dataflows.errors import VendorNotConfiguredError
    from tradingagents.dataflows.vendors import oanda
    from tradingagents.fx import DEFAULT_UNIVERSE, scan
    from tradingagents.fx.report import AGENT_DISCLAIMER, DISCLAIMER, save as save_scan
    from tradingagents.fx.smc_scanner import scan_smc

    names = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else list(DEFAULT_UNIVERSE)
    if max_per_currency < 1 or stop_atr <= 0 or top < 1 or final < 1:
        console.print("[red]--max-per-currency, --top and --final must be at least 1, and --stop-atr above 0.[/red]")
        raise typer.Exit(code=1)
    try:
        oanda.get_quote(names[0])          # fail fast on a missing or refused token
    except VendorNotConfiguredError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except Exception:
        pass                               # a problem with one pair is reported by the scan

    # The agents choose from a wider field than the final book allows, so a
    # setup they reject can be replaced; the book's own cap is enforced later.
    scan_top = max(top, final + 4) if agents else top
    scan_cap = max_per_currency + 1 if agents else max_per_currency
    with console.status(f"Scanning {len(names)} instruments..."):
        if strategy == "smc":
            result = scan_smc(oanda.get_candles, oanda.get_quote, names,
                              min_rr=min_rr, top=scan_top, valid_hours=valid_hours,
                              max_per_currency=scan_cap, any_session=any_session)
        else:
            result = scan(oanda.get_candles, oanda.get_quote, names,
                          min_rr=min_rr, top=scan_top, valid_hours=valid_hours,
                          max_per_currency=scan_cap, stop_atr=stop_atr)

    _print_scan(result, quiet=agents)
    review = None
    if agents:
        review = _run_fx_agents(result, final=final, max_per_currency=max_per_currency, min_rr=min_rr)
    console.print(f"\n[dim]{AGENT_DISCLAIMER if agents else DISCLAIMER}[/dim]")

    if save:
        md, _ = save_scan(result, DEFAULT_CONFIG["results_dir"], review)
        console.print(f"Saved: {md}")


def _print_scan(result, quiet: bool = False):
    from rich.table import Table

    title = "Scanner candidates" if quiet else "Setups"
    if result.setups:
        table = Table(title=f"{title} — {result.scanned_at:%Y-%m-%d %H:%M} UTC")
        for column in ("#", "Symbol", "Order", "Entry", "Stop", "Target", "RR", "Risk pips", "Score"):
            table.add_column(column, justify="right" if column not in ("Symbol", "Order") else "left")
        for i, s in enumerate(result.setups, 1):
            colour = "green" if s.direction == "long" else "red"
            table.add_row(str(i), s.symbol, f"[{colour}]{s.order_type}[/{colour}]", str(s.entry),
                          str(s.stop), str(s.target), f"{s.rr:.2f}", str(s.risk_pips), f"{s.score:.0f}")
        console.print(table)
    else:
        console.print("[yellow]No setups met the rules right now.[/yellow]")
    if not quiet or not result.setups:
        for symbol, reason in result.skipped:
            console.print(f"[dim]{symbol}: {reason}[/dim]")


def _quiet_fx_logs():
    """Hide library chatter during the review; its problems are reported in one line each.

    Yahoo's "no news" warnings and Google's notes to developers would otherwise
    print over the progress spinner, and a quota error would print its whole
    JSON payload.
    """
    import logging
    import warnings

    for name in ("tradingagents.dataflows.router", "tradingagents.agents.structured",
                 "google_genai", "google_genai.models", "google.genai", "httpx", "yfinance"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"google\..*")


def _run_fx_agents(result, *, final: int, max_per_currency: int, min_rr: float):
    """Stage 2: the agent team reviews the candidates. Returns the Review, or exits."""
    from datetime import UTC, datetime

    from rich.table import Table

    from cli.stats_handler import StatsCallbackHandler
    from tradingagents.fx import agents as fx_agents
    from tradingagents.fx.context import gather
    from tradingagents.llm_clients.factory import create_tier_client

    if not result.setups:
        console.print("[yellow]No candidates, so there is nothing for the agents to review.[/yellow]")
        return fx_agents.Review(orders=[], dropped=[], summary="The scanner found no candidates.")

    _quiet_fx_logs()
    stats = StatsCallbackHandler()
    try:
        quick = create_tier_client(DEFAULT_CONFIG, "quick", callbacks=[stats]).get_llm()
        deep = create_tier_client(DEFAULT_CONFIG, "deep", callbacks=[stats]).get_llm()
    except Exception as exc:
        console.print(f"[red]Could not start the AI models: {exc}[/red]")
        console.print("[dim]Check the provider, models and API key in your .env file.[/dim]")
        raise typer.Exit(code=1) from None

    now = datetime.now(UTC)
    with console.status("Reading the economic calendar and headlines...") as status:
        context = gather(result.setups, now)
        try:
            review = fx_agents.review(
                result.setups, context, quick, deep,
                max_orders=final, max_per_currency=max_per_currency, min_rr=min_rr,
                progress=lambda msg: status.update(f"{msg}..."), now=now,
            )
        except Exception as exc:
            console.print(f"[red]The agent review stopped: {exc}[/red]")
            console.print("[dim]The scanner candidates above are still valid; the report is saved without the review.[/dim]")
            return None

    if context.calendar_note:
        console.print(f"[yellow]{context.calendar_note}[/yellow]")
    missing_news = [sym for sym, text in context.news.items()
                    if text.startswith(("(news unavailable", "(FXStreet: no headlines"))]
    if missing_news:
        console.print(f"[dim]No headlines found for {', '.join(missing_news)}; the agents were told so.[/dim]")
    for problem in review.problems:
        console.print(f"[yellow]Model problem: {problem}[/yellow]")
    for note in review.fallbacks:
        console.print(f"[yellow]Partial review: {note}[/yellow]")
    if any("quota" in p for p in review.problems):
        console.print("[dim]A free-tier daily quota resets after about 24 hours; a paid key removes the limit.[/dim]")

    if review.orders:
        table = Table(title=f"Final orders ({len(review.orders)} of up to {final})")
        for column in ("#", "Symbol", "Order", "Entry", "Stop", "Target", "RR", "Cancel by", "Conviction"):
            table.add_column(column, justify="right" if column not in ("Symbol", "Order", "Conviction") else "left")
        for i, o in enumerate(review.orders, 1):
            colour = "green" if o.direction == "long" else "red"
            table.add_row(str(i), o.symbol, f"[{colour}]{o.order_type}[/{colour}]", str(o.entry),
                          str(o.stop), str(o.target), f"{o.rr:.2f}",
                          f"{o.expires_at:%H:%M} UTC", o.conviction)
        console.print(table)
        for i, o in enumerate(review.orders, 1):
            console.print(f"[bold]{i}. {o.symbol}[/bold] {o.rationale}")
            console.print(f"   [dim]Watch for: {o.watch_for}[/dim]")
            for note in o.notes:
                console.print(f"   [yellow]{note}[/yellow]")
    else:
        console.print("[yellow]The agents chose no orders today.[/yellow]")
    console.print(f"\n{review.summary}")
    for symbol, reason in review.dropped:
        console.print(f"[dim]{symbol}: {reason}[/dim]")

    used = stats.get_stats()
    console.print(f"\n[dim]AI usage: {used['llm_calls']} calls, {used['tokens_in']:,} tokens in, "
                  f"{used['tokens_out']:,} out[/dim]")
    return review


if __name__ == "__main__":
    app()
