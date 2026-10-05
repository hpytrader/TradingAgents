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
    top: int = typer.Option(10, "--top", help="How many setups to keep"),
    min_rr: float = typer.Option(2.0, "--min-rr", help="Minimum reward-to-risk after the spread"),
    valid_hours: float = typer.Option(8.0, "--valid-hours", help="Hours before an unfilled limit order should be cancelled"),
    max_per_currency: int = typer.Option(
        2, "--max-per-currency", help="Most setups long, or short, the same currency (e.g. 3 on a strong-dollar day)"
    ),
    stop_atr: float = typer.Option(
        1.0, "--stop-atr", help="Stop distance beyond the entry level, in 1-hour ATRs"
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Save the scan as Markdown and JSON"),
):
    """Scan forex and metals for intraday limit-order setups (no AI, uses OANDA prices)."""
    from rich.table import Table

    from tradingagents.dataflows.errors import VendorNotConfiguredError
    from tradingagents.dataflows.vendors import oanda
    from tradingagents.fx import DEFAULT_UNIVERSE, scan
    from tradingagents.fx.report import DISCLAIMER, save as save_scan

    names = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else list(DEFAULT_UNIVERSE)
    if max_per_currency < 1 or stop_atr <= 0 or top < 1:
        console.print("[red]--max-per-currency and --top must be at least 1, and --stop-atr above 0.[/red]")
        raise typer.Exit(code=1)
    try:
        oanda.get_quote(names[0])          # fail fast on a missing or refused token
    except VendorNotConfiguredError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except Exception:
        pass                               # a problem with one pair is reported by the scan

    with console.status(f"Scanning {len(names)} instruments..."):
        result = scan(oanda.get_candles, oanda.get_quote, names,
                      min_rr=min_rr, top=top, valid_hours=valid_hours,
                      max_per_currency=max_per_currency, stop_atr=stop_atr)

    if result.setups:
        table = Table(title=f"Setups — {result.scanned_at:%Y-%m-%d %H:%M} UTC")
        for column in ("#", "Symbol", "Order", "Entry", "Stop", "Target", "RR", "Risk pips", "Score"):
            table.add_column(column, justify="right" if column not in ("Symbol", "Order") else "left")
        for i, s in enumerate(result.setups, 1):
            colour = "green" if s.direction == "long" else "red"
            table.add_row(str(i), s.symbol, f"[{colour}]{s.order_type}[/{colour}]", str(s.entry),
                          str(s.stop), str(s.target), f"{s.rr:.2f}", str(s.risk_pips), f"{s.score:.0f}")
        console.print(table)
    else:
        console.print("[yellow]No setups met the rules right now.[/yellow]")
    for symbol, reason in result.skipped:
        console.print(f"[dim]{symbol}: {reason}[/dim]")
    console.print(f"\n[dim]{DISCLAIMER}[/dim]")

    if save:
        md, _ = save_scan(result, DEFAULT_CONFIG["results_dir"])
        console.print(f"Saved: {md}")


if __name__ == "__main__":
    app()
