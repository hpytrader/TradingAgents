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
        "smc", "--strategy", help="smc: H1 bias, M5 sweep → CHoCH → order block / FVG, inside the scan window. "
                                  "trend: H4 trend, H1 pullback to a swing level or the 50 EMA"
    ),
    window: str = typer.Option(
        "02:00-12:00", "--window", help="smc: hours to build setups in, New York (= Toronto) time; may cross midnight"
    ),
    any_session: bool = typer.Option(
        False, "--any-session", help="smc: scan outside the window too"
    ),
    valid_hours: float = typer.Option(
        None, "--valid-hours", help="Hours before an unfilled limit order is cancelled (smc: 4, capped at the window's close; trend: 8)"
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
    from tradingagents.fx.smc_scanner import ScanWindow
    try:
        scan_window = ScanWindow.parse(window)
    except ValueError as exc:
        console.print(f"[red]--window: {exc}[/red]")
        raise typer.Exit(code=1) from None
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
                              max_per_currency=scan_cap, window=scan_window,
                              any_session=any_session)
        else:
            result = scan(oanda.get_candles, oanda.get_quote, names,
                          min_rr=min_rr, top=scan_top, valid_hours=valid_hours,
                          max_per_currency=scan_cap, stop_atr=stop_atr)

    if not result.ran:
        # Outside the window nothing was scanned: say when it opens, and do
        # not save an empty report or run the agents.
        console.print(f"[yellow]{result.skipped[0][1]}[/yellow]")
        return

    _print_scan(result, quiet=agents)
    review = None
    if agents:
        review = _run_fx_agents(result, final=final, max_per_currency=max_per_currency, min_rr=min_rr)
    if review is not None and review.orders:
        console.print(f"\n[dim]{AGENT_DISCLAIMER}[/dim]")
    elif review is None and result.setups:
        console.print(f"\n[dim]{DISCLAIMER}[/dim]")

    md = None
    if save:
        md, _ = save_scan(result, DEFAULT_CONFIG["results_dir"], review)
        console.print(f"Saved: {md}")
    if review is not None and review.orders:
        from datetime import UTC, datetime

        added = _fx_journal().record(review.orders, now=datetime.now(UTC), report=str(md or ""))
        repeat = len(review.orders) - len(added)
        console.print(f"Journal: {len(added)} order(s) added"
                      + (f", {repeat} already pending or open" if repeat else "")
                      + ". See them with: tradingagents fx-journal --open")


def _fx_journal():
    from pathlib import Path

    from tradingagents.fx.journal import Journal

    return Journal(Path(DEFAULT_CONFIG["results_dir"]) / "fx_journal.db")


def _fx_dashboard_path():
    from pathlib import Path

    return Path(DEFAULT_CONFIG["results_dir"]) / "fx_dashboard.html"


def _plain(html_text: str) -> str:
    import re
    from html import unescape

    return unescape(re.sub(r"<[^>]+>", "", html_text))


def _require_oanda(symbol: str):
    from tradingagents.dataflows.errors import VendorNotConfiguredError
    from tradingagents.dataflows.vendors import oanda

    try:
        oanda.get_quote(symbol)
    except VendorNotConfiguredError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except Exception:
        pass


@app.command("fx-journal")
def fx_journal(
    settle: bool = typer.Option(True, "--settle/--no-settle", help="Replay OANDA prices to update open orders first"),
    open_page: bool = typer.Option(False, "--open", help="Open the dashboard in your browser"),
    limit: int = typer.Option(15, "--limit", help="How many recent orders to list"),
    import_reports: bool = typer.Option(
        False, "--import-reports", help="Add final orders from earlier saved agent reports (before the journal existed)"
    ),
):
    """The paper-trade journal of the agents' final orders: results, scorecard and dashboard."""
    import webbrowser
    from datetime import UTC, datetime

    from rich.table import Table

    from tradingagents.dataflows.vendors import oanda
    from tradingagents.fx import dashboard
    from tradingagents.fx.journal import stats
    from tradingagents.fx.telegram import result_message

    book = _fx_journal()
    now = datetime.now(UTC)
    if import_reports:
        from pathlib import Path

        folder = Path(DEFAULT_CONFIG["results_dir"]) / "fx_scans"
        added = [e for path in sorted(folder.glob("*_agents.json")) for e in book.import_report(path)]
        console.print(f"Imported {len(added)} order(s) from saved agent reports"
                      + (": " + ", ".join(f"{e.symbol} ({e.created_at:%d %b %H:%M} UTC)" for e in added)
                         if added else "."))
    if settle and book.entries(("pending", "open")):
        _require_oanda("EURUSD")
        with console.status("Replaying prices for pending and open orders..."):
            changes = book.settle(oanda.get_candles, now)
        for change in changes:
            console.print(_plain(result_message(change.entry)))

    entries = book.entries()
    s = stats(entries)
    if not entries:
        console.print("[yellow]The journal is empty. Orders are added by "
                      "`tradingagents fx-scan --agents` and `tradingagents fx-watch`.[/yellow]")
    else:
        table = Table(title=f"FX journal: {s.total} orders")
        for col in ("Suggested (NY)", "Symbol", "Order", "Entry", "Stop", "Target", "Status", "Result"):
            table.add_column(col, justify="right" if col in ("Entry", "Stop", "Target", "Result") else "left")
        from tradingagents.fx.smc import NEW_YORK
        for e in sorted(entries, key=lambda e: e.created_at, reverse=True)[:limit]:
            colour = {"won": "green", "lost": "red"}.get(e.status, "white")
            table.add_row(e.created_at.astimezone(NEW_YORK).strftime("%a %d %b %H:%M"), e.symbol, e.order_type,
                          str(e.entry), str(e.stop), str(e.target), f"[{colour}]{e.status}[/{colour}]",
                          "—" if e.result_r is None else f"{e.result_r:+.2f}R")
        console.print(table)
        if s.finished:
            console.print(f"Filled {s.finished}: win rate {s.win_rate:.0%} · total {s.total_r:+.2f}R · "
                          f"expectancy {s.avg_r:+.2f}R · max drawdown {s.max_drawdown_r:.2f}R")
        console.print(f"Open {s.open} · pending {s.pending} · expired {s.expired} · missed {s.missed}")
        if s.finished < 30:
            console.print("[dim]Fewer than 30 finished trades: too few to judge the win rate yet.[/dim]")
    path = dashboard.write(_fx_dashboard_path(), entries, s, now=now)
    console.print(f"Dashboard: {path}")
    if open_page:
        webbrowser.open(path.as_uri())


@app.command("fx-telegram")
def fx_telegram():
    """Set up and test Telegram alerts (finds your chat id, then sends a test message)."""
    from tradingagents.fx import telegram

    if not telegram._token():
        console.print(
            "1. In Telegram, message [bold]@BotFather[/bold], send /newbot and follow the prompts.\n"
            "2. Copy the token it gives you into .env as [bold]TELEGRAM_BOT_TOKEN=...[/bold]\n"
            "3. Send your new bot any message, then run this command again.")
        raise typer.Exit(code=1)
    try:
        if not telegram.chat_id():
            chats = telegram.find_chats()
            if not chats:
                console.print("[yellow]The bot has no messages yet. Send it any message in Telegram, "
                              "then run this again.[/yellow]")
                raise typer.Exit(code=1)
            console.print("Add this line to .env, then run this command again to send a test message:")
            for chat, name in chats:
                console.print(f"  [bold]TELEGRAM_CHAT_ID={chat}[/bold]   ({name})")
            return
        telegram.send("<b>FX desk</b>\nTelegram alerts are working. New orders and their results will arrive here.")
    except telegram.TelegramError as exc:
        console.print(f"[red]Telegram: {exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print("[green]Test message sent. Check Telegram.[/green]")


@app.command("fx-watch")
def fx_watch(
    interval: int = typer.Option(10, "--interval", help="Minutes between scans inside the window"),
    max_agent_runs: int = typer.Option(12, "--max-agent-runs", help="Most agent reviews per trading day (cost cap)"),
    final: int = typer.Option(6, "--final", help="The most orders in a review's final book"),
    min_rr: float = typer.Option(2.0, "--min-rr", help="Minimum reward-to-risk after the spread"),
    max_per_currency: int = typer.Option(2, "--max-per-currency", help="Most orders long, or short, one currency"),
    window: str = typer.Option("02:00-12:00", "--window", help="Hours to build setups in, New York (= Toronto) time"),
    symbols: str = typer.Option(None, "--symbols", help="Comma-separated pairs; omit for the default list"),
    open_page: bool = typer.Option(False, "--open", help="Open the dashboard in your browser at the start"),
):
    """Watch the market all morning: scan every few minutes, review new setups with the agents,
    journal the orders, settle them, and send alerts to Telegram."""
    import time as clock
    import webbrowser
    from datetime import UTC, datetime, timedelta

    from tradingagents.dataflows.vendors import oanda
    from tradingagents.fx import DEFAULT_UNIVERSE, dashboard, telegram
    from tradingagents.fx.journal import stats
    from tradingagents.fx.smc import NEW_YORK
    from tradingagents.fx.smc_scanner import ScanWindow, scan_smc
    from tradingagents.fx.watch import WatchState, cycle

    try:
        scan_window = ScanWindow.parse(window)
    except ValueError as exc:
        console.print(f"[red]--window: {exc}[/red]")
        raise typer.Exit(code=1) from None
    if interval < 1 or max_agent_runs < 0 or final < 1 or max_per_currency < 1:
        console.print("[red]--interval and --final must be at least 1; --max-agent-runs at least 0.[/red]")
        raise typer.Exit(code=1)
    names = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else list(DEFAULT_UNIVERSE)
    _require_oanda(names[0])

    alerts = telegram.configured()

    def notify(text: str):
        console.print(_plain(text))
        if alerts:
            try:
                telegram.send(text)
            except telegram.TelegramError as exc:
                console.print(f"[yellow]Telegram: {exc}[/yellow]")

    def scan_fn(now):
        return scan_smc(oanda.get_candles, oanda.get_quote, names, min_rr=min_rr, top=final + 4,
                        max_per_currency=max_per_currency + 1, window=scan_window, now=now)

    review_fn = _fx_reviewer(final=final, max_per_currency=max_per_currency, min_rr=min_rr)
    book = _fx_journal()
    state = WatchState()
    page = _fx_dashboard_path()

    console.print(f"[bold]FX watcher[/bold] · window {scan_window.label()} · every {interval} min · "
                  f"up to {max_agent_runs} agent reviews a day")
    console.print("Telegram alerts: " + ("[green]on[/green]" if alerts else
                  "[yellow]off[/yellow] (run `tradingagents fx-telegram` to set them up)"))
    console.print(f"Dashboard: {page}\n[dim]Keep this window open and the Mac awake "
                  "(start it with `caffeinate -i tradingagents fx-watch`). Ctrl+C stops it.[/dim]\n")
    dashboard.write(page, book.entries(), stats(book.entries()), live=True, window=scan_window)
    if open_page:
        webbrowser.open(page.as_uri())

    try:
        while True:
            now = datetime.now(UTC)
            c = cycle(now, window=scan_window, journal=book, candles=oanda.get_candles, scan_fn=scan_fn,
                      review_fn=review_fn, notify=notify, state=state, max_agent_runs=max_agent_runs)
            stamp = now.astimezone(NEW_YORK).strftime("%H:%M")
            parts = [f"[dim]{stamp} NY[/dim]", "window open" if c.in_window else "window closed"]
            if c.in_window:
                parts.append(f"{c.candidates} candidate(s), {c.new_setups} new")
                if c.reviewed:
                    parts.append(f"agents reviewed → {len(c.new_orders)} new order(s)")
            parts += c.notes
            console.print(" · ".join(parts))
            entries = book.entries()
            dashboard.write(page, entries, stats(entries), now=now, live=True, window=scan_window,
                            last_cycle=f"{stamp} NY")

            if c.in_window:
                wait = timedelta(minutes=interval)
            elif any(e.status in ("pending", "open") for e in entries):
                wait = timedelta(minutes=15)          # keep settling open trades until the close
            else:
                wait = min(scan_window.next_open(now) - now, timedelta(hours=1))
            clock.sleep(max(wait.total_seconds(), 30))
    except KeyboardInterrupt:
        entries = book.entries()
        dashboard.write(page, entries, stats(entries), window=scan_window)
        console.print("\nWatcher stopped.")


def _fx_reviewer(*, final: int, max_per_currency: int, min_rr: float):
    """A reviewer for the watcher: models built once, report saved, no exit on failure."""
    from tradingagents.fx import agents as fx_agents
    from tradingagents.fx.context import gather
    from tradingagents.fx.report import save as save_scan
    from tradingagents.llm_clients.factory import create_tier_client

    models = {}

    def review(result, now):
        if not models:
            _quiet_fx_logs()
            models["quick"] = create_tier_client(DEFAULT_CONFIG, "quick").get_llm()
            models["deep"] = create_tier_client(DEFAULT_CONFIG, "deep").get_llm()
        context = gather(result.setups, now)
        outcome = fx_agents.review(result.setups, context, models["quick"], models["deep"],
                                   max_orders=final, max_per_currency=max_per_currency, min_rr=min_rr, now=now)
        for problem in outcome.problems:
            console.print(f"[yellow]Model problem: {problem}[/yellow]")
        md, _ = save_scan(result, DEFAULT_CONFIG["results_dir"], outcome)
        return outcome, str(md)

    return review


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
