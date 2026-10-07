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
        None, "--valid-hours", help="Hours before an unfilled limit order is cancelled (smc: 4, and by --latest-cancel; trend: 8)"
    ),
    latest_cancel: str = typer.Option(
        "14:00", "--latest-cancel", help="smc: cancel unfilled orders by this New York time at the latest"
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
    cancel_by = _parse_clock(latest_cancel, "--latest-cancel")
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
                              any_session=any_session, latest_cancel=cancel_by)
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
    if review is not None and review.messages:
        from datetime import UTC, datetime

        from tradingagents.dataflows.vendors import oanda
        from tradingagents.fx.watch import file_review

        added, applied = file_review(review, _fx_journal(), now=datetime.now(UTC), report=str(md or ""),
                                     candles=oanda.get_candles)
        for a in applied:
            console.print(("[green]✓[/green] " if a.ok else "[yellow]✕[/yellow] ") + a.text)
        if review.orders:
            repeat = len(review.orders) - len(added)
            console.print(f"Journal: {', '.join(e.label for e in added) or 'no new orders'}"
                          + (f" ({repeat} already pending or open)" if repeat else ""))
        console.print("Desk chat saved: see it with tradingagents fx-journal --open")


def _parse_clock(text: str, flag: str):
    from datetime import time

    try:
        return time.fromisoformat(text.strip())
    except ValueError:
        console.print(f"[red]{flag} must be a time like 14:00, not {text!r}.[/red]")
        raise typer.Exit(code=1) from None


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
    path = dashboard.write(_fx_dashboard_path(), entries, s, now=now, reviews=book.reviews())
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
    ward_every: int = typer.Option(
        30, "--ward-every", help="Minutes between the trade manager's own checks of open and pending trades (0 = only in reviews)"
    ),
    final: int = typer.Option(6, "--final", help="The most orders in a review's final book"),
    min_rr: float = typer.Option(2.0, "--min-rr", help="Minimum reward-to-risk after the spread"),
    max_per_currency: int = typer.Option(2, "--max-per-currency", help="Most orders long, or short, one currency"),
    window: str = typer.Option("02:00-12:00", "--window", help="Hours to build setups in, New York (= Toronto) time"),
    latest_cancel: str = typer.Option("14:00", "--latest-cancel", help="Cancel unfilled orders by this New York time at the latest"),
    symbols: str = typer.Option(None, "--symbols", help="Comma-separated pairs; omit for the default list"),
    open_page: bool = typer.Option(False, "--open", help="Open the dashboard in your browser at the start"),
    mt4: bool = typer.Option(False, "--mt4", help="Also place and manage the orders in MT4 (set up with `tradingagents fx-mt4 --setup`)"),
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
    cancel_by = _parse_clock(latest_cancel, "--latest-cancel")
    if interval < 1 or max_agent_runs < 0 or final < 1 or max_per_currency < 1 or ward_every < 0:
        console.print("[red]--interval and --final must be at least 1; --max-agent-runs and --ward-every at least 0.[/red]")
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
                        max_per_currency=max_per_currency + 1, window=scan_window, now=now,
                        latest_cancel=cancel_by)

    models: dict = {}
    review_fn = _fx_reviewer(final=final, max_per_currency=max_per_currency, min_rr=min_rr, models=models)
    ward_fn = _fx_ward(models) if ward_every else None
    book = _fx_journal()
    state = WatchState()
    page = _fx_dashboard_path()

    console.print(f"[bold]FX watcher[/bold] · window {scan_window.label()} · every {interval} min · "
                  f"up to {max_agent_runs} agent reviews a day")
    from tradingagents.fx.team import team
    ward_name = team()["trade_manager"].name
    console.print(f"{ward_name} checks open and pending trades "
                  + (f"every {ward_every} min, until they finish" if ward_every else "only during agent reviews"))
    console.print("Telegram alerts: " + ("[green]on[/green]" if alerts else
                  "[yellow]off[/yellow] (run `tradingagents fx-telegram` to set them up)"))
    bridge = _fx_bridge(book) if mt4 else None
    console.print(f"Dashboard: {page}\n[dim]Keep this window open and the Mac awake "
                  "(start it with `caffeinate -i tradingagents fx-watch`). Ctrl+C stops it.[/dim]\n")
    dashboard.write(page, book.entries(), stats(book.entries()), live=True, window=scan_window,
                    reviews=book.reviews())
    if open_page:
        webbrowser.open(page.as_uri())

    try:
        while True:
            now = datetime.now(UTC)
            c = cycle(now, window=scan_window, journal=book, candles=oanda.get_candles, scan_fn=scan_fn,
                      review_fn=review_fn, notify=notify, state=state, max_agent_runs=max_agent_runs,
                      ward_fn=ward_fn, ward_every=timedelta(minutes=ward_every or 30))
            stamp = now.astimezone(NEW_YORK).strftime("%H:%M")
            parts = [f"[dim]{stamp} NY[/dim]", "window open" if c.in_window else "window closed"]
            if c.in_window:
                parts.append(f"{c.candidates} candidate(s), {c.new_setups} new")
                if c.reviewed:
                    parts.append(f"agents reviewed → {len(c.new_orders)} new order(s)")
            if c.ward_checked:
                parts.append(f"{ward_name} checked the live trades")
            parts += c.notes
            if bridge is not None:
                parts += _fx_bridge_sync(bridge, datetime.now(UTC), notify)   # the cycle may have taken minutes
            console.print(" · ".join(parts))
            entries = book.entries()
            dashboard.write(page, entries, stats(entries), now=now, live=True, window=scan_window,
                            last_cycle=f"{stamp} NY", reviews=book.reviews())

            if c.in_window:
                wait = timedelta(minutes=interval)
            elif any(e.status in ("pending", "open") for e in entries):
                wait = timedelta(minutes=15)          # keep settling open trades until the close
            else:
                wait = min(scan_window.next_open(now) - now, timedelta(hours=1))
            until = now + max(wait, timedelta(seconds=30))
            while datetime.now(UTC) < until:          # MT4 fills and closes are reported within ~20 s
                clock.sleep(min(20.0, max((until - datetime.now(UTC)).total_seconds(), 0.1)))
                if bridge is not None:
                    for note in _fx_bridge_sync(bridge, datetime.now(UTC), notify):
                        console.print(f"[dim]MT4: {note}[/dim]")
    except KeyboardInterrupt:
        entries = book.entries()
        dashboard.write(page, entries, stats(entries), window=scan_window, reviews=book.reviews())
        console.print("\nWatcher stopped.")


def _fx_bridge(book):
    """The MT4 bridge from ~/.tradingagents/fx_mt4.json, or exit with how to set it up."""
    from datetime import UTC, datetime

    from tradingagents.fx.mt4 import Bridge, Mt4Config, config_path

    cfg = Mt4Config.load()
    if cfg is None or not cfg.files_dir:
        console.print(f"[red]MT4 is not set up yet ({config_path()} missing). "
                      "Run `tradingagents fx-mt4 --setup` first.[/red]")
        raise typer.Exit(code=1)
    bridge = Bridge(cfg.files_dir, book, lots=cfg.lots, symbols=cfg.symbols)
    state = bridge.state()
    ok, why = bridge.healthy(datetime.now(UTC), state)
    if state is not None:
        kind = "[yellow]DEMO[/yellow]" if state.demo else "[bold red]LIVE[/bold red]"
        console.print(f"MT4: {kind} account {state.account} on {state.server} · "
                      f"{state.balance:,.2f} {state.currency} · {cfg.lots:g} lot per order · EA cap {state.max_lots:g}")
    console.print("MT4 bridge: " + ("[green]ready[/green]" if ok else f"[yellow]{why}[/yellow]"))
    return bridge


def _fx_bridge_sync(bridge, now, notify) -> list[str]:
    try:
        return bridge.sync(now, notify)
    except Exception as exc:                      # a file problem must not stop the watcher
        return [f"MT4 sync failed: {exc}"]


@app.command("fx-mt4")
def fx_mt4(
    setup: bool = typer.Option(False, "--setup", help="Find MT4's files folder, save it, and install the EA"),
    folder: str = typer.Option(None, "--folder", help="MT4's MQL4/Files folder, if --setup can't find it"),
    lots: float = typer.Option(None, "--lots", help="Lot size for every order (default 0.01)"),
    symbol: str = typer.Option(None, "--symbol", help="Map symbols to the broker's names, e.g. XAUUSD=GOLD,XAGUSD=SILVER"),
    flatten: bool = typer.Option(False, "--flatten", help="Cancel every pending order and close every trade of the desk in MT4 now"),
    test_order: str = typer.Option(None, "--test-order", help="Place a buy limit 1% under the price on this symbol (e.g. EURUSD or USDJPY.r) and cancel it at once"),
):
    """Connect the desk to MetaTrader 4 (e.g. CMC Markets): setup, status, a test order, and an emergency flatten."""
    import time as clock
    from datetime import UTC, datetime
    from pathlib import Path

    from tradingagents.fx import DEFAULT_UNIVERSE
    from tradingagents.fx.mt4 import (
        Bridge,
        Mt4Config,
        config_path,
        find_files_folders,
        install_ea,
        map_symbol,
        symbol_choices,
    )

    cfg = Mt4Config.load() or Mt4Config()
    if setup or folder:
        if folder:
            chosen = Path(folder).expanduser()
        else:
            console.print("Looking for MetaTrader 4 on this computer (this can take a minute)…")
            found = find_files_folders()
            if not found:
                console.print("[red]No MT4 data folder found.[/red] In MT4 use File → Open Data Folder, open "
                              "MQL4 → Files, and pass that path with --folder.")
                raise typer.Exit(code=1)
            for i, f in enumerate(found, 1):
                console.print(f"  {i}. {f}")
            pick = 1 if len(found) == 1 else typer.prompt("Which one is your CMC MT4", type=int, default=1)
            chosen = found[pick - 1]
        if not chosen.is_dir():
            console.print(f"[red]Not a folder: {chosen}[/red]")
            raise typer.Exit(code=1)
        cfg.files_dir = str(chosen)
        ea = install_ea(chosen)
        console.print(f"[green]EA copied to[/green] {ea}")
    if lots is not None:
        if not 0 < lots <= 1:
            console.print("[red]--lots must be between 0.01 and 1.[/red]")
            raise typer.Exit(code=1)
        cfg.lots = lots
    for pair in (symbol or "").split(","):
        ours, _, theirs = pair.partition("=")
        if theirs:
            cfg.symbols[ours.strip().upper()] = theirs.strip()
    if setup or folder or lots is not None or symbol:
        console.print(f"Saved {cfg.save()}")
    if not cfg.files_dir:
        console.print(f"[yellow]Not set up: run `tradingagents fx-mt4 --setup` ({config_path()}).[/yellow]")
        raise typer.Exit(code=1)

    bridge = Bridge(cfg.files_dir, _fx_journal(), lots=cfg.lots, symbols=cfg.symbols)
    now = datetime.now(UTC)
    if flatten:
        if not typer.confirm("Cancel ALL of the desk's pending orders and close ALL its trades in MT4 now?"):
            raise typer.Exit()
        cid = bridge.flatten(now)
    else:
        cid = bridge.ping(now)
    state = bridge.state()
    ok, why = bridge.healthy(now, state)
    console.print(f"Folder: {cfg.files_dir}\nLots per order: {cfg.lots:g}")
    if state is None:
        console.print("[yellow]The EA hasn't written anything yet: attach TradingAgentsBridge to a chart "
                      "and turn AutoTrading on.[/yellow]")
    else:
        kind = "DEMO" if state.demo else "LIVE"
        console.print(f"Account: {kind} {state.account} · {state.company} · {state.server}\n"
                      f"Balance {state.balance:,.2f} {state.currency} · equity {state.equity:,.2f} · "
                      f"EA lot cap {state.max_lots:g} · desk orders in MT4: {len(state.orders)}")
        names = bridge.broker_symbols() or list(state.quotes)
        mapped = {s: map_symbol(s, names, cfg.symbols) for s in DEFAULT_UNIVERSE}
        console.print("Symbols: " + ", ".join(f"{s}→{m}" if m and m != s else (s if m else f"[red]{s}→?[/red]")
                                             for s, m in mapped.items()))
        if any(m is None for m in mapped.values()):
            console.print("[yellow]Map a missing one with e.g. --symbol XAUUSD=GOLD[/yellow]")
        for s, m in mapped.items():
            others = [c for c in symbol_choices(s, names) if c != m]
            if m and others and s not in cfg.symbols:
                console.print(f"[yellow]{s}: using {m}, but MT4 also has {', '.join(others)}. Check which one you "
                              f"can trade (--test-order), then fix it with --symbol {s}=<name>.[/yellow]")
    console.print("Bridge: " + ("[green]ready[/green]" if ok else f"[yellow]{why}[/yellow]"))
    if test_order and ok and state is not None:
        bridge.answers()                           # clear the ping's answer
        names = bridge.broker_symbols() or list(state.quotes)
        target = test_order if test_order in names else map_symbol(test_order.upper(), names, cfg.symbols)
        if target is None:
            console.print(f"[red]MT4 has no symbol like {test_order}.[/red]")
            raise typer.Exit(code=1)
        for line in bridge.test_order(target, datetime.now(UTC)):
            console.print(line)
        return
    for _ in range(10):                            # the EA answers within a second or two
        clock.sleep(1)
        answer = bridge.answers().get(cid)
        if answer:
            colour = "green" if answer.get("ok") == "1" else "red"
            console.print(f"MT4 answered: [{colour}]{answer.get('message', '')}[/{colour}]")
            break
    else:
        console.print("[yellow]No answer from MT4 within 10 s: is the EA on a chart with a smiley face?[/yellow]")


@app.command("fx-lab")
def fx_lab(
    months: int = typer.Option(12, "--months", help="How many months of history to replay (each year, with --experiments)"),
    step: int = typer.Option(10, "--step", help="Minutes between scans, as the watcher runs them"),
    symbols: str = typer.Option(None, "--symbols", help="Comma-separated pairs; omit for the default list"),
    min_rr: float = typer.Option(2.0, "--min-rr", help="Minimum reward-to-risk after the spread"),
    final: int = typer.Option(6, "--final", help="Most new orders per scan"),
    experiments: bool = typer.Option(False, "--experiments", help="Run Quinn's ideas: design year, then the locked year"),
    batch: int = typer.Option(None, "--batch", help="Which batch of Quinn's ideas to run (default: the newest)"),
    anatomy: bool = typer.Option(False, "--anatomy", help="Study how the design year's trades played out: costs, stops, exits"),
    refresh: bool = typer.Option(False, "--refresh", help="Download the history again instead of topping it up"),
    notify: bool = typer.Option(False, "--notify", help="Send the summary to Telegram"),
):
    """Quinn's lab: replay today's scanner over past OANDA prices and grade it, or test new ideas with --experiments."""
    from datetime import UTC, datetime, timedelta

    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
    from rich.table import Table

    from tradingagents.dataflows.vendors import oanda
    from tradingagents.fx import DEFAULT_UNIVERSE, lab, quinn, telegram
    from tradingagents.fx.smc import NEW_YORK

    if months < 1 or step < 1 or final < 1:
        console.print("[red]--months, --step and --final must be at least 1.[/red]")
        raise typer.Exit(code=1)
    names = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else list(DEFAULT_UNIVERSE)
    _require_oanda(names[0])
    end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    span = timedelta(days=round(months * 30.44))
    start = end - span
    locked_start = start - span                     # the year before: Quinn's locked test year
    history = lab.History(lab.lab_dir() / "history")
    rules = lab.Rules(step_minutes=step, min_rr=min_rr, final=final)
    title = "anatomy" if anatomy else "experiments" if experiments else "baseline"
    console.print(f"[bold]Quinn's lab · {title}[/bold] · design year {start:%d %b %Y} to {end:%d %b %Y}"
                  + (f" · locked year {locked_start:%d %b %Y} to {start:%d %b %Y}" if experiments and not anatomy else "")
                  + f" · {len(names)} instruments · scans every {step} min, 02:00-12:00 New York")

    columns = (TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
               TimeRemainingColumn())
    first = (locked_start if experiments and not anatomy else start) - lab.WARMUP
    frames = {}
    with Progress(*columns, console=console) as bar:
        job = bar.add_task("Price history", total=len(names) * len(lab.GRANULARITIES))
        for symbol in names:
            for gran in lab.GRANULARITIES:
                bar.update(job, description=f"Price history · {symbol} {gran}")
                try:
                    frames[(symbol, gran)] = history.ensure(symbol, gran, first, end,
                                                            oanda.get_candle_history, refresh=refresh)
                except Exception as exc:
                    console.print(f"[yellow]{symbol} {gran}: {exc}; left out.[/yellow]")
                bar.advance(job)
    feed = lab.HistoricalFeed(frames)
    used = sorted({s for s, _ in frames})

    def scans(a, b, label):
        days = max((b - a).days, 1)
        with Progress(*columns, console=console) as bar:
            job = bar.add_task(label, total=days)

            def moved(t, found):
                bar.update(job, completed=(t - a).days,
                           description=f"{label} · {t.astimezone(NEW_YORK):%d %b %Y} · {found} setups")

            log = lab.cached_scan_log(lab.lab_dir() / "scans", feed, used, a, b, rules, progress=moved)
            bar.update(job, completed=days)
        return log

    design_log = scans(start, end, "Scanning the design year")
    if anatomy:
        from tradingagents.fx import anatomy as anat

        console.print("Trading the design year and reading every trade's path…")
        entries = lab.trade(design_log, feed, end, rules)
        study, _ = anat.study(entries, frames)
        console.print()
        for line in study.findings:
            head, _, rest = line.partition(": ")
            console.print(f"[bold]{head}[/bold]: {rest}\n")
        table = Table(title=f"Trade anatomy · {study.trades} filled trades")
        table.add_column("Measure")
        table.add_column("Value", justify="right")
        for label, value in (
            ("Per trade after / before the spread", f"{study.net_r:+.2f}R / {study.gross_r:+.2f}R"),
            ("Spread as a share of the risk", f"{study.spread_share:.0%}"),
            ("Losers +0.5R / +1R / +1.5R up first", " / ".join(f"{study.losers_mfe[k]:.0%}" for k in ("0.5R", "1R", "1.5R"))),
            ("Winners 0.5R against us first", f"{study.winners_mae['0.5R']:.0%}"),
            ("Losers stopped within 15 / 60 min", f"{study.stopped_within_15:.0%} / {study.stopped_within_60:.0%}"),
            ("Losers that hit the target after the stop", f"{study.target_after_stop:.0%}"),
            ("Closed at 16:55", f"{study.closed_at_17:.0%}"),
            ("What if: break-even at +1R", f"{study.breakeven_r:+.2f}R"),
            ("What if: half off at +1R", f"{study.half_off_r:+.2f}R"),
            ("Losses with stop and target in one bar", f"{study.same_bar_losses:.0%}"),
        ):
            table.add_row(label, value)
        console.print(table)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
        md = lab.lab_dir() / f"anatomy-{stamp}.md"
        md.write_text(anat.to_markdown(study, f"{start:%d %b %Y} to {end:%d %b %Y}"), encoding="utf-8")
        console.print(f"Full report: {md}")
        if notify and telegram.configured():
            try:
                telegram.send(anat.telegram_summary(study))
            except telegram.TelegramError as exc:
                console.print(f"[yellow]Telegram: {exc}[/yellow]")
        return
    if not experiments:
        entries = lab.trade(design_log, feed, end, rules)
        report = lab.summarize(entries, start, end, used, rules)
        md, js = lab.save(report, entries, lab.lab_dir())
        console.print(f"\n[bold]{report.verdict}[/bold]\n")
        table = Table(title="Against the benchmarks")
        for col in ("Measure", "Result", "Red line", "Target", "Strong", "Grade"):
            table.add_column(col, justify="right" if col not in ("Measure", "Grade") else "left")
        for key, (label, red, target, strong, _, fmt) in lab.BENCHMARKS.items():
            value = report.metrics.get(key)
            table.add_row(label, "—" if value is None else fmt.format(value), fmt.format(red), fmt.format(target),
                          fmt.format(strong), report.grades[key])
        console.print(table)
        m = report.metrics
        console.print(f"{report.orders} orders, {report.filled} filled ({m['filled_per_day']} a day), "
                      f"total {m['total_r']:+.2f}R.")
        console.print(f"Full report: {md}\nTrades for Quinn: {js}")
        if notify and telegram.configured():
            try:
                telegram.send(lab.telegram_summary(report))
            except telegram.TelegramError as exc:
                console.print(f"[yellow]Telegram: {exc}[/yellow]")
        return

    locked_log = scans(locked_start, start, "Scanning the locked year")
    ledger = quinn.Ledger(lab.lab_dir() / "ledger.json")
    console.print("Trading the unchanged rules on both years, for comparison…")
    base_design = quinn.measure(lab.trade(design_log, feed, end, rules))
    base_locked = quinn.measure(lab.trade(locked_log, feed, start, rules))
    rows = []
    number = batch or max(quinn.BATCHES)
    if number not in quinn.BATCHES:
        console.print(f"[red]No batch {number}; there are {sorted(quinn.BATCHES)}.[/red]")
        raise typer.Exit(code=1)
    console.print(f"Batch {number}: {len(quinn.BATCHES[number])} ideas")
    for exp in quinn.BATCHES[number]:
        console.print(f"Trying [bold]{exp.name}[/bold]: {exp.describe()}")
        row = quinn.run(exp, lambda e: lab.trade(design_log, feed, end, rules, e),
                        lambda e: lab.trade(locked_log, feed, start, rules, e), base_design, ledger)
        d = row["design"]
        console.print(f"  [dim]{d['trades']} trades, "
                      f"{'—' if d['expectancy_r'] is None else format(d['expectancy_r'], '+.2f') + 'R'} per trade · "
                      f"{row['status']}[/dim]")
        rows.append(row)
    table = Table(title=f"Unchanged rules: {base_design.expectancy_r:+.2f}R per trade on the design year")
    for col in ("Idea", "Design trades", "Per trade", "PF", "Locked per trade", "t / bar", "Status"):
        table.add_column(col)
    for r in rows:
        d, lk = r["design"], r["locked"]
        table.add_row(r["name"], str(d["trades"]),
                      "—" if d["expectancy_r"] is None else f"{d['expectancy_r']:+.2f}R", str(d["profit_factor"]),
                      "—" if lk is None else f"{lk['expectancy_r']:+.2f}R", "—" if lk is None
                      else f"{lk['t_score']} / {r['locked_bar']}",
                      {"rejected on the design year": "✕ design year", "failed on the locked year": "✕ locked year"}
                      .get(r["status"], "✅ candidate"))
    console.print(table)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    md = lab.lab_dir() / f"experiments-{stamp}.md"
    md.write_text(quinn.report(rows, base_design, base_locked, ledger), encoding="utf-8")
    console.print(f"Full report: {md}\nLedger: {ledger.path}")
    if notify and telegram.configured():
        try:
            telegram.send(quinn.telegram_summary(rows))
        except telegram.TelegramError as exc:
            console.print(f"[yellow]Telegram: {exc}[/yellow]")

def _fx_desk_inputs(setups, now):
    """Tickets for the candidates and a snapshot of the live book, for a review."""
    from tradingagents.dataflows.vendors import oanda
    from tradingagents.fx.journal import ACTIVE
    from tradingagents.fx.manage import snapshot

    book = _fx_journal()
    tickets = book.next_tickets(s.symbol for s in setups)
    positions = snapshot(book.entries(ACTIVE), oanda.get_candles, oanda.get_quote, now)
    return tickets, positions


def _fx_load_models(models: dict) -> dict:
    """Build the quick and deep models once, on first use, and keep them in ``models``."""
    if not models:
        from tradingagents.llm_clients.factory import create_tier_client

        _quiet_fx_logs()
        models["quick"] = create_tier_client(DEFAULT_CONFIG, "quick").get_llm()
        models["deep"] = create_tier_client(DEFAULT_CONFIG, "deep").get_llm()
    return models


def _fx_ward(models: dict):
    """The trade manager's own check of the live book, for the watcher."""
    from types import SimpleNamespace

    from tradingagents.dataflows.vendors import oanda
    from tradingagents.fx import agents as fx_agents
    from tradingagents.fx.context import gather
    from tradingagents.fx.journal import ACTIVE, OPEN, flat_by
    from tradingagents.fx.manage import snapshot

    def ward(now):
        positions = snapshot(_fx_journal().entries(ACTIVE), oanda.get_candles, oanda.get_quote, now)
        if not positions:
            return None
        _fx_load_models(models)
        # Calendar and headlines up to when each trade can still be affected.
        horizon = [SimpleNamespace(symbol=p.entry.symbol,
                                   expires_at=flat_by(p.entry.filled_at) if p.entry.status == OPEN
                                   else p.entry.expires_at) for p in positions]
        context = gather(horizon, now)
        outcome = fx_agents.check_trades(positions, context, models["quick"], models["deep"], now=now)
        for problem in outcome.problems:
            console.print(f"[yellow]Model problem: {problem}[/yellow]")
        return outcome

    return ward


def _fx_reviewer(*, final: int, max_per_currency: int, min_rr: float, models: dict | None = None):
    """A reviewer for the watcher: models built once, report saved, no exit on failure."""
    from tradingagents.fx import agents as fx_agents
    from tradingagents.fx.context import gather
    from tradingagents.fx.report import save as save_scan

    models = {} if models is None else models

    def review(result, now):
        _fx_load_models(models)
        context = gather(result.setups, now)
        tickets, positions = _fx_desk_inputs(result.setups, now)
        outcome = fx_agents.review(result.setups, context, models["quick"], models["deep"],
                                   max_orders=final, max_per_currency=max_per_currency, min_rr=min_rr, now=now,
                                   tickets=tickets, positions=positions)
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
        tickets, positions = _fx_desk_inputs(result.setups, now)
        try:
            review = fx_agents.review(
                result.setups, context, quick, deep,
                max_orders=final, max_per_currency=max_per_currency, min_rr=min_rr,
                progress=lambda msg: status.update(f"{msg}..."), now=now,
                tickets=tickets, positions=positions,
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
        for column in ("Ticket", "Symbol", "Order", "Entry", "Stop", "Target", "RR", "Cancel by", "Conviction"):
            table.add_column(column, justify="right" if column not in ("Ticket", "Symbol", "Order", "Conviction") else "left")
        for o in review.orders:
            colour = "green" if o.direction == "long" else "red"
            table.add_row(review.tickets.get(o.symbol, ""), o.symbol, f"[{colour}]{o.order_type}[/{colour}]", str(o.entry),
                          str(o.stop), str(o.target), f"{o.rr:.2f}",
                          f"{o.expires_at:%H:%M} UTC", o.conviction)
        console.print(table)
        for o in review.orders:
            label = f"{review.tickets.get(o.symbol, '')} {o.symbol}".strip()
            why = o.rationale.removeprefix(label).lstrip(" :-–—")   # the agents often start with it too
            console.print(f"[bold]{label}[/bold] {why}")
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
