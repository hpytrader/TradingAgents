"""Telegram alerts for new orders and their results.

Set up once:

1. In Telegram, message @BotFather, send ``/newbot`` and follow the prompts;
   it replies with a token like ``123456:ABC-...``.
2. Put it in ``.env`` as ``TELEGRAM_BOT_TOKEN``.
3. Send your new bot any message, then run ``tradingagents fx-telegram``: it
   finds your chat id and tells you the line to add (``TELEGRAM_CHAT_ID``).

The token sits in the Bot API's URL path, so every error raised here has it
replaced with ``***`` before it can reach a log or the terminal.
"""

from __future__ import annotations

import os
from datetime import datetime
from html import escape

import requests

from tradingagents.fx.instruments import spec_for
from tradingagents.fx.journal import CLOSED, EXPIRED, LOST, MISSED, OPEN, WON, Entry, Stats
from tradingagents.fx.smc import NEW_YORK

API = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 15


class TelegramError(Exception):
    """Telegram could not be reached or refused the request (token removed)."""


def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "").strip()


def configured() -> bool:
    return bool(_token() and chat_id())


def _call(method: str, payload: dict | None = None) -> dict:
    token = _token()
    if not token:
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set")
    try:
        response = requests.post(API.format(token=token, method=method), json=payload or {},
                                 timeout=TIMEOUT)
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise TelegramError(str(exc).replace(token, "***")) from None
    if not data.get("ok"):
        raise TelegramError(str(data.get("description", f"HTTP {response.status_code}")).replace(token, "***"))
    return data.get("result") or {}


def send(text: str) -> None:
    """Send ``text`` (Telegram HTML) to the configured chat."""
    if not chat_id():
        raise TelegramError("TELEGRAM_CHAT_ID is not set")
    _call("sendMessage", {"chat_id": chat_id(), "text": text[:4000], "parse_mode": "HTML",
                          "disable_web_page_preview": True})


def find_chats() -> list[tuple[str, str]]:
    """``(chat id, name)`` of everyone who has messaged the bot recently."""
    seen: dict[str, str] = {}
    for update in _call("getUpdates") or []:
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat") or {}
        if "id" in chat:
            name = chat.get("title") or " ".join(
                p for p in (chat.get("first_name"), chat.get("last_name")) if p) or chat.get("username", "")
            seen[str(chat["id"])] = name
    return list(seen.items())


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def _local(t: datetime) -> str:
    return t.astimezone(NEW_YORK).strftime("%H:%M")


def _price(symbol: str, value: float) -> str:
    return f"{value:.{spec_for(symbol).decimals}f}"


def order_message(e: Entry) -> str:
    arrow = "▲" if e.long else "▼"
    pips = spec_for(e.symbol).pips(e.risk)
    lines = [
        f"{arrow} <b>{e.order_type} {e.symbol}</b> @ {_price(e.symbol, e.entry)}",
        f"SL {_price(e.symbol, e.stop)} ({pips:.1f} pips) · TP {_price(e.symbol, e.target)} · "
        f"{e.planned_rr:.2f}R",
        f"Cancel {_local(e.expires_at)} New York · conviction {escape(e.conviction)}",
    ]
    if e.rationale:
        lines.append(f"<i>{escape(e.rationale[:500])}</i>")
    if e.watch_for:
        lines.append(f"Watch: {escape(e.watch_for[:300])}")
    lines.append("Paper trade: check your chart before acting.")
    return "\n".join(lines)


def result_message(e: Entry) -> str:
    r = e.result_r or 0.0
    heads = {
        OPEN: lambda: f"⏵ FILLED {e.symbol} {e.order_type} @ {_price(e.symbol, e.entry)}",
        WON: lambda: f"✓ WON {e.symbol} {r:+.2f}R",
        LOST: lambda: f"✕ LOST {e.symbol} {r:+.2f}R",
        CLOSED: lambda: f"■ CLOSED {e.symbol} at the NY close {r:+.2f}R",
        EXPIRED: lambda: f"○ EXPIRED {e.symbol}: not filled by {_local(e.expires_at)}",
        MISSED: lambda: f"○ MISSED {e.symbol}: target reached before the fill",
    }
    head = heads[e.status]() if e.status in heads else f"{e.symbol}: {e.status}"
    return f"<b>{escape(head)}</b>" + (f"\n{escape(e.note)}" if e.note and e.status in (WON, LOST, CLOSED) else "")


def summary_message(s: Stats, today: str) -> str:
    if not s.finished:
        return f"<b>FX desk · {escape(today)}</b>\nNo finished trades yet. Open {s.open}, pending {s.pending}."
    return (f"<b>FX desk · {escape(today)}</b>\n"
            f"Trades {s.finished} · win rate {s.win_rate:.0%} · total {s.total_r:+.2f}R · "
            f"avg {s.avg_r:+.2f}R\nOpen {s.open} · pending {s.pending}")
