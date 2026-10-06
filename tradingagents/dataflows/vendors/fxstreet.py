"""Forex headlines from FXStreet's public news feed.

Yahoo's news is built around stocks: a search for ``EURUSD=X`` often returns
nothing, or a general markets story. FXStreet writes for forex traders, posts
within minutes, and reports releases with their actual figures ("US ISM
Services PMI rises to 54.9"), which the economic calendar feed lacks.

The feed is fetched once per scan and cached for ten minutes; headlines are
matched to a pair by the currencies, central banks and nicknames they mention.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorUnavailableError

FEED_URL = "https://www.fxstreet.com/rss/news"
REQUEST_TIMEOUT = 20
CACHE_SECONDS = 600

# Words that tie a headline to a currency. Short codes are matched as whole
# words so "CAD" does not match "decade".
KEYWORDS: dict[str, tuple[str, ...]] = {
    "USD": ("usd", "dollar", "greenback", "fed", "fomc", "powell", "treasury", "nonfarm",
            "nfp", "us cpi", "us pce", "ism", "jolts", "us gdp", "us retail", "white house",
            "trump", "wall street", "dxy"),
    "EUR": ("eur", "euro", "ecb", "lagarde", "eurozone", "euro area", "germany", "german",
            "france", "french", "italy", "italian", "bund"),
    "GBP": ("gbp", "pound", "sterling", "cable", "boe", "bank of england", "bailey", "uk ",
            "britain", "british", "gilt"),
    "JPY": ("jpy", "yen", "boj", "bank of japan", "ueda", "japan", "japanese", "jgb"),
    "CHF": ("chf", "franc", "snb", "swiss", "switzerland"),
    "AUD": ("aud", "aussie", "rba", "reserve bank of australia", "australia", "australian"),
    "NZD": ("nzd", "kiwi", "rbnz", "new zealand"),
    "CAD": ("cad", "loonie", "boc", "bank of canada", "canada", "canadian", "oil", "crude", "wti"),
    "XAU": ("xau", "gold", "bullion", "precious metal"),
    "XAG": ("xag", "silver", "precious metal"),
}


@dataclass(frozen=True)
class Headline:
    title: str
    published: datetime
    summary: str = ""

    def describe(self) -> str:
        return f"{self.published:%a %H:%M} UTC · {self.title}"


def _cache_path() -> Path:
    return Path(get_config()["data_cache_dir"]) / "fxstreet" / "news.json"


def _download() -> str:
    try:
        response = requests.get(FEED_URL, timeout=REQUEST_TIMEOUT,
                                headers={"User-Agent": "TradingAgents fx-scan (RSS reader)"})
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"FXStreet news unreachable: {exc}") from None
    if response.status_code != 200:
        raise VendorUnavailableError(f"FXStreet news returned HTTP {response.status_code}")
    return response.text


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(text or ""))).strip()


def parse(xml_text: str) -> list[Headline]:
    """RSS items → headlines, newest first; unreadable items are skipped."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        raise VendorUnavailableError("FXStreet news feed is not valid RSS") from None
    out = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title", ""))
        try:
            published = parsedate_to_datetime(item.findtext("pubDate", "")).astimezone(UTC)
        except (TypeError, ValueError):
            continue
        if title:
            out.append(Headline(title=title, published=published,
                                summary=_clean(item.findtext("description", ""))[:300]))
    return sorted(out, key=lambda h: h.published, reverse=True)


def _headlines() -> list[Headline]:
    path = _cache_path()
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [Headline(r["title"], datetime.fromisoformat(r["published"]), r.get("summary", ""))
                for r in rows]
    items = parse(_download())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps([{**asdict(h), "published": h.published.isoformat()} for h in items]),
                   encoding="utf-8")
    tmp.replace(path)
    return items


def _mentions(text: str, currency: str) -> bool:
    lowered = f" {text.lower()} "
    for word in KEYWORDS.get(currency, (currency.lower(),)):
        if len(word.strip()) <= 4:
            if re.search(rf"(?<![a-z]){re.escape(word.strip())}(?![a-z])", lowered):
                return True
        elif word in lowered:
            return True
    return False


def headlines_for(symbol: str, now: datetime, hours: float = 12,
                  limit: int = 12, items: list[Headline] | None = None) -> list[Headline]:
    """Recent headlines mentioning either currency of ``symbol``, newest first."""
    s = symbol.upper()
    legs = {s[:3], s[3:]}
    pair = f"{s[:3]}/{s[3:]}".lower()
    since = now - timedelta(hours=hours)
    pool = items if items is not None else _headlines()
    found = []
    for h in pool:
        if not since <= h.published <= now + timedelta(minutes=5):
            continue
        text = f"{h.title} {h.summary}"
        if pair in text.lower() or any(_mentions(text, leg) for leg in legs):
            found.append(h)
        if len(found) >= limit:
            break
    return found


def market_headlines(now: datetime, hours: float = 12, limit: int = 15,
                     items: list[Headline] | None = None) -> list[Headline]:
    """The most recent headlines of any kind, for the macro picture."""
    since = now - timedelta(hours=hours)
    pool = items if items is not None else _headlines()
    return [h for h in pool if since <= h.published <= now + timedelta(minutes=5)][:limit]
