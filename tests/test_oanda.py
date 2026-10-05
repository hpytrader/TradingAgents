"""The OANDA price vendor, against recorded-shape responses (no network)."""

import pytest
import requests

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import oanda

TOKEN = "secret-token-123"


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _candle(time, o, h, low, c, complete=True):
    return {"time": time, "complete": complete, "volume": 42,
            "mid": {"o": str(o), "h": str(h), "l": str(low), "c": str(c)}}


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("OANDA_API_TOKEN", TOKEN)
    monkeypatch.delenv("OANDA_ENVIRONMENT", raising=False)


def _serve(monkeypatch, status, payload, seen=None):
    def fake_get(url, params, headers, timeout):
        if seen is not None:
            seen.update(url=url, params=params, headers=headers)
        return _Response(status, payload)
    monkeypatch.setattr(oanda.requests, "get", fake_get)


@pytest.mark.unit
def test_symbols_map_to_oanda_instruments():
    assert oanda.to_oanda_instrument("EURUSD") == "EUR_USD"
    assert oanda.to_oanda_instrument("xauusd+") == "XAU_USD"
    assert oanda.to_oanda_instrument("GBP/JPY") == "GBP_JPY"
    assert oanda.to_oanda_instrument("EUR_USD") == "EUR_USD"
    with pytest.raises(ValueError):
        oanda.to_oanda_instrument("AAPL")


@pytest.mark.unit
def test_missing_token_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
    with pytest.raises(oanda.OandaNotConfiguredError, match="OANDA_API_TOKEN"):
        oanda.get_candles("EURUSD")


@pytest.mark.unit
def test_candles_parse_and_drop_the_unfinished_bar(monkeypatch, token):
    seen = {}
    _serve(monkeypatch, 200, {"candles": [
        _candle("2026-10-05T10:00:00.000000000Z", 1.1, 1.2, 1.0, 1.15),
        _candle("2026-10-05T11:00:00.000000000Z", 1.15, 1.25, 1.1, 1.2),
        _candle("2026-10-05T12:00:00.000000000Z", 1.2, 1.3, 1.2, 1.28, complete=False),
    ]}, seen)

    frame = oanda.get_candles("EURUSD", "H1", 3)

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 2
    assert frame["close"].iloc[-1] == pytest.approx(1.2)
    assert str(frame.index.tz) == "UTC"
    assert seen["url"].startswith("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/")
    assert seen["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in seen["url"] and TOKEN not in str(seen["params"])


@pytest.mark.unit
def test_live_environment_uses_the_live_host(monkeypatch, token):
    monkeypatch.setenv("OANDA_ENVIRONMENT", "live")
    seen = {}
    _serve(monkeypatch, 200, {"candles": [_candle("2026-10-05T10:00:00Z", 1, 1, 1, 1)]}, seen)
    oanda.get_candles("EURUSD")
    assert seen["url"].startswith("https://api-fxtrade.oanda.com")


@pytest.mark.unit
def test_no_candles_is_no_market_data(monkeypatch, token):
    _serve(monkeypatch, 200, {"candles": []})
    with pytest.raises(NoMarketDataError):
        oanda.get_candles("EURUSD")


@pytest.mark.unit
def test_refused_token_says_how_to_fix_it(monkeypatch, token):
    _serve(monkeypatch, 401, {"errorMessage": "Insufficient authorization"})
    with pytest.raises(oanda.OandaNotConfiguredError, match="practice or live"):
        oanda.get_candles("EURUSD")


@pytest.mark.unit
@pytest.mark.parametrize("status", [429, 503])
def test_throttling_and_outages_are_unavailable(monkeypatch, token, status):
    _serve(monkeypatch, status, {})
    with pytest.raises(VendorUnavailableError):
        oanda.get_candles("EURUSD")


@pytest.mark.unit
def test_network_errors_do_not_leak_the_token(monkeypatch, token):
    def boom(url, params, headers, timeout):
        raise requests.ConnectionError("connection reset")
    monkeypatch.setattr(oanda.requests, "get", boom)
    with pytest.raises(VendorUnavailableError) as info:
        oanda.get_candles("EURUSD")
    assert TOKEN not in str(info.value)


@pytest.mark.unit
def test_quote_reads_bid_and_ask(monkeypatch, token):
    _serve(monkeypatch, 200, {"candles": [{
        "time": "2026-10-05T12:00:05.000000000Z", "complete": False,
        "bid": {"c": "2401.10"}, "ask": {"c": "2401.40"},
    }]})
    q = oanda.get_quote("XAUUSD")
    assert q.bid == pytest.approx(2401.10)
    assert q.spread == pytest.approx(0.30)
    assert q.mid == pytest.approx(2401.25)
