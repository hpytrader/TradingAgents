"""The MT4 bridge: the journal's orders placed, managed and reported through files."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tradingagents.fx import mt4
from tradingagents.fx.journal import Journal
from tradingagents.fx.mt4 import Bridge, Mt4Config, map_symbol, parse_state

NOW = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)          # 06:00 New York


def _order(symbol="EURUSD", entry=1.1700, stop=1.1680, target=1.1750, direction="long"):
    setup = SimpleNamespace(spread_pips=1, strategy="smc", score=70)
    return SimpleNamespace(symbol=symbol, direction=direction, entry=entry, stop=stop, target=target, rr=2.4,
                           expires_at=NOW + timedelta(hours=8), conviction="medium", rationale="",
                           watch_for="", setup=setup)


class FakeEA:
    """Does what TradingAgentsBridge.mq4 does with the files, on a pretend account."""

    def __init__(self, folder, symbols=("EURUSD", "GBPUSD", "GOLD")):
        self.dir = folder
        self.orders, self.history, self.next = {}, {}, 5001
        self.time, self.trade_allowed = NOW, True
        self.quotes = {"EURUSD": (1.1719, 1.1721), "GBPUSD": (1.3400, 1.3402), "GOLD": (4150.0, 4150.4)}
        self.received = []
        (folder / "ta_symbols.txt").write_text("\r\n".join(symbols))
        self.write_state()

    def write_state(self):
        lines = [f"time={int(self.time.timestamp())}", "account=123456", "server=CMC-Live",
                 "company=CMC Markets", "currency=CAD", "balance=1000.00", "equity=1000.00",
                 "demo=0", "connected=1", f"trade_allowed={int(self.trade_allowed)}", "max_lots=0.01"]
        for t, o in self.orders.items():
            lines.append(f"order={t}|{o['comment']}|{o['symbol']}|{o['type']}|0.01|{o['price']}|{o['sl']}|{o['tp']}|0.00")
        for t, o in self.history.items():
            lines.append(f"hist={t}|{o['comment']}|{o['symbol']}|{o['type']}|0.01|{o['price']}|{o['sl']}|{o['tp']}|"
                         f"{o['profit']}|{o['close']}|{int(self.time.timestamp())}")
        lines += [f"quote={s}|{b}|{a}" for s, (b, a) in self.quotes.items()]
        (self.dir / "ta_state.txt").write_text("\r\n".join(lines) + "\r\n")

    def run(self):
        for path in sorted(self.dir.glob("ta_cmd_*.txt")):
            cmd = mt4._kv(path.read_text())
            path.unlink()
            self.received.append(cmd)
            ok, ticket, msg = True, int(cmd.get("mt4") or 0), "done"
            if cmd["action"] == "place":
                ticket, self.next = self.next, self.next + 1
                self.orders[ticket] = {"comment": cmd["comment"], "symbol": cmd["symbol"],
                                       "type": 2 if cmd["side"] == "buy" else 3, "price": cmd["price"],
                                       "sl": cmd["sl"], "tp": cmd["tp"]}
            elif cmd["action"] == "modify":
                self.orders[ticket].update(sl=cmd["sl"], tp=cmd["tp"])
            elif cmd["action"] in ("cancel", "close"):
                o = self.orders.pop(ticket)
                self.history[ticket] = {**o, "profit": "0.00", "close": o["price"]}
            price = cmd["price"] if cmd["action"] == "place" else "0"
            text = f"id={cmd['id']}\r\nok={int(ok)}\r\nmt4={ticket}\r\nerror=0\r\nprice={price}\r\nmessage={msg}\r\n"
            (self.dir / f"ta_res_{cmd['id']}.txt").write_text(text)
        self.write_state()

    def fill(self, ticket):
        self.orders[ticket]["type"] -= 2
        self.write_state()

    def stop_out(self, ticket, price, profit):
        o = self.orders.pop(ticket)
        self.history[ticket] = {**o, "profit": profit, "close": price}
        self.write_state()


@pytest.fixture
def desk(tmp_path):
    files = tmp_path / "MQL4" / "Files"
    files.mkdir(parents=True)
    book = Journal(tmp_path / "j.db")
    ea = FakeEA(files)
    sent: list[str] = []
    bridge = Bridge(files, book, lots=0.01)
    return SimpleNamespace(book=book, ea=ea, sent=sent, bridge=bridge,
                           sync=lambda now=NOW: bridge.sync(now, sent.append))


@pytest.mark.unit
def test_a_new_journal_order_is_placed_once_with_its_times(desk):
    [e] = desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    desk.ea.run()
    desk.sync()
    [cmd] = desk.ea.received
    assert cmd["action"] == "place" and cmd["symbol"] == "EURUSD" and cmd["side"] == "buy"
    # CMC quotes 1.1719/1.1721: the long's levels move half that spread so they trigger on the mid
    assert cmd["lots"] == "0.01" and cmd["price"] == "1.1701" and cmd["sl"] == "1.1679" and cmd["tp"] == "1.1749"
    assert cmd["comment"] == "TA#1043"
    assert int(cmd["expires"]) == int(e.expires_at.timestamp())
    assert int(cmd["close_by"]) == int(datetime(2026, 10, 6, 20, 55, tzinfo=UTC).timestamp())  # 16:55 NY
    assert int(cmd["valid_until"]) == int((NOW + timedelta(minutes=2)).timestamp())
    assert any("placed" in m and "#1043" in m for m in desk.sent)
    assert desk.bridge.rows()["#1043"].state == "pending"
    desk.sync(NOW + timedelta(minutes=1))
    desk.ea.run()
    assert len(desk.ea.received) == 1                                  # not sent twice


@pytest.mark.unit
def test_fills_and_closes_at_cmc_are_reported_with_the_profit(desk):
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    desk.ea.run()
    desk.sync()
    desk.ea.fill(5001)
    desk.sync()
    assert any("CMC filled" in m and "1.17" in m for m in desk.sent)
    desk.ea.stop_out(5001, "1.1680", "-2.74")
    desk.sync()
    assert any("CMC closed" in m and "−2.74 CAD" in m for m in desk.sent)
    assert desk.bridge.rows()["#1043"].state == "closed"


@pytest.mark.unit
def test_the_trade_managers_changes_reach_mt4(desk):
    [e] = desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    desk.ea.run()
    desk.sync()
    desk.ea.fill(5001)
    desk.sync()
    later = NOW + timedelta(minutes=30)
    desk.ea.time = later                                               # the EA is still writing
    desk.ea.write_state()
    desk.book.modify(e, now=later, by="Ward", reason="lock in", stop=1.1702)
    desk.sync(later)
    desk.ea.run()
    desk.sync(later)
    # a long's stop sits half CMC's 0.2-pip spread under the desk's mid-price level
    assert desk.ea.received[-1]["action"] == "modify" and desk.ea.received[-1]["sl"] == "1.1701"
    assert desk.ea.orders[5001]["sl"] == "1.1701"
    desk.sync(later)
    desk.ea.run()
    assert len(desk.ea.received) == 2                                  # no repeat once applied

    [e] = desk.book.entries(("open", "pending"))
    desk.book.close_now(e, price=1.1730, now=later, by="Ward", reason="news")
    desk.sync(later)
    desk.ea.run()
    assert desk.ea.received[-1]["action"] == "close" and 5001 in desk.ea.history


@pytest.mark.unit
def test_an_order_the_journal_stopped_waiting_on_is_cancelled_at_cmc(desk):
    [e] = desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    desk.ea.run()
    desk.sync()
    e.status = "missed"                                                # target traded before the fill
    desk.book.save(e)
    desk.sync()
    desk.ea.run()
    desk.sync()
    assert desk.ea.received[-1]["action"] == "cancel"
    assert any("cancelled, unfilled" in m for m in desk.sent)


@pytest.mark.unit
def test_a_paper_result_does_not_close_a_trade_that_filled_at_cmc(desk):
    [e] = desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    desk.ea.run()
    desk.sync()
    desk.ea.fill(5001)
    desk.sync()
    e = desk.book.entries()[0]
    e.status, e.result_r = "won", 2.4                                  # OANDA's mid reached the target
    desk.book.save(e)
    desk.sync()
    desk.ea.run()
    assert [c["action"] for c in desk.ea.received] == ["place"]       # CMC's own target decides


@pytest.mark.unit
def test_nothing_is_sent_while_mt4_is_silent_and_it_warns_once(desk):
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    later = NOW + timedelta(minutes=5)                                 # the EA last wrote at NOW
    desk.sync(later)
    desk.sync(later + timedelta(seconds=20))
    assert not list(desk.ea.dir.glob("ta_cmd_*.txt"))
    assert sum("silent" in m for m in desk.sent) == 1
    desk.ea.time = later
    desk.ea.write_state()
    desk.sync(later)
    assert any("back" in m for m in desk.sent)
    assert list(desk.ea.dir.glob("ta_cmd_*.txt"))


@pytest.mark.unit
def test_autotrading_off_holds_the_orders(desk):
    desk.ea.trade_allowed = False
    desk.ea.write_state()
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    assert not list(desk.ea.dir.glob("ta_cmd_*.txt"))
    assert any("AutoTrading is off" in m for m in desk.sent)


@pytest.mark.unit
def test_gold_maps_to_the_brokers_name_and_wrong_prices_are_refused(desk):
    desk.book.record([_order("XAUUSD", 4140.0, 4130.0, 4170.0)], now=NOW, tickets={"XAUUSD": "#1050"})
    desk.book.record([_order("GBPUSD", 1.2000, 1.1980, 1.2050)], now=NOW, tickets={"GBPUSD": "#1051"})
    desk.sync()
    [cmd] = [mt4._kv(p.read_text()) for p in desk.ea.dir.glob("ta_cmd_*.txt")]
    assert cmd["symbol"] == "GOLD"
    assert any("#1051" in m and "far from CMC" in m for m in desk.sent)    # 1.2000 vs 1.3401


@pytest.mark.unit
def test_an_unanswered_order_is_reported_and_sent_again_then_given_up(desk):
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    later = NOW
    for _ in range(4):                                                 # MT4 never answers
        later += timedelta(minutes=4)
        desk.ea.time = later
        desk.ea.write_state()
        desk.sync(later)
    assert sum(": MT4 did not answer" in m for m in desk.sent) == 4     # the first send and three retries
    assert any("gave up" in m for m in desk.sent)
    assert desk.bridge.rows()["#1043"].state == "failed"


@pytest.mark.unit
def test_orders_about_to_be_cancelled_are_not_sent(desk):
    o = _order()
    o.expires_at = NOW + timedelta(minutes=3)
    desk.book.record([o], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    assert not list(desk.ea.dir.glob("ta_cmd_*.txt"))


@pytest.mark.unit
def test_symbol_mapping():
    names = ["EURUSD.", "EURUSDm", "GBPUSD", "GOLD", "SILVER", "US500"]
    assert map_symbol("EURUSD", names) == "EURUSD."
    assert map_symbol("GBPUSD", names) == "GBPUSD"
    assert map_symbol("XAUUSD", names) == "GOLD"
    assert map_symbol("XAGUSD", names) == "SILVER"
    assert map_symbol("USDJPY", names) is None
    assert map_symbol("XAUUSD", names, {"XAUUSD": "XAU/USD"}) == "XAU/USD"


@pytest.mark.unit
def test_state_file_parsing():
    s = parse_state("time=1791280800\r\naccount=42\r\ndemo=1\r\nconnected=1\r\ntrade_allowed=1\r\n"
                    "balance=1000.50\r\norder=7|TA#1001|EURUSD|2|0.01|1.17|1.168|1.175|0.00\r\n"
                    "hist=8|TA#1002|GOLD|1|0.01|4150|4160|4120|3.10|4120|1791284400\r\n"
                    "quote=EURUSD|1.1719|1.1721\r\n")
    assert s.account == "42" and s.demo and s.balance == 1000.5
    assert s.orders[7].pending and s.orders[7].comment == "TA#1001"
    assert not s.history[8].pending and s.history[8].close_price == 4120 and s.history[8].profit == 3.1
    assert s.quotes["EURUSD"] == (1.1719, 1.1721)


@pytest.mark.unit
def test_setup_finds_the_folder_saves_settings_and_installs_the_ea(tmp_path, monkeypatch):
    files = tmp_path / "App" / "drive_c" / "Program Files" / "CMC MT4" / "MQL4" / "Files"
    files.mkdir(parents=True)
    assert mt4.find_files_folders([tmp_path]) == [files]
    ea = mt4.install_ea(files)
    assert ea == files.parent / "Experts" / "TradingAgentsBridge.mq4"
    assert "MaxLots" in ea.read_text()
    monkeypatch.setenv("TRADINGAGENTS_FX_MT4", str(tmp_path / "fx_mt4.json"))
    Mt4Config(files_dir=str(files), symbols={"XAUUSD": "GOLD"}).save()
    cfg = Mt4Config.load()
    assert cfg.files_dir == str(files) and cfg.lots == 0.01 and cfg.symbols == {"XAUUSD": "GOLD"}


@pytest.mark.unit
def test_the_ea_keeps_its_safety_rails():
    src = mt4.EA_FILE.read_text()
    assert "input double MaxLots        = 0.01;" in src
    for rail in ("lots > MaxLots", "CountOurs() >= MaxOpenOrders", "TimeGMT() > validUntil",
                 "OrderMagicNumber() == MagicNumber", "bookExpires[k]", "bookCloseBy[k]"):
        assert rail in src


@pytest.mark.unit
def test_symbol_choices_show_every_variant():
    names = ["USDJPY", "USDJPY.r", "EURUSD.r", "GOLD", "XAUUSD.r"]
    assert mt4.symbol_choices("USDJPY", names) == ["USDJPY", "USDJPY.r"]
    assert mt4.symbol_choices("XAUUSD", names) == ["GOLD", "XAUUSD.r"]


@pytest.mark.unit
def test_a_test_order_is_placed_under_the_price_and_cancelled(desk):
    lines = desk.bridge.test_order("EURUSD", NOW, sleep=lambda _: desk.ea.run())
    place, cancel = desk.ea.received
    assert place["action"] == "place" and place["side"] == "buy" and place["lots"] == "0.01"
    assert float(place["price"]) == pytest.approx(1.1719 * 0.99)
    assert float(place["sl"]) < float(place["price"]) < float(place["tp"])
    assert cancel["action"] == "cancel" and cancel["mt4"] == "5001"
    assert "works end to end" in lines[-1] and not desk.ea.orders
    assert not desk.bridge.rows()                                     # not part of the desk's book


@pytest.mark.unit
def test_a_test_order_reports_a_refusal(desk):
    def refuse(_):
        for path in desk.ea.dir.glob("ta_cmd_*.txt"):
            cmd = mt4._kv(path.read_text())
            path.unlink()
            (desk.ea.dir / f"ta_res_{cmd['id']}.txt").write_text(
                f"id={cmd['id']}\r\nok=0\r\nmt4=0\r\nerror=133\r\nmessage=trade is disabled\r\n")
    lines = desk.bridge.test_order("EURUSD", NOW, sleep=refuse)
    assert "REFUSED by MT4: trade is disabled (error 133)" in lines[-1]
    assert "No price" in desk.bridge.test_order("USDJPY", NOW, sleep=refuse)[0]


@pytest.mark.unit
def test_an_order_that_reached_mt4_late_is_sent_again(desk):
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    [path] = desk.ea.dir.glob("ta_cmd_*.txt")                          # MT4 answers it as stale
    cmd = mt4._kv(path.read_text())
    path.unlink()
    (desk.ea.dir / f"ta_res_{cmd['id']}.txt").write_text(
        f"id={cmd['id']}\r\nok=0\r\nmt4=0\r\nerror=0\r\nmessage=stale instruction, ignored\r\n")
    desk.sync()
    assert any("reached MT4 late" in m for m in desk.sent)
    assert not any("refused" in m for m in desk.sent)
    desk.sync()                                                        # sent again, and placed
    desk.ea.run()
    desk.sync()
    assert desk.bridge.rows()["#1043"].state == "pending"
    assert any("placed" in m for m in desk.sent)


@pytest.mark.unit
def test_an_order_mt4_refused_is_not_sent_again(desk):
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    [path] = desk.ea.dir.glob("ta_cmd_*.txt")
    cmd = mt4._kv(path.read_text())
    path.unlink()
    (desk.ea.dir / f"ta_res_{cmd['id']}.txt").write_text(
        f"id={cmd['id']}\r\nok=0\r\nmt4=0\r\nerror=134\r\nmessage=not enough money\r\n")
    for _ in range(3):
        desk.sync()
    assert not list(desk.ea.dir.glob("ta_cmd_*.txt"))
    assert desk.bridge.rows()["#1043"].state == "failed"


@pytest.mark.unit
def test_a_late_order_is_retried_at_most_three_times(desk):
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    sends = 0
    for _ in range(6):
        desk.sync()
        for path in desk.ea.dir.glob("ta_cmd_*.txt"):
            sends += 1
            cmd = mt4._kv(path.read_text())
            path.unlink()
            (desk.ea.dir / f"ta_res_{cmd['id']}.txt").write_text(
                f"id={cmd['id']}\r\nok=0\r\nmt4=0\r\nerror=0\r\nmessage=stale instruction, ignored\r\n")
    assert sends == 4                                                  # the first send and three retries


@pytest.mark.unit
def test_levels_shift_half_a_spread_so_mt4_acts_on_the_mid_price():
    # a buy fills on the ask and exits on the bid
    assert mt4.broker_levels(True, 1.1700, 1.1680, 1.1750, 0.0001) == pytest.approx((1.1701, 1.1679, 1.1749))
    # a sell fills on the bid and exits on the ask
    assert mt4.broker_levels(False, 4118.93, 4123.40, 4103.82, 0.15) == pytest.approx((4118.78, 4123.55, 4103.97))
    entry, sl, tp = mt4.broker_levels(False, 4118.93, 4123.40, 4103.82, 0.15)
    assert sl - entry == pytest.approx(4123.40 - 4118.93 + 0.30)     # risk grows by one spread
    assert entry - tp == pytest.approx(4118.93 - 4103.82 - 0.30)     # reward shrinks by one spread


@pytest.mark.unit
def test_the_spread_used_is_cmcs_now_but_never_a_blow_out():
    assert mt4.half_spread((1.1719, 1.1721), 0.0001) == pytest.approx(0.0001)
    assert mt4.half_spread((1.1700, 1.1740), 0.0001) == pytest.approx(0.00015)     # capped at 3x planned
    assert mt4.half_spread(None, 0.0002) == pytest.approx(0.0001)                  # no CMC price: planned


@pytest.mark.unit
def test_the_placed_message_shows_the_adjusted_levels(desk):
    desk.book.record([_order()], now=NOW, tickets={"EURUSD": "#1043"})
    desk.sync()
    desk.ea.run()
    desk.sync()
    [msg] = [m for m in desk.sent if "placed" in m]
    assert "at 1.1701 · SL 1.1679 · TP 1.1749" in msg and "shifted 1.0 pips for CMC's spread" in msg


@pytest.mark.unit
def test_an_older_bridge_table_gains_the_spread_column(tmp_path):
    import sqlite3
    book = Journal(tmp_path / "j.db")
    with sqlite3.connect(book.path) as db:
        db.execute(mt4.TABLE.replace(",\n    half_spread REAL", ""))
    (tmp_path / "Files").mkdir()
    Bridge(tmp_path / "Files", book)
    with sqlite3.connect(book.path) as db:
        assert "half_spread" in {r[1] for r in db.execute("PRAGMA table_info(mt4_orders)")}


@pytest.mark.unit
def test_the_bridge_records_cmcs_spreads_by_new_york_hour(desk):
    desk.sync()
    desk.sync()                                                        # same state file: not counted twice
    desk.ea.time = NOW + timedelta(minutes=1)
    desk.ea.write_state()
    desk.sync(NOW + timedelta(minutes=1))
    rec = desk.bridge.spreads()
    n, avg, low, high = rec["EURUSD"][6]                               # 10:00 UTC is 06:00 New York
    assert n == 2 and avg == pytest.approx(2.0) and low == high
    assert rec["XAUUSD"][6][1] == pytest.approx(4.0)                   # GOLD, 0.4 wide, in 0.1 pips
    from tradingagents.fx import lab
    costs = lab.broker_spreads(rec, min_samples=2)
    assert costs["EURUSD"] == pytest.approx(2.0) and costs["USDJPY"] == lab.TYPICAL_SPREAD_PIPS["USDJPY"]
    assert lab.broker_spreads(rec)["EURUSD"] == lab.TYPICAL_SPREAD_PIPS["EURUSD"]    # too few readings yet
