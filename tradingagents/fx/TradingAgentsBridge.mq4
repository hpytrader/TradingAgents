//+------------------------------------------------------------------+
//| TradingAgentsBridge.mq4                                          |
//| Places and manages the FX desk's orders from instruction files.  |
//|                                                                  |
//| The watcher (tradingagents fx-watch --mt4) writes one file per   |
//| instruction into MQL4/Files: ta_cmd_<id>.txt. Every second this  |
//| EA reads them in order, acts, and answers in ta_res_<id>.txt.    |
//| It also writes ta_state.txt (account, its orders, recent         |
//| history, prices) and ta_symbols.txt (every broker symbol).       |
//|                                                                  |
//| Safety rails that hold even if the watcher stops:                |
//|  - it only touches orders with its own MagicNumber;              |
//|  - it refuses any order larger than MaxLots;                     |
//|  - it never holds more than MaxOpenOrders of its own orders;     |
//|  - it ignores an instruction older than its valid_until time;    |
//|  - it deletes each pending order at its cancel time and closes   |
//|    each filled trade at its close time (16:55 New York), both    |
//|    remembered in ta_book.txt so a restart keeps them.            |
//+------------------------------------------------------------------+
#property strict
#property description "TradingAgents FX desk bridge: orders from files, with lot and time limits."

#include <stdlib.mqh>

input int    MagicNumber    = 20261006;  // marks this desk's orders
input double MaxLots        = 0.01;      // refuse anything larger
input int    MaxOpenOrders  = 6;         // pending + open orders of this desk
input int    SlippagePoints = 30;        // for closing at market

string CMD_PREFIX = "ta_cmd_";
string RES_PREFIX = "ta_res_";
string STATE_FILE = "ta_state.txt";
string BOOK_FILE  = "ta_book.txt";
string SYMS_FILE  = "ta_symbols.txt";

int      bookTicket[];
datetime bookExpires[];   // GMT; 0 = none
datetime bookCloseBy[];   // GMT; 0 = none
datetime bookTried[];     // last attempt to enforce, to retry every 30 s

//+------------------------------------------------------------------+
int OnInit()
  {
   LoadBook();
   WriteSymbols();
   EventSetTimer(1);
   WriteState();
   Print("TradingAgents bridge started: magic ", MagicNumber, ", max lots ", DoubleToString(MaxLots, 2),
         ", max orders ", MaxOpenOrders);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTimer()
  {
   ProcessCommands();
   EnforceTimes();
   WriteState();
  }

void OnTick()
  {
  }

//+------------------------------------------------------------------+
//| Small helpers                                                    |
//+------------------------------------------------------------------+
string Trim(string s)
  {
   int start = 0, end = StringLen(s) - 1;
   while(start <= end)
     {
      ushort c = StringGetCharacter(s, start);
      if(c == ' ' || c == '\t' || c == '\r' || c == '\n') start++;
      else break;
     }
   while(end >= start)
     {
      ushort c = StringGetCharacter(s, end);
      if(c == ' ' || c == '\t' || c == '\r' || c == '\n') end--;
      else break;
     }
   if(end < start) return("");
   return(StringSubstr(s, start, end - start + 1));
  }

string Value(string &keys[], string &vals[], string key)
  {
   for(int i = 0; i < ArraySize(keys); i++)
      if(keys[i] == key) return(vals[i]);
   return("");
  }

bool ReadCommand(string file, string &keys[], string &vals[])
  {
   int fh = FileOpen(file, FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
   if(fh == INVALID_HANDLE) return(false);
   int n = 0;
   while(!FileIsEnding(fh))
     {
      string line = FileReadString(fh);
      int eq = StringFind(line, "=");
      if(eq <= 0) continue;
      ArrayResize(keys, n + 1);
      ArrayResize(vals, n + 1);
      keys[n] = Trim(StringSubstr(line, 0, eq));
      vals[n] = Trim(StringSubstr(line, eq + 1));
      n++;
     }
   FileClose(fh);
   return(true);
  }

void WriteAtomic(string name, string text)
  {
   string tmp = name + ".part";
   int fh = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      Print("TradingAgents bridge: cannot write ", tmp, " error ", GetLastError());
      return;
     }
   FileWriteString(fh, text);
   FileClose(fh);
   if(!FileMove(tmp, 0, name, FILE_REWRITE))
      Print("TradingAgents bridge: cannot replace ", name, " error ", GetLastError());
  }

void Answer(string id, bool ok, int ticket, int code, string message, double price)
  {
   string text = "id=" + id + "\r\n";
   text += "ok=" + (ok ? "1" : "0") + "\r\n";
   text += "mt4=" + IntegerToString(ticket) + "\r\n";
   text += "error=" + IntegerToString(code) + "\r\n";
   text += "price=" + DoubleToString(price, 6) + "\r\n";
   text += "message=" + message + "\r\n";
   text += "time=" + IntegerToString((long)TimeGMT()) + "\r\n";
   WriteAtomic(RES_PREFIX + id + ".txt", text);
  }

bool Ours(int ticket)
  {
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) return(false);
   return(OrderMagicNumber() == MagicNumber);
  }

int CountOurs()
  {
   int n = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderMagicNumber() == MagicNumber) n++;
   return(n);
  }

int FindByComment(string comment)
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderMagicNumber() == MagicNumber
         && StringFind(OrderComment(), comment) == 0)
         return(OrderTicket());
   return(-1);
  }

int ServerOffset()
  {
   // server time minus GMT, rounded to the half hour
   return((int)(MathRound((double)(TimeCurrent() - TimeGMT()) / 1800.0) * 1800));
  }

//+------------------------------------------------------------------+
//| The book of cancel and close times                               |
//+------------------------------------------------------------------+
void LoadBook()
  {
   ArrayResize(bookTicket, 0);
   ArrayResize(bookExpires, 0);
   ArrayResize(bookCloseBy, 0);
   ArrayResize(bookTried, 0);
   int fh = FileOpen(BOOK_FILE, FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
   if(fh == INVALID_HANDLE) return;
   while(!FileIsEnding(fh))
     {
      string parts[];
      string line = Trim(FileReadString(fh));
      if(StringSplit(line, ',', parts) != 3) continue;
      AddToBook((int)StringToInteger(parts[0]), (datetime)StringToInteger(parts[1]),
                (datetime)StringToInteger(parts[2]), false);
     }
   FileClose(fh);
  }

void SaveBook()
  {
   string text = "";
   for(int i = 0; i < ArraySize(bookTicket); i++)
      text += IntegerToString(bookTicket[i]) + "," + IntegerToString((long)bookExpires[i]) + ","
              + IntegerToString((long)bookCloseBy[i]) + "\r\n";
   WriteAtomic(BOOK_FILE, text);
  }

void AddToBook(int ticket, datetime expires, datetime closeBy, bool save)
  {
   int n = ArraySize(bookTicket);
   ArrayResize(bookTicket, n + 1);
   ArrayResize(bookExpires, n + 1);
   ArrayResize(bookCloseBy, n + 1);
   ArrayResize(bookTried, n + 1);
   bookTicket[n] = ticket;
   bookExpires[n] = expires;
   bookCloseBy[n] = closeBy;
   bookTried[n] = 0;
   if(save) SaveBook();
  }

void RemoveFromBook(int k)
  {
   int n = ArraySize(bookTicket);
   for(int i = k; i < n - 1; i++)
     {
      bookTicket[i] = bookTicket[i + 1];
      bookExpires[i] = bookExpires[i + 1];
      bookCloseBy[i] = bookCloseBy[i + 1];
      bookTried[i] = bookTried[i + 1];
     }
   ArrayResize(bookTicket, n - 1);
   ArrayResize(bookExpires, n - 1);
   ArrayResize(bookCloseBy, n - 1);
   ArrayResize(bookTried, n - 1);
  }

void EnforceTimes()
  {
   if(!IsTradeAllowed()) return;
   datetime now = TimeGMT();
   bool changed = false;
   for(int k = ArraySize(bookTicket) - 1; k >= 0; k--)
     {
      int t = bookTicket[k];
      if(!OrderSelect(t, SELECT_BY_TICKET) || OrderCloseTime() != 0)
        {
         RemoveFromBook(k);            // finished, or gone
         changed = true;
         continue;
        }
      if(now - bookTried[k] < 30) continue;
      int type = OrderType();
      if(type >= 2 && bookExpires[k] > 0 && now >= bookExpires[k])
        {
         bookTried[k] = now;
         if(OrderDelete(t)) Print("TradingAgents bridge: cancelled unfilled #", t, " at its cancel time");
         else Print("TradingAgents bridge: could not cancel #", t, ": ", ErrorDescription(GetLastError()));
        }
      else if(type < 2 && bookCloseBy[k] > 0 && now >= bookCloseBy[k])
        {
         bookTried[k] = now;
         if(CloseMarket(t)) Print("TradingAgents bridge: closed #", t, " at the New York close");
         else Print("TradingAgents bridge: could not close #", t, ": ", ErrorDescription(GetLastError()));
        }
     }
   if(changed) SaveBook();
  }

bool CloseMarket(int ticket)
  {
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) return(false);
   RefreshRates();
   string sym = OrderSymbol();
   double price = (OrderType() == OP_BUY) ? MarketInfo(sym, MODE_BID) : MarketInfo(sym, MODE_ASK);
   return(OrderClose(ticket, OrderLots(), price, SlippagePoints, clrNONE));
  }

//+------------------------------------------------------------------+
//| Instructions                                                     |
//+------------------------------------------------------------------+
void ProcessCommands()
  {
   string names[];
   string name;
   int n = 0;
   long h = FileFindFirst(CMD_PREFIX + "*.txt", name);
   if(h == INVALID_HANDLE) return;
   do
     {
      ArrayResize(names, n + 1);
      names[n] = name;
      n++;
     }
   while(FileFindNext(h, name));
   FileFindClose(h);

   for(int i = 0; i < n - 1; i++)             // oldest instruction first
      for(int j = 0; j < n - 1 - i; j++)
         if(StringCompare(names[j], names[j + 1]) > 0)
           {
            string swap = names[j];
            names[j] = names[j + 1];
            names[j + 1] = swap;
           }

   for(int i = 0; i < n; i++)
     {
      string keys[], vals[];
      if(!ReadCommand(names[i], keys, vals)) continue;     // still being written: next second
      Handle(keys, vals);
      FileDelete(names[i]);
     }
  }

void Handle(string &keys[], string &vals[])
  {
   string id = Value(keys, vals, "id");
   string action = Value(keys, vals, "action");
   datetime validUntil = (datetime)StringToInteger(Value(keys, vals, "valid_until"));
   if(validUntil > 0 && TimeGMT() > validUntil)
     {
      Answer(id, false, 0, 0, "stale instruction, ignored", 0);
      return;
     }
   if(action == "ping")
     {
      Answer(id, true, 0, 0, "pong: account " + IntegerToString(AccountNumber()) + ", trading "
             + (IsTradeAllowed() ? "allowed" : "NOT allowed (turn AutoTrading on)"), 0);
      return;
     }
   if(!IsTradeAllowed())
     {
      Answer(id, false, 0, 0, "AutoTrading is off in MT4", 0);
      return;
     }
   if(action == "place") Place(id, keys, vals);
   else if(action == "modify") Modify(id, keys, vals);
   else if(action == "cancel" || action == "close") CloseOrCancel(id, keys, vals);
   else if(action == "flatten") Flatten(id);
   else Answer(id, false, 0, 0, "unknown action " + action, 0);
  }

void Place(string id, string &keys[], string &vals[])
  {
   string sym = Value(keys, vals, "symbol");
   string side = Value(keys, vals, "side");
   string comment = Value(keys, vals, "comment");
   double lots = StringToDouble(Value(keys, vals, "lots"));

   int existing = FindByComment(comment);
   if(comment != "" && existing > 0)
     {
      Answer(id, true, existing, 0, "already placed", 0);
      return;
     }
   if(lots <= 0 || lots > MaxLots + 0.0000001)
     {
      Answer(id, false, 0, 0, "lot size " + DoubleToString(lots, 2) + " is above MaxLots "
             + DoubleToString(MaxLots, 2), 0);
      return;
     }
   if(CountOurs() >= MaxOpenOrders)
     {
      Answer(id, false, 0, 0, "already " + IntegerToString(MaxOpenOrders) + " orders open (MaxOpenOrders)", 0);
      return;
     }
   if(!SymbolSelect(sym, true))
     {
      Answer(id, false, 0, 0, "unknown symbol " + sym, 0);
      return;
     }
   RefreshRates();
   int digits = (int)MarketInfo(sym, MODE_DIGITS);
   double point = MarketInfo(sym, MODE_POINT);
   double stopLevel = MarketInfo(sym, MODE_STOPLEVEL) * point;
   double minLot = MarketInfo(sym, MODE_MINLOT);
   double ask = MarketInfo(sym, MODE_ASK);
   double bid = MarketInfo(sym, MODE_BID);
   double price = NormalizeDouble(StringToDouble(Value(keys, vals, "price")), digits);
   double sl = NormalizeDouble(StringToDouble(Value(keys, vals, "sl")), digits);
   double tp = NormalizeDouble(StringToDouble(Value(keys, vals, "tp")), digits);
   if(lots < minLot - 0.0000001)
     {
      Answer(id, false, 0, 0, "lot size below the broker minimum " + DoubleToString(minLot, 2), 0);
      return;
     }
   int type = -1;
   if(side == "buy")
     {
      type = OP_BUYLIMIT;
      if(price >= ask - stopLevel)
        {
         Answer(id, false, 0, 0, "price is already at or through the entry (ask " + DoubleToString(ask, digits) + ")", ask);
         return;
        }
     }
   else if(side == "sell")
     {
      type = OP_SELLLIMIT;
      if(price <= bid + stopLevel)
        {
         Answer(id, false, 0, 0, "price is already at or through the entry (bid " + DoubleToString(bid, digits) + ")", bid);
         return;
        }
     }
   else
     {
      Answer(id, false, 0, 0, "unknown side " + side, 0);
      return;
     }

   datetime expires = (datetime)StringToInteger(Value(keys, vals, "expires"));
   datetime closeBy = (datetime)StringToInteger(Value(keys, vals, "close_by"));
   datetime brokerExpiry = 0;
   if(expires > 0 && expires - TimeGMT() > 15 * 60)
      brokerExpiry = expires + ServerOffset();

   int ticket = OrderSend(sym, type, lots, price, 0, sl, tp, comment, MagicNumber, brokerExpiry, clrNONE);
   int err = (ticket < 0) ? GetLastError() : 0;
   if(ticket < 0 && brokerExpiry > 0 && err == 147)   // expirations denied: the cancel time still holds
     {
      ticket = OrderSend(sym, type, lots, price, 0, sl, tp, comment, MagicNumber, 0, clrNONE);
      err = (ticket < 0) ? GetLastError() : 0;
     }
   if(ticket < 0)
     {
      Answer(id, false, 0, err, ErrorDescription(err), 0);
      return;
     }
   AddToBook(ticket, expires, closeBy, true);
   Answer(id, true, ticket, 0, "placed", price);
  }

void Modify(string id, string &keys[], string &vals[])
  {
   int t = (int)StringToInteger(Value(keys, vals, "mt4"));
   if(!Ours(t))
     {
      Answer(id, false, t, 0, "not one of this desk's orders", 0);
      return;
     }
   if(OrderCloseTime() != 0)
     {
      Answer(id, false, t, 0, "already finished", OrderClosePrice());
      return;
     }
   int digits = (int)MarketInfo(OrderSymbol(), MODE_DIGITS);
   double sl = NormalizeDouble(StringToDouble(Value(keys, vals, "sl")), digits);
   double tp = NormalizeDouble(StringToDouble(Value(keys, vals, "tp")), digits);
   double point = MarketInfo(OrderSymbol(), MODE_POINT);
   if(MathAbs(sl - OrderStopLoss()) < point / 2 && MathAbs(tp - OrderTakeProfit()) < point / 2)
     {
      Answer(id, true, t, 0, "unchanged", 0);
      return;
     }
   if(OrderModify(t, OrderOpenPrice(), sl, tp, OrderExpiration(), clrNONE))
      Answer(id, true, t, 0, "modified", 0);
   else
     {
      int err = GetLastError();
      Answer(id, false, t, err, ErrorDescription(err), 0);
     }
  }

void CloseOrCancel(string id, string &keys[], string &vals[])
  {
   int t = (int)StringToInteger(Value(keys, vals, "mt4"));
   if(!Ours(t))
     {
      Answer(id, false, t, 0, "not one of this desk's orders", 0);
      return;
     }
   if(OrderCloseTime() != 0)
     {
      Answer(id, true, t, 0, "already finished", OrderClosePrice());
      return;
     }
   bool done;
   if(OrderType() >= 2) done = OrderDelete(t);
   else done = CloseMarket(t);
   if(done)
     {
      double price = 0;
      if(OrderSelect(t, SELECT_BY_TICKET)) price = OrderClosePrice();
      Answer(id, true, t, 0, "done", price);
     }
   else
     {
      int err = GetLastError();
      Answer(id, false, t, err, ErrorDescription(err), 0);
     }
  }

void Flatten(string id)
  {
   int failed = 0, done = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES) || OrderMagicNumber() != MagicNumber) continue;
      int t = OrderTicket();
      bool ok = (OrderType() >= 2) ? OrderDelete(t) : CloseMarket(t);
      if(ok) done++;
      else failed++;
     }
   Answer(id, failed == 0, 0, 0, IntegerToString(done) + " closed or cancelled, " + IntegerToString(failed) + " failed", 0);
  }

//+------------------------------------------------------------------+
//| State for the watcher                                            |
//+------------------------------------------------------------------+
string OrderLine(string tag, bool closed)
  {
   int digits = (int)MarketInfo(OrderSymbol(), MODE_DIGITS);
   string s = tag + "=" + IntegerToString(OrderTicket()) + "|" + OrderComment() + "|" + OrderSymbol() + "|"
              + IntegerToString(OrderType()) + "|" + DoubleToString(OrderLots(), 2) + "|"
              + DoubleToString(OrderOpenPrice(), digits) + "|" + DoubleToString(OrderStopLoss(), digits) + "|"
              + DoubleToString(OrderTakeProfit(), digits) + "|"
              + DoubleToString(OrderProfit() + OrderSwap() + OrderCommission(), 2);
   if(closed)
      s += "|" + DoubleToString(OrderClosePrice(), digits) + "|"
           + IntegerToString((long)(OrderCloseTime() - ServerOffset()));
   return(s + "\r\n");
  }

void WriteState()
  {
   string text = "time=" + IntegerToString((long)TimeGMT()) + "\r\n";
   text += "account=" + IntegerToString(AccountNumber()) + "\r\n";
   text += "server=" + AccountServer() + "\r\n";
   text += "company=" + AccountCompany() + "\r\n";
   text += "currency=" + AccountCurrency() + "\r\n";
   text += "balance=" + DoubleToString(AccountBalance(), 2) + "\r\n";
   text += "equity=" + DoubleToString(AccountEquity(), 2) + "\r\n";
   text += "demo=" + (IsDemo() ? "1" : "0") + "\r\n";
   text += "connected=" + (IsConnected() ? "1" : "0") + "\r\n";
   text += "trade_allowed=" + (IsTradeAllowed() ? "1" : "0") + "\r\n";
   text += "magic=" + IntegerToString(MagicNumber) + "\r\n";
   text += "max_lots=" + DoubleToString(MaxLots, 2) + "\r\n";
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderMagicNumber() == MagicNumber)
         text += OrderLine("order", false);
   int listed = 0;
   for(int i = OrdersHistoryTotal() - 1; i >= 0 && listed < 60; i--)
      if(OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderMagicNumber() == MagicNumber
         && TimeCurrent() - OrderCloseTime() < 4 * 86400)
        {
         text += OrderLine("hist", true);
         listed++;
        }
   for(int i = 0; i < SymbolsTotal(true); i++)
     {
      string sym = SymbolName(i, true);
      int digits = (int)MarketInfo(sym, MODE_DIGITS);
      text += "quote=" + sym + "|" + DoubleToString(MarketInfo(sym, MODE_BID), digits) + "|"
              + DoubleToString(MarketInfo(sym, MODE_ASK), digits) + "\r\n";
     }
   WriteAtomic(STATE_FILE, text);
  }

void WriteSymbols()
  {
   string text = "";
   for(int i = 0; i < SymbolsTotal(false); i++)
      text += SymbolName(i, false) + "\r\n";
   WriteAtomic(SYMS_FILE, text);
  }
//+------------------------------------------------------------------+
