//+------------------------------------------------------------------+
//|                                            ZeroMQ_Listener.mq5   |
//|        QuantMind-PropFirm-Orchestrator  -  MT5 execution bridge  |
//+------------------------------------------------------------------+
//| Receives flat JSON commands from execution/mt5_bridge.py over a  |
//| ZeroMQ REP socket (default tcp://127.0.0.1:5555) and publishes   |
//| heartbeats / trade events on a PUB socket (tcp://127.0.0.1:5556).|
//|                                                                  |
//| Guarantees enforced HERE, independently of Python:               |
//|  * SEND_BRACKET requires SL and TP; they travel inside the same  |
//|    TRADE_ACTION_DEAL request as the entry (atomic). If the       |
//|    resulting position is somehow missing SL/TP it is repaired    |
//|    or closed immediately.                                        |
//|  * Every request must carry the shared auth token.               |
//|  * Duplicate client_id -> cached result, never a second order.   |
//|  * Weekend guard (backup to Python): no new orders and auto-     |
//|    flatten from Friday 19:30 to Sunday 22:00 UTC.                |
//|  * Optional dead-man switch: flatten if Python goes silent.      |
//|  * Every command and server response is appended to a JSONL      |
//|    audit log with millisecond UTC timestamps.                    |
//|                                                                  |
//| Requires the mql-zmq binding (github.com/dingmaotu/mql-zmq):     |
//|   MQL5/Include/Zmq/*.mqh, MQL5/Libraries/libzmq.dll,             |
//|   MQL5/Libraries/libsodium.dll, and "Allow DLL imports".         |
//|   See mql5/README_MT5.md.                                        |
//+------------------------------------------------------------------+
#property copyright "QuantMind-PropFirm-Orchestrator"
#property version   "1.00"
#property description "ZeroMQ execution listener: atomic brackets, FLATTEN_ALL, weekend guard, audit log"

#include <Zmq/Zmq.mqh>
#include <Trade/Trade.mqh>

//--- connection
input string InpRepEndpoint        = "tcp://127.0.0.1:5555"; // REP endpoint (commands)
input string InpPubEndpoint        = "tcp://127.0.0.1:5556"; // PUB endpoint (events)
input string InpAuthToken          = "";                     // Shared secret (REQUIRED, must match Python)
input int    InpTimerMs            = 5;                      // Poll interval, ms
input int    InpHeartbeatPubMs     = 1000;                   // PUB heartbeat interval, ms
//--- safety
input bool   InpWeekendGuard       = true;                   // Block + flatten over the weekend
input int    InpFlattenDow         = 5;                      // Flatten day (MQL: 0=Sun ... 5=Fri)
input int    InpFlattenHourUTC     = 19;                     // Flatten hour UTC (broker closes 20:00 UTC)
input int    InpFlattenMinuteUTC   = 30;                     // Flatten minute UTC
input int    InpReopenDow          = 0;                      // Reopen day (0=Sun)
input int    InpReopenHourUTC      = 22;                     // Reopen hour UTC
input int    InpDeadManSec         = 0;                      // Flatten if Python silent N sec (0=off)
input bool   InpAccountWideFlatten = true;                   // Guards flatten ALL positions (not only magic)
input long   InpMagic              = 260926;                 // Magic used when not account-wide
input int    InpMaxDeviationPts    = 50;                     // Cap on requested deviation, points
//--- logging
input string InpAuditFile          = "QuantMind\\ea_audit.jsonl"; // In MQL5/Files
input bool   InpDemoOnly           = true;                   // Refuse to run on REAL (live-money) accounts
input int    InpMaxRatesBars       = 5000;                   // Cap on bars returned by RATES

#define PROTOCOL_VERSION 1
#define DEDUP_SIZE       256
#define MAX_MSGS_PER_TICK 64

Context  g_ctx("QuantMind");   // global object: destroyed after OnDeinit deletes the sockets
Socket  *g_rep = NULL;
Socket  *g_pub = NULL;
CTrade   g_trade;

int      g_audit = INVALID_HANDLE;
long     g_epochMs = 0;          // wall-clock ms at init (from TimeGMT)
ulong    g_epochUs = 0;          // GetMicrosecondCount() at init
ulong    g_lastHeartbeatUs = 0;
ulong    g_lastGuardAttemptUs = 0;
datetime g_lastPythonMsg = 0;
bool     g_deadManFired = false;

string   g_dedupKeys[DEDUP_SIZE];
string   g_dedupData[DEDUP_SIZE];
int      g_dedupNext = 0;

//+------------------------------------------------------------------+
//| Time helpers                                                     |
//+------------------------------------------------------------------+
long NowMs() { return g_epochMs + (long)((GetMicrosecondCount() - g_epochUs) / 1000); }

string IsoMs(const long ms)
  {
   datetime sec = (datetime)(ms / 1000);
   string s = TimeToString(sec, TIME_DATE | TIME_SECONDS);   // yyyy.mm.dd hh:mi:ss
   StringReplace(s, ".", "-");
   StringReplace(s, " ", "T");
   return s + "." + StringFormat("%03d", (int)(ms % 1000)) + "Z";
  }

//+------------------------------------------------------------------+
//| JSON helpers (flat, strict: the Python side guarantees no quotes |
//| or backslashes inside string values)                             |
//+------------------------------------------------------------------+
string JsonEscape(const string s)
  {
   string out = s;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", "\\r");
   StringReplace(out, "\n", "\\n");
   StringReplace(out, "\t", "\\t");
   return out;
  }

string JStr(const string key, const string value) { return "\"" + key + "\":\"" + JsonEscape(value) + "\""; }
string JNum(const string key, const double value, const int digits) { return "\"" + key + "\":" + DoubleToString(value, digits); }
string JInt(const string key, const long value) { return "\"" + key + "\":" + IntegerToString(value); }
string JBool(const string key, const bool value) { return "\"" + key + "\":" + (value ? "true" : "false"); }

// Locate the first character of the value for "key": ... Returns -1 if absent.
int JsonValueStart(const string &json, const string key)
  {
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0)
      return -1;
   pos += StringLen(pattern);
   while(pos < StringLen(json) && StringGetCharacter(json, pos) == ' ')
      pos++;
   return pos;
  }

bool JsonGetString(const string &json, const string key, string &out)
  {
   int p = JsonValueStart(json, key);
   if(p < 0 || StringGetCharacter(json, p) != '"')
      return false;
   int end = StringFind(json, "\"", p + 1);
   if(end < 0)
      return false;
   out = StringSubstr(json, p + 1, end - p - 1);
   return true;
  }

string JsonRawToken(const string &json, const int start)
  {
   int i = start;
   int n = StringLen(json);
   while(i < n)
     {
      ushort c = StringGetCharacter(json, i);
      if(c == ',' || c == '}' || c == ' ')
         break;
      i++;
     }
   return StringSubstr(json, start, i - start);
  }

bool JsonGetDouble(const string &json, const string key, double &out)
  {
   int p = JsonValueStart(json, key);
   if(p < 0)
      return false;
   string tok = JsonRawToken(json, p);
   if(tok == "" || tok == "null")
      return false;
   out = StringToDouble(tok);
   return true;
  }

bool JsonGetLong(const string &json, const string key, long &out)
  {
   int p = JsonValueStart(json, key);
   if(p < 0)
      return false;
   string tok = JsonRawToken(json, p);
   if(tok == "" || tok == "null")
      return false;
   out = StringToInteger(tok);
   return true;
  }

string ReplyOk(const string id, const string data) { return "{\"ok\":true,\"id\":\"" + JsonEscape(id) + "\",\"data\":" + data + "}"; }
string ReplyErr(const string id, const string err) { return "{\"ok\":false,\"id\":\"" + JsonEscape(id) + "\",\"error\":\"" + JsonEscape(err) + "\"}"; }

//+------------------------------------------------------------------+
//| Audit log & event publishing                                     |
//+------------------------------------------------------------------+
void Publish(const string json)
  {
   if(g_pub == NULL)
      return;
   ZmqMsg msg(json);
   g_pub.send(msg, true);   // non-blocking; PUB drops if no subscriber
  }

void Audit(const string event, const string fields)
  {
   long ms = NowMs();
   string line = "{\"ts\":\"" + IsoMs(ms) + "\",\"ts_ms\":" + IntegerToString(ms) +
                 ",\"server_time\":\"" + TimeToString(TimeTradeServer(), TIME_DATE | TIME_SECONDS) +
                 "\",\"event\":\"" + event + "\"" + (fields == "" ? "" : "," + fields) + "}";
   if(g_audit != INVALID_HANDLE)
     {
      FileSeek(g_audit, 0, SEEK_END);
      FileWriteString(g_audit, line + "\r\n");
      FileFlush(g_audit);
     }
   Publish("{\"type\":\"audit\",\"line\":" + line + "}");
  }

//+------------------------------------------------------------------+
//| Guards                                                           |
//+------------------------------------------------------------------+
bool InWeekendBlackout()
  {
   if(!InpWeekendGuard)
      return false;
   MqlDateTime t;
   TimeToStruct(TimeGMT(), t);
   int mow   = t.day_of_week * 1440 + t.hour * 60 + t.min;
   int start = InpFlattenDow * 1440 + InpFlattenHourUTC * 60 + InpFlattenMinuteUTC;
   int end   = InpReopenDow * 1440 + InpReopenHourUTC * 60;
   if(start < end)
      return (mow >= start && mow < end);
   return (mow >= start || mow < end);   // window wraps past Saturday->Sunday
  }

bool TradingPermitted(string &why)
  {
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))       { why = "terminal not connected"; return false; }
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))   { why = "Algo Trading disabled in terminal"; return false; }
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))             { why = "live trading not allowed for this EA"; return false; }
   if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))     { why = "account does not allow trading"; return false; }
   if(!AccountInfoInteger(ACCOUNT_TRADE_EXPERT))      { why = "account forbids expert trading"; return false; }
   return true;
  }

bool MagicMatches(const long positionMagic, const long filter) { return filter < 0 || positionMagic == filter; }

//+------------------------------------------------------------------+
//| Dedup cache (client_id -> result JSON)                           |
//+------------------------------------------------------------------+
bool DedupLookup(const string key, string &data)
  {
   if(key == "")
      return false;
   for(int i = 0; i < DEDUP_SIZE; i++)
      if(g_dedupKeys[i] == key)
        {
         data = g_dedupData[i];
         return true;
        }
   return false;
  }

void DedupStore(const string key, const string data)
  {
   if(key == "")
      return;
   g_dedupKeys[g_dedupNext] = key;
   g_dedupData[g_dedupNext] = data;
   g_dedupNext = (g_dedupNext + 1) % DEDUP_SIZE;
  }

//+------------------------------------------------------------------+
//| Core trade operations                                            |
//+------------------------------------------------------------------+
int CancelAll(const long magicFilter, string &failed)
  {
   int cancelled = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0 || !MagicMatches(OrderGetInteger(ORDER_MAGIC), magicFilter))
         continue;
      if(g_trade.OrderDelete(ticket))
         cancelled++;
      else
         failed += (failed == "" ? "" : ",") + IntegerToString((long)ticket);
     }
   return cancelled;
  }

bool ClosePositionWithRetry(const ulong ticket, const ulong deviation)
  {
   for(int attempt = 0; attempt < 3; attempt++)
     {
      if(!PositionSelectByTicket(ticket))
         return true;                                   // already closed
      if(g_trade.PositionClose(ticket, deviation))
        {
         uint rc = g_trade.ResultRetcode();
         if(rc == TRADE_RETCODE_DONE || rc == TRADE_RETCODE_DONE_PARTIAL || rc == TRADE_RETCODE_PLACED)
            return true;
        }
      uint rc2 = g_trade.ResultRetcode();
      if(rc2 != TRADE_RETCODE_REQUOTE && rc2 != TRADE_RETCODE_PRICE_CHANGED && rc2 != TRADE_RETCODE_PRICE_OFF)
         break;                                         // non-transient error: stop retrying
     }
   return !PositionSelectByTicket(ticket);
  }

// Cancel pending orders first (nothing can fill behind us), then close every position.
string FlattenAll(const long magicFilter, const ulong deviation, const string reason)
  {
   ulong t0 = GetMicrosecondCount();
   string failed = "";
   int cancelled = CancelAll(magicFilter, failed);
   int closed = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !MagicMatches(PositionGetInteger(POSITION_MAGIC), magicFilter))
         continue;
      if(ClosePositionWithRetry(ticket, deviation))
         closed++;
      else
         failed += (failed == "" ? "" : ",") + IntegerToString((long)ticket);
     }
   double elapsedMs = (double)(GetMicrosecondCount() - t0) / 1000.0;
   string data = "{" + JInt("closed", closed) + "," + JInt("cancelled", cancelled) +
                 ",\"failed\":[" + failed + "]," + JNum("elapsed_ms", elapsedMs, 3) + "}";
   Audit("flatten_all", JStr("reason", reason) + "," + JInt("magic_filter", magicFilter) +
         ",\"result\":" + data);
   return data;
  }

string PositionsJson(const long magicFilter)
  {
   string out = "[";
   bool first = true;
   for(int i = 0; i < PositionsTotal(); i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !MagicMatches(PositionGetInteger(POSITION_MAGIC), magicFilter))
         continue;
      string sym = PositionGetString(POSITION_SYMBOL);
      int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
      out += (first ? "" : ",") + "{" + JInt("ticket", (long)ticket) + "," + JStr("symbol", sym) + "," +
             JInt("type", PositionGetInteger(POSITION_TYPE)) + "," +
             JNum("volume", PositionGetDouble(POSITION_VOLUME), 8) + "," +
             JNum("price_open", PositionGetDouble(POSITION_PRICE_OPEN), digits) + "," +
             JNum("sl", PositionGetDouble(POSITION_SL), digits) + "," +
             JNum("tp", PositionGetDouble(POSITION_TP), digits) + "," +
             JInt("magic", PositionGetInteger(POSITION_MAGIC)) + "," +
             JStr("comment", PositionGetString(POSITION_COMMENT)) + "," +
             JNum("profit", PositionGetDouble(POSITION_PROFIT), 2) + "}";
      first = false;
     }
   return out + "]";
  }

string PendingOrdersJson(const long magicFilter)
  {
   string out = "[";
   bool first = true;
   for(int i = 0; i < OrdersTotal(); i++)
     {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0 || !MagicMatches(OrderGetInteger(ORDER_MAGIC), magicFilter))
         continue;
      out += (first ? "" : ",") + "{" + JInt("ticket", (long)ticket) + "," +
             JStr("symbol", OrderGetString(ORDER_SYMBOL)) + "," + JInt("type", OrderGetInteger(ORDER_TYPE)) + "," +
             JInt("magic", OrderGetInteger(ORDER_MAGIC)) + "}";
      first = false;
     }
   return out + "]";
  }

//+------------------------------------------------------------------+
//| SEND_BRACKET: entry + SL + TP in ONE TRADE_ACTION_DEAL request   |
//+------------------------------------------------------------------+
string SendBracket(const string &req, string &err)
  {
   string symbol, side, clientId, comment;
   double volume = 0, sl = 0, tp = 0;
   long deviation = 20, magic = 0;
   if(!JsonGetString(req, "symbol", symbol) || !JsonGetString(req, "side", side) ||
      !JsonGetDouble(req, "volume", volume))
     { err = "symbol/side/volume required"; return ""; }
   // Hard rule: no bracket, no trade.
   if(!JsonGetDouble(req, "sl", sl) || !JsonGetDouble(req, "tp", tp) || sl <= 0 || tp <= 0)
     { err = "SL and TP are mandatory"; return ""; }
   JsonGetLong(req, "deviation", deviation);
   JsonGetLong(req, "magic", magic);
   JsonGetString(req, "client_id", clientId);
   JsonGetString(req, "comment", comment);

   string cached;
   if(DedupLookup(clientId, cached))
     {
      Audit("send_bracket_duplicate", JStr("client_id", clientId));
      return cached;
     }
   if(InpDemoOnly && IsRealAccount())
     { err = "demo guard: REAL account - orders refused"; return ""; }
   if(InWeekendBlackout())
     { err = "weekend guard: new orders blocked"; return ""; }
   string why;
   if(!TradingPermitted(why))
     { err = why; return ""; }
   if(!SymbolSelect(symbol, true))
     { err = "unknown symbol " + symbol; return ""; }
   if(AccountInfoInteger(ACCOUNT_MARGIN_MODE) == ACCOUNT_MARGIN_MODE_RETAIL_NETTING && PositionSelect(symbol))
     { err = "netting account already has a position on " + symbol + " (would merge and overwrite SL/TP)"; return ""; }

   bool isBuy = (side == "BUY");
   if(!isBuy && side != "SELL")
     { err = "side must be BUY or SELL"; return ""; }
   double ask   = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid   = SymbolInfoDouble(symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits   = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double minDist = (double)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double ref = isBuy ? ask : bid;
   if(isBuy && !(sl < ref && tp > ref))   { err = "BUY requires sl < ask < tp"; return ""; }
   if(!isBuy && !(tp < ref && sl > ref))  { err = "SELL requires tp < bid < sl"; return ""; }
   if(MathAbs(ref - sl) < minDist || MathAbs(tp - ref) < minDist)
     { err = "SL/TP inside broker stops level"; return ""; }

   double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vmax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   volume = MathFloor(volume / vstep + 1e-9) * vstep;
   volume = MathMin(volume, vmax);
   if(volume < vmin)
     { err = "volume below symbol minimum"; return ""; }

   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   g_trade.SetExpertMagicNumber((ulong)magic);
   g_trade.SetDeviationInPoints((ulong)MathMin(deviation, (long)InpMaxDeviationPts));
   g_trade.SetTypeFillingBySymbol(symbol);
   g_trade.SetAsyncMode(false);

   ulong t0 = GetMicrosecondCount();
   bool sent = isBuy ? g_trade.Buy(volume, symbol, 0.0, sl, tp, comment)
                     : g_trade.Sell(volume, symbol, 0.0, sl, tp, comment);
   double latencyMs = (double)(GetMicrosecondCount() - t0) / 1000.0;
   uint rc = g_trade.ResultRetcode();
   ulong deal = g_trade.ResultDeal();
   string repair = "";

   // Defence in depth: the position MUST carry SL and TP. Repair or close it otherwise.
   if(sent && deal > 0 && HistoryDealSelect(deal))
     {
      ulong posId = (ulong)HistoryDealGetInteger(deal, DEAL_POSITION_ID);
      if(posId > 0 && PositionSelectByTicket(posId) &&
         (PositionGetDouble(POSITION_SL) == 0.0 || PositionGetDouble(POSITION_TP) == 0.0))
        {
         if(g_trade.PositionModify(posId, sl, tp))
            repair = "sl_tp_repaired";
         else
           {
            ClosePositionWithRetry(posId, (ulong)InpMaxDeviationPts);
            repair = "sl_tp_missing_position_closed";
           }
        }
     }

   string data = "{" + JInt("retcode", rc) + "," + JInt("order", (long)g_trade.ResultOrder()) + "," +
                 JInt("deal", (long)deal) + "," + JNum("price", g_trade.ResultPrice(), digits) + "," +
                 JNum("volume", g_trade.ResultVolume(), 8) + "," +
                 JStr("comment", g_trade.ResultRetcodeDescription()) + "," + JStr("repair", repair) + "," +
                 JNum("latency_ms", latencyMs, 3) + "}";
   Audit("send_bracket", JStr("client_id", clientId) + "," + JStr("symbol", symbol) + "," + JStr("side", side) + "," +
         JNum("volume", volume, 8) + "," + JNum("sl", sl, digits) + "," + JNum("tp", tp, digits) + "," +
         JInt("magic", magic) + ",\"result\":" + data);
   if(repair == "sl_tp_missing_position_closed")
     { err = "position opened without SL/TP and was closed"; return ""; }
   DedupStore(clientId, data);
   return data;
  }

//+------------------------------------------------------------------+
//| Account type, server clock offset, timeframes                    |
//+------------------------------------------------------------------+
string TradeModeName(const long mode)
  {
   if(mode == ACCOUNT_TRADE_MODE_DEMO)    return "DEMO";
   if(mode == ACCOUNT_TRADE_MODE_CONTEST) return "CONTEST";
   if(mode == ACCOUNT_TRADE_MODE_REAL)    return "REAL";
   return "UNKNOWN";
  }

bool IsRealAccount() { return AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_REAL; }

// Trade-server clock minus UTC, rounded to 15 minutes (MT5 bar times are server time).
long ServerOffsetSec()
  {
   long raw = (long)(TimeTradeServer() - TimeGMT());
   return (long)MathRound((double)raw / 900.0) * 900;
  }

bool ParseTimeframe(const string tf, ENUM_TIMEFRAMES &out)
  {
   if(tf == "M1")  { out = PERIOD_M1;  return true; }
   if(tf == "M5")  { out = PERIOD_M5;  return true; }
   if(tf == "M15") { out = PERIOD_M15; return true; }
   if(tf == "M30") { out = PERIOD_M30; return true; }
   if(tf == "H1")  { out = PERIOD_H1;  return true; }
   if(tf == "H4")  { out = PERIOD_H4;  return true; }
   if(tf == "D1")  { out = PERIOD_D1;  return true; }
   return false;
  }

// Closed bars only: CopyRates starting at shift 1 skips the still-forming bar.
string RatesJson(const string sym, const ENUM_TIMEFRAMES tf, const int count, string &err)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int n = CopyRates(sym, tf, 1, count, rates);
   if(n <= 0)
     {
      err = "CopyRates failed (" + IntegerToString(GetLastError()) + ") - history not loaded yet?";
      return "";
     }
   int d = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   string out = "{" + JStr("symbol", sym) + "," + JInt("server_offset", ServerOffsetSec()) + ",\"bars\":[";
   for(int i = 0; i < n; i++)
     {
      StringAdd(out, (i == 0 ? "" : ",") + "[" + IntegerToString((long)rates[i].time) + "," +
                DoubleToString(rates[i].open, d) + "," + DoubleToString(rates[i].high, d) + "," +
                DoubleToString(rates[i].low, d) + "," + DoubleToString(rates[i].close, d) + "," +
                IntegerToString((long)rates[i].tick_volume) + "]");
     }
   StringAdd(out, "]}");
   return out;
  }

//+------------------------------------------------------------------+
//| Request dispatcher                                               |
//+------------------------------------------------------------------+
string HandleRequest(const string raw)
  {
   string id = "", token = "", action = "";
   long v = 0;
   JsonGetString(raw, "id", id);
   if(!JsonGetLong(raw, "v", v) || v != PROTOCOL_VERSION)
      return ReplyErr(id, "unsupported protocol version");
   if(!JsonGetString(raw, "token", token) || token != InpAuthToken)
     {
      Audit("auth_rejected", JStr("id", id));
      return ReplyErr(id, "authentication failed");
     }
   if(!JsonGetString(raw, "action", action))
      return ReplyErr(id, "missing action");
   g_lastPythonMsg = TimeGMT();
   g_deadManFired = false;

   long magic = -1, deviation = 20, ticket = 0;
   JsonGetLong(raw, "magic", magic);
   JsonGetLong(raw, "deviation", deviation);
   ulong dev = (ulong)MathMin(deviation, (long)InpMaxDeviationPts);
   string err = "";

   if(action == "PING")
     {
      // trade_allowed = terminal/account permissions only. The weekend window is a separate
      // signal (weekend_blackout); SEND_BRACKET enforces it independently.
      string why;
      bool permitted = TradingPermitted(why);
      return ReplyOk(id, "{" + JBool("trade_allowed", permitted) + "," + JStr("reason", why) + "," +
                     JBool("connected", (bool)TerminalInfoInteger(TERMINAL_CONNECTED)) + "," +
                     JBool("weekend_blackout", InWeekendBlackout()) + "," +
                     JInt("server_time", (long)TimeTradeServer()) + "," + JInt("gmt_ms", NowMs()) + "," +
                     JNum("equity", AccountInfoDouble(ACCOUNT_EQUITY), 2) + "," +
                     JInt("positions", PositionsTotal()) + "," +
                     JInt("trade_mode", AccountInfoInteger(ACCOUNT_TRADE_MODE)) + "," +
                     JStr("trade_mode_name", TradeModeName(AccountInfoInteger(ACCOUNT_TRADE_MODE))) + "," +
                     JInt("login", AccountInfoInteger(ACCOUNT_LOGIN)) + "}");
     }
   if(action == "ACCOUNT_INFO")
     {
      return ReplyOk(id, "{" + JInt("login", AccountInfoInteger(ACCOUNT_LOGIN)) + "," +
                     JInt("trade_mode", AccountInfoInteger(ACCOUNT_TRADE_MODE)) + "," +
                     JStr("trade_mode_name", TradeModeName(AccountInfoInteger(ACCOUNT_TRADE_MODE))) + "," +
                     JStr("server", AccountInfoString(ACCOUNT_SERVER)) + "," +
                     JStr("currency", AccountInfoString(ACCOUNT_CURRENCY)) + "," +
                     JNum("equity", AccountInfoDouble(ACCOUNT_EQUITY), 2) + "," +
                     JNum("balance", AccountInfoDouble(ACCOUNT_BALANCE), 2) + "," +
                     JNum("margin_free", AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2) + "," +
                     JInt("leverage", AccountInfoInteger(ACCOUNT_LEVERAGE)) + "," +
                     JBool("trade_allowed", (bool)AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) + "," +
                     JBool("trade_expert", (bool)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) &&
                           (bool)MQLInfoInteger(MQL_TRADE_ALLOWED) &&
                           (bool)AccountInfoInteger(ACCOUNT_TRADE_EXPERT)) + "}");
     }
   if(action == "SYMBOL_INFO")
     {
      string sym;
      if(!JsonGetString(raw, "symbol", sym) || !SymbolSelect(sym, true))
         return ReplyErr(id, "unknown symbol");
      int d = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
      return ReplyOk(id, "{" + JStr("symbol", sym) + "," + JInt("digits", d) + "," +
                     JNum("point", SymbolInfoDouble(sym, SYMBOL_POINT), 10) + "," +
                     JNum("trade_tick_size", SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE), 10) + "," +
                     JNum("bid", SymbolInfoDouble(sym, SYMBOL_BID), d) + "," +
                     JNum("ask", SymbolInfoDouble(sym, SYMBOL_ASK), d) + "," +
                     JNum("volume_min", SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN), 8) + "," +
                     JNum("volume_max", SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX), 8) + "," +
                     JNum("volume_step", SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP), 8) + "," +
                     JNum("trade_contract_size", SymbolInfoDouble(sym, SYMBOL_TRADE_CONTRACT_SIZE), 4) + "," +
                     JInt("trade_stops_level", SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL)) + "," +
                     JInt("filling_mode", SymbolInfoInteger(sym, SYMBOL_FILLING_MODE)) + "}");
     }
   if(action == "SEND_BRACKET")
     {
      string data = SendBracket(raw, err);
      if(data == "")
        {
         Audit("send_bracket_rejected", JStr("id", id) + "," + JStr("error", err));
         return ReplyErr(id, err);
        }
      return ReplyOk(id, data);
     }
   if(action == "MODIFY_SLTP")
     {
      double sl = 0, tp = 0;
      JsonGetLong(raw, "ticket", ticket);
      if(!JsonGetDouble(raw, "sl", sl) || !JsonGetDouble(raw, "tp", tp) || sl <= 0 || tp <= 0)
         return ReplyErr(id, "MODIFY_SLTP requires both SL and TP (never strips a bracket)");
      bool ok = g_trade.PositionModify((ulong)ticket, sl, tp);
      Audit("modify_sltp", JInt("ticket", ticket) + "," + JNum("sl", sl, 8) + "," + JNum("tp", tp, 8) + "," +
            JInt("retcode", g_trade.ResultRetcode()));
      return ReplyOk(id, "{" + JInt("retcode", ok ? (long)g_trade.ResultRetcode() : 0) + "}");
     }
   if(action == "POSITIONS")
      return ReplyOk(id, PositionsJson(magic));
   if(action == "PENDING_ORDERS")
      return ReplyOk(id, PendingOrdersJson(magic));
   if(action == "CANCEL_ALL")
     {
      string failed = "";
      int n = CancelAll(magic, failed);
      Audit("cancel_all", JInt("cancelled", n) + ",\"failed\":[" + failed + "]");
      return ReplyOk(id, "{" + JInt("cancelled", n) + ",\"failed\":[" + failed + "]}");
     }
   if(action == "CLOSE_POSITION")
     {
      JsonGetLong(raw, "ticket", ticket);
      bool ok = ClosePositionWithRetry((ulong)ticket, dev);
      Audit("close_position", JInt("ticket", ticket) + "," + JBool("ok", ok) + "," + JInt("retcode", g_trade.ResultRetcode()));
      return ReplyOk(id, "{" + JInt("retcode", ok ? (long)TRADE_RETCODE_DONE : (long)g_trade.ResultRetcode()) + "}");
     }
   if(action == "RATES")
     {
      string sym, tfs;
      long count = 500;
      ENUM_TIMEFRAMES tf;
      if(!JsonGetString(raw, "symbol", sym) || !SymbolSelect(sym, true))
         return ReplyErr(id, "unknown symbol");
      if(!JsonGetString(raw, "timeframe", tfs) || !ParseTimeframe(tfs, tf))
         return ReplyErr(id, "timeframe must be one of M1,M5,M15,M30,H1,H4,D1");
      JsonGetLong(raw, "count", count);
      count = MathMax(1, MathMin(count, (long)InpMaxRatesBars));
      string data = RatesJson(sym, tf, (int)count, err);
      if(data == "")
         return ReplyErr(id, err);
      return ReplyOk(id, data);
     }
   if(action == "FLATTEN_ALL")
      return ReplyOk(id, FlattenAll(magic, dev, "python FLATTEN_ALL"));

   return ReplyErr(id, "unknown action " + action);
  }

//+------------------------------------------------------------------+
//| Lifecycle                                                        |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpAuthToken == "")
     {
      Print("QuantMind: InpAuthToken is empty - refusing to start (fail closed).");
      return INIT_PARAMETERS_INCORRECT;
     }
   if(InpDemoOnly && IsRealAccount())
     {
      Print("QuantMind: REAL (live-money) account detected and InpDemoOnly=true - refusing to start.");
      return INIT_FAILED;
     }
   g_epochMs = (long)TimeGMT() * 1000;
   g_epochUs = GetMicrosecondCount();

   FolderCreate("QuantMind");
   g_audit = FileOpen(InpAuditFile, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
   if(g_audit == INVALID_HANDLE)
      PrintFormat("QuantMind: cannot open audit file %s (err %d)", InpAuditFile, GetLastError());

   g_rep = new Socket(g_ctx, ZMQ_REP);
   g_pub = new Socket(g_ctx, ZMQ_PUB);
   g_rep.setLinger(0);
   g_pub.setLinger(0);
   if(!g_rep.bind(InpRepEndpoint) || !g_pub.bind(InpPubEndpoint))
     {
      PrintFormat("QuantMind: failed to bind %s / %s (port in use?)", InpRepEndpoint, InpPubEndpoint);
      return INIT_FAILED;
     }
   if(!EventSetMillisecondTimer(MathMax(1, InpTimerMs)))
      return INIT_FAILED;
   Audit("ea_start", JStr("rep", InpRepEndpoint) + "," + JStr("pub", InpPubEndpoint) + "," +
         JInt("login", AccountInfoInteger(ACCOUNT_LOGIN)) + "," + JStr("server", AccountInfoString(ACCOUNT_SERVER)) +
         "," + JBool("weekend_guard", InpWeekendGuard) + "," + JInt("dead_man_sec", InpDeadManSec));
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   Audit("ea_stop", JInt("reason", reason));
   if(g_rep != NULL) { g_rep.unbind(InpRepEndpoint); delete g_rep; g_rep = NULL; }
   if(g_pub != NULL) { g_pub.unbind(InpPubEndpoint); delete g_pub; g_pub = NULL; }
   if(g_audit != INVALID_HANDLE) { FileClose(g_audit); g_audit = INVALID_HANDLE; }
  }

void OnTimer()
  {
   // 1) Drain every pending command (REP: exactly one reply per request).
   for(int i = 0; i < MAX_MSGS_PER_TICK; i++)
     {
      ZmqMsg request;
      if(!g_rep.recv(request, true))
         break;
      string reply = HandleRequest(request.getData());
      ZmqMsg response(reply);
      g_rep.send(response);
     }

   ulong nowUs = GetMicrosecondCount();

   // 2) PUB heartbeat.
   if(nowUs - g_lastHeartbeatUs >= (ulong)InpHeartbeatPubMs * 1000)
     {
      g_lastHeartbeatUs = nowUs;
      Publish("{\"type\":\"heartbeat\"," + JInt("gmt_ms", NowMs()) + "," +
              JBool("weekend_blackout", InWeekendBlackout()) + "," + JInt("positions", PositionsTotal()) + "," +
              JNum("equity", AccountInfoDouble(ACCOUNT_EQUITY), 2) + "}");
     }

   // 3) Backup guards, rate-limited to one attempt per second.
   if(nowUs - g_lastGuardAttemptUs < 1000000 || PositionsTotal() + OrdersTotal() == 0)
      return;
   long scope = InpAccountWideFlatten ? -1 : InpMagic;
   if(InWeekendBlackout())
     {
      g_lastGuardAttemptUs = nowUs;
      FlattenAll(scope, (ulong)InpMaxDeviationPts, "EA weekend guard");
      return;
     }
   if(InpDeadManSec > 0 && g_lastPythonMsg > 0 && !g_deadManFired &&
      TimeGMT() - g_lastPythonMsg > InpDeadManSec)
     {
      g_lastGuardAttemptUs = nowUs;
      g_deadManFired = true;
      FlattenAll(scope, (ulong)InpMaxDeviationPts, "EA dead-man switch: Python silent");
     }
  }

void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   string evt = "{\"type\":\"deal\"," + JInt("deal", (long)trans.deal) + "," + JInt("order", (long)trans.order) + "," +
                JInt("position", (long)trans.position) + "," + JStr("symbol", trans.symbol) + "," +
                JNum("price", trans.price, 8) + "," + JNum("volume", trans.volume, 8) + "," +
                JInt("gmt_ms", NowMs()) + "}";
   Publish(evt);
   if(g_audit != INVALID_HANDLE)
     {
      FileSeek(g_audit, 0, SEEK_END);
      FileWriteString(g_audit, evt + "\r\n");
      FileFlush(g_audit);
     }
  }
//+------------------------------------------------------------------+
