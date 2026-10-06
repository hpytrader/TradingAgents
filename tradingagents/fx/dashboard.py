"""The FX desk dashboard: one self-contained HTML page of the journal.

A heads-up-display look on a dark surface: the scorecard, the cumulative-R
curve, live orders, results by symbol and conviction, and every journal entry
with the agents' reasoning. It loads nothing from the network. While the
watcher runs it rewrites the page each cycle and the page reloads itself every
minute.

Colour carries meaning only alongside a label: wins and losses always show a
mark (✓ / ✕) and a signed R, never green or red alone. All text that came from
the agents is escaped.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from tradingagents.fx import journal as jr
from tradingagents.fx.instruments import spec_for
from tradingagents.fx.journal import Entry, Stats
from tradingagents.fx.smc import NEW_YORK
from tradingagents.fx.smc_scanner import ScanWindow, market_open
from tradingagents.fx.team import Profile, team

STATUS = {
    jr.PENDING: ("◌", "Pending", "pending"),
    jr.OPEN: ("⏵", "Open", "open"),
    jr.WON: ("✓", "Won", "good"),
    jr.LOST: ("✕", "Lost", "bad"),
    jr.CLOSED: ("■", "Closed at NY close", "neutral"),
    jr.EXPIRED: ("○", "Expired", "muted"),
    jr.MISSED: ("○", "Missed", "muted"),
    jr.CANCELLED: ("○", "Cancelled", "muted"),
}

CSS = """
:root{--bg:#05090d;--panel:#0b1117;--panel-2:#0f1820;--line:#16323d;--line-2:#1f4552;
--accent:#5ce1f0;--accent-dim:#2aa9b8;--series:#19a7b8;--good:#0ca30c;--bad:#d03b3b;
--text:#e6f1f5;--text-2:#9db6c1;--muted:#5f7782;color-scheme:dark}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
background-image:radial-gradient(ellipse at 50% -10%,rgba(92,225,240,.10),transparent 60%),
linear-gradient(rgba(92,225,240,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(92,225,240,.035) 1px,transparent 1px);
background-size:100% 100%,32px 32px,32px 32px;min-height:100vh}
body:after{content:"";position:fixed;inset:0;pointer-events:none;
background:repeating-linear-gradient(0deg,rgba(255,255,255,.012) 0 1px,transparent 1px 3px)}
.mono,.num,td.n,th.n{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1280px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;justify-content:space-between;
padding:14px 18px;border:1px solid var(--line);background:linear-gradient(90deg,rgba(92,225,240,.08),transparent 40%),var(--panel);
clip-path:polygon(0 0,calc(100% - 18px) 0,100% 18px,100% 100%,18px 100%,0 calc(100% - 18px))}
.brand{display:flex;align-items:center;gap:14px}
.brand h1{font-size:15px;letter-spacing:.32em;margin:0;font-weight:600;text-transform:uppercase}
.brand h1 b{color:var(--accent);font-weight:600}
.brand small{display:block;color:var(--muted);letter-spacing:.2em;font-size:11px;text-transform:uppercase}
.core{flex:none;width:34px;height:34px;border-radius:50%;border:1px solid var(--accent-dim);position:relative;
box-shadow:0 0 18px rgba(92,225,240,.35),inset 0 0 10px rgba(92,225,240,.35)}
.core:after{content:"";position:absolute;inset:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent)}
.live .core:after{animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35;transform:scale(.8)}}
.status{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:12px;color:var(--text-2);letter-spacing:.06em;text-transform:uppercase}
.status span b{color:var(--text);font-weight:500}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:1px;background:var(--muted)}
.dot.on{background:var(--accent);box-shadow:0 0 8px var(--accent)}
.grid{display:grid;gap:14px;margin-top:14px}
.top{grid-template-columns:260px 1fr}
.mid{grid-template-columns:1.7fr 1fr}
.low{grid-template-columns:1fr 1fr}
@media (max-width:900px){.top,.mid,.low{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);position:relative;padding:16px 18px;min-width:0}
.panel:before,.panel:after{content:"";position:absolute;width:12px;height:12px;border-color:var(--accent-dim);border-style:solid}
.panel:before{top:-1px;left:-1px;border-width:1px 0 0 1px}.panel:after{bottom:-1px;right:-1px;border-width:0 1px 1px 0}
.panel h2{margin:0 0 12px;font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:var(--text-2);font-weight:500}
.panel h2 i{font-style:normal;color:var(--accent)}
.ring{display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring svg{display:block}
.ring .cap{color:var(--muted);font-size:12px;margin-top:4px;text-align:center}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
@media (max-width:600px){.tiles{grid-template-columns:repeat(2,1fr)}}
.tile{background:var(--panel-2);border:1px solid var(--line);padding:12px 14px}
.tile .k{font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-size:26px;margin-top:4px;font-weight:500}
.tile .s{font-size:12px;color:var(--text-2)}
.chart{width:100%;height:auto;display:block}
.axis{stroke:var(--line-2);stroke-width:1}.zero{stroke:var(--muted);stroke-dasharray:3 4}
.tick{fill:var(--muted);font:11px ui-monospace,Menlo,monospace}
.empty{color:var(--muted);padding:28px 4px;text-align:center}
.orders{display:flex;flex-direction:column;gap:10px}
.ord{border:1px solid var(--line);background:var(--panel-2);padding:10px 12px}
.ord .h{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.ord .h b{letter-spacing:.06em}
.ord .lv{color:var(--text-2);font-size:12px;margin-top:4px}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
padding:2px 8px;border:1px solid var(--line-2);color:var(--text-2);white-space:nowrap}
.chip.good{border-color:rgba(12,163,12,.6);color:var(--text)}.chip.good i{color:var(--good)}
.chip.bad{border-color:rgba(208,59,59,.6);color:var(--text)}.chip.bad i{color:var(--bad)}
.chip.open,.chip.pending{border-color:var(--accent-dim);color:var(--text)}.chip.open i,.chip.pending i{color:var(--accent)}
.chip i{font-style:normal}
.bars{display:grid;grid-template-columns:72px 1fr auto;gap:9px 12px;align-items:center}
.bar{display:contents;font-size:13px}
.bar .track{position:relative;height:14px;border-left:0}
.bar .track:before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;border-left:1px dashed var(--muted)}
.bar .fill{position:absolute;top:2px;height:10px;border-radius:2px}
.bar .fill.pos{left:50%;background:var(--good);border-radius:0 3px 3px 0}
.bar .fill.neg{right:50%;background:var(--bad);border-radius:3px 0 0 3px}
.bar .lab{color:var(--text-2);text-align:right;white-space:nowrap}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:500;text-align:left;
padding:8px 8px;border-bottom:1px solid var(--line-2);white-space:nowrap}
td{padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th.n,td.n{text-align:right}
tr:hover td{background:rgba(92,225,240,.035)}
.scroll{overflow-x:auto}
details summary{cursor:pointer;color:var(--accent-dim);font-size:12px;list-style:none}
details summary::-webkit-details-marker{display:none}
details p{margin:6px 0 0;color:var(--text-2);max-width:60ch;font-size:12.5px}
.foot{margin-top:18px;color:var(--muted);font-size:12px;line-height:1.6}
.chat{display:grid;grid-template-columns:240px 1fr;gap:14px;min-height:420px}
@media (max-width:900px){.chat{grid-template-columns:1fr}}
.sessions{display:flex;flex-direction:column;gap:6px;max-height:620px;overflow-y:auto}
.sessions button{all:unset;cursor:pointer;display:block;padding:9px 11px;border:1px solid var(--line);
background:var(--panel-2);font-size:12px;color:var(--text-2)}
.sessions button b{display:block;color:var(--text);font-family:ui-monospace,Menlo,monospace;font-weight:500;font-size:12.5px}
.sessions button[aria-pressed="true"]{border-color:var(--accent-dim);box-shadow:inset 3px 0 0 var(--accent)}
.sessions button:focus-visible{outline:1px solid var(--accent)}
.log{display:flex;flex-direction:column;gap:12px;max-height:620px;overflow-y:auto;padding-right:6px}
.msg{display:grid;grid-template-columns:40px 1fr;gap:10px}
.av{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font:600 12px ui-monospace,Menlo,monospace;
color:var(--text);background:#0d1a22;border:1.5px solid var(--ring,#2aa9b8);box-shadow:0 0 10px -2px var(--ring,#2aa9b8)}
.who{font-size:12px;color:var(--text-2);margin-bottom:4px}.who b{color:var(--text);font-weight:600;letter-spacing:.04em}
.who .t{display:inline-block;margin-left:8px;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.bubble{background:var(--panel-2);border:1px solid var(--line);padding:10px 12px;white-space:pre-wrap;font-size:13px;line-height:1.55;
color:var(--text);overflow-wrap:anywhere}
.msg.decision .bubble{border-color:var(--accent-dim);background:linear-gradient(90deg,rgba(92,225,240,.06),transparent 60%),var(--panel-2)}
.msg.system .bubble{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--text-2);background:#091016}
.team{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{border:1px solid var(--line);background:var(--panel-2);padding:14px}
.card .top{display:flex;gap:12px;align-items:center;margin-bottom:10px}
.card h3{margin:0;font-size:15px;letter-spacing:.06em}.card .r{font-size:12px;color:var(--text-2)}
.card p{margin:8px 0 0;font-size:12.5px;color:var(--text-2);line-height:1.55}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.tags span{font-size:10.5px;letter-spacing:.06em;padding:2px 7px;border:1px solid var(--line-2);color:var(--text-2)}
.tier{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
a.disc{color:var(--accent);font-size:12px;text-decoration:none;display:inline-block;margin-top:4px}
.tip{position:fixed;pointer-events:none;background:#0d1a22;border:1px solid var(--accent-dim);padding:6px 9px;
font:12px ui-monospace,Menlo,monospace;color:var(--text);display:none;z-index:5;box-shadow:0 0 16px rgba(92,225,240,.2)}
"""

JS = """
(function(){
  var clock=document.getElementById('clock');
  function tick(){var d=new Date();
    var ny=d.toLocaleTimeString('en-GB',{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit'});
    var utc=d.toISOString().substr(11,8);
    if(clock)clock.textContent=ny+' NY  ·  '+utc+' UTC';}
  tick();setInterval(tick,1000);
  var tip=document.getElementById('tip');
  document.querySelectorAll('[data-tip]').forEach(function(el){
    el.addEventListener('mousemove',function(e){tip.textContent=el.getAttribute('data-tip');tip.style.display='block';
      tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+12)+'px';});
    el.addEventListener('mouseleave',function(){tip.style.display='none';});
  });
  var raw=document.getElementById('chat-data');var log=document.getElementById('log');
  if(!raw||!log)return;
  var data=JSON.parse(raw.textContent);var byId={};
  data.reviews.forEach(function(r){byId[r.id]=r;});
  function text(parent,str){            // plain text, with **bold** kept: never parsed as HTML
    str.split('**').forEach(function(part,i){
      if(!part)return;var node=i%2?document.createElement('b'):document.createTextNode(part);
      if(i%2)node.textContent=part;parent.appendChild(node);});
  }
  function show(id){
    var r=byId[id];if(!r)return;log.textContent='';
    document.querySelectorAll('.sessions button').forEach(function(b){
      b.setAttribute('aria-pressed',b.getAttribute('data-review')===id?'true':'false');});
    r.messages.forEach(function(m){
      var row=document.createElement('div');row.className='msg '+(m.kind||'message');
      var av=document.createElement('div');av.className='av';av.textContent=(m.name||'?').slice(0,2).toUpperCase();
      av.style.setProperty('--ring',data.accents[m.agent]||'#2aa9b8');
      var body=document.createElement('div');var who=document.createElement('div');who.className='who';
      var b=document.createElement('b');b.textContent=m.name;who.appendChild(b);
      who.appendChild(document.createTextNode(' · '+(m.role||'')));
      var t=document.createElement('span');t.className='t';t.textContent=m.title||'';who.appendChild(t);
      var bubble=document.createElement('div');bubble.className='bubble';text(bubble,m.text||'');
      body.appendChild(who);body.appendChild(bubble);row.appendChild(av);row.appendChild(body);log.appendChild(row);
    });
    log.scrollTop=0;
  }
  document.querySelectorAll('.sessions button').forEach(function(b){
    b.addEventListener('click',function(){show(b.getAttribute('data-review'));});});
  document.querySelectorAll('a.disc').forEach(function(a){
    a.addEventListener('click',function(){show(a.getAttribute('data-review'));});});
  if(data.reviews.length)show(data.reviews[0].id);
})();
"""


def _ny(t: datetime | None, fmt: str = "%a %d %b %H:%M") -> str:
    return t.astimezone(NEW_YORK).strftime(fmt) if t else "—"


def _p(symbol: str, value: float | None) -> str:
    return "—" if value is None else f"{value:.{spec_for(symbol).decimals}f}"


def _r(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}R"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _chip(status: str) -> str:
    icon, label, cls = STATUS.get(status, ("·", status, "muted"))
    return f'<span class="chip {cls}"><i>{icon}</i>{escape(label)}</span>'


def _ring(s: Stats) -> str:
    rate = s.win_rate or 0.0
    circ = 2 * 3.14159 * 70
    arc = circ * rate
    ticks = "".join(
        f'<line x1="90" y1="8" x2="90" y2="{14 if i % 5 else 18}" stroke="#1f4552" stroke-width="1" '
        f'transform="rotate({i * 6} 90 90)"/>' for i in range(60))
    value = _pct(s.win_rate) if s.win_rate is not None else "—"
    return f"""<div class="panel ring"><h2><i>◆</i> Win rate</h2>
<svg width="180" height="180" viewBox="0 0 180 180" role="img" aria-label="Win rate {value}">
{ticks}<circle cx="90" cy="90" r="70" fill="none" stroke="#132a33" stroke-width="8"/>
<circle cx="90" cy="90" r="70" fill="none" stroke="#5ce1f0" stroke-width="8" stroke-linecap="round"
stroke-dasharray="{arc:.1f} {circ:.1f}" transform="rotate(-90 90 90)" style="filter:drop-shadow(0 0 6px rgba(92,225,240,.6))"/>
<circle cx="90" cy="90" r="54" fill="none" stroke="#16323d" stroke-dasharray="2 5"/>
<text x="90" y="92" text-anchor="middle" fill="#e6f1f5" font-size="32" font-family="ui-monospace,Menlo,monospace">{value}</text>
<text x="90" y="116" text-anchor="middle" fill="#5f7782" font-size="10" letter-spacing="3">OF {s.finished} FILLED</text>
</svg><div class="cap">{s.won} won · {s.lost} lost · {s.closed} closed at NY close</div></div>"""


def _tiles(s: Stats) -> str:
    pf = "—" if s.profit_factor is None else f"{s.profit_factor:.2f}"
    tiles = [
        ("Total result", _r(s.total_r) if s.finished else "—", f"{s.finished} filled trades"),
        ("Expectancy", _r(s.avg_r), "average R per filled trade"),
        ("Profit factor", pf, "gains ÷ losses"),
        ("Max drawdown", f"{-s.max_drawdown_r:.2f}R" if s.finished else "—", "peak to trough"),
        ("Fill rate", _pct(s.fill_rate), f"{s.expired} expired · {s.missed} missed"),
        ("Live", str(s.open + s.pending), f"{s.open} open · {s.pending} pending"),
    ]
    body = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v num">{escape(v)}</div>'
                   f'<div class="s">{escape(sub)}</div></div>' for k, v, sub in tiles)
    return f'<div class="panel"><h2><i>◆</i> Scorecard</h2><div class="tiles">{body}</div></div>'


def _curve(s: Stats, finished: list[Entry]) -> str:
    head = '<div class="panel"><h2><i>◆</i> Cumulative result <span style="color:var(--muted)">· R, by trade</span></h2>'
    if not s.curve:
        return head + '<div class="empty">The curve starts with the first filled trade.</div></div>'
    w, h, pl, pr, pt, pb = 760, 260, 44, 16, 14, 30
    ys = [0.0] + [r for _, r in s.curve]
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.12, 0.5)
    lo, hi = lo - pad, hi + pad
    n = len(ys) - 1

    def x(i):
        return pl + (w - pl - pr) * (i / max(n, 1))

    def y(v):
        return pt + (h - pt - pb) * (hi - v) / (hi - lo)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(ys))
    area = f"{x(0):.1f},{y(0):.1f} {pts} {x(n):.1f},{y(0):.1f}"
    step = max(round((hi - lo) / 4 * 2) / 2, 0.5)
    first = int(lo // step) * step
    grid = []
    v = first
    while v <= hi:
        if v >= lo:
            grid.append(f'<line class="axis" x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}" opacity=".5"/>'
                        f'<text class="tick" x="{pl - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:+g}</text>')
        v += step
    dots = []
    for i, ((_, cum), e) in enumerate(zip(s.curve, finished, strict=False), start=1):
        tip = f"#{i} {e.symbol} {_r(e.result_r)} · total {cum:+.2f}R · {_ny(e.exit_at)}"
        dots.append(f'<circle cx="{x(i):.1f}" cy="{y(cum):.1f}" r="4.5" fill="#19a7b8" stroke="#0b1117" '
                    f'stroke-width="2"/><circle cx="{x(i):.1f}" cy="{y(cum):.1f}" r="12" fill="transparent" '
                    f'data-tip="{escape(tip, quote=True)}"/>')
    xt = "".join(f'<text class="tick" x="{x(i):.1f}" y="{h - 10}" text-anchor="middle">{i}</text>'
                 for i in range(1, n + 1) if n <= 20 or i % max(n // 10, 1) == 0)
    svg = f"""<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Cumulative R by trade, now {s.total_r:+.2f}R">
<defs><linearGradient id="fade" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#19a7b8" stop-opacity=".28"/>
<stop offset="1" stop-color="#19a7b8" stop-opacity="0"/></linearGradient></defs>
{''.join(grid)}<line class="zero" x1="{pl}" x2="{w - pr}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>
<polygon points="{area}" fill="url(#fade)"/>
<polyline points="{pts}" fill="none" stroke="#19a7b8" stroke-width="2" stroke-linejoin="round"
style="filter:drop-shadow(0 0 4px rgba(25,167,184,.7))"/>{''.join(dots)}{xt}</svg>"""
    return head + svg + "</div>"


def _live(entries: list[Entry]) -> str:
    live = [e for e in entries if e.status in jr.ACTIVE]
    head = '<div class="panel"><h2><i>◆</i> Live orders</h2>'
    if not live:
        return head + '<div class="empty">No pending or open orders.</div></div>'
    rows = []
    for e in sorted(live, key=lambda e: e.created_at, reverse=True):
        when = (f"filled {_ny(e.filled_at, '%H:%M')}" if e.status == jr.OPEN
                else f"cancel {_ny(e.expires_at, '%H:%M')} NY")
        rows.append(f"""<div class="ord"><div class="h"><b>{'▲' if e.long else '▼'} <span class="mono" style="color:var(--accent)">{escape(e.ticket)}</span> {escape(e.symbol)}</b>{_chip(e.status)}</div>
<div class="lv mono">{e.order_type} {_p(e.symbol, e.entry)} · SL {_p(e.symbol, e.stop)} · TP {_p(e.symbol, e.target)}</div>
<div class="lv">{(e.planned_rr or 0):.2f}R planned · {escape(e.conviction)} conviction · {escape(when)}</div></div>""")
    return head + f'<div class="orders">{"".join(rows)}</div></div>'


def _breakdown(title: str, groups: dict[str, dict]) -> str:
    head = f'<div class="panel"><h2><i>◆</i> {escape(title)}</h2>'
    if not groups:
        return head + '<div class="empty">Appears once trades finish.</div></div>'
    scale = max(abs(g["total_r"]) for g in groups.values()) or 1
    rows = []
    for name, g in groups.items():
        width = 50 * abs(g["total_r"]) / scale
        cls = "pos" if g["total_r"] >= 0 else "neg"
        mark = "✓" if g["total_r"] > 0 else "✕" if g["total_r"] < 0 else "·"
        rows.append(f'<div class="bar"><span class="mono">{escape(name)}</span>'
                    f'<span class="track"><span class="fill {cls}" style="width:{width:.1f}%"></span></span>'
                    f'<span class="lab mono">{mark} {g["total_r"]:+.2f}R · {g["n"]} trade{"s" if g["n"] != 1 else ""} · '
                    f'{g["win_rate"]:.0%} won</span></div>')
    return head + f'<div class="bars">{"".join(rows)}</div></div>'


def _table(entries: list[Entry], linked: set[str] | None = None) -> str:
    linked = linked or set()
    head = '<div class="panel" style="margin-top:14px"><h2><i>◆</i> Journal <span style="color:var(--muted)">· every final order</span></h2>'
    if not entries:
        return head + ('<div class="empty">The journal is empty. It fills as the agents give final orders '
                       '(<span class="mono">fx-scan --agents</span> or <span class="mono">fx-watch</span>).</div></div>')
    rows = []
    for e in sorted(entries, key=lambda e: e.created_at, reverse=True):
        why = ""
        if e.rationale or e.watch_for or e.note or e.changes:
            changes = "".join(
                f"<p><b>{escape(c.get('by', ''))}, {_ny(datetime.fromisoformat(c['at']), '%H:%M')}:</b> "
                f"{escape(c['action'].replace('_', ' '))} "
                f"{escape(str(c.get('old', '')))}{' → ' if 'new' in c else ''}{escape(str(c.get('new', c.get('price', ''))))}"
                f" · {escape(c.get('reason', ''))}</p>" for c in e.changes)
            parts = [f"<p>{escape(e.rationale)}</p>" if e.rationale else "",
                     f"<p><b>Watch for:</b> {escape(e.watch_for)}</p>" if e.watch_for else "",
                     changes,
                     f"<p><b>Outcome:</b> {escape(e.note)}</p>" if e.note else ""]
            why = f'<details><summary>reasoning ▸</summary>{"".join(parts)}</details>'
        if e.review_id in linked:
            why += (f'<a class="disc" href="#chat" data-review="{escape(e.review_id, quote=True)}">'
                    'desk chat ▸</a>')
        rows.append(f"""<tr><td class="mono" style="color:var(--accent)">{escape(e.ticket)}</td><td class="mono">{_ny(e.created_at)}</td><td><b>{escape(e.symbol)}</b><br>
<span style="color:var(--text-2);font-size:12px">{e.order_type}</span></td>
<td class="n">{_p(e.symbol, e.entry)}</td><td class="n">{_p(e.symbol, e.stop)}</td><td class="n">{_p(e.symbol, e.target)}</td>
<td class="n">{(e.planned_rr or 0):.2f}</td><td>{escape(e.conviction)}</td><td>{_chip(e.status)}</td>
<td class="n">{_r(e.result_r)}</td><td>{why}</td></tr>""")
    return head + f"""<div class="scroll"><table><thead><tr><th>Ticket</th><th>Suggested (NY)</th><th>Order</th><th class="n">Entry</th>
<th class="n">Stop</th><th class="n">Target</th><th class="n">Plan R</th><th>Conviction</th><th>Status</th>
<th class="n">Result</th><th>Why</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>"""


def _chat(reviews: list[dict]) -> str:
    head = ('<div class="panel" id="chat" style="margin-top:14px"><h2><i>◆</i> Desk chat '
            '<span style="color:var(--muted)">· how each decision was argued</span></h2>')
    if not reviews:
        return head + ('<div class="empty">No conversations yet. Each agent review is saved here, '
                       'from the scan to the final check.</div></div>')
    buttons = []
    for i, r in enumerate(reviews):
        at = datetime.fromisoformat(r["at"])
        label = f"{r['orders']} order{'s' if r['orders'] != 1 else ''}" if r["orders"] else "no orders"
        buttons.append(f'<button type="button" data-review="{escape(r["id"], quote=True)}" '
                       f'aria-pressed="{"true" if i == 0 else "false"}"><b>{_ny(at)}</b>{escape(label)}</button>')
    return head + (f'<div class="chat"><div class="sessions" role="list">{"".join(buttons)}</div>'
                   f'<div class="log" id="log" aria-live="polite"></div></div></div>')


def _team(people: dict[str, Profile]) -> str:
    tiers = {"deep": "Strong model", "quick": "Fast model", "code": "Rule engine"}
    cards = []
    for p in people.values():
        initials = escape(p.name[:2].upper())
        tags = "".join(f"<span>{escape(t)}</span>" for t in p.expertise)
        cards.append(f"""<div class="card"><div class="top"><div class="av" style="--ring:{escape(p.accent, quote=True)}">{initials}</div>
<div><h3>{escape(p.name)}</h3><div class="r">{escape(p.role)}</div><div class="tier">{tiers.get(p.tier, p.tier)}</div></div></div>
<div class="tags">{tags}</div><p>{escape(p.bio)}</p></div>""")
    return ('<div class="panel" style="margin-top:14px"><h2><i>◆</i> The desk '
            f'<span style="color:var(--muted)">· who decides</span></h2><div class="team">{"".join(cards)}</div></div>')


def _json_script(element_id: str, data) -> str:
    """Embed ``data`` for the page script; ``</`` is escaped so text cannot close the tag."""
    text = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{text}</script>'


def render(entries: list[Entry], s: Stats, *, now: datetime | None = None, live: bool = False,
           window: ScanWindow | None = None, last_cycle: str = "",
           reviews: list[dict] | None = None) -> str:
    now = now or datetime.now(UTC)
    window = window or ScanWindow()
    reviews = reviews or []
    people = team()
    accents = {p.key: p.accent for p in people.values()}
    closes = window.current_end(now)
    if closes:
        win = f'<span><i class="dot on"></i>Window <b>open</b> · closes {_ny(closes, "%H:%M")} NY</span>'
    else:
        opens = window.next_open(now)
        reason = "market closed" if not market_open(now) else "window closed"
        win = f'<span><i class="dot"></i>{reason} · opens <b>{_ny(opens, "%a %H:%M")}</b> NY</span>'
    mode = ('<span><i class="dot on"></i>Watcher <b>live</b></span>' if live
            else '<span><i class="dot"></i>Snapshot</span>')
    finished = sorted((e for e in entries if e.status in jr.FINISHED and e.result_r is not None),
                      key=lambda e: e.exit_at or e.created_at)
    refresh = '<meta http-equiv="refresh" content="60">' if live else ""
    cycle = f"<span>Last cycle <b>{escape(last_cycle)}</b></span>" if last_cycle else ""
    data = json.dumps({"generated": now.isoformat(), "total_r": s.total_r, "trades": s.finished})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>FX Desk</title><style>{CSS}</style></head>
<body class="{'live' if live else ''}"><div class="wrap">
<header><div class="brand"><div class="core"></div><div><h1>Trading<b>Agents</b> // FX Desk</h1>
<small>Agent-reviewed limit orders · paper journal</small></div></div>
<div class="status">{mode}{win}{cycle}<span class="mono" id="clock"></span></div></header>
<div class="grid top">{_ring(s)}{_tiles(s)}</div>
<div class="grid mid">{_curve(s, finished)}{_live(entries)}</div>
<div class="grid low">{_breakdown("By symbol", s.by_symbol)}{_breakdown("By conviction", s.by_conviction)}</div>
{_table(entries, {r["id"] for r in reviews})}
{_chat(reviews)}
{_team(people)}
<div class="foot">Paper results: each order is replayed on OANDA one-minute mid prices after it was suggested. A loss is −1R;
a win or a close at the New York 17:00 close is charged the spread recorded at the scan. A minute that touches the entry
and target together counts as no fill, and one that touches the stop and target as the stop. Real fills differ.
Generated {_ny(now)} New York.</div></div>
<div class="tip" id="tip"></div><script type="application/json" id="summary">{escape(data)}</script>
{_json_script("chat-data", {"reviews": reviews, "accents": accents})}
<script>{JS}</script></body></html>"""


def write(path: str | Path, entries: list[Entry], s: Stats, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(render(entries, s, **kwargs), encoding="utf-8")
    tmp.replace(path)
    return path
