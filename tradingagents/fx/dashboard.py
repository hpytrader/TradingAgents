"""Happy People: the FX desk dashboard, one self-contained HTML page of the journal.

A neon trading console: the latest review stage by stage (scan to book), the
trading floor with every agent at a desk saying what they said in that review
(and a replay that plays it back in speaking order), the scorecard, the
cumulative-R curve, live orders, results by symbol and conviction, and every
journal entry with the agents' reasoning. It loads nothing from the network. While the
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
:root{--void:#0c0718;--deck:#150d27;--deck-2:#1c1233;--wire:#2f1d52;--wire-2:#46297a;
--neon:#ff3fd8;--neon-dim:#b02d97;--ion:#43e8ff;--ion-dim:#1f9fb4;--sun:#ffb547;--win:#3ee08f;--loss:#ff5470;
--ink:#f3ecff;--ink-2:#b3a6d6;--mute:#7a6b9e;color-scheme:dark;
--face:Bahnschrift,"DIN Alternate","DIN Condensed","Arial Narrow",system-ui,sans-serif;
--body:system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--void);color:var(--ink);font:14px/1.5 var(--body);min-height:100vh;
background-image:radial-gradient(ellipse 80% 50% at 50% -5%,rgba(255,63,216,.16),transparent 70%),
radial-gradient(ellipse 60% 40% at 50% 105%,rgba(67,232,255,.10),transparent 70%),
linear-gradient(rgba(179,166,214,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(179,166,214,.04) 1px,transparent 1px);
background-size:100% 100%,100% 100%,28px 28px,28px 28px}
.num,.mono,td.n,th.n{font-family:var(--face);font-variant-numeric:tabular-nums;letter-spacing:.02em}
.wrap{max-width:1440px;margin:0 auto;padding:18px 16px 40px}
.console{border:1.5px solid var(--neon);border-radius:14px;padding:14px;position:relative;
box-shadow:0 0 0 1px rgba(255,63,216,.25),0 0 28px rgba(255,63,216,.35),inset 0 0 40px rgba(255,63,216,.08);
background:linear-gradient(180deg,rgba(28,18,51,.55),rgba(12,7,24,.7))}
header{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:center;justify-content:space-between;padding:4px 6px 12px;
border-bottom:1px solid var(--wire)}
.brand{display:flex;align-items:center;gap:14px}
.brand h1{font:600 26px/1 var(--face);letter-spacing:.04em;margin:0;color:var(--ink);
text-shadow:0 0 18px rgba(255,63,216,.55)}
.brand small{display:block;color:var(--ink-2);font-size:12.5px;margin-top:4px}
.mark{flex:none;width:40px;height:40px}
.live .mark .pulse{animation:pulse 2.4s ease-in-out infinite;transform-origin:20px 20px}
@keyframes pulse{50%{opacity:.35;transform:scale(.82)}}
.status{display:flex;flex-wrap:wrap;gap:6px 18px;font:13px var(--face);color:var(--ink-2);letter-spacing:.03em}
.status span b{color:var(--ink);font-weight:600}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:1px;background:var(--mute)}
.dot.on{background:var(--win);box-shadow:0 0 8px var(--win)}
/* pipeline */
.pipe{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;margin:12px 0}
@media (max-width:1000px){.pipe{grid-template-columns:repeat(4,minmax(0,1fr))}.stage:after{display:none}}
@media (max-width:560px){.pipe{grid-template-columns:repeat(2,minmax(0,1fr))}}
.stage{position:relative;background:var(--deck);border:1px solid var(--wire-2);border-radius:8px;padding:9px 11px 10px;min-width:0}
.stage:after{content:"";position:absolute;right:-7px;top:50%;width:6px;height:1px;background:var(--neon-dim)}
.stage:last-child:after{display:none}
.stage .k{font:600 12px var(--face);letter-spacing:.14em;color:var(--neon);display:flex;align-items:center;gap:7px}
.stage .k i{width:6px;height:6px;border-radius:50%;background:var(--wire-2)}
.stage.done .k i{background:var(--neon);box-shadow:0 0 8px var(--neon)}
.stage .v{font:600 20px/1.2 var(--face);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stage .s{font-size:12px;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* main grid */
.grid{display:grid;gap:12px;margin-top:12px}
.main{grid-template-columns:280px minmax(0,1fr) 300px;align-items:start}
.pair{grid-template-columns:1.6fr 1fr}
@media (max-width:1180px){.main{grid-template-columns:1fr 1fr}.main .floorwrap{grid-column:1/-1;order:-1}}
@media (max-width:760px){.main,.pair{grid-template-columns:1fr}}
.panel{background:var(--deck);border:1px solid var(--wire);border-radius:10px;padding:14px 16px;min-width:0}
.panel h2{margin:0 0 10px;font:600 15px var(--face);letter-spacing:.04em;color:var(--ink)}
.panel h2 span{color:var(--mute);font-weight:400}
.stack{display:flex;flex-direction:column;gap:12px}
.ring{display:flex;flex-direction:column;align-items:center}
.ring svg{display:block}.ring .cap{color:var(--ink-2);font-size:12px;margin-top:2px;text-align:center}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.tile{background:var(--deck-2);border:1px solid var(--wire);border-radius:7px;padding:9px 11px}
.tile .k{font-size:12px;color:var(--ink-2)}
.tile .v{font:600 22px/1.25 var(--face)}
.tile .s{font-size:11.5px;color:var(--mute)}
/* the trading floor */
.floorwrap{padding:12px 14px 14px;background:radial-gradient(ellipse at 50% 40%,rgba(255,63,216,.08),transparent 70%),var(--deck)}
.floorhead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px}
.floorhead h2{margin:0}
.floorhead .when{color:var(--ink-2);font:13px var(--face)}
.replay{all:unset;cursor:pointer;font:600 13px var(--face);letter-spacing:.06em;color:var(--void);background:var(--neon);
padding:6px 14px;border-radius:99px;box-shadow:0 0 14px rgba(255,63,216,.6)}
.replay:focus-visible{outline:2px solid var(--ion);outline-offset:2px}
.replay[aria-pressed="true"]{background:var(--ion);box-shadow:0 0 14px rgba(67,232,255,.6)}
.ticker{font:13px var(--face);color:var(--ion);background:#0a0616;border:1px solid var(--wire);border-radius:6px;
padding:7px 10px;margin-bottom:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ticker b{color:var(--ink-2);font-weight:400;margin-right:8px}
.floor{display:grid;gap:10px;grid-template-columns:repeat(4,minmax(0,1fr));
grid-template-areas:"macro price_action bull bear" "research_manager portfolio_manager portfolio_manager trader"
"risk_aggressive risk_neutral risk_conservative trade_manager"}
@media (max-width:760px){.floor{grid-template-columns:1fr 1fr;grid-template-areas:"portfolio_manager portfolio_manager"
"macro price_action" "bull bear" "research_manager trader" "risk_aggressive risk_conservative" "risk_neutral trade_manager"}}
.desk{--a:#b3a6d6;position:relative;background:linear-gradient(180deg,#1a1030,#120b22);border:1px solid var(--wire);
border-radius:9px;padding:9px 10px 10px;min-width:0;cursor:pointer;transition:border-color .25s,box-shadow .25s,opacity .25s}
.desk:focus-visible{outline:2px solid var(--ion);outline-offset:2px}
.desk.quiet{opacity:.55}
.desk.talking{border-color:var(--a);box-shadow:0 0 0 1px var(--a),0 0 22px -4px var(--a);opacity:1}
.desk .rig{display:flex;align-items:flex-end;gap:9px;padding-right:12px}
.desk svg.screen{flex:none;width:50px;height:35px}
.desk .who{min-width:0}
.desk .who b{display:block;font:600 15px/1.1 var(--face);letter-spacing:.03em}
.desk .role{font-size:11.5px;color:var(--ink-2);margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.desk .seq{position:absolute;top:7px;right:9px;font:12px var(--face);color:var(--mute)}
.desk .say{margin-top:8px;background:rgba(12,7,24,.75);border:1px solid var(--wire);border-left:2px solid var(--a);
border-radius:6px;padding:6px 8px}
.desk .say p{margin:0;font-size:12.5px;line-height:1.45;color:var(--ink);min-height:2.9em;
display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.desk .say p:empty:before{content:"No word in this review";color:var(--mute)}
.floor.playing .say p:empty:before{content:"Waiting to speak"}
.desk.boss{grid-area:portfolio_manager;background:linear-gradient(180deg,#241240,#150a29);border-color:var(--neon-dim)}
.desk.boss .who b{font-size:19px}
.desk.boss .say p{-webkit-line-clamp:5}
.avatar{flex:none;width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font:600 12px var(--face);
background:#0c0718;border:1.5px solid var(--a);box-shadow:0 0 10px -2px var(--a);margin-left:-18px;margin-bottom:-4px}
.detail{margin-top:10px;background:#0a0616;border:1px solid var(--wire);border-radius:8px;padding:10px 12px;font-size:13px;
white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow-y:auto;display:none}
.detail.on{display:block}
.detail .h{font:600 14px var(--face);color:var(--ink);margin-bottom:4px}
/* chart, orders, bars */
.chart{width:100%;height:auto;display:block}
.axis{stroke:var(--wire-2);stroke-width:1}.zero{stroke:var(--mute);stroke-dasharray:3 4}
.tick{fill:var(--mute);font:12px var(--face)}
.empty{color:var(--ink-2);padding:22px 4px;text-align:center}
.orders{display:flex;flex-direction:column;gap:8px}
.ord{border:1px solid var(--wire);background:var(--deck-2);border-radius:7px;padding:9px 11px}
.ord .h{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.ord .h b{font:600 15px var(--face);letter-spacing:.03em}
.ord .lv{color:var(--ink-2);font-size:12px;margin-top:3px}
.tkt{color:var(--ion)}
.chip{display:inline-flex;align-items:center;gap:5px;font:12px var(--face);letter-spacing:.03em;
padding:1px 8px;border:1px solid var(--wire-2);border-radius:99px;color:var(--ink-2);white-space:nowrap}
.chip i{font-style:normal}
.chip.good{border-color:rgba(62,224,143,.6);color:var(--ink)}.chip.good i{color:var(--win)}
.chip.bad{border-color:rgba(255,84,112,.6);color:var(--ink)}.chip.bad i{color:var(--loss)}
.chip.open{border-color:var(--ion-dim);color:var(--ink)}.chip.open i{color:var(--ion)}
.chip.pending{border-color:rgba(255,181,71,.55);color:var(--ink)}.chip.pending i{color:var(--sun)}
.bars{display:grid;grid-template-columns:62px 1fr;gap:4px 10px;align-items:center}
.bar .lab{grid-column:1/-1;margin-bottom:6px}
.bar{display:contents;font-size:13px}
.bar .track{position:relative;height:14px}
.bar .track:before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;border-left:1px dashed var(--mute)}
.bar .fill{position:absolute;top:2px;height:10px}
.bar .fill.pos{left:50%;background:var(--win);border-radius:0 3px 3px 0}
.bar .fill.neg{right:50%;background:var(--loss);border-radius:3px 0 0 3px}
.bar .lab{color:var(--ink-2);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font:500 12.5px var(--face);letter-spacing:.03em;color:var(--mute);text-align:left;padding:8px;border-bottom:1px solid var(--wire-2);white-space:nowrap}
td{padding:9px 8px;border-bottom:1px solid var(--wire);vertical-align:top}
th.n,td.n{text-align:right}
tr:hover td{background:rgba(255,63,216,.04)}
.scroll{overflow-x:auto}
details summary{cursor:pointer;color:var(--ion);font-size:12px;list-style:none}
details summary::-webkit-details-marker{display:none}
details p{margin:6px 0 0;color:var(--ink-2);max-width:62ch;font-size:12.5px}
.foot{margin-top:16px;color:var(--mute);font-size:12px;line-height:1.6;max-width:110ch}
.chat{display:grid;grid-template-columns:230px 1fr;gap:12px;min-height:380px}
@media (max-width:900px){.chat{grid-template-columns:1fr}}
.sessions{display:flex;flex-direction:column;gap:6px;max-height:600px;overflow-y:auto}
.sessions button{all:unset;cursor:pointer;display:block;padding:8px 11px;border:1px solid var(--wire);border-radius:7px;
background:var(--deck-2);font-size:12px;color:var(--ink-2)}
.sessions button b{display:block;color:var(--ink);font:600 13.5px var(--face)}
.sessions button[aria-pressed="true"]{border-color:var(--neon);box-shadow:inset 3px 0 0 var(--neon)}
.sessions button:focus-visible{outline:2px solid var(--ion)}
.log{display:flex;flex-direction:column;gap:12px;max-height:600px;overflow-y:auto;padding-right:6px}
.msg{display:grid;grid-template-columns:40px 1fr;gap:10px}
.av{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font:600 13px var(--face);
color:var(--ink);background:#0c0718;border:1.5px solid var(--ring,#b3a6d6);box-shadow:0 0 10px -2px var(--ring,#b3a6d6)}
.msg .who{font-size:12px;color:var(--ink-2);margin-bottom:4px}.msg .who b{color:var(--ink);font:600 14px var(--face)}
.msg .who .t{margin-left:8px;color:var(--mute)}
.bubble{background:var(--deck-2);border:1px solid var(--wire);border-radius:8px;padding:9px 12px;white-space:pre-wrap;font-size:13px;
line-height:1.55;overflow-wrap:anywhere}
.msg.decision .bubble{border-color:var(--neon-dim);background:linear-gradient(90deg,rgba(255,63,216,.07),transparent 60%),var(--deck-2)}
.msg.system .bubble{font-size:12.5px;color:var(--ink-2);background:#0a0616}
.team{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}
.card{border:1px solid var(--wire);border-radius:8px;background:var(--deck-2);padding:12px 14px}
.card .top{display:flex;gap:12px;align-items:center;margin-bottom:8px}
.card h3{margin:0;font:600 16px var(--face);letter-spacing:.03em}.card .r{font-size:12px;color:var(--ink-2)}
.card p{margin:8px 0 0;font-size:12.5px;color:var(--ink-2);line-height:1.55}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.tags span{font-size:11px;padding:1px 8px;border:1px solid var(--wire-2);border-radius:99px;color:var(--ink-2)}
.tier{font-size:11.5px;color:var(--neon)}
a.disc{color:var(--ion);font-size:12px;text-decoration:none;display:inline-block;margin-top:4px}
.tip{position:fixed;pointer-events:none;background:#0c0718;border:1px solid var(--neon-dim);border-radius:6px;padding:6px 9px;
font:12.5px var(--face);color:var(--ink);display:none;z-index:5;box-shadow:0 0 16px rgba(255,63,216,.25)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = """
(function(){
  var clock=document.getElementById('clock');
  function tick(){var d=new Date();
    var ny=d.toLocaleTimeString('en-GB',{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit'});
    if(clock)clock.textContent=ny+' New York';}
  tick();setInterval(tick,1000);
  var tip=document.getElementById('tip');
  document.querySelectorAll('[data-tip]').forEach(function(el){
    el.addEventListener('mousemove',function(e){tip.textContent=el.getAttribute('data-tip');tip.style.display='block';
      tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+12)+'px';});
    el.addEventListener('mouseleave',function(){tip.style.display='none';});
  });
  var raw=document.getElementById('chat-data');if(!raw)return;
  var data=JSON.parse(raw.textContent);var byId={};
  data.reviews.forEach(function(r){byId[r.id]=r;});
  var log=document.getElementById('log');
  function text(parent,str){            // plain text, with **bold** kept: never parsed as HTML
    str.split('**').forEach(function(part,i){
      if(!part)return;var node=i%2?document.createElement('b'):document.createTextNode(part);
      if(i%2)node.textContent=part;parent.appendChild(node);});
  }
  function plain(str){return (str||'').replace(/\\*\\*/g,'').replace(/^\\s*[-\\u2022]\\s+/gm,'').replace(/^#{1,6}\\s+/gm,'').replace(/\\s+/g,' ').trim();}
  function fmt(iso){try{return new Date(iso).toLocaleString('en-GB',{timeZone:'America/New_York',weekday:'short',
    hour:'2-digit',minute:'2-digit'})+' New York';}catch(e){return iso;}}

  // --- the pipeline strip -------------------------------------------------
  function stage(key,value,sub,done){var el=document.querySelector('.stage[data-stage="'+key+'"]');if(!el)return;
    el.querySelector('.v').textContent=value;el.querySelector('.s').textContent=sub;el.classList.toggle('done',!!done);}
  function pipeline(r){
    var said={};(r.messages||[]).forEach(function(m){(said[m.agent]=said[m.agent]||[]).push(m.text||'');});
    var scan=(said.desk||[''])[0];
    var n=(scan.match(/(\\d+) setup/)||[])[1];var at=(scan.match(/at (\\d\\d:\\d\\d) UTC/)||[])[1];
    var ev=(scan.match(/Calendar: (\\d+)/)||[])[1];
    var ward=!said.macro&&said.trade_manager;
    if(ward){stage('scan','Ward check','live trades only',true);stage('macro','—','',false);stage('debate','—','',false);
      stage('verdict','—','',false);stage('risk','—','',false);stage('book','No new orders','trade manager only',true);return;}
    stage('scan',n?n+' setup'+(n==='1'?'':'s'):'—',at?'at '+at+' UTC':'',!!n);
    stage('macro',ev?ev+' event'+(ev==='1'?'':'s'):'No events',said.macro?'before the close':'',!!said.macro);
    stage('debate',said.bull&&said.bear?'Leo vs Ursa':'—',said.bear?'bull and bear cases':'',!!(said.bull&&said.bear));
    var v=(said.research_manager||[''])[0];var keep=(v.match(/: KEEP/g)||[]).length,drop=(v.match(/: DROP/g)||[]).length;
    stage('verdict',said.research_manager?keep+' kept':'—',said.research_manager?drop+' dropped':'',!!said.research_manager);
    var risk=['risk_aggressive','risk_conservative','risk_neutral'].filter(function(k){return said[k];}).length;
    stage('risk',risk?risk+' views':'—',risk?'press, cut, weigh':'',risk>0);
    stage('book',r.orders?r.orders+' order'+(r.orders===1?'':'s'):'No orders',said.portfolio_manager?'Donna\\u2019s final book':'',
      !!said.portfolio_manager);
  }

  // --- the trading floor --------------------------------------------------
  var desks={};document.querySelectorAll('.desk').forEach(function(d){desks[d.getAttribute('data-agent')]=d;});
  var detail=document.getElementById('detail'),ticker=document.getElementById('ticker'),when=document.getElementById('floor-when');
  var current=null,timer=null,replayBtn=document.getElementById('replay');
  function speak(m,i){var d=desks[m.agent];if(!d)return;
    d.querySelector('.say p').textContent=plain(m.text).slice(0,320);
    d.querySelector('.seq').textContent=String(i+1);d._msg=m;}
  function floor(r){current=r;stop();
    Object.keys(desks).forEach(function(k){var d=desks[k];d.querySelector('.say p').textContent='';
      d.querySelector('.seq').textContent='';d._msg=null;d.classList.remove('talking');d.classList.add('quiet');});
    var order=0;(r.messages||[]).forEach(function(m){if(desks[m.agent]){speak(m,order++);desks[m.agent].classList.remove('quiet');}});
    var scan=(r.messages||[]).filter(function(m){return m.agent==='desk';});
    ticker.textContent='';var b=document.createElement('b');b.textContent='Desk system';ticker.appendChild(b);
    ticker.appendChild(document.createTextNode(scan.length?plain(scan[scan.length-1].text).slice(0,400):'No scan in this session.'));
    when.textContent=fmt(r.at);detail.classList.remove('on');pipeline(r);
    var last=(r.messages||[]).filter(function(m){return desks[m.agent];}).pop();
    if(last)desks[last.agent].classList.add('talking');
  }
  function open(d){if(!d._msg)return;detail.textContent='';var h=document.createElement('div');h.className='h';
    h.textContent=d._msg.name+' \\u2014 '+(d._msg.title||d._msg.role||'');detail.appendChild(h);
    var body=document.createElement('div');text(body,d._msg.text||'');detail.appendChild(body);detail.classList.add('on');
    Object.keys(desks).forEach(function(k){desks[k].classList.toggle('talking',desks[k]===d);});}
  Object.keys(desks).forEach(function(k){var d=desks[k];d.addEventListener('click',function(){stop();open(d);});
    d.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();stop();open(d);}});});
  var floorEl=document.querySelector('.floor');
  function stop(){if(floorEl)floorEl.classList.remove('playing');if(timer){clearInterval(timer);timer=null;}if(replayBtn){replayBtn.setAttribute('aria-pressed','false');
    replayBtn.textContent='Replay review';}}
  if(replayBtn)replayBtn.addEventListener('click',function(){
    if(timer){stop();return;}if(!current)return;
    var seq=(current.messages||[]).filter(function(m){return desks[m.agent];});if(!seq.length)return;
    Object.keys(desks).forEach(function(k){desks[k].querySelector('.say p').textContent='';desks[k].classList.add('quiet');
      desks[k].classList.remove('talking');});
    var i=0;floorEl.classList.add('playing');replayBtn.setAttribute('aria-pressed','true');replayBtn.textContent='Stop replay';
    function step(){if(i>=seq.length){stop();return;}var m=seq[i];
      Object.keys(desks).forEach(function(k){desks[k].classList.remove('talking');});
      speak(m,i);desks[m.agent].classList.remove('quiet');desks[m.agent].classList.add('talking');i++;}
    step();timer=setInterval(step,2200);
  });

  // --- the full chat ------------------------------------------------------
  function show(id){
    var r=byId[id];if(!r)return;floor(r);if(!log)return;log.textContent='';
    document.querySelectorAll('.sessions button').forEach(function(b){
      b.setAttribute('aria-pressed',b.getAttribute('data-review')===id?'true':'false');});
    r.messages.forEach(function(m){
      var row=document.createElement('div');row.className='msg '+(m.kind||'message');
      var av=document.createElement('div');av.className='av';av.textContent=(m.name||'?').slice(0,2).toUpperCase();
      av.style.setProperty('--ring',data.accents[m.agent]||'#b3a6d6');
      var body=document.createElement('div');var who=document.createElement('div');who.className='who';
      var b=document.createElement('b');b.textContent=m.name;who.appendChild(b);
      who.appendChild(document.createTextNode(' \\u00b7 '+(m.role||'')));
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
        f'<line x1="90" y1="8" x2="90" y2="{14 if i % 5 else 18}" stroke="#46297a" stroke-width="1" '
        f'transform="rotate({i * 6} 90 90)"/>' for i in range(60))
    value = _pct(s.win_rate) if s.win_rate is not None else "—"
    return f"""<div class="panel ring"><h2>Win rate</h2>
<svg width="170" height="170" viewBox="0 0 180 180" role="img" aria-label="Win rate {value}">
{ticks}<circle cx="90" cy="90" r="70" fill="none" stroke="#2f1d52" stroke-width="8"/>
<circle cx="90" cy="90" r="70" fill="none" stroke="#ff3fd8" stroke-width="8" stroke-linecap="round"
stroke-dasharray="{arc:.1f} {circ:.1f}" transform="rotate(-90 90 90)" style="filter:drop-shadow(0 0 6px rgba(255,63,216,.7))"/>
<circle cx="90" cy="90" r="54" fill="none" stroke="#2f1d52" stroke-dasharray="2 5"/>
<text x="90" y="98" text-anchor="middle" fill="#f3ecff" font-size="36" font-weight="600" font-family="Bahnschrift,DIN Alternate,Arial Narrow,sans-serif">{value}</text>
<text x="90" y="120" text-anchor="middle" fill="#b3a6d6" font-size="12" font-family="Bahnschrift,DIN Alternate,Arial Narrow,sans-serif">of {s.finished} filled</text>
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
    return f'<div class="panel"><h2>Scorecard</h2><div class="tiles">{body}</div></div>'


def _curve(s: Stats, finished: list[Entry]) -> str:
    head = '<div class="panel"><h2>Cumulative result <span>in R, trade by trade</span></h2>'
    if not s.curve:
        return head + '<div class="empty">The curve starts with the first filled trade.</div></div>'
    w, h, pl, pr, pt, pb = 1200, 240, 44, 16, 14, 30
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
        dots.append(f'<circle cx="{x(i):.1f}" cy="{y(cum):.1f}" r="4.5" fill="#43e8ff" stroke="#150d27" '
                    f'stroke-width="2"/><circle cx="{x(i):.1f}" cy="{y(cum):.1f}" r="12" fill="transparent" '
                    f'data-tip="{escape(tip, quote=True)}"/>')
    xt = "".join(f'<text class="tick" x="{x(i):.1f}" y="{h - 10}" text-anchor="middle">{i}</text>'
                 for i in range(1, n + 1) if n <= 20 or i % max(n // 10, 1) == 0)
    svg = f"""<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Cumulative R by trade, now {s.total_r:+.2f}R">
<defs><linearGradient id="fade" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#43e8ff" stop-opacity=".28"/>
<stop offset="1" stop-color="#43e8ff" stop-opacity="0"/></linearGradient></defs>
{''.join(grid)}<line class="zero" x1="{pl}" x2="{w - pr}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>
<polygon points="{area}" fill="url(#fade)"/>
<polyline points="{pts}" fill="none" stroke="#43e8ff" stroke-width="2" stroke-linejoin="round"
style="filter:drop-shadow(0 0 4px rgba(67,232,255,.7))"/>{''.join(dots)}{xt}</svg>"""
    return head + svg + "</div>"


def _live(entries: list[Entry]) -> str:
    live = [e for e in entries if e.status in jr.ACTIVE]
    head = '<div class="panel"><h2>Live orders</h2>'
    if not live:
        return head + '<div class="empty">No pending or open orders.</div></div>'
    rows = []
    for e in sorted(live, key=lambda e: e.created_at, reverse=True):
        when = (f"filled {_ny(e.filled_at, '%H:%M')}" if e.status == jr.OPEN
                else f"cancel {_ny(e.expires_at, '%H:%M')} NY")
        rows.append(f"""<div class="ord"><div class="h"><b>{'▲' if e.long else '▼'} <span class="tkt">{escape(e.ticket)}</span> {escape(e.symbol)}</b>{_chip(e.status)}</div>
<div class="lv num">{e.order_type} {_p(e.symbol, e.entry)} · SL {_p(e.symbol, e.stop)} · TP {_p(e.symbol, e.target)}</div>
<div class="lv">{(e.planned_rr or 0):.2f}R planned · {escape(e.conviction)} conviction · {escape(when)}</div></div>""")
    return head + f'<div class="orders">{"".join(rows)}</div></div>'


def _breakdown(title: str, groups: dict[str, dict]) -> str:
    head = f'<div class="panel"><h2>{escape(title)}</h2>'
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
    head = '<div class="panel" style="margin-top:12px"><h2>Journal <span>every final order</span></h2>'
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
        rows.append(f"""<tr><td class="num tkt">{escape(e.ticket)}</td><td class="num">{_ny(e.created_at)}</td><td><b>{escape(e.symbol)}</b><br>
<span style="color:var(--ink-2);font-size:12px">{e.order_type}</span></td>
<td class="n">{_p(e.symbol, e.entry)}</td><td class="n">{_p(e.symbol, e.stop)}</td><td class="n">{_p(e.symbol, e.target)}</td>
<td class="n">{(e.planned_rr or 0):.2f}</td><td>{escape(e.conviction)}</td><td>{_chip(e.status)}</td>
<td class="n">{_r(e.result_r)}</td><td>{why}</td></tr>""")
    return head + f"""<div class="scroll"><table><thead><tr><th>Ticket</th><th>Suggested (NY)</th><th>Order</th><th class="n">Entry</th>
<th class="n">Stop</th><th class="n">Target</th><th class="n">Plan R</th><th>Conviction</th><th>Status</th>
<th class="n">Result</th><th>Why</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>"""


def _chat(reviews: list[dict]) -> str:
    head = ('<div class="panel" id="chat" style="margin-top:12px"><h2>Desk chat '
            '<span>how each decision was argued</span></h2>')
    if not reviews:
        return head + ('<div class="empty">No conversations yet. Each agent review is saved here, '
                       'from the scan to the final check.</div></div>')
    buttons = []
    for i, r in enumerate(reviews):
        at = datetime.fromisoformat(r["at"])
        label = f"{r['orders']} order{'s' if r['orders'] != 1 else ''}" if r["orders"] else "no orders"
        if r["messages"] and all(m.get("agent") in ("desk", "trade_manager") for m in r["messages"]):
            ward = next((m["name"] for m in r["messages"] if m.get("agent") == "trade_manager"), "Trade manager")
            changed = [m for m in r["messages"] if m.get("title") == "Trade changes"]
            label = f"{ward} check" + (" · changes" if changed else " · held")
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
    return ('<div class="panel" style="margin-top:12px"><h2>The desk '
            f'<span>who decides</span></h2><div class="team">{"".join(cards)}</div></div>')


STAGES = (("scan", "SCAN"), ("macro", "MACRO"), ("debate", "DEBATE"), ("verdict", "VERDICT"),
          ("risk", "RISK"), ("book", "BOOK"), ("fills", "FILLS"))


def _pipeline(s: Stats) -> str:
    """The review's path from scan to book; the page script fills it from the latest review."""
    cards = []
    for key, label in STAGES:
        if key == "fills":
            value = " · ".join(x for x in (f"{s.open} open" if s.open else "", f"{s.pending} pending" if s.pending else "")
                               if x) or "None live"
            sub = f"{s.won} won · {s.lost} lost" if s.finished else "no filled trades yet"
            cards.append(f'<div class="stage{" done" if s.open + s.pending else ""}" data-stage="fills">'
                         f'<div class="k"><i></i>{label}</div><div class="v">{escape(value)}</div>'
                         f'<div class="s">{escape(sub)}</div></div>')
        else:
            cards.append(f'<div class="stage" data-stage="{key}"><div class="k"><i></i>{label}</div>'
                         f'<div class="v">—</div><div class="s">waiting for a review</div></div>')
    return f'<div class="pipe" aria-label="Latest review, stage by stage">{"".join(cards)}</div>'


def _screen(name: str, accent: str) -> str:
    """A small monitor with a trace in the agent's colour: decoration, so hidden from screen readers."""
    seed = sum(ord(c) * (i + 3) for i, c in enumerate(name))
    ys = [20 + ((seed >> (i * 2)) % 13) - 6 for i in range(9)]
    pts = " ".join(f"{6 + i * 5.75:.1f},{y}" for i, y in enumerate(ys))
    return (f'<svg class="screen" viewBox="0 0 58 40" aria-hidden="true">'
            f'<rect x="1" y="1" width="56" height="31" rx="3" fill="#0c0718" stroke="{accent}" stroke-opacity=".7"/>'
            f'<polyline points="{pts}" fill="none" stroke="{accent}" stroke-width="1.6" stroke-linejoin="round"/>'
            f'<rect x="24" y="32" width="10" height="4" fill="#46297a"/><rect x="16" y="36" width="26" height="3" rx="1.5" '
            f'fill="#46297a"/></svg>')


FLOOR = ("macro", "price_action", "bull", "bear", "research_manager", "portfolio_manager", "trader",
         "risk_aggressive", "risk_neutral", "risk_conservative", "trade_manager")


def _floor(people: dict[str, Profile]) -> str:
    """Every agent at a desk; the page script fills in what each said in the chosen review."""
    desks = []
    for key in FLOOR:
        p = people.get(key)
        if p is None:
            continue
        boss = " boss" if key == "portfolio_manager" else ""
        desks.append(
            f'<div class="desk{boss} quiet" data-agent="{key}" tabindex="0" role="button" '
            f'style="--a:{escape(p.accent, quote=True)};grid-area:{key}" '
            f'aria-label="{escape(p.name, quote=True)}, {escape(p.role, quote=True)}: show what they said">'
            f'<span class="seq"></span><div class="rig">{_screen(p.name, p.accent)}'
            f'<span class="avatar">{escape(p.name[:2].upper())}</span>'
            f'<div class="who"><b>{escape(p.name)}</b></div></div>'
            f'<div class="role" title="{escape(p.role, quote=True)}">{escape(p.role)}</div>'
            f'<div class="say"><p></p></div></div>')
    return ('<div class="panel floorwrap"><div class="floorhead"><h2>Trading floor <span id="floor-when"></span></h2>'
            '<button class="replay" id="replay" type="button" aria-pressed="false">Replay review</button></div>'
            '<div class="ticker" id="ticker"><b>Desk system</b>Waiting for the first review of the day.</div>'
            f'<div class="floor">{"".join(desks)}</div>'
            '<div class="detail" id="detail" aria-live="polite"></div></div>')


def _mark() -> str:
    return ('<svg class="mark" viewBox="0 0 40 40" aria-hidden="true">'
            '<circle cx="20" cy="20" r="18" fill="none" stroke="#ff3fd8" stroke-width="1.5"/>'
            '<path d="M12 23 q8 8 16 0" fill="none" stroke="#43e8ff" stroke-width="2" stroke-linecap="round"/>'
            '<circle class="pulse" cx="14" cy="15" r="2.2" fill="#ff3fd8"/><circle class="pulse" cx="26" cy="15" r="2.2" '
            'fill="#ff3fd8"/></svg>')


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
        win = f'<span><i class="dot on"></i>Window <b>open</b> until {_ny(closes, "%H:%M")} NY</span>'
    else:
        opens = window.next_open(now)
        reason = "market closed" if not market_open(now) else "window closed"
        win = f'<span><i class="dot"></i>{reason}, opens <b>{_ny(opens, "%a %H:%M")}</b> NY</span>'
    mode = ('<span><i class="dot on"></i>Watcher <b>live</b></span>' if live
            else '<span><i class="dot"></i>Snapshot</span>')
    finished = sorted((e for e in entries if e.status in jr.FINISHED and e.result_r is not None),
                      key=lambda e: e.exit_at or e.created_at)
    refresh = '<meta http-equiv="refresh" content="60">' if live else ""
    cycle = f"<span>Last cycle <b>{escape(last_cycle)}</b></span>" if last_cycle else ""
    data = json.dumps({"generated": now.isoformat(), "total_r": s.total_r, "trades": s.finished})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>Happy People</title><style>{CSS}</style></head>
<body class="{'live' if live else ''}"><div class="wrap"><div class="console">
<header><div class="brand">{_mark()}<div><h1>Happy People</h1>
<small>An AI trading desk for forex and metals: agent-reviewed limit orders</small></div></div>
<div class="status">{mode}{win}{cycle}<span class="num" id="clock"></span></div></header>
{_pipeline(s)}
<div class="grid main"><div class="stack">{_ring(s)}{_tiles(s)}</div>{_floor(people)}
<div class="stack">{_live(entries)}{_breakdown("By symbol", s.by_symbol)}{_breakdown("By conviction", s.by_conviction)}</div></div>
<div class="grid">{_curve(s, finished)}</div>
{_table(entries, {r["id"] for r in reviews})}
{_chat(reviews)}
{_team(people)}
<div class="foot">Paper results: each order is replayed on OANDA one-minute mid prices after it was suggested. A loss is −1R;
a win or a close at 16:55 New York (before the 17:00 rollover) is charged the spread recorded at the scan. A minute that touches the entry
and target together counts as no fill, and one that touches the stop and target as the stop. Real fills differ.
Generated {_ny(now)} New York.</div></div></div>
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
