"""
dashboard_web.py — Browser dashboard for the CiboBuono pipeline.

Reads ``logs/dashboard_live.json`` (written by run_pipeline / Dashboard) and
serves a live-updating HTML page.

Usage:
    python -m scripts.dashboard_web
    python -m scripts.dashboard_web --port 8765 --host 0.0.0.0

Open http://localhost:8765/ while the pipeline runs (or after a run).
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.dashboard import DASHBOARD_SNAPSHOT_PATH, Dashboard

_HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CiboBuono — Pipeline Dashboard</title>
<style>
  :root {
    --bg: #0f1419; --card: #1a2332; --border: #2d3a4f;
    --text: #e7ecf3; --muted: #8b9cb3; --accent: #f59e0b;
    --green: #22c55e; --cyan: #06b6d4; --red: #ef4444; --yellow: #eab308;
  }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg);
    color: var(--text); margin: 0; padding: 1rem; line-height: 1.45; }
  h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
  .sub { color: var(--muted); font-size: .85rem; margin-bottom: 1rem; }
  .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
  .card h2 { font-size: .95rem; margin: 0 0 .75rem; color: var(--accent); text-transform: uppercase;
    letter-spacing: .04em; }
  .stat-row { display: flex; justify-content: space-between; padding: .25rem 0; border-bottom: 1px solid var(--border); }
  .stat-row:last-child { border: none; }
  .big { font-size: 1.8rem; font-weight: 700; color: var(--cyan); }
  .pending { color: var(--cyan); font-weight: 700; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th, td { text-align: left; padding: .4rem .35rem; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  .conf-high { color: var(--green); }
  .conf-mid { color: var(--yellow); }
  .conf-low { color: var(--red); }
  .flag { color: var(--red); font-size: .75rem; }
  .step-active { color: var(--accent); font-weight: 600; }
  .sources li { margin: .2rem 0; }
  .log { font-family: ui-monospace, monospace; font-size: .75rem; color: var(--muted);
    max-height: 160px; overflow-y: auto; white-space: pre-wrap; }
  .pill { display: inline-block; background: #243044; padding: .15rem .5rem; border-radius: 999px;
    font-size: .75rem; margin-right: .35rem; }
  .bar { background: #243044; border-radius: 6px; height: 10px; overflow: hidden; margin: .2rem 0 .5rem; }
  .bar > span { display: block; height: 100%; }
  .bar-ok > span { background: var(--green); }
  .bar-warn > span { background: var(--yellow); }
  .bar-hot > span { background: var(--red); }
  .hw-label { display: flex; justify-content: space-between; font-size: .82rem; }
  .pressure-ok { color: var(--green); }
  .pressure-hot { color: var(--red); font-weight: 700; }
  @media (min-width: 900px) {
    .wide { grid-column: 1 / -1; }
  }
</style>
</head>
<body>
<h1>🍕 CiboBuono Pipeline</h1>
<p class="sub">Aggiornamento live · <span id="updated">—</span></p>

<div class="grid">
  <div class="card">
    <h2>Coda &amp; statistiche</h2>
    <div class="stat-row"><span>Video in coda</span><span class="pending big" id="pending">—</span></div>
    <div class="stat-row"><span>Processati (DB)</span><span id="processed">—</span></div>
    <div class="stat-row"><span>Errori</span><span id="errored">—</span></div>
    <div class="stat-row"><span>Locali in DB</span><span id="locales_db">—</span></div>
    <div class="stat-row"><span>Locali questa run</span><span id="locales_run">—</span></div>
    <div class="stat-row"><span>Visite in DB</span><span id="visits">—</span></div>
  </div>

  <div class="card">
    <h2>Hardware (live)</h2>
    <div id="hw_body">
      <div class="hw-label"><span>RAM</span><span id="hw_ram">—</span></div>
      <div class="bar" id="hw_ram_bar"><span style="width:0%"></span></div>
      <div class="hw-label"><span>VRAM</span><span id="hw_vram">—</span></div>
      <div class="bar" id="hw_vram_bar"><span style="width:0%"></span></div>
      <div class="hw-label"><span>CPU load/core</span><span id="hw_cpu">—</span></div>
      <div class="hw-label"><span>Swap</span><span id="hw_swap">—</span></div>
      <p id="hw_pressure" style="margin:.5rem 0 0">—</p>
    </div>
  </div>

  <div class="card">
    <h2>Tempi</h2>
    <div class="stat-row"><span>Run totale</span><span id="run_elapsed">—</span></div>
    <div class="stat-row"><span>Video corrente</span><span id="video_elapsed">—</span></div>
    <div class="stat-row"><span>Step corrente</span><span id="step_elapsed">—</span></div>
    <div class="stat-row"><span>Media per video</span><span id="avg_video">—</span></div>
    <div class="stat-row"><span>Completati (run)</span><span id="completed_run">—</span></div>
  </div>

  <div class="card wide">
    <h2>Video in elaborazione</h2>
    <p><span class="pill" id="phase">—</span> <span class="pill" id="step">—</span></p>
    <p id="video_title" style="font-weight:600;margin:.5rem 0">—</p>
    <p class="sub" id="video_progress">—</p>
  </div>

  <div class="card">
    <h2>Dati video usati</h2>
    <ul class="sources" id="sources"></ul>
  </div>

  <div class="card wide">
    <h2>Locali — video corrente</h2>
    <table><thead><tr><th>Locale</th><th>Città</th><th>Certezza</th><th>Voto</th><th>Stato</th></tr></thead>
    <tbody id="current_locales"></tbody></table>
  </div>

  <div class="card wide">
    <h2>Locali — questa run</h2>
    <table><thead><tr><th>Locale</th><th>Città</th><th>Certezza</th><th>Video</th><th>Stato</th></tr></thead>
    <tbody id="run_locales"></tbody></table>
  </div>

  <div class="card wide">
    <h2>Database locali (top per certezza)</h2>
    <table><thead><tr><th>Locale</th><th>Città</th><th>Certezza</th><th>Coord.</th></tr></thead>
    <tbody id="db_locales"></tbody></table>
  </div>

  <div class="card wide">
    <h2>Ultimi video completati</h2>
    <table><thead><tr><th>Titolo</th><th>Esito</th><th>Durata</th><th>Visite</th><th>Locali</th></tr></thead>
    <tbody id="recent"></tbody></table>
  </div>

  <div class="card wide">
    <h2>Log</h2>
    <div class="log" id="log"></div>
  </div>
</div>

<script>
function fmt(s) {
  if (s == null || s === undefined) return '—';
  s = Math.floor(s);
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  if (h) return h+'h '+String(m).padStart(2,'0')+'m';
  return m+'m '+String(sec).padStart(2,'0')+'s';
}
function confClass(c) {
  if (c == null) return '';
  if (c >= 0.72) return 'conf-high';
  if (c >= 0.5) return 'conf-mid';
  return 'conf-low';
}
function confPct(c) { return c == null ? '—' : Math.round(c*100)+'%'; }

function barClass(pct) {
  if (pct == null) return 'bar';
  if (pct >= 90) return 'bar bar-hot';
  if (pct >= 75) return 'bar bar-warn';
  return 'bar bar-ok';
}
function setBar(id, pct) {
  const el = document.getElementById(id);
  el.className = barClass(pct);
  el.firstElementChild.style.width = (pct == null ? 0 : Math.min(100, pct)) + '%';
}
function renderHardware(hw) {
  if (!hw) {
    document.getElementById('hw_ram').textContent = 'n/d';
    document.getElementById('hw_pressure').textContent = '—';
    return;
  }
  document.getElementById('hw_ram').textContent =
    hw.ram_available_gb.toFixed(1) + ' / ' + hw.ram_total_gb.toFixed(1) + ' GB liberi · ' +
    hw.ram_used_percent.toFixed(0) + '% usata';
  setBar('hw_ram_bar', hw.ram_used_percent);

  if (hw.gpu_total_gb) {
    document.getElementById('hw_vram').textContent =
      hw.gpu_free_gb.toFixed(1) + ' / ' + hw.gpu_total_gb.toFixed(1) + ' GB liberi · ' +
      (hw.gpu_used_percent != null ? hw.gpu_used_percent.toFixed(0) + '%' : '—');
    setBar('hw_vram_bar', hw.gpu_used_percent);
  } else {
    document.getElementById('hw_vram').textContent = 'n/d (CPU / Metal)';
    setBar('hw_vram_bar', null);
  }
  document.getElementById('hw_cpu').textContent =
    hw.load_per_core.toFixed(2) + ' (' + hw.cpu_count + ' core)';
  document.getElementById('hw_swap').textContent =
    hw.swap_used_gb.toFixed(1) + ' GB (' + hw.swap_used_percent.toFixed(0) + '%)';

  const p = document.getElementById('hw_pressure');
  if (hw.under_pressure) {
    p.className = 'pressure-hot';
    p.textContent = '⚠ Sotto pressione: ' + (hw.pressure_reason || '');
  } else {
    p.className = 'pressure-ok';
    p.textContent = '✓ Risorse OK';
  }
}

function render(d) {
  if (!d) {
    document.getElementById('updated').textContent = 'Nessuno snapshot (pipeline non avviata?)';
    return;
  }
  document.getElementById('updated').textContent = d.updated_at || '—';
  const st = d.stats || {};
  document.getElementById('pending').textContent = st.pending ?? '—';
  document.getElementById('processed').textContent = st.processed ?? '—';
  document.getElementById('errored').textContent = st.errored ?? '—';
  document.getElementById('locales_db').textContent = st.locales_in_db ?? '—';
  document.getElementById('locales_run').textContent = st.run_locales_count ?? '—';
  document.getElementById('visits').textContent = st.visits_in_db ?? '—';

  renderHardware(d.hardware);

  const t = d.timing || {};
  document.getElementById('run_elapsed').textContent = fmt(t.run_elapsed_s);
  document.getElementById('video_elapsed').textContent = fmt(t.current_video_elapsed_s);
  document.getElementById('step_elapsed').textContent = fmt(t.current_step_elapsed_s);
  document.getElementById('avg_video').textContent = fmt(t.avg_video_s);
  document.getElementById('completed_run').textContent = t.videos_completed_this_run ?? '—';

  document.getElementById('phase').textContent = d.phase || '—';
  const cv = d.current_video || {};
  document.getElementById('step').textContent = cv.step || '—';
  document.getElementById('video_title').textContent = cv.title || '—';
  document.getElementById('video_progress').textContent =
    cv.index ? ('Video '+cv.index+' / '+cv.total+(cv.video_id ? ' · '+cv.video_id : '')) : 'In attesa…';

  const src = cv.sources || {};
  const ul = document.getElementById('sources');
  ul.innerHTML = '';
  const items = [];
  if (src.uses_title) items.push('Titolo YouTube');
  if (src.description_chars) items.push('Descrizione ('+src.description_chars+' char)');
  if (src.chapters_count) items.push('Capitoli ('+src.chapters_count+')');
  if (src.description_timestamps_count) items.push('Timestamp descrizione ('+src.description_timestamps_count+')');
  if (src.venue_hints_count) items.push('Hint locali ('+src.venue_hints_count+')');
  if (src.transcript_source) items.push('Trascrizione: '+src.transcript_source+' ('+(src.transcript_chars||0)+' char)');
  if (src.uses_ner) items.push('NER (GLiNER)');
  if (src.uses_llm) items.push('LLM (estrazione + verifica)');
  if (src.intel_type) items.push('Tipo video: '+src.intel_type);
  if (src.intel_city) items.push('Città intel: '+src.intel_city);
  if (src.food_gate) items.push('Food gate: '+src.food_gate);
  items.forEach(x => { const li = document.createElement('li'); li.textContent = x; ul.appendChild(li); });
  if (!items.length) { const li = document.createElement('li'); li.textContent = '—'; ul.appendChild(li); }

  function localeRows(arr, tbodyId, withVideo) {
    const tb = document.getElementById(tbodyId);
    tb.innerHTML = '';
    (arr || []).slice(-30).reverse().forEach(h => {
      const tr = document.createElement('tr');
      const c = h.confidence;
      tr.innerHTML = '<td>'+ (h.name||'?') +'</td><td>'+(h.city||'—')+'</td>'+
        '<td class="'+confClass(c)+'">'+confPct(c)+'</td>'+
        (withVideo ? '<td style="max-width:180px;overflow:hidden;text-overflow:ellipsis">'+(h.video_title||'—')+'</td>' : '<td>'+(h.rating||'—')+'</td>')+
        '<td>'+(h.flagged ? '<span class="flag">flag</span>' : 'ok')+'</td>';
      tb.appendChild(tr);
    });
  }
  localeRows(cv.extractions, 'current_locales', false);
  localeRows(d.run_locales, 'run_locales', true);

  const dbt = document.getElementById('db_locales');
  dbt.innerHTML = '';
  (d.database_locales || []).slice(0, 40).forEach(h => {
    const tr = document.createElement('tr');
    const c = h.confidence;
    tr.innerHTML = '<td>'+(h.name||'?')+'</td><td>'+(h.city||'—')+'</td>'+
      '<td class="'+confClass(c)+'">'+confPct(c)+'</td>'+
      '<td>'+(h.lat!=null ? h.lat+', '+h.lon : '—')+'</td>';
    dbt.appendChild(tr);
  });

  const rt = document.getElementById('recent');
  rt.innerHTML = '';
  (d.recent_videos || []).forEach(v => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td style="max-width:240px">'+v.title+'</td><td>'+v.outcome+'</td><td>'+
      fmt(v.duration_s)+'</td><td>'+(v.visits||0)+'</td><td>'+(v.locales||[]).length+'</td>';
    rt.appendChild(tr);
  });

  document.getElementById('log').textContent = (d.log_tail || []).join('\\n');
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    render(await r.json());
  } catch (e) { console.error(e); }
}
setInterval(poll, 2000);
poll();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        pass  # quiet

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._respond(200, "text/html; charset=utf-8", _HTML.encode("utf-8"))
        elif self.path == "/api/state":
            data = Dashboard.load_snapshot() or {}
            self._respond(200, "application/json", json.dumps(data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_error(404)

    def _respond(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="CiboBuono web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    snap = DASHBOARD_SNAPSHOT_PATH
    print(f"Dashboard web su http://{args.host}:{args.port}/")
    print(f"Snapshot: {snap} ({'presente' if snap.exists() else 'assente — avvia run_pipeline'})")
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStop.")


if __name__ == "__main__":
    main()
