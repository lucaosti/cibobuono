"""
dashboard_web.py — Browser dashboard for the CiboBuono pipeline.

Live monitoring, JSON data editing, and pipeline control (start N / all pending,
pause, stop).

Usage:
    python -m scripts.dashboard_web
    python -m scripts.dashboard_web --port 8765 --host 0.0.0.0
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from scripts.dashboard import DASHBOARD_SNAPSHOT_PATH, Dashboard
from scripts.pipeline_control import (
    EDITABLE_FILES,
    read_editable,
    request_pause,
    request_resume,
    request_stop,
    start_pipeline,
    sync_status,
    write_editable,
)

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
  .sources li { margin: .2rem 0; }
  .log { font-family: ui-monospace, monospace; font-size: .75rem; color: var(--muted);
    max-height: 160px; overflow-y: auto; white-space: pre-wrap; }
  .pill { display: inline-block; background: #243044; padding: .15rem .5rem; border-radius: 999px;
    font-size: .75rem; margin-right: .35rem; }
  .pill-running { background: #14532d; color: var(--green); }
  .pill-paused { background: #713f12; color: var(--yellow); }
  .pill-idle { background: #243044; }
  .btn-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .75rem; align-items: center; }
  button, select, input[type=number] {
    background: #243044; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: .45rem .75rem; font-size: .85rem; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: #1e3a5f; border-color: var(--cyan); }
  button.danger { border-color: var(--red); color: #fca5a5; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  input[type=number] { width: 5rem; }
  textarea.editor {
    width: 100%; min-height: 280px; font-family: ui-monospace, monospace; font-size: .78rem;
    background: #0f1419; color: var(--text); border: 1px solid var(--border); border-radius: 8px;
    padding: .75rem; resize: vertical;
  }
  .msg { font-size: .82rem; margin-top: .5rem; min-height: 1.2em; }
  .msg.ok { color: var(--green); }
  .msg.err { color: var(--red); }
  @media (min-width: 900px) { .wide { grid-column: 1 / -1; } }
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
    <h2>Controllo pipeline</h2>
    <p><span class="pill" id="ctrl_status">—</span> <span id="ctrl_msg" class="sub"></span></p>
    <div class="btn-row">
      <label>Video: <input type="number" id="max_videos" min="1" value="5" placeholder="N"/></label>
      <button class="primary" id="btn_start_n">Avvia N video</button>
      <button class="primary" id="btn_start_all">Avvia tutti i pending</button>
    </div>
    <div class="btn-row">
      <button id="btn_pause">Pausa</button>
      <button id="btn_resume">Riprendi</button>
      <button class="danger" id="btn_stop">Stop</button>
    </div>
    <p class="msg" id="ctrl_feedback"></p>
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
    <h2>Modifica dati JSON</h2>
    <div class="btn-row">
      <select id="data_file"></select>
      <button id="btn_load_data">Carica</button>
      <button class="primary" id="btn_save_data">Salva</button>
    </div>
    <textarea class="editor" id="data_editor" spellcheck="false" placeholder="Seleziona un file e clicca Carica…"></textarea>
    <p class="msg" id="data_feedback"></p>
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
  if (c >= 0.65) return 'conf-high';
  if (c >= 0.5) return 'conf-mid';
  return 'conf-low';
}
function confPct(c) { return c == null ? '—' : Math.round(c*100)+'%'; }

async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function setFeedback(id, msg, ok) {
  const el = document.getElementById(id);
  el.textContent = msg || '';
  el.className = 'msg ' + (ok ? 'ok' : msg ? 'err' : '');
}

function renderControl(ctrl) {
  const st = (ctrl && ctrl.status) || 'idle';
  const pill = document.getElementById('ctrl_status');
  pill.textContent = st;
  pill.className = 'pill ' + (st === 'running' ? 'pill-running' : st === 'paused' ? 'pill-paused' : 'pill-idle');
  document.getElementById('ctrl_msg').textContent = (ctrl && ctrl.message) || '';
  const running = st === 'running' || st === 'paused' || st === 'stopping';
  document.getElementById('btn_start_n').disabled = running;
  document.getElementById('btn_start_all').disabled = running;
  document.getElementById('btn_pause').disabled = st !== 'running';
  document.getElementById('btn_resume').disabled = st !== 'paused';
  document.getElementById('btn_stop').disabled = !running;
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
  renderControl(d.control);

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
    const d = await api('/api/state');
    render(d);
  } catch (e) { console.error(e); }
}

async function ctrlAction(action, maxVideos) {
  try {
    const body = { action };
    if (maxVideos != null) body.max_videos = maxVideos;
    const r = await api('/api/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    setFeedback('ctrl_feedback', r.message, true);
    poll();
  } catch (e) {
    setFeedback('ctrl_feedback', e.message, false);
  }
}

async function loadDataFiles() {
  const r = await api('/api/data');
  const sel = document.getElementById('data_file');
  sel.innerHTML = '';
  (r.files || []).forEach(f => {
    const o = document.createElement('option');
    o.value = f; o.textContent = f;
    sel.appendChild(o);
  });
}

async function loadData() {
  const name = document.getElementById('data_file').value;
  if (!name) return;
  try {
    const r = await api('/api/data/' + encodeURIComponent(name));
    document.getElementById('data_editor').value = r.content;
    setFeedback('data_feedback', 'Caricato ' + name, true);
  } catch (e) {
    setFeedback('data_feedback', e.message, false);
  }
}

async function saveData() {
  const name = document.getElementById('data_file').value;
  const raw = document.getElementById('data_editor').value;
  if (!name) return;
  try {
    let content = raw;
    if (!name.endsWith('.txt')) content = JSON.parse(raw);
    const r = await api('/api/data/' + encodeURIComponent(name), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ content }),
    });
    setFeedback('data_feedback', r.message || 'Salvato', true);
  } catch (e) {
    setFeedback('data_feedback', e.message, false);
  }
}

document.getElementById('btn_start_n').onclick = () => {
  const n = parseInt(document.getElementById('max_videos').value, 10);
  if (!n || n < 1) { setFeedback('ctrl_feedback', 'Inserisci un numero valido', false); return; }
  ctrlAction('start', n);
};
document.getElementById('btn_start_all').onclick = () => ctrlAction('start', 0);
document.getElementById('btn_pause').onclick = () => ctrlAction('pause');
document.getElementById('btn_resume').onclick = () => ctrlAction('resume');
document.getElementById('btn_stop').onclick = () => ctrlAction('stop');
document.getElementById('btn_load_data').onclick = loadData;
document.getElementById('btn_save_data').onclick = saveData;

loadDataFiles();
setInterval(poll, 2000);
poll();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        pass

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._respond(200, "text/html; charset=utf-8", _HTML.encode("utf-8"))
        elif path == "/api/state":
            data = Dashboard.load_snapshot() or {}
            data["control"] = sync_status()
            self._json(data)
        elif path == "/api/data":
            self._json({"files": sorted(EDITABLE_FILES.keys())})
        elif path.startswith("/api/data/"):
            name = unquote(path.split("/api/data/", 1)[1])
            try:
                content, kind = read_editable(name)
                if kind == "json":
                    self._json({"name": name, "kind": kind, "content": json.dumps(content, ensure_ascii=False, indent=2)})
                else:
                    self._json({"name": name, "kind": kind, "content": content})
            except KeyError:
                self._json({"error": "File non consentito"}, status=404)
            except OSError as e:
                self._json({"error": str(e)}, status=500)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_json()

        if path == "/api/control":
            action = (body or {}).get("action", "")
            try:
                if action == "start":
                    mv = int((body or {}).get("max_videos", 0))
                    ok, msg = start_pipeline(max_videos=max(0, mv))
                elif action == "pause":
                    ok, msg = request_pause()
                elif action == "resume":
                    ok, msg = request_resume()
                elif action == "stop":
                    ok, msg = request_stop()
                else:
                    self._json({"error": "Azione sconosciuta"}, status=400)
                    return
                self._json({"ok": ok, "message": msg, "control": sync_status()})
            except Exception as e:
                self._json({"error": str(e)}, status=500)
            return

        if path.startswith("/api/data/"):
            name = unquote(path.split("/api/data/", 1)[1])
            try:
                write_editable(name, (body or {}).get("content"))
                self._json({"ok": True, "message": f"Salvato {name}"})
            except KeyError:
                self._json({"error": "File non consentito"}, status=404)
            except (ValueError, TypeError) as e:
                self._json({"error": str(e)}, status=400)
            except OSError as e:
                self._json({"error": str(e)}, status=500)
            return

        self.send_error(404)

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _json(self, data: dict, *, status: int = 200) -> None:
        self._respond(status, "application/json", json.dumps(data, ensure_ascii=False).encode("utf-8"))

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
