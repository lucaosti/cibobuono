"""
dashboard_web.py — Browser dashboard for the CiboBuono pipeline.

Live monitoring, manual review queue, locale reports, JSON editing, pipeline control.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
from scripts.review_queue import (
    list_reports,
    pending_reviews,
    resolve_report,
    resolve_review,
    submit_report,
    visits_with_links,
)

_STATIC_HTML = Path(__file__).parent / "static" / "dashboard.html"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        pass

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = _STATIC_HTML.read_text(encoding="utf-8")
            self._respond(200, "text/html; charset=utf-8", html.encode("utf-8"))
        elif path == "/api/state":
            data = Dashboard.load_snapshot() or {}
            data["control"] = sync_status()
            data["review_pending_count"] = len(pending_reviews(limit=500))
            data["reports_open_count"] = len(list_reports(limit=500))
            data["recent_visits"] = visits_with_links(limit=25)
            self._json(data)
        elif path == "/api/review":
            self._json({"items": pending_reviews(limit=80)})
        elif path == "/api/reports":
            self._json({"items": list_reports(limit=50)})
        elif path == "/api/data":
            self._json({"files": sorted(EDITABLE_FILES.keys())})
        elif path.startswith("/api/data/"):
            name = unquote(path.split("/api/data/", 1)[1])
            try:
                content, kind = read_editable(name)
                if kind == "json":
                    payload = json.dumps(content, ensure_ascii=False, indent=2)
                else:
                    payload = content
                self._json({"name": name, "kind": kind, "content": payload})
            except KeyError:
                self._json({"error": "File non consentito"}, status=404)
            except OSError as e:
                self._json({"error": str(e)}, status=500)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_json() or {}

        if path == "/api/control":
            action = body.get("action", "")
            try:
                if action == "start":
                    mv = int(body.get("max_videos", 0))
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

        if path == "/api/review":
            action = body.get("action", "approve")
            try:
                ok, msg = resolve_review(
                    video_id=str(body.get("video_id", "")),
                    timestamp_start=str(body.get("timestamp_start", "")),
                    action="reject" if action == "reject" else "approve",
                    locale_name=body.get("locale_name"),
                    notes=str(body.get("notes", "")),
                )
                if not ok:
                    self._json({"error": msg}, status=404)
                    return
                self._json({"ok": True, "message": msg})
            except Exception as e:
                self._json({"error": str(e)}, status=500)
            return

        if path == "/api/reports":
            action = body.get("action", "submit")
            try:
                if action == "resolve":
                    ok, msg = resolve_report(str(body.get("report_id", "")))
                    if not ok:
                        self._json({"error": msg}, status=404)
                        return
                    self._json({"ok": True, "message": msg})
                    return
                entry = submit_report(
                    locale_name=str(body.get("locale_name", "")),
                    reason=str(body.get("reason", "")),
                    video_id=str(body.get("video_id", "")),
                    visit_id=str(body.get("visit_id", "")),
                    youtube_url=str(body.get("youtube_url", "")),
                )
                self._json({"ok": True, "report": entry})
            except Exception as e:
                self._json({"error": str(e)}, status=500)
            return

        if path.startswith("/api/data/"):
            name = unquote(path.split("/api/data/", 1)[1])
            try:
                write_editable(name, body.get("content"))
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

    if not _STATIC_HTML.exists():
        raise SystemExit(f"Missing dashboard UI: {_STATIC_HTML}")

    snap = DASHBOARD_SNAPSHOT_PATH
    print(f"Dashboard web su http://{args.host}:{args.port}/")
    print(f"Snapshot: {snap} ({'presente' if snap.exists() else 'assente'})")
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStop.")


if __name__ == "__main__":
    main()
