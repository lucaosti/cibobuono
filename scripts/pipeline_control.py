"""
pipeline_control.py — File-based pipeline control for the web dashboard.

The dashboard writes commands here; run_pipeline reads them between videos
for pause/stop. Status survives dashboard restarts.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.utils import (
    CHANNELS_INPUT,
    CHANNELS_JSON,
    CORRECTIONS_JSON,
    DATA_DIR,
    FLAGGED_SEGMENTS_JSON,
    LOCALES_JSON,
    LOGS_DIR,
    PROCESSED_VIDEOS_JSON,
    PROJECT_ROOT,
    SKIPPED_VIDEOS_JSON,
    VIDEOS_JSON,
    VISITS_JSON,
    load_json,
    save_json,
)

CONTROL_PATH = LOGS_DIR / "pipeline_control.json"
PID_PATH = LOGS_DIR / "pipeline.pid"

EDITABLE_FILES: dict[str, Path] = {
    "channels_input.txt": CHANNELS_INPUT,
    "channels.json": CHANNELS_JSON,
    "videos.json": VIDEOS_JSON,
    "locales.json": LOCALES_JSON,
    "visits.json": VISITS_JSON,
    "processed_videos.json": PROCESSED_VIDEOS_JSON,
    "flagged_segments.json": FLAGGED_SEGMENTS_JSON,
    "skipped_videos.json": SKIPPED_VIDEOS_JSON,
    "corrections.json": CORRECTIONS_JSON,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "pid": None,
        "max_videos": 0,
        "started_at": None,
        "updated_at": _now_iso(),
        "message": "",
        "pause_requested": False,
        "stop_requested": False,
    }


def read_state() -> dict[str, Any]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONTROL_PATH.exists():
        return default_state()
    try:
        with open(CONTROL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            base = default_state()
            base.update(data)
            return base
    except (json.JSONDecodeError, OSError):
        pass
    return default_state()


def write_state(state: dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = _now_iso()
    tmp = CONTROL_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(CONTROL_PATH)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def sync_status() -> dict[str, Any]:
    """Reconcile control file with actual process state."""
    state = read_state()
    pid = state.get("pid")
    if state.get("status") in ("running", "paused", "stopping") and not _pid_alive(pid):
        state.update(
            {
                "status": "idle",
                "pid": None,
                "pause_requested": False,
                "stop_requested": False,
                "message": "Pipeline terminata",
            }
        )
        write_state(state)
        PID_PATH.unlink(missing_ok=True)
    return state


def mark_started(*, pid: int, max_videos: int) -> None:
    write_state(
        {
            "status": "running",
            "pid": pid,
            "max_videos": max_videos,
            "started_at": _now_iso(),
            "message": f"In esecuzione ({max_videos or 'tutti'} video)",
            "pause_requested": False,
            "stop_requested": False,
        }
    )
    PID_PATH.write_text(str(pid), encoding="utf-8")


def mark_finished(message: str = "Completato") -> None:
    write_state(
        {
            **default_state(),
            "message": message,
        }
    )
    PID_PATH.unlink(missing_ok=True)


def request_pause() -> tuple[bool, str]:
    state = sync_status()
    if state.get("status") not in ("running", "paused"):
        return False, "Nessuna pipeline in esecuzione"
    state["pause_requested"] = True
    state["status"] = "paused"
    state["message"] = "Pausa richiesta (dopo il video corrente)"
    write_state(state)
    return True, state["message"]


def request_resume() -> tuple[bool, str]:
    state = sync_status()
    if state.get("status") not in ("running", "paused"):
        return False, "Nessuna pipeline in pausa"
    state["pause_requested"] = False
    state["status"] = "running"
    state["message"] = "Ripresa"
    write_state(state)
    return True, state["message"]


def request_stop() -> tuple[bool, str]:
    state = sync_status()
    pid = state.get("pid")
    if not _pid_alive(pid):
        mark_finished("Fermata")
        return False, "Nessuna pipeline in esecuzione"

    state["stop_requested"] = True
    state["status"] = "stopping"
    state["message"] = "Stop richiesto (dopo il video corrente)"
    write_state(state)

    try:
        os.kill(int(pid), signal.SIGINT)
    except OSError:
        pass
    return True, state["message"]


def start_pipeline(*, max_videos: int = 0) -> tuple[bool, str]:
    state = sync_status()
    if _pid_alive(state.get("pid")):
        return False, f"Pipeline già in esecuzione (pid {state['pid']})"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_pipeline",
        "--skip-push",
        "--no-dashboard",
        "--max-videos",
        str(max_videos),
    ]
    log_path = LOGS_DIR / "pipeline.log"
    log_fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    mark_started(pid=proc.pid, max_videos=max_videos)
    label = f"{max_videos} video" if max_videos > 0 else "tutti i pending"
    return True, f"Pipeline avviata (pid {proc.pid}, {label})"


def wait_if_paused(*, should_abort=None, poll_s: float = 2.0) -> bool:
    """Block while pause is requested. Returns False if stop was requested."""
    while True:
        state = read_state()
        if state.get("stop_requested"):
            return False
        if not state.get("pause_requested"):
            if state.get("status") == "paused":
                state = dict(state)
                state["status"] = "running"
                state["message"] = "In esecuzione"
                write_state(state)
            return True
        if should_abort is not None and should_abort():
            return False
        if state.get("status") != "paused":
            state = dict(state)
            state["status"] = "paused"
            state["message"] = "In pausa"
            write_state(state)
        time.sleep(poll_s)


def read_editable(name: str) -> tuple[Any, str]:
    path = EDITABLE_FILES.get(name)
    if path is None:
        raise KeyError(name)
    if name.endswith(".txt"):
        return path.read_text(encoding="utf-8"), "text"
    return load_json(path), "json"


def write_editable(name: str, content: Any) -> None:
    path = EDITABLE_FILES.get(name)
    if path is None:
        raise KeyError(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".txt"):
        if not isinstance(content, str):
            raise ValueError("Expected string for text file")
        path.write_text(content, encoding="utf-8")
        return
    if not isinstance(content, list):
        raise ValueError("Expected JSON array")
    save_json(path, content)
