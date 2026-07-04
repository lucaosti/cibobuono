"""
pipeline_control.py — File-based pipeline control (pause / resume / stop).

Commands are written to ``logs/pipeline_control.json``; run_pipeline reads
them between videos for pause/stop. Status survives process restarts.

CLI:
    python -m scripts.pipeline_control {status|pause|resume|stop}
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any

from scripts.utils import LOGS_DIR

CONTROL_PATH = LOGS_DIR / "pipeline_control.json"
PID_PATH = LOGS_DIR / "pipeline.pid"


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
                "message": "Pipeline terminated",
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
            "message": f"Running ({max_videos or 'all'} videos)",
            "pause_requested": False,
            "stop_requested": False,
        }
    )
    PID_PATH.write_text(str(pid), encoding="utf-8")


def mark_finished(message: str = "Completed") -> None:
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
        return False, "No pipeline running"
    state["pause_requested"] = True
    state["status"] = "paused"
    state["message"] = "Pause requested (after current video)"
    write_state(state)
    return True, state["message"]


def request_resume() -> tuple[bool, str]:
    state = sync_status()
    if state.get("status") not in ("running", "paused"):
        return False, "No pipeline paused"
    state["pause_requested"] = False
    state["status"] = "running"
    state["message"] = "Resumed"
    write_state(state)
    return True, state["message"]


def request_stop() -> tuple[bool, str]:
    state = sync_status()
    pid = state.get("pid")
    if not _pid_alive(pid):
        mark_finished("Stopped")
        return False, "No pipeline running"

    state["stop_requested"] = True
    state["status"] = "stopping"
    state["message"] = "Stop requested (after current video)"
    write_state(state)

    try:
        os.kill(int(pid), signal.SIGINT)
    except OSError:
        pass
    return True, state["message"]


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
                state["message"] = "Running"
                write_state(state)
            return True
        if should_abort is not None and should_abort():
            return False
        if state.get("status") != "paused":
            state = dict(state)
            state["status"] = "paused"
            state["message"] = "Paused"
            write_state(state)
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# CLI: python -m scripts.pipeline_control {status|pause|resume|stop}
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Control a running pipeline")
    parser.add_argument("command", choices=["status", "pause", "resume", "stop"])
    args = parser.parse_args(argv)

    if args.command == "status":
        state = sync_status()
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0

    action = {
        "pause": request_pause,
        "resume": request_resume,
        "stop": request_stop,
    }[args.command]
    ok, message = action()
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
