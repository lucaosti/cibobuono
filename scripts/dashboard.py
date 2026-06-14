"""
dashboard.py — Live pipeline dashboard (terminal + JSON snapshot for web UI).

Terminal (Rich): run_pipeline with dashboard enabled (default).
Web: ``python -m scripts.dashboard_web`` reads ``logs/dashboard_live.json``.

Shows pending/processed counts, current video + step timing, data sources in
use (title, description, chapters, transcript, NER, LLM), and each locale with
confidence — including locales found during the current run and in the DB.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from scripts.utils import LOCALES_JSON, LOGS_DIR, VISITS_JSON, load_json

# ── Pipeline steps ─────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "Fetch channels",
    "Catalog videos",
    "Prefetch audio",
    "Download audio",
    "Video intel",
    "Transcribe",
    "Chunk",
    "Extract (LLM)",
    "Geocode",
    "Verify (OSM)",
    "Deduplicate",
    "Populate",
    "Update status",
]

DASHBOARD_SNAPSHOT_PATH = LOGS_DIR / "dashboard_live.json"


@dataclass
class VideoSourcesInfo:
    """Which parts of the video feed the extraction pipeline."""

    video_id: str = ""
    title: str = ""
    uses_title: bool = True
    description_chars: int = 0
    chapters_count: int = 0
    description_timestamps_count: int = 0
    venue_hints_count: int = 0
    intel_city: str = ""
    intel_type: str = ""
    intel_series: str = ""
    transcript_source: str = ""  # faster_whisper | youtube_subs_manual | cached | ""
    transcript_chars: int = 0
    comments_count: int = 0
    uses_ner: bool = False
    uses_llm: bool = False
    food_gate: str = ""


@dataclass
class LocaleHit:
    name: str
    city: str = ""
    confidence: float = 0.0
    rating: str | None = None
    flagged: bool = False
    flag_reason: str | None = None
    video_id: str = ""
    video_title: str = ""
    youtube_url: str = ""
    mention_timestamp: str = ""


@dataclass
class CompletedVideo:
    video_id: str
    title: str
    outcome: str
    duration_s: float
    visits: int = 0
    flagged: int = 0
    locales: list[LocaleHit] = field(default_factory=list)


@dataclass
class DashboardState:
    phase: str = "Idle"
    current_video_index: int = 0
    total_videos: int = 0
    current_video_id: str = ""
    current_video_title: str = ""
    current_step: str = ""
    current_step_index: int = 0

    total_in_db: int = 0
    pending: int = 0
    processed: int = 0
    skipped: int = 0
    errored: int = 0
    locales_found: int = 0
    visits_created: int = 0
    flagged: int = 0
    channels: int = 0

    start_time: float = field(default_factory=time.time)
    video_start_time: float = 0.0
    step_start_time: float = 0.0

    sources: VideoSourcesInfo = field(default_factory=VideoSourcesInfo)
    current_extractions: list[LocaleHit] = field(default_factory=list)
    run_locales: list[LocaleHit] = field(default_factory=list)
    recent_completed: list[CompletedVideo] = field(default_factory=list)

    log_lines: list[str] = field(default_factory=list)
    max_log_lines: int = 12

    def avg_video_seconds(self) -> float | None:
        if not self.recent_completed:
            return None
        return sum(v.duration_s for v in self.recent_completed) / len(self.recent_completed)


class Dashboard:
    """Pipeline dashboard: optional Rich live UI + always-on JSON snapshot."""

    def __init__(self, *, live: bool = True) -> None:
        self.live_enabled = live
        self.console = Console()
        self.state = DashboardState()
        self._live: Live | None = None
        self._video_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        )
        self._video_task = self._video_progress.add_task("Videos", total=0)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.persist_snapshot()

    def start(self) -> None:
        if not self.live_enabled:
            return
        self.state.start_time = time.time()
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._refresh()
            self._live.stop()
            self._live = None
        self.persist_snapshot()

    # ── Mutations ──────────────────────────────────────────────────────

    def set_phase(self, phase: str) -> None:
        self.state.phase = phase
        self._touch()

    def set_totals(
        self,
        total_in_db: int = 0,
        pending: int = 0,
        processed: int = 0,
        skipped: int = 0,
        errored: int = 0,
        channels: int = 0,
        locales_found: int | None = None,
        visits_created: int | None = None,
        flagged: int | None = None,
    ) -> None:
        s = self.state
        s.total_in_db = total_in_db
        s.pending = pending
        s.processed = processed
        s.skipped = skipped
        s.errored = errored
        s.channels = channels
        if locales_found is not None:
            s.locales_found = locales_found
        if visits_created is not None:
            s.visits_created = visits_created
        if flagged is not None:
            s.flagged = flagged
        self._touch()

    def set_video_batch(self, total: int) -> None:
        self.state.total_videos = total
        self._video_progress.update(self._video_task, total=total, completed=0)
        self._touch()

    def update_video(self, index: int, title: str, video_id: str = "") -> None:
        s = self.state
        s.current_video_index = index
        s.current_video_title = title
        s.current_video_id = video_id
        s.current_step = "Download audio"
        s.current_step_index = PIPELINE_STEPS.index("Download audio")
        s.video_start_time = time.time()
        s.step_start_time = s.video_start_time
        s.sources = VideoSourcesInfo(video_id=video_id, title=title[:200])
        s.current_extractions = []
        self._video_progress.update(self._video_task, completed=index - 1)
        self._touch()

    def set_step(self, step: str) -> None:
        self.state.current_step = step
        self.state.step_start_time = time.time()
        if step in PIPELINE_STEPS:
            self.state.current_step_index = PIPELINE_STEPS.index(step)
        self._touch()

    def set_video_sources(self, **kwargs: Any) -> None:
        src = self.state.sources
        for k, v in kwargs.items():
            if hasattr(src, k):
                setattr(src, k, v)
        self._touch()

    def set_extractions(
        self,
        extractions: list[dict],
        flagged: list[dict] | None = None,
    ) -> None:
        flagged = flagged or []
        hits: list[LocaleHit] = []
        for e in extractions:
            mt = e.get("mention_time", e.get("chunk_start_seconds", 0))
            yt = f"https://youtu.be/{self.state.current_video_id}?t={int(mt or 0)}"
            ts = e.get("mention_timestamp") or e.get("chunk_start", "")
            hits.append(
                LocaleHit(
                    name=str(e.get("locale_name", "?")),
                    city=str(e.get("city", "")),
                    confidence=float(e.get("confidence", 0)),
                    rating=e.get("rating"),
                    flagged=False,
                    video_id=self.state.current_video_id,
                    video_title=self.state.current_video_title[:120],
                    youtube_url=yt,
                    mention_timestamp=str(ts),
                )
            )
        for e in flagged:
            mt = e.get("mention_time", e.get("chunk_start_seconds", 0))
            yt = f"https://youtu.be/{self.state.current_video_id}?t={int(mt or 0)}"
            ts = e.get("mention_timestamp") or e.get("chunk_start", "")
            hits.append(
                LocaleHit(
                    name=str(e.get("locale_name", "?")),
                    city=str(e.get("city", "")),
                    confidence=float(e.get("confidence", 0)),
                    rating=e.get("rating"),
                    flagged=True,
                    flag_reason=e.get("_flag_reason"),
                    video_id=self.state.current_video_id,
                    video_title=self.state.current_video_title[:120],
                    youtube_url=yt,
                    mention_timestamp=str(ts),
                )
            )
        self.state.current_extractions = hits
        # Merge into run totals (dedupe by video+name)
        seen = {(h.video_id, h.name.lower()) for h in self.state.run_locales}
        for h in hits:
            key = (h.video_id, h.name.lower())
            if key not in seen:
                self.state.run_locales.append(h)
                seen.add(key)
        self._touch()

    def complete_video(
        self,
        *,
        outcome: str = "processed",
        visits: int = 0,
        flagged: int = 0,
    ) -> None:
        s = self.state
        duration = time.time() - s.video_start_time if s.video_start_time else 0.0
        rec = CompletedVideo(
            video_id=s.current_video_id,
            title=s.current_video_title[:200],
            outcome=outcome,
            duration_s=round(duration, 1),
            visits=visits,
            flagged=flagged,
            locales=list(s.current_extractions),
        )
        s.recent_completed.insert(0, rec)
        s.recent_completed = s.recent_completed[:20]
        self._video_progress.update(self._video_task, completed=s.current_video_index)
        self._touch()

    def tick_stat(self, stat: str, delta: int = 1) -> None:
        cur = getattr(self.state, stat, 0)
        setattr(self.state, stat, cur + delta)
        self._touch()

    def log(self, message: str) -> None:
        self.state.log_lines.append(message)
        if len(self.state.log_lines) > self.state.max_log_lines:
            self.state.log_lines = self.state.log_lines[-self.state.max_log_lines :]
        self._touch()

    # ── Snapshot (web UI + headless watch mode) ────────────────────────

    def to_snapshot_dict(self) -> dict:
        s = self.state
        elapsed = time.time() - s.start_time
        video_elapsed = (
            time.time() - s.video_start_time if s.video_start_time else 0.0
        )
        step_elapsed = time.time() - s.step_start_time if s.step_start_time else 0.0

        db_locales = _database_locales_with_confidence(limit=50)

        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": s.phase,
            "stats": {
                "channels": s.channels,
                "total_videos": s.total_in_db,
                "pending": s.pending,
                "processed": s.processed,
                "skipped": s.skipped,
                "errored": s.errored,
                "locales_in_db": s.locales_found,
                "visits_in_db": s.visits_created,
                "flagged_in_db": s.flagged,
                "run_locales_count": len(s.run_locales),
            },
            "timing": {
                "run_elapsed_s": round(elapsed, 1),
                "current_video_elapsed_s": round(video_elapsed, 1),
                "current_step_elapsed_s": round(step_elapsed, 1),
                "avg_video_s": round(s.avg_video_seconds(), 1)
                if s.avg_video_seconds() is not None
                else None,
                "videos_completed_this_run": len(s.recent_completed),
            },
            "current_video": {
                "index": s.current_video_index,
                "total": s.total_videos,
                "video_id": s.current_video_id,
                "title": s.current_video_title,
                "step": s.current_step,
                "step_index": s.current_step_index,
                "sources": asdict(s.sources),
                "extractions": [asdict(h) for h in s.current_extractions],
            },
            "run_locales": [asdict(h) for h in s.run_locales[-100:]],
            "recent_videos": [
                {**asdict(v), "locales": [asdict(l) for l in v.locales]}
                for v in s.recent_completed
            ],
            "database_locales": db_locales,
            "log_tail": list(s.log_lines[-20:]),
        }

    def persist_snapshot(self) -> None:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.to_snapshot_dict(), ensure_ascii=False, indent=2)
            fd, tmp = tempfile.mkstemp(
                suffix=".json.tmp",
                prefix=".dashboard_live.",
                dir=str(LOGS_DIR),
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp, DASHBOARD_SNAPSHOT_PATH)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as _e:
            logger.debug("Dashboard snapshot write failed (non-critical): %s", _e)

    @staticmethod
    def load_snapshot() -> dict | None:
        if not DASHBOARD_SNAPSHOT_PATH.exists():
            return None
        try:
            with open(DASHBOARD_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def reset_to_idle(self) -> None:
        """Clear ephemeral run state so the web dashboard does not show a stale run."""
        s = self.state
        s.phase = "Idle"
        s.current_video_index = 0
        s.total_videos = 0
        s.current_video_id = ""
        s.current_video_title = ""
        s.current_step = ""
        s.current_step_index = 0
        s.current_extractions = []
        s.run_locales = []
        s.recent_completed = []
        s.sources = VideoSourcesInfo()
        s.video_start_time = 0.0
        s.step_start_time = 0.0
        live = compute_live_stats()
        s.total_in_db = live["total_videos"]
        s.pending = live["pending"]
        s.processed = live["processed"]
        s.skipped = live["skipped"]
        s.errored = live["errored"]
        s.channels = live["channels"]
        s.locales_found = live["locales_in_db"]
        s.visits_created = live["visits_in_db"]
        s.flagged = live["flagged_in_db"]
        self.persist_snapshot()

    def _touch(self) -> None:
        self.persist_snapshot()
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._build_layout())

    def _build_layout(self) -> Panel:
        cols, rows = os.get_terminal_size()
        s = self.state

        header = Text()
        header.append("🍕 CiboBuono Pipeline ", style="bold yellow")
        header.append(f"— {s.phase}", style="bold white")
        header.append(f"  ⏱ {self._fmt_time(time.time() - s.start_time)}", style="dim")
        if s.pending:
            header.append(f"  · {s.pending} in coda", style="cyan")

        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(justify="right", style="dim")
        stats_table.add_column(style="bold")
        stats_table.add_column(justify="right", style="dim")
        stats_table.add_column(style="bold")
        stats_table.add_row("Canali", str(s.channels), "Video", str(s.total_in_db))
        stats_table.add_row(
            "In coda", f"[cyan bold]{s.pending}[/]",
            "Processati", f"[green]{s.processed}[/]",
        )
        stats_table.add_row(
            "Saltati", f"[yellow]{s.skipped}[/]",
            "Errori", f"[red]{s.errored}[/]",
        )
        stats_table.add_row(
            "Locali DB", f"[magenta]{s.locales_found}[/]",
            "Visite DB", f"[magenta]{s.visits_created}[/]",
        )
        avg = s.avg_video_seconds()
        stats_table.add_row(
            "Locali run", f"[magenta]{len(s.run_locales)}[/]",
            "Media/video", self._fmt_time(avg) if avg else "—",
        )
        stats_panel = Panel(stats_table, title="[bold]Statistiche", border_style="blue", expand=True)

        if s.current_video_index > 0:
            video_info = Text()
            video_info.append(f"[{s.current_video_index}/{s.total_videos}] ", style="bold")
            video_info.append(s.current_video_title[: cols - 20], style="italic")
            elapsed_parts: list[str] = []
            if s.video_start_time:
                elapsed_parts.append(f"video {self._fmt_time(time.time() - s.video_start_time)}")
            if s.current_step:
                step_t = self._fmt_time(time.time() - s.step_start_time) if s.step_start_time else "—"
                elapsed_parts.append(f"step «{s.current_step}» {step_t}")
            if elapsed_parts:
                video_info.append(f"\n  {' · '.join(elapsed_parts)}", style="dim")
            done = " ".join(
                f"[green]✓{st}[/]" if idx < s.current_step_index
                else f"[bold yellow]▸{st}[/]" if idx == s.current_step_index
                else f"[dim]{st}[/]"
                for idx, st in enumerate(PIPELINE_STEPS[2:], start=2)
            )
            video_info.append(f"\n  {done}")
            video_panel = Panel(video_info, title="[bold]Video corrente", border_style="green", expand=True)
        else:
            video_panel = Panel(
                Align.center(Text("In attesa di video…", style="dim")),
                title="[bold]Video corrente", border_style="green", expand=True,
            )

        loc_table = Table(show_header=True, header_style="bold", expand=True)
        loc_table.add_column("Locale", ratio=4)
        loc_table.add_column("Conf.", justify="right", width=6)
        loc_table.add_column("Voto", width=5)
        loc_table.add_column("Stato", width=6)
        shown = s.current_extractions or s.run_locales[-8:]
        for h in shown[-10:]:
            conf_style = "green" if h.confidence >= 0.72 else "yellow" if h.confidence >= 0.5 else "red"
            loc_table.add_row(
                h.name[:55],
                f"[{conf_style}]{h.confidence:.0%}[/]",
                str(h.rating or "—"),
                "[red]⚑[/]" if h.flagged else "ok",
            )
        if not shown:
            loc_table.add_row("—", "—", "—", "—")
        locales_panel = Panel(
            loc_table,
            title=f"[bold]Locali ({len(s.run_locales)} in questa run)",
            border_style="magenta", expand=True,
        )

        progress_panel = Panel(
            self._video_progress,
            title="[bold]Avanzamento batch", border_style="cyan", expand=True,
        )

        max_log = max(3, rows - 28)
        s.max_log_lines = max_log
        log_text = Text()
        for line in s.log_lines[-max_log:]:
            log_text.append(f"  {line[: cols - 6]}\n", style="dim")
        if not s.log_lines:
            log_text.append("  (nessun log)\n", style="dim italic")
        log_panel = Panel(log_text, title="[bold]Log", border_style="dim", expand=True)

        body = Group(
            Align.center(header),
            "",
            stats_panel,
            video_panel,
            locales_panel,
            progress_panel,
            log_panel,
        )
        return Panel(
            body,
            title="[bold white on blue] CiboBuono ",
            border_style="blue",
            expand=True,
            padding=(0, 1),
        )

    @staticmethod
    def _fmt_time(seconds: float | None) -> str:
        if seconds is None:
            return "—"
        m, sec = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m {sec:02d}s"
        return f"{m}m {sec:02d}s"


def compute_live_stats() -> dict[str, int]:
    """Read current counts directly from data JSON files (always fresh)."""
    from scripts.utils import (
        CHANNELS_JSON,
        FLAGGED_SEGMENTS_JSON,
        SKIPPED_VIDEOS_JSON,
        VIDEOS_JSON,
    )

    videos = load_json(VIDEOS_JSON)
    counts = {"total": len(videos), "pending": 0, "processed": 0, "errored": 0}
    for v in videos:
        s = v.get("status", "pending")
        if s in counts:
            counts[s] += 1
    return {
        "channels": len(load_json(CHANNELS_JSON)),
        "total_videos": counts["total"],
        "pending": counts["pending"],
        "processed": counts["processed"],
        "skipped": len(load_json(SKIPPED_VIDEOS_JSON)),
        "errored": counts["errored"],
        "locales_in_db": len(load_json(LOCALES_JSON)),
        "visits_in_db": len(load_json(VISITS_JSON)),
        "flagged_in_db": len(load_json(FLAGGED_SEGMENTS_JSON)),
    }


def build_web_state(snapshot: dict | None, *, pipeline_running: bool) -> dict:
    """Merge pipeline snapshot with live DB stats; hide stale run data when idle."""
    live = compute_live_stats()
    base: dict = dict(snapshot) if snapshot else {}

    stats = dict(base.get("stats") or {})
    stats.update(live)
    stats["run_locales_count"] = (
        len(base.get("run_locales") or []) if pipeline_running else 0
    )
    base["stats"] = stats

    base["database_locales"] = _database_locales_with_confidence(limit=50)
    base["pipeline_running"] = pipeline_running

    if not pipeline_running:
        base["phase"] = "Idle"
        snap_phase = (snapshot or {}).get("phase", "Idle")
        had_run = bool(
            (snapshot or {}).get("run_locales")
            or (snapshot or {}).get("recent_videos")
            or ((snapshot or {}).get("current_video") or {}).get("video_id")
            or snap_phase not in ("Idle", "")
        )
        base["stale_snapshot"] = had_run
        base["run_locales"] = []
        base["recent_videos"] = []
        base["current_video"] = {
            "index": 0,
            "total": 0,
            "video_id": "",
            "title": "",
            "step": "",
            "step_index": 0,
            "sources": asdict(VideoSourcesInfo()),
            "extractions": [],
        }
        base["timing"] = {
            "run_elapsed_s": None,
            "current_video_elapsed_s": None,
            "current_step_elapsed_s": None,
            "avg_video_s": None,
            "videos_completed_this_run": 0,
        }
    else:
        base["stale_snapshot"] = False

    return base


def _database_locales_with_confidence(limit: int = 50) -> list[dict]:
    """Merge locales.json + latest visit confidence for the web dashboard."""
    locales = load_json(LOCALES_JSON)
    visits = load_json(VISITS_JSON)
    best_conf: dict[str, float] = {}
    for v in visits:
        lid = v.get("locale_id", "")
        conf = float(v.get("llm_confidence", 0) or 0)
        if lid and conf >= best_conf.get(lid, 0):
            best_conf[lid] = conf
    out: list[dict] = []
    for loc in locales[-limit:]:
        lid = loc.get("locale_id", "")
        out.append(
            {
                "name": loc.get("name", "?"),
                "city": loc.get("city", ""),
                "lat": loc.get("lat"),
                "lon": loc.get("lon"),
                "confidence": best_conf.get(lid),
                "locale_id": lid,
            }
        )
    out.sort(key=lambda x: x.get("confidence") or 0, reverse=True)
    return out
