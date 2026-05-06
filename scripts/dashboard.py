"""
dashboard.py — Live terminal dashboard for the CiboBuono pipeline.

Uses the `rich` library to render a live-updating display that fills the
terminal window.  The dashboard shows:
  • Current phase (Catalog / Processing)
  • Video being processed (title, progress)
  • Step progress within the current video
  • Global statistics (pending, processed, skipped, errored, locales, visits)
  • Elapsed time and ETA

Usage (integrated into run_pipeline.py):
    from scripts.dashboard import Dashboard
    dash = Dashboard()
    dash.start()
    dash.set_phase("Catalog")
    dash.update_video(1, "Video title")
    dash.set_step("Transcribing")
    dash.tick_stat("processed")
    dash.stop()
"""

__author__ = "Luca Ostinelli"

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

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


# ── Pipeline step definitions ──────────────────────────────────────────

PIPELINE_STEPS = [
    "Fetch channels",
    "Catalog videos",
    "Prefetch audio",
    "Download audio",
    "Transcribe",
    "Chunk",
    "Extract (LLM)",
    "Geocode",
    "Verify (OSM)",
    "Deduplicate",
    "Populate",
    "Update status",
]


@dataclass
class DashboardState:
    """All mutable state for the dashboard display."""

    phase: str = "Idle"
    current_video_index: int = 0
    total_videos: int = 0
    current_video_title: str = ""
    current_step: str = ""
    current_step_index: int = 0

    # Counters
    total_in_db: int = 0
    pending: int = 0
    processed: int = 0
    skipped: int = 0
    errored: int = 0
    locales_found: int = 0
    visits_created: int = 0
    flagged: int = 0
    channels: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    video_start_time: float = 0.0

    # Log lines (last N messages)
    log_lines: list[str] = field(default_factory=list)
    max_log_lines: int = 12


class Dashboard:
    """Live terminal dashboard for the CiboBuono pipeline."""

    def __init__(self) -> None:
        self.console = Console()
        self.state = DashboardState()
        self._live: Live | None = None

        # Video-level progress bar
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

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the live display."""
        self.state.start_time = time.time()
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display gracefully."""
        if self._live:
            self._refresh()
            self._live.stop()
            self._live = None



    # ── State mutations ────────────────────────────────────────────────

    def set_phase(self, phase: str) -> None:
        self.state.phase = phase
        self._refresh()

    def set_totals(
        self,
        total_in_db: int = 0,
        pending: int = 0,
        processed: int = 0,
        skipped: int = 0,
        errored: int = 0,
        channels: int = 0,
    ) -> None:
        """Set the global counters read from disk."""
        self.state.total_in_db = total_in_db
        self.state.pending = pending
        self.state.processed = processed
        self.state.skipped = skipped
        self.state.errored = errored
        self.state.channels = channels
        self._refresh()

    def set_video_batch(self, total: int) -> None:
        """Set number of videos to process in this run."""
        self.state.total_videos = total
        self._video_progress.update(self._video_task, total=total, completed=0)
        self._refresh()

    def update_video(self, index: int, title: str) -> None:
        """Signal that we are now processing video #index."""
        self.state.current_video_index = index
        self.state.current_video_title = title
        self.state.current_step = PIPELINE_STEPS[3]  # starts at Download audio
        self.state.current_step_index = 3
        self.state.video_start_time = time.time()
        self._video_progress.update(self._video_task, completed=index - 1)
        self._refresh()

    def set_step(self, step: str) -> None:
        """Update the pipeline step for the current video."""
        self.state.current_step = step
        if step in PIPELINE_STEPS:
            self.state.current_step_index = PIPELINE_STEPS.index(step)
        self._refresh()

    def complete_video(self) -> None:
        """Mark the current video as done (progress bar only; stats refreshed by caller)."""
        self._video_progress.update(
            self._video_task,
            completed=self.state.current_video_index,
        )
        self._refresh()

    def tick_stat(self, stat: str, delta: int = 1) -> None:
        """Increment a counter."""
        cur = getattr(self.state, stat, 0)
        setattr(self.state, stat, cur + delta)
        self._refresh()

    def log(self, message: str) -> None:
        """Append a log line to the dashboard footer."""
        self.state.log_lines.append(message)
        if len(self.state.log_lines) > self.state.max_log_lines:
            self.state.log_lines = self.state.log_lines[-self.state.max_log_lines :]
        self._refresh()

    # ── Layout builder ─────────────────────────────────────────────────

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._build_layout())

    def _build_layout(self) -> Panel:
        """Build the full dashboard layout sized to terminal."""
        cols, rows = os.get_terminal_size()

        # ── Header ────────────────────────────────────────────────────
        header = Text()
        header.append("🍕 CiboBuono Pipeline ", style="bold yellow")
        header.append(f"— {self.state.phase}", style="bold white")
        elapsed = time.time() - self.state.start_time
        header.append(f"  ⏱  {self._fmt_time(elapsed)}", style="dim")

        # ── Statistics table ──────────────────────────────────────────
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(justify="right", style="dim")
        stats_table.add_column(style="bold")
        stats_table.add_column(justify="right", style="dim")
        stats_table.add_column(style="bold")

        s = self.state
        stats_table.add_row(
            "Channels", str(s.channels),
            "Total videos", str(s.total_in_db),
        )
        stats_table.add_row(
            "Pending", f"[cyan]{s.pending}[/]",
            "Processed", f"[green]{s.processed}[/]",
        )
        stats_table.add_row(
            "Skipped", f"[yellow]{s.skipped}[/]",
            "Errored", f"[red]{s.errored}[/]",
        )
        stats_table.add_row(
            "Locales", f"[magenta]{s.locales_found}[/]",
            "Visits", f"[magenta]{s.visits_created}[/]",
        )
        stats_table.add_row(
            "Flagged", f"[yellow]{s.flagged}[/]",
            "", "",
        )

        stats_panel = Panel(
            stats_table,
            title="[bold]Statistics",
            border_style="blue",
            expand=True,
        )

        # ── Current video info ────────────────────────────────────────
        if s.current_video_index > 0:
            title_trunc = s.current_video_title[:cols - 30] if s.current_video_title else "—"
            video_info = Text()
            video_info.append(f"Video {s.current_video_index}/{s.total_videos}: ", style="bold")
            video_info.append(title_trunc, style="italic")
            video_info.append("\n")

            # Step indicator
            for i, step in enumerate(PIPELINE_STEPS[3:], start=3):
                if i < s.current_step_index:
                    video_info.append(f" ✓ {step} ", style="green")
                elif i == s.current_step_index:
                    video_info.append(f" ▸ {step} ", style="bold yellow")
                else:
                    video_info.append(f"   {step} ", style="dim")
                video_info.append("\n")

            if s.video_start_time > 0:
                vt = time.time() - s.video_start_time
                video_info.append(f"\n  Video elapsed: {self._fmt_time(vt)}", style="dim")

            video_panel = Panel(
                video_info,
                title="[bold]Current Video",
                border_style="green",
                expand=True,
            )
        else:
            video_panel = Panel(
                Align.center(Text("Waiting for videos…", style="dim")),
                title="[bold]Current Video",
                border_style="green",
                expand=True,
            )

        # ── Progress bar ──────────────────────────────────────────────
        progress_panel = Panel(
            self._video_progress,
            title="[bold]Batch Progress",
            border_style="cyan",
            expand=True,
        )

        # ── Log tail ─────────────────────────────────────────────────
        max_log = max(4, rows - 28)
        self.state.max_log_lines = max_log
        log_text = Text()
        for line in self.state.log_lines[-max_log:]:
            # Truncate to terminal width
            display = line[: cols - 6]
            log_text.append(f"  {display}\n", style="dim")
        if not self.state.log_lines:
            log_text.append("  (no log messages yet)\n", style="dim italic")

        log_panel = Panel(
            log_text,
            title="[bold]Log",
            border_style="dim",
            expand=True,
        )

        body = Group(
            Align.center(header),
            "",
            stats_panel,
            video_panel,
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
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"
