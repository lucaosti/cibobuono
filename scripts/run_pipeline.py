"""
run_pipeline.py — Main pipeline orchestrator.

All YouTube sources are Italian; downloader, subtitles, Whisper, geocoding labels,
and LLM prompts are tuned for Italian (see scripts.utils.CONTENT_LANGUAGE).

Two-phase design:
  Phase 1 (catalog):
      1. Fetch channels from channels_input.txt
      2. Catalog ALL videos into videos.json with status="pending"
         (recipe videos, Shorts, and non-food videos → skipped_videos.json)
  Phase 2 (process):
      For each pending video (newest first, limited by --max-videos):
      3. Prefetch audio (sliding window of 20 files)
      4. Download audio via yt-dlp
      5. Transcribe: Whisper large-v3-turbo (primary) + YouTube manual subs (when present)
      6. Chunk transcription (90s chunks, 15s overlap)
      6b. Food-relevance gate (LLM classifies: is this a food review video?)
      7. Extract locales with local LLM (GGUF, auto-selected by RAM + video description)
      8. Self-verify extractions (Generate then Verify pattern)
      9. Geocode via Nominatim (free)
      10. Verify locales exist on OpenStreetMap (Overpass API)
      11. Deduplicate (thefuzz + haversine)
      12. Populate visits.json, flagged_segments.json
      13. Update video status to "processed" in videos.json
  Final:
      14. Git commit & push (optional)
      15. GitHub Actions builds React site and deploys to Pages

Usage:
    python -m scripts.run_pipeline --max-videos 10
    python -m scripts.run_pipeline --reset --max-videos 0 --no-dashboard
    python -m scripts.run_pipeline --push --max-videos 10   # also commit+push data to GitHub
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from scripts.chunk_transcription import seconds_to_timestamp as _stt
from scripts.dashboard import Dashboard
from scripts.hardware import get_profile
from scripts.schemas import VideoStatus
from scripts.utils import (
    CACHE_DIR,
    CHANNELS_JSON,
    CORRECTIONS_JSON,
    DATA_DIR,
    FLAGGED_SEGMENTS_JSON,
    LOCALES_JSON,
    LOGS_DIR,
    PREFETCH_WINDOW,
    PROCESSED_VIDEOS_JSON,
    SKIPPED_VIDEOS_JSON,
    VIDEOS_JSON,
    VISITS_JSON,
    WATCH_MIN_INTERVAL_SECONDS,
    WATCH_POLL_INTERVAL_SECONDS,
    WHISPER_DEFAULT_MODEL,
    cleanup_cache,
    ensure_dirs,
    load_json,
    save_json,
    setup_logging,
    today_str,
)

logger = setup_logging("pipeline")

# First SIGINT/SIGTERM: finish current video, then stop. Second: KeyboardInterrupt.
_pipeline_shutdown = {"graceful": False}

_prefetch_executor: ThreadPoolExecutor | None = None


def _prefetch_pool() -> ThreadPoolExecutor:
    global _prefetch_executor
    if _prefetch_executor is None:
        workers = int(os.environ.get("CIBOBUONO_IO_WORKERS", "4"))
        _prefetch_executor = ThreadPoolExecutor(
            max_workers=max(2, workers), thread_name_prefix="prefetch"
        )
    return _prefetch_executor


def _schedule_audio_prefetch(
    videos: list[dict], start_index: int, *, count: int = 3
) -> None:
    """Download audio for upcoming videos while GPU works on the current one."""
    from scripts.fetch_videos import download_audio

    pool = _prefetch_pool()
    for j in range(start_index, min(start_index + count, len(videos))):
        v = videos[j]
        pool.submit(download_audio, v["video_id"], v["url"])
_sig_previous: dict[int, object] = {}


def _install_pipeline_signal_handlers() -> None:
    def handler(signum, frame):
        if _pipeline_shutdown["graceful"]:
            logger.warning(
                "Second interrupt: aborting immediately "
                "(run --repair-stale-state if visits look inconsistent)"
            )
            raise KeyboardInterrupt
        _pipeline_shutdown["graceful"] = True
        logger.warning(
            "Interrupt: will stop after the current video. "
            "Interrupt again to quit immediately."
        )

    _sig_previous[signal.SIGINT] = signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        _sig_previous[signal.SIGTERM] = signal.signal(signal.SIGTERM, handler)


def _restore_pipeline_signal_handlers() -> None:
    for sig, prev in list(_sig_previous.items()):
        signal.signal(sig, prev)
    _sig_previous.clear()


def _count_videos_by_status() -> dict:
    """Count videos in each status from videos.json."""
    videos = load_json(VIDEOS_JSON)
    counts = {"total": len(videos), "pending": 0, "processed": 0, "errored": 0}
    for v in videos:
        s = v.get("status", "pending")
        if s in counts:
            counts[s] += 1
    return counts


def run_pipeline(
    skip_fetch: bool = False,
    skip_transcribe: bool = False,
    skip_extract: bool = False,
    skip_push: bool = True,
    whisper_model: str = WHISPER_DEFAULT_MODEL,
    max_videos: int = 100,
    no_dashboard: bool = False,
    auto_models: bool = True,
    parallel_postprocess: bool | None = None,
    _external_setup: bool = False,
):
    """
    Run the full pipeline (one shot).

    Args:
        skip_fetch: Skip cataloging new videos (work only on existing pending)
        skip_transcribe: Skip transcription (use cached transcripts)
        skip_extract: Skip LLM extraction (useful for testing geocoding/dedup)
        skip_push: Skip git commit and push (default True — pass --push to enable)
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        max_videos: Max pending videos to process in this run (0 = all pending)
        no_dashboard: Disable live dashboard (log-only mode)
        _external_setup: Internal flag; True when invoked from run_pipeline_watch
            so signal handlers and the shutdown flag are managed by the caller.
    """
    ensure_dirs()

    if not _external_setup:
        _pipeline_shutdown["graceful"] = False
        _install_pipeline_signal_handlers()

    # Yield CPU to interactive work; keeps the OS responsive on shared machines.
    from scripts import resource_monitor as rm
    rm.apply_friendly_priority()

    # ── Hardware profile + auto-select models ─────────────────────────
    # get_profile() already logs the detected capacity once; no need to repeat it.
    from scripts.utils import select_optimal_models

    profile = get_profile()
    logger.info("Live resources at start: %s", rm.snapshot(include_gpu=profile.has_cuda).summary())

    if parallel_postprocess is None:
        from scripts.pipeline_executor import parallel_postprocess_enabled

        parallel_postprocess = parallel_postprocess_enabled(profile.has_cuda)
    if parallel_postprocess and profile.has_cuda:
        logger.info(
            "Parallel postprocess enabled — geocode/OSM runs while GPU processes next video"
        )

    if auto_models:
        selected = select_optimal_models()
        whisper_model = selected["whisper_model"]
        chosen_llm = selected["llm_model_path"]
        logger.info(
            f"Auto-selected models: Whisper={whisper_model}, "
            f"LLM={chosen_llm.name if chosen_llm else 'none'} ({selected['llm_tier']})"
        )
        if chosen_llm:
            import os as _os
            _os.environ.setdefault("CIBOBUONO_LLM_MODEL", str(chosen_llm))

    # ── Graceful LLM degradation on low-end hardware ──────────────────
    if not skip_extract and not profile.enable_llm:
        logger.warning(
            "Hardware below LLM threshold (%.1f GB RAM, %s) — forcing "
            "--skip-extract automatically. Extraction will use NER + rules "
            "only and quality will be lower.",
            profile.total_ram_gb, profile.platform.value,
        )
        skip_extract = True

    # ── Dashboard (terminal UI optional; JSON snapshot always on) ───
    dash = Dashboard(live=not no_dashboard)
    use_dash = not no_dashboard

    if use_dash:
        dash.start()

    def _log(msg: str) -> None:
        logger.info(msg)
        dash.log(msg)

    # Pre-load Whisper once so the first video does not wait on model load.
    if not skip_transcribe:
        from scripts.transcribe_video import _get_whisper_model

        _log("Pre-loading Whisper model…")
        _get_whisper_model(whisper_model)

    # Snapshot counts once at the start of the run; update incrementally
    # instead of re-reading all JSON files after every video.
    _stats: dict = {
        "total": 0, "pending": 0, "processed": 0, "errored": 0,
        "skipped": len(load_json(SKIPPED_VIDEOS_JSON)),
        "channels": len(load_json(CHANNELS_JSON)),
        "locales_found": len(load_json(LOCALES_JSON)),
        "visits_created": len(load_json(VISITS_JSON)),
        "flagged": len(load_json(FLAGGED_SEGMENTS_JSON)),
    }
    _vc = _count_videos_by_status()
    _stats.update(_vc)

    def _refresh_stats(
        *,
        delta_pending: int = 0,
        delta_processed: int = 0,
        delta_errored: int = 0,
        delta_visits: int = 0,
        delta_flagged: int = 0,
        delta_locales: int = 0,
    ) -> None:
        _stats["pending"] += delta_pending
        _stats["processed"] += delta_processed
        _stats["errored"] += delta_errored
        _stats["visits_created"] += delta_visits
        _stats["flagged"] += delta_flagged
        _stats["locales_found"] += delta_locales
        dash.set_totals(
            total_in_db=_stats["total"],
            pending=_stats["pending"],
            processed=_stats["processed"],
            errored=_stats["errored"],
            skipped=_stats["skipped"],
            channels=_stats["channels"],
            locales_found=_stats["locales_found"],
            visits_created=_stats["visits_created"],
            flagged=_stats["flagged"],
        )

    def _dash_finish(
        outcome: str,
        *,
        visits: int = 0,
        flagged: int = 0,
    ) -> None:
        dash.complete_video(outcome=outcome, visits=visits, flagged=flagged)
        if outcome in ("processed", "processed_empty"):
            _refresh_stats(delta_pending=-1, delta_processed=1, delta_visits=visits, delta_flagged=flagged)
        elif outcome == "errored":
            _refresh_stats(delta_pending=-1, delta_errored=1)
        else:
            _refresh_stats(delta_pending=-1, delta_processed=1)

    run_report: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "videos": [],
    }
    executor = None

    try:
        _log("Pipeline started")
        from scripts import pipeline_control as pc

        pc.mark_started(pid=os.getpid(), max_videos=max_videos)
        dash.set_phase("Phase 1 — Catalog")

        dash.set_step("Fetch channels")
        _log("Fetching channels…")
        from scripts.fetch_channels import fetch_channels
        new_channels = fetch_channels()
        _log(f"{len(new_channels)} new channels")
        if new_channels:
            _stats["channels"] += len(new_channels)
        _refresh_stats()

        if not skip_fetch:
            dash.set_step("Catalog videos")
            _log("Cataloging channel videos…")
            from scripts.fetch_videos import catalog_channel_videos
            n = catalog_channel_videos()
            _log(f"{n} new videos cataloged")
            # Catalog can add many videos — re-read counts once after Phase 1.
            _vc2 = _count_videos_by_status()
            _stats.update(_vc2)
            _stats["skipped"] = len(load_json(SKIPPED_VIDEOS_JSON))
            _refresh_stats()
        else:
            _log("Catalog skipped (--skip-fetch)")

        dash.set_phase("Phase 2 — Processing")
        _refresh_stats()

        from scripts.fetch_videos import (
            download_audio,
            fetch_video_upload_date,
            get_pending_videos,
            update_video_status,
        )

        def _err(vid: str, ch: str, pub: str, step: str, recap: dict) -> None:
            """Mark a video errored, update status/processed/dashboard."""
            update_video_status(vid, VideoStatus.ERRORED, pub)
            _update_processed(vid, ch, VideoStatus.ERRORED)
            recap.update({"outcome": "errored", "step": step})
            dash.tick_stat("errored")
            _dash_finish("errored")

        def _skip(vid: str, ch: str, pub: str, outcome: str, recap: dict) -> None:
            """Mark a video as processed/skipped, update status/processed/dashboard."""
            update_video_status(vid, VideoStatus.PROCESSED, pub)
            _update_processed(vid, ch, VideoStatus.PROCESSED, 0, 0)
            recap["outcome"] = outcome
            dash.tick_stat("processed")
            _dash_finish("processed")

        pending = get_pending_videos()
        to_process = pending if max_videos <= 0 else pending[:max_videos]

        if _pipeline_shutdown["graceful"]:
            _log("Graceful shutdown requested before Phase 2; exiting")
            return

        _log(
            f"Phase 2: {len(pending)} pending videos, "
            f"processing {len(to_process)} (newest first)"
        )

        if not to_process:
            _log("No pending videos to process")
            if not skip_push:
                if _git_push_if_changed():
                    _log("Pushed to GitHub")
            _log("Pipeline complete (no pending content)")
            return

        if not skip_extract:
            from scripts.utils import resolve_llm_model_path
            mp = resolve_llm_model_path()
            if mp is None:
                if profile.enable_llm:
                    _log(
                        "No GGUF model found but hardware supports LLM. "
                        "Add a .gguf under models/ (recommended tier: "
                        f"{profile.llm_tier}) or set CIBOBUONO_LLM_MODEL. "
                        "Falling back to --skip-extract for this run."
                    )
                else:
                    _log(
                        "No GGUF model found and hardware below LLM threshold "
                        "— running with --skip-extract."
                    )
                skip_extract = True

        dash.set_video_batch(len(to_process))

        stopped_gracefully = False

        from scripts.pipeline_executor import (
            FinalizeJob,
            PipelineExecutor,
            parallel_whisper_enabled,
        )

        io_workers = int(os.environ.get("CIBOBUONO_IO_WORKERS", "4"))
        _par_whisper = (
            parallel_whisper_enabled(profile.has_cuda, profile.gpu_vram_gb)
            and not skip_transcribe
            and not skip_extract
        )
        if _par_whisper:
            _log(
                f"Parallel Whisper enabled (VRAM={profile.gpu_vram_gb} GB) — "
                "Whisper(N+1) will run alongside LLM(N)"
            )
        executor = PipelineExecutor(
            parallel_postprocess=parallel_postprocess and not skip_extract,
            parallel_whisper=_par_whisper,
            io_workers=io_workers,
        )

        # Prefetch intel for the first video while initial audio downloads run.
        if not skip_extract and to_process:
            v0 = to_process[0]
            executor.schedule_intel_prep(v0["video_id"], v0.get("title", v0["video_id"]))

        # ── Sliding window: pre-download audio for first batch ────────
        prefetch_n = min(PREFETCH_WINDOW, len(to_process))
        _log(f"Prefetch: downloading audio for first {prefetch_n} videos...")
        dash.set_step("Prefetch audio")
        for pv in to_process[:prefetch_n]:
            audio = download_audio(pv["video_id"], pv["url"])
            if not audio:
                _log(f"  ✗ Prefetch failed: {pv['title'][:55]}")
        _log(f"Prefetch complete: {prefetch_n} audio files ready")

        # Load channel info for rubrica inference
        channels = load_json(CHANNELS_JSON)
        channel_map = {ch["channel_id"]: ch for ch in channels}

        from scripts import pipeline_control as pc

        for i, video in enumerate(to_process, 1):
            if pc.read_state().get("stop_requested"):
                _pipeline_shutdown["graceful"] = True
            if _pipeline_shutdown["graceful"]:
                stopped_gracefully = True
                _log("Graceful shutdown: not starting another video")
                break
            if not pc.wait_if_paused(should_abort=lambda: _pipeline_shutdown["graceful"]):
                _pipeline_shutdown["graceful"] = True
                stopped_gracefully = True
                _log("Stop requested via dashboard")
                break

            if executor:
                for fr in executor.poll_completed():
                    if fr.outcome == "processed":
                        dash.tick_stat("processed")
                        _refresh_stats(delta_visits=fr.visits_created, delta_flagged=fr.flagged_segments)
                    elif fr.outcome == "errored":
                        dash.tick_stat("errored")
                        _log(f"  ✗ Background finalize failed: {fr.error[:200]}")
                        _refresh_stats()

            if executor:
                executor.schedule_io_prep(to_process, i - 1, count=4)
                if not skip_extract and i < len(to_process):
                    nv = to_process[i]
                    executor.schedule_intel_prep(
                        nv["video_id"], nv.get("title", nv["video_id"])
                    )

            if _pipeline_shutdown["graceful"]:
                stopped_gracefully = True
                _log("Graceful shutdown: not starting another video")
                break

            # Back-pressure: before taking on another (heavy) video, make sure
            # the system isn't already saturated. Wait briefly for it to calm
            # down; if it doesn't, proceed anyway (the per-model loaders apply
            # their own adaptive limits) rather than stalling forever.
            stressed, why = rm.under_pressure(include_gpu=profile.has_cuda)
            if stressed:
                _log(f"  System under pressure ({why}); pausing before next video…")
                rm.wait_until_calm(
                    include_gpu=profile.has_cuda,
                    should_abort=lambda: _pipeline_shutdown["graceful"],
                )

            recap: dict = {
                "video_id": video["video_id"],
                "title": (video.get("title") or "")[:200],
            }
            food_confirmed_by_rules = False
            try:
                # Prefetch upcoming audio while this video is processed (I/O overlap).
                _schedule_audio_prefetch(to_process, i - 1, count=3)

                # Sliding window: trim cache to window size
                deleted = cleanup_cache(PREFETCH_WINDOW)
                if deleted:
                    _log(f"  Cache trimmed: {len(deleted)} old files removed")

                video_id = video["video_id"]
                channel_id = video["channel_id"]
                channel_info = channel_map.get(channel_id, {})
                channel_rubriche = channel_info.get("rubriche", [])
                v_title = video.get("title", video_id)

                dash.update_video(i, v_title, video_id=video_id)

                _log(f"[{i}/{len(to_process)}] {v_title[:70]}")

                # Resolve publish_date if not yet known
                publish_date = video.get("publish_date", "")
                if not publish_date:
                    _log(f"  Resolving publish date for {video_id}...")
                    publish_date = fetch_video_upload_date(video_id)
                    if not publish_date:
                        publish_date = today_str()

                # Step 3: Download audio
                dash.set_step("Download audio")
                _log("  Downloading audio...")
                audio_path = download_audio(video_id, video["url"])
                if not audio_path:
                    _log(f"  ✗ Audio download failed for {video_id}")
                    _err(video_id, channel_id, publish_date, "download_audio", recap)
                    continue

                # Step 3.5: Metadata + video intelligence (pre-Whisper filters)
                # Fetch title/description intel and run cheap skip checks before
                # spending minutes on transcription.
                video_intel = None
                youtube_extra: dict | None = None
                video_description: str | None = None

                if not skip_extract:
                    dash.set_step("Video intel")
                    from scripts.fetch_videos import detect_non_food_video

                    intel = executor.take_intel_prep(video_id, v_title)
                    if intel.error and intel.video_intel is None:
                        _log(f"  Intel prep error: {intel.error[:120]}")
                    video_intel = intel.video_intel
                    youtube_extra = intel.youtube_extra
                    video_description = intel.video_description or ""
                    comments: list = []

                    chapters = (youtube_extra or {}).get("chapters") or []
                    dts = (youtube_extra or {}).get("description_timestamps") or []
                    _log(
                        f"  Intel: type={video_intel.video_type if video_intel else '?'}"
                        f", city={video_intel.city if video_intel else ''}"
                        f", hints={len(video_intel.venue_hints) if video_intel else 0}"
                        f", desc={len(video_description)} ch"
                        f", chapters={len(chapters)}, ts={len(dts)}"
                        f", comments={intel.comments_count}"
                    )

                    # Title-based skip before Whisper
                    if video_intel and video_intel.video_type == "non_review" and video_intel.skip_reason:
                        _log(f"  ✗ Skipped (non-review): {video_intel.skip_reason}")
                        _skip(video_id, channel_id, publish_date, "skipped_non_review", recap)
                        continue

                    # Description-based non-food skip before Whisper
                    is_nf, nf_reason = detect_non_food_video(v_title, video_description)
                    if is_nf:
                        _log(f"  ✗ Skipped (non-food): {nf_reason}")
                        _skip(video_id, channel_id, publish_date, "skipped_non_food", recap)
                        continue

                    if not skip_extract:
                        from scripts.extract_locales import check_food_video

                        is_food_pre, food_pre_reason = check_food_video(
                            v_title,
                            video_description=video_description or "",
                            video_intel=video_intel,
                        )
                        if food_pre_reason.startswith("Rules:"):
                            food_confirmed_by_rules = True
                            _log(f"  Food-check (pre-transcript): {food_pre_reason}")

                # Step 4: Transcribe
                dash.set_step("Transcribe")
                if not skip_transcribe:
                    # Parallel mode: both Whisper and LLM stay in VRAM together.
                    # Sequential mode: release LLM first so Whisper can use VRAM.
                    if profile.has_cuda and not _par_whisper:
                        from scripts.extract_locales import release_llm

                        release_llm()
                    _log("  Transcribing...")
                    transcript = executor.take_transcript_prefetch(video_id, whisper_model)
                    if not transcript:
                        _log(f"  ✗ Transcription failed for {video_id}")
                        _err(video_id, channel_id, publish_date, "transcribe", recap)
                        continue
                else:
                    _log("  Transcription skipped (cached)")
                    transcript_path = CACHE_DIR / f"{video_id}_transcript.json"
                    if transcript_path.exists():
                        with open(transcript_path, "r", encoding="utf-8") as f:
                            transcript = json.load(f)
                    else:
                        _log(f"  ✗ No cached transcript for {video_id}")
                        _err(video_id, channel_id, publish_date, "transcript_cache_missing", recap)
                        continue

                dash.set_video_sources(
                    transcript_source=transcript.get("source", "whisper"),
                    transcript_chars=len(transcript.get("text", "")),
                )

                # Hand VRAM from Whisper to LLM (sequential) -OR- submit the
                # next Whisper to a background thread while keeping both models
                # in VRAM (parallel mode).
                if not skip_transcribe:
                    if _par_whisper:
                        # Schedule Whisper for the next video while LLM runs now.
                        if i < len(to_process):
                            next_vid = to_process[i]
                            executor.schedule_transcript_prefetch(
                                next_vid["video_id"], whisper_model
                            )
                    else:
                        from scripts.transcribe_video import release_whisper_model

                        release_whisper_model()
                        if not skip_extract and profile.has_cuda:
                            from scripts.extract_locales import preload_llm

                            preload_llm(executor._io_pool if executor else None)

                # Step 5: Chunk
                dash.set_step("Chunk")
                _log("  Chunking transcription...")
                from scripts.chunk_transcription import chunk_transcription
                chunks = chunk_transcription(transcript)
                _log(f"  {len(chunks)} chunks created")

                if not chunks:
                    _log(f"  ✗ No chunks for {video_id}")
                    _err(video_id, channel_id, publish_date, "chunk", recap)
                    continue

                # Step 6: Extract locales with LLM
                dash.set_step("Extract (LLM)")
                if not skip_extract:
                    _log("  Extracting locales with LLM...")
                    from scripts.extract_locales import check_food_video
                    from scripts.extract_pipeline import extract_from_video

                    transcript_text = transcript.get("text", "")

                    if food_confirmed_by_rules:
                        food_reason = "Rules: confirmed pre-transcript"
                        _log(f"  Food-check: {food_reason}")
                    else:
                        is_food, food_reason = check_food_video(
                            v_title,
                            transcript_text,
                            video_description or "",
                            video_intel=video_intel,
                        )
                        if not is_food:
                            _log(f"  ✗ Skipped (food-check): {food_reason}")
                            _skip(video_id, channel_id, publish_date, "skipped_food_gate", recap)
                            continue
                        _log(f"  Food-check: {food_reason}")

                    dash.set_video_sources(food_gate=food_reason[:120])

                    extractions, flagged_extractions = extract_from_video(
                        video_id,
                        chunks,
                        channel_rubriche,
                        video_description,
                        video_title=v_title,
                        video_intel=video_intel,
                        youtube_extra=youtube_extra,
                        transcript=transcript,
                    )
                    _log(f"  {len(extractions)} locales, {len(flagged_extractions)} flagged")
                else:
                    _log("  Extraction skipped")
                    extractions = []
                    flagged_extractions = []

                # Apply title rating to extractions that don't have one
                if video_intel and video_intel.title_rating:
                    for e in extractions:
                        if not e.get("rating"):
                            e["rating"] = video_intel.title_rating
                            _log(f"  Applied title rating '{video_intel.title_rating}' to '{e.get('locale_name')}'")

                # Flag strong disagreement between title rating and transcript rating
                if video_intel and video_intel.title_rating and extractions:
                    from scripts.extract_locales import rating_numeric_core
                    tr_num = rating_numeric_core(video_intel.title_rating)
                    if tr_num is not None:
                        for e in extractions:
                            er_num = rating_numeric_core(e.get("rating"))
                            if er_num is None:
                                continue
                            if abs(er_num - tr_num) >= 2.0:
                                _log(
                                    f"  ⚠ Rating mismatch title={video_intel.title_rating} "
                                    f"vs extraction={e.get('rating')} ({e.get('locale_name')})"
                                )
                                fe = dict(e)
                                fe["_flag_reason"] = "rating_title_transcript_mismatch"
                                fe["confidence"] = min(float(e.get("confidence", 0.5)), 0.55)
                                flagged_extractions.append(fe)

                # Correct timestamps: scan full transcript for best mention time
                from scripts.extract_locales import find_best_timestamp_in_transcript

                for e in extractions:
                    name = e.get("locale_name", "")
                    best_ts = find_best_timestamp_in_transcript(name, transcript)
                    if best_ts is not None:
                        old_ts = e.get("mention_time", e.get("chunk_start_seconds", 0))
                        e["mention_time"] = best_ts
                        e["mention_timestamp"] = _stt(best_ts)
                        e["chunk_start_seconds"] = best_ts
                        e["chunk_start"] = _stt(best_ts)
                        end_ts = best_ts + 90
                        e["chunk_end"] = _stt(end_ts)
                        _log(f"  Timestamp corrected for '{name}': {old_ts:.0f}s → {best_ts:.0f}s")

                dash.set_extractions(extractions, flagged_extractions)

                if not extractions and not flagged_extractions:
                    _log(f"  No locales found in {video_id}")
                    update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
                    _update_processed(video_id, channel_id, VideoStatus.PROCESSED, 0, 0)
                    recap["outcome"] = "processed_empty"
                    _dash_finish("processed_empty")
                    continue

                # Build set of trusted venue names. Only STRUCTURED sources
                # (title/chapter) are trusted enough to bypass geocoding/OSM
                # verification — description/comment hints are too noisy.
                trusted_venue_names: set[str] = set()
                if video_intel:
                    for hint in video_intel.venue_hints:
                        if hint.get("source") not in ("title", "chapter"):
                            continue
                        if hint.get("confidence") != "very_high":
                            continue
                        trusted_venue_names.add(hint["name"].lower().strip())

                # Steps 7–10: geocode / OSM / dedupe / populate (overlap with next GPU work)
                finalize_job = FinalizeJob(
                    video_id=video_id,
                    channel_id=channel_id,
                    publish_date=publish_date,
                    extractions=extractions,
                    flagged_extractions=flagged_extractions,
                    trusted_venue_names=trusted_venue_names,
                    intel_city=(video_intel.city if video_intel else ""),
                )

                if executor and executor.parallel_postprocess:
                    dash.set_step("Finalize (background)")
                    _log(
                        "  Geocode/OSM/populate queued in background — "
                        "GPU free for next video"
                    )
                    executor.submit_finalize(finalize_job, _log)
                    recap["outcome"] = "processed"
                    recap["visits_created"] = len(extractions)
                    recap["flagged_segments"] = len(flagged_extractions)
                    _dash_finish(
                        "processed",
                        visits=len(extractions),
                        flagged=len(flagged_extractions),
                    )
                else:
                    from scripts.pipeline_executor import finalize_video

                    dash.set_step("Geocode")
                    result = finalize_video(finalize_job, _log)
                    recap["outcome"] = result.outcome
                    recap["visits_created"] = result.visits_created
                    recap["flagged_segments"] = result.flagged_segments
                    if result.outcome == "errored":
                        dash.tick_stat("errored")
                        _dash_finish("errored")
                    else:
                        _log(
                            f"  ✓ Done: {result.visits_created} visits, "
                            f"{result.flagged_segments} flagged"
                        )
                        _dash_finish(
                            "processed",
                            visits=result.visits_created,
                            flagged=result.flagged_segments,
                        )

                # Sliding window: prefetch next video outside current window
                next_prefetch = i - 1 + PREFETCH_WINDOW  # i is 1-based
                if next_prefetch < len(to_process):
                    nv = to_process[next_prefetch]
                    download_audio(nv["video_id"], nv["url"])
            finally:
                recap.setdefault("outcome", "unknown")
                run_report["videos"].append(recap)

        finalize_results: list = []
        if executor:
            finalize_results = executor.drain_finalize(_log)
            for fr in executor.poll_completed():
                finalize_results.append(fr)
                if fr.outcome == "processed":
                    dash.tick_stat("processed")
                    _refresh_stats(delta_visits=fr.visits_created, delta_flagged=fr.flagged_segments)
                elif fr.outcome == "errored":
                    dash.tick_stat("errored")
                    _refresh_stats()

        try:
            from scripts.pipeline_metrics import record_run_metrics
            record_run_metrics(
                finalize_results,
                run_id=run_report["started_at"],
            )
        except Exception as exc:
            logger.warning("Could not record pipeline metrics: %s", exc)

        run_report["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_report["summary"] = {
            "videos_attempted": len(run_report["videos"]),
            "processed": sum(1 for v in run_report["videos"] if v.get("outcome") == "processed"),
            "errored": sum(1 for v in run_report["videos"] if v.get("outcome") == "errored"),
            "stopped_gracefully": stopped_gracefully,
        }
        report_path = (
            LOGS_DIR
            / f"run_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        try:
            ensure_dirs()
            with open(report_path, "w", encoding="utf-8") as rf:
                json.dump(run_report, rf, ensure_ascii=False, indent=2)
            _log(f"Run report written to {report_path}")
        except OSError as e:
            logger.warning("Could not write run report: %s", e)

        # Final cache cleanup
        deleted = cleanup_cache(PREFETCH_WINDOW)
        if deleted:
            _log(f"Cleaned up {len(deleted)} old cached files")

        # Git push (only when something under data/ actually changed)
        if not skip_push:
            pushed = _git_push_if_changed()
            if pushed:
                _log("Pushed to GitHub")
            else:
                _log("No data changes — git push skipped")
        else:
            _log("Git push skipped (use --push to enable)")

        dash.set_phase("Complete ✓")
        _refresh_stats()

        _log("Pipeline complete!")

    finally:
        if executor is not None:
            executor.shutdown()
        global _prefetch_executor
        if _prefetch_executor is not None:
            _prefetch_executor.shutdown(wait=False, cancel_futures=True)
            _prefetch_executor = None
        if not _external_setup:
            _restore_pipeline_signal_handlers()
        try:
            from scripts import pipeline_control as pc

            pc.mark_finished()
        except Exception:
            pass
        dash.reset_to_idle()
        if use_dash:
            dash.stop()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_processed(
    video_id: str,
    channel_id: str,
    status: VideoStatus,
    visits: int = 0,
    flagged: int = 0,
):
    """Write entry to processed_videos.json."""
    from scripts.populate_json import update_processed_videos
    update_processed_videos(video_id, channel_id, status, visits, flagged)


def _git_push():
    logger.info("Committing and pushing to GitHub...")
    from scripts.push_to_github import git_commit_and_push
    git_commit_and_push()


def _data_dir_has_uncommitted_changes() -> bool:
    """Return True iff `git status --porcelain` reports any change under data/."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(DATA_DIR)],
            capture_output=True,
            text=True,
            cwd=str(DATA_DIR.parent),
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("git status check failed (%s); assuming changes exist", e)
        return True

    if result.returncode != 0:
        logger.warning(
            "git status returned %d; assuming changes exist. stderr=%s",
            result.returncode, result.stderr[:200],
        )
        return True

    return bool(result.stdout.strip())


def _git_push_if_changed() -> bool:
    """Commit and push only if there are uncommitted changes under data/.

    Returns True if a push attempt was made, False if nothing changed.
    """
    if not _data_dir_has_uncommitted_changes():
        return False
    _git_push()
    return True


def _interruptible_sleep(total_seconds: float, slice_seconds: float = 5.0) -> None:
    """Sleep up to total_seconds, returning early if a graceful shutdown is requested.

    Polling in short slices keeps SIGINT/SIGTERM response time bounded.
    """
    import time

    remaining = max(0.0, float(total_seconds))
    while remaining > 0:
        if _pipeline_shutdown["graceful"]:
            return
        chunk = min(slice_seconds, remaining)
        time.sleep(chunk)
        remaining -= chunk


def run_pipeline_watch(
    poll_interval: float = WATCH_POLL_INTERVAL_SECONDS,
    skip_fetch: bool = False,
    skip_transcribe: bool = False,
    skip_extract: bool = False,
    skip_push: bool = True,
    whisper_model: str = WHISPER_DEFAULT_MODEL,
    max_videos: int = 100,
    auto_models: bool = True,
) -> None:
    """Run the pipeline continuously: catalog + process + push + sleep, repeat.

    Each cycle invokes :func:`run_pipeline` exactly as a one-shot run would, so
    every guarantee (atomic JSON writes, --repair-stale-state, sliding window
    cache cleanup, model caching) still holds.

    Sleep is interruptible: SIGINT/SIGTERM stops the loop at the cycle boundary
    (or at the next 5 s sleep slice). A second signal aborts the in-flight
    cycle immediately, as in one-shot mode.

    Models (Whisper, NER, LLM) are loaded lazily on the first cycle and reused
    across cycles via the module-level caches in transcribe_video / extract_locales /
    ner_candidates — no per-cycle memory growth.
    """
    poll_interval = max(float(poll_interval), float(WATCH_MIN_INTERVAL_SECONDS))

    logger.info(
        f"Starting watch mode: poll_interval={poll_interval:.0f}s, "
        f"max_videos_per_cycle={max_videos}, skip_push={skip_push}"
    )

    _pipeline_shutdown["graceful"] = False
    _install_pipeline_signal_handlers()

    cycle_n = 0
    try:
        while not _pipeline_shutdown["graceful"]:
            cycle_n += 1
            logger.info("── Watch cycle #%d starting ──", cycle_n)
            cycle_started_at = datetime.now(timezone.utc)

            try:
                run_pipeline(
                    skip_fetch=skip_fetch,
                    skip_transcribe=skip_transcribe,
                    skip_extract=skip_extract,
                    skip_push=skip_push,
                    whisper_model=whisper_model,
                    max_videos=max_videos,
                    no_dashboard=True,  # watch mode is log-only
                    auto_models=auto_models,
                    _external_setup=True,
                )
            except SystemExit:
                raise
            except Exception as e:
                logger.exception(f"Watch cycle #{cycle_n} crashed: {e}")

            elapsed = (datetime.now(timezone.utc) - cycle_started_at).total_seconds()
            logger.info(
                f"── Watch cycle #{cycle_n} finished in {elapsed:.0f}s ──"
            )

            if _pipeline_shutdown["graceful"]:
                break

            logger.info("Sleeping %.0fs before next cycle...", poll_interval)
            _interruptible_sleep(poll_interval)

        logger.info(f"Watch mode stopped after {cycle_n} cycle(s)")
    finally:
        _restore_pipeline_signal_handlers()


def _print_status():
    """Print a summary of the current pipeline state and exit."""
    videos = load_json(VIDEOS_JSON)
    locales = load_json(LOCALES_JSON)
    visits = load_json(VISITS_JSON)
    flagged = load_json(FLAGGED_SEGMENTS_JSON)
    skipped = load_json(SKIPPED_VIDEOS_JSON)
    processed_log = load_json(PROCESSED_VIDEOS_JSON)
    channels = load_json(CHANNELS_JSON)

    by_status = {s: [v for v in videos if v.get("status") == s] for s in ("pending", "processed", "errored")}
    print(f"\n{'='*50}\n  CiboBuono Pipeline — Status\n{'='*50}")
    print(f"  Channels: {len(channels)}  Videos: {len(videos)}"
          f"  (pending={len(by_status['pending'])}, processed={len(by_status['processed'])}, errored={len(by_status['errored'])})")
    print(f"  Skipped: {len(skipped)}  Locales: {len(locales)}  Visits: {len(visits)}  Flagged: {len(flagged)}\n")

    for v in by_status["processed"]:
        pv = next((p for p in processed_log if p["video_id"] == v["video_id"]), {})
        print(f"  [{v.get('publish_date','')}] {v['title'][:60]}  "
              f"(visits={pv.get('visits_extracted',0)}, flagged={pv.get('flagged_segments',0)})")
    if by_status["processed"]:
        print()
    for loc in locales:
        print(f"  {loc['name']} | {loc.get('city','')} | ({loc.get('lat','')}, {loc.get('lon','')})")
    loc_by_id = {l["locale_id"]: l["name"] for l in locales}
    for vis in visits:
        print(f"  {loc_by_id.get(vis['locale_id'],'?')} | {vis['sentiment']} | rating={vis['rating']} | video={vis['video_id']}")
    for f in flagged:
        print(f"  {f.get('locale_name','?')} | reason={f.get('reason','')} | video={f['video_id']}")
    print()


def _reset_all(reset_all_data: bool = False):
    """Reset all data, cache, and logs for a fresh start.

    By default, corrections.json is preserved (manual overrides for the site).
    Pass reset_all_data=True to clear corrections as well.
    """
    print("Resetting all data, cache, and logs...")

    # Reset JSON data files to empty arrays
    json_files = [
        CHANNELS_JSON, VIDEOS_JSON, LOCALES_JSON, VISITS_JSON,
        FLAGGED_SEGMENTS_JSON, SKIPPED_VIDEOS_JSON, PROCESSED_VIDEOS_JSON,
    ]

    for f in json_files:
        save_json(f, [])

    if reset_all_data:
        save_json(CORRECTIONS_JSON, [])
        print("Also cleared corrections.json")

    # Clear cache
    if CACHE_DIR.exists():
        for item in CACHE_DIR.iterdir():
            if item.is_file():
                item.unlink()

    # Clear logs
    if LOGS_DIR.exists():
        for item in LOGS_DIR.iterdir():
            if item.is_file():
                item.unlink()

    print("Reset complete: all data, cache, and logs cleared.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CiboBuono pipeline — extract locale reviews from YouTube videos"
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip cataloging new videos (process existing pending only)",
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true",
        help="Skip transcription (use cached transcripts)",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip LLM extraction",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Commit and push data/*.json to GitHub after the run (default: off)",
    )
    parser.add_argument(
        "--whisper-model", default=None,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"],
        help="Whisper model size (default: auto-selected based on hardware)",
    )
    parser.add_argument(
        "--no-auto-models", action="store_true",
        help="Disable automatic model selection; use --whisper-model and CIBOBUONO_LLM_MODEL env instead",
    )
    parser.add_argument(
        "--max-videos", type=int, default=100,
        help="Max pending videos per run (default: 100); 0 means all pending",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable live terminal dashboard (log-only mode)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset pipeline JSON, cache, and logs before running (keeps corrections unless --reset-all-data)",
    )
    parser.add_argument(
        "--reset-all-data", action="store_true",
        help="With --reset, also clear corrections.json",
    )
    parser.add_argument(
        "--repair-stale-state", action="store_true",
        help="Mark pending videos that already have visits/flagged as processed, then exit",
    )
    parser.add_argument(
        "--repair-dry-run", action="store_true",
        help="With --repair-stale-state, only print what would change",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show pipeline status summary and exit (no processing)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Continuous mode: keep cataloging+processing new videos in a loop "
             "(SIGINT/SIGTERM stops gracefully between cycles)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=WATCH_POLL_INTERVAL_SECONDS,
        help=f"With --watch, seconds between cycles (default: {WATCH_POLL_INTERVAL_SECONDS}, "
             f"min: {WATCH_MIN_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--no-parallel-postprocess", action="store_true",
        help="Run geocode/OSM/populate synchronously (disable GPU/CPU overlap)",
    )
    parser.add_argument(
        "--print-hardware", action="store_true",
        help="Print the detected hardware profile (Whisper + LLM params) as JSON and exit",
    )

    args = parser.parse_args()

    if args.print_hardware:
        profile = get_profile()
        json.dump(profile.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    if args.repair_dry_run and not args.repair_stale_state:
        parser.error("--repair-dry-run requires --repair-stale-state")

    if args.repair_stale_state:
        from scripts.repair_stale_state import repair_stale_video_state

        summary = repair_stale_video_state(dry_run=args.repair_dry_run)
        print(
            f"{'Would repair' if summary['dry_run'] else 'Repaired'} "
            f"{summary['count']} video(s)"
        )
        for vid in summary["repaired_video_ids"]:
            print(f"  - {vid}")
        return

    if args.status:
        _print_status()
        return

    if args.reset_all_data and not args.reset:
        parser.error("--reset-all-data requires --reset")

    if args.reset:
        _reset_all(reset_all_data=args.reset_all_data)

    if args.watch:
        if args.no_dashboard is False:
            logger.info("Watch mode: dashboard disabled (log-only).")
        run_pipeline_watch(
            poll_interval=args.poll_interval,
            skip_fetch=args.skip_fetch,
            skip_transcribe=args.skip_transcribe,
            skip_extract=args.skip_extract,
            skip_push=not args.push,
            whisper_model=args.whisper_model or WHISPER_DEFAULT_MODEL,
            max_videos=args.max_videos,
            auto_models=not args.no_auto_models,
        )
        return

    run_pipeline(
        skip_fetch=args.skip_fetch,
        skip_transcribe=args.skip_transcribe,
        skip_extract=args.skip_extract,
        skip_push=not args.push,
        whisper_model=args.whisper_model or WHISPER_DEFAULT_MODEL,
        max_videos=args.max_videos,
        no_dashboard=args.no_dashboard,
        auto_models=not args.no_auto_models,
        parallel_postprocess=not args.no_parallel_postprocess,
    )


if __name__ == "__main__":
    main()
