"""
run_pipeline.py — Main pipeline orchestrator.

Two-phase design:
  Phase 1 (catalog):
      1. Fetch channels from channels_input.txt
      2. Catalog ALL videos into videos.json with status="pending"
         (recipe videos, Shorts, and non-food videos → skipped_videos.json)
  Phase 2 (process):
      For each pending video (newest first, limited by --max-videos):
      3. Prefetch audio (sliding window of 20 files)
      4. Download audio via yt-dlp
      5. Transcribe: YouTube subtitles first, Whisper medium fallback
      6. Chunk transcription (90s chunks, 15s overlap)
      6b. Food-relevance gate (LLM classifies: is this a food review video?)
      7. Extract locales with LLM (Mistral 7B GGUF + video description)
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
    python -m scripts.run_pipeline --skip-push --max-videos 10
"""

import argparse
import json

from scripts.utils import (
    CACHE_DIR,
    CHANNELS_JSON,
    LOCALES_JSON,
    LOGS_DIR,
    PREFETCH_WINDOW,
    PROCESSED_VIDEOS_JSON,
    SKIPPED_VIDEOS_JSON,
    VIDEOS_JSON,
    VISITS_JSON,
    FLAGGED_SEGMENTS_JSON,
    cleanup_cache,
    ensure_dirs,
    load_json,
    save_json,
    setup_logging,
    today_str,
)
from scripts.schemas import VideoStatus
from scripts.dashboard import Dashboard

logger = setup_logging("pipeline")


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
    skip_push: bool = False,
    whisper_model: str = "medium",
    max_videos: int = 100,
    no_dashboard: bool = False,
):
    """
    Run the full pipeline.

    Args:
        skip_fetch: Skip cataloging new videos (work only on existing pending)
        skip_transcribe: Skip transcription (use cached transcripts)
        skip_extract: Skip LLM extraction (useful for testing geocoding/dedup)
        skip_push: Skip git commit and push
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        max_videos: Max pending videos to process in this run
        no_dashboard: Disable live dashboard (log-only mode)
    """
    ensure_dirs()

    # ── Dashboard setup ───────────────────────────────────────────────
    dash = Dashboard()
    use_dash = not no_dashboard

    if use_dash:
        dash.start()

    def _log(msg: str) -> None:
        logger.info(msg)
        if use_dash:
            dash.log(msg)

    def _refresh_stats() -> None:
        if not use_dash:
            return
        vc = _count_videos_by_status()
        dash.set_totals(
            total_in_db=vc["total"],
            pending=vc["pending"],
            processed=vc["processed"],
            errored=vc["errored"],
            skipped=len(load_json(SKIPPED_VIDEOS_JSON)),
            channels=len(load_json(CHANNELS_JSON)),
        )
        dash.state.locales_found = len(load_json(LOCALES_JSON))
        dash.state.visits_created = len(load_json(VISITS_JSON))
        dash.state.flagged = len(load_json(FLAGGED_SEGMENTS_JSON))

    try:
        _log("Pipeline started")
        if use_dash:
            dash.set_phase("Phase 1 — Catalog")

        # ------------------------------------------------------------------
        # PHASE 1: Catalog
        # ------------------------------------------------------------------

        # Step 1: Fetch channels
        if use_dash:
            dash.set_step("Fetch channels")
        _log("Step 1: Fetching channels from channels_input.txt...")
        from scripts.fetch_channels import fetch_channels
        new_channels = fetch_channels()
        _log(f"Step 1 complete: {len(new_channels)} new channels")
        _refresh_stats()

        # Step 2: Catalog ALL videos as pending (no download yet)
        if not skip_fetch:
            if use_dash:
                dash.set_step("Catalog videos")
            _log("Step 2: Cataloging channel videos...")
            from scripts.fetch_videos import catalog_channel_videos
            n = catalog_channel_videos()
            _log(f"Step 2 complete: {n} new videos cataloged")
            _refresh_stats()
        else:
            _log("Step 2: Skipped (--skip-fetch)")

        # ------------------------------------------------------------------
        # PHASE 2: Process pending videos (newest first)
        # ------------------------------------------------------------------

        if use_dash:
            dash.set_phase("Phase 2 — Processing")
            _refresh_stats()

        from scripts.fetch_videos import (
            download_audio,
            fetch_video_upload_date,
            get_pending_videos,
            update_video_status,
        )

        pending = get_pending_videos()
        to_process = pending[:max_videos]

        _log(
            f"Phase 2: {len(pending)} pending videos, "
            f"processing {len(to_process)} (newest first)"
        )

        if not to_process:
            _log("No pending videos to process")
            if not skip_push:
                _git_push()
            _log("Pipeline complete (no pending content)")
            return

        if use_dash:
            dash.set_video_batch(len(to_process))

        # ── Sliding window: pre-download audio for first batch ────────
        prefetch_n = min(PREFETCH_WINDOW, len(to_process))
        _log(f"Prefetch: downloading audio for first {prefetch_n} videos...")
        if use_dash:
            dash.set_step("Prefetch audio")
        for pv in to_process[:prefetch_n]:
            audio = download_audio(pv["video_id"], pv["url"])
            if not audio:
                _log(f"  ✗ Prefetch failed: {pv['title'][:55]}")
        _log(f"Prefetch complete: {prefetch_n} audio files ready")

        # Load channel info for rubrica inference
        channels = load_json(CHANNELS_JSON)
        channel_map = {ch["channel_id"]: ch for ch in channels}

        for i, video in enumerate(to_process, 1):
            # Sliding window: trim cache to window size
            deleted = cleanup_cache(PREFETCH_WINDOW)
            if deleted:
                _log(f"  Cache trimmed: {len(deleted)} old files removed")

            video_id = video["video_id"]
            channel_id = video["channel_id"]
            channel_info = channel_map.get(channel_id, {})
            channel_rubriche = channel_info.get("rubriche", [])
            v_title = video.get("title", video_id)

            if use_dash:
                dash.update_video(i, v_title)

            _log(f"[{i}/{len(to_process)}] {v_title[:70]}")

            # Resolve publish_date if not yet known
            publish_date = video.get("publish_date", "")
            if not publish_date:
                _log(f"  Resolving publish date for {video_id}...")
                publish_date = fetch_video_upload_date(video_id)
                if not publish_date:
                    publish_date = today_str()

            # Step 3: Download audio
            if use_dash:
                dash.set_step("Download audio")
            _log("  Downloading audio...")
            audio_path = download_audio(video_id, video["url"])
            if not audio_path:
                _log(f"  ✗ Audio download failed for {video_id}")
                update_video_status(video_id, VideoStatus.ERRORED, publish_date)
                _update_processed(video_id, channel_id, VideoStatus.ERRORED)
                if use_dash:
                    dash.tick_stat("errored")
                    dash.complete_video()
                    _refresh_stats()
                continue

            # Step 4: Transcribe
            if use_dash:
                dash.set_step("Transcribe")
            if not skip_transcribe:
                _log("  Transcribing...")
                from scripts.transcribe_video import transcribe_audio
                transcript = transcribe_audio(video_id, whisper_model)
                if not transcript:
                    _log(f"  ✗ Transcription failed for {video_id}")
                    update_video_status(video_id, VideoStatus.ERRORED, publish_date)
                    _update_processed(video_id, channel_id, VideoStatus.ERRORED)
                    if use_dash:
                        dash.tick_stat("errored")
                        dash.complete_video()
                        _refresh_stats()
                    continue
            else:
                _log("  Transcription skipped (cached)")
                transcript_path = CACHE_DIR / f"{video_id}_transcript.json"
                if transcript_path.exists():
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcript = json.load(f)
                else:
                    _log(f"  ✗ No cached transcript for {video_id}")
                    update_video_status(video_id, VideoStatus.ERRORED, publish_date)
                    _update_processed(video_id, channel_id, VideoStatus.ERRORED)
                    if use_dash:
                        dash.tick_stat("errored")
                        dash.complete_video()
                        _refresh_stats()
                    continue

            # Step 5: Chunk
            if use_dash:
                dash.set_step("Chunk")
            _log("  Chunking transcription...")
            from scripts.chunk_transcription import chunk_transcription
            chunks = chunk_transcription(transcript)
            _log(f"  {len(chunks)} chunks created")

            if not chunks:
                _log(f"  ✗ No chunks for {video_id}")
                update_video_status(video_id, VideoStatus.ERRORED, publish_date)
                _update_processed(video_id, channel_id, VideoStatus.ERRORED)
                if use_dash:
                    dash.tick_stat("errored")
                    dash.complete_video()
                    _refresh_stats()
                continue

            # Step 6: Extract locales with LLM
            if use_dash:
                dash.set_step("Extract (LLM)")
            if not skip_extract:
                _log("  Extracting locales with LLM...")
                from scripts.extract_locales import extract_from_video, is_food_review_video
                from scripts.fetch_videos import fetch_video_description, detect_non_food_video
                from scripts.video_intelligence import (
                    analyze_title, analyze_description, get_ground_truth_for_video,
                )
                video_description = fetch_video_description(video_id)
                if video_description:
                    _log(f"  Video description: {len(video_description)} chars")

                # ── Title/description intelligence ──────────────────
                video_intel = analyze_title(v_title)
                video_intel = analyze_description(video_description or "", video_intel)

                # Check ground truth for known overrides
                gt = get_ground_truth_for_video(video_id)
                if gt:
                    _log(f"  Ground truth available for {video_id}")
                    if gt.get("video_type") == "non_review":
                        _log(f"  ✗ Skipped (ground truth: non-review): {gt.get('notes', '')}")
                        update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
                        _update_processed(video_id, channel_id, VideoStatus.PROCESSED, 0, 0)
                        if use_dash:
                            dash.tick_stat("processed")
                            dash.complete_video()
                            _refresh_stats()
                        continue
                    # Inject ground truth venue hints
                    for v in gt.get("venues", []):
                        video_intel.venue_hints.append({
                            "name": v["name"],
                            "address": v.get("address", ""),
                            "source": "ground_truth",
                            "confidence": "very_high",
                        })
                    if gt.get("city"):
                        video_intel.city = gt["city"]

                # Title-based skip for non-review videos
                if video_intel.video_type == "non_review" and video_intel.skip_reason:
                    _log(f"  ✗ Skipped (title analysis: non-review): {video_intel.skip_reason}")
                    update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
                    _update_processed(video_id, channel_id, VideoStatus.PROCESSED, 0, 0)
                    if use_dash:
                        dash.tick_stat("processed")
                        dash.complete_video()
                        _refresh_stats()
                    continue

                _log(f"  Intel: type={video_intel.video_type}, city={video_intel.city}, "
                     f"hints={len(video_intel.venue_hints)}")

                # Re-check non-food with description
                is_nf, nf_reason = detect_non_food_video(v_title, video_description)
                if is_nf:
                    _log(f"  ✗ Skipped (non-food via description): {nf_reason}")
                    update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
                    _update_processed(video_id, channel_id, VideoStatus.PROCESSED, 0, 0)
                    if use_dash:
                        dash.tick_stat("processed")
                        dash.complete_video()
                        _refresh_stats()
                    continue

                # LLM food-relevance gate
                transcript_text = transcript.get("text", "")
                is_food, food_reason = is_food_review_video(
                    v_title, transcript_text, video_description
                )
                if not is_food:
                    _log(f"  ✗ Skipped (LLM food-check): {food_reason}")
                    update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
                    _update_processed(video_id, channel_id, VideoStatus.PROCESSED, 0, 0)
                    if use_dash:
                        dash.tick_stat("processed")
                        dash.complete_video()
                        _refresh_stats()
                    continue
                _log(f"  Food-check: {food_reason}")

                extractions, flagged_extractions = extract_from_video(
                    video_id, chunks, channel_rubriche, video_description,
                    video_title=v_title, video_intel=video_intel,
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

            # Correct timestamps: scan full transcript for best mention time
            from scripts.extract_locales import find_best_timestamp_in_transcript
            gt_lookup = {}
            if gt:
                for v in gt.get("venues", []):
                    gt_lookup[v["name"].lower()] = v.get("asr_variants", [])

            from scripts.chunk_transcription import seconds_to_timestamp as _stt
            for e in extractions:
                name = e.get("locale_name", "")
                variants = gt_lookup.get(name.lower(), [])
                best_ts = find_best_timestamp_in_transcript(name, transcript, variants)
                if best_ts is not None:
                    old_ts = e.get("mention_time", e.get("chunk_start_seconds", 0))
                    e["mention_time"] = best_ts
                    e["mention_timestamp"] = _stt(best_ts)
                    e["chunk_start_seconds"] = best_ts
                    e["chunk_start"] = _stt(best_ts)
                    # Also fix chunk_end to be ~90s after start (typical segment)
                    end_ts = best_ts + 90
                    e["chunk_end"] = _stt(end_ts)
                    _log(f"  Timestamp corrected for '{name}': {old_ts:.0f}s → {best_ts:.0f}s")

            if not extractions and not flagged_extractions:
                _log(f"  No locales found in {video_id}")
                update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
                _update_processed(video_id, channel_id, VideoStatus.PROCESSED, 0, 0)
                if use_dash:
                    dash.complete_video()
                    _refresh_stats()
                continue

            # Build set of trusted venue names (ground truth + high-confidence title hints)
            trusted_venue_names: set[str] = set()
            if video_intel:
                for hint in video_intel.venue_hints:
                    if hint.get("confidence") in ("very_high", "high"):
                        trusted_venue_names.add(hint["name"].lower().strip())

            def _is_trusted(ext: dict) -> bool:
                from thefuzz import fuzz
                name_lower = ext.get("locale_name", "").lower().strip()
                for tn in trusted_venue_names:
                    if fuzz.ratio(name_lower, tn) >= 70 or tn in name_lower or name_lower in tn:
                        return True
                return False

            # Step 7: Geocode
            if use_dash:
                dash.set_step("Geocode")
            if extractions:
                _log(f"  Geocoding {len(extractions)} locales...")
                from scripts.geocode_locales import geocode_extractions
                geocoded, non_geocoded = geocode_extractions(extractions)
                for e in non_geocoded:
                    if _is_trusted(e):
                        _log(f"  Geocoding failed for trusted venue '{e.get('locale_name')}' — keeping with default coords")
                        e["lat"] = 0.0
                        e["lon"] = 0.0
                        geocoded.append(e)
                    else:
                        e["confidence"] = 0.0
                        e["_flag_reason"] = "geocoding_failed"
                        flagged_extractions.append(e)
                extractions = geocoded
                _log(f"  Geocoded: {len(extractions)}, failed: {len(non_geocoded)}")

            # Step 7b: Verify locales exist on OpenStreetMap
            if use_dash:
                dash.set_step("Verify (OSM)")
            if extractions:
                _log(f"  Verifying {len(extractions)} locales on OpenStreetMap...")
                from scripts.verify_locales import verify_extractions
                verified, not_verified = verify_extractions(extractions)
                for e in not_verified:
                    if _is_trusted(e):
                        _log(f"  OSM not found for trusted venue '{e.get('locale_name')}' — keeping anyway")
                        e["osm_verified"] = False
                        verified.append(e)
                    else:
                        e["confidence"] = 0.0
                        e["_flag_reason"] = "osm_not_found"
                        flagged_extractions.append(e)
                extractions = verified
                _log(f"  Verified: {len(extractions)}, not found: {len(not_verified)}")

            # Step 8: Deduplicate
            if use_dash:
                dash.set_step("Deduplicate")
            if extractions:
                _log("  Deduplicating locales...")
                from scripts.deduplicate_locales import deduplicate_locales
                _, locale_mapping = deduplicate_locales(extractions)
            else:
                locale_mapping = []

            # Step 9: Populate visits + flagged
            if use_dash:
                dash.set_step("Populate")
            _log("  Populating visits...")
            from scripts.populate_json import populate_visits, populate_flagged

            new_visits = populate_visits(
                locale_mapping, video_id, channel_id, publish_date
            )

            if flagged_extractions:
                _log(f"  Saving {len(flagged_extractions)} flagged segments...")
                populate_flagged(flagged_extractions, video_id, channel_id)

            # Step 10: Update status
            if use_dash:
                dash.set_step("Update status")
            update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
            _update_processed(
                video_id, channel_id, VideoStatus.PROCESSED,
                len(new_visits), len(flagged_extractions),
            )

            _log(f"  ✓ Done: {len(new_visits)} visits, {len(flagged_extractions)} flagged")

            if use_dash:
                dash.complete_video()
                _refresh_stats()

            # Sliding window: prefetch next video outside current window
            next_prefetch = i - 1 + PREFETCH_WINDOW  # i is 1-based
            if next_prefetch < len(to_process):
                nv = to_process[next_prefetch]
                download_audio(nv["video_id"], nv["url"])

        # Final cache cleanup
        deleted = cleanup_cache(PREFETCH_WINDOW)
        if deleted:
            _log(f"Cleaned up {len(deleted)} old cached files")

        # Git push
        if not skip_push:
            _git_push()
            _log("Pushed to GitHub")
        else:
            _log("Git push skipped (--skip-push)")

        if use_dash:
            dash.set_phase("Complete ✓")
            _refresh_stats()

        _log("Pipeline complete!")

    finally:
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


def _print_status():
    """Print a summary of the current pipeline state and exit."""
    videos = load_json(VIDEOS_JSON)
    locales = load_json(LOCALES_JSON)
    visits = load_json(VISITS_JSON)
    flagged = load_json(FLAGGED_SEGMENTS_JSON)
    skipped = load_json(SKIPPED_VIDEOS_JSON)
    processed = load_json(PROCESSED_VIDEOS_JSON)
    channels = load_json(CHANNELS_JSON)

    pending = [v for v in videos if v.get("status") == "pending"]
    proc = [v for v in videos if v.get("status") == "processed"]
    errored = [v for v in videos if v.get("status") == "errored"]

    print(f"\n{'='*50}")
    print(f"  CiboBuono Pipeline — Status")
    print(f"{'='*50}")
    print(f"  Channels:       {len(channels)}")
    print(f"  Total videos:   {len(videos)}")
    print(f"    Pending:      {len(pending)}")
    print(f"    Processed:    {len(proc)}")
    print(f"    Errored:      {len(errored)}")
    print(f"  Skipped:        {len(skipped)} (recipe)")
    print(f"  Locales found:  {len(locales)}")
    print(f"  Visits:         {len(visits)}")
    print(f"  Flagged:        {len(flagged)}")
    print()

    if proc:
        print("  Processed videos:")
        for v in proc:
            # Find visit/flagged counts from processed_videos.json
            pv = next((p for p in processed if p["video_id"] == v["video_id"]), {})
            vc = pv.get("visits_extracted", 0)
            fc = pv.get("flagged_segments", 0)
            print(f"    [{v.get('publish_date','')}] {v['title'][:55]}  (visits={vc}, flagged={fc})")
        print()

    if locales:
        print("  Locales:")
        for loc in locales:
            print(f"    {loc['name']} | {loc.get('city','')} | ({loc.get('lat','')}, {loc.get('lon','')})")
        print()

    if visits:
        print("  Visits:")
        for vis in visits:
            loc_name = next((l["name"] for l in locales if l["locale_id"] == vis["locale_id"]), "?")
            print(f"    {loc_name} | {vis['sentiment']} | rating={vis['rating']} | video={vis['video_id']}")
        print()

    if flagged:
        print("  Flagged segments:")
        for f in flagged:
            print(f"    {f.get('locale_name','?')} | reason={f.get('reason','')} | video={f['video_id']}")
        print()


def _reset_all():
    """Reset all data, cache, and logs for a fresh start."""
    print("Resetting all data, cache, and logs...")

    # Reset JSON data files to empty arrays
    json_files = [
        CHANNELS_JSON, VIDEOS_JSON, LOCALES_JSON, VISITS_JSON,
        FLAGGED_SEGMENTS_JSON, SKIPPED_VIDEOS_JSON, PROCESSED_VIDEOS_JSON,
    ]

    for f in json_files:
        save_json(f, [])

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
        "--skip-push", action="store_true",
        help="Skip git commit and push",
    )
    parser.add_argument(
        "--whisper-model", default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium, best quality/speed for Italian)",
    )
    parser.add_argument(
        "--max-videos", type=int, default=100,
        help="Max pending videos to process in this run (default: 100)",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable live terminal dashboard (log-only mode)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset all data, cache, and logs before running",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show pipeline status summary and exit (no processing)",
    )

    args = parser.parse_args()

    if args.status:
        _print_status()
        return

    if args.reset:
        _reset_all()

    run_pipeline(
        skip_fetch=args.skip_fetch,
        skip_transcribe=args.skip_transcribe,
        skip_extract=args.skip_extract,
        skip_push=args.skip_push,
        whisper_model=args.whisper_model,
        max_videos=args.max_videos,
        no_dashboard=args.no_dashboard,
    )


if __name__ == "__main__":
    main()
