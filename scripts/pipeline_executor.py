"""
pipeline_executor.py — Overlap CPU/network postprocess with GPU work on the next video.

Videos are independent; the GPU runs Whisper OR LLM (not both). While the GPU
transcribes/extracts video N+1, video N is geocoded/verified/populated on a
background thread.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import os
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Any, Callable

from thefuzz import fuzz

from scripts.schemas import VideoStatus
from scripts.utils import setup_logging

from scripts.manual_edits import MIN_PUBLISH_OSM_SCORE

logger = setup_logging("pipeline_exec")

LogFn = Callable[[str], None]


def parallel_postprocess_enabled(has_cuda: bool) -> bool:
    raw = os.environ.get("CIBOBUONO_PARALLEL_POSTPROCESS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return has_cuda


@dataclass
class IntelPrepResult:
    video_id: str
    video_intel: Any | None = None
    youtube_extra: dict | None = None
    video_description: str = ""
    comments_count: int = 0
    error: str = ""


def _prepare_video_intel(video_id: str, title: str) -> IntelPrepResult:
    """Fetch metadata/comments and build VideoIntel (CPU + network, no GPU)."""
    try:
        from scripts.fetch_videos import (
            fetch_video_description,
            fetch_video_metadata,
            fetch_video_comments,
        )
        from scripts.video_intelligence import (
            analyze_title,
            analyze_description,
            analyze_description_timestamps,
            analyze_chapters,
            analyze_comments,
            parse_description_timestamps,
        )

        ym = fetch_video_metadata(video_id)
        video_description = fetch_video_description(video_id) or ym.get("description") or ""
        chapters = ym.get("chapters") or []
        dts = parse_description_timestamps(video_description)
        youtube_extra = {
            "chapters": chapters,
            "description_timestamps": dts,
        }

        video_intel = analyze_title(title)
        video_intel = analyze_description(video_description, video_intel)
        video_intel = analyze_description_timestamps(dts, video_intel)
        video_intel = analyze_chapters(chapters, video_intel)

        comments = fetch_video_comments(video_id)
        comments_count = len(comments)
        if comments:
            video_intel = analyze_comments(comments, video_intel)

        return IntelPrepResult(
            video_id=video_id,
            video_intel=video_intel,
            youtube_extra=youtube_extra,
            video_description=video_description,
            comments_count=comments_count,
        )
    except Exception as exc:
        logger.warning("Intel prep failed for %s: %s", video_id, exc)
        return IntelPrepResult(video_id=video_id, error=str(exc))


@dataclass
class FinalizeJob:
    video_id: str
    channel_id: str
    publish_date: str
    extractions: list[dict]
    flagged_extractions: list[dict]
    trusted_venue_names: set[str] = field(default_factory=set)
    intel_city: str = ""


@dataclass
class FinalizeResult:
    video_id: str
    visits_created: int = 0
    flagged_segments: int = 0
    outcome: str = "processed"
    error: str = ""
    # Per-video pipeline health counters (aggregated into run metrics)
    extractions_attempted: int = 0
    geocoded: int = 0
    osm_verified: int = 0
    published: int = 0
    city_mismatches: int = 0
    confidences: list[float] = field(default_factory=list)


_CITY_COHERENCE_THRESHOLD = 75


def _check_city_coherence(ext: dict, intel_city: str) -> bool:
    """True = geocoded city is consistent with expected intel city (or either is unknown)."""
    geocoded_city = (ext.get("geocoded_city") or "").strip()
    if not intel_city or not geocoded_city:
        return True
    return fuzz.partial_ratio(intel_city.lower(), geocoded_city.lower()) >= _CITY_COHERENCE_THRESHOLD


def _is_trusted(ext: dict, trusted_names: set[str]) -> bool:
    name_lower = ext.get("locale_name", "").lower().strip()
    for tn in trusted_names:
        if fuzz.ratio(name_lower, tn) >= 70 or tn in name_lower or name_lower in tn:
            return True
    return False


def _publishable_extraction(ext: dict, trusted: set[str]) -> tuple[bool, str]:
    """Return (True, '') if extraction may become a published visit."""
    if _is_trusted(ext, trusted):
        lat, lon = ext.get("lat"), ext.get("lon")
        if (lat in (None, 0, 0.0) or lon in (None, 0, 0.0)) and not ext.get("osm_verified"):
            return False, "trusted_no_coords"
        return True, ""

    if not ext.get("osm_verified"):
        return False, "osm_not_found"

    score = int(ext.get("osm_match_score") or 0)
    if score < MIN_PUBLISH_OSM_SCORE:
        return False, f"osm_weak_match_{score}"

    lat, lon = ext.get("lat"), ext.get("lon")
    if lat in (None, 0, 0.0) or lon in (None, 0, 0.0):
        return False, "missing_coords"

    return True, ""


def finalize_video(job: FinalizeJob, log: LogFn | None = None) -> FinalizeResult:
    """Geocode → city-coherence → OSM verify → dedupe → populate JSON → mark processed."""
    _log = log or logger.info

    try:
        from scripts.fetch_videos import update_video_status
        from scripts.geocode_locales import geocode_extractions
        from scripts.verify_locales import verify_extractions
        from scripts.deduplicate_locales import deduplicate_locales
        from scripts.populate_json import populate_visits, populate_flagged, update_processed_videos

        extractions = list(job.extractions)
        flagged_extractions = list(job.flagged_extractions)
        trusted = job.trusted_venue_names
        n_attempted = len(extractions)
        n_geocoded = n_osm = n_published = n_city_mismatch = 0
        confidences: list[float] = []

        if extractions:
            _log(f"  [{job.video_id}] Geocoding {len(extractions)} locales (background)…")
            geocoded, non_geocoded = geocode_extractions(extractions)
            n_geocoded = len(geocoded)
            for e in non_geocoded:
                if _is_trusted(e, trusted):
                    _log(f"  [{job.video_id}] Geocoding failed for trusted '{e.get('locale_name')}' — keeping with default coords")
                    e["lat"] = 0.0
                    e["lon"] = 0.0
                    geocoded.append(e)
                else:
                    e["confidence"] = 0.0
                    e["_flag_reason"] = "geocoding_failed"
                    flagged_extractions.append(e)
            extractions = geocoded

            coherent: list[dict] = []
            for e in extractions:
                if _check_city_coherence(e, job.intel_city):
                    coherent.append(e)
                else:
                    n_city_mismatch += 1
                    fe = dict(e)
                    fe["confidence"] = min(float(fe.get("confidence", 0.5)), 0.45)
                    fe["_flag_reason"] = "city_mismatch"
                    flagged_extractions.append(fe)
                    _log(f"  [{job.video_id}] City mismatch '{e.get('locale_name')}': geocoded={e.get('geocoded_city')!r} vs expected={job.intel_city!r}")
            extractions = coherent

        if extractions:
            _log(f"  [{job.video_id}] Verifying {len(extractions)} locales on OSM…")
            verified, not_verified = verify_extractions(extractions)
            n_osm = len(verified)
            for e in not_verified:
                if _is_trusted(e, trusted):
                    e["osm_verified"] = False
                    verified.append(e)
                else:
                    e["confidence"] = 0.0
                    e["_flag_reason"] = "osm_not_found"
                    flagged_extractions.append(e)
            extractions = verified

        publishable: list[dict] = []
        for e in extractions:
            ok_pub, why = _publishable_extraction(e, trusted)
            if ok_pub:
                publishable.append(e)
                confidences.append(float(e.get("confidence", 0.0)))
            else:
                fe = dict(e)
                fe["confidence"] = min(float(fe.get("confidence", 0.5)), 0.45)
                fe["_flag_reason"] = why
                flagged_extractions.append(fe)
                _log(f"  [{job.video_id}] Non pubblicato '{e.get('locale_name')}': {why}")
        extractions = publishable
        n_published = len(extractions)

        if extractions:
            _log(f"  [{job.video_id}] Deduplicating…")
            _, locale_mapping = deduplicate_locales(extractions)
        else:
            locale_mapping = []

        _log(f"  [{job.video_id}] Populating visits…")
        new_visits = populate_visits(
            locale_mapping, job.video_id, job.channel_id, job.publish_date
        )
        if flagged_extractions:
            populate_flagged(flagged_extractions, job.video_id, job.channel_id)

        update_video_status(job.video_id, VideoStatus.PROCESSED, job.publish_date)
        update_processed_videos(
            job.video_id,
            job.channel_id,
            VideoStatus.PROCESSED,
            len(new_visits),
            len(flagged_extractions),
        )
        _log(
            f"  [{job.video_id}] ✓ Finalized: {len(new_visits)} visits, "
            f"{len(flagged_extractions)} flagged"
        )
        return FinalizeResult(
            video_id=job.video_id,
            visits_created=len(new_visits),
            flagged_segments=len(flagged_extractions),
            outcome="processed",
            extractions_attempted=n_attempted,
            geocoded=n_geocoded,
            osm_verified=n_osm,
            published=n_published,
            city_mismatches=n_city_mismatch,
            confidences=confidences,
        )
    except Exception as exc:
        logger.exception("Finalize failed for %s", job.video_id)
        return FinalizeResult(
            video_id=job.video_id,
            outcome="errored",
            error=str(exc),
        )


def _warm_video_cache(video_id: str, url: str) -> None:
    """Prefetch audio + metadata while GPU is busy on another video."""
    try:
        from scripts.fetch_videos import (
            download_audio,
            fetch_video_description,
            fetch_video_metadata,
        )

        download_audio(video_id, url)
        fetch_video_metadata(video_id)
        fetch_video_description(video_id)
    except Exception as exc:
        logger.debug("IO warm cache failed for %s: %s", video_id, exc)


class PipelineExecutor:
    """Background IO warmup + deferred geocode/OSM/populate."""

    def __init__(
        self,
        *,
        parallel_postprocess: bool,
        io_workers: int = 4,
        max_pending_finalize: int = 2,
    ):
        self.parallel_postprocess = parallel_postprocess
        self.max_pending_finalize = max(1, max_pending_finalize)
        self._io_pool = ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="io_prep"
        )
        self._post_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="postprocess"
        )
        self._finalize_futures: list[Future] = []
        self._completed: list[FinalizeResult] = []
        self._intel_futures: dict[str, Future] = {}

    def schedule_intel_prep(self, video_id: str, title: str) -> None:
        """Prefetch metadata + intel for a video while GPU works on another."""
        if video_id in self._intel_futures:
            return
        self._intel_futures[video_id] = self._io_pool.submit(
            _prepare_video_intel, video_id, title
        )

    def take_intel_prep(self, video_id: str, title: str) -> IntelPrepResult:
        """Return prefetched intel, or compute synchronously if not scheduled."""
        fut = self._intel_futures.pop(video_id, None)
        if fut is None:
            return _prepare_video_intel(video_id, title)
        try:
            return fut.result()
        except Exception as exc:
            logger.warning("Prefetched intel failed for %s: %s", video_id, exc)
            return IntelPrepResult(video_id=video_id, error=str(exc))

    def schedule_io_prep(
        self,
        videos: list[dict],
        start_index: int,
        *,
        count: int = 4,
    ) -> None:
        for j in range(start_index, min(start_index + count, len(videos))):
            v = videos[j]
            self._io_pool.submit(_warm_video_cache, v["video_id"], v["url"])

    def submit_finalize(
        self,
        job: FinalizeJob,
        log: LogFn | None = None,
    ) -> Future | FinalizeResult:
        if not self.parallel_postprocess:
            result = finalize_video(job, log)
            self._completed.append(result)
            return result

        self._wait_for_finalize_slot()
        fut = self._post_pool.submit(finalize_video, job, log)
        self._finalize_futures.append(fut)
        return fut

    def poll_completed(self) -> list[FinalizeResult]:
        """Return finalize results that finished since last poll."""
        still: list[Future] = []
        for fut in self._finalize_futures:
            if fut.done():
                try:
                    self._completed.append(fut.result())
                except Exception as exc:
                    self._completed.append(
                        FinalizeResult(outcome="errored", error=str(exc))
                    )
            else:
                still.append(fut)
        self._finalize_futures = still
        ready = self._completed
        self._completed = []
        return ready

    def drain_finalize(self, log: LogFn | None = None) -> list[FinalizeResult]:
        """Wait for all pending finalize jobs (end of batch / shutdown)."""
        results: list[FinalizeResult] = []
        if not self.parallel_postprocess:
            return results

        pending = list(self._finalize_futures)
        self._finalize_futures.clear()
        for fut in pending:
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append(FinalizeResult(outcome="errored", error=str(exc)))
        results.extend(self.poll_completed())
        if log and results:
            n = sum(r.visits_created for r in results)
            log(f"Drained {len(results)} background finalize job(s), {n} visits total")
        return results

    def _wait_for_finalize_slot(self) -> None:
        while len(self._finalize_futures) >= self.max_pending_finalize:
            done, _ = wait(self._finalize_futures, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    self._completed.append(fut.result())
                except Exception as exc:
                    self._completed.append(
                        FinalizeResult(outcome="errored", error=str(exc))
                    )
            self._finalize_futures = [f for f in self._finalize_futures if not f.done()]

    def shutdown(self) -> None:
        self.drain_finalize()
        self._intel_futures.clear()
        self._io_pool.shutdown(wait=False, cancel_futures=True)
        self._post_pool.shutdown(wait=True)
