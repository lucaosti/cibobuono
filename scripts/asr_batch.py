"""
asr_batch.py — Transcribe a list of videos with Whisper, downloads overlapped with ASR.

Input: a meta JSON list (``{video_id, ...}``, the :mod:`scripts.gold_sample`
output). For every video without a cached ``cache/<video_id>_transcript.json``
the audio is downloaded with :func:`scripts.fetch_videos.download_audio` and
transcribed with :func:`scripts.transcribe_video.transcribe_audio`. Downloads
run in a small thread pool ahead of the single GPU transcription loop; the
audio file is deleted once its transcript is written.

Stages (``--stage``) split the work for clusters whose GPU nodes have no
internet access: ``download`` fetches audio and manual subtitles only (login
node), ``asr`` transcribes what is already cached without any network access
(GPU job), ``all`` does both in one process.

Per-video timings go to ``--times`` (JSONL) so the throughput of the host can
be reported. Safe to re-run: cached transcripts are skipped.

CLI:
    python -m scripts.asr_batch --meta runs/gold200/gold_meta.json \
        --times runs/gold200/asr_times.jsonl [--stage all|download|asr] [--model large-v3-turbo] [--prefetch 3]
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.fetch_videos import download_audio
from scripts.transcribe_video import _download_youtube_manual_subs, find_audio_file, transcribe_audio
from scripts.utils import CACHE_DIR, WHISPER_DEFAULT_MODEL, setup_logging

logger = setup_logging("asr_batch")


def _transcript_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}_transcript.json"


def _download(video_id: str, with_subs: bool = False) -> tuple[str, Path | None, float]:
    t0 = time.time()
    if with_subs:
        _download_youtube_manual_subs(video_id)
    path = download_audio(video_id, f"https://youtu.be/{video_id}")
    return video_id, path, time.time() - t0


def _cached_audio(video_id: str) -> tuple[str, Path | None, float]:
    return video_id, find_audio_file(video_id), 0.0


def _cleanup_audio(video_id: str) -> None:
    for _ in range(4):
        path = find_audio_file(video_id)
        if not path:
            return
        try:
            path.unlink()
        except OSError:
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Whisper transcription with prefetching downloads")
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--times", required=True, type=Path, help="JSONL with per-video timings")
    parser.add_argument("--stage", choices=("all", "download", "asr"), default="all")
    parser.add_argument("--model", default=WHISPER_DEFAULT_MODEL)
    parser.add_argument("--prefetch", type=int, default=3, help="audio downloads kept ahead of the GPU")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(args.meta.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    todo = [r["video_id"] for r in rows if not _transcript_path(r["video_id"]).exists()]
    logger.info("%d videos, %d without transcript (stage %s)", len(rows), len(todo), args.stage)
    if not todo:
        return 0

    durations = {r["video_id"]: r.get("duration") for r in rows}
    args.times.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    batch_t0 = time.time()
    if args.stage == "download":
        fetch = lambda v: _download(v, with_subs=True)  # noqa: E731
    elif args.stage == "asr":
        fetch = _cached_audio
    else:
        fetch = _download
    with ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as pool, \
            open(args.times, "a", encoding="utf-8") as times:
        futures = [pool.submit(fetch, v) for v in todo]
        for i, fut in enumerate(futures, 1):
            video_id, audio, dl_seconds = fut.result()
            if audio is None:
                failures += 1
                logger.error("%s: audio %s, skipped", video_id,
                             "not in cache" if args.stage == "asr" else "download failed")
                times.write(json.dumps({"video_id": video_id, "ok": False, "stage": "download"}) + "\n")
                times.flush()
                continue
            if args.stage == "download":
                logger.info("[%d/%d] %s: downloaded in %.0fs", i, len(todo), video_id, dl_seconds)
                continue
            t0 = time.time()
            transcript = transcribe_audio(video_id, args.model, fetch_subs=args.stage == "all")
            asr_seconds = time.time() - t0
            if not args.keep_audio:
                _cleanup_audio(video_id)
            if transcript is None:
                failures += 1
                logger.error("%s: transcription failed", video_id)
                times.write(json.dumps({"video_id": video_id, "ok": False, "stage": "asr"}) + "\n")
                times.flush()
                continue
            record = {
                "video_id": video_id,
                "ok": True,
                "duration": durations.get(video_id),
                "download_seconds": round(dl_seconds, 1),
                "asr_seconds": round(asr_seconds, 1),
                "segments": len(transcript.get("segments") or []),
                "source": transcript.get("source"),
            }
            times.write(json.dumps(record) + "\n")
            times.flush()
            speed = (durations.get(video_id) or 0) / asr_seconds if asr_seconds else 0
            logger.info("[%d/%d] %s: %.0fs audio in %.0fs (%.1fx realtime), elapsed %.0f min",
                        i, len(todo), video_id, durations.get(video_id) or 0, asr_seconds, speed,
                        (time.time() - batch_t0) / 60)
    logger.info("Done: %d ok, %d failed, %.1f min", len(todo) - failures, failures,
                (time.time() - batch_t0) / 60)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
