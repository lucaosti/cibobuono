"""
perceptor.py — Audio/video perception orchestrator.

Runs the full perception pass on one video:

1. Audio (CPU, every tier): Silero VAD → speaker embeddings → diarization →
   per-channel voice registry matching (:mod:`scripts.perceptor_audio`).
2. Video (GPU tiers): low-res download → frame sampling + phash novelty →
   VLM captioning of novel frames (:mod:`scripts.perceptor_video`). The video
   file is deleted right after frame extraction.

Results are persisted to ``data/perception.json`` (JSON array, one record per
video — ``load_json`` coerces non-list payloads to ``[]``). Every sub-step is
individually guarded: a failure downgrades the record to ``status="partial"``
(or ``"errored"`` when nothing succeeded) but NEVER propagates to the caller —
the main pipeline must not die because perception did.

Enable with ``--perceptor`` or ``CIBOBUONO_PERCEPTOR=1``.

CLI (one-shot on a cached video):
    python -m scripts.perceptor <video_id> [--force]
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils import DATA_DIR, load_json, save_json_split, setup_logging

logger = setup_logging("perceptor")

PERCEPTION_JSON = DATA_DIR / "perception.json"

_missing_deps_logged = False


def _deps_available() -> bool:
    """True when the Perceptor runtime deps are importable."""
    import importlib.util

    return all(
        importlib.util.find_spec(m) is not None
        for m in ("sherpa_onnx", "numpy", "av", "imagehash")
    )


def perceptor_enabled(cli_flag: bool | None = None) -> bool:
    """Whether the Perceptor stage should run.

    CLI flag wins; otherwise CIBOBUONO_PERCEPTOR ("1"/"true"/"yes"); default
    off. Missing dependencies log once and behave as disabled.
    """
    global _missing_deps_logged
    if cli_flag is not None:
        enabled = cli_flag
    else:
        enabled = os.environ.get("CIBOBUONO_PERCEPTOR", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    if enabled and not _deps_available():
        if not _missing_deps_logged:
            logger.warning(
                "Perceptor requested but dependencies are missing "
                "(sherpa-onnx / numpy / av / imagehash) — stage disabled. "
                "Install with: pip install -r requirements.txt"
            )
            _missing_deps_logged = True
        return False
    return enabled


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def perceive_video(
    video_id: str,
    url: str,
    channel_id: str,
    transcript: dict | None,
) -> dict:
    """Full perception pass on one video. Returns the perception record.

    Sub-steps are individually guarded; partial results are kept and the
    record status reflects what succeeded ("ok" | "partial" | "errored").
    """
    from scripts.hardware import get_profile
    from scripts.transcribe_video import find_audio_file

    profile = get_profile()
    errors: list[str] = []
    record: dict = {
        "video_id": video_id,
        "channel_id": channel_id,
        "created_at": _now_iso(),
        "status": "ok",
        "error": None,
        "asr_backend": profile.asr_backend,
        "audio": None,
        "video": None,
    }

    # ── Audio: VAD → embeddings → diarization → voice registry ───────────
    try:
        from scripts import perceptor_audio as pa

        audio_path = find_audio_file(video_id)
        if audio_path is None:
            raise FileNotFoundError(f"no cached audio for {video_id}")

        segments = pa.detect_speech_segments(audio_path)
        duration = max((s["end"] for s in segments), default=0.0)
        speech_s = sum(s["end"] - s["start"] for s in segments)

        audio_info: dict = {
            "duration_s": round(duration, 1),
            "speech_ratio": round(speech_s / duration, 3) if duration else 0.0,
            "vad_segments": segments,
            "speakers": [],
            "segment_speakers": [],
        }

        if segments:
            embeddings = pa.embed_segments(audio_path, segments)
            labels = pa.cluster_speakers(embeddings)
            centroids = pa.speaker_centroids(embeddings, labels)

            def _label_stats(label: str) -> tuple[list[int], float]:
                lab_idx = int(label[1:])
                idx = [i for i, x in enumerate(labels) if x == lab_idx]
                talk = sum(segments[i]["end"] - segments[i]["start"] for i in idx)
                return idx, talk

            # Only substantial speakers enter the channel voice registry —
            # 1-segment micro-clusters are background voices/jingles.
            substantial: dict = {}
            for label, emb in centroids.items():
                seg_idx, talk = _label_stats(label)
                if (
                    len(seg_idx) >= pa.MIN_REGISTRY_SEGMENTS
                    and talk >= pa.MIN_REGISTRY_TALK_TIME_S
                ):
                    substantial[label] = emb

            voice_results: list[dict] = []
            try:
                if channel_id and substantial:
                    voice_results = pa.match_or_register_voices(
                        channel_id, video_id, substantial
                    )
            except Exception as e:
                errors.append(f"voice_registry: {e}")
                logger.warning("Voice registry failed for %s: %s", video_id, e)
            by_label = {r["label"]: r for r in voice_results}

            for label in sorted(centroids):
                seg_idx, talk = _label_stats(label)
                match = by_label.get(label, {})
                audio_info["speakers"].append(
                    {
                        "label": label,
                        "talk_time_s": round(talk, 1),
                        "segments": len(seg_idx),
                        "voice_id": match.get("voice_id"),
                        "voice_match_score": (
                            None if match.get("new") else match.get("score")
                        ),
                    }
                )

            if transcript:
                audio_info["segment_speakers"] = pa.assign_speakers_to_transcript(
                    transcript, segments, labels
                )

        record["audio"] = audio_info
    except Exception as e:
        errors.append(f"audio: {e}")
        logger.warning("Audio perception failed for %s: %s", video_id, e)

    # ── Video: download → novel frames → captions ────────────────────────
    try:
        from scripts import perceptor_video as pv
        from scripts.fetch_videos import download_video

        video_info: dict = {
            "downloaded": False,
            "max_height": 480,
            "frame_interval_s": profile.frame_interval_s,
            "frames_sampled": 0,
            "novelty_frames": 0,
            "captioned": 0,
            "captions": [],
        }
        video_path = download_video(video_id, url)
        if video_path is not None:
            video_info["downloaded"] = True
            try:
                frames = pv.sample_novel_frames(
                    video_path, interval_s=profile.frame_interval_s
                )
            finally:
                # Frames are in memory; the 480p file is no longer needed.
                try:
                    video_path.unlink(missing_ok=True)
                except OSError:
                    pass
            video_info["frames_sampled"] = len(frames)
            video_info["novelty_frames"] = sum(1 for f in frames if f.novel)
            captions = pv.caption_frames(frames, budget=profile.caption_budget)
            video_info["captioned"] = len(captions)
            video_info["captions"] = captions
        else:
            errors.append("video: download failed")
        record["video"] = video_info
    except Exception as e:
        errors.append(f"video: {e}")
        logger.warning("Video perception failed for %s: %s", video_id, e)

    if errors:
        both_failed = record["audio"] is None and record["video"] is None
        record["status"] = "errored" if both_failed else "partial"
        record["error"] = "; ".join(errors)[:500]
    return record


def run_perceptor_stage(video: dict, transcript: dict | None, log=None) -> dict | None:
    """Best-effort wrapper used by the pipeline: NEVER raises.

    Returns the perception record (possibly errored) or None on a failure so
    unexpected that not even an error record could be built.
    """
    _say = log or logger.info
    video_id = video.get("video_id", "")
    try:
        record = perceive_video(
            video_id,
            video.get("url", ""),
            video.get("channel_id", ""),
            transcript,
        )
        if record["status"] != "ok":
            _say(f"Perceptor {record['status']} for {video_id}: {record['error']}")
        return record
    except Exception as e:  # pragma: no cover — belt and braces
        logger.error("Perceptor stage crashed for %s: %s", video_id, e)
        try:
            return {
                "video_id": video_id,
                "channel_id": video.get("channel_id", ""),
                "created_at": _now_iso(),
                "status": "errored",
                "error": str(e)[:500],
                "asr_backend": "",
                "audio": None,
                "video": None,
            }
        except Exception:
            return None


def upsert_perception(record: dict) -> None:
    """Insert or replace the perception record for record['video_id']."""
    data = load_json(PERCEPTION_JSON)
    data = [r for r in data if r.get("video_id") != record.get("video_id")]
    data.append(record)
    save_json_split(PERCEPTION_JSON, data)


def get_perception(video_id: str) -> dict | None:
    """Stored perception record for *video_id*, or None."""
    for r in load_json(PERCEPTION_JSON):
        if r.get("video_id") == video_id:
            return r
    return None


# ---------------------------------------------------------------------------
# CLI: one-shot perception on a (cached) video
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse
    import json as _json

    from scripts.utils import VIDEOS_JSON

    parser = argparse.ArgumentParser(description="Run Perceptor on one video")
    parser.add_argument("video_id")
    parser.add_argument(
        "--force", action="store_true", help="re-run even if a record exists"
    )
    args = parser.parse_args()

    if not _deps_available():
        print("Perceptor dependencies missing — pip install -r requirements.txt")
        return 1
    existing = get_perception(args.video_id)
    if existing and not args.force:
        print(f"Perception record already exists for {args.video_id} (use --force):")
        print(_json.dumps({k: existing[k] for k in ("status", "error")}, indent=2))
        return 0

    video = next(
        (v for v in load_json(VIDEOS_JSON) if v.get("video_id") == args.video_id),
        {"video_id": args.video_id, "url": "", "channel_id": ""},
    )

    transcript = None
    from scripts.utils import CACHE_DIR

    tpath = CACHE_DIR / f"{args.video_id}_transcript.json"
    if tpath.exists():
        transcript = _json.loads(tpath.read_text(encoding="utf-8"))

    record = run_perceptor_stage(video, transcript)
    if record is None:
        print("Perceptor failed irrecoverably")
        return 1
    upsert_perception(record)

    summary = {
        "status": record["status"],
        "error": record["error"],
        "speakers": [
            {k: s[k] for k in ("label", "talk_time_s", "voice_id")}
            for s in (record.get("audio") or {}).get("speakers", [])
        ],
        "speech_ratio": (record.get("audio") or {}).get("speech_ratio"),
        "frames_sampled": (record.get("video") or {}).get("frames_sampled"),
        "novelty_frames": (record.get("video") or {}).get("novelty_frames"),
        "captioned": (record.get("video") or {}).get("captioned"),
        "sample_captions": [
            c["caption"] for c in (record.get("video") or {}).get("captions", [])[:3]
        ],
    }
    print(_json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
