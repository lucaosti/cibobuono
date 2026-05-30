"""
transcribe_video.py — Transcribe video audio to text.

Strategy (in priority order):
1. Try downloading YouTube's *manual* (human-written) subtitles via yt-dlp.
   These are very rare but, when present, more accurate than any ASR.
2. Otherwise transcribe locally with Whisper (faster-whisper on Apple Silicon /
   CUDA; openai-whisper pure-Python fallback). YouTube's auto-generated
   subtitles are NOT used because they mangle Italian proper nouns
   (restaurant names, street names) which are the whole point of this dataset.

Saves transcriptions as JSON in cache/ with segment-level timestamps.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import re
import subprocess
from pathlib import Path

from scripts.hardware import get_profile
from scripts.utils import (
    CACHE_DIR,
    CONTENT_LANGUAGE,
    WHISPER_DEFAULT_MODEL,
    ensure_dirs,
    setup_logging,
    yt_dlp_command,
)

logger = setup_logging("transcribe")

# Backend name constants — used as the "source" field in transcript dicts
# and written to cached JSON, so keep stable.
_BACKEND_FASTER = "faster_whisper"
_BACKEND_OPENAI = "openai_whisper"

# Global Whisper model cache (loaded once per session, like the LLM)
_whisper_model = None
_whisper_model_name = None
_whisper_backend = None
_whisper_device: str | None = None  # actual device used: "cuda" | "cpu" | "mps"


# ---------------------------------------------------------------------------
# YouTube manual subtitles (rare but highest quality when present)
# ---------------------------------------------------------------------------
#
# Note: We deliberately do NOT try YouTube's auto-generated subtitles.  They
# are produced by an ASR model that systematically mangles Italian proper
# nouns ("Raimond di Garibaldi" instead of "Raimondi di Garibaldi",
# "l'onoreficienza più scra" instead of "l'onorificenza più sacra", etc.),
# which destroys downstream venue extraction.  Local Whisper large-v3-turbo
# is significantly more accurate on the names we care about.

def _download_youtube_manual_subs(video_id: str) -> Path | None:
    """
    Download YouTube *manual* (human-written) subtitles via yt-dlp.

    Auto-generated subs are explicitly disabled — see module docstring.

    Returns path to the downloaded .vtt file, or None if no manual subs exist.
    """
    ensure_dirs()
    output_template = str(CACHE_DIR / f"{video_id}_subs")

    try:
        subprocess.run(
            [
                *yt_dlp_command(),
                "--skip-download",
                "--write-subs",
                "--no-write-auto-subs",
                "--sub-langs", CONTENT_LANGUAGE,
                "--sub-format", "vtt",
                "--convert-subs", "vtt",
                "-o", output_template,
                "--no-playlist",
                f"https://youtu.be/{video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        for suffix in [".it.vtt", ".it.srt"]:
            candidate = CACHE_DIR / f"{video_id}_subs{suffix}"
            if candidate.exists() and candidate.stat().st_size > 100:
                logger.info(
                    f"Downloaded YouTube manual subs for {video_id}: "
                    f"{candidate.name} ({candidate.stat().st_size} bytes)"
                )
                return candidate

    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Manual sub download failed for {video_id}: {e}")

    logger.info(f"No YouTube manual subtitles available for {video_id}")
    return None


def _parse_vtt(vtt_path: Path) -> dict | None:
    """
    Parse a WebVTT subtitle file into our transcript format.
    
    Returns:
        dict with 'video_id', 'language', 'text', 'segments', 'source'
    """
    try:
        raw = vtt_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.error(f"Could not read VTT file {vtt_path}: {e}")
        return None

    video_id = vtt_path.stem.replace("_subs.it", "").replace("_subs", "")

    # Parse VTT timestamps: "00:01:23.456 --> 00:01:26.789"
    # Each cue block: timestamp line + one or more text lines
    timestamp_re = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3})"
    )

    segments = []
    full_text_parts = []
    seg_id = 0
    prev_cue_lines: list[str] = []  # Track lines of previous cue for overlap

    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = timestamp_re.match(line)
        if m:
            start_str, end_str = m.group(1), m.group(2)
            start_sec = _vtt_time_to_seconds(start_str)
            end_sec = _vtt_time_to_seconds(end_str)

            # Collect text lines until empty line or next timestamp
            text_lines: list[str] = []
            i += 1
            while i < len(lines):
                tl = lines[i].strip()
                if not tl or timestamp_re.match(tl):
                    break
                # Strip VTT formatting tags like <c>, </c>, <00:01:23.456>
                tl = re.sub(r"<[^>]+>", "", tl)
                if tl:
                    text_lines.append(tl)
                i += 1

            if not text_lines:
                continue

            # --- Scrolling-subtitle dedup ---
            # YouTube auto-generated subs use a "scrolling" format:
            #   Cue N:   [line_A, line_B]
            #   Cue N+1: [line_B, line_C]
            # The trailing lines of the previous cue overlap with the
            # leading lines of the current cue.  Remove the overlap so
            # each line is emitted only once.
            new_lines = text_lines[:]
            if prev_cue_lines:
                # Find the longest k such that the last k lines of
                # prev_cue_lines equal the first k lines of text_lines.
                max_k = min(len(prev_cue_lines), len(new_lines))
                overlap = 0
                for k in range(1, max_k + 1):
                    if prev_cue_lines[-k:] == new_lines[:k]:
                        overlap = k
                new_lines = new_lines[overlap:]

            prev_cue_lines = text_lines  # update for next iteration

            text = " ".join(new_lines).strip()
            if text:
                segments.append({
                    "id": seg_id,
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "text": text,
                })
                full_text_parts.append(text)
                seg_id += 1
            elif segments:
                # All lines overlap — just extend the end time
                segments[-1]["end"] = end_sec
        else:
            i += 1

    if not segments:
        logger.warning(f"VTT parsed but no segments found: {vtt_path.name}")
        return None

    transcript = {
        "video_id": video_id,
        "language": CONTENT_LANGUAGE,
        "text": " ".join(full_text_parts),
        "segments": segments,
        "source": "youtube_subs_manual",
    }

    logger.info(
        f"Parsed YouTube manual subs: {len(segments)} segments, "
        f"{len(transcript['text'])} chars"
    )
    return transcript


def _vtt_time_to_seconds(time_str: str) -> float:
    """Convert VTT timestamp 'HH:MM:SS.mmm' to seconds.

    Distinct from schemas.timestamp_to_seconds: VTT uses fractional seconds
    (e.g. '00:01:23.456'), whereas schemas handles integer-only 'HH:MM:SS'.
    """
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


# ---------------------------------------------------------------------------
# Local Whisper transcription (primary ASR path)
# ---------------------------------------------------------------------------


def _openai_whisper_device() -> str:
    """Pick a device string for the openai-whisper fallback only.

    openai-whisper *can* use Metal via PyTorch MPS; faster-whisper cannot
    (CTranslate2 has no Metal backend), so this helper is intentionally
    decoupled from the main DeviceProfile.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def _get_whisper_model(model_name: str = WHISPER_DEFAULT_MODEL):
    """Load and cache the Whisper model.

    Tries faster-whisper first (CTranslate2 — CUDA fp16 / CPU int8) and falls
    back to openai-whisper, which can use Metal via PyTorch MPS. Device and
    compute-type defaults come from :func:`scripts.hardware.get_profile`.
    """
    global _whisper_model, _whisper_model_name, _whisper_backend, _whisper_device

    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model, _whisper_backend

    from scripts.utils import MODELS_DIR
    download_root = str(MODELS_DIR / "whisper")
    profile = get_profile()

    # Runtime governance: pause if the system is under pressure, then pick a
    # Whisper size that fits the memory that's actually free right now (VRAM on
    # CUDA, RAM otherwise). Quality stays first: we only ever downgrade when the
    # requested model genuinely won't fit.
    from scripts import resource_monitor as rm

    if _whisper_model is None:
        rm.wait_until_calm(include_gpu=profile.whisper_device == "cuda")
    fitted, note = rm.fit_whisper_model(profile, model_name)
    if fitted != model_name:
        logger.warning("Whisper model selection: %s", note)
        model_name = fitted
    else:
        logger.info("Whisper model selection: %s", note)

    # faster-whisper has no Metal backend: on Apple Silicon we run CPU + int8.
    # On CUDA we use fp16 (or int8_float16 for small VRAM, picked in hardware.py).
    fw_device = profile.whisper_device
    compute_type = profile.whisper_compute_type

    try:
        from faster_whisper import WhisperModel
        logger.info(
            "Loading faster-whisper model: %s (device=%s, compute=%s, cpu_threads=%d)",
            model_name, fw_device, compute_type, profile.whisper_cpu_threads,
        )
        kwargs: dict = {
            "device": fw_device,
            "compute_type": compute_type,
            "download_root": download_root,
        }
        if fw_device == "cpu" and profile.whisper_cpu_threads > 0:
            kwargs["cpu_threads"] = profile.whisper_cpu_threads
        _whisper_model = WhisperModel(model_name, **kwargs)
        _whisper_model_name = model_name
        _whisper_backend = _BACKEND_FASTER
        _whisper_device = fw_device
        logger.info("faster-whisper loaded successfully")
        return _whisper_model, _whisper_backend
    except ImportError:
        logger.info("faster-whisper not installed; falling back to openai-whisper")
    except Exception as e:
        logger.warning(f"faster-whisper load failed ({e}); falling back to openai-whisper")

    try:
        import whisper
    except ImportError:
        logger.error(
            "Neither faster-whisper nor openai-whisper is installed. "
            "Install with: pip install faster-whisper  (or: pip install openai-whisper)"
        )
        return None, None

    # openai-whisper can use Metal (MPS) when PyTorch is built with it.
    device = _openai_whisper_device()
    logger.info(f"Loading openai-whisper model: {model_name} on {device}")
    _whisper_model = whisper.load_model(model_name, download_root=download_root, device=device)
    _whisper_model_name = model_name
    _whisper_backend = _BACKEND_OPENAI
    _whisper_device = device
    return _whisper_model, _whisper_backend


def release_whisper_model() -> None:
    """Unload Whisper from GPU/RAM so the LLM can use VRAM without contention."""
    global _whisper_model, _whisper_model_name, _whisper_backend, _whisper_device
    if _whisper_model is None:
        return
    _whisper_model = None
    _whisper_model_name = None
    _whisper_backend = None
    _whisper_device = None
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Whisper model released — VRAM freed for LLM extraction")


def find_audio_file(video_id: str) -> Path | None:
    """Find the cached audio file for a given video ID."""
    for ext in [".wav", ".mp3", ".m4a", ".webm"]:
        path = CACHE_DIR / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def transcribe_audio(video_id: str, model_name: str = WHISPER_DEFAULT_MODEL) -> dict | None:
    """
    Get transcription for a video.

    Strategy:
    1. Return cached transcript if available.
    2. Try YouTube *manual* (human-written) subtitles — rare but, when
       present, more accurate than any ASR.
    3. Otherwise transcribe locally with Whisper (faster-whisper preferred,
       openai-whisper fallback). This is the *primary* path for almost every
       video, because YouTube auto-subs mangle Italian proper nouns.

    Args:
        video_id: YouTube video ID
        model_name: Whisper model size (default: large-v3-turbo)

    Returns:
        dict with 'text' (full transcript) and 'segments' (timestamped chunks)
    """
    # Check if transcription is already cached
    transcript_path = CACHE_DIR / f"{video_id}_transcript.json"
    if transcript_path.exists():
        logger.info(f"Transcription already cached: {video_id}")
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            source = cached.get("source", "whisper")
            logger.info(f"  Source: {source}")
            return cached
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Cached transcription corrupted, re-transcribing: {video_id}")

    # --- Strategy 1: YouTube *manual* subtitles (rare, but best when present) ---
    subs_path = _download_youtube_manual_subs(video_id)
    if subs_path:
        transcript = _parse_vtt(subs_path)
        if transcript and len(transcript.get("segments", [])) > 0:
            ensure_dirs()
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
            logger.info(
                f"✓ Using YouTube manual subs for {video_id} "
                f"({len(transcript['segments'])} segments)"
            )
            return transcript

    # --- Strategy 2: Local Whisper (primary path) ---
    logger.info(f"Transcribing with Whisper ({model_name}) for {video_id}")
    audio_path = find_audio_file(video_id)
    if not audio_path:
        logger.error(f"No audio file found for video {video_id}")
        return None

    model, backend = _get_whisper_model(model_name)
    if model is None:
        return None

    logger.info(f"Transcribing: {audio_path.name} via {backend} (this may take a while...)")

    # Biases Whisper toward Italian food terminology and common proper nouns.
    initial_prompt = (
        "Trascrizione di un video food review italiano. "
        "Pizzeria, trattoria, forno, ristorante, osteria, gelateria, "
        "rosticceria, friggitoria, kebabbaro, pub, enoteca. "
        "Vittoria Spaziale, Roscioli, Antico Forno Roscioli, Bonci, "
        "Da Marione, Franco Pepe, Trapizzino, Pinsa Romana, "
        "Carbonara, Amatriciana, Supplì, Pizza al taglio. "
        "Roma, Napoli, Milano, Firenze, Bologna, Torino. "
        "Andiamo a mangiare, entriamo, assaggiamo, ordiniamo, prendiamo."
    )

    segments = []
    full_text_parts = []

    if backend == _BACKEND_FASTER:
        segs_gen, info = model.transcribe(
            str(audio_path),
            language=CONTENT_LANGUAGE,
            initial_prompt=initial_prompt,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        detected_lang = info.language
        for i, seg in enumerate(segs_gen):
            text = (seg.text or "").strip()
            if text:
                segments.append({
                    "id": i,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                })
                full_text_parts.append(text)
    else:
        result = model.transcribe(
            str(audio_path),
            language=CONTENT_LANGUAGE,
            verbose=False,
            fp16=(_whisper_device != "cpu"),
            initial_prompt=initial_prompt,
        )
        detected_lang = result.get("language", CONTENT_LANGUAGE)
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            if text:
                segments.append({
                    "id": seg.get("id", 0),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": text,
                })
                full_text_parts.append(text)

    transcript = {
        "video_id": video_id,
        "language": detected_lang,
        "text": " ".join(full_text_parts),
        "segments": segments,
        "source": backend,
    }

    ensure_dirs()
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    logger.info(f"Transcription complete: {len(segments)} segments, {len(transcript['text'])} chars")
    return transcript


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m scripts.transcribe_video <video_id> [model_name]")
        sys.exit(1)

    vid = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else WHISPER_DEFAULT_MODEL
    result = transcribe_audio(vid, model)
    if result:
        print(f"Transcribed {vid}: {len(result['segments'])} segments")
    else:
        print(f"Failed to transcribe {vid}")
