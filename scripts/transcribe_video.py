"""
transcribe_video.py — Transcribe video audio to text.

Strategy (in priority order):
1. Try downloading YouTube's own subtitles (auto-generated or manual) via yt-dlp.
   YouTube's ASR models are far more accurate than local Whisper, especially for
   Italian proper nouns (restaurant names, addresses, etc.).
2. If YouTube subtitles are not available, fall back to local Whisper transcription.

Saves transcriptions as JSON in cache/ with segment-level timestamps.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import re
import subprocess
from pathlib import Path

from scripts.utils import (
    CACHE_DIR,
    CONTENT_LANGUAGE,
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
_whisper_device = None  # cached so transcribe_audio avoids a second probe


# ---------------------------------------------------------------------------
# YouTube subtitles (preferred — much higher quality)
# ---------------------------------------------------------------------------

def _download_youtube_subs(video_id: str) -> Path | None:
    """
    Download YouTube subtitles for a video via yt-dlp.
    
    Tries (in order):
    1. Manual Italian subtitles (human-written — best quality)
    2. Auto-generated Italian subtitles (YouTube ASR — very good)
    
    Returns path to the downloaded .vtt/.srt file, or None if unavailable.
    """
    ensure_dirs()
    output_template = str(CACHE_DIR / f"{video_id}_subs")

    # First try manual Italian subs, then auto-generated
    for sub_args in [
        # Manual subs only
        ["--write-subs", "--no-write-auto-subs", "--sub-langs", CONTENT_LANGUAGE],
        # Auto-generated subs
        ["--write-auto-subs", "--sub-langs", CONTENT_LANGUAGE],
    ]:
        try:
            result = subprocess.run(
                [
                    *yt_dlp_command(),
                    "--skip-download",
                    *sub_args,
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

            # yt-dlp appends language code and extension
            for suffix in [".it.vtt", ".it.srt"]:
                candidate = CACHE_DIR / f"{video_id}_subs{suffix}"
                if candidate.exists() and candidate.stat().st_size > 100:
                    sub_type = "manual" if "--no-write-auto-subs" in sub_args else "auto"
                    logger.info(
                        f"Downloaded YouTube subs ({sub_type}) for {video_id}: "
                        f"{candidate.name} ({candidate.stat().st_size} bytes)"
                    )
                    return candidate

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Sub download attempt failed for {video_id}: {e}")
            continue

    logger.info(f"No YouTube subtitles available for {video_id}")
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
        "source": "youtube_subs",
    }

    logger.info(
        f"Parsed YouTube subs: {len(segments)} segments, "
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
# Local Whisper transcription (fallback)
# ---------------------------------------------------------------------------


def _detect_whisper_device() -> str:
    """Detect the best available device for Whisper inference."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def _get_whisper_model(model_name: str = "large-v3"):
    """
    Load and cache the Whisper model. Tries faster-whisper first (CTranslate2,
    2-4× faster on Apple Silicon Metal), falls back to openai-whisper.
    Returns (model, backend) tuple.
    """
    global _whisper_model, _whisper_model_name, _whisper_backend, _whisper_device

    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model, _whisper_backend

    from scripts.utils import MODELS_DIR
    download_root = str(MODELS_DIR / "whisper")
    device = _detect_whisper_device()
    # faster-whisper uses "auto" on Apple Silicon to pick Metal automatically
    fw_device = "auto" if device == "mps" else device
    compute_type = "float16" if device != "cpu" else "int8"

    try:
        from faster_whisper import WhisperModel
        logger.info(
            f"Loading faster-whisper model: {model_name} "
            f"(device={fw_device}, compute={compute_type})"
        )
        _whisper_model = WhisperModel(
            model_name,
            device=fw_device,
            compute_type=compute_type,
            download_root=download_root,
        )
        _whisper_model_name = model_name
        _whisper_backend = _BACKEND_FASTER
        _whisper_device = device
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

    logger.info(f"Loading openai-whisper model: {model_name} on {device}")
    _whisper_model = whisper.load_model(model_name, download_root=download_root, device=device)
    _whisper_model_name = model_name
    _whisper_backend = _BACKEND_OPENAI
    _whisper_device = device
    return _whisper_model, _whisper_backend


def find_audio_file(video_id: str) -> Path | None:
    """Find the cached audio file for a given video ID."""
    for ext in [".wav", ".mp3", ".m4a", ".webm"]:
        path = CACHE_DIR / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def transcribe_audio(video_id: str, model_name: str = "large-v3") -> dict | None:
    """
    Get transcription for a video.
    
    Strategy:
    1. Return cached transcript if available
    2. Try downloading YouTube subtitles (auto-generated or manual)
    3. Fall back to local Whisper transcription
    
    Args:
        video_id: YouTube video ID
        model_name: Whisper model size (used only as fallback)
    
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

    # --- Strategy 1: Try YouTube subtitles first ---
    subs_path = _download_youtube_subs(video_id)
    if subs_path:
        transcript = _parse_vtt(subs_path)
        if transcript and len(transcript.get("segments", [])) > 0:
            # Cache the transcript
            ensure_dirs()
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
            logger.info(
                f"✓ Using YouTube subtitles for {video_id} "
                f"({len(transcript['segments'])} segments)"
            )
            return transcript

    # --- Strategy 2: Fall back to local Whisper ---
    logger.info(f"Falling back to Whisper ({model_name}) for {video_id}")
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
    model = sys.argv[2] if len(sys.argv) > 2 else "large-v3"
    result = transcribe_audio(vid, model)
    if result:
        print(f"Transcribed {vid}: {len(result['segments'])} segments")
    else:
        print(f"Failed to transcribe {vid}")
