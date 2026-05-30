"""
Shared utility functions for the pipeline.
Handles JSON I/O, logging, paths, and common operations.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import logging
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

# --- Paths ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All channels in this project are Italian food YouTube; keep yt-dlp / ASR / geocoding / LLM aligned.
CONTENT_LANGUAGE = "it"
YOUTUBE_EXTRACTOR_ARGS: tuple[str, str] = ("--extractor-args", f"youtube:lang={CONTENT_LANGUAGE}")

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

CHANNELS_INPUT = PROJECT_ROOT / "channels_input.txt"
CHANNELS_JSON = DATA_DIR / "channels.json"
VIDEOS_JSON = DATA_DIR / "videos.json"
LOCALES_JSON = DATA_DIR / "locales.json"
VISITS_JSON = DATA_DIR / "visits.json"
PROCESSED_VIDEOS_JSON = DATA_DIR / "processed_videos.json"
FLAGGED_SEGMENTS_JSON = DATA_DIR / "flagged_segments.json"
SKIPPED_VIDEOS_JSON = DATA_DIR / "skipped_videos.json"
CORRECTIONS_JSON = DATA_DIR / "corrections.json"

# --- Confidence threshold ---
CONFIDENCE_THRESHOLD = 0.65

# --- Deduplication settings ---
DEDUP_DISTANCE_METERS = 200
DEDUP_NAME_SIMILARITY_THRESHOLD = 70  # thefuzz score 0-100

# --- LLM settings ---
# All prompts use create_chat_completion (model-agnostic format via GGUF chat template).
#
# 32 GB Apple Silicon:
#   Primary:  Qwen2.5-32B-Instruct-Q4_K_M.gguf   (~20 GB, strong Italian)
#   Upgrade:  gemma-3-27b-it-Q4_K_M.gguf          (~16 GB, 140+ languages, more headroom)
#   Italian:  Velvet-14B-Q4_K_M.gguf              (~9 GB, 23% Italian training data)
#   Fallback: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (~5 GB)
#
# 64 GB Apple Silicon:
#   Primary:  Qwen2.5-72B-Instruct-Q4_K_M.gguf   (~43 GB, best open multilingual)
#   Alt:      Llama-3.3-70B-Instruct-Q4_K_M.gguf (~40 GB)
LLM_MODEL_FILENAME = "Qwen2.5-32B-Instruct-Q4_K_M.gguf"
LLM_CONTEXT_SIZE = 8192
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.1

# --- NER (GLiNER zero-shot, HuggingFace id) ---
# gliner-x-large-v0.5: MT5-backbone, explicit Italian support, replaces gliner_large-v2.1.
# SLIMER-IT (expertai/LLaMAntino-3-SLIMER-IT) offers higher accuracy but requires
# LLM-class inference; use gliner-x-large-v0.5 for batch processing.
NER_MODEL_NAME = os.environ.get("CIBOBUONO_NER_MODEL", "knowledgator/gliner-x-large-v0.5")

# --- Video cleanup ---
MAX_CACHED_VIDEOS = 20  # Delete oldest videos when cache exceeds this

# --- Prefetch / Sliding window ---
PREFETCH_WINDOW = 20  # Max audio files to keep pre-downloaded at any time

# --- Continuous mode (run_pipeline --watch) ---
WATCH_POLL_INTERVAL_SECONDS = 1800  # 30 min default between catalog+process cycles
WATCH_MIN_INTERVAL_SECONDS = 60     # safety floor for --poll-interval

# --- Whisper default ---
# Updated to large-v3-turbo: same quality as large-v3 for Italian proper nouns
# at ~3× the throughput, and the default for every hardware tier ≥8 GB RAM.
# Lower tiers (Raspberry Pi, low-RAM x86) downgrade automatically via
# scripts.hardware.get_profile().whisper_model.
WHISPER_DEFAULT_MODEL = "large-v3-turbo"


def yt_dlp_command() -> list[str]:
    """Argv prefix to run yt-dlp via the current interpreter (works inside a venv)."""
    return [sys.executable, "-m", "yt_dlp"]


def resolve_llm_model_path() -> Optional[Path]:
    """
    Resolve path to a GGUF model for llama-cpp.
    Order: CIBOBUONO_LLM_MODEL env (file path), preferred filename in models/,
    else first *.gguf in models/ (sorted by name).
    """
    env = (os.environ.get("CIBOBUONO_LLM_MODEL") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    preferred = MODELS_DIR / LLM_MODEL_FILENAME
    if preferred.is_file():
        return preferred
    matches = sorted(MODELS_DIR.glob("*.gguf"))
    if matches:
        return matches[0]
    return None


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    for d in [DATA_DIR, CACHE_DIR, LOGS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "pipeline") -> logging.Logger:
    """Set up logging to both console and file."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # File handler
    today = date.today().strftime("%Y%m%d")
    fh = logging.FileHandler(LOGS_DIR / f"pipeline_{today}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_json(path: Path) -> list:
    """Load a JSON list, transparently reassembling paged files written by
    :func:`save_json_split` (sentinel ``{"_pages": N}`` + ``stem_0.json`` …)."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "_pages" in data:
            n = int(data["_pages"])
            merged: list = []
            for i in range(n):
                chunk_path = path.parent / f"{path.stem}_{i}.json"
                if not chunk_path.exists():
                    logging.getLogger("utils").warning(
                        "Missing page %s for %s", chunk_path.name, path.name
                    )
                    continue
                with open(chunk_path, "r", encoding="utf-8") as cf:
                    chunk = json.load(cf)
                if isinstance(chunk, list):
                    merged.extend(chunk)
            return merged
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        logging.getLogger("utils").warning(f"Corrupted JSON in {path}: {e}")
        return []
    except OSError as e:
        logging.getLogger("utils").error(f"Cannot read {path}: {e}")
        return []


def save_json(path: Path, data: list) -> None:
    """Save data to a JSON file with pretty formatting (atomic replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json.tmp",
        prefix=f".{path.name}.",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


JSON_SPLIT_THRESHOLD_MB = 45


def save_json_split(path: Path, data: list) -> None:
    """Save a list as JSON, splitting into pages when the payload exceeds
    JSON_SPLIT_THRESHOLD_MB.

    Small files: saved normally at ``path`` (no sentinel, no extra files).

    Large files: data is split into ``path.stem_0.json``, ``path.stem_1.json``,
    … and ``path`` is written as a sentinel ``{"_pages": N}`` so callers know
    to fetch the numbered chunks. Old single-file and old chunk files are
    cleaned up atomically.
    """
    encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    threshold = JSON_SPLIT_THRESHOLD_MB * 1024 * 1024

    # Remove any stale chunk files left from a previous split run
    for old_chunk in sorted(path.parent.glob(f"{path.stem}_[0-9]*.json")):
        old_chunk.unlink(missing_ok=True)

    if len(encoded) <= threshold:
        save_json(path, data)
        return

    # Calculate chunk size so each chunk stays under the threshold
    n_chunks = len(encoded) // threshold + 1
    chunk_size = max(1, len(data) // n_chunks)
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    for i, chunk in enumerate(chunks):
        save_json(path.parent / f"{path.stem}_{i}.json", chunk)

    # Write sentinel in place of the monolithic file
    sentinel = {"_pages": len(chunks)}
    fd, tmp = tempfile.mkstemp(
        suffix=".json.tmp", prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sentinel, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logging.getLogger("utils").info(
        f"Split {path.name} into {len(chunks)} pages "
        f"({len(encoded) / 1024 / 1024:.1f} MB total)"
    )


def today_str() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return date.today().strftime("%Y-%m-%d")


def cleanup_cache(max_files: int = MAX_CACHED_VIDEOS) -> list[str]:
    """
    Delete oldest cached media files when cache exceeds max_files.
    Returns list of deleted file paths.
    """
    ensure_dirs()
    media_extensions = {".mp4", ".mp3", ".wav", ".webm", ".m4a"}
    media_files = [
        f for f in CACHE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in media_extensions
    ]

    if len(media_files) <= max_files:
        return []

    # Sort by modification time, oldest first
    media_files.sort(key=lambda f: f.stat().st_mtime)
    to_delete = media_files[:len(media_files) - max_files]
    deleted = []
    for f in to_delete:
        f.unlink()
        deleted.append(str(f))
        stem = f.stem
        companions = [
            f.parent / f"{stem}_transcript.json",
            f.parent / f"{stem}_metadata.json",
            f.parent / f"{stem}_description.txt",
        ]
        for companion in companions:
            if companion.exists():
                companion.unlink()
        for sub in f.parent.glob(f"{stem}_subs.*"):
            sub.unlink()
    return deleted


def detect_hardware() -> dict:
    """Legacy hardware-detection shim — kept for backwards compatibility.

    Delegates to :func:`scripts.hardware.get_profile` and returns the subset of
    fields that legacy callers expect (``n_threads``, ``n_gpu_layers``,
    ``use_mlock``, ``n_batch``, ``cpu_count``, ``total_ram_gb``,
    ``is_apple_silicon``). New code should call ``get_profile()`` directly.
    """
    from .hardware import get_profile

    profile = get_profile()
    return {
        "n_threads": profile.n_threads,
        "n_gpu_layers": profile.n_gpu_layers,
        "use_mlock": profile.use_mlock,
        "n_batch": profile.n_batch,
        "n_ctx": profile.n_ctx,
        "use_mmap": profile.use_mmap,
        "cpu_count": profile.cpu_count_logical,
        "total_ram_gb": profile.total_ram_gb,
        "is_apple_silicon": profile.platform.value == "apple_silicon",
        "platform": profile.platform.value,
    }


# LLM tier definitions — single source of truth lives in scripts.hardware.
# Re-exported here under the legacy name so existing imports keep working.
from .hardware import LLM_TIERS as LLM_TIER_CANDIDATES  # noqa: E402


def resolve_llm_model_path_for_tier(tier: str) -> Optional[Path]:
    """Return the best GGUF in ``models/`` whose filename contains *tier*
    (e.g. "32B"). Falls back to the first GGUF available; returns None when
    the directory is empty or the tier is "none"."""
    if tier == "none":
        return None
    if tier:
        matches = sorted(MODELS_DIR.glob(f"*{tier}*.gguf"))
        if matches:
            return matches[0]
    all_ggufs = sorted(MODELS_DIR.glob("*.gguf"))
    return all_ggufs[0] if all_ggufs else None


def select_optimal_models(hw: dict | None = None) -> dict:
    """Pick the best Whisper + LLM models for the detected hardware profile.

    The Whisper choice and LLM tier come from :mod:`scripts.hardware`; this
    function only resolves the tier to an actual GGUF file in ``models/``.

    Tiers (RAM-based):
        ≥40 GB → 72B / 70B   (Qwen2.5-72B, Llama-3.3-70B)
        ≥24 GB → 32B / 27B   (Qwen2.5-32B, Gemma-3-27B)
        ≥12 GB → 14B         (Velvet-14B)
        ≥6  GB → 8B          (Llama-3.1-8B)
        ≥3  GB → 3B          (Phi-3-mini, Qwen2.5-3B) — Raspberry Pi 5 8 GB
        ≥1.5GB → 1B          (TinyLlama, Qwen2.5-1.5B) — Raspberry Pi 4 4 GB
        else    → none       (NER+rules-only extraction)

    The ``hw`` argument is accepted for backwards compatibility but ignored —
    we always go through ``get_profile()`` so the answer is deterministic.
    """
    from .hardware import get_profile

    profile = get_profile()
    llm_path = resolve_llm_model_path_for_tier(profile.llm_tier)

    return {
        "whisper_model": profile.whisper_model,
        "llm_model_path": llm_path,
        "llm_tier": profile.llm_tier,
        "enable_llm": profile.enable_llm,
        "profile": profile,
        # Back-compat: legacy callers (validate_data, dashboard) read hw.* keys
        "hw": detect_hardware(),
    }
