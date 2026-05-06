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
CONFIDENCE_THRESHOLD = 0.72

# --- Deduplication settings ---
DEDUP_DISTANCE_METERS = 200
DEDUP_NAME_SIMILARITY_THRESHOLD = 70  # thefuzz score 0-100

# --- LLM settings ---
# Recommended: Qwen2.5-32B-Instruct-Q4_K_M.gguf (~20 GB, fits in 32 GB unified memory)
# or Llama-3.3-70B-Instruct-Q2_K.gguf (~22 GB) for best Italian understanding.
# Current 8B fallback: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LLM_MODEL_FILENAME = "Qwen2.5-32B-Instruct-Q4_K_M.gguf"
LLM_CONTEXT_SIZE = 8192
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.1

# --- NER (GLiNER zero-shot, HuggingFace id) ---
# gliner_large-v2.1 offers significantly better zero-shot performance than multi-v2.1.
NER_MODEL_NAME = os.environ.get("CIBOBUONO_NER_MODEL", "urchade/gliner_large-v2.1")

# --- Video cleanup ---
MAX_CACHED_VIDEOS = 20  # Delete oldest videos when cache exceeds this

# --- Prefetch / Sliding window ---
PREFETCH_WINDOW = 20  # Max audio files to keep pre-downloaded at any time

# --- Verification ---
LLM_VERIFY = True  # Enable LLM self-verification pass on extracted locales


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
    """Load a JSON file, returning empty list if file doesn't exist or is empty."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
        # Also clean up corresponding transcript
        transcript = f.parent / f"{f.stem}_transcript.json"
        if transcript.exists():
            transcript.unlink()
    return deleted


def detect_hardware() -> dict:
    """Detect system capabilities for optimal LLM configuration.

    Auto-detects CPU cores, Apple Silicon, unified memory size, and configures:
    - n_threads: performance cores (half logical on Apple Silicon)
    - n_gpu_layers: -1 for Metal GPU, 0 for CPU-only
    - n_batch: tuned for available memory (2048 with ≥16 GB unified)
    - use_mlock: keep model locked in RAM
    """
    import os as _os
    import platform as _platform

    cpu_count = _os.cpu_count() or 4
    is_apple_silicon = (
        _platform.machine() == "arm64" and _platform.system() == "Darwin"
    )

    # Detect total system memory (GB)
    total_ram_gb = 8
    try:
        import subprocess as _sp
        if _platform.system() == "Darwin":
            r = _sp.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            total_ram_gb = int(r.stdout.strip()) / (1024 ** 3)
        else:
            total_ram_gb = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        pass

    if is_apple_silicon:
        optimal_threads = max(4, cpu_count // 2)
        n_gpu_layers = -1
        use_mlock = True
        n_batch = 2048 if total_ram_gb >= 16 else 1024
    else:
        optimal_threads = max(2, cpu_count - 2)
        n_gpu_layers = 0
        use_mlock = False
        n_batch = 512

    return {
        "n_threads": optimal_threads,
        "n_gpu_layers": n_gpu_layers,
        "use_mlock": use_mlock,
        "n_batch": n_batch,
        "cpu_count": cpu_count,
        "total_ram_gb": round(total_ram_gb, 1),
        "is_apple_silicon": is_apple_silicon,
    }


def select_optimal_models(hw: dict | None = None) -> dict:
    """
    Select the best available models based on hardware and what's already downloaded.

    LLM tiers (RAM-based, GGUF in models/):
      ≥40 GB → prefer 70B Q3 (fits with headroom)
      ≥24 GB → prefer 32B Q4  (~20 GB)
      ≥16 GB → prefer 14B Q4  (~9 GB)
       <16 GB → use 8B Q4     (~5 GB)

    Whisper tiers:
      Apple Silicon or ≥8 GB → large-v3-turbo (best quality/speed)
      <8 GB CPU-only         → medium

    NER: always gliner_large-v2.1 when available, multi-v2.1 as fallback.
    """
    if hw is None:
        hw = detect_hardware()

    ram = hw.get("total_ram_gb", 8)
    is_apple = hw.get("is_apple_silicon", False)

    # ── Whisper ──────────────────────────────────────────────────────────────
    if is_apple or ram >= 8:
        whisper_model = "large-v3-turbo"
    else:
        whisper_model = "medium"

    # ── LLM (check what's actually present in models/) ───────────────────────
    # Priority list: (filename_fragment, min_ram_gb)
    llm_candidates = [
        ("70B", 40),
        ("32B", 24),
        ("14B", 16),
        ("8B",   0),
    ]

    llm_model: Optional[Path] = None
    llm_tier = "8B"
    for fragment, min_ram in llm_candidates:
        if ram < min_ram:
            continue
        matches = sorted(MODELS_DIR.glob(f"*{fragment}*.gguf"))
        if matches:
            llm_model = matches[0]
            llm_tier = fragment
            break

    # Fall back to any GGUF if none of the above matched
    if llm_model is None:
        all_ggufs = sorted(MODELS_DIR.glob("*.gguf"))
        if all_ggufs:
            llm_model = all_ggufs[0]
            llm_tier = "unknown"

    return {
        "whisper_model": whisper_model,
        "llm_model_path": llm_model,
        "llm_tier": llm_tier,
        "hw": hw,
    }
