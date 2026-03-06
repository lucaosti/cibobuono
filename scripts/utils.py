"""
Shared utility functions for the pipeline.
Handles JSON I/O, logging, paths, and common operations.
"""

import json
import logging
import sys
from datetime import date
from pathlib import Path

# --- Paths ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

# --- Confidence threshold ---
CONFIDENCE_THRESHOLD = 0.65

# --- Deduplication settings ---
DEDUP_DISTANCE_METERS = 200
DEDUP_NAME_SIMILARITY_THRESHOLD = 70  # thefuzz score 0-100

# --- LLM settings ---
LLM_MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLM_CONTEXT_SIZE = 8192
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.1

# --- Video cleanup ---
MAX_CACHED_VIDEOS = 20  # Delete oldest videos when cache exceeds this

# --- Prefetch / Sliding window ---
PREFETCH_WINDOW = 20  # Max audio files to keep pre-downloaded at any time

# --- Verification ---
LLM_VERIFY = True  # Enable LLM self-verification pass on extracted locales


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
    """Save data to a JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
