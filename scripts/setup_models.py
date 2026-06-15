"""
setup_models.py — Download pipeline assets that cannot live in git.

Downloads (as needed):
  • GGUF LLM weights under models/  (tier chosen from hardware profile)
  • faster-whisper cache under models/whisper/
  • GLiNER weights (Hugging Face cache, warmed on first load)

Usage:
    python -m scripts.setup_models              # auto tier from hardware
    python -m scripts.setup_models --tier 8B    # force a tier
    python -m scripts.setup_models --whisper-only
    python -m scripts.setup_models --verify   # check files, no downloads
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# The Hugging Face Xet transfer backend can stall on some networks (KVM guests,
# restrictive egress) leaving 0-byte downloads. Prefer the classic, resumable
# HTTP downloader unless the user explicitly opts back in.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from scripts.hardware import get_profile
from scripts.utils import MODELS_DIR, NER_MODEL_NAME, WHISPER_DEFAULT_MODEL, ensure_dirs

logger = logging.getLogger("setup_models")


@dataclass(frozen=True)
class GgufAsset:
    tier: str
    filename: str
    repo_id: str
    min_ram_gb: float


# Hugging Face GGUF mirrors (bartowski quantizations — widely used, stable URLs).
GGUF_CATALOG: tuple[GgufAsset, ...] = (
    GgufAsset("72B", "Qwen2.5-72B-Instruct-Q4_K_M.gguf", "bartowski/Qwen2.5-72B-Instruct-GGUF", 40),
    GgufAsset("70B", "Llama-3.3-70B-Instruct-Q4_K_M.gguf", "bartowski/Llama-3.3-70B-Instruct-GGUF", 40),
    GgufAsset("32B", "Qwen2.5-32B-Instruct-Q4_K_M.gguf", "bartowski/Qwen2.5-32B-Instruct-GGUF", 24),
    GgufAsset("27B", "gemma-3-27b-it-Q4_K_M.gguf", "bartowski/gemma-3-27b-it-GGUF", 20),
    GgufAsset("14B", "Qwen2.5-14B-Instruct-Q4_K_M.gguf", "bartowski/Qwen2.5-14B-Instruct-GGUF", 12),
    GgufAsset("8B", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", 6),
    GgufAsset("3B", "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "bartowski/Qwen2.5-3B-Instruct-GGUF", 3),
    GgufAsset("1B", "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf", "bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF", 1.5),
)


def _pick_tier(explicit: str | None = None) -> str:
    if explicit:
        t = explicit.strip().upper()
        if t.isdigit():
            t = f"{t}B"
        return t
    profile = get_profile()
    if not profile.enable_llm:
        return "none"
    return profile.llm_tier


def _asset_for_tier(tier: str) -> GgufAsset | None:
    tier = tier.upper() if tier != "none" else tier
    for asset in GGUF_CATALOG:
        if asset.tier == tier:
            return asset
    # Fuzzy: "8" → "8B"
    if not tier.endswith("B") and tier != "none":
        return _asset_for_tier(f"{tier}B")
    return None


def _hf_download(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Download a single file from Hugging Face into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.is_file() and dest.stat().st_size > 1024 * 1024:
        logger.info("Already present: %s (%.1f GB)", dest.name, dest.stat().st_size / 1024**3)
        return dest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for model downloads. "
            "Install with: pip install huggingface_hub"
        ) from exc

    logger.info("Downloading %s from %s …", filename, repo_id)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    path = Path(cached)
    if path.resolve() != dest.resolve() and path.is_file():
        shutil.copy2(path, dest)
    if not dest.is_file():
        raise RuntimeError(f"Download failed: {dest}")
    logger.info("Saved %s (%.1f GB)", dest.name, dest.stat().st_size / 1024**3)
    return dest


def download_llm(tier: str | None = None, *, models_dir: Path = MODELS_DIR) -> Path | None:
    """Download the GGUF for *tier* (or hardware auto-tier)."""
    chosen = _pick_tier(tier)
    if chosen == "none":
        logger.warning("Hardware below LLM threshold — skipping GGUF download.")
        return None

    asset = _asset_for_tier(chosen)
    if asset is None:
        logger.error("Unknown LLM tier %r. Known: %s", chosen, ", ".join(a.tier for a in GGUF_CATALOG))
        return None

    # If any GGUF already exists, keep it unless --force (caller handles force).
    existing = sorted(models_dir.glob("*.gguf"))
    if existing:
        best = max(existing, key=lambda p: p.stat().st_size)
        logger.info("GGUF already in models/: %s — skipping download (use --force-llm to re-fetch)", best.name)
        return best

    return _hf_download(asset.repo_id, asset.filename, models_dir)


def download_whisper(model_name: str = WHISPER_DEFAULT_MODEL) -> None:
    """Warm faster-whisper cache (downloads weights on first load)."""
    ensure_dirs()
    profile = get_profile()
    download_root = str(MODELS_DIR / "whisper")
    logger.info(
        "Warming Whisper cache: model=%s device=%s compute=%s",
        model_name, profile.whisper_device, profile.whisper_compute_type,
    )
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("faster-whisper not installed. pip install faster-whisper") from exc

    kwargs: dict = {
        "device": profile.whisper_device,
        "compute_type": profile.whisper_compute_type,
        "download_root": download_root,
    }
    if profile.whisper_device == "cpu" and profile.whisper_cpu_threads > 0:
        kwargs["cpu_threads"] = profile.whisper_cpu_threads
    WhisperModel(model_name, **kwargs)
    logger.info("Whisper model cached under %s", download_root)


def download_ner() -> None:
    """Warm GLiNER Hugging Face cache."""
    logger.info("Warming NER model: %s", NER_MODEL_NAME)
    try:
        import langdetect  # noqa: F401 — required by gliner at runtime
    except ImportError as exc:
        raise SystemExit(
            "langdetect not installed. pip install langdetect"
        ) from exc
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise SystemExit("gliner not installed. pip install gliner transformers torch") from exc
    from scripts.ner_candidates import _patch_gliner_tokenizer

    _patch_gliner_tokenizer()
    GLiNER.from_pretrained(NER_MODEL_NAME)
    logger.info("NER model cached (Hugging Face hub cache)")


def download_title_classifier() -> None:
    """Warm the zero-shot video title classifier (xlm-roberta-large-xnli)."""
    from scripts.fetch_videos import _ZSC_MODEL_NAME
    logger.info("Warming title classifier: %s", _ZSC_MODEL_NAME)
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("transformers not installed. pip install transformers") from exc
    pipeline("zero-shot-classification", model=_ZSC_MODEL_NAME, device=-1)
    logger.info("Title classifier cached (%s)", _ZSC_MODEL_NAME)


def verify_assets(*, models_dir: Path = MODELS_DIR) -> list[str]:
    """Return list of missing/problem descriptions (empty = OK)."""
    issues: list[str] = []
    profile = get_profile()
    if profile.enable_llm and not list(models_dir.glob("*.gguf")):
        issues.append("No *.gguf under models/ (run: python -m scripts.setup_models)")

    whisper_dir = models_dir / "whisper"
    if not whisper_dir.exists() or not any(whisper_dir.iterdir()):
        issues.append(f"Whisper cache empty under {whisper_dir}")

    try:
        import langdetect  # noqa: F401
    except ImportError:
        issues.append("langdetect not installed (required by GLiNER)")

    try:
        from gliner import GLiNER
        from scripts.ner_candidates import _patch_gliner_tokenizer

        _patch_gliner_tokenizer()
        GLiNER.from_pretrained(NER_MODEL_NAME)
    except ImportError as exc:
        issues.append(f"gliner import failed: {exc}")
    except Exception as exc:
        issues.append(f"NER model {NER_MODEL_NAME} not ready: {exc}")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download CiboBuono pipeline models")
    parser.add_argument("--tier", help="Force LLM tier (8B, 14B, 32B, …)")
    parser.add_argument("--whisper-only", action="store_true")
    parser.add_argument("--ner-only", action="store_true")
    parser.add_argument("--llm-only", action="store_true")
    parser.add_argument("--classifier-only", action="store_true", help="Download only the title classifier")
    parser.add_argument("--force-llm", action="store_true", help="Re-download GGUF even if present")
    parser.add_argument("--verify", action="store_true", help="Check assets only, no downloads")
    parser.add_argument("--whisper-model", default=WHISPER_DEFAULT_MODEL)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    ensure_dirs()
    profile = get_profile()
    logger.info(
        "Hardware: %s, %.1f GB RAM, CUDA=%s, LLM tier=%s",
        profile.platform.value, profile.total_ram_gb, profile.has_cuda, profile.llm_tier,
    )

    if args.verify:
        issues = verify_assets()
        if issues:
            for i in issues:
                logger.error("MISSING: %s", i)
            return 1
        logger.info("All model assets present.")
        return 0

    only_flags = sum([args.whisper_only, args.ner_only, args.llm_only, args.classifier_only])
    do_all = only_flags == 0

    if args.force_llm:
        for f in MODELS_DIR.glob("*.gguf"):
            f.unlink()
            logger.info("Removed %s", f.name)

    if do_all or args.llm_only:
        download_llm(args.tier)
    if do_all or args.whisper_only:
        download_whisper(args.whisper_model)
    if do_all or args.ner_only:
        download_ner()
    if do_all or args.classifier_only:
        download_title_classifier()

    issues = verify_assets()
    if issues:
        for i in issues:
            logger.warning("Post-setup: %s", i)
    logger.info("Setup models complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
