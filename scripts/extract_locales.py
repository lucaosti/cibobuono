"""
extract_locales.py — Shared helpers + local LLM loader + food-relevance gate.

Venue extraction orchestration lives in scripts.extract_pipeline (NER + rules + LLM).
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import os
import re

from thefuzz import fuzz

from scripts.hardware import get_profile
from scripts.utils import (
    CACHE_DIR,
    LLM_CONTEXT_SIZE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODELS_DIR,
    resolve_llm_model_path,
    setup_logging,
)

logger = setup_logging("extract")

# Global LLM instance (loaded once per session)
_llm_instance = None


def _should_abort() -> bool:
    """Honour a pipeline graceful-shutdown request while waiting for headroom."""
    try:
        from scripts.run_pipeline import _pipeline_shutdown

        return bool(_pipeline_shutdown.get("graceful"))
    except Exception:
        return False


def get_llm():
    """Load and cache the LLM model with hardware-optimal, *runtime-adaptive* settings.

    The static :class:`~scripts.hardware.DeviceProfile` says what the machine
    *could* run; :mod:`scripts.resource_monitor` decides what actually fits the
    free RAM / VRAM right now (waiting briefly for headroom, then downgrading to
    a smaller GGUF if needed) so we never push the OS into swap or a GPU OOM.

    Returns ``None`` (without errors) when the detected hardware profile has
    ``enable_llm=False`` — typically a Raspberry Pi Zero / Pi 3 / very small VM.
    Callers (``is_food_review_video``, ``visit_classifier.classify_candidate``)
    handle a None return gracefully and degrade to NER+rules-only extraction.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    profile = get_profile()
    if not profile.enable_llm:
        logger.warning(
            "LLM disabled: hardware profile %s (%.1f GB RAM) is below the LLM "
            "threshold. Running NER+rules-only extraction.",
            profile.platform.value, profile.total_ram_gb,
        )
        return None

    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error(
            "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        )
        return None

    from scripts import resource_monitor as rm

    # If the user pinned a specific model, respect it; otherwise let the
    # resource monitor choose the largest GGUF that currently fits free memory.
    pinned = resolve_llm_model_path() if os.environ.get("CIBOBUONO_LLM_MODEL") else None
    if pinned is not None:
        models = [(pinned, pinned.stat().st_size / (1024**3))]
    else:
        models = rm.list_gguf_models(MODELS_DIR)

    if not models:
        logger.error(
            "No GGUF model found. Set CIBOBUONO_LLM_MODEL to a file path, "
            "or add *.gguf under models/. "
            "Recommended filename: see LLM_MODEL_FILENAME in scripts/utils.py."
        )
        return None

    plan = rm.plan_llm_load(profile, models, should_abort=_should_abort)
    if plan.model_path is None:
        logger.error("No loadable GGUF model: %s", plan.note)
        return None

    model_path = plan.model_path

    # Honor the project-wide LLM_CONTEXT_SIZE when the profile allows a larger
    # context than the default; otherwise clamp down (Pi-class hardware).
    n_ctx = min(LLM_CONTEXT_SIZE, profile.n_ctx) if profile.n_ctx else LLM_CONTEXT_SIZE
    logger.info("Loading LLM model: %s (%.1f GB) — %s", model_path.name, plan.size_gb, plan.note)
    logger.info(
        "  Runtime config: platform=%s, threads=%d, gpu_layers=%d (ceiling %d), "
        "batch=%d, ctx=%d, mlock=%s, pool=%s, metal=%s, cuda=%s",
        profile.platform.value, profile.n_threads, plan.n_gpu_layers,
        profile.n_gpu_layers, profile.n_batch, n_ctx, plan.use_mlock,
        plan.pool, profile.has_metal, profile.has_cuda,
    )

    llm_kwargs: dict = {
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "n_gpu_layers": plan.n_gpu_layers,
        "n_threads": profile.n_threads,
        "n_batch": profile.n_batch,
        "use_mlock": plan.use_mlock,
        "use_mmap": profile.use_mmap,
        "verbose": False,
    }

    # Flash Attention + quantized KV cache are well-tested on the Metal
    # backend; on CUDA they're also supported but more recent. Limit to Metal
    # for now to avoid surprises on older NVIDIA drivers.
    extra_kwargs: dict = {}
    if profile.has_metal:
        extra_kwargs["flash_attn"] = True
        try:
            from llama_cpp import GGML_TYPE_Q8_0
            extra_kwargs["type_k"] = GGML_TYPE_Q8_0
            extra_kwargs["type_v"] = GGML_TYPE_Q8_0
        except ImportError:
            pass

    if extra_kwargs:
        try:
            _llm_instance = Llama(**llm_kwargs, **extra_kwargs)
            logger.info("LLM loaded with flash attention + quantized KV cache (Metal)")
        except TypeError:
            _llm_instance = Llama(**llm_kwargs)
            logger.info("LLM loaded (advanced features not supported in this version)")
    else:
        _llm_instance = Llama(**llm_kwargs)
        logger.info("LLM loaded successfully")

    return _llm_instance


# ---------------------------------------------------------------------------
# Food-relevance gate — lightweight LLM check before extraction
# ---------------------------------------------------------------------------

_FOOD_SYSTEM = (
    "You classify Italian YouTube videos. "
    "Answer SI or NO — is the video about physically visiting food venues?"
)

_FOOD_USER_TEMPLATE = (
    'VIDEO TITLE: "{title}"\n'
    "{description_section}"
    'TRANSCRIPT EXCERPT (first minutes):\n"{transcript_sample}"\n\n'
    "RULES:\n"
    "- SI = the blogger clearly goes to named food businesses (restaurant, bakery, "
    "street stall, etc.) and eats/tastes there in the video.\n"
    "- NO = sports, boxing, gaming, interviews, activism, recipe-only at home, "
    "generic vlogs without venue visits, or only talking about food without being on location.\n"
    '- "Cena a 4 mani" or similar = private dinner event, NOT a venue review → NO\n'
    '- "Salviamo il X" = activism video → NO\n'
    '- Titles with "criminale" (e.g., "CITY criminale", "Forni criminali CITY") = food review → SI\n'
    '- Titles with "Hit di Franchino:" = single venue food review → SI\n'
    "- Borderline food/travel vlog with on-location eating → SI\n\n"
    "Answer with EXACTLY one word: SI or NO"
)


def is_food_review_video(
    title: str,
    transcript_text: str,
    video_description: str = "",
) -> tuple[bool, str]:
    """Use the LLM to classify whether a video is about visiting food locales."""
    llm = get_llm()
    if llm is None:
        return True, "LLM not available, skipping check"

    sample = transcript_text[:2000].strip()
    mid = len(transcript_text) // 2
    if len(transcript_text) > 2500:
        sample = (transcript_text[:1000] + " … " + transcript_text[mid : mid + 1000]).strip()
    if not sample:
        return True, "Empty transcript, skipping check"

    description_section = ""
    if video_description:
        desc_trunc = video_description[:500]
        description_section = f'DESCRIPTION: "{desc_trunc}"\n'

    user_msg = _FOOD_USER_TEMPLATE.format(
        title=title[:200].replace('"', "'"),
        transcript_sample=sample.replace('"', "'"),
        description_section=description_section,
    )

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _FOOD_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=20,
            temperature=0.05,
            stop=["\n"],
        )
        answer = response["choices"][0]["message"]["content"].strip().upper()
        first_token = re.split(r"[\s.,;:!?]+", answer, maxsplit=1)[0]
        is_yes = first_token in ("SI", "SÌ", "SÍ")

        if is_yes:
            return True, f"LLM food-check: food video ({answer})"
        return False, f"LLM food-check: NOT a food video ({answer})"

    except Exception as e:
        logger.warning(f"Food-relevance check failed: {e}, defaulting to True")
        return True, f"Error in food-check: {e}"


GENERIC_WORDS = {
    "forno", "forni", "panificio", "ristorante", "pizzeria", "trattoria",
    "bar", "pasticceria", "osteria", "taverna", "locanda", "gelateria",
    "macelleria", "salumeria", "rosticceria", "friggitoria", "caffè",
    "caffe", "cafe", "pub", "birreria", "enoteca", "pescheria",
    "supermercato", "mercato", "bottega", "negozio", "locale", "posto",
    "ristoranti", "pizzerie", "fornaio", "bakery", "restaurant",
    "panifici", "trattorie", "osterie", "gelaterie", "pasticcerie",
    "taverne", "locande", "macellerie", "salumerie", "rosticcerie",
    "friggitorie", "birrerie", "enoteche", "pescherie", "botteghe",
    "negozi", "locali", "posti", "supermercati", "mercati",
}


def _clean_locale_name(name: str) -> str:
    """Clean up transcription artifacts from locale names."""
    name = re.sub(r'[ß©®™•°§†‡¶]', '', name)
    name = re.sub(r'(.)\1{3,}', r'\1\1', name)
    name = name.strip(' .,;:!?-–—"\'()[]{}/')
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


_FILLER_WORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "del", "dei", "della", "delle", "dello", "degli",
    "da", "dal", "dai", "dalla", "dalle", "dallo", "dagli",
    "a", "al", "ai", "alla", "alle", "allo", "agli",
    "in", "nel", "nei", "nella", "nelle", "nello", "negli",
    "su", "sul", "sui", "sulla", "sulle", "sullo", "sugli",
    "con", "per", "tra", "fra", "e", "o", "de",
}


def _is_valid_locale_name(name: str) -> bool:
    """Check if extracted locale name is a real proper name, not a generic word."""
    if not name or len(name) < 3:
        return False
    if name.lower().strip() in GENERIC_WORDS:
        return False
    words = name.lower().strip().split()
    meaningful = [w for w in words if w not in _FILLER_WORDS]
    if all(w in GENERIC_WORDS for w in meaningful):
        return False
    return True


def extract_hints_from_description(description: str) -> list[str]:
    """Extract possible venue names from the YouTube video description."""
    if not description:
        return []
    hints: list[str] = []

    for m in re.finditer(
        r"\bda\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*)", description
    ):
        hints.append(m.group(1))

    for m in re.finditer(
        r"\bfritti\s+di\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*)", description
    ):
        hints.append(m.group(1))

    for m in re.finditer(
        r"(?:siamo|andiamo|entriamo|mangio|mangiamo|provo|provare|visito|visitiamo)"
        r".*?\b([A-ZÀ-Ú][a-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,4})",
        description,
        re.IGNORECASE,
    ):
        candidate = m.group(1).strip()
        if len(candidate) >= 3:
            hints.append(candidate)

    for m in re.finditer(
        r"\bin\s+via\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú0-9a-zà-ú]+)*)", description
    ):
        hints.append(f"via {m.group(1)}")

    seen: set[str] = set()
    unique: list[str] = []
    for h in hints:
        key = h.lower().strip()
        if key not in seen and len(key) >= 3:
            seen.add(key)
            unique.append(h)
    return unique


def find_best_timestamp_in_transcript(
    locale_name: str,
    transcript: dict,
) -> float | None:
    """Scan the FULL transcript for the best timestamp where a venue name appears."""
    segments = transcript.get("segments", [])
    if not segments:
        return None

    search_terms = [locale_name.lower()]
    name_tokens = [t for t in locale_name.lower().split() if len(t) >= 3]

    best_time = None
    best_score = 0.0

    for seg in segments:
        seg_text = seg.get("text", "").lower()
        seg_start = seg.get("start", 0)

        for term in search_terms:
            if term in seg_text:
                return seg_start

        if name_tokens:
            matched = 0
            for token in name_tokens:
                for word in seg_text.split():
                    if fuzz.ratio(token, word) >= 75:
                        matched += 1
                        break
            score = matched / len(name_tokens)
            if score > best_score and score >= 0.5:
                best_score = score
                best_time = seg_start

    return best_time


def rating_numeric_core(value: object) -> float | None:
    """Parse leading numeric part of a blogger rating (e.g. '8--' -> 8.0)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)", s.replace(",", "."))
    if not m:
        return None
    return float(m.group(1))


def _normalize_rating(value: object) -> str | None:
    """Normalize rating to a string preserving modifiers like '8--', '6++', '10'."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "n/a", ""):
        return None

    import re as _re
    m = _re.search(r"(\d+(?:\.\d+)?)\s*([+\-]{1,2})?", s)
    if m:
        num = float(m.group(1))
        modifier = m.group(2) or ""
        if 1 <= num <= 10:
            base = str(int(num)) if num == int(num) else str(num)
            return f"{base}{modifier}" if modifier else base
    return None


def _normalize_sentiment(value: object) -> str:
    """Normalize sentiment to one of three values."""
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ("positive", "positivo"):
            return "positive"
        if v in ("negative", "negativo"):
            return "negative"
    return "neutral"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m scripts.extract_locales <video_id>")
        sys.exit(1)

    vid = sys.argv[1]
    transcript_path = CACHE_DIR / f"{vid}_transcript.json"
    if not transcript_path.exists():
        print(f"Transcript not found: {transcript_path}")
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    from scripts.chunk_transcription import chunk_transcription
    from scripts.extract_pipeline import extract_from_video

    chunks = chunk_transcription(transcript)

    extractions, flagged = extract_from_video(
        vid, chunks, transcript=transcript
    )
    print(f"\nExtractions ({len(extractions)}):")
    for e in extractions:
        print(f"  {e['locale_name']} ({e['city']}) - rating: {e['rating']} conf: {e['confidence']}")

    print(f"\nFlagged ({len(flagged)}):")
    for f_ in flagged:
        print(f"  {f_['locale_name']} ({f_['city']}) - conf: {f_['confidence']}")
