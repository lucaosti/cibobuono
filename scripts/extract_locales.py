"""
extract_locales.py — Shared helpers + local LLM loader + food-relevance gate.

Venue extraction orchestration lives in scripts.extract_pipeline (NER + rules + LLM).
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import json
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor

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
_llm_load_future: Future | None = None
_llm_load_lock = threading.Lock()


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
    global _llm_instance, _llm_load_future

    if _llm_instance is not None:
        return _llm_instance

    with _llm_load_lock:
        if _llm_instance is not None:
            return _llm_instance
        if _llm_load_future is not None:
            fut = _llm_load_future
        else:
            result = _load_llm_impl()
            if result is not None:
                _llm_instance = result  # cache successful load
            return result

    try:
        loaded = fut.result()
    except Exception as exc:
        logger.error("Background LLM preload failed: %s", exc)
        with _llm_load_lock:
            _llm_load_future = None
        result = _load_llm_impl()
        if result is not None:
            with _llm_load_lock:
                _llm_instance = result
        return result

    with _llm_load_lock:
        if _llm_instance is None and loaded is not None:
            _llm_instance = loaded
        _llm_load_future = None
    return _llm_instance


def preload_llm(pool: ThreadPoolExecutor | None = None) -> None:
    """Start loading the LLM on a background thread (e.g. while chunking after Whisper)."""
    global _llm_load_future

    if _llm_instance is not None:
        return

    with _llm_load_lock:
        if _llm_instance is not None or _llm_load_future is not None:
            return
        target_pool = pool
        if target_pool is None:
            target_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_preload")
            _llm_load_future = target_pool.submit(_load_llm_impl)
            return
        _llm_load_future = target_pool.submit(_load_llm_impl)


def _load_llm_impl():

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

    # Flash Attention + Q8_0 quantized KV cache: supported on both Metal and
    # CUDA backends. Q8_0 KV halves cache memory (~1.3 GB for 14B at 16384
    # context vs ~2.6 GB fp16), enabling the larger context window.
    extra_kwargs: dict = {}
    if profile.has_metal or profile.has_cuda:
        extra_kwargs["flash_attn"] = True
        try:
            from llama_cpp import GGML_TYPE_Q8_0
            extra_kwargs["type_k"] = GGML_TYPE_Q8_0
            extra_kwargs["type_v"] = GGML_TYPE_Q8_0
        except ImportError:
            pass

    backend_label = "Metal" if profile.has_metal else "CUDA" if profile.has_cuda else ""
    if extra_kwargs:
        try:
            instance = Llama(**llm_kwargs, **extra_kwargs)
            logger.info("LLM loaded with flash attention + quantized KV cache (%s)", backend_label)
        except TypeError:
            instance = Llama(**llm_kwargs)
            logger.info("LLM loaded (flash attention not supported in this llama-cpp-python build)")
    else:
        instance = Llama(**llm_kwargs)
        logger.info("LLM loaded successfully")

    return instance


def release_llm() -> None:
    """Unload LLM from GPU/RAM so Whisper can use VRAM for the next video."""
    global _llm_instance, _llm_load_future
    with _llm_load_lock:
        if _llm_load_future is not None:
            _llm_load_future.cancel()
            _llm_load_future = None
        if _llm_instance is None:
            return
        _llm_instance = None
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("LLM released — VRAM freed for Whisper transcription")


# ---------------------------------------------------------------------------
# Food-relevance gate — lightweight LLM check before extraction
# ---------------------------------------------------------------------------

_FOOD_SYSTEM = (
    "You classify Italian YouTube food-vlog videos. "
    "These channels visit restaurants, pizzerias, bakeries, and street-food stalls. "
    "Answer SI or NO — is the video about physically visiting food venues?"
)

_FOOD_USER_TEMPLATE = (
    'VIDEO TITLE: "{title}"\n'
    "{description_section}"
    "{transcript_section}"
    "RULES (IMPORTANT — title outweighs ambiguous transcript excerpts):\n"
    "- SI = the blogger goes to named food businesses and eats/tastes on location.\n"
    "- SI if the title indicates a food tour/review (criminale, Forni criminali, "
    "Hit di Franchino, Cosa mangia, pizzeria/ristorante names, assaggio, giro del…).\n"
    "- SI if the description lists venues, addresses, or timestamped venue names.\n"
    "- NO = sports/boxing/gaming, activism (Salviamo…), private dinner events "
    "(Cena a N mani), recipe-only at home, no on-location eating.\n"
    "- When title/description clearly indicate food venues, answer SI even if the "
    "transcript excerpt sounds generic.\n\n"
    "Answer with EXACTLY one word: SI or NO"
)


def _food_llm_check(
    title: str,
    transcript_text: str,
    video_description: str,
) -> tuple[bool, str]:
    """LLM food gate (called only when rules are inconclusive)."""
    llm = get_llm()
    if llm is None:
        return True, "LLM not available, skipping check"

    transcript_section = ""
    if transcript_text.strip():
        sample = transcript_text[:2000].strip()
        mid = len(transcript_text) // 2
        if len(transcript_text) > 2500:
            sample = (
                transcript_text[:1000] + " … " + transcript_text[mid : mid + 1000]
            ).strip()
        transcript_section = (
            f'TRANSCRIPT EXCERPT:\n"{sample.replace(chr(34), chr(39))}"\n\n'
        )

    description_section = ""
    if video_description:
        safe_desc = video_description[:12000].replace('"', "'")
        description_section = f"DESCRIPTION (full):\n\"{safe_desc}\"\n\n"

    user_msg = _FOOD_USER_TEMPLATE.format(
        title=title[:200].replace('"', "'"),
        description_section=description_section,
        transcript_section=transcript_section,
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


def check_food_video(
    title: str,
    transcript_text: str = "",
    video_description: str = "",
    video_intel=None,
) -> tuple[bool, str]:
    """Decide if a video is a food-venue visit video.

    Order: rule-based signals (title/intel/description) → LLM → override LLM
    false negatives when rules strongly indicate food.
    """
    from scripts.video_intelligence import (
        VideoIntel,
        food_video_confidence,
        title_suggests_food,
    )

    intel = video_intel if isinstance(video_intel, VideoIntel) else None
    rules_yes, rules_reason = food_video_confidence(
        title, intel, video_description or ""
    )
    if rules_yes is True:
        return True, f"Rules: {rules_reason}"

    if not transcript_text.strip() and rules_yes is None:
        return True, "Rules inconclusive pre-transcript — proceeding to transcribe"

    is_food, llm_reason = _food_llm_check(title, transcript_text, video_description)
    if is_food:
        return True, llm_reason

    # LLM false negative guard: trust title/intel/description over a misleading excerpt.
    if intel and intel.video_type in ("single_venue", "multi_venue_tour"):
        return True, f"Override LLM NO → SI (intel: {intel.video_type})"
    if title_suggests_food(title):
        return True, "Override LLM NO → SI (food title keywords)"
    ovr, oreason = food_video_confidence(title, intel, video_description or "")
    if ovr is True:
        return True, f"Override LLM NO → SI ({oreason})"

    return False, llm_reason


def is_food_review_video(
    title: str,
    transcript_text: str,
    video_description: str = "",
    video_intel=None,
) -> tuple[bool, str]:
    """Backward-compatible wrapper around :func:`check_food_video`."""
    return check_food_video(
        title,
        transcript_text=transcript_text,
        video_description=video_description,
        video_intel=video_intel,
    )


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
