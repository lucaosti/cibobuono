"""
ner_candidates.py — Zero-shot NER for food-venue candidates (GLiNER) + heuristic fallback.

Maps character spans in chunk text to approximate segment start times for visit classification.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"


import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from scripts.utils import NER_MODEL_NAME, setup_logging

logger = setup_logging("ner_candidates")

_gliner_model = None
_gliner_predict_lock = threading.Lock()
_gliner_load_lock = threading.Lock()

# GLiNER often includes trigger words in the span; strip common Italian prefixes.
_VENUE_PREFIX = re.compile(
    r"(?i)^(siamo|andiamo|eccoci|qui|oggi)\s+da\s+",
)


def _refine_candidate_text(raw: str) -> str:
    s = raw.strip()
    s = _VENUE_PREFIX.sub("", s).strip()
    return s

# GLiNER labels — broad semantic phrases so the model generalises to venue types
# not explicitly listed (taverna, kebabbaro, focacceria, etc.).  Fewer labels
# also speed up GLiNER inference.  The LLM validation step downstream filters
# false positives, so higher recall here is more important than precision.
NER_LABELS = [
    # Venue labels (5 broad categories)
    "restaurant, trattoria, osteria or food venue",
    "pizzeria",
    "bakery, pastry shop or gelateria",
    "bar, cafe, wine bar or enoteca",
    "street food stall, rosticceria, friggitoria or food market",
    # Context labels (unchanged)
    "city",
    "neighborhood",
    "country",
    "person",
    "brand",
    "food dish",
]

VENUE_LABELS = frozenset({
    "restaurant, trattoria, osteria or food venue",
    "pizzeria",
    "bakery, pastry shop or gelateria",
    "bar, cafe, wine bar or enoteca",
    "street food stall, rosticceria, friggitoria or food market",
})

CONTEXT_LABELS = frozenset({
    "city",
    "neighborhood",
    "country",
    "person",
    "brand",
    "food dish",
})


@dataclass
class Candidate:
    name: str
    label: str
    start_char: int
    end_char: int
    start_time: float
    chunk_index: int
    ner_score: float


def _build_char_ranges(
    full_text: str,
    segment_timestamps: list[tuple[float, str]],
) -> list[tuple[int, int, float]]:
    """Map (char_start, char_end_exclusive, segment_start_time) for each segment in full_text."""
    ranges: list[tuple[int, int, float]] = []
    pos = 0
    for i, (t, seg_text) in enumerate(segment_timestamps):
        if i > 0:
            if pos < len(full_text) and full_text[pos] == " ":
                pos += 1
        start_c = pos
        end_c = pos + len(seg_text)
        ranges.append((start_c, end_c, float(t)))
        pos = end_c
    return ranges


def _char_span_to_start_time(
    span_start: int,
    span_end: int,
    ranges: list[tuple[int, int, float]],
    fallback: float,
) -> float:
    """Pick segment start time for entity span overlapping chunk text."""
    for start_c, end_c, t in ranges:
        if span_start < end_c and span_end > start_c:
            return t
    return fallback


def _patch_gliner_tokenizer() -> None:
    """Avoid GLiNER eager-loading mecab/janome/jieba at init (breaks on minimal installs).

    Italian food vlogs only need langdetect + whitespace fallback for token spans.
    """
    try:
        from gliner.data_processing import tokenizer as gt
    except ImportError:
        return
    if getattr(gt, "_cibobuono_patched", False):
        return

    class _ItalianMultiLangSplitter(gt.TokenSplitterBase):
        def __init__(self, logging: bool = False, use_spacy: bool = True):
            if not gt.is_module_available("langdetect"):
                raise ImportError("Please install langdetect with: `pip install langdetect`")
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 0
            self.detect = detect
            self.lang2splitter: dict = {}
            self.universal_splitter = gt.WhitespaceTokenSplitter()
            self.logging = logging

        def __call__(self, text):
            lang = "unknown"
            splitter = self.universal_splitter
            try:
                lang = self.detect(text)
            except Exception:
                pass
            else:
                splitter = self.lang2splitter.get(lang, self.universal_splitter)
                if lang not in self.lang2splitter:
                    self.lang2splitter[lang] = splitter
            if self.logging and lang != "unknown":
                logger.debug(f"GLiNER tokenizer lang={lang}")
            yield from splitter(text)

    gt.MultiLangWordsSplitter = _ItalianMultiLangSplitter  # type: ignore[misc,assignment]
    gt._cibobuono_patched = True


_GLINER_GPU_MIN_FREE_VRAM_GB = 2.5  # GLiNER x-large is ~1.5 GB on GPU


def _gliner_use_cpu() -> bool:
    """Return True if GLiNER should be pinned to CPU.

    In auto mode (default): use GPU when free VRAM >= 2.5 GB at load time.
    This covers the typical case where the LLM is already loaded (8–9 GB)
    but 7+ GB of VRAM is still free — more than enough for GLiNER x-large.
    """
    mode = os.environ.get("CIBOBUONO_GLINER_CPU", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    try:
        from scripts.hardware import get_profile
        if not get_profile().has_cuda:
            return False  # Non-CUDA: no GPU to reserve; let GLiNER choose
        # Check runtime free VRAM so we don't pin unnecessarily when there is
        # enough headroom (e.g. 7 GB free after loading a 14B LLM on 16 GB).
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info(0)
                if free_bytes / 1024**3 >= _GLINER_GPU_MIN_FREE_VRAM_GB:
                    return False  # plenty of room → GPU
        except Exception:
            pass
        return True  # conservative fallback: CPU
    except Exception:
        return False


def _maybe_pin_gliner_cpu(model) -> None:
    """Pin GLiNER to CPU when VRAM headroom is insufficient for GPU inference."""
    if not _gliner_use_cpu():
        logger.info("GLiNER on GPU — sufficient free VRAM for parallel inference")
        return
    try:
        import torch

        model.to(torch.device("cpu"))
        logger.info("GLiNER pinned to CPU — insufficient free VRAM for GPU inference")
    except Exception as e:
        logger.debug(f"GLiNER CPU pin skipped: {e}")


def get_gliner():
    """Lazy-load GLiNER model (cached process-wide, thread-safe)."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model
    with _gliner_load_lock:
        # Re-check under lock: another NER worker thread may have loaded it
        # while we waited (avoids loading the model N times concurrently).
        if _gliner_model is not None:
            return _gliner_model
        try:
            from gliner import GLiNER
        except ImportError:
            logger.warning("gliner not installed; using heuristic NER fallback only")
            return None
        try:
            _patch_gliner_tokenizer()
            logger.info(f"Loading GLiNER model: {NER_MODEL_NAME}")
            model = GLiNER.from_pretrained(NER_MODEL_NAME)
            _maybe_pin_gliner_cpu(model)
            _gliner_model = model
            return _gliner_model
        except Exception as e:
            logger.warning(f"GLiNER load failed ({e}); heuristic fallback only")
            return None


# Only used when GLiNER is completely unavailable. Requires an explicit venue
# trigger immediately before a capitalized name so we do not grab arbitrary
# capitalized words (a major false-positive source).
_NAME_RE = r"[A-ZÀ-Ú][\wà-ú'’]+(?:\s+[A-ZÀ-Ú][\wà-ú'’]+){0,2}"

# Venue-type nouns: the trigger word is PART of the name (Forno Roscioli).
# Case-insensitivity is scoped to the trigger so the name pattern stays
# case-SENSITIVE (only real Capitalized proper nouns are captured).
_TRIGGER_TYPE = re.compile(
    r"\b(?i:(pizzeria|trattoria|ristorante|osteria|forno|panificio|pasticceria|"
    r"gelateria|friggitoria|rosticceria|enoteca|hosteria|locanda))\s+(" + _NAME_RE + r")"
)
# Prepositions: the trigger word is NOT part of the name (da Michele -> Michele).
_TRIGGER_PREP = re.compile(
    r"\b(?i:da|al|alla|allo|dal|dalla|presso)\s+(" + _NAME_RE + r")"
)


def _heuristic_venue_spans(text: str) -> list[tuple[str, int, int, float]]:
    """Trigger-anchored heuristic, used ONLY when GLiNER cannot load.

    Requires a venue trigger word right before the candidate name, so it cannot
    pick up arbitrary capitalized tokens.
    """
    from scripts.extract_locales import GENERIC_WORDS, _is_valid_locale_name

    out: list[tuple[str, int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for m in _TRIGGER_TYPE.finditer(text):
        name = f"{m.group(1).capitalize()} {m.group(2).strip()}"
        start, end = m.start(0), m.end(0)
        if not _is_valid_locale_name(name) or name.lower() in GENERIC_WORDS:
            continue
        seen.add((start, end))
        out.append((name, start, end, 0.5))

    for m in _TRIGGER_PREP.finditer(text):
        name = m.group(1).strip()
        start, end = m.start(1), m.end(1)
        if any(s <= start < e for s, e in seen):
            continue
        if not _is_valid_locale_name(name) or name.lower() in GENERIC_WORDS:
            continue
        out.append((name, start, end, 0.5))
    return out


def extract_chunk_candidates(
    chunk: dict,
    *,
    threshold: float = 0.5,
) -> tuple[list[Candidate], list[Candidate]]:
    """
    Run NER on one transcription chunk.

    Returns:
        (venue_candidates, context_entities) — context used for blacklisting in visit_classifier.
    """
    text = (chunk.get("text") or "").strip()
    chunk_index = int(chunk.get("chunk_index", 0))
    seg_ts: list[tuple[float, str]] = chunk.get("segment_timestamps") or []
    t_fallback = float(chunk.get("start_time", 0.0))

    if len(text) < 12:
        return [], []

    ranges = _build_char_ranges(text, seg_ts) if seg_ts else []

    model = get_gliner()
    venues: list[Candidate] = []
    context: list[Candidate] = []

    if model is not None:
        try:
            with _gliner_predict_lock:
                raw = model.predict_entities(text, NER_LABELS, threshold=threshold)
        except Exception as e:
            logger.warning(f"GLiNER predict failed: {e}")
            raw = []

        for ent in raw:
            if not isinstance(ent, dict):
                continue
            label = str(ent.get("label", "")).strip().lower()
            name = _refine_candidate_text(str(ent.get("text", "")))
            if not name or len(name) < 2:
                continue
            start = int(ent.get("start", 0))
            end = int(ent.get("end", 0))
            score = float(ent.get("score", 0.5))

            st = (
                _char_span_to_start_time(start, end, ranges, t_fallback)
                if ranges
                else t_fallback
            )

            c = Candidate(
                name=name,
                label=label,
                start_char=start,
                end_char=end,
                start_time=st,
                chunk_index=chunk_index,
                ner_score=score,
            )
            if label in VENUE_LABELS:
                venues.append(c)
            elif label in CONTEXT_LABELS:
                context.append(c)

        # Union trigger-anchored heuristics (high precision) — catches "siamo da X"
        # when GLiNER misses or mislabels ASR-garbled names.
        seen_spans = {(c.start_char, c.end_char) for c in venues}
        for name, start, end, score in _heuristic_venue_spans(text):
            if any(abs(start - s) < 3 and abs(end - e) < 3 for s, e in seen_spans):
                continue
            st = (
                _char_span_to_start_time(start, end, ranges, t_fallback)
                if ranges
                else t_fallback
            )
            venues.append(
                Candidate(
                    name=name,
                    label="restaurant",
                    start_char=start,
                    end_char=end,
                    start_time=st,
                    chunk_index=chunk_index,
                    ner_score=score,
                )
            )
            seen_spans.add((start, end))
        return venues, context

    # Heuristic fallback: treat all as venue candidates, no structured context
    for name, start, end, score in _heuristic_venue_spans(text):
        st = (
            _char_span_to_start_time(start, end, ranges, t_fallback)
            if ranges
            else t_fallback
        )
        venues.append(
            Candidate(
                name=name,
                label="restaurant",
                start_char=start,
                end_char=end,
                start_time=st,
                chunk_index=chunk_index,
                ner_score=score,
            )
        )
    return venues, context


def extract_all_chunks_candidates(
    chunks: list[dict],
    *,
    max_workers: int | None = None,
) -> dict[int, tuple[list[Candidate], list[Candidate]]]:
    """Run NER on all chunks in parallel (CPU threads — GLiNER stays off GPU)."""
    if max_workers is None:
        max_workers = int(os.environ.get("CIBOBUONO_NER_WORKERS", "4"))

    if len(chunks) <= 1 or max_workers <= 1:
        return {
            int(c.get("chunk_index", i)): extract_chunk_candidates(c)
            for i, c in enumerate(chunks)
        }

    out: dict[int, tuple[list[Candidate], list[Candidate]]] = {}
    workers = min(max_workers, len(chunks))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gliner") as pool:
        futs = {pool.submit(extract_chunk_candidates, c): c for c in chunks}
        for fut in as_completed(futs):
            chunk = futs[fut]
            idx = int(chunk.get("chunk_index", 0))
            try:
                out[idx] = fut.result()
            except Exception as exc:
                logger.warning("NER failed for chunk %d: %s", idx, exc)
                out[idx] = ([], [])
    logger.info(f"Parallel NER: {len(chunks)} chunks with {workers} workers")
    return out
