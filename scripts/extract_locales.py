"""
extract_locales.py — Use local LLM to extract locale info from transcription chunks.

Uses llama-cpp-python with a GGUF model (Mistral 7B) to extract:
- Locale name, address, city
- Rating (1-10 scale)
- Sentiment (positive/neutral/negative)
- Rubrica
- Confidence score

Low-confidence extractions are sent to flagged_segments.json.
"""

import json
import re

from scripts.utils import (
    CACHE_DIR,
    CONFIDENCE_THRESHOLD,
    LLM_CONTEXT_SIZE,
    LLM_MAX_TOKENS,
    LLM_MODEL_FILENAME,
    LLM_TEMPERATURE,
    LLM_VERIFY,
    MODELS_DIR,
    detect_hardware,
    setup_logging,
)

logger = setup_logging("extract")

# Global LLM instance (loaded once per session)
_llm_instance = None


def get_llm():
    """Load and cache the LLM model with hardware-optimal settings."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error(
            "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        )
        return None

    model_path = MODELS_DIR / LLM_MODEL_FILENAME
    if not model_path.exists():
        logger.error(
            f"LLM model not found at {model_path}. "
            f"Download a GGUF model and place it in {MODELS_DIR}/. "
            f"Recommended: Meta-Llama-3.1-8B-Instruct Q4_K_M from HuggingFace."
        )
        return None

    hw = detect_hardware()
    logger.info(f"Loading LLM model: {model_path.name}")
    logger.info(
        f"  Hardware config: threads={hw['n_threads']}, "
        f"gpu_layers={hw['n_gpu_layers']}, batch={hw['n_batch']}, "
        f"mlock={hw['use_mlock']}, apple_silicon={hw['is_apple_silicon']}"
    )

    llm_kwargs = {
        "model_path": str(model_path),
        "n_ctx": LLM_CONTEXT_SIZE,
        "n_gpu_layers": hw["n_gpu_layers"],
        "n_threads": hw["n_threads"],
        "n_batch": hw["n_batch"],
        "use_mlock": hw["use_mlock"],
        "verbose": False,
    }

    if hw["is_apple_silicon"]:
        # Metal GPU: enable flash attention + quantized KV cache (Q8_0)
        # to reduce VRAM usage and increase throughput on unified memory.
        extra_kwargs: dict = {}
        extra_kwargs["flash_attn"] = True
        try:
            from llama_cpp import GGML_TYPE_Q8_0
            extra_kwargs["type_k"] = GGML_TYPE_Q8_0
            extra_kwargs["type_v"] = GGML_TYPE_Q8_0
        except ImportError:
            pass
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

FOOD_RELEVANCE_PROMPT = """<|start_header_id|>system<|end_header_id|>

You classify Italian YouTube videos. Answer SI or NO — is the video about physically visiting food venues?<|eot_id|><|start_header_id|>user<|end_header_id|>

VIDEO TITLE: "{title}"
{description_section}
TRANSCRIPT EXCERPT (first minutes):
"{transcript_sample}"

RULES:
- SI = the blogger physically goes to a named food venue and eats there.
- NO = sports, boxing, gaming, interviews, non-food vlogs, social causes.
- "Cena a 4 mani" or similar = private dinner event, NOT a venue review → NO
- "Salviamo il X" = activism video → NO
- Titles with "criminale" (e.g., "CITY criminale", "Forni criminali CITY") = food review → SI
- Titles with "Hit di Franchino:" = single venue food review → SI
- When in doubt → SI

Answer with EXACTLY one word: SI or NO<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def is_food_review_video(
    title: str,
    transcript_text: str,
    video_description: str = "",
) -> tuple[bool, str]:
    """Use the LLM to classify whether a video is about visiting food locales.

    Returns:
        (is_food, reason) — True if the video appears to be about food reviews.
        On error or ambiguity, defaults to True (better to process than miss).
    """
    llm = get_llm()
    if llm is None:
        return True, "LLM not available, skipping check"

    # Use a sample of the transcript (first ~2000 chars)
    sample = transcript_text[:2000].strip()
    if not sample:
        return True, "Empty transcript, skipping check"

    description_section = ""
    if video_description:
        desc_trunc = video_description[:500]
        description_section = f'\nDESCRIPTION: "{desc_trunc}"\n'

    prompt = FOOD_RELEVANCE_PROMPT.format(
        title=title[:200],
        transcript_sample=sample,
        description_section=description_section,
    )

    try:
        response = llm(
            prompt,
            max_tokens=20,
            temperature=0.05,
            stop=["\n"],
        )
        answer = response["choices"][0]["text"].strip().upper()

        # Match only standalone "SI" / "SÌ" — not substrings like "viSIte".
        # The LLM is instructed to answer with a single word, but sometimes
        # emits a full sentence.  We check the first token only.
        first_token = re.split(r"[\s.,;:!?]+", answer, maxsplit=1)[0]
        is_yes = first_token in ("SI", "SÌ", "SÍ")

        if is_yes:
            return True, f"LLM food-check: food video ({answer})"
        return False, f"LLM food-check: NOT a food video ({answer})"

    except Exception as e:
        logger.warning(f"Food-relevance check failed: {e}, defaulting to True")
        return True, f"Error in food-check: {e}"


EXTRACTION_PROMPT = """<|start_header_id|>system<|end_header_id|>

You extract food venue data from Italian YouTube video transcripts. Return ONLY valid JSON arrays.<|eot_id|><|start_header_id|>user<|end_header_id|>

VIDEO TITLE: "{title}"
VIDEO TYPE: {video_type}
{venue_hints_section}{description_section}{few_shot_section}
TRANSCRIPT ({start} to {end}):
"{text}"

YOUR TASK: Identify food venues the blogger PHYSICALLY VISITS and EATS at in this segment.

CRITICAL RULES:
1. A venue is "visited" when the blogger enters, orders food, tastes food, or describes dishes being served.
2. DO NOT extract venues that are only mentioned, compared, or announced for future visits ("prossima tappa", "ci andiamo dopo").
3. VENUE HINTS (if provided above) are STRONG signals — if you hear a phonetic match to a hint in this segment, that venue is almost certainly correct. Use the hint's exact name.
4. Italian ASR is very garbled. A venue name like "PezZ de Pane" might appear as "pezze di pane". Listen for phonetic similarity.
5. RATING: Only the blogger's OVERALL venue rating as stated (e.g., "gli do un 8", "8 pieno", "da dieci", "6++"). Preserve exact wording: "8", "8--", "6++", "10". Use null if not stated.
6. SENTIMENT: "positive" = praise, "negative" = complaints, "neutral" = mixed.
7. NOTES: Briefly list food items tried and any per-item comments (e.g., "maritozzo excellent, pizza bianca 8/10, cornetto average").
8. Names under 3 characters or generic words alone ("forno", "bar") are invalid.
9. DO NOT extract: person names, landmarks, neighborhoods, video title text as venue name.
10. If no venue is being visited in this segment: return [].

RESPOND with ONLY a JSON array:
[{{"locale_name": "Venue Name", "address": "", "city": "Roma", "category": ["pizzeria"], "rating": null, "sentiment": "positive", "notes": "maritozzo excellent, pizza bianca 8/10", "rubrica": "", "confidence": 0.85}}]

If no venues visited: []
{rubriche_section}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


# Post-extraction filter: reject generic/invalid names
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
    # Remove non-printable and weird Unicode artifacts
    name = re.sub(r'[ß©®™•°§†‡¶]', '', name)
    # Remove excessive repeated chars (e.g. "Naaapola" stays, but "Naaaaapola" becomes "Naapola")
    name = re.sub(r'(.)\1{3,}', r'\1\1', name)
    # Remove leading/trailing punctuation and whitespace
    name = name.strip(' .,;:!?-–—"\'()[]{}/')
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


# Italian filler words (articles, prepositions, contractions)
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
    """Check if extracted locale name is a real proper name, not a generic word.

    Keeps names like "Forno Roscioli", "Pizzeria Napoli", "Da Marione".
    Rejects names composed entirely of generic category words and filler,
    e.g. "la pizzeria", "il ristorante", "forni e panifici".
    """
    if not name or len(name) < 3:
        return False
    # Reject if it's just a single generic category word ("forno", "pizzeria")
    if name.lower().strip() in GENERIC_WORDS:
        return False
    # Split and check: at least one word must be neither generic nor filler
    words = name.lower().strip().split()
    meaningful = [w for w in words if w not in _FILLER_WORDS]
    if all(w in GENERIC_WORDS for w in meaningful):
        return False
    return True


def extract_hints_from_description(description: str) -> list[str]:
    """Extract possible venue names from the YouTube video description.

    Descriptions often contain the real venue name (sometimes with address)
    in plain text.  This lightweight regex/heuristic pass pulls out likely
    proper nouns so they can be fed as hints to the LLM, dramatically
    improving recall on garbled transcripts.

    Returns a list of candidate venue name strings.
    """
    if not description:
        return []
    hints: list[str] = []

    # Pattern 1: "da <Name>" / "da <Name> in <address>"
    for m in re.finditer(
        r"\bda\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*)", description
    ):
        hints.append(m.group(1))

    # Pattern 2: "di <Name>" when preceded by food context
    for m in re.finditer(
        r"\bfritti\s+di\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*)", description
    ):
        hints.append(m.group(1))

    # Pattern 3: quoted or capitalised venue names like "Sant'Isidoro"
    for m in re.finditer(
        r"(?:siamo|andiamo|entriamo|mangio|mangiamo|provo|provare|visito|visitiamo)"
        r".*?\b([A-ZÀ-Ú][a-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,4})",
        description,
        re.IGNORECASE,
    ):
        candidate = m.group(1).strip()
        if len(candidate) >= 3:
            hints.append(candidate)

    # Pattern 4: "in via <Street> a <City>" — extract surrounding proper nouns
    for m in re.finditer(
        r"\bin\s+via\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú0-9a-zà-ú]+)*)", description
    ):
        hints.append(f"via {m.group(1)}")

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for h in hints:
        key = h.lower().strip()
        if key not in seen and len(key) >= 3:
            seen.add(key)
            unique.append(h)
    return unique


def _find_mention_timestamp(
    locale_name: str,
    segment_timestamps: list[tuple[float, str]],
    chunk_start: float,
) -> float:
    """Find the segment timestamp where the locale is most likely mentioned.

    Searches chunk segments for the best fuzzy match of the locale name and
    returns that segment's start time.  Falls back to chunk_start if no
    reasonable match is found.
    """
    if not segment_timestamps or not locale_name:
        return chunk_start

    name_lower = locale_name.lower()
    tokens = name_lower.split()

    best_score = 0.0
    best_time = chunk_start

    for seg_start, seg_text in segment_timestamps:
        seg_lower = seg_text.lower()

        # Exact substring match
        if name_lower in seg_lower:
            return seg_start

        # Token overlap: count how many name tokens appear in the segment
        matched = sum(1 for t in tokens if t in seg_lower)
        score = matched / len(tokens) if tokens else 0
        if score > best_score:
            best_score = score
            best_time = seg_start

    # Require at least 40% of the name tokens to match
    if best_score >= 0.4:
        return best_time

    return chunk_start


def find_best_timestamp_in_transcript(
    locale_name: str,
    transcript: dict,
    ground_truth_variants: list[str] | None = None,
) -> float | None:
    """Scan the FULL transcript for the best timestamp where a venue name appears.

    This is a post-extraction pass that searches ALL segments (not just
    one chunk) and uses ground truth ASR variants for better matching.

    Returns seconds, or None if no match found.
    """
    segments = transcript.get("segments", [])
    if not segments:
        return None

    search_terms = [locale_name.lower()]
    if ground_truth_variants:
        search_terms.extend(v.lower() for v in ground_truth_variants)

    # Also generate simple tokens from the name
    name_tokens = [t for t in locale_name.lower().split() if len(t) >= 3]

    best_time = None
    best_score = 0.0

    for seg in segments:
        seg_text = seg.get("text", "").lower()
        seg_start = seg.get("start", 0)

        for term in search_terms:
            if term in seg_text:
                return seg_start  # exact match — best possible

        # Token overlap with fuzzy matching
        if name_tokens:
            from thefuzz import fuzz
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


def extract_from_chunk(
    chunk: dict,
    channel_rubriche: list[str] | None = None,
    video_description: str = "",
    video_intel: "VideoIntel | None" = None,
    video_title: str = "",
) -> list[dict]:
    """
    Extract locale information from a single transcription chunk using LLM.
    
    Args:
        chunk: dict with video_id, start_timestamp, end_timestamp, text
        channel_rubriche: list of known rubrica names for better inference
        video_description: YouTube video description for additional context
        video_intel: pre-analyzed intelligence from title/description
        video_title: original video title
    
    Returns:
        list of extracted locale dicts
    """
    llm = get_llm()
    if llm is None:
        return []

    text = chunk.get("text", "").strip()
    if not text or len(text) < 20:
        return []

    rubriche_section = ""
    if channel_rubriche:
        rubriche_section = f"\nKnown channel shows: {', '.join(channel_rubriche)}\n"

    description_section = ""
    if video_description:
        desc_trunc = video_description[:1500]
        description_section = (
            f"\nVIDEO DESCRIPTION:\n\"{desc_trunc}\"\n"
        )

    # Build venue hints section from title/description intelligence
    venue_hints_section = ""
    video_type = "unknown"
    if video_intel:
        video_type = video_intel.video_type
        if video_intel.venue_hints:
            hints_list = []
            for h in video_intel.venue_hints:
                hint_str = f'"{h["name"]}"'
                if h.get("address"):
                    hint_str += f' ({h["address"]})'
                hint_str += f' [source: {h.get("source", "unknown")}, confidence: {h.get("confidence", "medium")}]'
                hints_list.append(hint_str)
            venue_hints_section = (
                "\nVENUE HINTS (from title/description — these are LIKELY correct, "
                "look for phonetic matches in the transcript):\n"
                + "\n".join(f"  • {h}" for h in hints_list)
                + "\n"
            )
        if video_intel.city and not venue_hints_section:
            venue_hints_section = f"\nCITY: {video_intel.city}\n"

    # Few-shot examples from ground truth
    from scripts.video_intelligence import build_few_shot_examples
    few_shot_section = build_few_shot_examples()

    prompt = EXTRACTION_PROMPT.format(
        title=video_title[:200],
        video_type=video_type,
        start=chunk.get("start_timestamp", "?"),
        end=chunk.get("end_timestamp", "?"),
        text=text[:3000],
        rubriche_section=rubriche_section,
        description_section=description_section,
        venue_hints_section=venue_hints_section,
        few_shot_section=few_shot_section,
    )

    try:
        response = llm(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stop=["```", "\n\n\n"],
        )

        output = response["choices"][0]["text"].strip()

        # Try to extract JSON from the response
        extracted = _parse_llm_json(output)
        if extracted is None:
            logger.debug(f"No valid JSON in LLM response for chunk {chunk.get('chunk_index', '?')}")
            return []

        # Validate and normalize each extraction
        results = []
        for item in extracted:
            if not isinstance(item, dict):
                continue
            if not item.get("locale_name"):
                continue

            locale_name = _clean_locale_name(str(item.get("locale_name", "")).strip())

            # Apply strict validation filter
            if not _is_valid_locale_name(locale_name):
                logger.debug(f"Rejected generic/invalid name: '{locale_name}'")
                continue

            seg_ts = chunk.get("segment_timestamps", [])
            chunk_start_s = chunk.get("start_time", 0)
            mention_time = _find_mention_timestamp(locale_name, seg_ts, chunk_start_s)

            from scripts.chunk_transcription import seconds_to_timestamp

            normalized = {
                "locale_name": locale_name,
                "address": str(item.get("address", "")).strip(),
                "city": str(item.get("city", "")).strip(),
                "category": item.get("category", []) if isinstance(item.get("category"), list) else [],
                "rating": _normalize_rating(item.get("rating")),
                "sentiment": _normalize_sentiment(item.get("sentiment", "neutral")),
                "notes": str(item.get("notes", "")).strip(),
                "rubrica": str(item.get("rubrica", "")).strip(),
                "confidence": _normalize_confidence(item.get("confidence", 0.5)),
                "chunk_start": chunk.get("start_timestamp", "0:00"),
                "chunk_end": chunk.get("end_timestamp", "0:00"),
                "chunk_start_seconds": chunk.get("start_time", 0),
                "mention_time": mention_time,
                "mention_timestamp": seconds_to_timestamp(mention_time),
            }
            results.append(normalized)

        return results

    except Exception as e:
        logger.error(f"LLM extraction failed for chunk {chunk.get('chunk_index', '?')}: {e}")
        return []


def _parse_llm_json(text: str) -> list | None:
    """Try to extract a JSON array from LLM output."""
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Try to find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass

    return None


def _normalize_rating(value: object) -> str | None:
    """Normalize rating to a string preserving modifiers like '8--', '6++', '10'.

    Accepts numeric values, strings like "8", "8--", "6++", "7.5",
    "da 8", "voto 7", etc. Returns None if no valid rating found.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "n/a", ""):
        return None

    import re as _re
    # Match patterns like "8", "8--", "6++", "7.5", "8+", "9-", "10"
    m = _re.search(r'(\d+(?:\.\d+)?)\s*([+\-]{1,2})?', s)
    if m:
        num = float(m.group(1))
        modifier = m.group(2) or ""
        if 1 <= num <= 10:
            base = str(int(num)) if num == int(num) else str(num)
            return f"{base}{modifier}" if modifier else base
    return None


def _normalize_sentiment(value) -> str:
    """Normalize sentiment to one of three values."""
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ("positive", "positivo"):
            return "positive"
        if v in ("negative", "negativo"):
            return "negative"
    return "neutral"


def _normalize_confidence(value) -> float:
    """Normalize confidence to 0-1 float."""
    try:
        conf = float(value)
        return max(0.0, min(1.0, round(conf, 2)))
    except (ValueError, TypeError):
        return 0.5


# ---------------------------------------------------------------------------
# Self-verification ("Generate then Verify" pattern)
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """<|start_header_id|>system<|end_header_id|>

You verify extracted food venue data from Italian YouTube transcripts. Return ONLY valid JSON arrays.<|eot_id|><|start_header_id|>user<|end_header_id|>

TRANSCRIPT ({start} to {end}):
"{text}"

EXTRACTED VENUES TO VERIFY:
{extractions_json}

RULES:
- KEEP if the blogger is at the venue: eating, tasting, commenting on food, discussing prices, ordering.
- KEEP if the name sounds phonetically like something in the transcript (ASR garbles names heavily).
- EXCLUDE ONLY if: name has zero phonetic trace in transcript, OR it's ONLY mentioned as future destination, OR it's a person/landmark/neighborhood.
- Be CONSERVATIVE with rejections — only reject if HIGHLY CONFIDENT the venue is wrong.
- Add "notes" with food items mentioned for this venue.
- RATING: only overall venue rating as stated (e.g., "8", "8--", "6++"). null if none given.
- SENTIMENT: praise="positive", complaints="negative", mixed="neutral".

Return JSON array (confidence 0.75–0.95):
[{{"locale_name": "Name", "city": "Roma", "rating": null, "sentiment": "positive", "notes": "items eaten", "confidence": 0.85}}]

If none confirmed: []<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def _verify_extractions(
    chunk: dict, extractions: list[dict], protected_names: set[str] | None = None,
) -> list[dict]:
    """Run a verification pass on extracted locales using the same LLM.

    Protected names (from title/description intelligence) are NEVER rejected
    by verification — they are always kept regardless of LLM output.

    Fallback behaviour for non-protected venues:
    - Valid JSON with confirmed venues → use those.
    - Valid JSON with empty list → rejection (but protected names still kept).
    - Unparseable output → keep originals with confidence penalty.
    - Exception → same fallback with penalty.
    """
    if protected_names is None:
        protected_names = set()
    llm = get_llm()
    if llm is None or not extractions:
        return extractions

    simple = [
        {
            "locale_name": e["locale_name"],
            "city": e.get("city", ""),
            "rating": e.get("rating"),
            "sentiment": e.get("sentiment", "neutral"),
        }
        for e in extractions
    ]

    prompt = VERIFY_PROMPT.format(
        start=chunk.get("start_timestamp", "?"),
        end=chunk.get("end_timestamp", "?"),
        text=chunk.get("text", "")[:2000],
        extractions_json=json.dumps(simple, ensure_ascii=False),
    )

    CONFIDENCE_PENALTY = 0.10

    def _is_protected(name: str) -> bool:
        """Check if a venue name is protected by title/description intelligence."""
        name_lower = name.lower().strip()
        from thefuzz import fuzz
        for pn in protected_names:
            if fuzz.ratio(name_lower, pn) >= 70 or pn in name_lower or name_lower in pn:
                return True
        return False

    def _penalised_fallback(reason: str) -> list[dict]:
        """Return original extractions with reduced confidence."""
        logger.debug(f"Verification fallback ({reason}), keeping with penalty")
        for e in extractions:
            e["confidence"] = max(0.0, e["confidence"] - CONFIDENCE_PENALTY)
            e["verified"] = False
        return extractions

    try:
        response = llm(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.05,
            stop=["```", "\n\n\n"],
        )
        output = response["choices"][0]["text"].strip()
        verified = _parse_llm_json(output)

        if verified is None:
            return _penalised_fallback("no valid JSON in LLM response")

        verified_names = {
            v.get("locale_name", "").lower().strip()
            for v in verified
            if isinstance(v, dict)
        } if verified else set()

        confirmed = []
        for ext in extractions:
            name_lower = ext["locale_name"].lower().strip()
            if name_lower in verified_names:
                for v in verified:
                    if (
                        isinstance(v, dict)
                        and v.get("locale_name", "").lower().strip() == name_lower
                    ):
                        new_conf = _normalize_confidence(
                            v.get("confidence", ext["confidence"])
                        )
                        ext["confidence"] = max(ext["confidence"], new_conf)
                        ext["verified"] = True
                        if v.get("rating") is not None:
                            ext["rating"] = _normalize_rating(v["rating"])
                        if v.get("sentiment"):
                            ext["sentiment"] = _normalize_sentiment(v["sentiment"])
                        break
                confirmed.append(ext)
            elif _is_protected(ext["locale_name"]):
                logger.info(f"Verification rejected '{ext['locale_name']}' but PROTECTED by title/description — keeping")
                ext["verified"] = False
                confirmed.append(ext)
            else:
                logger.debug(f"Verification rejected: '{ext['locale_name']}'")

        if not confirmed:
            logger.debug("Verification rejected all extractions")

        return confirmed

    except Exception as e:
        logger.warning(f"Verification pass failed: {e}")
        return _penalised_fallback(str(e))


def extract_from_video(
    video_id: str,
    chunks: list[dict],
    channel_rubriche: list[str] = None,
    video_description: str = "",
    video_title: str = "",
    video_intel: "VideoIntel | None" = None,
) -> tuple[list[dict], list[dict]]:
    """
    Extract locale info from all chunks of a video.

    Uses cross-chunk consensus and title/description intelligence.
    
    Args:
        video_id: YouTube video ID
        chunks: list of transcription chunk dicts
        channel_rubriche: list of known rubrica names
        video_description: YouTube video description for additional context
        video_title: original video title
        video_intel: pre-analyzed title/description intelligence
    
    Returns:
        (extractions, flagged) — high confidence and low confidence results
    """
    # Build set of "protected" venue names from title/description intelligence
    protected_names: set[str] = set()
    if video_intel:
        for hint in video_intel.venue_hints:
            protected_names.add(hint["name"].lower().strip())

    # Step 1: collect per-chunk extractions
    all_chunk_results: list[list[dict]] = []

    for chunk in chunks:
        results = extract_from_chunk(
            chunk, channel_rubriche, video_description,
            video_intel=video_intel, video_title=video_title,
        )

        # Self-verification pass — but protect title-derived venues
        if results and LLM_VERIFY:
            results = _verify_extractions(chunk, results, protected_names)

        all_chunk_results.append(results)

    # Step 2: cross-chunk frequency as confidence boost (soft signal, not hard filter)
    flat_results = [r for chunk_results in all_chunk_results for r in chunk_results]

    if len(chunks) > 1 and flat_results:
        # Count how many distinct chunks mention each locale name (case-insensitive)
        from collections import Counter
        chunk_counts: Counter = Counter()
        for chunk_idx, chunk_results in enumerate(all_chunk_results):
            seen_in_chunk: set[str] = set()
            for r in chunk_results:
                name_key = r["locale_name"].lower().strip()
                if name_key not in seen_in_chunk:
                    chunk_counts[name_key] += 1
                    seen_in_chunk.add(name_key)

        # Boost confidence for locales mentioned in multiple chunks
        multi_mention = {name for name, count in chunk_counts.items() if count >= 2}
        if multi_mention:
            logger.info(
                f"Video {video_id}: multi-chunk locales (boosted): {multi_mention}"
            )
        for r in flat_results:
            name_key = r["locale_name"].lower().strip()
            if name_key in multi_mention:
                r["confidence"] = min(1.0, r["confidence"] + 0.05)

    flagged_singles = []

    # Step 3: deduplicate same locale across chunks — keep highest confidence,
    # merge notes from all chunks
    best_by_name: dict[str, dict] = {}
    notes_by_name: dict[str, list[str]] = {}
    for r in flat_results:
        name_key = r["locale_name"].lower().strip()
        if r.get("notes"):
            notes_by_name.setdefault(name_key, []).append(r["notes"])
        if name_key not in best_by_name or r["confidence"] > best_by_name[name_key]["confidence"]:
            best_by_name[name_key] = r

    # Merge all notes for each venue
    for name_key, best in best_by_name.items():
        all_notes = notes_by_name.get(name_key, [])
        if all_notes:
            seen = set()
            unique_notes = []
            for n in all_notes:
                if n.lower() not in seen:
                    seen.add(n.lower())
                    unique_notes.append(n)
            best["notes"] = "; ".join(unique_notes)

    # Step 4: inject city from VideoIntel when LLM didn't provide one
    if video_intel and video_intel.city:
        for r in best_by_name.values():
            if not r.get("city"):
                r["city"] = video_intel.city

    # Step 5: inject rubrica from VideoIntel series name
    if video_intel and video_intel.series_name:
        for r in best_by_name.values():
            if not r.get("rubrica"):
                r["rubrica"] = video_intel.series_name

    # Step 6: split by confidence threshold
    extractions = []
    flagged = []

    for r in best_by_name.values():
        if r["confidence"] >= CONFIDENCE_THRESHOLD:
            extractions.append(r)
        else:
            flagged.append(r)

    # Add flagged singles
    flagged.extend(flagged_singles)

    logger.info(
        f"Video {video_id}: {len(extractions)} extractions, {len(flagged)} flagged"
    )
    return extractions, flagged


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
    chunks = chunk_transcription(transcript)

    extractions, flagged = extract_from_video(vid, chunks)
    print(f"\nExtractions ({len(extractions)}):")
    for e in extractions:
        print(f"  {e['locale_name']} ({e['city']}) - rating: {e['rating']} conf: {e['confidence']}")

    print(f"\nFlagged ({len(flagged)}):")
    for f_ in flagged:
        print(f"  {f_['locale_name']} ({f_['city']}) - conf: {f_['confidence']}")
