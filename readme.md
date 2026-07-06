# CiboBuono — YouTube Food Locale Reviews Dataset

**[🇮🇹 Leggi in italiano](#-versione-italiana)**

## What is this?

An open-source dataset and interactive map of food locales (bakeries, pizzerias, restaurants, etc.) reviewed by Italian YouTube food bloggers. All data is extracted automatically via a **fully local pipeline** using open-source tools only — no paid APIs.

The pipeline uses **inference only** (no model training or fine-tuning): a local ASR model, a **GLiNER** NER model for venue spans, deterministic rules, and a local instruction-tuned LLM (GGUF) for verification and visit details. **Python 3.10+** is required.

**Language:** all catalogued videos are **Italian**; yt-dlp, subtitle download, Whisper, Nominatim result language, and LLM prompts use Italian (`CONTENT_LANGUAGE` in `scripts/utils.py`).

The pipeline transcribes YouTube videos, proposes venue **candidates** with a local multilingual NER (GLiNER), applies **deterministic Italian visit-vs-mention rules**, and calls the local LLM only as a binary **verifier** (with cited evidence) when rules are ambiguous; ratings and sentiment are filled by the LLM only on accepted visits over a short transcript window. Results are geocoded via OpenStreetMap and written as normalized JSON on GitHub. The React site deploys automatically via GitHub Pages on every push.

## Live Map

The interactive map is deployed on **GitHub Pages** and updates automatically on every `git push` to `main`. It allows you to explore all reviewed locales with search and sentiment filters.

The web UI is **bilingual (English / Italiano)**: the language is auto-detected from your browser, and a toggle in the header switches between EN and IT on the fly.

> **Local preview**: `cd site && npm install && npm run dev`

> **Python venv (pipeline)**: from repo root, `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then run `python -m scripts.run_pipeline ...`. Optional: `CIBOBUONO_LLM_MODEL` → a `.gguf` under `models/` or elsewhere; `CIBOBUONO_NER_MODEL` overrides the Hugging Face id for GLiNER (default `knowledgator/gliner-x-large-v0.5`). **PyTorch** is required for NER (CPU or MPS on Apple Silicon); if GLiNER cannot load, a lightweight heuristic fallback still runs.

> **ffmpeg**: required on your `PATH` for yt-dlp audio extraction and for Whisper (`brew install ffmpeg` on macOS).

## Repository Structure

```
/cibobuono
├── channels_input.txt              # YouTube channel URLs (one per line)
├── requirements.txt                # Python dependencies
├── readme.md                       # This file
│
├── .github/workflows/
│   └── deploy.yml                  # GitHub Actions: build React + deploy Pages
│
├── data/                           # Normalized JSON dataset
│   ├── channels.json               # YouTube channels
│   ├── videos.json                 # All cataloged videos with status
│   ├── locales.json                # Normalized locales with coordinates
│   ├── visits.json                 # Locale visits/reviews from videos
│   ├── processed_videos.json       # Incrementality tracker
│   ├── flagged_segments.json       # Low-confidence segments for review
│   ├── skipped_videos.json         # Skipped videos (recipes, Shorts)
│   ├── corrections.json            # Manual hide/edit overrides (preserved by --reset)
│   ├── voices.json                 # Per-channel recurring-voice registry (Perceptor)
│   ├── perception.json             # Per-video audio/video perception records (Perceptor)
│   ├── calibration.json            # Fitted Platt-scaling params (regenerated; absent until fitted)
│   └── eval_set.json               # Hand-labeled gold set for the visit classifier eval harness
│
├── scripts/                        # Pipeline modules (Python)
│   ├── schemas.py                  # Pydantic models & validation
│   ├── utils.py                    # Shared utilities, paths, config, hardware shim
│   ├── hardware.py                 # Cross-platform DeviceProfile (Whisper + llama.cpp params)
│   ├── resource_monitor.py         # Live RAM/GPU/CPU monitoring with back-pressure
│   ├── fetch_channels.py           # Extract channel metadata via yt-dlp
│   ├── fetch_videos.py             # Catalog videos, detect recipes/non-food/Shorts, download audio
│   ├── transcribe_video.py         # Transcription: Whisper large-v3-turbo (primary) + YouTube manual subs
│   ├── chunk_transcription.py      # Split transcripts into 90s chunks with 15s overlap
│   ├── video_intelligence.py       # Title + description + chapters + comments analysis (hints, rating, rubrica)
│   ├── extract_locales.py          # Shared helpers: food gate, hints, timestamps, LLM handle
│   ├── ner_candidates.py           # GLiNER venue candidates (+ heuristic fallback; parallel workers)
│   ├── visit_classifier.py         # Italian rules + LLM yes/no for ambiguous cases
│   ├── batch_visit_llm.py          # Batch LLM evaluation for NER candidates (GPU-optimised)
│   ├── venue_discovery.py          # Holistic LLM venue discovery from full timestamped transcript
│   ├── vote_aggregator.py          # Weighted log-odds fusion of Perceptor signals into visit confidence
│   ├── calibrate_confidence.py     # Platt-scaling recalibration of confidence from corrections.json
│   ├── eval_pipeline.py            # Precision/recall/F1 of the visit classifier against data/eval_set.json
│   ├── extract_pipeline.py         # Orchestrator: NER → batch classify → detail LLM → cross-chunk filter → holistic merge
│   ├── geocode_locales.py          # Nominatim geocoding (free, rate-limited, file-cached)
│   ├── verify_locales.py           # OSM verification via Overpass API (anti-false-positive)
│   ├── deduplicate_locales.py      # Fuzzy name + haversine distance deduplication
│   ├── populate_json.py            # Write visits, flagged segments
│   ├── handle_flagged_segments.py  # Import manually reviewed segments → locale + visit
│   ├── manual_edits.py             # Dashboard-driven manual corrections (remove/add visits)
│   ├── review_queue.py             # Pending-review queue + user locale reports
│   ├── repair_stale_state.py       # Repair pending-but-already-extracted videos
│   ├── pipeline_executor.py        # PipelineExecutor: background finalize (GPU/CPU overlap)
│   ├── pipeline_metrics.py         # Per-run metrics aggregation → logs/pipeline_metrics.json
│   ├── pipeline_control.py         # Runtime pause/resume/stop/status CLI (file-based)
│   ├── dashboard.py                # Rich live terminal dashboard + JSON snapshot
│   ├── perceptor.py                # Audio/video perception orchestrator (--perceptor)
│   ├── perceptor_audio.py          # Silero VAD + speaker diarization + channel voice registry
│   ├── perceptor_video.py          # Frame sampling + phash novelty + Qwen2-VL captioning
│   ├── setup_models.py             # Model download + verification helper
│   ├── push_to_github.py           # Git commit & push
│   ├── validate_data.py            # Validate data/*.json against Pydantic schemas (CI / local)
│   ├── run_pipeline.py             # Main pipeline orchestrator (2-phase; --watch supported)
│   └── com.cibobuono.pipeline.plist.example  # Example macOS LaunchAgent for --watch mode
│
├── site/                           # React + Vite + TypeScript (GitHub Pages, bilingual EN/IT)
│   ├── package.json                # Node dependencies
│   ├── vite.config.ts              # Vite config (base: /cibobuono/)
│   ├── index.html                  # HTML entry — title + meta description
│   ├── src/
│   │   ├── App.tsx                 # Main app: data loading, filtering, layout
│   │   ├── api.ts                  # Fetch JSON from data/ at runtime
│   │   ├── types.ts                # TypeScript types (mirrors Python schemas)
│   │   ├── i18n/
│   │   │   ├── messages.ts         # Typed EN + IT message dictionaries
│   │   │   ├── LanguageContext.tsx # Provider (persists language in localStorage)
│   │   │   └── useLanguage.ts      # useLanguage / useT hooks
│   │   └── components/
│   │       ├── MapView.tsx         # Leaflet map with sentiment-colored markers
│   │       ├── LocaleList.tsx      # Sidebar list with expandable visit details
│   │       ├── Header.tsx          # Search + filters + EN/IT language toggle
│   │       └── StatusBar.tsx       # Loading/error states
│   └── ...
│
├── tests/                          # Pytest test suite (464 tests)
│   ├── test_schemas.py             # Pydantic model validation
│   ├── test_dedup.py               # Fuzzy dedup + haversine
│   ├── test_chunks.py              # Chunking + timestamp helpers
│   ├── test_extraction.py          # Food gate, LLM caching, locale validation, description hints
│   ├── test_ner_candidates.py      # GLiNER candidates + parallel exception recovery
│   ├── test_visit_classifier.py    # Italian rule classifier + venue name checks + self-consistency ensemble
│   ├── test_batch_visit_llm.py     # Batch LLM evaluation
│   ├── test_venue_discovery.py     # Holistic LLM discovery + transcript formatting
│   ├── test_extract_pipeline.py    # Full NER→classify→merge pipeline + Perceptor fusion + agreement bonus
│   ├── test_vote_aggregator.py     # Perceptor OCR/speaker votes + log-odds confidence combination
│   ├── test_calibrate_confidence.py # Platt-scaling fit/apply + save/load roundtrip
│   ├── test_eval_pipeline.py       # Visit-classifier precision/recall/F1 harness
│   ├── test_verify.py              # OSM/Overpass verification
│   ├── test_geocode.py             # Geocoding cache logic + batch geocode
│   ├── test_perceptor.py           # Novelty dedup, diarization, voice registry, stage guards
│   ├── test_validate_data.py       # validate_data.py gate (all files OK / errors reported)
│   ├── test_utils.py               # Shared utilities
│   ├── test_data_integrity.py      # JSON referential integrity
│   ├── test_hardware.py            # DeviceProfile across simulated platforms
│   ├── test_populate_json.py       # populate_visits + populate_flagged
│   ├── test_manual_edits.py        # Dashboard manual visit corrections
│   ├── test_repair_stale_state.py  # pending → processed repair
│   ├── test_resource_monitor.py    # RAM/GPU/CPU monitoring
│   ├── test_pipeline_control.py    # Pause/stop/status control
│   ├── test_pipeline_executor.py   # City coherence, finalize, GPU/CPU overlap
│   ├── test_pipeline_metrics.py    # Run metrics + eval metrics aggregation + append
│   ├── test_review_queue.py        # Pending reviews + reports
│   ├── test_dashboard.py           # Dashboard state + snapshot
│   ├── test_transcribe.py          # VTT parser + Whisper backend selection
│   ├── test_video_intelligence.py  # Title / description / chapter / comment parsing
│   └── test_watch_loop.py          # --watch daemon mode
│
├── models/                         # Local LLM GGUF models (gitignored)
├── cache/                          # Temp audio & transcripts (gitignored)
└── logs/                           # Pipeline logs (gitignored)
```

## Data Model

All JSON files are **normalized** — no nested arrays of IDs inside records. Visits reference locales, videos, and channels by ID. The site computes relationships at load time.

### channels.json

| Field         | Type     | Description                           |
|---------------|----------|---------------------------------------|
| `channel_id`  | string   | Deterministic ID from channel URL     |
| `name`        | string   | Channel display name                  |
| `url`         | string   | YouTube channel URL                   |
| `description` | string   | Channel description                   |
| `rubriche`    | string[] | Show/series names inferred from titles|

### videos.json

| Field            | Type   | Description                              |
|------------------|--------|------------------------------------------|
| `video_id`       | string | YouTube video ID                         |
| `channel_id`     | string | FK → channels.json                       |
| `title`          | string | Video title                              |
| `url`            | string | YouTube URL                              |
| `publish_date`   | string | YYYY-MM-DD                               |
| `processed_date` | string | YYYY-MM-DD (empty if pending)            |
| `status`         | enum   | `pending` / `processed` / `errored`      |

### locales.json

| Field       | Type     | Description                                |
|-------------|----------|--------------------------------------------|
| `locale_id` | string   | SHA256 hash of normalized name + coords    |
| `name`      | string   | Primary locale name                        |
| `aliases`   | string[] | Alternative names (deduplication)          |
| `address`   | string   | Street address                             |
| `city`      | string   | City name                                  |
| `lat`       | float    | Latitude (4 decimal places)                |
| `lon`       | float    | Longitude (4 decimal places)               |
| `category`  | string[] | Business categories (forno, ristorante...) |

### visits.json

| Field             | Type   | Description                                    |
|-------------------|--------|------------------------------------------------|
| `visit_id`        | string | `visit_{video_id}_{timestamp_seconds}`          |
| `locale_id`       | string | FK → locales.json                               |
| `video_id`        | string | FK → videos.json                                |
| `channel_id`      | string | FK → channels.json                              |
| `timestamp_start` | string | MM:SS or HH:MM:SS                               |
| `timestamp_end`   | string | MM:SS or HH:MM:SS                               |
| `youtube_url`     | string | Direct link with `?t=` parameter                |
| `rating`          | string? | Blogger-stated overall score (e.g. `8`, `8--`, `6++`), null if not stated |
| `sentiment`       | enum   | `positive` / `neutral` / `negative`              |
| `rubrica`         | string | Show/series name                                 |
| `llm_confidence`  | float  | 0-1 extraction confidence                        |
| `extraction_date` | string | YYYY-MM-DD                                       |
| `date`            | string | Approx. visit date (= video publish date)        |

### processed_videos.json

| Field              | Type   | Description                     |
|--------------------|--------|---------------------------------|
| `video_id`         | string | YouTube video ID                |
| `channel_id`       | string | FK → channels.json              |
| `processed_date`   | string | YYYY-MM-DD                      |
| `status`           | enum   | Processing result               |
| `visits_extracted`  | int    | Number of visits found          |
| `flagged_segments`  | int    | Number of flagged segments      |

### flagged_segments.json

| Field              | Type    | Description                           |
|--------------------|---------|---------------------------------------|
| `video_id`         | string  | FK → videos.json                      |
| `channel_id`       | string  | FK → channels.json                    |
| `timestamp_start`  | string  | Segment start                         |
| `timestamp_end`    | string  | Segment end                           |
| `youtube_url`      | string  | Direct link                           |
| `reason`           | enum    | Why it was flagged                     |
| `extracted_text`   | string  | Raw transcription text                |
| `llm_confidence`   | float   | Confidence score                      |
| `reviewed_by_human`| bool    | Has been manually reviewed            |
| `reviewed_date`    | string? | Date of review                        |
| `locale_name`      | string? | Fill in during review                 |
| `rating`           | string? | Fill in during review (e.g. `8`, `8--`) |
| `city`             | string? | Fill in during review                 |

### skipped_videos.json

| Field          | Type   | Description                                |
|----------------|--------|--------------------------------------------|
| `video_id`     | string | YouTube video ID                           |
| `channel_id`   | string | FK → channels.json                         |
| `title`        | string | Video title                                |
| `url`          | string | YouTube URL                                |
| `reason`       | string | Reason for skipping (recipe keyword, Short)|
| `skipped_date` | string | YYYY-MM-DD                                 |

### corrections.json

| Field       | Type    | Description                                          |
|-------------|---------|-------------------------------------------------------|
| `locale_id` | string  | FK → locales.json                                     |
| `type`      | enum    | `hide` (false positive) / `edit` (field override)     |
| `reason`    | string? | Human-readable reason                                 |
| `overrides` | object? | For `type=edit`: `name`/`city`/`rating`/`sentiment` overrides |

Preserved across `--reset` (cleared only with `--reset-all-data`). `hide` entries are also the weak-supervision ground truth `scripts/calibrate_confidence.py` fits against — see [Confidence & Redundancy](#confidence--redundancy).

### voices.json

Per-channel registry of recurring speaker voices, written by `scripts/perceptor_audio.py`. One entry per detected voice:

| Field         | Type     | Description                                          |
|---------------|----------|-------------------------------------------------------|
| `voice_id`    | string   | `voice_{channel_id}_{NNN}`                             |
| `channel_id`  | string   | FK → channels.json                                     |
| `centroid`    | float[]  | Running-mean 192-dim TitaNet embedding                 |
| `n_samples`   | int      | Number of videos merged into the centroid              |
| `videos`      | string[] | Video IDs this voice was seen in                       |
| `created_at`  | string   | ISO-8601 timestamp                                     |
| `updated_at`  | string   | ISO-8601 timestamp                                     |

### perception.json

One record per video processed with `--perceptor`, written by `scripts/perceptor.py`:

| Field        | Type    | Description                                             |
|--------------|---------|-----------------------------------------------------------|
| `video_id`   | string  | FK → videos.json                                          |
| `channel_id` | string  | FK → channels.json                                         |
| `status`     | enum    | `ok` / `partial` / `errored`                               |
| `error`      | string? | Truncated error summary if not fully `ok`                  |
| `asr_backend`| string  | Backend used for this video (`mlx_whisper`/`faster_whisper`)|
| `audio`      | object? | VAD segments, per-speaker talk time + matched `voice_id`, per-transcript-segment speaker labels |
| `video`      | object? | Frames sampled/captioned, novelty-deduped VLM captions with timestamps |

`extract_pipeline.py` reads this record per video (`get_perception`) and feeds its captions/diarization into `scripts/vote_aggregator.py` as independent votes on the visit/mention decision — see [Confidence & Redundancy](#confidence--redundancy).

### calibration.json

Written by `scripts/calibrate_confidence.py`; absent until the script has been run once.

| Field       | Type   | Description                                              |
|-------------|--------|-------------------------------------------------------------|
| `fitted`    | bool   | Whether enough labeled data existed to fit a calibration     |
| `a`, `b`    | float  | Platt-scaling params (present only if `fitted=true`)         |
| `n_samples` | int    | Number of (confidence, outcome) pairs the fit used            |

### eval_set.json

Hand-labeled gold set consumed by `scripts/eval_pipeline.py`. Plain JSON list, starts empty until populated by hand:

| Field            | Type    | Description                                    |
|------------------|---------|--------------------------------------------------|
| `candidate_name` | string  | Venue name as it appears in the transcript window |
| `window_text`    | string  | Transcript excerpt around the mention              |
| `start_time`     | float?  | Mention time in seconds (default 0.0)              |
| `ner_score`      | float?  | Simulated NER span confidence (default 0.6)        |
| `gold_label`     | enum    | `visit` / `mention` — the correct answer            |

## Entity Relationships

```
channels.json ←── videos.json ←── visits.json ──→ locales.json
                       │
                skipped_videos.json
                       │
                   flagged_segments.json (manual review → updates above)
```

All IDs are **deterministic** to guarantee idempotency.

## Pipeline

Run locally (on-demand or scheduled):

```bash
python -m scripts.run_pipeline --skip-push --max-videos 10
```

### Architecture

The pipeline runs in two phases:

**Phase 1 — Catalog**: Fetch all channel videos via yt-dlp. Insert into `videos.json` as `pending`. Recipe videos and YouTube Shorts are detected and moved to `skipped_videos.json`.

**Phase 2 — Process**: For each pending video (newest first, up to `--max-videos`; use `0` for all pending):

| #  | Step                 | Tool                         | Description                                      |
|----|----------------------|------------------------------|--------------------------------------------------|
| 1  | Fetch channels       | yt-dlp                       | Extract metadata from channel URLs               |
| 2  | Catalog videos       | yt-dlp                       | Catalog all videos, skip recipes & Shorts        |
| 3  | Prefetch audio       | yt-dlp                       | Sliding window: pre-download up to 20 audio files|
| 4  | Transcribe           | Whisper `large-v3-turbo` (faster-whisper) + YouTube manual subs | Whisper primary; YouTube *manual* subs preferred when present (auto-subs ignored — they mangle Italian proper nouns) |
| 5  | Chunk                | Python                       | Split into 90s chunks with 15s overlap           |
| 6  | Extract              | GLiNER + rules + local LLM   | NER candidates → visit/mention → detail JSON (ratings) only if visit |
| 7  | Cross-chunk filter   | Python                       | Keep venue if ≥2 chunks **or** title/description hint protects it |
| 8  | Geocode              | Nominatim (OSM, free)        | Name + city → lat/lon                            |
| 9  | Verify (OSM)         | Overpass API (OSM, free)     | Confirm locale exists as real food business      |
| 10 | Deduplicate          | thefuzz + haversine          | <200m AND name similarity ≥70%                   |
| 11 | Populate             | Python                       | Write visits.json, update locales.json           |
| 12 | Flag                 | Python                       | Low confidence → flagged_segments.json           |
| 13 | Push                 | git                          | Commit & push updated data                       |
| 14 | Deploy               | GitHub Actions                | Build React site, deploy to Pages                |

### Extraction — Neuro-Symbolic Detail

The extraction step (step 6 above) combines three layers working together:

```
Full timestamped transcript
     │
     ├─→ [A] Holistic LLM discovery  ──────────────────────────────────→ venue list + timestamps
     │         (venue_discovery.py)                                               │
     │                                                                            ↓
     └─→ [B] 90 s chunks (15 s overlap)                          [C] Merge + cross-chunk filter
               │                                                                  │
               ├─→ GLiNER NER (parallel, 4 workers)                              ↓
               │   knowledgator/gliner-x-large-v0.5            Final extractions with
               │   19 zero-shot labels:                         rating + sentiment + timestamp
               │   Venue: restaurant · ristorante · pizzeria
               │          trattoria · forno · panificio
               │          pasticceria · gelateria · osteria
               │          bakery · street food stall
               │          food market · bar or cafe
               │   Context: city · neighborhood · country
               │            person · brand · food dish
               │
               └─→ Visit classifier ──→ confirmed visits ──→ Batch LLM detail
                   (visit_classifier.py)                     (batch_visit_llm.py)
                   Italian rules first:                      rating: "8" / "8--" / "6++"
                   • explicit visit verbs                    sentiment: positive/neutral/negative
                   • movement prepositions
                   LLM binary yes/no only if ambiguous
                   (must cite a verbatim evidence span)
```

**[A] Holistic LLM discovery** (`venue_discovery.py`): The full timestamped transcript is sent to the LLM in a single structured-JSON prompt. The LLM returns a list of visited venues with approximate start timestamps and any stated rating or sentiment. This large-context pass catches venues that chunk-level NER misses (e.g. a place briefly mentioned at the opening and revisited later).

**[B] GLiNER NER** (`ner_candidates.py`): Each 90-second chunk is scored independently by GLiNER x-large, a zero-shot generative NER model (mT5 backbone, multilingual), running in parallel across 4 CPU workers. On CUDA systems GLiNER is pinned to CPU by default to leave all VRAM for Whisper and the LLM. Set `CIBOBUONO_GLINER_CPU=0` to enable GPU inference if your free VRAM allows it (GLiNER x-large is ~1.5 GB on GPU; on machines with ≥16 GB VRAM and a 14B LLM there is ample headroom, making NER 10–20× faster).

**Visit classifier** (`visit_classifier.py`): Deterministic Italian regex rules decide the majority of cases (explicit visit verbs, movement prepositions, presence in the video location). The LLM is called only for genuinely ambiguous candidates and must return a strict binary yes/no with a verbatim quoted evidence span.

**Batch LLM evaluation** (`batch_visit_llm.py`): All NER candidates confirmed as visits are sent to the LLM in a single batched call that extracts `rating` and `sentiment` for every visit at once. On CPU-only systems this falls back to sequential calls.

**[C] Merge + confidence** now goes beyond "keep the higher-confidence row": cross-source agreement, self-consistency voting on ambiguous cases, and Perceptor's audio/video signals all feed into the final confidence — see [Confidence & Redundancy](#confidence--redundancy) below.

**Chunk parameters**: 90 s with 15-second overlap was chosen to balance two needs: a typical restaurant visit in these videos spans 60–180 s of commentary (so 90 s is a good fit), and the overlap prevents a mention straddling a boundary from being missed.

**Rating format**: The `rating` field stores the blogger's exact expressed score as a string:

| Format | Meaning |
|--------|---------|
| `"8"` | Exact integer score (most common) |
| `"8--"` | Slightly below 8 (`"quasi un 8 meno meno"`) |
| `"8-"` | A little below 8 |
| `"7+"` | A little above 7 |
| `"7++"` | Approaching 8 |
| `null` | No numeric score stated; sentiment is still extracted |

The numeric core must be 1–10; the Pydantic validator rejects values outside this range.

**Whisper `initial_prompt`**: Every transcription is primed with a short Italian preamble listing venue types (`pizzeria`, `trattoria`, `forno`, …), well-known restaurant names, Italian cities, and visit phrases (`andiamo a mangiare`, `entriamo`, …). This significantly reduces ASR errors on restaurant proper nouns compared to unprompted decoding.

**Model choices rationale**:

| Component | Model | Why |
|-----------|-------|-----|
| ASR | `whisper-large-v3-turbo` (faster-whisper, CUDA fp16) | Best speed/accuracy ratio for Italian; full `large-v3` is ~2× slower for marginal gain |
| NER | `knowledgator/gliner-x-large-v0.5` | Largest available zero-shot NER; mT5 backbone handles Italian venue names well with no fine-tuning |
| LLM | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` | Fully CUDA-offloadable at 8.4 GB; strong Italian instruction following and structured-JSON output; tier-27B default (`gemma-3-27b`) requires HF auth |
| Geocoding | Nominatim + Overpass (OSM) | No paid API; Nominatim covers Italian cities well; Overpass `amenity=*` verification is the strongest anti-false-positive gate |

### Confidence & Redundancy

Every classification step above (rules, LLM arbiter, batch LLM, holistic discovery) used to run **once** per candidate, and results were merged by "keep whichever confidence is higher" — agreement between independent sources was thrown away, and Perceptor's audio/video signals (captions, diarization) were computed but never consulted. Four mechanisms close that gap, all evaluated against the Condorcet-jury intuition that *independent* agreeing signals are more informative than re-asking a single correlated one:

1. **Cross-source agreement bonus** (`extract_pipeline._merge_extraction_rows`): when the chunk-level NER+rules pipeline and the holistic LLM discovery pass (`venue_discovery.py`) independently agree on the same venue, the merged row's confidence is boosted (`_SOURCE_AGREEMENT_BONUS`) instead of only keeping the higher of the two.

2. **Gated self-consistency for genuinely ambiguous cases** (`visit_classifier.classify_with_llm_ensemble`): when the rule engine's own reason is truly ambiguous (`conflict_patterns`, `no_clear_signal` — both a mention and a visit pattern fire, or neither), 3 diversified LLM samples vote instead of 1. (`empty_window` — no transcript text at all — is deliberately excluded: there's no evidence for 3 samples to disagree over, so it stays a single call.) (baseline greedy, a higher-temperature resample, and a "devil's advocate" reframing that argues the mention case first). Confidence scales with how much the samples agreed. This only fires for the ambiguous subset — most `unsure` cases (e.g. a visit pattern with weak food evidence) still resolve with a single call, since 3× the LLM cost is only worth it where the rules genuinely can't decide.

3. **Perceptor signal fusion** (`vote_aggregator.py`): the only *independent-modality* voters in the pipeline. A nearby VLM caption mentioning the venue name/signage text votes "visit"; the diarized speaker at the candidate's timestamp being the channel's dominant/registered voice (heuristic proxy for "the host") votes "visit", while a different speaker (guest/bystander) votes "mention" — catching a guest describing their own visit being misread as the host's. Votes combine with the text-pipeline confidence via a weighted log-odds sum (`combine_confidence`), not a naive average; Perceptor voters currently carry a lower weight than the text pipeline because there isn't yet enough corrected-outcome history to validate their reliability. No-op (falls back to the plain text confidence) when Perceptor is disabled or has nothing to say near that timestamp.

4. **Confidence recalibration from corrections** (`calibrate_confidence.py`): the hand-tuned linear blend above has never been checked against real outcomes. `python -m scripts.calibrate_confidence` fits a 2-parameter Platt scaling (`data/calibration.json`) from `data/corrections.json`'s `hide` entries (confirmed false positives) vs. `data/visits.json`'s `llm_confidence`. Deliberately *not* isotonic regression: that needs far more labeled points than a slowly-growing corrections file will have for a while, and a 2-parameter fit degrades gracefully on small data. Below `MIN_SAMPLES` (30) labeled pairs, calibration stays unfitted and the linear formula is used unchanged.

Measuring whether any of this actually helps: `python -m scripts.eval_pipeline [--with-llm]` runs the visit classifier against a hand-labeled gold set (`data/eval_set.json`, starts empty — see [Data Model](#eval_setjson)) and reports precision/recall/F1, trend-tracked in `logs/eval_metrics.json` via `pipeline_metrics.record_eval_metrics`. Run it before and after enabling a redundancy mechanism to attribute the gain (or confirm there isn't one) instead of judging by feel.

### Pipeline Options

```
python -m scripts.run_pipeline --help

Options:
  --skip-fetch         Skip cataloging new videos (process existing pending only)
  --skip-transcribe    Use cached transcripts
  --skip-extract       Skip LLM extraction
  --skip-push          Don't commit/push to GitHub
  --whisper-model      tiny|base|small|medium|large|large-v2|large-v3|large-v3-turbo
                       (default: auto-selected — large-v3-turbo on Apple Silicon / ≥8 GB)
  --max-videos N       Max pending videos per run (default: 100); 0 = all pending
  --no-dashboard       Disable live terminal dashboard (log-only mode)
  --reset              Reset pipeline data JSON, cache, and logs (keeps corrections unless --reset-all-data)
  --reset-all-data     With --reset, also clear corrections.json
  --repair-stale-state Mark pending videos that already have visits/flagged as processed, then exit
  --repair-dry-run     With --repair-stale-state, print only (no writes)
  --status             Show pipeline status summary and exit
  --watch              Continuous mode: keep cataloging + processing new videos in a loop
  --poll-interval N    With --watch, seconds between cycles (default: 1800, min: 60)
  --no-parallel-postprocess  Run geocode/OSM/populate synchronously (disable GPU/CPU overlap)
  --print-hardware     Print the detected hardware profile as JSON and exit
  --perceptor          Enable audio/video perception (VAD, diarization, voice
                       registry, frame captioning); also via CIBOBUONO_PERCEPTOR=1
```

### Perceptor (audio/video perception, `--perceptor`)

An optional per-video perception stage that enriches the pipeline with what
can be *heard* and *seen*, using the best backend the machine supports:

| Component | Technology | Backend selection |
|---|---|---|
| Voice activity | Silero VAD via sherpa-onnx | CPU, every tier |
| Speaker diarization | TitaNet-small embeddings + cosine clustering | CPU, every tier |
| Recurring voices | Per-channel registry (`data/voices.json`) | CPU, every tier |
| ASR | Whisper large-v3-turbo | mlx-whisper (Metal) on Apple Silicon; faster-whisper fp16 on CUDA; faster-whisper int8 on CPU |
| Frame novelty | PyAV sampling + perceptual hash (phash) | CPU; interval 2–5 s scaled to hardware |
| Captioning | Qwen2-VL | mlx-vlm 4-bit on Apple Silicon; transformers on CUDA (7B AWQ ≥16 GB VRAM, 2B fp16 ≥8 GB); disabled on CPU-only |

Results land in `data/perception.json` (per video: VAD segments, speaker
labels with talk time, matched channel voices, novelty-frame captions with
timestamps). The stage is best-effort: any perception failure is recorded and
the main pipeline continues. When enabled, these signals also feed back into
extraction confidence as independent votes — see [Confidence & Redundancy](#confidence--redundancy).
Setup:

```bash
python -m scripts.setup_models --perceptor-only   # ONNX models + VLM cache warm
python -m scripts.perceptor <video_id>            # one-shot on a cached video
CIBOBUONO_PERCEPTOR=1 python -m scripts.run_pipeline --max-videos 1
```

Pipeline control (pause/resume/stop a running pipeline):

```bash
python -m scripts.pipeline_control status|pause|resume|stop
```

### Continuous mode (`--watch`)

Run the pipeline as a long-lived daemon. Every poll interval it re-catalogs all channels, queues new videos as `pending` in `videos.json`, drains up to `--max-videos` (newest first), and pushes the resulting JSON delta to GitHub. If the process is killed and restarted later, any videos uploaded in the meantime are picked up on the next cycle.

```bash
# Local, no pushes (good for development):
python -m scripts.run_pipeline --watch --poll-interval 1800 --skip-push

# Production-ish: poll every 30 min, auto-push to GitHub on every change:
python -m scripts.run_pipeline --watch
```

Behaviour notes:
- Whisper, NER, and LLM models are loaded **once** on the first cycle and reused across cycles — RAM stays flat.
- The terminal dashboard is force-disabled in `--watch` (log-only) so the process plays nicely with `tmux`/`launchd`.
- `git push` only fires when `git status --porcelain data/` shows actual changes. Idle cycles (no new uploads on any followed channel) make zero commits.
- **Ctrl+C / SIGTERM** stops gracefully at the cycle boundary (or, if mid-video, after the current video like in one-shot mode). Press a second time to abort immediately. The next start resumes from where you left off thanks to the `pending` status in `videos.json` and atomic JSON writes.
- A transient exception in a cycle (e.g. yt-dlp network blip) is logged and the loop continues; `SystemExit` (e.g. missing GGUF model) terminates the daemon.

For autostart on macOS, see `scripts/com.cibobuono.pipeline.plist.example`.

**Interrupts (Ctrl+C / SIGTERM)** — First signal: finish the current video, then stop (coherent between videos). Second signal: quit immediately (run `python -m scripts.run_pipeline --repair-stale-state` if a video was left pending after visits were already saved). JSON files are saved via atomic replace to reduce corruption if the process is killed during a write.

**Full catalog re-run example** (after reset, process every pending video, no git push):

```bash
python -m scripts.run_pipeline --reset --skip-push --max-videos 0 --no-dashboard
```

### Key Features

- **Newest-first processing**: Videos are processed newest-first so the freshest uploads land on the map quickest.
- **Whisper-primary ASR**: Local Whisper `large-v3-turbo` is the primary transcription engine — via `mlx-whisper` (Metal GPU) on Apple Silicon, `faster-whisper` (CTranslate2) on CUDA/CPU. YouTube's *manual* subtitles are still preferred when an author has uploaded them (rare but high-quality); YouTube's *auto-generated* subtitles are explicitly **not** used because their Italian ASR mangles the proper nouns (venue/street names) we extract.
- **Whisper with `initial_prompt`**: Whisper runs with an Italian food terminology prompt to bias transcription toward restaurant names, and with `vad_filter=True` on faster-whisper to drop silence/jingles.
- **Continuous mode (`--watch`)**: Optional daemon loop that catalogs and processes new uploads forever, with model caching across cycles and graceful Ctrl+C.
- **Video descriptions as context**: Descriptions supply regex-extracted **venue hints** (names/links); hints can protect a candidate so a single-chunk mention still counts as a catalogued visit when rules agree it was a visit.
- **OSM real-place verification**: After geocoding, each locale is verified against OpenStreetMap via the Overpass API (500m radius, fuzzy name match ≥ 80) — if no matching food establishment exists near the coordinates, the extraction is rejected. This is the strongest anti-false-positive measure.
- **Sliding window**: Audio files are pre-downloaded in a window of 20. As each video is processed, the oldest is cleaned up and the next is fetched.
- **Neuro-symbolic extraction**: GLiNER proposes spans; Italian regex/heuristics decide many cases; the LLM answers visit yes/no with a quoted evidence span only when rules are unsure (no monolithic “extract everything” prompt).
- **Non-food video filtering**: Videos with non-food keywords in the title or description (boxing, gaming, fitness, etc.) are automatically skipped before transcription.
- **Holistic venue discovery**: A single structured LLM pass over the full timestamped transcript (`venue_discovery.py`) finds visits that chunk-level NER missed. Results are merged with NER output, keeping the highest-confidence entry per venue name.
- **Batch LLM evaluation**: On CUDA, NER candidates are evaluated in batches rather than one at a time, reducing per-video LLM call overhead (`batch_visit_llm.py`).
- **GPU/CPU overlap (`PipelineExecutor`)**: Geocoding, OSM verification, deduplication, and JSON population run in a background thread while the GPU processes the next video. Configurable with `--no-parallel-postprocess`.
- **Geographic coherence check**: After geocoding, each extraction's city is compared to the video's `video_intel.city` via fuzzy matching. Mismatches downgrade confidence rather than discard the extraction.
- **Pipeline metrics**: Each run appends a structured JSON record to `logs/pipeline_metrics.json` with geocode/OSM/publish rates, city-mismatch rate, and confidence distribution.
- **Live resource back-pressure**: `resource_monitor.py` samples RAM, CPU, and GPU every few seconds. If the system is under memory or compute pressure between videos, the pipeline waits briefly before starting the next one.
- **Cross-platform hardware profiling**: `scripts/hardware.py` builds a frozen `DeviceProfile` at startup that recognises Apple Silicon (P/E cores), NVIDIA CUDA, AMD ROCm, Raspberry Pi 3/4/5, generic ARM/x86 Linux, Intel Macs, Windows CPU/CUDA, and VM/container environments. Whisper device + compute type, llama.cpp `n_threads` / `n_gpu_layers` / `n_batch` / `n_ctx`, Flash Attention + Q8_0 KV cache (Metal), and `use_mlock` are all tuned per profile. Run `python -m scripts.run_pipeline --print-hardware` to dump the detected profile as JSON.
- **Bilingual web UI**: The React site is fully bilingual (English / Italiano). The language is auto-detected from `navigator.language`, user choice persists in `localStorage`, and an EN/IT toggle in the header lets visitors switch on the fly.
- **Graceful LLM degradation**: On very low-RAM hardware (Raspberry Pi 3, Pi Zero 2W, or <1.5 GB containers), the pipeline automatically runs in NER+rules-only mode instead of crashing.
- **Shorts filtering**: YouTube Shorts (URL `/shorts/` or duration ≤60s) are automatically skipped.
- **Recipe filtering**: Videos with recipe keywords in the title are automatically skipped.
- **Global model caching**: Whisper, NER, and LLM models are loaded once per session (NER via `transformers`/`gliner`). On low-end hardware GLiNER is pinned to CPU to leave VRAM for Whisper and the LLM.

## Tech Stack (100% Open Source, No Paid APIs)

| Component      | Tool                             | License    |
|----------------|----------------------------------|------------|
| Video download | yt-dlp                           | Unlicense  |
| Transcription  | faster-whisper / openai-whisper (primary) + YouTube manual subs via yt-dlp (when present) | MIT |
| Venue NER      | GLiNER + PyTorch + Hugging Face `transformers` | Apache-2.0 / BSD |
| LLM inference  | llama-cpp-python + GGUF (Qwen 2.5 32B default; auto-selected per RAM, from 72B down to TinyLlama 1B on Raspberry Pi) | MIT     |
| Geocoding      | Nominatim (OpenStreetMap)        | ODbL       |
| Place verification | Overpass API (OpenStreetMap)  | ODbL       |
| Deduplication  | thefuzz (fuzzy matching)         | MIT        |
| Validation     | Pydantic                         | MIT        |
| Frontend       | React + TypeScript + Vite        | MIT        |
| Map            | Leaflet + react-leaflet + OSM    | BSD-2/ODbL |
| CI/CD          | GitHub Actions + GitHub Pages    | —          |
| Testing        | pytest                           | MIT        |

## Hardware Support

The pipeline detects the host environment exactly once at startup
(`scripts.hardware.get_profile()`) and configures every model accordingly.
Run `python -m scripts.run_pipeline --print-hardware` to see the active
profile.

| Profile                         | Whisper                          | LLM tier               | `n_threads`  | `n_gpu_layers` |
|---------------------------------|----------------------------------|------------------------|--------------|----------------|
| Apple Silicon (M-class) ≥ 16 GB | `large-v3-turbo` (CPU int8)†     | 14B → 32B              | P-cores      | -1 (Metal)     |
| Linux + NVIDIA, VRAM ≥ 8 GB     | `large-v3-turbo` (CUDA fp16)     | 8B → 72B               | physical-1   | -1             |
| Linux + NVIDIA, VRAM < 8 GB     | `large-v3-turbo` (CUDA int8_fp16)| 8B → 72B               | physical-1   | -1             |
| Linux + AMD ROCm                | `large-v3-turbo` (CPU int8)‡     | per RAM                | physical-1   | -1             |
| Linux x86_64 CPU-only           | `large-v3-turbo` / `medium`      | 8B / 14B               | physical-1   | 0              |
| Linux generic ARM64             | `large-v3-turbo` / `small`       | per RAM                | physical-1   | 0              |
| Raspberry Pi 5 (8 GB)           | `small`                          | 3B (e.g. Phi-3-mini)   | 4            | 0              |
| Raspberry Pi 4 (4 GB)           | `small`                          | 1B (e.g. TinyLlama)    | 4            | 0              |
| Raspberry Pi 3 / Zero 2W (1 GB) | `tiny`                           | none — rules-only mode | 4            | 0              |
| Intel Mac                       | `large-v3-turbo` / `medium`      | per RAM                | physical-1   | 0              |
| Windows + CUDA                  | as Linux CUDA                    | as Linux CUDA          | physical-1   | -1             |
| Windows CPU-only                | as Linux CPU                     | as Linux CPU           | physical-1   | 0              |
| VM / container (any of above)   | as base profile                  | as base                | base-1       | as base        |

† faster-whisper has no Metal backend — CPU int8 with P-core threads is
strictly faster than the (broken) CTranslate2-MPS path; llama.cpp still uses
Metal via `n_gpu_layers=-1`.
‡ faster-whisper has no native ROCm path; llama.cpp uses ROCm when built with
`-DGGML_HIPBLAS`. The pipeline falls back to CPU Whisper but offloads the LLM.

In every VM/container the pipeline disables `mlock` (it usually fails because
of `RLIMIT_MEMLOCK`) and reserves one core for the host. On Apple Silicon it
enables Flash Attention + Q8_0 quantized KV cache (~25 % memory saving) when
the linked `llama-cpp-python` build supports it.

If the detected RAM is below the LLM threshold, the pipeline transparently
runs with `--skip-extract` so it never crashes on low-end hardware.

## Setup

### Pipeline (Python)

```bash
# Clone
git clone https://github.com/your-user/cibobuono.git
cd cibobuono

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download a GGUF LLM model
mkdir -p models
# Recommended (auto-selected by RAM in this order):
#   ≥40 GB → Qwen2.5-72B-Instruct-Q4_K_M.gguf  or  Llama-3.3-70B-Instruct-Q4_K_M.gguf
#   ≥24 GB → Qwen2.5-32B-Instruct-Q4_K_M.gguf
#            gemma-3-27b-it-Q4_K_M.gguf         (requires HF login: huggingface-cli login)
#   ≥12 GB → Qwen2.5-14B-Instruct-Q4_K_M.gguf  (strong Italian, full CUDA offload ≤16 GB VRAM)
#   ≥6  GB → Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
#   ≥3  GB → Phi-3-mini-4k-Instruct-Q4_K_M.gguf  or  Qwen2.5-3B-Instruct-Q4_K_M.gguf
#   ≥1.5GB → tinyllama-1.1b-chat-v1.0-Q4_K_M.gguf  (Raspberry Pi 4)
# Place the file(s) anywhere under models/. The pipeline auto-selects the
# largest model that fits the detected hardware.

# Inspect the active hardware profile (Whisper + llama.cpp params, JSON)
python -m scripts.run_pipeline --print-hardware

# Add channel URLs
echo "https://www.youtube.com/@your-channel" >> channels_input.txt

# Run the pipeline
python -m scripts.run_pipeline --skip-push --max-videos 5

# Continuous mode (catalog + process new uploads forever)
python -m scripts.run_pipeline --watch --poll-interval 1800

# Check status
python -m scripts.run_pipeline --status

# Run tests
pytest tests/ -v
```

### Site (React)

```bash
cd site
npm install
npm run dev      # Development server (http://localhost:5173)
npm run build    # Production build → site/dist/
```

The site is deployed automatically via GitHub Actions on push to `main`. Go to **Settings → Pages → Source: GitHub Actions** to enable deployment.

## Deployment Flow

```
Pipeline → data/*.json → git push → GitHub Actions → npm build → inject data/ → GitHub Pages
```

The React app fetches `data/*.json` at runtime via relative URLs. No data is duplicated — the CI workflow copies `data/` into the build output at deploy time.

## Deduplication Strategy

The deduplication uses **two conditions (AND)**:

1. **Geographic proximity**: haversine distance < 200 meters
2. **Name similarity**: ≥ 70 after noise-word normalization

Before comparing, common noise words (generic categories like "pizzeria", "forno", articles like "il", "la", "di") are stripped so that the core proper name is compared. This makes "Sant'Isidoro pizza e bolle" correctly match "Sant'Isidoro Pizze Boiler".

When a duplicate is found, the new name is added to `aliases[]`.

## Manual Review

1. Open `data/flagged_segments.json`
2. Find segments with `"reviewed_by_human": false`
3. Fill in `locale_name`, `city`, `rating` as needed
4. Set `"reviewed_by_human": true`
5. Run: `python -m scripts.handle_flagged_segments`

## Contributing & Reporting Issues

This project extracts data automatically from YouTube videos. Errors are **expected and inevitable** — the pipeline may:

- **False positives**: Extract a "locale" that doesn't actually exist or wasn't visited in the video
- **False negatives**: Miss a real locale that was visited
- **Wrong names**: Misspell or garble a locale name (especially from Whisper transcription)
- **Wrong coordinates**: Geocode to the wrong location
- **Wrong ratings/sentiment**: Misinterpret the blogger's opinion

### How to Report an Error

Please [open a GitHub Issue](../../issues/new) with one of these templates:

#### 🔴 False Positive (locale shouldn't be there)
```
Title: [False Positive] "Locale Name" in video VIDEO_ID
Body:
- Locale name: ...
- Video: https://youtu.be/VIDEO_ID
- Why it's wrong: (the blogger never visited this place / it doesn't exist / etc.)
```

#### 🟡 Wrong Data (name, coordinates, rating, etc.)
```
Title: [Wrong Data] "Locale Name" — wrong name/coordinates/rating
Body:
- Current data: ...
- Correct data: ...
- Video: https://youtu.be/VIDEO_ID
- Timestamp: MM:SS
```

#### 🟢 Missing Locale (false negative)
```
Title: [Missing] "Locale Name" from video VIDEO_ID
Body:
- Locale name: ...
- City: ...
- Video: https://youtu.be/VIDEO_ID
- Timestamp: MM:SS (where the blogger visits it)
```

#### 💡 New Channel Suggestion
```
Title: [Channel] Suggest @ChannelName
Body:
- Channel URL: https://www.youtube.com/@...
- Why: (Italian food review channel)
```

All issues are welcome — even a simple "this is wrong" helps improve the dataset.

## License

- **Dataset** (`data/` directory): Proprietary — © Luca Ostinelli. All rights reserved. Usage only with explicit permission. See [LICENSE-DATASET](LICENSE-DATASET).
- **Code & Site** (`scripts/`, `site/`, `tests/`): [MIT](LICENSE-CODE) — free to use, modify, and distribute.

---

[⬆ Back to English version](#what-is-this)

# 🇮🇹 Versione Italiana

## Cos'è?

Un dataset open source e una mappa interattiva dei locali (forni, panifici, ristoranti, pizzerie, ecc.) recensiti da food YouTuber italiani. Tutti i dati vengono estratti automaticamente con una **pipeline completamente locale** usando solo strumenti open source — nessuna API a pagamento.

La pipeline usa **solo inferenza** (nessun training o fine-tuning): ASR locale, modello **GLiNER** per gli span dei locali, regole deterministiche e LLM istruito locale (GGUF) per verifica e dettagli sulle visite. È richiesto **Python 3.10+**.

**Lingua:** tutti i video catalogati sono in **italiano**; yt-dlp, sottotitoli, Whisper, lingua dei risultati Nominatim e prompt LLM sono allineati all’italiano (`CONTENT_LANGUAGE` in `scripts/utils.py`).

La pipeline trascrive i video YouTube, propone **candidati locale** con NER multilingue locale (GLiNER), applica **regole deterministiche** (italiano) visita vs. semplice menzione e interroga l’LLM locale solo come **verificatore** binario (con evidenza citata) quando le regole sono ambigue; voti e sentiment sono richiesti all’LLM solo per le visite accettate su una finestra breve di trascrizione. Geocoding OSM e JSON normalizzati come sopra; deploy del sito React su GitHub Pages ad ogni push.

## Mappa Interattiva

La mappa è deployata su **GitHub Pages** e si aggiorna automaticamente ad ogni `git push` su `main`. Permette di esplorare tutti i locali recensiti con ricerca e filtri per sentiment.

La web app è **bilingue (Italiano / English)**: la lingua viene rilevata automaticamente dal browser e un toggle IT/EN nell'header permette di cambiarla al volo.

> **Anteprima locale**: `cd site && npm install && npm run dev`

> **Venv Python (pipeline)**: dalla root, `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, poi `python -m scripts.run_pipeline ...`. Opzionale: `CIBOBUONO_LLM_MODEL` → `.gguf`; `CIBOBUONO_NER_MODEL` sovrascrive l’id Hugging Face di GLiNER (default `knowledgator/gliner-x-large-v0.5`). **PyTorch** serve per il NER (CPU o MPS su Apple Silicon); se GLiNER non si carica, resta un fallback euristico leggero.

> **ffmpeg**: necessario nel `PATH` per estrazione audio (yt-dlp) e Whisper (su macOS: `brew install ffmpeg`).

## Struttura della Repository

```
/cibobuono
├── channels_input.txt              # URL dei canali YouTube (uno per riga)
├── requirements.txt                # Dipendenze Python
├── readme.md                       # Questo file
│
├── .github/workflows/
│   └── deploy.yml                  # GitHub Actions: build React + deploy Pages
│
├── data/                           # Dataset JSON normalizzato
│   ├── channels.json               # Canali YouTube
│   ├── videos.json                 # Tutti i video catalogati con stato
│   ├── locales.json                # Locali normalizzati con coordinate
│   ├── visits.json                 # Visite/recensioni dei locali dai video
│   ├── processed_videos.json       # Tracker per incrementalità
│   ├── flagged_segments.json       # Segmenti a bassa confidence per review
│   ├── skipped_videos.json         # Video saltati (ricette, Shorts)
│   ├── corrections.json            # Override manuali hide/edit (preservato da --reset)
│   ├── voices.json                 # Registro voci ricorrenti per canale (Perceptor)
│   ├── perception.json             # Record di percezione audio/video per video (Perceptor)
│   ├── calibration.json            # Parametri Platt-scaling fittati (rigenerato; assente finché non fittato)
│   └── eval_set.json               # Gold set etichettato a mano per l'harness di valutazione del classificatore
│
├── scripts/                        # Moduli della pipeline (Python)
│   ├── schemas.py                  # Modelli Pydantic e validazione
│   ├── utils.py                    # Utilità condivise, percorsi, config, shim hardware
│   ├── hardware.py                 # DeviceProfile cross-platform (parametri Whisper + llama.cpp)
│   ├── resource_monitor.py         # Monitoraggio live RAM/GPU/CPU con back-pressure
│   ├── fetch_channels.py           # Estrazione metadati canali via yt-dlp
│   ├── fetch_videos.py             # Catalogo video, detection ricette/non-food/Shorts, download audio
│   ├── transcribe_video.py         # Trascrizione: Whisper large-v3-turbo (primario) + sottotitoli YouTube manuali
│   ├── chunk_transcription.py      # Divisione trascrizioni in chunk da 90s con 15s overlap
│   ├── video_intelligence.py       # Analisi titolo + descrizione + capitoli + commenti (hint, voto, rubrica)
│   ├── extract_locales.py          # Helper condivisi: gate food, hints, timestamp, handle LLM
│   ├── ner_candidates.py           # Candidati GLiNER (+ fallback euristico; worker paralleli)
│   ├── visit_classifier.py         # Regole italiane + LLM sì/no se ambiguo
│   ├── batch_visit_llm.py          # Valutazione LLM in batch dei candidati NER (ottimizzato GPU)
│   ├── venue_discovery.py          # Discovery olistica LLM su trascrizione completa con timestamp
│   ├── vote_aggregator.py          # Fusione log-odds pesata dei segnali Perceptor nella confidenza visita
│   ├── calibrate_confidence.py     # Ricalibrazione Platt-scaling della confidenza da corrections.json
│   ├── eval_pipeline.py            # Precision/recall/F1 del classificatore visite su data/eval_set.json
│   ├── extract_pipeline.py         # Orchestratore: NER → batch classify → LLM dettagli → filtro cross-chunk → merge olistico
│   ├── geocode_locales.py          # Geocoding Nominatim (gratuito, rate-limited, cache su file)
│   ├── verify_locales.py           # Verifica OSM via Overpass API (anti-falsi positivi)
│   ├── deduplicate_locales.py      # Deduplicazione fuzzy nome + distanza haversine
│   ├── populate_json.py            # Scrittura visite e segmenti flaggati
│   ├── handle_flagged_segments.py  # Importazione segmenti revisionati manualmente → locale + visita
│   ├── manual_edits.py             # Correzioni manuali dalla dashboard (rimozione/aggiunta visite)
│   ├── review_queue.py             # Coda revisioni in attesa + segnalazioni utente
│   ├── repair_stale_state.py       # Ripara video pending già estratti
│   ├── pipeline_executor.py        # PipelineExecutor: finalize in background (overlap GPU/CPU)
│   ├── pipeline_metrics.py         # Aggregazione metriche per run → logs/pipeline_metrics.json
│   ├── pipeline_control.py         # CLI pause/resume/stop/status (file-based)
│   ├── dashboard.py                # Dashboard live nel terminale (Rich) + snapshot JSON
│   ├── perceptor.py                # Orchestratore percezione audio/video (--perceptor)
│   ├── perceptor_audio.py          # Silero VAD + diarizzazione + registro voci per canale
│   ├── perceptor_video.py          # Campionamento frame + novelty phash + captioning Qwen2-VL
│   ├── setup_models.py             # Helper per download e verifica modelli
│   ├── push_to_github.py           # Commit e push su Git
│   ├── validate_data.py            # Validazione data/*.json vs Pydantic (CI / locale)
│   ├── run_pipeline.py             # Orchestratore principale della pipeline (2 fasi; supporta --watch)
│   └── com.cibobuono.pipeline.plist.example  # Esempio LaunchAgent macOS per modalità --watch
│
├── site/                           # React + Vite + TypeScript (GitHub Pages, bilingue EN/IT)
│   ├── package.json                # Dipendenze Node
│   ├── vite.config.ts              # Configurazione Vite (base: /cibobuono/)
│   ├── index.html                  # Entry HTML — titolo + meta description
│   ├── src/
│   │   ├── App.tsx                 # App principale: caricamento dati, filtri, layout
│   │   ├── api.ts                  # Fetch JSON da data/ a runtime
│   │   ├── types.ts                # Tipi TypeScript (mirror degli schema Python)
│   │   ├── i18n/
│   │   │   ├── messages.ts         # Dizionari EN + IT tipizzati
│   │   │   ├── LanguageContext.tsx # Provider (persiste la lingua in localStorage)
│   │   │   └── useLanguage.ts      # Hook useLanguage / useT
│   │   └── components/
│   │       ├── MapView.tsx         # Mappa Leaflet con marker colorati per sentiment
│   │       ├── LocaleList.tsx      # Lista nella sidebar con dettagli visite espandibili
│   │       ├── Header.tsx          # Ricerca + filtri + toggle lingua EN/IT
│   │       └── StatusBar.tsx       # Stati di caricamento/errore
│   └── ...
│
├── tests/                          # Suite di test pytest (464 test)
│   ├── test_schemas.py             # Validazione modelli Pydantic
│   ├── test_dedup.py               # Dedup fuzzy + haversine
│   ├── test_chunks.py              # Chunking + helper timestamp
│   ├── test_extraction.py          # Gate food, caching LLM, validazione locale, hint descrizione
│   ├── test_ner_candidates.py      # Candidati GLiNER + recovery eccezioni parallele
│   ├── test_visit_classifier.py    # Classificatore regole italiane + controlli nome locale + ensemble self-consistency
│   ├── test_batch_visit_llm.py     # Valutazione LLM in batch
│   ├── test_venue_discovery.py     # Discovery olistica LLM + formattazione trascrizione
│   ├── test_extract_pipeline.py    # Pipeline completa NER→classify→merge + fusione Perceptor + bonus accordo
│   ├── test_vote_aggregator.py     # Voti OCR/speaker Perceptor + combinazione log-odds della confidenza
│   ├── test_calibrate_confidence.py # Fit/apply Platt-scaling + roundtrip save/load
│   ├── test_eval_pipeline.py       # Harness precision/recall/F1 del classificatore visite
│   ├── test_verify.py              # Verifica OSM/Overpass
│   ├── test_geocode.py             # Logica cache geocoding + batch geocode
│   ├── test_perceptor.py           # Novelty dedup, diarizzazione, registro voci, guardie stage
│   ├── test_validate_data.py       # Gate validate_data.py (tutti OK / errori riportati)
│   ├── test_utils.py               # Utilità condivise
│   ├── test_data_integrity.py      # Integrità referenziale JSON
│   ├── test_hardware.py            # DeviceProfile su piattaforme simulate
│   ├── test_populate_json.py       # populate_visits + populate_flagged
│   ├── test_manual_edits.py        # Correzioni manuali visite dalla dashboard
│   ├── test_repair_stale_state.py  # Riparazione pending → processed
│   ├── test_resource_monitor.py    # Monitoraggio RAM/GPU/CPU
│   ├── test_pipeline_control.py    # Controllo pause/stop/stato
│   ├── test_pipeline_executor.py   # Coerenza geografica, finalize, overlap GPU/CPU
│   ├── test_pipeline_metrics.py    # Aggregazione metriche run + eval metrics + append
│   ├── test_review_queue.py        # Revisioni in attesa + segnalazioni
│   ├── test_dashboard.py           # Stato dashboard + snapshot
│   ├── test_transcribe.py          # Parser VTT + selezione backend Whisper
│   ├── test_video_intelligence.py  # Analisi titolo/descrizione/capitoli/commenti
│   └── test_watch_loop.py          # Modalità --watch (daemon)
│
├── models/                         # Modelli LLM GGUF (gitignored)
├── cache/                          # Audio e trascrizioni temporanee (gitignored)
└── logs/                           # Log della pipeline (gitignored)
```

## Modello Dati

Tutti i file JSON sono **normalizzati** — nessun array di ID annidato nei record. Le visite fanno riferimento a locali, video e canali tramite ID. Il sito calcola le relazioni a runtime.

### channels.json

| Campo         | Tipo     | Descrizione                                |
|---------------|----------|--------------------------------------------|
| `channel_id`  | string   | ID deterministico dall'URL del canale      |
| `name`        | string   | Nome del canale                            |
| `url`         | string   | URL del canale YouTube                     |
| `description` | string   | Descrizione del canale                     |
| `rubriche`    | string[] | Nomi delle rubriche inferiti dai titoli    |

### videos.json

| Campo            | Tipo   | Descrizione                               |
|------------------|--------|-------------------------------------------|
| `video_id`       | string | ID video YouTube                          |
| `channel_id`     | string | FK → channels.json                        |
| `title`          | string | Titolo del video                          |
| `url`            | string | URL YouTube                               |
| `publish_date`   | string | YYYY-MM-DD                                |
| `processed_date` | string | YYYY-MM-DD (vuoto se pending)             |
| `status`         | enum   | `pending` / `processed` / `errored`       |

### locales.json

| Campo       | Tipo     | Descrizione                                    |
|-------------|----------|------------------------------------------------|
| `locale_id` | string   | Hash SHA256 di nome normalizzato + coordinate  |
| `name`      | string   | Nome principale del locale                     |
| `aliases`   | string[] | Nomi alternativi (deduplicazione)              |
| `address`   | string   | Indirizzo                                      |
| `city`      | string   | Città                                          |
| `lat`       | float    | Latitudine (4 decimali)                        |
| `lon`       | float    | Longitudine (4 decimali)                       |
| `category`  | string[] | Categorie (forno, ristorante...)               |

### visits.json

| Campo             | Tipo   | Descrizione                                         |
|-------------------|--------|-----------------------------------------------------|
| `visit_id`        | string | `visit_{video_id}_{secondi_timestamp}`               |
| `locale_id`       | string | FK → locales.json                                    |
| `video_id`        | string | FK → videos.json                                     |
| `channel_id`      | string | FK → channels.json                                   |
| `timestamp_start` | string | MM:SS o HH:MM:SS                                     |
| `timestamp_end`   | string | MM:SS o HH:MM:SS                                     |
| `youtube_url`     | string | Link diretto con parametro `?t=`                     |
| `rating`          | string? | Voto complessivo come detto dal blogger (es. `8`, `8--`, `6++`), null se assente |
| `sentiment`       | enum   | `positive` / `neutral` / `negative`                   |
| `rubrica`         | string | Nome della rubrica                                    |
| `llm_confidence`  | float  | Confidenza dell'estrazione (0-1)                      |
| `extraction_date` | string | YYYY-MM-DD                                            |
| `date`            | string | Data approssimativa della visita (= data pubblicazione)|

### processed_videos.json

| Campo              | Tipo   | Descrizione                      |
|--------------------|--------|----------------------------------|
| `video_id`         | string | ID video YouTube                 |
| `channel_id`       | string | FK → channels.json               |
| `processed_date`   | string | YYYY-MM-DD                       |
| `status`           | enum   | Risultato del processamento      |
| `visits_extracted`  | int    | Numero di visite trovate         |
| `flagged_segments`  | int    | Numero di segmenti flaggati      |

### flagged_segments.json

| Campo              | Tipo    | Descrizione                              |
|--------------------|---------|------------------------------------------|
| `video_id`         | string  | FK → videos.json                         |
| `channel_id`       | string  | FK → channels.json                       |
| `timestamp_start`  | string  | Inizio segmento                          |
| `timestamp_end`    | string  | Fine segmento                            |
| `youtube_url`      | string  | Link diretto                             |
| `reason`           | enum    | Motivo del flag                           |
| `extracted_text`   | string  | Testo della trascrizione                 |
| `llm_confidence`   | float   | Punteggio di confidenza                  |
| `reviewed_by_human`| bool    | Revisionato manualmente                  |
| `reviewed_date`    | string? | Data della revisione                     |
| `locale_name`      | string? | Da compilare durante la revisione        |
| `rating`           | string? | Da compilare in revisione (es. `8`, `8--`) |
| `city`             | string? | Da compilare durante la revisione        |

### skipped_videos.json

| Campo          | Tipo   | Descrizione                                          |
|----------------|--------|------------------------------------------------------|
| `video_id`     | string | ID video YouTube                                     |
| `channel_id`   | string | FK → channels.json                                   |
| `title`        | string | Titolo del video                                     |
| `url`          | string | URL YouTube                                          |
| `reason`       | string | Motivo dello skip (keyword ricetta, Short)            |
| `skipped_date` | string | YYYY-MM-DD                                           |

### corrections.json

| Campo       | Tipo    | Descrizione                                              |
|-------------|---------|-----------------------------------------------------------|
| `locale_id` | string  | FK → locales.json                                          |
| `type`      | enum    | `hide` (falso positivo) / `edit` (override di un campo)    |
| `reason`    | string? | Motivo leggibile                                           |
| `overrides` | object? | Per `type=edit`: override di `name`/`city`/`rating`/`sentiment` |

Preservato tra i `--reset` (azzerato solo con `--reset-all-data`). Le voci `hide` sono anche la ground truth debole su cui `scripts/calibrate_confidence.py` fitta la calibrazione — vedi [Confidenza e Ridondanza](#confidenza-e-ridondanza).

### voices.json

Registro per canale delle voci ricorrenti, scritto da `scripts/perceptor_audio.py`. Una voce per ogni speaker rilevato:

| Campo         | Tipo     | Descrizione                                            |
|---------------|----------|------------------------------------------------------------|
| `voice_id`    | string   | `voice_{channel_id}_{NNN}`                                  |
| `channel_id`  | string   | FK → channels.json                                           |
| `centroid`    | float[]  | Embedding TitaNet 192-dim a media mobile                     |
| `n_samples`   | int      | Numero di video uniti nel centroide                          |
| `videos`      | string[] | Video ID in cui questa voce è comparsa                       |
| `created_at`  | string   | Timestamp ISO-8601                                           |
| `updated_at`  | string   | Timestamp ISO-8601                                           |

### perception.json

Un record per ogni video processato con `--perceptor`, scritto da `scripts/perceptor.py`:

| Campo        | Tipo    | Descrizione                                                |
|--------------|---------|-----------------------------------------------------------------|
| `video_id`   | string  | FK → videos.json                                                 |
| `channel_id` | string  | FK → channels.json                                                |
| `status`     | enum    | `ok` / `partial` / `errored`                                       |
| `error`      | string? | Riassunto troncato dell'errore se non `ok`                         |
| `asr_backend`| string  | Backend usato per questo video (`mlx_whisper`/`faster_whisper`)    |
| `audio`      | object? | Segmenti VAD, tempo di parola per speaker + `voice_id` associato, speaker per segmento di trascrizione |
| `video`      | object? | Frame campionati/caption, caption VLM deduplicate per novelty con timestamp |

`extract_pipeline.py` legge questo record per video (`get_perception`) e passa le sue caption/diarizzazione a `scripts/vote_aggregator.py` come voti indipendenti sulla decisione visita/menzione — vedi [Confidenza e Ridondanza](#confidenza-e-ridondanza).

### calibration.json

Scritto da `scripts/calibrate_confidence.py`; assente finché lo script non viene eseguito almeno una volta.

| Campo       | Tipo   | Descrizione                                                 |
|-------------|--------|-------------------------------------------------------------------|
| `fitted`    | bool   | Se c'erano abbastanza dati etichettati per fittare una calibrazione |
| `a`, `b`    | float  | Parametri Platt-scaling (presenti solo se `fitted=true`)            |
| `n_samples` | int    | Numero di coppie (confidenza, esito) usate per il fit                |

### eval_set.json

Gold set etichettato a mano usato da `scripts/eval_pipeline.py`. Lista JSON semplice, parte vuota finché non viene popolata a mano:

| Campo            | Tipo    | Descrizione                                                |
|------------------|---------|-------------------------------------------------------------------|
| `candidate_name` | string  | Nome del locale come compare nella finestra di trascrizione         |
| `window_text`    | string  | Estratto della trascrizione attorno alla menzione                   |
| `start_time`     | float?  | Momento della menzione in secondi (default 0.0)                     |
| `ner_score`      | float?  | Confidenza NER simulata dello span (default 0.6)                    |
| `gold_label`     | enum    | `visit` / `mention` — la risposta corretta                          |

## Relazioni tra Entità

```
channels.json ←── videos.json ←── visits.json ──→ locales.json
                       │
                skipped_videos.json
                       │
                   flagged_segments.json (revisione manuale → aggiorna i JSON sopra)
```

Tutti gli ID sono **deterministici** per garantire idempotenza.

## Pipeline

Esecuzione locale (on-demand o schedulata):

```bash
python -m scripts.run_pipeline --skip-push --max-videos 10
```

### Architettura

La pipeline è divisa in due fasi:

**Fase 1 — Catalogo**: Fetch di tutti i video del canale via yt-dlp. Inserimento in `videos.json` con `status=pending`. I video di ricette e gli YouTube Shorts vengono rilevati e spostati in `skipped_videos.json`.

**Fase 2 — Processamento**: Per ogni video pending (dal più recente, fino a `--max-videos`; `0` = tutti i pending):

| #  | Passaggio            | Strumento                    | Descrizione                                          |
|----|----------------------|------------------------------|------------------------------------------------------|
| 1  | Fetch canali         | yt-dlp                       | Estrai metadati dagli URL dei canali                |
| 2  | Catalogo video       | yt-dlp                       | Cataloga tutti i video, salta ricette e Shorts      |
| 3  | Prefetch audio       | yt-dlp                       | Finestra mobile: pre-scarica fino a 20 file audio   |
| 4  | Trascrizione         | Whisper `large-v3-turbo` (faster-whisper) + sottotitoli YT manuali | Whisper come primario; sottotitoli YouTube *manuali* preferiti quando presenti (gli auto-generati sono esclusi: massacrano i nomi propri italiani) |
| 5  | Chunking             | Python                       | Dividi in chunk da 90s con overlap 15s              |
| 6  | Estrazione           | GLiNER + regole + LLM locale | Candidati NER → visita/menzione → JSON dettaglio (voti) solo se visita |
| 7  | Filtro cross-chunk   | Python                       | Mantieni il locale se ≥2 chunk **oppure** hint titolo/descrizione      |
| 8  | Geocoding            | Nominatim (OSM, gratis)      | Nome + città → lat/lon                              |
| 9  | Verifica (OSM)       | Overpass API (OSM, gratis)   | Conferma che il locale esiste come attività reale   |
| 10 | Deduplicazione       | thefuzz + haversine          | <200m E similarità nome ≥70%                        |
| 11 | Popolamento          | Python                       | Scrivi visits.json, aggiorna locales.json           |
| 12 | Flag                 | Python                       | Bassa confidence → flagged_segments.json            |
| 13 | Push                 | git                          | Commit e push dei dati aggiornati                   |
| 14 | Deploy               | GitHub Actions                | Build sito React, deploy su Pages                   |

### Estrazione — Dettaglio Neuro-Simbolico

La fase di estrazione (passo 6 nella tabella) combina tre livelli:

```
Trascrizione completa con timestamp
     │
     ├─→ [A] Discovery olistica LLM  ──────────────────────────────────→ lista locali + timestamp
     │         (venue_discovery.py)                                               │
     │                                                                            ↓
     └─→ [B] Chunk da 90 s (overlap 15 s)                        [C] Merge + filtro cross-chunk
               │                                                                  │
               ├─→ GLiNER NER (parallelo, 4 worker)                              ↓
               │   knowledgator/gliner-x-large-v0.5            Estrazioni finali con
               │   19 etichette zero-shot:                      rating + sentiment + timestamp
               │   Locale: restaurant · ristorante · pizzeria
               │           trattoria · forno · panificio
               │           pasticceria · gelateria · osteria
               │           bakery · street food stall
               │           food market · bar or cafe
               │   Contesto: city · neighborhood · country
               │             person · brand · food dish
               │
               └─→ Classificatore visita ──→ visite confermate ──→ Dettaglio LLM in batch
                   (visit_classifier.py)                           (batch_visit_llm.py)
                   Regole italiane prima:                          rating: "8" / "8--" / "6++"
                   • verbi di visita espliciti                     sentiment: positive/neutral/negative
                   • preposizioni di movimento
                   LLM sì/no solo se ambiguo
                   (deve citare uno span testuale verbatim)
```

**[A] Discovery olistica** (`venue_discovery.py`): L'intera trascrizione con timestamp viene inviata all'LLM in un unico prompt JSON strutturato. L'LLM restituisce una lista di locali visitati con timestamp approssimativi e qualsiasi voto o sentiment dichiarato. Questo passaggio a contesto esteso cattura i locali che il NER a chunk ha mancato.

**[B] NER GLiNER** (`ner_candidates.py`): Ogni chunk da 90 secondi viene analizzato da GLiNER x-large in parallelo su 4 CPU worker. Su sistemi CUDA, GLiNER è fissato su CPU per lasciare la VRAM a Whisper e all'LLM. Impostando `CIBOBUONO_GLINER_CPU=0` si abilita l'inferenza GPU (GLiNER x-large occupa ~1.5 GB su GPU; su macchine con ≥16 GB VRAM e un LLM 14B c'è spazio abbondante, rendendo il NER 10–20× più veloce).

**Classificatore visita** (`visit_classifier.py`): Regole regex italiane decidono la maggior parte dei casi. L'LLM viene chiamato solo per i candidati ambigui e deve rispondere con un sì/no binario e uno span di evidenza testuale verbatim.

**Valutazione LLM in batch** (`batch_visit_llm.py`): Tutti i candidati NER confermati come visite vengono inviati all'LLM in una singola chiamata batch che estrae `rating` e `sentiment` per ogni visita in una volta sola.

**[C] Merge + confidenza** ora va oltre "tieni la riga con confidenza più alta": accordo tra fonti, voto self-consistency sui casi ambigui e i segnali audio/video di Perceptor concorrono tutti alla confidenza finale — vedi [Confidenza e Ridondanza](#confidenza-e-ridondanza) più sotto.

**Parametri dei chunk**: 90 s con 15 s di overlap garantisce che una menzione a cavallo di un confine venga catturata. Una tipica visita a un ristorante in questi video dura 60–180 s di commento, quindi 90 s è una finestra naturale.

**Formato del voto**: Il campo `rating` memorizza il punteggio esatto espresso dal blogger come stringa:

| Formato | Significato |
|---------|-------------|
| `"8"` | Punteggio intero esatto (più comune) |
| `"8--"` | Poco sotto l'8 (`"quasi un 8 meno meno"`) |
| `"8-"` | Appena sotto l'8 |
| `"7+"` | Appena sopra il 7 |
| `"7++"` | Quasi un 8 |
| `null` | Nessun punteggio numerico dichiarato; il sentiment viene comunque estratto |

Il core numerico deve essere compreso tra 1 e 10; il validator Pydantic rifiuta valori fuori range.

**`initial_prompt` di Whisper**: Ogni trascrizione viene introdotta da un breve preambolo italiano con tipi di locali, nomi noti, città italiane e frasi di visita. Questo riduce significativamente gli errori ASR sui nomi propri dei ristoranti rispetto alla decodifica senza prompt.

**Scelta dei modelli**:

| Componente | Modello | Perché |
|------------|---------|--------|
| ASR | `whisper-large-v3-turbo` (faster-whisper, CUDA fp16) | Miglior rapporto velocità/accuratezza per l'italiano; il `large-v3` full è ~2× più lento per un guadagno marginale |
| NER | `knowledgator/gliner-x-large-v0.5` | Il più grande GLiNER zero-shot disponibile; backbone mT5 multilingue, nessun fine-tuning richiesto |
| LLM | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` | Full CUDA offload a 8.4 GB; ottimo italiano e output JSON strutturato; il tier 27B default (`gemma-3-27b`) richiede autenticazione HF |
| Geocoding | Nominatim + Overpass (OSM) | Nessuna API a pagamento; Nominatim copre bene le città italiane; la verifica Overpass `amenity=*` è il filtro anti-falsi-positivi più efficace |

### Confidenza e Ridondanza

Ogni fase di classificazione sopra descritta (regole, arbiter LLM, LLM in batch, discovery olistica) girava **una sola volta** per candidato, e i risultati venivano uniti tenendo "la confidenza più alta" — l'accordo tra fonti indipendenti veniva scartato, e i segnali audio/video di Perceptor (caption, diarizzazione) venivano calcolati ma mai consultati. Quattro meccanismi colmano questo divario, tutti valutati secondo l'intuizione del teorema della giuria di Condorcet: segnali *indipendenti* che concordano sono più informativi che richiedere di nuovo a un unico votante correlato:

1. **Bonus di accordo tra fonti** (`extract_pipeline._merge_extraction_rows`): quando la pipeline NER+regole a chunk e il passaggio di discovery olistica LLM (`venue_discovery.py`) concordano indipendentemente sullo stesso locale, la confidenza della riga unita viene alzata (`_SOURCE_AGREEMENT_BONUS`) invece di tenere solo la più alta delle due.

2. **Self-consistency selettiva per i casi realmente ambigui** (`visit_classifier.classify_with_llm_ensemble`): quando il motivo del rule engine è genuinamente ambiguo (`conflict_patterns`, `no_clear_signal` — scattano sia un pattern di menzione sia uno di visita, oppure nessuno dei due), 3 campioni LLM diversificati votano invece di 1. (`empty_window` — nessun testo di trascrizione disponibile — è deliberatamente escluso: non c'è evidenza su cui i 3 campioni possano dissentire in modo utile, quindi resta una singola chiamata.) (baseline greedy, un ricampionamento a temperatura più alta, e una riformulazione "avvocato del diavolo" che argomenta prima il caso menzione). La confidenza scala in base a quanto i campioni concordano. Questo scatta solo per il sottoinsieme ambiguo — la maggior parte dei casi "unsure" (es. un pattern di visita con evidenza food debole) si risolve ancora con una singola chiamata, perché il costo 3x dell'LLM vale solo dove le regole non riescono davvero a decidere.

3. **Fusione dei segnali Perceptor** (`vote_aggregator.py`): gli unici votanti a *modalità indipendente* della pipeline. Una caption VLM vicina che menziona il nome del locale/testo di insegna vota "visita"; lo speaker diarizzato al timestamp del candidato che è la voce dominante/registrata del canale (proxy euristico per "l'host") vota "visita", mentre uno speaker diverso (ospite/bystander) vota "menzione" — intercettando un ospite che descrive la propria visita letta erroneamente come visita dell'host. I voti si combinano con la confidenza della pipeline testuale tramite una somma log-odds pesata (`combine_confidence`), non una media ingenua; i votanti Perceptor hanno oggi un peso più basso della pipeline testuale perché non c'è ancora abbastanza storico di esiti corretti per validarne l'affidabilità. Non fa nulla (torna alla sola confidenza testuale) quando Perceptor è disabilitato o non ha nulla da dire vicino a quel timestamp.

4. **Ricalibrazione della confidenza dalle correzioni** (`calibrate_confidence.py`): la formula lineare tarata a mano sopra non è mai stata verificata contro esiti reali. `python -m scripts.calibrate_confidence` fitta un Platt scaling a 2 parametri (`data/calibration.json`) dalle voci `hide` di `data/corrections.json` (falsi positivi confermati) contro `llm_confidence` di `data/visits.json`. Deliberatamente *non* isotonic regression: richiede molti più punti etichettati di quanti ne avrà per un po' un file corrections a crescita lenta, e un fit a 2 parametri degrada meglio su pochi dati. Sotto `MIN_SAMPLES` (30) coppie etichettate, la calibrazione resta non fittata e la formula lineare resta invariata.

Per misurare se tutto questo aiuta davvero: `python -m scripts.eval_pipeline [--with-llm]` esegue il classificatore visite contro un gold set etichettato a mano (`data/eval_set.json`, parte vuoto — vedi [Modello Dati](#eval_setjson)) e riporta precision/recall/F1, tracciati nel tempo in `logs/eval_metrics.json` via `pipeline_metrics.record_eval_metrics`. Eseguirlo prima e dopo aver abilitato un meccanismo di ridondanza per attribuire il guadagno (o confermare che non c'è) invece di giudicare a sensazione.

### Opzioni della Pipeline

```
python -m scripts.run_pipeline --help

Opzioni:
  --skip-fetch         Salta il catalogo dei nuovi video (lavora solo sui pending)
  --skip-transcribe    Usa trascrizioni in cache
  --skip-extract       Salta l'estrazione LLM
  --skip-push          Non fare commit/push su GitHub
  --whisper-model      tiny|base|small|medium|large|large-v2|large-v3|large-v3-turbo
                       (default: auto-selezionato — large-v3-turbo su Apple Silicon / ≥8 GB)
  --max-videos N       Max video pending per run (default: 100); 0 = tutti i pending
  --no-dashboard       Disabilita la dashboard live nel terminale
  --reset              Resetta i JSON della pipeline, cache e log (mantiene corrections salvo --reset-all-data)
  --reset-all-data     Con --reset, azzera anche corrections.json
  --repair-stale-state Allinea videos.json/processed_videos.json se un video è pending ma ha già visite/flagged, poi esce
  --repair-dry-run     Con --repair-stale-state, solo stampa (nessuna scrittura)
  --status             Mostra il riepilogo dello stato e termina
  --watch              Modalità continua: cataloga + processa nuovi video in loop
  --poll-interval N    Con --watch, secondi tra un ciclo e l'altro (default: 1800, min: 60)
  --no-parallel-postprocess  Geocoding/OSM/popolamento sincrono (disabilita overlap GPU/CPU)
  --print-hardware     Stampa il profilo hardware rilevato come JSON ed esce
  --perceptor          Abilita la percezione audio/video (VAD, diarizzazione,
                       registro voci, captioning frame); anche via CIBOBUONO_PERCEPTOR=1
```

### Perceptor (percezione audio/video, `--perceptor`)

Stage opzionale per video che arricchisce la pipeline con ciò che si *sente*
e si *vede*, scegliendo il backend migliore per la macchina:

| Componente | Tecnologia | Selezione backend |
|---|---|---|
| Attività vocale | Silero VAD via sherpa-onnx | CPU, ogni tier |
| Diarizzazione speaker | Embedding TitaNet-small + clustering coseno | CPU, ogni tier |
| Voci ricorrenti | Registro per canale (`data/voices.json`) | CPU, ogni tier |
| ASR | Whisper large-v3-turbo | mlx-whisper (Metal) su Apple Silicon; faster-whisper fp16 su CUDA; faster-whisper int8 su CPU |
| Novelty frame | Campionamento PyAV + hash percettivo (phash) | CPU; intervallo 2–5 s scalato sull'hardware |
| Captioning | Qwen2-VL | mlx-vlm 4-bit su Apple Silicon; transformers su CUDA (7B AWQ ≥16 GB VRAM, 2B fp16 ≥8 GB); disattivato su CPU-only |

I risultati finiscono in `data/perception.json` (per video: segmenti VAD,
speaker con tempo di parola, voci del canale riconosciute, caption dei frame
nuovi con timestamp). Lo stage è best-effort: un errore di percezione viene
registrato e la pipeline principale prosegue. Quando abilitati, questi segnali
tornano indietro anche nella confidenza di estrazione come voti indipendenti —
vedi [Confidenza e Ridondanza](#confidenza-e-ridondanza). Setup:

```bash
python -m scripts.setup_models --perceptor-only   # modelli ONNX + warm cache VLM
python -m scripts.perceptor <video_id>            # one-shot su un video in cache
CIBOBUONO_PERCEPTOR=1 python -m scripts.run_pipeline --max-videos 1
```

Controllo pipeline (pausa/ripresa/stop di una pipeline in esecuzione):

```bash
python -m scripts.pipeline_control status|pause|resume|stop
```

### Modalità continua (`--watch`)

Esegui la pipeline come daemon. A ogni intervallo di poll rilegge tutti i canali, mette in coda i video nuovi come `pending`, ne processa fino a `--max-videos` (dal più recente) e pusha il delta JSON su GitHub. Se il processo viene fermato e riavviato in seguito, tutti i video pubblicati nel frattempo entrano in coda al primo ciclo successivo.

```bash
# Locale, senza push (utile in sviluppo):
python -m scripts.run_pipeline --watch --poll-interval 1800 --skip-push

# Produzione: poll ogni 30 min, push automatico a ogni cambio:
python -m scripts.run_pipeline --watch
```

Comportamento:
- Whisper, NER e LLM vengono caricati **una sola volta** al primo ciclo e riutilizzati — la RAM non cresce nel tempo.
- La dashboard live viene forzata su off in `--watch` (solo log) per non rompere con `tmux`/`launchd`.
- `git push` parte solo se `git status --porcelain data/` mostra cambiamenti reali. Cicli senza nuovi upload = zero commit.
- **Ctrl+C / SIGTERM** ferma in modo pulito al confine del ciclo (o, se siamo in mezzo a un video, alla fine del video corrente come in modalità one-shot). Premi due volte per uscita immediata. Al riavvio si riprende dallo stato `pending` in `videos.json` grazie alle scritture atomiche.
- Un'eccezione transitoria in un ciclo (es. yt-dlp che fa cilecca) viene loggata e il loop continua; `SystemExit` (es. modello GGUF mancante) termina il daemon.

Per autostart su macOS vedi `scripts/com.cibobuono.pipeline.plist.example`.

**Interruzioni (Ctrl+C / SIGTERM)** — Primo segnale: termina il video corrente, poi si ferma (stato coerente tra un video e l’altro). Secondo segnale: uscita immediata (se serve, `python -m scripts.run_pipeline --repair-stale-state`). I JSON vengono scritti con sostituzione atomica per ridurre file corrotti se il processo muore durante il salvataggio.

**Esempio run completo sul catalogo** (dopo reset, tutti i pending, senza push):

```bash
python -m scripts.run_pipeline --reset --skip-push --max-videos 0 --no-dashboard
```

### Funzionalità Chiave

- **Processamento dal più recente**: I video vengono processati dal più recente così gli upload più freschi arrivano sulla mappa per primi.
- **Whisper come ASR primario**: Whisper locale `large-v3-turbo` è il motore di trascrizione principale — via `mlx-whisper` (GPU Metal) su Apple Silicon, `faster-whisper` (CTranslate2) su CUDA/CPU. I sottotitoli *manuali* di YouTube sono ancora preferiti quando l'autore li ha caricati (rari ma di altissima qualità); i sottotitoli *auto-generati* di YouTube sono esclusi perché l'ASR italiano massacra i nomi propri (locali, vie) che sono il cuore del dataset.
- **Whisper con `initial_prompt`**: Whisper gira con un prompt di terminologia food italiana per orientare la trascrizione sui nomi dei locali, e con `vad_filter=True` su faster-whisper per scartare silenzi/sigle.
- **Modalità continua (`--watch`)**: Loop daemon opzionale che cataloga e processa i nuovi upload all'infinito, con i modelli caricati una sola volta tra i cicli e Ctrl+C graceful.
- **Descrizioni video come contesto**: Dalla descrizione si estraggono via regex gli **hint** sui nomi dei locali; un hint può proteggere un candidato così che una sola menzione in un chunk resti una visita catalogata quando le regole confermano la visita.
- **Verifica OSM dei locali reali**: Dopo il geocoding, ogni locale viene verificato su OpenStreetMap tramite Overpass API (raggio 500m, fuzzy name match ≥ 80) — se nessun locale di ristorazione corrispondente esiste vicino alle coordinate, l'estrazione viene rifiutata. È la misura anti-falsi-positivi più potente.
- **Finestra mobile**: I file audio vengono pre-scaricati in una finestra di 20. Man mano che un video viene processato, il più vecchio viene cancellato e il prossimo viene scaricato.
- **Estrazione neuro-simbolica**: GLiNER propone gli span; regex/euristiche italiane decidono molti casi; l’LLM risponde sì/no visita con uno span di evidenza citato solo se le regole sono incerte (niente prompt monolitico “estrai tutto”).
- **Filtro video non-food**: I video con keyword non-food nel titolo o descrizione (boxing, gaming, fitness, ecc.) vengono automaticamente saltati prima della trascrizione.
- **Holistic venue discovery**: Un singolo passaggio LLM strutturato sull'intera trascrizione con timestamp (`venue_discovery.py`) trova visite che il NER a chunk ha mancato. I risultati vengono uniti all'output NER mantenendo la voce a confidenza più alta per ogni nome.
- **Batch LLM evaluation**: Su CUDA, i candidati NER vengono valutati in batch invece che uno alla volta, riducendo il numero di chiamate LLM per video (`batch_visit_llm.py`).
- **Overlap GPU/CPU (`PipelineExecutor`)**: Geocoding, verifica OSM, deduplicazione e scrittura JSON girano in un thread in background mentre la GPU processa il video successivo. Configurabile con `--no-parallel-postprocess`.
- **Coerenza geografica**: Dopo il geocoding, la città di ogni estrazione viene confrontata con la `video_intel.city` via fuzzy matching. Le discrepanze abbassano la confidenza invece di scartare l'estrazione.
- **Pipeline metrics**: Ogni run aggiunge un record JSON strutturato a `logs/pipeline_metrics.json` con tassi geocoding/OSM/pubblicazione, tasso city-mismatch e distribuzione delle confidenze.
- **Back-pressure risorse**: `resource_monitor.py` campiona RAM, CPU e GPU ogni pochi secondi. Se il sistema è sotto pressione tra un video e l'altro, la pipeline attende brevemente prima di iniziare il prossimo.
- **Profiling hardware cross-platform**: `scripts/hardware.py` costruisce all'avvio un `DeviceProfile` immutabile che riconosce Apple Silicon (core P/E), NVIDIA CUDA, AMD ROCm, Raspberry Pi 3/4/5, ARM/x86 Linux generico, Intel Mac, Windows CPU/CUDA e ambienti VM/container. Device + compute type di Whisper, `n_threads` / `n_gpu_layers` / `n_batch` / `n_ctx` di llama.cpp, Flash Attention + KV cache Q8_0 (Metal) e `use_mlock` sono tarati per profilo. `python -m scripts.run_pipeline --print-hardware` stampa il profilo in JSON.
- **Sito web bilingue**: La web app React è completamente bilingue (Italiano / English). La lingua viene rilevata automaticamente da `navigator.language`, la scelta persiste in `localStorage` e un toggle IT/EN nell'header permette di cambiarla al volo.
- **Degradazione LLM graceful**: Su hardware con pochissima RAM (Raspberry Pi 3, Pi Zero 2W, container <1.5 GB) la pipeline parte automaticamente in modalità solo NER+regole invece di crashare.
- **Filtro Shorts**: Gli YouTube Shorts (URL `/shorts/` o durata ≤60s) vengono automaticamente saltati.
- **Filtro ricette**: I video con parole chiave di ricette nel titolo vengono automaticamente saltati.
- **Caching globale dei modelli**: Whisper, NER e LLM vengono caricati una sola volta per sessione (NER via `transformers`/`gliner`). Sull'hardware con CUDA, GLiNER è fissato su CPU per liberare VRAM per Whisper e l'LLM.

## Stack Tecnologico (100% Open Source, Nessuna API a Pagamento)

| Componente     | Strumento                        | Licenza    |
|----------------|----------------------------------|------------|
| Download video | yt-dlp                           | Unlicense  |
| Trascrizione   | faster-whisper / openai-whisper (primario) + sottotitoli YouTube manuali via yt-dlp (quando presenti) | MIT |
| NER locali     | GLiNER + PyTorch + Hugging Face `transformers` | Apache-2.0 / BSD |
| Inferenza LLM  | llama-cpp-python + GGUF (Qwen 2.5 32B default; auto-selezione per RAM, da 72B fino a TinyLlama 1B su Raspberry Pi) | MIT     |
| Geocoding      | Nominatim (OpenStreetMap)        | ODbL       |
| Verifica locali | Overpass API (OpenStreetMap)     | ODbL       |
| Deduplicazione | thefuzz (matching fuzzy)         | MIT        |
| Validazione    | Pydantic                         | MIT        |
| Frontend       | React + TypeScript + Vite        | MIT        |
| Mappa          | Leaflet + react-leaflet + OSM    | BSD-2/ODbL |
| CI/CD          | GitHub Actions + GitHub Pages    | —          |
| Testing        | pytest                           | MIT        |

## Supporto Hardware

La pipeline rileva l'ambiente host una sola volta all'avvio
(`scripts.hardware.get_profile()`) e configura ogni modello di conseguenza.
Esegui `python -m scripts.run_pipeline --print-hardware` per vedere il profilo
attivo.

| Profilo                          | Whisper                          | Tier LLM              | `n_threads`  | `n_gpu_layers` |
|----------------------------------|----------------------------------|-----------------------|--------------|----------------|
| Apple Silicon (serie M) ≥ 16 GB  | `large-v3-turbo` (CPU int8)†     | 14B → 32B             | core P       | -1 (Metal)     |
| Linux + NVIDIA, VRAM ≥ 8 GB      | `large-v3-turbo` (CUDA fp16)     | 8B → 72B              | fisici-1     | -1             |
| Linux + NVIDIA, VRAM < 8 GB      | `large-v3-turbo` (CUDA int8_fp16)| 8B → 72B              | fisici-1     | -1             |
| Linux + AMD ROCm                 | `large-v3-turbo` (CPU int8)‡     | per RAM               | fisici-1     | -1             |
| Linux x86_64 solo CPU            | `large-v3-turbo` / `medium`      | 8B / 14B              | fisici-1     | 0              |
| Linux ARM64 generico             | `large-v3-turbo` / `small`       | per RAM               | fisici-1     | 0              |
| Raspberry Pi 5 (8 GB)            | `small`                          | 3B (Phi-3-mini ecc.)  | 4            | 0              |
| Raspberry Pi 4 (4 GB)            | `small`                          | 1B (TinyLlama ecc.)   | 4            | 0              |
| Raspberry Pi 3 / Zero 2W (1 GB)  | `tiny`                           | none — solo regole    | 4            | 0              |
| Mac Intel                        | `large-v3-turbo` / `medium`      | per RAM               | fisici-1     | 0              |
| Windows + CUDA                   | come Linux CUDA                  | come Linux CUDA       | fisici-1     | -1             |
| Windows solo CPU                 | come Linux CPU                   | come Linux CPU        | fisici-1     | 0              |
| VM / container (qualunque sopra) | come profilo base                | come base             | base-1       | come base      |

† faster-whisper non ha un backend Metal — CPU int8 con thread sui P-core è
strettamente più veloce del path CTranslate2-MPS (che non funziona); llama.cpp
usa comunque Metal con `n_gpu_layers=-1`.
‡ faster-whisper non ha un path ROCm nativo; llama.cpp usa ROCm quando
compilato con `-DGGML_HIPBLAS`. La pipeline ricade su Whisper CPU ma scarica
l'LLM sulla GPU.

In ogni VM/container la pipeline disabilita `mlock` (di solito fallisce per
`RLIMIT_MEMLOCK`) e riserva un core per l'host. Su Apple Silicon abilita
Flash Attention + KV cache quantizzata Q8_0 (~25 % di memoria in meno) se la
build di `llama-cpp-python` collegata lo supporta.

Se la RAM rilevata è sotto soglia per l'LLM, la pipeline esegue
trasparentemente in `--skip-extract` evitando crash sull'hardware più piccolo.

## Installazione

### Pipeline (Python)

```bash
# Clona
git clone https://github.com/your-user/cibobuono.git
cd cibobuono

# Crea virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt

# Scarica un modello LLM GGUF
mkdir -p models
# Consigliati (auto-selezionati in base alla RAM in quest'ordine):
#   ≥40 GB → Qwen2.5-72B-Instruct-Q4_K_M.gguf  oppure  Llama-3.3-70B-Instruct-Q4_K_M.gguf
#   ≥24 GB → Qwen2.5-32B-Instruct-Q4_K_M.gguf
#            gemma-3-27b-it-Q4_K_M.gguf         (richiede login HF: huggingface-cli login)
#   ≥12 GB → Qwen2.5-14B-Instruct-Q4_K_M.gguf  (ottimo italiano, full CUDA offload ≤16 GB VRAM)
#   ≥6  GB → Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
#   ≥3  GB → Phi-3-mini-4k-Instruct-Q4_K_M.gguf  oppure  Qwen2.5-3B-Instruct-Q4_K_M.gguf
#   ≥1.5GB → tinyllama-1.1b-chat-v1.0-Q4_K_M.gguf  (Raspberry Pi 4)
# Metti i file in models/. La pipeline seleziona automaticamente il modello
# più grande che entra nell'hardware rilevato.

# Ispeziona il profilo hardware attivo (parametri Whisper + llama.cpp, JSON)
python -m scripts.run_pipeline --print-hardware

# Aggiungi URL dei canali
echo "https://www.youtube.com/@tuo-canale" >> channels_input.txt

# Esegui la pipeline
python -m scripts.run_pipeline --skip-push --max-videos 5

# Modalità continua (cataloga + processa i nuovi upload all'infinito)
python -m scripts.run_pipeline --watch --poll-interval 1800

# Controlla lo stato
python -m scripts.run_pipeline --status

# Esegui i test
pytest tests/ -v
```

### Sito (React)

```bash
cd site
npm install
npm run dev      # Server di sviluppo (http://localhost:5173)
npm run build    # Build di produzione → site/dist/
```

Il sito viene deployato automaticamente via GitHub Actions al push su `main`. Vai su **Settings → Pages → Source: GitHub Actions** per abilitare il deploy.

## Flusso di Deploy

```
Pipeline → data/*.json → git push → GitHub Actions → npm build → inject data/ → GitHub Pages
```

L'app React carica i `data/*.json` a runtime tramite URL relativi. Nessun dato viene duplicato — il workflow CI copia `data/` nella build output al momento del deploy.

## Strategia di Deduplicazione

La deduplicazione usa **due condizioni (AND)**:

1. **Prossimità geografica**: distanza haversine < 200 metri
2. **Similarità del nome**: ≥ 70 dopo normalizzazione noise-word

Prima del confronto, le parole generiche (categorie come "pizzeria", "forno", articoli come "il", "la", "di") vengono rimosse in modo che venga confrontato solo il nome proprio. Questo fa sì che "Sant'Isidoro pizza e bolle" venga correttamente unito a "Sant'Isidoro Pizze Boiler".

Quando viene trovato un duplicato, il nuovo nome viene aggiunto a `aliases[]`.

## Revisione Manuale

1. Apri `data/flagged_segments.json`
2. Trova i segmenti con `"reviewed_by_human": false`
3. Compila `locale_name`, `city`, `rating` dove necessario
4. Imposta `"reviewed_by_human": true`
5. Esegui: `python -m scripts.handle_flagged_segments`

## Contribuire e Segnalare Errori

Questo progetto estrae dati automaticamente da video YouTube. Gli errori sono **previsti e inevitabili** — la pipeline può:

- **Falsi positivi**: Estrarre un "locale" che in realtà non esiste o non è stato visitato nel video
- **Falsi negativi**: Non trovare un locale reale che è stato visitato
- **Nomi sbagliati**: Storpiare o confondere il nome di un locale (specialmente dalla trascrizione Whisper)
- **Coordinate sbagliate**: Geocodificare nella posizione sbagliata
- **Voti/sentiment sbagliati**: Interpretare male l'opinione del blogger

### Come Segnalare un Errore

[Apri una Issue su GitHub](../../issues/new) usando uno di questi template:

#### 🔴 Falso Positivo (il locale non dovrebbe esserci)
```
Titolo: [Falso Positivo] "Nome Locale" nel video VIDEO_ID
Corpo:
- Nome locale: ...
- Video: https://youtu.be/VIDEO_ID
- Perché è sbagliato: (il blogger non ha mai visitato questo posto / non esiste / ecc.)
```

#### 🟡 Dato Sbagliato (nome, coordinate, voto, ecc.)
```
Titolo: [Dato Sbagliato] "Nome Locale" — nome/coordinate/voto errati
Corpo:
- Dato attuale: ...
- Dato corretto: ...
- Video: https://youtu.be/VIDEO_ID
- Timestamp: MM:SS
```

#### 🟢 Locale Mancante (falso negativo)
```
Titolo: [Mancante] "Nome Locale" dal video VIDEO_ID
Corpo:
- Nome locale: ...
- Città: ...
- Video: https://youtu.be/VIDEO_ID
- Timestamp: MM:SS (dove il blogger lo visita)
```

#### 💡 Suggerimento Nuovo Canale
```
Titolo: [Canale] Suggerisco @NomeCanale
Corpo:
- URL canale: https://www.youtube.com/@...
- Perché: (canale di food review italiano)
```

Tutte le segnalazioni sono benvenute — anche un semplice "questo è sbagliato" aiuta a migliorare il dataset.

## Licenza

- **Dataset** (cartella `data/`): Proprietario — © Luca Ostinelli. Tutti i diritti riservati. Utilizzo solo con permesso esplicito. Vedi [LICENSE-DATASET](LICENSE-DATASET).
- **Codice e Sito** (`scripts/`, `site/`, `tests/`): [MIT](LICENSE-CODE) — libero di usare, modificare e distribuire.