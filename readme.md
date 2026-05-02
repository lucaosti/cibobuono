# CiboBuono — YouTube Food Locale Reviews Dataset

**[🇮🇹 Leggi in italiano](#-versione-italiana)**

## What is this?

An open-source dataset and interactive map of food locales (bakeries, pizzerias, restaurants, etc.) reviewed by Italian YouTube food bloggers. All data is extracted automatically via a **fully local pipeline** using open-source tools only — no paid APIs.

The pipeline uses **inference only** (no model training or fine-tuning): a local ASR model, a **GLiNER** NER model for venue spans, deterministic rules, and a local instruction-tuned LLM (GGUF) for verification and visit details. **Python 3.10+** is required.

**Language:** all catalogued videos are **Italian**; yt-dlp, subtitle download, Whisper, Nominatim result language, and LLM prompts use Italian (`CONTENT_LANGUAGE` in `scripts/utils.py`).

The pipeline transcribes YouTube videos, proposes venue **candidates** with a local multilingual NER (GLiNER), applies **deterministic Italian visit-vs-mention rules**, and calls the local LLM only as a binary **verifier** (with cited evidence) when rules are ambiguous; ratings and sentiment are filled by the LLM only on accepted visits over a short transcript window. Results are geocoded via OpenStreetMap and written as normalized JSON on GitHub. The React site deploys automatically via GitHub Pages on every push.

## Live Map

The interactive map is deployed on **GitHub Pages** and updates automatically on every `git push` to `main`. It allows you to explore all reviewed locales with search and sentiment filters.

> **Local preview**: `cd site && npm install && npm run dev`

> **Python venv (pipeline)**: from repo root, `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then run `python -m scripts.run_pipeline ...`. Optional: `CIBOBUONO_LLM_MODEL` → a `.gguf` under `models/` or elsewhere; `CIBOBUONO_NER_MODEL` overrides the Hugging Face id for GLiNER (default `urchade/gliner_multi-v2.1`). **PyTorch** is required for NER (CPU or MPS on Apple Silicon); if GLiNER cannot load, a lightweight heuristic fallback still runs.

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
│   └── skipped_videos.json         # Skipped videos (recipes, Shorts)
│
├── scripts/                        # Pipeline modules (Python)
│   ├── schemas.py                  # Pydantic models & validation
│   ├── utils.py                    # Shared utilities, paths, config, hardware detection
│   ├── fetch_channels.py           # Extract channel metadata via yt-dlp
│   ├── fetch_videos.py             # Catalog videos, detect recipes/Shorts, download audio
│   ├── transcribe_video.py         # Transcription: YouTube subs first, Whisper fallback
│   ├── chunk_transcription.py      # Split transcripts into 90s chunks with 15s overlap
│   ├── extract_locales.py          # Shared helpers: food gate, hints, timestamps, LLM handle
│   ├── ner_candidates.py           # GLiNER venue candidates (+ heuristic fallback)
│   ├── visit_classifier.py         # Italian rules + LLM yes/no for ambiguous cases
│   ├── extract_pipeline.py         # Orchestrator: NER → classify → detail LLM → cross-chunk filter
│   ├── geocode_locales.py          # Nominatim geocoding (free, rate-limited)
│   ├── verify_locales.py           # OSM verification via Overpass API (anti-false-positive)
│   ├── deduplicate_locales.py      # Fuzzy name + haversine distance deduplication
│   ├── populate_json.py            # Write visits, flagged segments
│   ├── handle_flagged_segments.py  # Import manually reviewed segments
│   ├── dashboard.py                # Rich live terminal dashboard
│   ├── push_to_github.py           # Git commit & push
│   ├── validate_data.py            # Validate data/*.json (Pydantic; CI / local)
│   └── run_pipeline.py             # Main pipeline orchestrator (2-phase)
│
├── site/                           # React + Vite + TypeScript (GitHub Pages)
│   ├── package.json                # Node dependencies
│   ├── vite.config.ts              # Vite config (base: /cibobuono/)
│   ├── src/
│   │   ├── App.tsx                 # Main app: data loading, filtering, layout
│   │   ├── api.ts                  # Fetch JSON from data/ at runtime
│   │   ├── types.ts                # TypeScript types (mirrors Python schemas)
│   │   └── components/
│   │       ├── MapView.tsx         # Leaflet map with sentiment-colored markers
│   │       ├── LocaleList.tsx      # Sidebar list with expandable visit details
│   │       ├── Header.tsx          # Search + sentiment filter bar
│   │       └── StatusBar.tsx       # Loading/error states
│   └── ...
│
├── tests/                          # Pytest test suite
│   ├── test_schemas.py
│   ├── test_dedup.py
│   ├── test_chunks.py
│   ├── test_extraction.py
│   ├── test_ner_candidates.py
│   ├── test_visit_classifier.py
│   ├── test_extract_pipeline.py
│   ├── test_verify.py
│   ├── test_utils.py
│   └── test_data_integrity.py
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
| 4  | Transcribe           | YouTube subs / Whisper `medium` | YouTube subtitles first, Whisper fallback       |
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

### Pipeline Options

```
python -m scripts.run_pipeline --help

Options:
  --skip-fetch         Skip cataloging new videos (process existing pending only)
  --skip-transcribe    Use cached transcripts
  --skip-extract       Skip LLM extraction
  --skip-push          Don't commit/push to GitHub
  --whisper-model      tiny|base|small|medium|large (default: medium)
  --max-videos N       Max pending videos per run (default: 100); 0 = all pending
  --no-dashboard       Disable live terminal dashboard (log-only mode)
  --reset              Reset pipeline data JSON, cache, and logs (keeps corrections unless --reset-all-data)
  --reset-all-data     With --reset, also clear corrections.json
  --repair-stale-state Mark pending videos that already have visits/flagged as processed, then exit
  --repair-dry-run     With --repair-stale-state, print only (no writes)
  --status             Show pipeline status summary and exit
```

**Interrupts (Ctrl+C / SIGTERM)** — First signal: finish the current video, then stop (coherent between videos). Second signal: quit immediately (run `python -m scripts.run_pipeline --repair-stale-state` if a video was left pending after visits were already saved). JSON files are saved via atomic replace to reduce corruption if the process is killed during a write.

**Full catalog re-run example** (after reset, process every pending video, no git push):

```bash
python -m scripts.run_pipeline --reset --skip-push --max-videos 0 --no-dashboard
```

### Key Features

- **Newest-first processing**: Videos are processed newest-first because recent YouTube ASR models produce significantly better subtitles for Italian proper nouns.
- **YouTube subtitles first**: Tries to download YouTube's own subtitles (auto-generated or manual) before falling back to local Whisper. YouTube ASR is far more accurate for Italian proper nouns.
- **Whisper with `initial_prompt`**: When YouTube subs are unavailable, Whisper `medium` runs with an Italian food terminology prompt to bias transcription toward restaurant names.
- **Video descriptions as context**: Descriptions supply regex-extracted **venue hints** (names/links); hints can protect a candidate so a single-chunk mention still counts as a catalogued visit when rules agree it was a visit.
- **OSM real-place verification**: After geocoding, each locale is verified against OpenStreetMap via the Overpass API (500m radius, fuzzy name match ≥ 80) — if no matching food establishment exists near the coordinates, the extraction is rejected. This is the strongest anti-false-positive measure.
- **Sliding window**: Audio files are pre-downloaded in a window of 20. As each video is processed, the oldest is cleaned up and the next is fetched.
- **Neuro-symbolic extraction**: GLiNER proposes spans; Italian regex/heuristics decide many cases; the LLM answers visit yes/no with a quoted evidence span only when rules are unsure (no monolithic “extract everything” prompt).
- **Non-food video filtering**: Videos with non-food keywords in the title (boxing, gaming, fitness, etc.) are automatically skipped.
- **Hardware auto-detection**: CPU cores, Apple Silicon / Metal GPU, unified memory are detected at startup. LLM threads, batch size, GPU layers, and mlock are configured dynamically.
- **Shorts filtering**: YouTube Shorts (URL `/shorts/` or duration ≤60s) are automatically skipped.
- **Recipe filtering**: Videos with recipe keywords in the title are automatically skipped.
- **Global model caching**: Whisper, NER, and LLM models are loaded once per session (NER via `transformers`/`gliner`).

## Tech Stack (100% Open Source, No Paid APIs)

| Component      | Tool                             | License    |
|----------------|----------------------------------|------------|
| Video download | yt-dlp                           | Unlicense  |
| Transcription  | YouTube subs (yt-dlp) + Whisper (fallback) | MIT |
| Venue NER      | GLiNER + PyTorch + Hugging Face `transformers` | Apache-2.0 / BSD |
| LLM inference  | llama-cpp-python + GGUF (e.g. Llama 3.1 8B) | MIT     |
| Geocoding      | Nominatim (OpenStreetMap)        | ODbL       |
| Place verification | Overpass API (OpenStreetMap)  | ODbL       |
| Deduplication  | thefuzz (fuzzy matching)         | MIT        |
| Validation     | Pydantic                         | MIT        |
| Frontend       | React + TypeScript + Vite        | MIT        |
| Map            | Leaflet + react-leaflet + OSM    | BSD-2/ODbL |
| CI/CD          | GitHub Actions + GitHub Pages    | —          |
| Testing        | pytest                           | MIT        |

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
# Recommended: Mistral 7B Instruct v0.2 Q4_K_M
# Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# Place: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Add channel URLs
echo "https://www.youtube.com/@your-channel" >> channels_input.txt

# Run the pipeline
python -m scripts.run_pipeline --skip-push --max-videos 5

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

# 🇮🇹 Versione Italiana

## Cos'è?

Un dataset open source e una mappa interattiva dei locali (forni, panifici, ristoranti, pizzerie, ecc.) recensiti da food YouTuber italiani. Tutti i dati vengono estratti automaticamente con una **pipeline completamente locale** usando solo strumenti open source — nessuna API a pagamento.

La pipeline usa **solo inferenza** (nessun training o fine-tuning): ASR locale, modello **GLiNER** per gli span dei locali, regole deterministiche e LLM istruito locale (GGUF) per verifica e dettagli sulle visite. È richiesto **Python 3.10+**.

**Lingua:** tutti i video catalogati sono in **italiano**; yt-dlp, sottotitoli, Whisper, lingua dei risultati Nominatim e prompt LLM sono allineati all’italiano (`CONTENT_LANGUAGE` in `scripts/utils.py`).

La pipeline trascrive i video YouTube, propone **candidati locale** con NER multilingue locale (GLiNER), applica **regole deterministiche** (italiano) visita vs. semplice menzione e interroga l’LLM locale solo come **verificatore** binario (con evidenza citata) quando le regole sono ambigue; voti e sentiment sono richiesti all’LLM solo per le visite accettate su una finestra breve di trascrizione. Geocoding OSM e JSON normalizzati come sopra; deploy del sito React su GitHub Pages ad ogni push.

## Mappa Interattiva

La mappa è deployata su **GitHub Pages** e si aggiorna automaticamente ad ogni `git push` su `main`. Permette di esplorare tutti i locali recensiti con ricerca e filtri per sentiment.

> **Anteprima locale**: `cd site && npm install && npm run dev`

> **Venv Python (pipeline)**: dalla root, `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, poi `python -m scripts.run_pipeline ...`. Opzionale: `CIBOBUONO_LLM_MODEL` → `.gguf`; `CIBOBUONO_NER_MODEL` sovrascrive l’id Hugging Face di GLiNER (default `urchade/gliner_multi-v2.1`). **PyTorch** serve per il NER (CPU o MPS su Apple Silicon); se GLiNER non si carica, resta un fallback euristico leggero.

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
│   └── skipped_videos.json         # Video saltati (ricette, Shorts)
│
├── scripts/                        # Moduli della pipeline (Python)
│   ├── schemas.py                  # Modelli Pydantic e validazione
│   ├── utils.py                    # Utilità condivise, percorsi, config, hardware detection
│   ├── fetch_channels.py           # Estrazione metadati canali via yt-dlp
│   ├── fetch_videos.py             # Catalogo video, detection ricette/Shorts, download audio
│   ├── transcribe_video.py         # Trascrizione: sottotitoli YouTube prima, Whisper fallback
│   ├── chunk_transcription.py      # Divisione trascrizioni in chunk da 90s con 15s overlap
│   ├── extract_locales.py          # Helper condivisi: gate food, hints, timestamp, handle LLM
│   ├── ner_candidates.py           # Candidati GLiNER (+ fallback euristico)
│   ├── visit_classifier.py         # Regole italiane + LLM sì/no se ambiguo
│   ├── extract_pipeline.py         # Orchestratore: NER → classificazione → LLM dettagli → filtro cross-chunk
│   ├── geocode_locales.py          # Geocoding Nominatim (gratuito, rate-limited)
│   ├── verify_locales.py           # Verifica OSM via Overpass API (anti-falsi positivi)
│   ├── deduplicate_locales.py      # Deduplicazione fuzzy nome + distanza haversine
│   ├── populate_json.py            # Scrittura visite e segmenti flaggati
│   ├── handle_flagged_segments.py  # Importazione segmenti revisionati manualmente
│   ├── dashboard.py                # Dashboard live nel terminale (Rich)
│   ├── push_to_github.py           # Commit e push su Git
│   ├── validate_data.py            # Validazione data/*.json vs Pydantic (CI / locale)
│   └── run_pipeline.py             # Orchestratore principale della pipeline (2 fasi)
│
├── site/                           # React + Vite + TypeScript (GitHub Pages)
│   ├── package.json                # Dipendenze Node
│   ├── vite.config.ts              # Configurazione Vite (base: /cibobuono/)
│   ├── src/
│   │   ├── App.tsx                 # App principale: caricamento dati, filtri, layout
│   │   ├── api.ts                  # Fetch JSON da data/ a runtime
│   │   ├── types.ts                # Tipi TypeScript (mirror degli schema Python)
│   │   └── components/
│   │       ├── MapView.tsx         # Mappa Leaflet con marker colorati per sentiment
│   │       ├── LocaleList.tsx      # Lista nella sidebar con dettagli visite espandibili
│   │       ├── Header.tsx          # Barra di ricerca + filtro sentiment
│   │       └── StatusBar.tsx       # Stati di caricamento/errore
│   └── ...
│
├── tests/                          # Suite di test (pytest)
│   ├── test_schemas.py
│   ├── test_dedup.py
│   ├── test_chunks.py
│   ├── test_extraction.py
│   ├── test_ner_candidates.py
│   ├── test_visit_classifier.py
│   ├── test_extract_pipeline.py
│   ├── test_verify.py
│   ├── test_utils.py
│   └── test_data_integrity.py
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
| 4  | Trascrizione         | Sottotitoli YT / Whisper `medium` | Sottotitoli YouTube prima, Whisper fallback     |
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

### Opzioni della Pipeline

```
python -m scripts.run_pipeline --help

Opzioni:
  --skip-fetch         Salta il catalogo dei nuovi video (lavora solo sui pending)
  --skip-transcribe    Usa trascrizioni in cache
  --skip-extract       Salta l'estrazione LLM
  --skip-push          Non fare commit/push su GitHub
  --whisper-model      tiny|base|small|medium|large (default: medium)
  --max-videos N       Max video pending per run (default: 100); 0 = tutti i pending
  --no-dashboard       Disabilita la dashboard live nel terminale
  --reset              Resetta i JSON della pipeline, cache e log (mantiene corrections salvo --reset-all-data)
  --reset-all-data     Con --reset, azzera anche corrections.json
  --repair-stale-state Allinea videos.json/processed_videos.json se un video è pending ma ha già visite/flagged, poi esce
  --repair-dry-run     Con --repair-stale-state, solo stampa (nessuna scrittura)
  --status             Mostra il riepilogo dello stato e termina
```

**Interruzioni (Ctrl+C / SIGTERM)** — Primo segnale: termina il video corrente, poi si ferma (stato coerente tra un video e l’altro). Secondo segnale: uscita immediata (se serve, `python -m scripts.run_pipeline --repair-stale-state`). I JSON vengono scritti con sostituzione atomica per ridurre file corrotti se il processo muore durante il salvataggio.

**Esempio run completo sul catalogo** (dopo reset, tutti i pending, senza push):

```bash
python -m scripts.run_pipeline --reset --skip-push --max-videos 0 --no-dashboard
```

### Funzionalità Chiave

- **Processamento dal più recente**: I video vengono processati dal più recente perché i modelli ASR di YouTube recenti producono sottotitoli significativamente migliori per i nomi propri italiani.
- **Sottotitoli YouTube prima**: Prova a scaricare i sottotitoli YouTube (auto-generati o manuali) prima di ricorrere a Whisper locale. L'ASR di YouTube è molto più accurato per i nomi propri italiani.
- **Whisper con `initial_prompt`**: Quando i sottotitoli YouTube non sono disponibili, Whisper `medium` usa un prompt con terminologia food italiana per migliorare il riconoscimento dei nomi dei locali.
- **Descrizioni video come contesto**: Dalla descrizione si estraggono via regex gli **hint** sui nomi dei locali; un hint può proteggere un candidato così che una sola menzione in un chunk resti una visita catalogata quando le regole confermano la visita.
- **Verifica OSM dei locali reali**: Dopo il geocoding, ogni locale viene verificato su OpenStreetMap tramite Overpass API (raggio 500m, fuzzy name match ≥ 80) — se nessun locale di ristorazione corrispondente esiste vicino alle coordinate, l'estrazione viene rifiutata. È la misura anti-falsi-positivi più potente.
- **Finestra mobile**: I file audio vengono pre-scaricati in una finestra di 20. Man mano che un video viene processato, il più vecchio viene cancellato e il prossimo viene scaricato.
- **Estrazione neuro-simbolica**: GLiNER propone gli span; regex/euristiche italiane decidono molti casi; l’LLM risponde sì/no visita con uno span di evidenza citato solo se le regole sono incerte (niente prompt monolitico “estrai tutto”).
- **Filtro video non-food**: I video con keyword non-food nel titolo (boxing, gaming, fitness, ecc.) vengono automaticamente saltati.
- **Hardware auto-detection**: CPU cores, Apple Silicon / Metal GPU, memoria unificata vengono rilevati allo startup. Thread, batch size, GPU layers e mlock dell'LLM vengono configurati dinamicamente.
- **Filtro Shorts**: Gli YouTube Shorts (URL `/shorts/` o durata ≤60s) vengono automaticamente saltati.
- **Filtro ricette**: I video con parole chiave di ricette nel titolo vengono automaticamente saltati.
- **Caching globale dei modelli**: Whisper, NER e LLM vengono caricati una sola volta per sessione (NER via `transformers`/`gliner`).

## Stack Tecnologico (100% Open Source, Nessuna API a Pagamento)

| Componente     | Strumento                        | Licenza    |
|----------------|----------------------------------|------------|
| Download video | yt-dlp                           | Unlicense  |
| Trascrizione   | Sottotitoli YouTube (yt-dlp) + Whisper (fallback) | MIT |
| NER locali     | GLiNER + PyTorch + Hugging Face `transformers` | Apache-2.0 / BSD |
| Inferenza LLM  | llama-cpp-python + GGUF (es. Llama 3.1 8B) | MIT     |
| Geocoding      | Nominatim (OpenStreetMap)        | ODbL       |
| Verifica locali | Overpass API (OpenStreetMap)     | ODbL       |
| Deduplicazione | thefuzz (matching fuzzy)         | MIT        |
| Validazione    | Pydantic                         | MIT        |
| Frontend       | React + TypeScript + Vite        | MIT        |
| Mappa          | Leaflet + react-leaflet + OSM    | BSD-2/ODbL |
| CI/CD          | GitHub Actions + GitHub Pages    | —          |
| Testing        | pytest                           | MIT        |

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
# Consigliato: Mistral 7B Instruct v0.2 Q4_K_M
# Scarica da: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# Posiziona: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Aggiungi URL dei canali
echo "https://www.youtube.com/@tuo-canale" >> channels_input.txt

# Esegui la pipeline
python -m scripts.run_pipeline --skip-push --max-videos 5

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