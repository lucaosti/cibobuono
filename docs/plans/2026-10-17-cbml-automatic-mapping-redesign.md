# CBML: fully automatic venue mapping — findings and redesign plan

Date: 2026-10-17. Status: proposal, measured on a 16-video sample.

Goal (owner's words): a pipeline that takes every video and maps it; human
input only as an optional extra. Training runs on the SLURM cluster, inference
on the Jetson; both hosts use the neutral codename `CBML` for every path, venv
and job name.

## 1. What was measured

Starting point: `data/*.json` are empty, `data/eval_set.json` is empty, no
model and no transcript existed locally. The only real output ever produced is
in the reverted commit `83eb93c` (3 videos processed, 1 visit published).

Evidence collected today:

| Source | Result |
|---|---|
| Reverted run (3 videos) | The single published visit, "Panificio Fiore", is a comparison ("il 10 lo lasciamo a Panificio Fiore a Bari") in a video filmed in Rome: false positive. "VIA SANNIO" (a street) was promoted from the title as a venue; the real shop was never found: false negative. |
| Catalog skip filters (2142 videos) | 1278 Shorts, 56 non-food, 25 recipe skips. Several recipe skips are on-location episodes ("Cucino con uno chef in un ristorante del centro"). |
| 16 real videos, transcribed (mlx-whisper, ~20x realtime) and traced per candidate (`scripts/trace_candidates.py`) | GLiNER produced 115 venue candidates over ~6 hours of video. Survivors of the name gate are mostly noise ("Dio", "Instagram", "Christopher Nolan", "Bosnia", "Comicon"). |
| Same 16 videos labelled by Qwen2.5-72B-Instruct (job `CBML-label` on an H100, 15-140 s per video) from title + description + chapters + full transcript | 32 on-site venues, 28 of them named, with street addresses for most Rome episodes (e.g. "Pizza e fritti magno, Via Sannio 41"; "Tiella, Piazza Regina Margherita 26"; "Clove, Via dei Gracchi 229"). |
| Current rules vs teacher, candidate level | precision 0.36, recall 0.19. The NER chain intercepts 12 of the 32 venues. |
| Where venue names live | speech 16, chapter 9, description 6, title 1; 4 venues are never named anywhere (only on screen). |
| Qwen2.5-32B-Instruct Q4_K_M on the same 16 videos (production tier) | 30 venues, 28 named, same addresses as the 72B, 9-79 s per video (about 2x faster). Weaker on names when the title names a place: "Regina Margherita" (the piazza) instead of "Tiella"; once returned a description ("Ristorante non nominato in Chinatown") instead of an empty name. Candidate labels: rules precision 0.50, recall 0.27 against it. |

## 2. Root causes, ranked by impact

1. **Architecture**: chunk-level NER on ASR text cannot see the venue. Hosts
   say "siamo qui a Piazza Regina Margherita", "questa pizzeria", "la pizzeria
   dei Pucciotti"; the proper name is spoken rarely and garbled by ASR
   ("Capani" for "Kapani", "Ernerio" for "Irnerio"). Names and addresses are
   in the description, the chapters and on screen.
2. **Blind promotion of metadata hints**: title fragments and chapter titles
   became venues with 0.80-0.85 confidence and, for chapters, bypassed OSM
   verification. Fixed today with transcript corroboration and a stricter
   chapter filter.
3. **Rules too permissive, LLM too permissive**: a regex hit within 80
   characters of any capitalized span published a visit; the LLM arbiter
   accepted comparisons. Rule hits without an LLM are now flagged, never
   published.
4. **Geocoding without an anchor**: "<name>, Italy" resolved to a homonym
   anywhere in the country and OSM then "verified" the same object. Fixed:
   no query without city or address.
5. **OSM presence as a hard publish gate**: small Italian venues are often
   missing from OSM, so correct extractions were flagged forever.
6. **Catalog filters** skipping on-location episodes with recipe words. Fixed.

Items 2, 3 (rules side), 4 and 6 are fixed in this branch with tests
(477 tests pass). Items 1, 3 (LLM side) and 5 need the redesign below.

## 3. Target architecture

One principle: **extract per video, not per chunk, from every source at
once; use the strongest model that the inference host can run; publish on
evidence, review only the residue.**

```
catalog (yt-dlp) ─► ASR (Whisper large-v3-turbo) ─► VIDEO UNDERSTANDING (one LLM call)
                                                   │  title + description + chapters + full transcript
                                                   │  → venues[name, address, city, on_site, t, rating, sentiment, evidence]
                                                   ▼
                       VISUAL RECOVERY (VLM on frames near t) ─► name/address from signs, overlays, rating card
                                                   ▼
                       GEO RESOLUTION: address+city → name+city; never name alone; OSM match = bonus
                                                   ▼
                       PUBLISH (confidence ≥ τ and city-coherent) │ RESIDUE → flagged (optional human pass)
```

### Stage by stage

- **Catalog / ASR**: unchanged tools. Whisper runs on Metal (Mac) or CUDA
  (Jetson). Keep the Italian initial prompt; add the venue names found in
  description and chapters to it per video so ASR spells them correctly.
- **Video understanding** (replaces `ner_candidates`, `visit_classifier`,
  `batch_visit_llm`, `venue_discovery`, cross-chunk consensus, hint
  promotion): the prompt in `scripts/teacher_label.py` is the prototype.
  Production model on the Jetson: Qwen2.5-32B-Instruct Q4_K_M (tier already
  recommended by the repo); 72B is the teacher for labelling and offline
  evaluation. Decision pending on the 32B-vs-72B comparison on the same
  sample.
- **Visual recovery** (new; the owner asked for a visual model): for every
  venue with empty name, missing address or confidence below τ, sample
  frames around its timestamp (the Perceptor already does sampling and
  perceptual-hash dedup) and ask a VLM (Qwen2.5-VL-7B on the Jetson,
  Qwen2-VL-2B on the Mac) for sign text, overlay text and the rating card.
  Also a cheap on-site confirmation for every venue (the sign is visible).
- **Geo resolution**: address-first. Nominatim with `address, city`, then
  `name, city`; Overpass match raises confidence instead of gating. City
  coherence against the video's city or the title's location hint stays.
- **Publication**: publish when the model confidence ≥ τ (τ chosen on the
  gold set for ≥ 0.95 precision) and the venue is geocoded coherently.
  Everything else goes to `flagged_segments.json` with a reason; a human may
  look at it, nothing waits for them.
- **Removed**: the Rich terminal dashboard (`dashboard.py`) and its control
  loop, the Perceptor audio diarization and voice registry (no measurable
  value for this goal; frame sampling is kept), `vote_aggregator.py`,
  `calibrate_confidence.py` (re-added later if corrections accumulate),
  GLiNER and its heuristic fallback. `corrections.json`, `manual_edits.py`,
  `review_queue.py` survive as optional CLI overrides.

### Learning on the cluster (`CBML-*` jobs)

- `CBML-label`: the 72B labels the whole catalogue (~800 non-Short videos
  after relaxed filters) → `venues` gold set plus candidate labels. About
  40 s per video on one H100: roughly 9 GPU-hours.
- `CBML-eval`: venue-level precision / recall of any candidate production
  model (32B, 14B, fine-tuned 7B) against the 72B labels, plus a human
  spot-check of ~50 videos to measure the teacher itself.
- `CBML-train`: LoRA fine-tune of Qwen2.5-7B/14B on the 72B labels (inputs:
  metadata + transcript; output: the venues JSON) so the Jetson runs a
  small, fast specialist instead of the 32B. Only worth it if 32B is too
  slow for the backlog; decided from `CBML-eval` numbers.

### Hosts

| Host | Role | State today |
|---|---|---|
| Mac | development, tests, ASR for samples | venv ready, 16 transcripts cached |
| `thor` (`~/CBML`) | all inference (ASR, LLM 32B, VLM) | venv with GPU torch 2.14 cu130, llama.cpp CUDA (compute 11.0), GLiNER, Whisper; no model weights yet |
| `libra` (`~/CBML`) | labelling, evaluation, training via `sbatch` | venv, llama.cpp cu124 wheel (needs gcc 14 libstdc++ at runtime), Qwen2.5 72B/32B/14B GGUF staged; `jobs/cbml_label.sbatch` |

## 4. Work plan

1. **Decide** (owner): production LLM tier on the Jetson; drop the dashboard;
   publication threshold policy; branch for this work.
2. **Pipeline core** (Mac, then Thor): new `scripts/video_understanding.py`
   built from `teacher_label.label_video`; wire it into `run_pipeline.py` in
   place of `extract_from_video`; address-first geocoding; OSM as bonus;
   delete the modules listed above with their tests; rewrite
   `scripts/eval_pipeline.py` for venue-level metrics against a gold JSONL.
3. **Visual recovery** (Thor): frame sampling around venue timestamps +
   Qwen2.5-VL prompt for sign/overlay/rating-card text; merge into the venue
   record; measure how many of the "inferred" venues it names.
4. **Scale the gold set** (Thor for ASR, Libra for `CBML-label`): 200 videos
   first, then the catalogue.
5. **Evaluate and pick the model** (`CBML-eval`), optionally `CBML-train`.
6. **Run the backlog on Thor**, publish via the existing GitHub Pages flow.

## 5. Open questions for the owner

- Keep the 32B on the Jetson or run the 72B there as well (slower, better
  names on this sample; the Jetson has 122 GB unified memory)?
- Delete the dashboard and the audio Perceptor outright, or keep them behind
  a flag for one release?
- Should the optional human pass live in the repo at all (`flagged_segments.json`
  plus the CLI), or only as a JSON people can edit?
