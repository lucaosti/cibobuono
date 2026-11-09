"""
teacher_label.py — Whole-video venue extraction with a large "teacher" LLM.

Two outputs per video, both meant for offline evaluation and for building the
gold set that the production chain is measured against:

1. ``venues``: every food business the host visits on camera, extracted from
   title + description + chapters + the FULL timestamped transcript in one
   call. This is the metadata-first, whole-transcript formulation that the
   chunk-level NER chain cannot express, so comparing the two tells whether
   the architecture (not the thresholds) is the limiting factor.
2. ``candidate_labels`` (optional, from a trace JSONL produced by
   :mod:`scripts.trace_candidates`): for each NER candidate, is the span a
   venue name and is the host on site there. These are the gold labels for
   precision / recall of the current rules + arbiter.

Two backends: a local GGUF through llama.cpp (``--model MODEL.gguf``), or any
OpenAI-compatible server such as vLLM (``--server URL --model SERVED_NAME``).
With ``--frames DIR`` (``DIR/<video_id>/<seconds>.jpg``, written by
:mod:`scripts.sample_frames`) the frames are attached to the request so a
vision-language teacher can read signs, menus and on-screen captions.

CLI:
    python -m scripts.teacher_label --model MODEL.gguf --meta meta.json \
        --transcripts DIR [--trace trace.jsonl] --out labels.jsonl
    python -m scripts.teacher_label --server http://127.0.0.1:8000/v1 --model NAME \
        --meta meta.json --transcripts DIR --frames FRAMES --out labels.jsonl --max-tokens 16384
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import base64
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from scripts.chunk_transcription import seconds_to_timestamp
from scripts.utils import setup_logging

logger = setup_logging("teacher_label")

MAX_TRANSCRIPT_CHARS = 60_000
MAX_DESCRIPTION_CHARS = 3_000

_VENUE_SYSTEM = (
    "You are an expert annotator of Italian food-vlog videos. You read the video's "
    "metadata and its full timestamped transcript and list the food businesses the "
    "host physically visits on camera. Return ONLY valid JSON."
)

_VENUE_USER = (
    'TITLE: "{title}"\n'
    'SERIES / FORMAT NAME IN TITLE (never a venue name): "{series}"\n'
    'LOCATION NAMED IN TITLE (street / area, not a venue): "{location_hint}"\n'
    "CHAPTERS:\n{chapters}\n\n"
    "DESCRIPTION:\n\"\"\"\n{description}\n\"\"\"\n\n"
    "TRANSCRIPT (automatic speech recognition, Italian, proper names may be garbled):\n"
    "\"\"\"\n{transcript}\n\"\"\"\n\n"
    "Return one JSON object:\n"
    '{{"video_type": "venue_review" | "recipe" | "other",\n'
    ' "venues": [\n'
    '   {{"name": "proper business name, correctly spelled (prefer description/chapters over ASR)",\n'
    '    "address": "street address if stated anywhere, else empty",\n'
    '    "city": "city or town, else empty",\n'
    '    "category": ["pizzeria" | "ristorante" | "forno" | "bar" | "street_food" | "mercato" | "gelateria" | "pasticceria"],\n'
    '    "on_site": true | false,\n'
    '    "timestamp": "M:SS when the host arrives / starts eating there",\n'
    '    "evidence": "short verbatim quote from the transcript proving the visit",\n'
    '    "rating": "overall grade given by the host, e.g. 8, 7.5, 6++, or null",\n'
    '    "sentiment": "positive" | "neutral" | "negative",\n'
    '    "name_source": ["title" | "description" | "chapter" | "speech" | "inferred"],\n'
    '    "confidence": 0.0-1.0}}\n'
    " ]}}\n\n"
    "Rules:\n"
    "- Include ONLY businesses where the host eats, orders or enters ON CAMERA in this video.\n"
    "- Exclude venues that are merely compared, remembered, recommended or planned.\n"
    "- Exclude dishes, ingredients, people, brands, streets, neighbourhoods, cities.\n"
    "- The recurring series name of the channel (e.g. 'Pizza a taglio criminale', 'Forni criminali',\n"
    "  'Hit di Franchino') is a video format, NOT the business name; never use it as name.\n"
    "- If the venue is never named anywhere (speech, description, chapters), still list it with\n"
    '  name "" and name_source ["inferred"] so it can be recovered from the video frames later.\n'
    "- Empty list if the video is not an on-location food visit."
)

_FRAMES_NOTE = (
    "\n\nFRAMES: still frames sampled from the video follow, each preceded by its timestamp. "
    "Use shop signs, menus, receipts, price boards and on-screen captions to read business "
    "names, addresses and grades; when a name comes from a frame, add \"frame\" to name_source."
)

_CAND_SYSTEM = (
    "You label spans extracted by a named-entity model from Italian food-vlog "
    "transcripts. For each span decide whether it is the proper name of a specific "
    "food business, and whether the host is physically at that business in the "
    "excerpt. Return ONLY valid JSON."
)

_CAND_USER = (
    'VIDEO TITLE: "{title}"\n\n'
    "SPANS:\n{blocks}\n\n"
    'Return {{"labels": [{{"id": "c0", "is_venue": true|false, "is_visit": true|false, '
    '"canonical_name": "corrected business name or empty"}}]}}\n'
    "Rules: is_venue=false for dishes, people, cities, streets, brands, generic words "
    "(e.g. 'pizzeria', 'Dio', 'Instagram'). is_visit=true only if the excerpt shows the host "
    "eating/ordering/being at that business now; comparisons, memories and plans are false."
)


def _load_model(model_path: str, n_ctx: int, n_gpu_layers: int):
    from llama_cpp import Llama

    kwargs: dict = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": 2048,
        "verbose": False,
    }
    try:
        from llama_cpp import GGML_TYPE_Q8_0

        return Llama(**kwargs, flash_attn=True, type_k=GGML_TYPE_Q8_0, type_v=GGML_TYPE_Q8_0)
    except (ImportError, TypeError):
        return Llama(**kwargs)


class OpenAIChat:
    """Client for an OpenAI-compatible server with llama.cpp's ``create_chat_completion`` shape."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.0, timeout: float = 1800):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def create_chat_completion(self, *, messages, max_tokens, temperature, response_format) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "response_format": response_format,
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


def frame_parts(frame_dir: Path) -> list[dict]:
    """Timestamp + image content parts for every ``<seconds>.jpg`` in *frame_dir*, in time order."""
    parts: list[dict] = []
    for path in sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem)):
        data = base64.b64encode(path.read_bytes()).decode()
        parts.append({"type": "text", "text": f"[{seconds_to_timestamp(int(path.stem))}]"})
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return parts


def _chat_json(llm, system: str, user: str, max_tokens: int, images: list[dict] | None = None) -> dict:
    content = [{"type": "text", "text": user + _FRAMES_NOTE}, *images] if images else user
    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = (response["choices"][0]["message"]["content"] or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start : end + 1]) if start >= 0 and end > start else {}
    return data if isinstance(data, dict) else {}


def _timestamped(transcript: dict, max_chars: int) -> str:
    lines = []
    for seg in transcript.get("segments") or []:
        t = (seg.get("text") or "").strip()
        if t:
            lines.append(f"[{seconds_to_timestamp(float(seg.get('start', 0)))}] {t}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... transcript truncated ...]"
    return text


def _chapter_lines(chapters: list[dict] | None) -> str:
    if not chapters:
        return "(none)"
    out = []
    for ch in chapters:
        if isinstance(ch, dict) and ch.get("title"):
            out.append(f"- [{seconds_to_timestamp(float(ch.get('start_time') or 0))}] {ch['title']}")
    return "\n".join(out) if out else "(none)"


def label_video(
    llm, meta: dict, transcript: dict, *, max_tokens: int = 1800, images: list[dict] | None = None
) -> dict | None:
    """Venue labels for one video, or None when the reply is empty or not JSON.

    A thinking model that spends the whole budget reasoning returns no content;
    that is a failed call, not a video without venues.
    """
    from scripts.video_intelligence import analyze_title

    intel = analyze_title(meta.get("title", ""))
    user = _VENUE_USER.format(
        title=(meta.get("title") or "").replace('"', "'"),
        series=(intel.series_name or "").replace('"', "'"),
        location_hint=(intel.location_hint or intel.city or "").replace('"', "'"),
        chapters=_chapter_lines(meta.get("chapters")),
        description=(meta.get("description") or "")[:MAX_DESCRIPTION_CHARS],
        transcript=_timestamped(transcript, MAX_TRANSCRIPT_CHARS),
    )
    data = _chat_json(llm, _VENUE_SYSTEM, user, max_tokens, images)
    if not data:
        return None
    venues = [v for v in (data.get("venues") or []) if isinstance(v, dict)]
    return {"video_type": data.get("video_type", ""), "venues": venues}


def label_candidates(llm, title: str, rows: list[dict], *, batch_size: int = 12) -> list[dict]:
    out: list[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        blocks = []
        for i, r in enumerate(batch):
            win = (r.get("window") or "")[:700].replace('"', "'")
            blocks.append(f'[c{i}] SPAN: "{r["name"]}" (t={r.get("start_time")})\nEXCERPT: "{win}"\n')
        data = _chat_json(
            llm,
            _CAND_SYSTEM,
            _CAND_USER.format(title=title.replace('"', "'"), blocks="\n".join(blocks)),
            max_tokens=60 * len(batch) + 100,
        )
        by_id = {str(l.get("id")): l for l in (data.get("labels") or []) if isinstance(l, dict)}
        for i, r in enumerate(batch):
            lab = by_id.get(f"c{i}", {})
            out.append(
                {
                    "name": r["name"],
                    "start_time": r.get("start_time"),
                    "chunk_index": r.get("chunk_index"),
                    "is_venue": bool(lab.get("is_venue")),
                    "is_visit": bool(lab.get("is_visit")),
                    "canonical_name": str(lab.get("canonical_name") or ""),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Teacher-LLM venue labels for evaluation")
    parser.add_argument("--model", required=True, help="GGUF path (first shard), or the served model name with --server")
    parser.add_argument("--server", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--frames", type=Path, default=None, help="directory with <video_id>/<seconds>.jpg")
    parser.add_argument("--max-tokens", type=int, default=1800, help="output budget (raise for thinking models)")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (server backend)")
    parser.add_argument("--meta", required=True, type=Path, help="JSON list of {video_id,title,description,chapters}")
    parser.add_argument("--transcripts", required=True, type=Path, help="directory with <video_id>_transcript.json")
    parser.add_argument("--trace", type=Path, default=None, help="trace JSONL from scripts.trace_candidates")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-ctx", type=int, default=32768)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    meta_rows = json.loads(args.meta.read_text(encoding="utf-8"))
    if args.limit:
        meta_rows = meta_rows[: args.limit]

    trace_by_video: dict[str, list[dict]] = defaultdict(list)
    if args.trace and args.trace.exists():
        with open(args.trace, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("valid_name") and row.get("looks_like_venue"):
                    trace_by_video[row["video_id"]].append(row)

    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(l)["video_id"] for l in f if l.strip()}

    if args.server:
        llm = OpenAIChat(args.server, args.model, temperature=args.temperature)
        model_name = args.model
    else:
        llm = _load_model(args.model, args.n_ctx, args.n_gpu_layers)
        model_name = Path(args.model).name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as out:
        for meta in meta_rows:
            vid = meta["video_id"]
            if vid in done:
                continue
            tpath = args.transcripts / f"{vid}_transcript.json"
            if not tpath.exists():
                logger.warning(f"{vid}: transcript missing, skipped")
                continue
            transcript = json.loads(tpath.read_text(encoding="utf-8"))
            t0 = time.time()
            images = None
            if args.frames and (args.frames / vid).is_dir():
                images = frame_parts(args.frames / vid)
            result = label_video(llm, meta, transcript, max_tokens=args.max_tokens, images=images)
            if result is None:
                logger.warning(f"{vid}: empty reply, retrying with {2 * args.max_tokens} tokens")
                result = label_video(llm, meta, transcript, max_tokens=2 * args.max_tokens, images=images)
            if result is None:
                # Not written, so a resumed run labels it again.
                logger.error(f"{vid}: empty reply twice, skipped")
                continue
            cand_rows = trace_by_video.get(vid, [])
            labels = label_candidates(llm, meta.get("title", ""), cand_rows) if cand_rows else []
            record = {
                "video_id": vid,
                "title": meta.get("title", ""),
                "model": model_name,
                "frames": len(images or []) // 2,
                "video_type": result["video_type"],
                "venues": result["venues"],
                "candidate_labels": labels,
                "seconds": round(time.time() - t0, 1),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            logger.info(
                f"{vid}: {len(result['venues'])} venues, {len(labels)} candidate labels, "
                f"{record['seconds']}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
