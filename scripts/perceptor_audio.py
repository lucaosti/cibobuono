"""
perceptor_audio.py — Audio perception: VAD, speaker diarization, voice registry.

All models run on CPU via sherpa-onnx (ONNX runtime), on every hardware tier:

- Voice Activity Detection: Silero VAD (``models/perceptor/silero_vad.onnx``).
  Chosen over webrtcvad, which classifies digital silence and low-level noise
  as speech at every aggressiveness level.
- Speaker embeddings: NeMo TitaNet-small (192-dim,
  ``models/perceptor/nemo_en_titanet_small.onnx``). English-trained but
  language-agnostic for speaker *identity*; the match threshold below may need
  tuning on Italian voices.

Diarization is a greedy cosine clustering over per-segment embeddings — no
external clustering dependency. Recurring voices are tracked per channel in
``data/voices.json`` (a JSON array — :func:`scripts.utils.load_json` coerces
any non-list payload to ``[]``) with running-mean centroids so the host of a
channel is recognized across videos.

Heavy imports (sherpa_onnx, numpy, av) are function-local so importing this
module never fails when Perceptor dependencies are not installed.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import threading
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils import DATA_DIR, MODELS_DIR, load_json, save_json, setup_logging

logger = setup_logging("perceptor_audio")

PERCEPTOR_MODELS_DIR = MODELS_DIR / "perceptor"
SILERO_VAD_ONNX = PERCEPTOR_MODELS_DIR / "silero_vad.onnx"
TITANET_ONNX = PERCEPTOR_MODELS_DIR / "nemo_en_titanet_small.onnx"
VOICES_JSON = DATA_DIR / "voices.json"

# Cosine similarity above which two clusters are the same speaker (diarization)
SPEAKER_CLUSTER_THRESHOLD = 0.38
# Cosine similarity above which a video speaker matches a registered channel
# voice. Empirical starting point — tune on Italian voices.
VOICE_MATCH_THRESHOLD = 0.60
# VAD segments shorter than this carry too little signal for an embedding.
MIN_EMBED_SEGMENT_S = 1.0
# Speakers below these floors are noise micro-clusters (background voices,
# TV audio, jingles): they stay in the diarization output but are NOT
# registered in the channel voice registry.
MIN_REGISTRY_TALK_TIME_S = 15.0
MIN_REGISTRY_SEGMENTS = 3

_models_lock = threading.Lock()
_vad_config = None
_embedding_extractor = None

VAD_SAMPLE_RATE = 16000
_VAD_WINDOW = 512  # samples per accept_waveform() push at 16 kHz


def _load_audio_16k_mono(audio_path: Path) -> "np.ndarray":  # noqa: F821
    """Decode any cached audio file to float32 mono 16 kHz via PyAV."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=VAD_SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(str(audio_path)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks)
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _get_embedding_extractor():
    """Cached sherpa-onnx SpeakerEmbeddingExtractor (thread-safe)."""
    global _embedding_extractor
    if _embedding_extractor is not None:
        return _embedding_extractor
    with _models_lock:
        if _embedding_extractor is not None:
            return _embedding_extractor
        import sherpa_onnx

        if not TITANET_ONNX.exists():
            raise FileNotFoundError(
                f"Speaker embedding model missing: {TITANET_ONNX}. "
                "Run: python -m scripts.setup_models --perceptor-only"
            )
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(TITANET_ONNX), num_threads=2
        )
        _embedding_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        logger.info("Speaker embedding extractor loaded (dim=%d)", _embedding_extractor.dim)
    return _embedding_extractor


def _new_vad():
    """Fresh VoiceActivityDetector (sherpa-onnx VADs are stateful per stream)."""
    import sherpa_onnx

    if not SILERO_VAD_ONNX.exists():
        raise FileNotFoundError(
            f"Silero VAD model missing: {SILERO_VAD_ONNX}. "
            "Run: python -m scripts.setup_models --perceptor-only"
        )
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(SILERO_VAD_ONNX)
    config.sample_rate = VAD_SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)


def detect_speech_segments(audio_path: Path) -> list[dict]:
    """Run Silero VAD over an audio file.

    Returns [{"start": seconds, "end": seconds}, ...] sorted by start time.
    """
    signal = _load_audio_16k_mono(audio_path)
    if signal.size == 0:
        return []
    vad = _new_vad()
    segments: list[dict] = []

    def _drain() -> None:
        while not vad.empty():
            seg = vad.front
            start = seg.start / VAD_SAMPLE_RATE
            end = start + len(seg.samples) / VAD_SAMPLE_RATE
            segments.append({"start": round(start, 2), "end": round(end, 2)})
            vad.pop()

    for i in range(0, len(signal), _VAD_WINDOW):
        vad.accept_waveform(signal[i : i + _VAD_WINDOW])
        _drain()
    vad.flush()
    _drain()
    segments.sort(key=lambda s: s["start"])
    return segments


def embed_segments(audio_path: Path, segments: list[dict]) -> "np.ndarray":  # noqa: F821
    """Extract one 192-dim speaker embedding per VAD segment.

    Segments shorter than MIN_EMBED_SEGMENT_S yield a zero vector so the
    output stays index-aligned with *segments* (callers skip zero rows).
    """
    import numpy as np

    extractor = _get_embedding_extractor()
    signal = _load_audio_16k_mono(audio_path)
    out = np.zeros((len(segments), extractor.dim), dtype=np.float32)
    for i, seg in enumerate(segments):
        if seg["end"] - seg["start"] < MIN_EMBED_SEGMENT_S:
            continue
        lo = int(seg["start"] * VAD_SAMPLE_RATE)
        hi = min(int(seg["end"] * VAD_SAMPLE_RATE), len(signal))
        if hi - lo < VAD_SAMPLE_RATE * MIN_EMBED_SEGMENT_S:
            continue
        stream = extractor.create_stream()
        stream.accept_waveform(VAD_SAMPLE_RATE, signal[lo:hi])
        stream.input_finished()
        out[i] = np.asarray(extractor.compute(stream), dtype=np.float32)
    return out


def _cosine(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cluster_speakers(
    embeddings: "np.ndarray",  # noqa: F821
    *,
    threshold: float = SPEAKER_CLUSTER_THRESHOLD,
) -> list[int]:
    """Greedy cosine clustering of segment embeddings into speaker labels.

    Returns one integer label per row; -1 for zero rows (too short to embed).
    Labels are renumbered by descending total segment count so label 0 is the
    most-talking speaker.
    """
    import numpy as np

    labels = [-1] * len(embeddings)
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    for i, emb in enumerate(embeddings):
        if not np.any(emb):
            continue
        best, best_sim = -1, threshold
        for c, centroid in enumerate(centroids):
            sim = _cosine(emb, centroid / counts[c])
            if sim > best_sim:
                best, best_sim = c, sim
        if best == -1:
            centroids.append(emb.astype(np.float64).copy())
            counts.append(1)
            labels[i] = len(centroids) - 1
        else:
            centroids[best] += emb
            counts[best] += 1
            labels[i] = best

    # Renumber by cluster size (0 = dominant speaker)
    order = sorted(range(len(counts)), key=lambda c: -counts[c])
    remap = {old: new for new, old in enumerate(order)}
    return [remap[lab] if lab >= 0 else -1 for lab in labels]


def speaker_centroids(
    embeddings: "np.ndarray",  # noqa: F821
    labels: list[int],
) -> dict[str, "np.ndarray"]:  # noqa: F821
    """Mean embedding per speaker label ("S0", "S1", ...)."""
    import numpy as np

    out: dict[str, np.ndarray] = {}
    for lab in sorted({l for l in labels if l >= 0}):
        rows = embeddings[[i for i, x in enumerate(labels) if x == lab]]
        out[f"S{lab}"] = rows.mean(axis=0)
    return out


def assign_speakers_to_transcript(
    transcript: dict, segments: list[dict], labels: list[int]
) -> list[dict]:
    """Label each Whisper transcript segment with the dominant speaker.

    A transcript segment gets the label of the VAD segment it overlaps most;
    "S?" when it overlaps no labelled VAD segment.
    """
    out: list[dict] = []
    for tseg in (transcript or {}).get("segments", []):
        t_start, t_end = tseg.get("start", 0.0), tseg.get("end", 0.0)
        best_label, best_overlap = -1, 0.0
        for vseg, lab in zip(segments, labels):
            if lab < 0:
                continue
            overlap = min(t_end, vseg["end"]) - max(t_start, vseg["start"])
            if overlap > best_overlap:
                best_overlap, best_label = overlap, lab
        out.append(
            {
                "start": t_start,
                "end": t_end,
                "speaker": f"S{best_label}" if best_label >= 0 else "S?",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Cross-video voice registry (per channel)
# ---------------------------------------------------------------------------


def load_voice_registry() -> list[dict]:
    return load_json(VOICES_JSON)


def match_or_register_voices(
    channel_id: str,
    video_id: str,
    centroids: dict[str, "np.ndarray"],  # noqa: F821
    *,
    threshold: float = VOICE_MATCH_THRESHOLD,
) -> list[dict]:
    """Match per-video speaker centroids against the channel's voice registry.

    Known voices (cosine >= threshold) update their running-mean centroid;
    unknown ones are registered as ``voice_{channel_id}_{NNN}``. Returns
    [{"label": "S0", "voice_id": ..., "score": float, "new": bool}, ...] and
    persists the updated registry.
    """
    import numpy as np

    registry = load_voice_registry()
    channel_voices = [v for v in registry if v.get("channel_id") == channel_id]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[dict] = []

    for label in sorted(centroids):
        emb = np.asarray(centroids[label], dtype=np.float64)
        best_voice, best_sim = None, threshold
        for voice in channel_voices:
            sim = _cosine(emb, np.asarray(voice["centroid"], dtype=np.float64))
            if sim > best_sim:
                best_voice, best_sim = voice, sim

        if best_voice is not None:
            n = int(best_voice.get("n_samples", 1))
            old = np.asarray(best_voice["centroid"], dtype=np.float64)
            best_voice["centroid"] = ((old * n + emb) / (n + 1)).tolist()
            best_voice["n_samples"] = n + 1
            if video_id not in best_voice.get("videos", []):
                best_voice.setdefault("videos", []).append(video_id)
            best_voice["updated_at"] = now
            results.append(
                {
                    "label": label,
                    "voice_id": best_voice["voice_id"],
                    "score": round(best_sim, 3),
                    "new": False,
                }
            )
        else:
            voice_id = f"voice_{channel_id}_{len(channel_voices) + 1:03d}"
            voice = {
                "voice_id": voice_id,
                "channel_id": channel_id,
                "centroid": emb.tolist(),
                "n_samples": 1,
                "videos": [video_id],
                "created_at": now,
                "updated_at": now,
            }
            registry.append(voice)
            channel_voices.append(voice)
            results.append(
                {"label": label, "voice_id": voice_id, "score": 1.0, "new": True}
            )

    save_json(VOICES_JSON, registry)
    return results
