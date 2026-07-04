"""
perceptor_video.py — Video perception: frame sampling, novelty dedup, captioning.

Pipeline: PyAV decodes one frame every ``frame_interval_s`` seconds from the
low-res download → perceptual hash (imagehash phash) marks a frame as *novel*
when it is far (Hamming distance > threshold) from every previously kept novel
frame → only novel frames are captioned by a Qwen2-VL vision-language model,
up to the hardware-derived ``caption_budget``.

VLM backends (selected by :func:`scripts.hardware.get_profile`):
- ``mlx_vlm``: Qwen2-VL-2B 4-bit on Apple Silicon (Metal, ~2.4 GB peak).
- ``transformers_cuda``: Qwen2-VL on NVIDIA (7B AWQ on >=16 GB VRAM, 2B fp16
  on 8-16 GB). The AWQ load falls back to the 2B fp16 repo when autoawq is
  missing.
- ``none``: phash novelty stats only, no captions (CPU-only machines).

Heavy imports are function-local; a single adapter wraps each backend so API
churn (mlx-vlm is 0.x) touches one place.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import threading
from dataclasses import dataclass, field
from pathlib import Path

from scripts.hardware import get_profile
from scripts.utils import setup_logging

logger = setup_logging("perceptor_video")

PHASH_NOVELTY_DISTANCE = 6  # Hamming distance beyond which a frame is "new"
MAX_SAMPLED_FRAMES = 2000

# Captions serve two purposes: human-readable evidence and a future NER hook,
# hence the explicit ask for exact signage text.
CAPTION_PROMPT = (
    "Describe this video frame from an Italian food review. "
    "If any text or signage is visible (restaurant names, street signs, menus), "
    "transcribe it exactly. Mention dishes and the type of venue if recognizable. "
    "Be concise (2-3 sentences)."
)
CAPTION_MAX_TOKENS = 120

_vlm_lock = threading.Lock()
_vlm = None  # (backend, model handle(s), model_id)


@dataclass
class FrameSample:
    """One sampled frame; ``image`` is kept only for novel frames."""

    t: float
    phash: str
    novel: bool
    image: "object | None" = field(default=None, repr=False)  # PIL.Image


def sample_novel_frames(
    video_path: Path,
    *,
    interval_s: float,
    phash_distance: int = PHASH_NOVELTY_DISTANCE,
    max_frames: int = MAX_SAMPLED_FRAMES,
) -> list[FrameSample]:
    """Sample one frame every *interval_s* seconds and flag novel ones.

    A frame is novel when its phash is farther than *phash_distance* from all
    previously kept novel frames. Images of non-novel frames are dropped
    immediately to bound memory.
    """
    import av
    import imagehash

    samples: list[FrameSample] = []
    novel_hashes: list = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        next_t = 0.0
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t < next_t:
                continue
            next_t = t + interval_s
            image = frame.to_image()
            h = imagehash.phash(image)
            novel = all(h - prev > phash_distance for prev in novel_hashes)
            if novel:
                novel_hashes.append(h)
            samples.append(
                FrameSample(
                    t=round(t, 2),
                    phash=str(h),
                    novel=novel,
                    image=image if novel else None,
                )
            )
            if len(samples) >= max_frames:
                logger.warning(
                    "Frame sampling capped at %d frames for %s",
                    max_frames,
                    video_path.name,
                )
                break

    logger.info(
        "Sampled %d frames (%d novel) from %s",
        len(samples),
        len(novel_hashes),
        video_path.name,
    )
    return samples


def _subsample_evenly(items: list, budget: int) -> list:
    """Pick at most *budget* items spread evenly across the list."""
    if budget <= 0:
        return []
    if len(items) <= budget:
        return list(items)
    step = len(items) / budget
    return [items[int(i * step)] for i in range(budget)]


# ---------------------------------------------------------------------------
# VLM adapter
# ---------------------------------------------------------------------------


def _load_mlx_vlm(model_id: str):
    from mlx_vlm import generate, load  # noqa: F401 — fail early if missing

    model, processor = load(model_id)
    return ("mlx_vlm", (model, processor), model_id)


def _load_transformers_cuda(model_id: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="cuda"
        )
    except Exception as e:
        # AWQ repos need autoawq; fall back to the 2B fp16 model.
        from scripts.hardware import _VLM_CUDA_2B

        if model_id == _VLM_CUDA_2B:
            raise
        logger.warning("VLM %s load failed (%s); falling back to %s", model_id, e, _VLM_CUDA_2B)
        model_id = _VLM_CUDA_2B
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda"
        )
    processor = AutoProcessor.from_pretrained(model_id)
    return ("transformers_cuda", (model, processor), model_id)


def _get_vlm():
    """Cached VLM handle: (backend, handles, model_id) or None when disabled."""
    global _vlm
    if _vlm is not None:
        return _vlm
    with _vlm_lock:
        if _vlm is not None:
            return _vlm
        profile = get_profile()
        if profile.vlm_backend == "none" or not profile.vlm_model:
            return None
        logger.info("Loading VLM: %s via %s", profile.vlm_model, profile.vlm_backend)
        if profile.vlm_backend == "mlx_vlm":
            _vlm = _load_mlx_vlm(profile.vlm_model)
        else:
            _vlm = _load_transformers_cuda(profile.vlm_model)
    return _vlm


def release_vlm() -> None:
    """Free the VLM's VRAM / unified memory before LLM extraction."""
    global _vlm
    with _vlm_lock:
        if _vlm is None:
            return
        _vlm = None
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    logger.info("VLM released — memory freed")


def _caption_one(vlm, image) -> str:
    """Caption a single PIL image with whichever backend is loaded."""
    backend, handles, model_id = vlm
    model, processor = handles

    if backend == "mlx_vlm":
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            processor, model.config, CAPTION_PROMPT, num_images=1
        )
        result = generate(
            model,
            processor,
            prompt,
            image=[image],
            max_tokens=CAPTION_MAX_TOKENS,
            verbose=False,
        )
        text = result.text if hasattr(result, "text") else str(result)
        return text.strip()

    # transformers_cuda
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=CAPTION_MAX_TOKENS)
    trimmed = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def caption_frames(frames: list[FrameSample], *, budget: int) -> list[dict]:
    """Caption novel frames up to *budget*, evenly spread when over budget.

    Returns [{"t", "phash", "model", "caption"}, ...]. Individual caption
    failures are logged and skipped; a load failure disables captioning for
    this call and returns whatever succeeded so far.
    """
    if budget <= 0:
        return []
    novel = [f for f in frames if f.novel and f.image is not None]
    if not novel:
        return []
    picked = _subsample_evenly(novel, budget)

    try:
        vlm = _get_vlm()
    except Exception as e:
        logger.warning("VLM unavailable, skipping captions: %s", e)
        return []
    if vlm is None:
        return []

    captions: list[dict] = []
    for frame in picked:
        try:
            text = _caption_one(vlm, frame.image)
        except Exception as e:
            logger.warning("Caption failed at t=%.1fs: %s", frame.t, e)
            continue
        if text:
            captions.append(
                {"t": frame.t, "phash": frame.phash, "model": vlm[2], "caption": text}
            )
    logger.info("Captioned %d/%d novel frames", len(captions), len(picked))
    return captions
