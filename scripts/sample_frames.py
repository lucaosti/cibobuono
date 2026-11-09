"""
sample_frames.py — Extract a budget of novel frames per video for the teacher LLM.

For every ``<video_id>.<ext>`` in ``--videos`` the Perceptor sampler
(:func:`scripts.perceptor_video.sample_novel_frames`) keeps one frame every
``--interval`` seconds when its perceptual hash is new; the novel frames are
spread evenly down to ``--budget`` and written as ``<out>/<video_id>/<t>.jpg``
(``t`` = seconds from the start), the layout read by :mod:`scripts.teacher_label`.

Videos that already have a frame directory are skipped, so the job resumes.

CLI:
    python -m scripts.sample_frames --videos runs/x/videos --out runs/x/frames \
        [--interval 5] [--budget 64] [--width 960] [--workers 8] [--delete-video]
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.perceptor_video import _subsample_evenly, sample_novel_frames
from scripts.utils import setup_logging

logger = setup_logging("sample_frames")

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov"}


def write_frames(video: Path, out_dir: Path, *, interval: float, budget: int, width: int) -> int:
    """Sample *video* into *out_dir*; returns the number of frames written."""
    samples = [s for s in sample_novel_frames(video, interval_s=interval) if s.novel]
    picked = _subsample_evenly(samples, budget)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for s in picked:
        image = s.image
        if image.width > width:
            image = image.resize((width, round(image.height * width / image.width)))
        image.convert("RGB").save(tmp / f"{int(s.t)}.jpg", quality=85)
    tmp.rename(out_dir)
    return len(picked)


def _sample_one(video: Path, out_dir: Path, interval: float, budget: int, width: int) -> tuple[int, float]:
    t0 = time.time()
    n = write_frames(video, out_dir, interval=interval, budget=budget, width=width)
    return n, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="Novel-frame sampling for teacher labelling")
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--delete-video", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="videos sampled in parallel")
    args = parser.parse_args()

    videos = sorted(
        p for p in args.videos.iterdir() if p.suffix in VIDEO_SUFFIXES and not (args.out / p.stem).exists()
    )
    failures = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_sample_one, v, args.out / v.stem, args.interval, args.budget, args.width): v
            for v in videos
        }
        for fut in as_completed(futures):
            video = futures[fut]
            try:
                n, seconds = fut.result()
            except Exception as e:  # corrupt or partial download: report, keep going
                failures += 1
                logger.error("%s: frame sampling failed: %s", video.stem, e)
                continue
            logger.info("%s: %d frames in %.0fs", video.stem, n, seconds)
            if args.delete_video:
                video.unlink()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
