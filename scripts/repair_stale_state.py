"""
repair_stale_state.py — Fix videos left pending after output was already written.

If the pipeline stops between populate_visits / populate_flagged and update_video_status,
visits or flagged segments exist on disk while videos.json still shows pending.
This pass marks those videos processed and ensures processed_videos.json has an entry.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from scripts.fetch_videos import update_video_status
from scripts.populate_json import update_processed_videos
from scripts.schemas import VideoStatus
from scripts.utils import (
    FLAGGED_SEGMENTS_JSON,
    PROCESSED_VIDEOS_JSON,
    VIDEOS_JSON,
    VISITS_JSON,
    load_json,
    setup_logging,
)

logger = setup_logging("repair_stale")


def repair_stale_video_state(dry_run: bool = False) -> dict:
    """
    For each video with status pending but existing visits or flagged rows,
    set status to processed and add processed_videos entry if missing.

    Returns a summary dict with keys: repaired_video_ids, dry_run, counts.
    """
    videos = load_json(VIDEOS_JSON)
    visits = load_json(VISITS_JSON)
    flagged = load_json(FLAGGED_SEGMENTS_JSON)
    processed = load_json(PROCESSED_VIDEOS_JSON)

    visit_counts: dict[str, int] = defaultdict(int)
    flagged_counts: dict[str, int] = defaultdict(int)
    for v in visits:
        vid = v.get("video_id")
        if vid:
            visit_counts[vid] += 1
    for f in flagged:
        vid = f.get("video_id")
        if vid:
            flagged_counts[vid] += 1

    processed_ids = {p.get("video_id") for p in processed if p.get("video_id")}
    repaired: list[str] = []

    for entry in videos:
        if entry.get("status") != VideoStatus.PENDING.value:
            continue
        video_id = entry.get("video_id")
        if not video_id:
            continue
        n_vis = visit_counts.get(video_id, 0)
        n_flag = flagged_counts.get(video_id, 0)
        if n_vis == 0 and n_flag == 0:
            continue

        repaired.append(video_id)
        publish_date = entry.get("publish_date") or ""
        channel_id = entry.get("channel_id") or ""

        if dry_run:
            logger.info(
                f"[dry-run] Would mark {video_id} processed "
                f"(visits={n_vis}, flagged={n_flag})"
            )
            continue

        update_video_status(video_id, VideoStatus.PROCESSED, publish_date)
        if video_id not in processed_ids:
            update_processed_videos(
                video_id,
                channel_id,
                VideoStatus.PROCESSED,
                n_vis,
                n_flag,
            )
            processed_ids.add(video_id)
        logger.info(
            f"Repaired stale state for {video_id} "
            f"(visits={n_vis}, flagged={n_flag})"
        )

    return {
        "dry_run": dry_run,
        "repaired_video_ids": repaired,
        "count": len(repaired),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair pending videos that already have visits or flagged segments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without writing files",
    )
    args = parser.parse_args()
    summary = repair_stale_video_state(dry_run=args.dry_run)
    print(
        f"{'Would repair' if summary['dry_run'] else 'Repaired'} "
        f"{summary['count']} video(s)"
    )
    if summary["repaired_video_ids"]:
        for vid in summary["repaired_video_ids"]:
            print(f"  - {vid}")


if __name__ == "__main__":
    main()
