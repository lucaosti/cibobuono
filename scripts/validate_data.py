#!/usr/bin/env python3
"""
Validate data/*.json against Pydantic schemas. Exit 0 if all OK, 1 on error.
Used in CI and locally before deploy.
"""

from __future__ import annotations

import sys

from scripts.utils import DATA_DIR, SKIPPED_VIDEOS_JSON, load_json
from scripts.schemas import (
    validate_channels,
    validate_flagged_segments,
    validate_locales,
    validate_processed_videos,
    validate_skipped_videos,
    validate_videos,
    validate_visits,
)


def main() -> int:
    checks = [
        ("channels.json", validate_channels),
        ("videos.json", validate_videos),
        ("locales.json", validate_locales),
        ("visits.json", validate_visits),
        ("flagged_segments.json", validate_flagged_segments),
        ("processed_videos.json", validate_processed_videos),
    ]
    errors: list[str] = []
    for name, fn in checks:
        path = DATA_DIR / name
        try:
            data = load_json(path)
            fn(data)
        except Exception as e:
            errors.append(f"{name}: {e}")

    try:
        data = load_json(SKIPPED_VIDEOS_JSON)
        validate_skipped_videos(data)
    except Exception as e:
        errors.append(f"skipped_videos.json: {e}")

    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print("All data files validate OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
