"""
Tests for gold_sample: publish-year assignment and stratified allocation.
"""

__author__ = "Luca Ostinelli"

import pytest

from scripts.gold_sample import MetaStore, allocate, years_by_position


class FakeStore(MetaStore):
    """MetaStore over a fixed id -> year table; ids mapped to None are unreadable."""

    def __init__(self, years: dict[str, int | None]):
        self.years = years
        self.requests = 0

    def get(self, video_id):
        self.requests += 1
        year = self.years[video_id]
        return None if year is None else {"id": video_id, "upload_date": f"{year}0101"}


def _ordered(n):
    return [{"id": f"v{i}"} for i in range(n)]


def test_years_follow_newest_first_order():
    years = [2026] * 4 + [2025] * 3 + [2023] * 5
    store = FakeStore({f"v{i}": y for i, y in enumerate(years)})
    assert years_by_position(_ordered(len(years)), store) == {f"v{i}": y for i, y in enumerate(years)}


def test_unreadable_tail_does_not_abort():
    """The oldest videos of a channel can all be members-only: the oldest
    readable video defines the last year instead of aborting the run."""
    years = [2026] * 5 + [2024] * 5 + [None] * 10
    store = FakeStore({f"v{i}": y for i, y in enumerate(years)})
    result = years_by_position(_ordered(len(years)), store)
    assert result["v0"] == 2026
    assert result["v9"] == 2024
    assert result["v19"] == 2024


def test_no_readable_video_raises():
    store = FakeStore({f"v{i}": None for i in range(5)})
    with pytest.raises(RuntimeError):
        years_by_position(_ordered(5), store)


def test_allocate_respects_budget_and_minimum():
    alloc = allocate({2026: 80, 2025: 15, 2020: 5}, 20)
    assert sum(alloc.values()) == 20
    assert all(v >= 1 for v in alloc.values())
    assert alloc[2026] > alloc[2025] > 0
