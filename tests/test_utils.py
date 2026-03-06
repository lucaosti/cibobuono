"""
Tests for utility functions.
"""

from scripts.utils import (
    load_json,
    save_json,
    today_str,
    cleanup_cache,
)


class TestLoadJson:
    def test_load_valid(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('[{"id": 1}]', encoding="utf-8")
        result = load_json(path)
        assert result == [{"id": 1}]

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        result = load_json(path)
        assert result == []

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "missing.json"
        result = load_json(path)
        assert result == []

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        result = load_json(path)
        assert result == []

    def test_load_dict_returns_empty(self, tmp_path):
        """load_json expects arrays, dicts return empty list."""
        path = tmp_path / "dict.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        result = load_json(path)
        assert result == []


class TestSaveJson:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "out.json"
        data = [{"name": "Forno Rossi", "city": "Roma"}]
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_save_creates_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "deep" / "file.json"
        save_json(path, [])
        assert path.exists()

    def test_save_unicode(self, tmp_path):
        path = tmp_path / "unicode.json"
        data = [{"name": "Caffè d'Artista", "city": "Città di Castello"}]
        save_json(path, data)
        loaded = load_json(path)
        assert loaded[0]["name"] == "Caffè d'Artista"


class TestTodayStr:
    def test_format(self):
        result = today_str()
        assert len(result) == 10
        assert result[4] == "-"
        assert result[7] == "-"


class TestCleanupCache:
    def test_no_cleanup_under_limit(self, tmp_path, monkeypatch):
        """Should not delete files when under the limit."""
        import scripts.utils
        monkeypatch.setattr(scripts.utils, "CACHE_DIR", tmp_path)

        # Create 3 files (under default limit of 20)
        for i in range(3):
            (tmp_path / f"video_{i}.wav").write_bytes(b"data")

        deleted = cleanup_cache(max_files=20)
        assert len(deleted) == 0

    def test_cleanup_over_limit(self, tmp_path, monkeypatch):
        """Should delete oldest files when over the limit."""
        import scripts.utils
        import time
        monkeypatch.setattr(scripts.utils, "CACHE_DIR", tmp_path)

        # Create 5 files
        for i in range(5):
            f = tmp_path / f"video_{i}.wav"
            f.write_bytes(b"data")
            # Stagger modification times
            import os
            os.utime(f, (time.time() + i, time.time() + i))

        deleted = cleanup_cache(max_files=3)
        assert len(deleted) == 2
        # Should still have 3 files
        remaining = list(tmp_path.glob("*.wav"))
        assert len(remaining) == 3

    def test_ignores_non_media(self, tmp_path, monkeypatch):
        """Should only count media files for cleanup."""
        import scripts.utils
        monkeypatch.setattr(scripts.utils, "CACHE_DIR", tmp_path)

        # Create media and non-media files
        (tmp_path / "video.wav").write_bytes(b"data")
        (tmp_path / "transcript.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("hello")

        deleted = cleanup_cache(max_files=5)
        assert len(deleted) == 0
