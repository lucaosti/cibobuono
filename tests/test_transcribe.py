"""
Tests for transcribe_video module: VTT parsing and scrolling-subtitle dedup.
"""

__author__ = "Luca Ostinelli"

import pytest
from pathlib import Path
from scripts.transcribe_video import _parse_vtt


@pytest.fixture
def tmp_vtt(tmp_path):
    """Helper to write a VTT file and return its path."""
    def _write(content: str) -> Path:
        p = tmp_path / "test_subs.it.vtt"
        p.write_text(content, encoding="utf-8")
        return p
    return _write


class TestParseVttScrollingDedup:
    """YouTube auto-subs use a scrolling format where each cue repeats
    lines from the previous cue.  Our parser should deduplicate them."""

    def test_basic_scrolling_dedup(self, tmp_vtt):
        """Each cue shows 2 lines; the first line repeats from the previous cue."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "ciao a tutti\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "ciao a tutti\n"
            "oggi andiamo a mangiare\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "oggi andiamo a mangiare\n"
            "la pizza migliore\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        # Each unique line should appear exactly once in the full text
        assert result["text"].count("ciao a tutti") == 1
        assert result["text"].count("oggi andiamo a mangiare") == 1
        assert result["text"].count("la pizza migliore") == 1

    def test_exact_duplicate_cues(self, tmp_vtt):
        """Consecutive cues with identical text → merge into one segment."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "stessa riga\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "stessa riga\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert result["text"].count("stessa riga") == 1
        # Only one segment but with extended end time
        assert len(result["segments"]) == 1
        assert result["segments"][0]["end"] == 4.0

    def test_no_overlap_preserved(self, tmp_vtt):
        """When cues don't overlap, all lines are preserved."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "prima frase\n\n"
            "00:00:02.000 --> 00:00:04.000\n"
            "seconda frase\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "terza frase\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert result["text"] == "prima frase seconda frase terza frase"
        assert len(result["segments"]) == 3

    def test_formatting_tags_stripped(self, tmp_vtt):
        """VTT tags like <c> and timestamps are removed."""
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:03.000\n"
            "<c>ciao</c> <00:00:01.500>mondo</c>\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert "<" not in result["text"]
        assert "ciao" in result["text"]
        assert "mondo" in result["text"]

    def test_source_field(self, tmp_vtt):
        """Parsed transcript should be tagged as YouTube manual subs.

        Manual VTTs are the only YouTube-sub path we still trust — the
        auto-generated branch was removed because Italian proper nouns get
        mangled (e.g. "Raimond di Garibaldi" instead of "Raimondi di Garibaldi").
        """
        vtt = tmp_vtt(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "test\n\n"
        )
        result = _parse_vtt(vtt)
        assert result is not None
        assert result["source"] == "youtube_subs_manual"

    def test_empty_vtt_returns_none(self, tmp_vtt):
        """VTT with no cues returns None."""
        vtt = tmp_vtt("WEBVTT\n\n")
        result = _parse_vtt(vtt)
        assert result is None


class TestMlxBackendSelection:
    """The mlx-whisper backend is used when the hardware profile selects it,
    with graceful fallback to faster-whisper when mlx is not importable."""

    @pytest.fixture(autouse=True)
    def _reset_model_cache(self):
        import scripts.transcribe_video as tv
        tv.release_whisper_model()
        yield
        tv.release_whisper_model()

    def _profile_with_backend(self, backend: str):
        from scripts.hardware import get_profile
        base = get_profile()
        import dataclasses
        return dataclasses.replace(base, asr_backend=backend)

    def test_mlx_backend_selected(self, monkeypatch):
        import sys
        import types
        import scripts.transcribe_video as tv

        fake_mlx = types.ModuleType("mlx_whisper")
        fake_mlx.transcribe = lambda *a, **k: {"language": "it", "segments": []}
        monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
        monkeypatch.setattr(
            tv, "get_profile", lambda: self._profile_with_backend("mlx_whisper")
        )

        with tv._whisper_lock:
            model, backend = tv._load_whisper_model_locked("large-v3-turbo")

        assert backend == tv._BACKEND_MLX
        assert isinstance(model, tv._MlxWhisperModel)
        assert model.repo_id == "mlx-community/whisper-large-v3-turbo"

    def test_mlx_wrapper_drops_fp16_kwarg(self, monkeypatch):
        import sys
        import types
        import scripts.transcribe_video as tv

        seen: dict = {}

        def fake_transcribe(audio, **kwargs):
            seen.update(kwargs)
            return {"language": "it", "segments": []}

        fake_mlx = types.ModuleType("mlx_whisper")
        fake_mlx.transcribe = fake_transcribe
        monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)

        wrapper = tv._MlxWhisperModel("mlx-community/whisper-large-v3-turbo")
        wrapper.transcribe("audio.wav", language="it", fp16=True, verbose=False)

        assert "fp16" not in seen
        assert seen["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
        assert seen["language"] == "it"

    def test_fallback_when_mlx_missing(self, monkeypatch):
        import builtins
        import sys
        import scripts.transcribe_video as tv

        monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)
        real_import = builtins.__import__

        def block_mlx(name, *args, **kwargs):
            if name == "mlx_whisper":
                raise ImportError("mlx_whisper unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_mlx)
        monkeypatch.setattr(
            tv, "get_profile", lambda: self._profile_with_backend("mlx_whisper")
        )

        class FakeWhisperModel:
            def __init__(self, *a, **k):
                pass

        fake_fw = pytest.importorskip("types").ModuleType("faster_whisper")
        fake_fw.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        with tv._whisper_lock:
            model, backend = tv._load_whisper_model_locked("large-v3-turbo")

        assert backend == tv._BACKEND_FASTER
        assert isinstance(model, FakeWhisperModel)
