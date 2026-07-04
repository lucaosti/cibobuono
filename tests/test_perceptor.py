"""Tests for the Perceptor modules: novelty dedup, diarization clustering,
voice registry, transcript speaker assignment, and stage failure guards."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import perceptor as perc
from scripts import perceptor_audio as pa
from scripts.perceptor_video import FrameSample, _subsample_evenly, caption_frames


# ---------------------------------------------------------------------------
# perceptor_video: novelty + budget
# ---------------------------------------------------------------------------


class TestSubsampleEvenly:
    def test_under_budget_returns_all(self):
        assert _subsample_evenly([1, 2, 3], 10) == [1, 2, 3]

    def test_zero_budget_returns_empty(self):
        assert _subsample_evenly([1, 2, 3], 0) == []

    def test_over_budget_spreads_evenly(self):
        picked = _subsample_evenly(list(range(100)), 10)
        assert len(picked) == 10
        assert picked[0] == 0
        assert picked[-1] == 90


class TestCaptionFrames:
    def test_zero_budget_skips_everything(self):
        frames = [FrameSample(t=0.0, phash="a" * 16, novel=True, image=object())]
        assert caption_frames(frames, budget=0) == []

    def test_no_novel_frames_returns_empty(self):
        frames = [FrameSample(t=0.0, phash="a" * 16, novel=False)]
        assert caption_frames(frames, budget=10) == []

    def test_vlm_load_failure_returns_empty(self, monkeypatch):
        import scripts.perceptor_video as pv

        monkeypatch.setattr(
            pv, "_get_vlm", lambda: (_ for _ in ()).throw(RuntimeError("no vlm"))
        )
        frames = [FrameSample(t=0.0, phash="a" * 16, novel=True, image=object())]
        assert pv.caption_frames(frames, budget=5) == []


# ---------------------------------------------------------------------------
# perceptor_audio: clustering + speaker assignment
# ---------------------------------------------------------------------------


def _fake_embeddings(n_a: int, n_b: int, dim: int = 192) -> np.ndarray:
    """Two well-separated clusters along different axes."""
    rng = np.random.RandomState(0)
    a = np.zeros((n_a, dim), dtype=np.float32)
    a[:, 0] = 1.0
    b = np.zeros((n_b, dim), dtype=np.float32)
    b[:, 1] = 1.0
    noise = rng.randn(n_a + n_b, dim).astype(np.float32) * 0.05
    return np.vstack([a, b]) + noise


class TestClusterSpeakers:
    def test_two_separated_clusters(self):
        emb = _fake_embeddings(5, 3)
        labels = pa.cluster_speakers(emb)
        assert len(set(labels)) == 2
        # Label 0 is the dominant (larger) cluster
        assert labels[:5] == [0] * 5
        assert labels[5:] == [1] * 3

    def test_zero_rows_get_minus_one(self):
        emb = _fake_embeddings(3, 2)
        emb[2] = 0.0  # too-short segment placeholder
        labels = pa.cluster_speakers(emb)
        assert labels[2] == -1
        assert all(l >= 0 for i, l in enumerate(labels) if i != 2)

    def test_single_speaker(self):
        emb = _fake_embeddings(4, 0)
        labels = pa.cluster_speakers(emb)
        assert labels == [0, 0, 0, 0]


class TestAssignSpeakers:
    def test_dominant_overlap_wins(self):
        transcript = {"segments": [{"start": 0.0, "end": 4.0, "text": "ciao"}]}
        segments = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 4.0}]
        labels = [1, 0]
        out = pa.assign_speakers_to_transcript(transcript, segments, labels)
        assert out == [{"start": 0.0, "end": 4.0, "speaker": "S0"}]

    def test_no_overlap_gives_unknown(self):
        transcript = {"segments": [{"start": 100.0, "end": 105.0, "text": "x"}]}
        segments = [{"start": 0.0, "end": 1.0}]
        out = pa.assign_speakers_to_transcript(transcript, segments, [0])
        assert out[0]["speaker"] == "S?"

    def test_empty_transcript(self):
        assert pa.assign_speakers_to_transcript(None, [], []) == []


# ---------------------------------------------------------------------------
# perceptor_audio: voice registry
# ---------------------------------------------------------------------------


@pytest.fixture
def voices_file(tmp_path, monkeypatch):
    dest = tmp_path / "voices.json"
    monkeypatch.setattr(pa, "VOICES_JSON", dest)
    return dest


class TestVoiceRegistry:
    def test_register_new_voice(self, voices_file):
        centroid = np.zeros(192)
        centroid[0] = 1.0
        results = pa.match_or_register_voices("UCx", "vid1", {"S0": centroid})
        assert results == [
            {"label": "S0", "voice_id": "voice_UCx_001", "score": 1.0, "new": True}
        ]
        registry = pa.load_voice_registry()
        assert len(registry) == 1
        assert registry[0]["videos"] == ["vid1"]

    def test_match_existing_voice_updates_centroid(self, voices_file):
        base = np.zeros(192)
        base[0] = 1.0
        pa.match_or_register_voices("UCx", "vid1", {"S0": base})

        similar = base + np.full(192, 0.01)
        results = pa.match_or_register_voices("UCx", "vid2", {"S0": similar})
        assert results[0]["new"] is False
        assert results[0]["voice_id"] == "voice_UCx_001"
        assert results[0]["score"] >= pa.VOICE_MATCH_THRESHOLD

        registry = pa.load_voice_registry()
        assert len(registry) == 1
        assert registry[0]["n_samples"] == 2
        assert set(registry[0]["videos"]) == {"vid1", "vid2"}
        # Running mean moved toward the new sample
        assert registry[0]["centroid"][1] == pytest.approx(0.005, abs=1e-6)

    def test_different_channel_never_matches(self, voices_file):
        base = np.zeros(192)
        base[0] = 1.0
        pa.match_or_register_voices("UCx", "vid1", {"S0": base})
        results = pa.match_or_register_voices("UCy", "vid2", {"S0": base})
        assert results[0]["new"] is True
        assert results[0]["voice_id"] == "voice_UCy_001"
        assert len(pa.load_voice_registry()) == 2

    def test_orthogonal_voice_registers_second(self, voices_file):
        a = np.zeros(192)
        a[0] = 1.0
        b = np.zeros(192)
        b[1] = 1.0
        pa.match_or_register_voices("UCx", "vid1", {"S0": a})
        results = pa.match_or_register_voices("UCx", "vid2", {"S0": b})
        assert results[0]["new"] is True
        assert results[0]["voice_id"] == "voice_UCx_002"


# ---------------------------------------------------------------------------
# perceptor: enablement + persistence + failure guard
# ---------------------------------------------------------------------------


@pytest.fixture
def perception_file(tmp_path, monkeypatch):
    dest = tmp_path / "perception.json"
    monkeypatch.setattr(perc, "PERCEPTION_JSON", dest)
    return dest


class TestPerceptorEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("CIBOBUONO_PERCEPTOR", raising=False)
        assert perc.perceptor_enabled() is False

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv("CIBOBUONO_PERCEPTOR", "1")
        monkeypatch.setattr(perc, "_deps_available", lambda: True)
        assert perc.perceptor_enabled() is True

    def test_cli_flag_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("CIBOBUONO_PERCEPTOR", "1")
        assert perc.perceptor_enabled(False) is False

    def test_missing_deps_disable(self, monkeypatch):
        monkeypatch.setattr(perc, "_deps_available", lambda: False)
        assert perc.perceptor_enabled(True) is False


class TestUpsertPerception:
    def test_insert_then_replace(self, perception_file):
        perc.upsert_perception({"video_id": "a", "status": "ok"})
        perc.upsert_perception({"video_id": "b", "status": "ok"})
        perc.upsert_perception({"video_id": "a", "status": "partial"})
        assert perc.get_perception("a")["status"] == "partial"
        assert perc.get_perception("b")["status"] == "ok"
        assert perc.get_perception("missing") is None


class TestRunPerceptorStage:
    def test_swallows_exceptions(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("perception exploded")

        monkeypatch.setattr(perc, "perceive_video", boom)
        record = perc.run_perceptor_stage(
            {"video_id": "v1", "url": "", "channel_id": "c1"}, None
        )
        assert record is not None
        assert record["status"] == "errored"
        assert "perception exploded" in record["error"]

    def test_partial_status_reported(self, monkeypatch):
        logged: list[str] = []
        monkeypatch.setattr(
            perc,
            "perceive_video",
            lambda *a, **k: {"video_id": "v1", "status": "partial", "error": "video: x"},
        )
        record = perc.run_perceptor_stage(
            {"video_id": "v1"}, None, log=logged.append
        )
        assert record["status"] == "partial"
        assert any("partial" in line for line in logged)
