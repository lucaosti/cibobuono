"""
Tests for chunk_transcription module.
"""

__author__ = "Luca Ostinelli"

from scripts.chunk_transcription import (
    chunk_transcription,
    seconds_to_timestamp,
    timestamp_to_seconds,
)


class TestSecondsToTimestamp:
    def test_zero(self):
        assert seconds_to_timestamp(0) == "0:00"

    def test_minutes_seconds(self):
        assert seconds_to_timestamp(754) == "12:34"

    def test_hours(self):
        assert seconds_to_timestamp(3661) == "1:01:01"

    def test_large(self):
        assert seconds_to_timestamp(7200) == "2:00:00"


class TestTimestampToSeconds:
    def test_mm_ss(self):
        assert timestamp_to_seconds("12:34") == 754

    def test_hh_mm_ss(self):
        assert timestamp_to_seconds("1:01:01") == 3661

    def test_zero(self):
        assert timestamp_to_seconds("0:00") == 0


class TestChunkTranscription:
    def _make_transcript(self, duration_seconds, segment_duration=5):
        """Create a mock Whisper transcript."""
        segments = []
        t = 0.0
        idx = 0
        while t < duration_seconds:
            end = min(t + segment_duration, duration_seconds)
            segments.append({
                "id": idx,
                "start": t,
                "end": end,
                "text": f"Segment {idx} from {t:.0f}s to {end:.0f}s.",
            })
            t = end
            idx += 1
        return {
            "video_id": "test_video",
            "language": "it",
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
        }

    def test_empty_transcript(self):
        transcript = {"video_id": "test", "segments": []}
        chunks = chunk_transcription(transcript)
        assert chunks == []

    def test_short_video(self):
        """Video shorter than chunk duration produces 1 chunk."""
        transcript = self._make_transcript(30)
        chunks = chunk_transcription(transcript, chunk_duration=90)
        assert len(chunks) == 1
        assert chunks[0]["video_id"] == "test_video"

    def test_multi_chunk(self):
        """5-minute video with 90s chunks produces multiple chunks."""
        transcript = self._make_transcript(300)
        chunks = chunk_transcription(transcript, chunk_duration=90, overlap=15)
        assert len(chunks) >= 3

    def test_chunk_has_required_fields(self):
        transcript = self._make_transcript(100)
        chunks = chunk_transcription(transcript, chunk_duration=90)
        for chunk in chunks:
            assert "video_id" in chunk
            assert "chunk_index" in chunk
            assert "start_time" in chunk
            assert "end_time" in chunk
            assert "start_timestamp" in chunk
            assert "end_timestamp" in chunk
            assert "text" in chunk
            assert len(chunk["text"]) > 0

    def test_overlap(self):
        """Chunks should overlap by the specified amount."""
        transcript = self._make_transcript(300)
        chunks = chunk_transcription(transcript, chunk_duration=90, overlap=15)
        if len(chunks) >= 2:
            # Second chunk should start before first chunk ends
            assert chunks[1]["start_time"] < chunks[0]["end_time"]

    def test_timestamps_ascending(self):
        """Chunk start times should be monotonically increasing."""
        transcript = self._make_transcript(600)
        chunks = chunk_transcription(transcript, chunk_duration=90, overlap=15)
        for i in range(1, len(chunks)):
            assert chunks[i]["start_time"] > chunks[i - 1]["start_time"]
