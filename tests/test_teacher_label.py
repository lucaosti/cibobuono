"""
Tests for the teacher_label server backend, frame attachment and frame sampling output.
"""

__author__ = "Luca Ostinelli"

import io
import json

import pytest
from PIL import Image

import scripts.teacher_label as tl


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.reply}}]}


def _jpg(path, color=(200, 0, 0)):
    Image.new("RGB", (8, 8), color).save(path)


def test_frame_parts_in_time_order_with_timestamps(tmp_path):
    for t in (125, 5, 60):
        _jpg(tmp_path / f"{t}.jpg")
    parts = tl.frame_parts(tmp_path)
    labels = [p["text"] for p in parts if p["type"] == "text"]
    assert labels == ["[0:05]", "[1:00]", "[2:05]"]
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in parts if p["type"] == "image_url")


def test_text_only_request_keeps_plain_prompt():
    llm = FakeLLM('{"venues": []}')
    tl._chat_json(llm, "sys", "user prompt", 100)
    content = llm.calls[0]["messages"][1]["content"]
    assert content == "user prompt"


def test_images_are_attached_after_prompt_with_frames_note(tmp_path):
    _jpg(tmp_path / "10.jpg")
    llm = FakeLLM('{"venues": [{"name": "Tiella"}]}')
    data = tl._chat_json(llm, "sys", "user prompt", 100, tl.frame_parts(tmp_path))
    content = llm.calls[0]["messages"][1]["content"]
    assert content[0]["type"] == "text" and "FRAMES:" in content[0]["text"]
    assert [c["type"] for c in content[1:]] == ["text", "image_url"]
    assert data["venues"][0]["name"] == "Tiella"


def test_null_content_yields_empty_dict():
    llm = FakeLLM(None)
    assert tl._chat_json(llm, "sys", "u", 10) == {}


def test_empty_reply_is_a_failed_label_not_zero_venues(monkeypatch):
    import scripts.video_intelligence as vi

    monkeypatch.setattr(vi, "analyze_title", lambda title: type("I", (), {"series_name": "", "location_hint": "", "city": ""})())
    assert tl.label_video(FakeLLM(None), {"title": "t"}, {"segments": []}) is None
    recipe = tl.label_video(FakeLLM('{"video_type": "recipe", "venues": []}'), {"title": "t"}, {"segments": []})
    assert recipe == {"video_type": "recipe", "venues": []}


def test_openai_chat_posts_completion_request(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr(tl.urllib.request, "urlopen", fake_urlopen)
    chat = tl.OpenAIChat("http://127.0.0.1:8000/v1/", "teacher", temperature=0.6)
    chat.create_chat_completion(messages=[{"role": "user", "content": "x"}], max_tokens=9,
                                temperature=0.0, response_format={"type": "json_object"})
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["body"]["model"] == "teacher"
    assert seen["body"]["temperature"] == 0.6
    assert seen["body"]["max_tokens"] == 9


def test_sample_frames_writes_budget_named_by_second(tmp_path, monkeypatch):
    import scripts.sample_frames as sf
    from scripts.perceptor_video import FrameSample

    samples = [FrameSample(t=float(t), phash="", novel=True, image=Image.new("RGB", (1920, 1080)))
               for t in range(0, 100, 10)]
    monkeypatch.setattr(sf, "sample_novel_frames", lambda video, interval_s: samples)
    out = tmp_path / "vid"
    assert sf.write_frames(tmp_path / "vid.mp4", out, interval=5, budget=4, width=960) == 4
    files = sorted(out.glob("*.jpg"), key=lambda p: int(p.stem))
    assert [p.stem for p in files] == ["0", "20", "50", "70"]
    assert Image.open(files[0]).size == (960, 540)
    assert not (tmp_path / "vid.tmp").exists()
