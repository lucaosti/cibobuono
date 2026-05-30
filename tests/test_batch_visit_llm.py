"""Tests for batch_visit_llm."""

__author__ = "Luca Ostinelli"

from unittest.mock import MagicMock

from scripts.batch_visit_llm import batch_evaluate_candidates


def test_batch_evaluate_parses_results():
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"results": ['
                        '{"id": "c0", "is_venue": true, "is_visit": true, "evidence": "siamo da Remo"},'
                        '{"id": "c1", "is_venue": false, "is_visit": false, "evidence": ""}'
                        "]}"
                    )
                }
            }
        ]
    }

    items = [
        {"id": "c0", "name": "Da Remo", "window": "Siamo da Remo e mangiamo la pizza."},
        {"id": "c1", "name": "carbonara", "window": "Parliamo della carbonara."},
    ]
    out = batch_evaluate_candidates(mock_llm, items, video_title="Roma", batch_size=10)
    assert out["c0"].is_venue is True
    assert out["c0"].is_visit is True
    assert "Remo" in out["c0"].evidence
    assert out["c1"].is_venue is False
    assert mock_llm.create_chat_completion.call_count == 1
