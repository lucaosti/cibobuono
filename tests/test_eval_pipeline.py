"""Tests for scripts.eval_pipeline — classifier precision/recall/F1 harness."""

__author__ = "Luca Ostinelli"

from scripts.eval_pipeline import evaluate, load_eval_set


class TestLoadEvalSet:
    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "eval_set.json"
        p.write_text("[]")
        assert load_eval_set(p) == []

    def test_drops_malformed_entries(self, tmp_path):
        p = tmp_path / "eval_set.json"
        p.write_text(
            '[{"candidate_name": "Roscioli", "gold_label": "visit"}, '
            '{"candidate_name": "", "gold_label": "visit"}, '
            '{"candidate_name": "X", "gold_label": "not_a_label"}]'
        )
        loaded = load_eval_set(p)
        assert len(loaded) == 1
        assert loaded[0]["candidate_name"] == "Roscioli"


class TestEvaluate:
    def test_empty_set(self):
        m = evaluate([])
        assert m["n"] == 0
        assert m["precision"] is None
        assert m["recall"] is None

    def test_rule_engine_correctly_classifies_clear_visit(self):
        examples = [
            {
                "candidate_name": "Roscioli",
                "window_text": "Siamo da Roscioli e assaggiamo la pizza bianca, buonissima.",
                "gold_label": "visit",
            }
        ]
        m = evaluate(examples)
        assert m["n"] == 1
        assert m["tp"] == 1
        assert m["fp"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_rule_engine_correctly_classifies_clear_mention(self):
        examples = [
            {
                "candidate_name": "Bonci",
                "window_text": "È come da Bonci questo panino, non è mica lo stesso livello.",
                "gold_label": "mention",
            }
        ]
        m = evaluate(examples)
        assert m["tn"] == 1
        assert m["fp"] == 0

    def test_unsure_without_llm_counts_as_false_negative_when_gold_is_visit(self):
        examples = [
            {
                "candidate_name": "Roscioli",
                "window_text": "Non succede niente di rilevante qui.",
                "gold_label": "visit",
            }
        ]
        m = evaluate(examples, llm=None)
        assert m["fn"] == 1

    def test_n_always_matches_input_length(self):
        examples = [
            {"candidate_name": "A", "window_text": "", "gold_label": "visit"},
            {"candidate_name": "B", "window_text": "", "gold_label": "mention"},
        ]
        m = evaluate(examples)
        assert m["n"] == 2
