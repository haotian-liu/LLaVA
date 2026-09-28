import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parent / "convert_seed_for_submission.py"
    spec = importlib.util.spec_from_file_location("seed_score_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score(module, question_id, text, answer):
    module.data = {
        "questions": [
            {
                "question_id": question_id,
                "data_type": "image",
                "question_type_id": 1,
                "answer": answer,
            }
        ]
    }
    module.ques_type_id_to_name = {1: "scene"}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        handle.write(json.dumps({"question_id": question_id, "text": text}) + "\n")
        path = handle.name
    buf = io.StringIO()
    with redirect_stdout(buf):
        module.eval_single(path)
    return buf.getvalue()


class SeedScoreTest(unittest.TestCase):
    def test_a_string_question_id_keeps_a_correct_answer(self):
        text = _score(_load(), "42", "A", "A")
        self.assertIn("Total accuracy: 100.00%", text)

    def test_an_int_question_id_still_scores(self):
        text = _score(_load(), 42, "A", "A")
        self.assertIn("Total accuracy: 100.00%", text)


if __name__ == "__main__":
    unittest.main()
