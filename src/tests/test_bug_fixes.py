"""
Unit tests for the two gameplay bug fixes:

1. select_liste: generator exhaustion bug — previously-asked question IDs
   were only excluded from the first iteration; subsequent iterations saw an
   exhausted generator and re-included already-asked questions.
   Fix: use a set comprehension instead of a generator expression.

2. find_similarities: returned raw numpy indices instead of character IDs,
   causing find_best_suggestions to look up wrong characters.
   Fix: return closest_character_ids instead of closest_indices.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: inject MagicMock stubs for all Django / ORM modules BEFORE
# importing anything from the project.  MagicMock is used (not bare
# ModuleType) so that any attribute access (CharField, IntegerField, …)
# automatically returns a MagicMock instead of raising AttributeError.
# ---------------------------------------------------------------------------

src_dir = os.path.join(os.path.dirname(__file__), "..")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# PlayerAnswer is a plain dataclass — define it before stubbing app.models
@dataclass
class _PlayerAnswer:
    question_id: int
    choice: str


_DJANGO_STUBS = [
    "django",
    "django.db",
    "django.db.models",
    "django.db.transaction",
    "django.db.models.base",
    "django.shortcuts",
]

for _name in _DJANGO_STUBS:
    sys.modules[_name] = MagicMock()

# app.models.* stubs — replace every submodule and the package __init__
_APP_MODEL_STUBS = [
    "app.models",
    "app.models.Answer",
    "app.models.AppSettings",
    "app.models.AutoEncoderTrainingRun",
    "app.models.dataclasses",
    "app.models.Doublon",
    "app.models.Item",
    "app.models.Question",
    "app.models.SerializableModel",
    "app.models.Suggestion",
    "app.models.Tree",
]
for _name in _APP_MODEL_STUBS:
    sys.modules[_name] = MagicMock()

# Expose PlayerAnswer as the real dataclass on the app.models stub
sys.modules["app.models"].PlayerAnswer = _PlayerAnswer
sys.modules["app.models.dataclasses"].PlayerAnswer = _PlayerAnswer

# admin.utils.tree.settings stub
_tree_settings = MagicMock()
_tree_settings.MINIMUM_RATIO_VALID_QUESTION = 0.5
sys.modules["admin.utils.tree"] = MagicMock()
sys.modules["admin.utils.tree.settings"] = _tree_settings

# ---------------------------------------------------------------------------
# Now we can safely import the modules under test
# ---------------------------------------------------------------------------

import app.utilities.utilities as utilities  # noqa: E402
import admin.utils.autoencoder.datas as ae_datas  # noqa: E402
from admin.utils.suggestions.find_similarities import find_similarities  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_player_answer(question_id: int) -> _PlayerAnswer:
    return _PlayerAnswer(question_id=question_id, choice="y")


class _FakeQuestion:
    """Minimal stand-in for a Question ORM object (only needs .id)."""

    def __init__(self, id_: int):
        self.id = id_


class _FakeValuesResult:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def iterator(self):
        return iter(self.rows)


# ---------------------------------------------------------------------------
# Tests for select_liste
# ---------------------------------------------------------------------------


class TestSelectListeGeneratorExhaustion(unittest.TestCase):
    """
    Bug: liste_id_questions_deja_posees was a generator expression.
    The list comprehension iterates get_liste_questions_a_poser(); for each
    element the `if elem.id not in liste_id_questions_deja_posees` check scans
    the generator from its current position.

    When an element that is NOT already-asked appears first, `not in` scans
    the entire generator looking for it, exhausts it, and returns True.
    All subsequent elements then see an empty generator and also pass — even
    if they were already asked.

    Example: already_asked=[3,5,7], all_ids=[1,2,3,...,9]
      - elem=1: scan generator [3,5,7] for 1 -> not found, generator exhausted -> 1 included (correct)
      - elem=2: generator empty -> 2 not in [] -> True -> included (correct)
      - elem=3: generator empty -> 3 not in [] -> True -> included (BUG: 3 was already asked)
      - ...all remaining ids included regardless

    Fix: use a set comprehension so the set is built once, membership checks
    are O(1) and always correct.
    """

    def _run_select_liste(
        self, all_question_ids: list[int], already_asked_ids: list[int]
    ) -> list[int]:
        fake_questions = [_FakeQuestion(id_) for id_ in all_question_ids]
        already_asked = [_make_player_answer(qid) for qid in already_asked_ids]

        with patch.object(
            utilities, "get_liste_questions_a_poser", return_value=fake_questions
        ):
            return utilities.select_liste(already_asked)

    def test_already_asked_excluded_when_unasked_come_first(self):
        """
        Core regression test: already-asked ids that appear AFTER an unasked id
        in the question list must still be excluded.

        With the generator bug, elem=1 (unasked) exhausts the generator while
        scanning for 1, so ids 3, 5, 7 later see an empty generator and are
        incorrectly included.
        """
        # ids 1,2 come before the asked ids 3,5,7 in the list
        all_ids = list(range(1, 10))  # [1,2,3,4,5,6,7,8,9]
        already_asked_ids = [3, 5, 7]  # appear later in all_ids
        # Need enough un-asked ids to fill 20 - 3 = 17 slots
        all_ids = list(range(1, 25))  # [1..24], asked = [3,5,7]

        result = self._run_select_liste(all_ids, already_asked_ids)

        for qid in already_asked_ids:
            self.assertNotIn(
                qid,
                result,
                f"Question id {qid} was already asked but appeared in result "
                f"(generator exhaustion bug)",
            )

    def test_already_asked_excluded_interleaved(self):
        """Already-asked ids scattered through the list are all excluded."""
        all_ids = list(range(1, 40))
        already_asked_ids = [2, 8, 14, 20, 26]  # interleaved throughout

        result = self._run_select_liste(all_ids, already_asked_ids)

        for qid in already_asked_ids:
            self.assertNotIn(
                qid,
                result,
                f"Already-asked id {qid} leaked into result",
            )

    def test_correct_number_of_questions_returned(self):
        """select_liste should return exactly 20 - len(already_asked) ids."""
        n_already_asked = 5
        all_ids = list(range(1, 40))
        already_asked_ids = list(range(1, n_already_asked + 1))

        result = self._run_select_liste(all_ids, already_asked_ids)
        self.assertEqual(len(result), 20 - n_already_asked)


# ---------------------------------------------------------------------------
# Tests for find_similarities
# ---------------------------------------------------------------------------


class TestFindSimilaritiesReturnsIds(unittest.TestCase):
    """
    Before the fix, find_similarities returned closest_indices — raw numpy
    array indices into list(set(answer.item_id ...)).  After the fix it
    returns closest_character_ids, i.e. actual item_id values.
    """

    def _make_answers(self, item_ids: list[int]) -> list:
        answers = []
        for item_id in item_ids:
            a = MagicMock()
            a.item_id = item_id
            answers.append(a)
        return answers

    def test_returns_character_ids_not_indices(self):
        """
        Returned values must be actual item IDs, not row indices into the
        similarity matrix.
        """
        # Use item_ids that are far from 0-based indices so we can tell them apart
        item_ids = [101, 202, 303, 404, 505]
        answers = self._make_answers(item_ids)

        # Row 0 is most similar to rows 2 and 4 (values 0.9 and 0.8)
        sim_matrix = np.array(
            [
                [1.0, 0.1, 0.9, 0.2, 0.8],
                [0.1, 1.0, 0.3, 0.4, 0.2],
                [0.9, 0.3, 1.0, 0.1, 0.5],
                [0.2, 0.4, 0.1, 1.0, 0.3],
                [0.8, 0.2, 0.5, 0.3, 1.0],
            ],
            dtype=float,
        )

        result = find_similarities(
            sim_matrix, character_idx=0, answers=answers, num_closest_characters=3
        )

        item_id_set = set(item_ids)
        raw_only_indices = {
            0,
            1,
            2,
            3,
            4,
        } - item_id_set  # indices that are NOT also item IDs

        for val in result:
            self.assertIn(
                val,
                item_id_set,
                f"find_similarities returned {val!r} which is not an item ID — likely a raw numpy index",
            )
            self.assertNotIn(
                val,
                raw_only_indices,
                f"find_similarities returned raw index {val} instead of an item ID",
            )

    def test_return_length(self):
        """Returns num_closest_characters - 1 results (argsort slice [-n:-1])."""
        item_ids = [10, 20, 30, 40, 50, 60]
        answers = self._make_answers(item_ids)
        sim_matrix = np.random.rand(6, 6)
        np.fill_diagonal(sim_matrix, 1.0)

        result = find_similarities(
            sim_matrix, character_idx=0, answers=answers, num_closest_characters=4
        )
        self.assertEqual(len(result), 3)  # [-4:-1] gives 3 elements


class TestAutoencoderPlayerVector(unittest.TestCase):
    """Regression tests for the autoencoder gameplay input representation."""

    def test_builds_interleaved_binary_yes_no_vector(self):
        answers = [
            _PlayerAnswer(question_id=101, choice="y"),
            _PlayerAnswer(question_id=202, choice="n"),
        ]
        vector = utilities.build_autoencoder_player_vector(
            answers,
            questions_to_index={101: 0, 202: 1, 303: 2},
            input_size=6,
        )

        np.testing.assert_array_equal(
            vector,
            np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        )

    def test_pass_and_unknown_questions_stay_zero(self):
        answers = [
            _PlayerAnswer(question_id=101, choice="p"),
            _PlayerAnswer(question_id=999, choice="y"),
        ]
        vector = utilities.build_autoencoder_player_vector(
            answers,
            questions_to_index={101: 0},
            input_size=2,
        )

        np.testing.assert_array_equal(
            vector,
            np.array([[0.0, 0.0]], dtype=np.float32),
        )


class TestAutoencoderExpectedMatrix(unittest.TestCase):
    """The trained target must match the benchmark/game yes/no threshold."""

    def test_expected_matrix_uses_count_threshold_not_tfidf(self):
        rows = [
            {"item_id": 10, "question_id": 101, "yes_count": 10, "no_count": 5},
            {"item_id": 10, "question_id": 202, "yes_count": 5, "no_count": 10},
            {"item_id": 20, "question_id": 101, "yes_count": 10, "no_count": 9},
        ]

        with patch.object(ae_datas, "MINIMUM_RATIO_VALID_QUESTION", 0.75), patch.object(
            ae_datas.Answer.objects,
            "values",
            return_value=_FakeValuesResult(rows),
        ):
            matrix = ae_datas.create_expected_output_matrix(
                nb_characters=2,
                nb_questions=2,
                characters_to_index={10: 0, 20: 1},
                questions_to_index={101: 0, 202: 1},
            )

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
        )

    def test_mask_includes_known_pass_answers(self):
        rows = [
            {"item_id": 10, "question_id": 101},
            {"item_id": 20, "question_id": 202},
        ]

        with patch.object(
            ae_datas.Answer.objects,
            "values",
            return_value=_FakeValuesResult(rows),
        ):
            matrix = ae_datas.create_answer_mask_matrix(
                nb_characters=2,
                nb_questions=2,
                characters_to_index={10: 0, 20: 1},
                questions_to_index={101: 0, 202: 1},
            )

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [[1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 1.0]],
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
