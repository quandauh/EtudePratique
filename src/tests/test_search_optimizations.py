"""
Unit tests for end-game search optimizations.

These tests are DB-free and rely on small fake managers so they can run in
sandboxed environments.
"""

import os
import sys
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch


src_dir = os.path.join(os.path.dirname(__file__), "..")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


@dataclass
class _PlayerAnswer:
    question_id: int
    choice: str


for _name in [
    "django",
    "django.db",
    "django.db.models",
    "django.db.transaction",
    "django.db.models.base",
    "django.shortcuts",
]:
    sys.modules[_name] = MagicMock()

for _name in [
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
]:
    sys.modules[_name] = MagicMock()

sys.modules["app.models"].PlayerAnswer = _PlayerAnswer
sys.modules["app.models.dataclasses"].PlayerAnswer = _PlayerAnswer

_tree_settings = MagicMock()
_tree_settings.MINIMUM_RATIO_VALID_QUESTION = 0.75
sys.modules["admin.utils.tree"] = MagicMock()
sys.modules["admin.utils.tree.settings"] = _tree_settings

import app.utilities.utilities as utilities  # noqa: E402


class _FakeValuesQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, **kwargs):
        if "question_id__in" in kwargs:
            allowed = set(kwargs["question_id__in"])
            return _FakeValuesQuery(
                [row for row in self.rows if row["question_id"] in allowed]
            )
        if "id__in" in kwargs:
            allowed = set(kwargs["id__in"])
            return _FakeValuesQuery([row for row in self.rows if row["id"] in allowed])
        return _FakeValuesQuery(self.rows)

    def iterator(self):
        return iter(self.rows)


class _FakeAnswerManager:
    def __init__(self, rows):
        self.rows = rows
        self.values_called = False
        self.last_filter_kwargs = None

    def values(self, *fields):
        self.values_called = True
        base_rows = [{field: row[field] for field in fields} for row in self.rows]
        manager = self

        class _TrackedQuery(_FakeValuesQuery):
            def filter(self, **kwargs):
                manager.last_filter_kwargs = kwargs
                return super().filter(**kwargs)

        return _TrackedQuery(base_rows)


class _FakeItemManager:
    def __init__(self, rows):
        self.rows = rows
        self.values_called = False
        self.last_filter_kwargs = None
        self.get_called = False

    def values(self, *fields):
        self.values_called = True
        base_rows = [{field: row[field] for field in fields} for row in self.rows]
        manager = self

        class _TrackedQuery(_FakeValuesQuery):
            def filter(self, **kwargs):
                manager.last_filter_kwargs = kwargs
                return super().filter(**kwargs)

        return _TrackedQuery(base_rows)

    def get(self, **kwargs):
        self.get_called = True
        raise AssertionError("Item.objects.get should not be used in optimized ranking")


class TestGenerateItemdictOptimized(unittest.TestCase):
    def test_returns_empty_without_player_answers_and_skips_query(self):
        fake_answers = _FakeAnswerManager(
            [
                {
                    "item_id": 1,
                    "question_id": 10,
                    "yes_count": 4,
                    "no_count": 0,
                    "yes_tfidf": 1.0,
                    "no_tfidf": 0.0,
                }
            ]
        )
        utilities.Answer.objects = fake_answers

        result = utilities.generate_itemdict([])

        self.assertEqual(result, {})
        self.assertFalse(fake_answers.values_called)

    def test_filters_to_asked_questions_and_encodes_values(self):
        fake_answers = _FakeAnswerManager(
            [
                {
                    "item_id": 1,
                    "question_id": 10,
                    "yes_count": 10,
                    "no_count": 0,
                    "yes_tfidf": 1.5,
                    "no_tfidf": 0.0,
                },
                {
                    "item_id": 1,
                    "question_id": 20,
                    "yes_count": 0,
                    "no_count": 10,
                    "yes_tfidf": 0.0,
                    "no_tfidf": 0.8,
                },
                {
                    "item_id": 2,
                    "question_id": 20,
                    "yes_count": 0,
                    "no_count": 5,
                    "yes_tfidf": 0.0,
                    "no_tfidf": 0.4,
                },
                {
                    "item_id": 2,
                    "question_id": 99,
                    "yes_count": 50,
                    "no_count": 0,
                    "yes_tfidf": 9.0,
                    "no_tfidf": 0.0,
                },
            ]
        )
        utilities.Answer.objects = fake_answers

        player_answers = [_PlayerAnswer(10, "y"), _PlayerAnswer(20, "n")]
        result = utilities.generate_itemdict(player_answers)

        self.assertEqual(
            set(fake_answers.last_filter_kwargs["question_id__in"]),
            {10, 20},
        )
        self.assertEqual(result[1][10], 1.5)
        self.assertEqual(result[1][20], -0.8)
        self.assertEqual(result[2][20], -0.4)
        self.assertNotIn(99, result.get(2, {}))


class TestSearchBestItemsOptimized(unittest.TestCase):
    def test_uses_games_tie_break_without_n_plus_one_get(self):
        fake_items = _FakeItemManager(
            [
                {"id": 1, "games": 4},
                {"id": 2, "games": 10},
                {"id": 3, "games": 100},
                {"id": 99, "games": 9999},
            ]
        )
        utilities.Item.objects = fake_items

        itemdict = {
            1: {10: 1.0},
            2: {10: 1.0},
            3: {10: -1.0},
        }
        with patch.object(utilities, "generate_itemdict", return_value=itemdict):
            result = utilities.search_best_items(
                [_PlayerAnswer(10, "y")],
                depth=1,
                num_suggestions=1,
            )

        self.assertEqual(result, [2, 1])
        self.assertEqual(set(fake_items.last_filter_kwargs["id__in"]), {1, 2, 3})
        self.assertFalse(fake_items.get_called)

    def test_returns_empty_without_player_answers(self):
        with patch.object(utilities, "generate_itemdict") as mocked_generate:
            result = utilities.search_best_items([], depth=1, num_suggestions=2)
        self.assertEqual(result, [])
        mocked_generate.assert_not_called()

    def test_returns_empty_when_no_positive_similarity(self):
        fake_items = _FakeItemManager([{"id": 1, "games": 4}])
        utilities.Item.objects = fake_items

        with patch.object(utilities, "generate_itemdict", return_value={1: {10: -2.0}}):
            result = utilities.search_best_items(
                [_PlayerAnswer(10, "y")],
                depth=1,
                num_suggestions=2,
            )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
