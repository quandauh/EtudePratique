import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


src_dir = os.path.join(os.path.dirname(__file__), "..")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


_DJANGO_STUBS = [
    "django",
    "django.db",
    "django.db.models",
    "django.db.transaction",
    "django.db.models.base",
    "django.shortcuts",
]

for _name in _DJANGO_STUBS:
    sys.modules.setdefault(_name, MagicMock())


_APP_MODEL_STUBS = [
    "app.models",
    "app.models.Tree",
]

for _name in _APP_MODEL_STUBS:
    sys.modules.setdefault(_name, MagicMock())


from admin.utils.generate_tree_json import build_tree_export_data  # noqa: E402


class TestGenerateTreeJson(unittest.TestCase):
    def test_build_tree_export_data_preserves_shape_and_branch_counts(self) -> None:
        app_tree = [
            (1, None, "", 10, "Root question"),
            (2, 1, "y", 20, "Yes child"),
            (3, 1, "n", 30, "No child"),
        ]
        item_order = [(1, "Alpha"), (2, "Beta"), (3, "Gamma")]
        question_order = [
            (10, "Root question"),
            (20, "Yes child"),
            (30, "No child"),
        ]
        answers = [
            (10, 1, 1.0, 0.0, 4, 1),
            (20, 1, 0.9, 0.1, 3, 1),
            (30, 1, 0.2, 0.8, 1, 3),
            (10, 2, 0.0, 1.0, 1, 4),
            (20, 2, 0.1, 0.9, 1, 3),
            (30, 2, 0.8, 0.2, 3, 1),
            (10, 3, 0.8, 0.2, 4, 1),
            (20, 3, 0.7, 0.3, 3, 1),
            (30, 3, 0.3, 0.7, 1, 3),
        ]

        with (
            patch(
                "admin.utils.generate_tree_json.extrait_app_tree",
                return_value=iter(app_tree),
            ),
            patch(
                "admin.utils.generate_tree_json.fetch_items_with_answers",
                return_value=item_order,
            ),
            patch(
                "admin.utils.generate_tree_json.fetch_questions_with_answers",
                return_value=question_order,
            ),
            patch("admin.utils.generate_tree_json.fetch_answers", return_value=answers),
            patch(
                "admin.utils.generate_tree_json.exemples",
                side_effect=lambda tfidf: list(np.argsort(np.sum(tfidf, axis=1))[:3]),
            ),
        ):
            export_data = build_tree_export_data(4)

        self.assertEqual([node["id"] for node in export_data], [1, 2, 3])

        root_node = export_data[0]
        yes_node = export_data[1]
        no_node = export_data[2]

        self.assertEqual(root_node["persos"], [])
        self.assertIsNone(root_node["parentId"])
        self.assertEqual(root_node["itemCount"], 3)

        self.assertEqual(yes_node["itemCount"], 2)
        self.assertEqual(no_node["itemCount"], 1)
        self.assertEqual(sorted(yes_node["persos"]), ["", "Alpha", "Gamma"])
        self.assertEqual(sorted(no_node["persos"]), ["", "", "Beta"])


if __name__ == "__main__":
    unittest.main()
