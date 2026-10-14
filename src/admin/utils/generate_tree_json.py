from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from django.db import connection

from app.models import Tree

TreeRow = tuple[int, int | None, str, int, str]
AnswerRow = tuple[int, int, float | None, float | None, int | None, int | None]


@dataclass(frozen=True)
class TreeExportContext:
    children_by_node_id: dict[int, tuple[TreeRow | None, TreeRow | None]]
    question_index_by_id: dict[int, int]
    tfidf: np.ndarray
    count: np.ndarray
    item_names: np.ndarray


def fetch_items_with_answers() -> list[tuple[int, str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT app_item.id, app_item.name
            FROM app_item
            WHERE EXISTS (SELECT 1 FROM app_answer WHERE item_id = app_item.id)
            ORDER BY app_item.name
            """
        )
        return [tuple(row) for row in cursor.fetchall()]


def fetch_questions_with_answers() -> list[tuple[int, str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT app_question.id, app_question.title
            FROM app_question
            WHERE EXISTS (SELECT 1 FROM app_answer WHERE question_id = app_question.id)
            ORDER BY app_question.title
            """
        )
        return [tuple(row) for row in cursor.fetchall()]


def fetch_answers() -> list[AnswerRow]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                question_id,
                item_id,
                yes_tfidf,
                no_tfidf,
                yes_count,
                no_count
            FROM app_answer
            """
        )
        return [tuple(row) for row in cursor.fetchall()]


def extrait_app_tree(profondeur: int) -> Iterator[TreeRow]:
    """
    Récupération des nœuds de l'arbre de la profondeur souhaitée.
    """
    return (
        Tree.objects.filter(depth__lte=profondeur)
        .exclude(choice="p")
        .values_list("id", "parent_id", "choice", "question_id", "question__title")
        .order_by("id")
        .iterator()
    )


def create_binarytree(tree: Iterable[TreeRow]) -> list[TreeRow]:
    """
    Conserve uniquement les noeuds accessibles depuis la racine en ignorant les branches "p".
    """
    reachable_parent_ids: set[int] = set()
    nodes: list[TreeRow] = []
    for node in tree:
        node_id, parent_id, *_ = node
        if not nodes:
            nodes.append(node)
            reachable_parent_ids.add(node_id)
            continue
        if parent_id in reachable_parent_ids:
            nodes.append(node)
            reachable_parent_ids.add(node_id)
    return nodes


def distEuclidienne(perso: np.ndarray, moyen: np.ndarray) -> np.ndarray:
    dist = (perso - moyen) ** 2
    return np.sum(dist, axis=1)


def exemples(tfidf: np.ndarray) -> list[int]:
    """
    Renvoie les indices des trois personnages les plus proches de la moyenne.
    """
    moyen = np.mean(tfidf, axis=0)
    dist = distEuclidienne(tfidf, moyen)
    return list(np.argsort(dist)[:3])


def tfidf_and_count(
    answers: Iterable[AnswerRow],
    question: list[tuple[int, str]],
    item: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    question_count = len(question)
    item_count = len(item)
    tfidf = np.zeros((item_count, 2 * question_count), dtype=float)
    count = np.zeros((item_count, 2 * question_count), dtype=float)
    question_index_by_id = {
        question_id: index for index, (question_id, _) in enumerate(question)
    }
    item_index_by_id = {item_id: index for index, (item_id, _) in enumerate(item)}

    for question_id, item_id, yes_tfidf, no_tfidf, yes_count, no_count in answers:
        item_index = item_index_by_id.get(item_id)
        question_index = question_index_by_id.get(question_id)
        if item_index is None or question_index is None:
            continue
        column_index = 2 * question_index
        tfidf[item_index, column_index] = 0.0 if yes_tfidf is None else yes_tfidf
        tfidf[item_index, column_index + 1] = 0.0 if no_tfidf is None else no_tfidf
        count[item_index, column_index] = 0.0 if yes_count is None else yes_count
        count[item_index, column_index + 1] = 0.0 if no_count is None else no_count

    return tfidf, count


def _build_children_by_node_id(
    app_tree: Iterable[TreeRow],
) -> dict[int, tuple[TreeRow | None, TreeRow | None]]:
    children_by_node_id: dict[int, list[TreeRow | None]] = defaultdict(
        lambda: [None, None]
    )
    for node in app_tree:
        parent_id = node[1]
        if parent_id is None:
            continue
        if node[2] == "y":
            children_by_node_id[parent_id][0] = node
        elif node[2] == "n":
            children_by_node_id[parent_id][1] = node
    return {
        parent_id: (children[0], children[1])
        for parent_id, children in children_by_node_id.items()
    }


def _build_question_index_by_id(
    question_order: Iterable[tuple[int, str]],
) -> dict[int, int]:
    return {question_id: index for index, (question_id, _) in enumerate(question_order)}


def _filter_indices(
    count: np.ndarray,
    row_indices: np.ndarray,
    answer_column: int,
    threshold: float,
    *,
    zero_zero_value: float,
) -> np.ndarray:
    selected = count[row_indices]
    opposite_column = answer_column + 1 if answer_column % 2 == 0 else answer_column - 1
    opposite = selected[:, opposite_column]
    ratio = np.ones(selected.shape[0], dtype=float)
    valid_opposite = opposite != 0
    ratio[valid_opposite] = (
        selected[valid_opposite, answer_column] / opposite[valid_opposite]
    )
    ratio[~valid_opposite & (selected[:, answer_column] == 0)] = zero_zero_value
    return row_indices[ratio >= threshold]


def _extract_examples(
    tfidf: np.ndarray,
    item_names: np.ndarray,
    row_indices: np.ndarray,
) -> list[str]:
    if row_indices.size == 0:
        persos = ["Aucun personnage avec suffisament de parties"]
    else:
        selected_indices = row_indices[exemples(tfidf[row_indices])]
        persos = [
            str(item_names[index]).replace('"', "").strip()
            for index in selected_indices
        ]
    while len(persos) < 3:
        persos.append("")
    return persos


def _build_node_payload(
    node: TreeRow,
    item_count: int,
    persos: list[str],
    is_root: bool,
) -> dict[str, object]:
    node_id, parent_id, choice, _, question_title = node
    return {
        "questionTitle": question_title.replace('"', "").strip(),
        "itemCount": item_count,
        "persos": [] if is_root else persos,
        "id": node_id,
        "parentId": None if is_root else parent_id,
        "colorId": 0 if is_root else (1 if choice == "y" else 2),
    }


def _walk_tree(
    node: TreeRow,
    context: TreeExportContext,
    item_indices: np.ndarray,
    example_indices: np.ndarray,
    exported_nodes: list[dict[str, object]],
) -> None:
    node_id, _, _, question_id, _ = node
    is_root = node[1] is None
    payload = _build_node_payload(
        node=node,
        item_count=int(item_indices.size),
        persos=_extract_examples(context.tfidf, context.item_names, example_indices),
        is_root=is_root,
    )
    exported_nodes.append(payload)

    children = context.children_by_node_id.get(node_id)
    if children is None:
        return

    question_index = context.question_index_by_id[question_id]
    for child, answer_column in (
        (children[0], question_index * 2),
        (children[1], (question_index * 2) + 1),
    ):
        if child is None:
            continue

        next_item_indices = _filter_indices(
            context.count,
            item_indices,
            answer_column,
            threshold=1.0,
            zero_zero_value=1.0,
        )
        next_example_indices = _filter_indices(
            context.count,
            example_indices,
            answer_column,
            threshold=0.75,
            zero_zero_value=0.0,
        )
        _walk_tree(
            node=child,
            context=context,
            item_indices=next_item_indices,
            example_indices=next_example_indices,
            exported_nodes=exported_nodes,
        )


def build_tree_export_data(profondeur: int) -> list[dict[str, object]]:
    app_tree = create_binarytree(extrait_app_tree(profondeur))
    if not app_tree:
        return []

    item_order = fetch_items_with_answers()
    question_order = fetch_questions_with_answers()
    answers = fetch_answers()
    tfidf, count = tfidf_and_count(answers, question_order, item_order)
    context = TreeExportContext(
        children_by_node_id=_build_children_by_node_id(app_tree),
        question_index_by_id=_build_question_index_by_id(question_order),
        tfidf=tfidf,
        count=count,
        item_names=np.array([item_name for _, item_name in item_order], dtype=object),
    )
    exported_nodes: list[dict[str, object]] = []
    all_indices = np.arange(len(item_order), dtype=np.int32)
    _walk_tree(
        node=app_tree[0],
        context=context,
        item_indices=all_indices,
        example_indices=all_indices,
        exported_nodes=exported_nodes,
    )
    return exported_nodes


def ecritureData(profondeur: int) -> None:
    export_data = build_tree_export_data(profondeur)
    file = (
        Path(__file__).resolve().parent.parent.parent
        / f"static/data/tree_{profondeur}.json"
    )
    with file.open("w", encoding="utf-8") as output_file:
        json.dump(export_data, output_file, ensure_ascii=True, separators=(",", ":"))


if __name__ == "__main__":
    print(len(build_tree_export_data(4)))
