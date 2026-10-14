"""
Benchmark Query Vector Module

Functions to build query vectors for similarity benchmarking between TF-IDF methods.
Allows simulation of player queries based on random subsets of questions.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from math import sqrt
from random import Random, sample
from typing import Iterable

import numpy as np
import django
from django.conf import settings
from sklearn.metrics.pairwise import pairwise_distances

if not settings.configured:
    # Add the 'src' directory to the path so that 'app' and 'admin' can be imported
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    # Set up Django environment
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "devinsa.settings")
    django.setup()

from admin.utils.tree.settings import MINIMUM_RATIO_VALID_QUESTION
from app.models import Answer, Item, PlayerAnswer, Question, Tree
from app.utilities.utilities import get_liste_questions_a_poser, search_best_item, search_best_items


def build_query_vector_for_item(item_id: int, num_questions: int = 20, 
                                 total_questions: list = None,
                                 question_index: dict = None,
                                 ordered_questions: list = None) -> np.ndarray:
    """
    Construit un vecteur de requête Q pour un personnage (item) donné,
    simulant les réponses du joueur à num_questions de l'arbre de décision.
    
    Args:
        item_id: ID du personnage (Item)
        num_questions: Nombre de questions à simuler (défaut 20)
        total_questions: Liste des tous les question IDs (optionnel, calculé si None)
        question_index: Dict {question_id -> index_colonne} (optionnel)
        ordered_questions: Liste des question IDs dans l'ordre de l'arbre (optionnel)
    
    Returns:
        Q_vec: np.ndarray de shape (2*Q,) où Q = len(total_questions)
    """
    # 1. Récupérer les réponses du personnage cible X
    answers_X = Answer.objects.filter(item_id=item_id).select_related('question')
    
    if not answers_X.exists():
        raise ValueError(f"No answers found for item {item_id}")
    
    # 2. Si question_index non fourni, le construire
    if question_index is None:
        if total_questions is None:
            total_questions = list(Question.objects.values_list('id', flat=True))
        question_index = {qid: idx for idx, qid in enumerate(total_questions)}
    else:
        if total_questions is None:
            total_questions = list(question_index.keys())
    
    Q_dim = 2 * len(total_questions)
    Q_vec = np.zeros(Q_dim)
    
    # 3. Sélectionner num_questions aléatoires
    available_questions = [a.question_id for a in answers_X]
    if len(available_questions) < num_questions:
        sampled_questions = available_questions
        print(f"Warning: Only {len(available_questions)} questions available for item {item_id}, "
              f"using all instead of {num_questions}")
    else:
        sampled_questions = sample(available_questions, num_questions)
    
    # 4. Remplir Q_vec avec les réponses du personnage X
    for answer in answers_X.filter(question_id__in=sampled_questions):
        ques_idx = question_index[answer.question_id]
        
        # Déterminer si c'est un "yes" ou "no" majoritaire pour le personnage X
        if answer.yes_count > answer.no_count:
            # Réponse "oui" dominante → utiliser yes_tfidf
            Q_vec[ques_idx] = answer.yes_tfidf
        elif answer.no_count > answer.yes_count:
            # Réponse "non" dominante → utiliser no_tfidf
            Q_vec[ques_idx + len(total_questions)] = answer.no_tfidf
        else:
            # Égalité → utiliser la moyenne
            Q_vec[ques_idx] = answer.yes_tfidf / 2.0
            Q_vec[ques_idx + len(total_questions)] = answer.no_tfidf / 2.0
    
    return Q_vec


def compute_similarity_for_query(Q_vec: np.ndarray, combined_matrix: np.ndarray,
                                  metric: str = 'cosine') -> np.ndarray:
    """
    Calcule les distances entre Q_vec et tous les vecteurs de combined_matrix.
    
    Args:
        Q_vec: Vecteur de requête, shape (2*Q,)
        combined_matrix: Matrice des caractères, shape (n_characters, 2*Q)
        metric: 'cosine', 'euclidean', ou 'manhattan'
    
    Returns:
        distances: Array shape (n_characters,) avec distances de Q à chaque caractère
    """
    # Q_vec doit être reshape en (1, 2*Q) pour pairwise_distances
    Q_reshaped = Q_vec.reshape(1, -1)
    distances = pairwise_distances(Q_reshaped, combined_matrix, metric=metric)[0]
    
    return distances


@dataclass
class BenchmarkStats:
    total: int
    correct: int
    incorrect: int
    not_found: int
    accuracy: float
    top_k_accuracy: float
    avg_tree_depth: float
    avg_questions: float


@dataclass
class BenchmarkCase:
    item_id: int
    item_name: str
    answers: list[PlayerAnswer]
    tree_depth: int


def _build_answer_stats_lookup() -> dict[int, dict[int, tuple[int, int, float, float]]]:
    answer_lookup: dict[int, dict[int, tuple[int, int, float, float]]] = {}
    for answer in Answer.objects.values(
        "item_id",
        "question_id",
        "yes_count",
        "no_count",
        "yes_tfidf",
        "no_tfidf",
    ).iterator():
        item_answers = answer_lookup.setdefault(answer["item_id"], {})
        item_answers[answer["question_id"]] = (
            answer["yes_count"],
            answer["no_count"],
            float(answer["yes_tfidf"]),
            float(answer["no_tfidf"]),
        )
    return answer_lookup


def _build_games_lookup() -> dict[int, int]:
    return {item["id"]: item["games"] for item in Item.objects.values("id", "games").iterator()}


def _build_tree_index() -> tuple[dict[int, dict[str, int | None]], dict[tuple[int, str], int]]:
    nodes: dict[int, dict[str, int | None]] = {}
    children: dict[tuple[int, str], int] = {}
    for node in Tree.objects.values("id", "parent_id", "choice", "question_id", "depth").iterator():
        nodes[node["id"]] = node
        if node["parent_id"] is not None:
            children[(node["parent_id"], node["choice"])] = node["id"]
    if 1 not in nodes:
        raise ValueError("Tree root (pk=1) not found.")
    return nodes, children


def _answer_for_item(
    answer_lookup: dict[int, dict[int, tuple[int, int, float, float]]],
    item_id: int,
    question_id: int,
    min_ratio: float = MINIMUM_RATIO_VALID_QUESTION,
) -> str:
    item_answers = answer_lookup.get(item_id)
    if not item_answers:
        return "p"
    answer = item_answers.get(question_id)
    if not answer:
        return "p"
    yes_count, no_count, _yes_tfidf, _no_tfidf = answer
    if yes_count * min_ratio > no_count:
        return "y"
    if no_count * min_ratio > yes_count:
        return "n"
    return "p"


def _select_random_questions(
    eligible_question_ids: list[int],
    already_asked: set[int],
    count: int,
    rng: Random,
) -> list[int]:
    pool = [qid for qid in eligible_question_ids if qid not in already_asked]
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return pool
    return rng.sample(pool, count)


def simulate_game_for_item(
    item_id: int,
    answer_lookup: dict[int, dict[int, tuple[int, int, float, float]]],
    tree_nodes: dict[int, dict[str, int | None]],
    tree_children: dict[tuple[int, str], int],
    eligible_question_ids: list[int],
    num_questions: int = 20,
    min_ratio: float = MINIMUM_RATIO_VALID_QUESTION,
    rng: Random | None = None,
    max_mistakes: int = 0,
    max_skips: int = 0,
) -> tuple[list[PlayerAnswer], int]:
    rng = rng or Random()
    answers: list[PlayerAnswer] = []
    asked_ids: set[int] = set()
    tree_depth = 0
    tree_id: int | None = 1
    mistakes_left = max_mistakes
    skips_left = max_skips

    while tree_id is not None and len(answers) < num_questions:
        node = tree_nodes.get(tree_id)
        if not node or not node.get("question_id"):
            break
        question_id = node["question_id"]
        if question_id in asked_ids:
            break

        real_choice = _answer_for_item(answer_lookup, item_id, question_id, min_ratio=min_ratio)

        # Introduire un skip aléatoire
        if skips_left > 0 and real_choice != "p" and rng.random() < 0.3:
            choice = "p"
            skips_left -= 1
        # Introduire une erreur aléatoire
        elif mistakes_left > 0 and real_choice in ("y", "n") and rng.random() < 0.3:
            choice = "n" if real_choice == "y" else "y"
            mistakes_left -= 1
        else:
            choice = real_choice

        answers.append(PlayerAnswer(question_id, choice))
        asked_ids.add(question_id)
        tree_depth += 1
        tree_id = tree_children.get((tree_id, choice))

    remaining = num_questions - len(answers)
    if remaining > 0:
        random_questions = _select_random_questions(eligible_question_ids, asked_ids, remaining, rng)
        for question_id in random_questions:
            real_choice = _answer_for_item(answer_lookup, item_id, question_id, min_ratio=min_ratio)

            if skips_left > 0 and real_choice != "p" and rng.random() < 0.3:
                choice = "p"
                skips_left -= 1
            elif mistakes_left > 0 and real_choice in ("y", "n") and rng.random() < 0.3:
                choice = "n" if real_choice == "y" else "y"
                mistakes_left -= 1
            else:
                choice = real_choice

            answers.append(PlayerAnswer(question_id, choice))

    return answers, tree_depth


def get_benchmark_items(
    min_games: int = 0,
    limit: int | None = None,
    item_ids: Iterable[int] | None = None,
):
    items = Item.objects.all().order_by("id")
    if min_games > 0:
        items = items.filter(games__gte=min_games)
    if item_ids is not None:
        items = items.filter(id__in=list(item_ids))
    if limit is not None:
        items = items[:limit]
    return items


def generate_benchmark_dataset(
    *,
    num_questions: int = 20,
    min_games: int = 0,
    seed: int | None = 0,
    limit: int | None = None,
    item_ids: Iterable[int] | None = None,
    min_ratio: float = MINIMUM_RATIO_VALID_QUESTION,
    max_mistakes: int = 0,
    max_skips: int = 0,
    answer_stats_lookup: dict[int, dict[int, tuple[int, int, float, float]]] | None = None,
    tree_nodes: dict[int, dict[str, int | None]] | None = None,
    tree_children: dict[tuple[int, str], int] | None = None,
    eligible_question_ids: list[int] | None = None,
) -> list[BenchmarkCase]:
    answer_stats_lookup = answer_stats_lookup or _build_answer_stats_lookup()
    if tree_nodes is None or tree_children is None:
        tree_nodes, tree_children = _build_tree_index()
    eligible_question_ids = eligible_question_ids or list(get_liste_questions_a_poser().values_list("id", flat=True))

    dataset: list[BenchmarkCase] = []
    for item in get_benchmark_items(
        min_games=min_games,
        limit=limit,
        item_ids=item_ids,
    ).iterator():
        rng = Random(seed + item.id) if seed is not None else Random()
        answers, tree_depth = simulate_game_for_item(
            item_id=item.id,
            answer_lookup=answer_stats_lookup,
            tree_nodes=tree_nodes,
            tree_children=tree_children,
            eligible_question_ids=eligible_question_ids,
            num_questions=num_questions,
            min_ratio=min_ratio,
            rng=rng,
            max_mistakes=max_mistakes,
            max_skips=max_skips,
        )
        dataset.append(
            BenchmarkCase(
                item_id=item.id,
                item_name=item.name,
                answers=answers,
                tree_depth=tree_depth,
            )
        )
    return dataset


def serialize_benchmark_dataset(dataset: Iterable[BenchmarkCase]) -> list[dict[str, object]]:
    return [
        {
            "item_id": case.item_id,
            "item_name": case.item_name,
            "tree_depth": case.tree_depth,
            "answers": [
                {
                    "question_id": answer.question_id,
                    "choice": answer.choice,
                }
                for answer in case.answers
            ],
        }
        for case in dataset
    ]


def deserialize_benchmark_dataset(rows: Iterable[dict[str, object]]) -> list[BenchmarkCase]:
    dataset: list[BenchmarkCase] = []
    for row in rows:
        dataset.append(
            BenchmarkCase(
                item_id=int(row["item_id"]),
                item_name=str(row.get("item_name") or row["item_id"]),
                tree_depth=int(row.get("tree_depth") or 0),
                answers=[
                    PlayerAnswer(
                        question_id=int(answer["question_id"]),
                        choice=str(answer["choice"]),
                    )
                    for answer in row.get("answers", [])
                ],
            )
        )
    return dataset


def _database_value_from_stats(
    answer_stats: tuple[int, int, float, float],
    min_ratio: float,
) -> float:
    yes_count, no_count, yes_tfidf, no_tfidf = answer_stats
    is_yes = yes_count * min_ratio > no_count
    is_no = no_count * min_ratio > yes_count
    is_pass = is_yes and is_no
    if is_pass:
        return 0.0
    if is_yes:
        return yes_tfidf
    return -no_tfidf


def _search_best_items_fast(
    player_answers: list[PlayerAnswer],
    depth: int,
    num_suggestions: int,
    answer_stats_lookup: dict[int, dict[int, tuple[int, int, float, float]]],
    games_lookup: dict[int, int],
    min_ratio: float,
) -> list[int]:
    similarity_check: dict[int, float] = {}
    item_ids = games_lookup.keys()

    for item_id in item_ids:
        no_info_count = 0
        scalar_product = 0.0
        answer_norm = 0.0
        database_norm = 0.0
        answer_counter = 0
        item_answers = answer_stats_lookup.get(item_id, {})

        for player_answer in player_answers:
            answer_counter += 1
            answer_stats = item_answers.get(player_answer.question_id)
            if answer_stats is None:
                if answer_counter <= depth:
                    no_info_count += 1
                continue

            database_value = _database_value_from_stats(answer_stats, min_ratio)
            database_norm += abs(database_value)

            answer_value = (
                1 if player_answer.choice == "y" else
                -1 if player_answer.choice == "n" else
                0
            )
            answer_norm += abs(database_value) * (answer_value ** 2)
            scalar_product += database_value * answer_value

        similarity = 0.0 if scalar_product == 0 else (scalar_product / sqrt(answer_norm * database_norm))
        similarity = (similarity * (20 - no_info_count)) / 20
        similarity_check[item_id] = similarity

    result_set = sorted(
        similarity_check.items(),
        key=lambda x: (x[1], games_lookup.get(x[0], 0)),
        reverse=True,
    )
    result = result_set[: num_suggestions + 1]
    return [res[0] for res in result if res[1] > 0.0]


def _search_best_item_fast(
    player_answers: list[PlayerAnswer],
    depth: int,
    answer_stats_lookup: dict[int, dict[int, tuple[int, int, float, float]]],
    games_lookup: dict[int, int],
    min_ratio: float,
) -> int:
    suggestions = _search_best_items_fast(
        player_answers=player_answers,
        depth=depth,
        num_suggestions=1,
        answer_stats_lookup=answer_stats_lookup,
        games_lookup=games_lookup,
        min_ratio=min_ratio,
    )
    return suggestions[0] if suggestions else -1


def run_benchmark(
    num_questions: int = 20,
    min_games: int = 0,
    seed: int | None = 0,
    limit: int | None = None,
    top_k: int = 3,
    item_ids: Iterable[int] | None = None,
    min_ratio: float = MINIMUM_RATIO_VALID_QUESTION,
    use_fast_search: bool = False,
) -> BenchmarkStats:
    answer_stats_lookup = _build_answer_stats_lookup()
    games_lookup = _build_games_lookup() if use_fast_search else {}
    tree_nodes, tree_children = _build_tree_index()
    eligible_question_ids = list(get_liste_questions_a_poser().values_list("id", flat=True))

    items = Item.objects.all().order_by("id")
    if min_games > 0:
        items = items.filter(games__gte=min_games)
    if item_ids is not None:
        items = items.filter(id__in=list(item_ids))
    if limit is not None:
        items = items[:limit]

    total_count = items.count()
    total = 0
    correct = 0
    incorrect = 0
    not_found = 0
    top_k_hit = 0
    total_tree_depth = 0
    total_questions = 0

    last_print_time = time.time()
    for item in items.iterator():
        current_time = time.time()
        if current_time - last_print_time >= 10:
            print(f"Progress: {total}/{total_count} ({total/total_count*100:.1f}%)" if total_count else f"Progress: {total}")
            last_print_time = current_time

        rng = Random(seed + item.id) if seed is not None else Random()
        answers, tree_depth = simulate_game_for_item(
            item_id=item.id,
            answer_lookup=answer_stats_lookup,
            tree_nodes=tree_nodes,
            tree_children=tree_children,
            eligible_question_ids=eligible_question_ids,
            num_questions=num_questions,
            min_ratio=min_ratio,
            rng=rng,
        )
        total += 1
        total_tree_depth += tree_depth
        total_questions += len(answers)

        if use_fast_search:
            guessed_item_id = _search_best_item_fast(
                player_answers=answers,
                depth=tree_depth,
                answer_stats_lookup=answer_stats_lookup,
                games_lookup=games_lookup,
                min_ratio=min_ratio,
            )
        else:
            guessed_item_id = search_best_item(answers, depth=tree_depth)
        if guessed_item_id == -1:
            not_found += 1
        elif guessed_item_id == item.id:
            correct += 1
        else:
            incorrect += 1

        if top_k and top_k > 1:
            if use_fast_search:
                suggestions = _search_best_items_fast(
                    player_answers=answers,
                    depth=tree_depth,
                    num_suggestions=top_k - 1,
                    answer_stats_lookup=answer_stats_lookup,
                    games_lookup=games_lookup,
                    min_ratio=min_ratio,
                )
            else:
                suggestions = search_best_items(answers, depth=tree_depth, num_suggestions=top_k - 1)
            if item.id in suggestions:
                top_k_hit += 1
        else:
            if guessed_item_id == item.id:
                top_k_hit += 1

    accuracy = correct / total if total else 0
    top_k_accuracy = top_k_hit / total if total else 0
    avg_tree_depth = total_tree_depth / total if total else 0
    avg_questions = total_questions / total if total else 0

    return BenchmarkStats(
        total=total,
        correct=correct,
        incorrect=incorrect,
        not_found=not_found,
        accuracy=accuracy,
        top_k_accuracy=top_k_accuracy,
        avg_tree_depth=avg_tree_depth,
        avg_questions=avg_questions,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Devinsa benchmark simulation.")
    parser.add_argument("--num-questions", type=int, default=20, help="Total questions per game (default: 20)")
    parser.add_argument("--min-games", type=int, default=0, help="Minimum games played filter")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (use -1 for non-deterministic)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items processed")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k accuracy (default: 3)")
    parser.add_argument(
        "--item-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of item ids to benchmark",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=MINIMUM_RATIO_VALID_QUESTION,
        help="Threshold ratio for yes/no decision",
    )
    parser.add_argument(
        "--use-fast-search",
        action="store_true",
        help="Use in-memory search (faster, but does not exercise live search functions)",
    )
    return parser.parse_args()


def _format_stats(stats: BenchmarkStats) -> str:
    return (
        "Benchmark results:\n"
        f"  total: {stats.total}\n"
        f"  correct: {stats.correct}\n"
        f"  incorrect: {stats.incorrect}\n"
        f"  not_found: {stats.not_found}\n"
        f"  accuracy: {stats.accuracy:.4f}\n"
        f"  top_k_accuracy: {stats.top_k_accuracy:.4f}\n"
        f"  avg_tree_depth: {stats.avg_tree_depth:.2f}\n"
        f"  avg_questions: {stats.avg_questions:.2f}"
    )


def main() -> None:
    args = _parse_args()
    seed = None if args.seed == -1 else args.seed
    stats = run_benchmark(
        num_questions=args.num_questions,
        min_games=args.min_games,
        seed=seed,
        limit=args.limit,
        top_k=args.top_k,
        item_ids=args.item_ids,
        min_ratio=args.min_ratio,
        use_fast_search=args.use_fast_search,
    )
    print(_format_stats(stats))


if __name__ == "__main__":
    main()
