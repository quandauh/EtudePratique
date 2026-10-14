from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

from admin.utils.tree import settings
from admin.utils.tree.node import Node, NodeType
from app.models import Item


class LinearTreeBuilder:
    """
    Decision-tree generator that favours balanced splits.
    Every node selects the question that maximises the balance between
    expected YES and NO answers within the active candidate set.
    """

    def __init__(
        self, max_depth: int, progress: Optional[Callable[[str], None]] = None
    ):
        self.max_depth = min(max_depth, settings.MAX_TREE_DEPTH)

        self.characters: List[Item] = list(
            Item.objects.prefetch_related("answers_set").order_by("id")
        )
        self.character_count: int = len(self.characters)
        self.all_mask: int = (
            (1 << self.character_count) - 1 if self.character_count else 0
        )

        self.question_yes_masks: Dict[int, int] = defaultdict(int)
        self.question_no_masks: Dict[int, int] = defaultdict(int)
        self.question_pass_masks: Dict[int, int] = defaultdict(int)
        self.question_answered_masks: Dict[int, int] = defaultdict(int)
        self.question_ids: List[int] = []

        # Progress callback
        self._notify: Callable[[str], None]
        if progress is not None:
            self._notify = progress
        else:
            self._notify = lambda _msg: None
        self.nodes_created: int = 0

        self._precompute_answer_data()

    def build(self) -> Node:
        if self.character_count == 0:
            raise ValueError("Cannot build a decision tree with no characters.")

        self._notify(
            f"Pré-calcul terminé: {self.character_count} personnages, {len(self.question_ids)} questions"
        )

        root_selection = self._select_best_question(self.all_mask, used_questions=set())
        if root_selection is None:
            raise ValueError("Unable to select a root question for the decision tree.")

        root_question, root_partition = root_selection
        self._notify(f"Racine sélectionnée: question #{root_question}")
        root = Node(parent=None, depth=1, question_id=root_question, type=NodeType.ROOT)

        used_questions: Set[int] = {root_question}
        self._expand_node(
            node=root,
            mask=self.all_mask,
            used_questions=used_questions,
            depth=1,
            partition=root_partition,
        )

        return root

    def _precompute_answer_data(self) -> None:
        question_ids_set: Set[int] = set()

        self._notify("Pré-calcul des réponses par personnage...")
        for idx, character in enumerate(self.characters):
            for answer in character.answers_set.all():
                question_id = answer.question_id
                question_ids_set.add(question_id)
                self.question_answered_masks[question_id] |= 1 << idx

                total = answer.yes_count + answer.no_count + answer.pass_count

                if total == 0:
                    self.question_pass_masks[question_id] |= 1 << idx
                    continue

                yes_avg = answer.yes_count / total
                no_avg = answer.no_count / total
                pass_avg = answer.pass_count / total

                yes_threshold = yes_avg >= settings.MINIMUM_RATIO_VALID_QUESTION
                no_threshold = no_avg >= settings.MINIMUM_RATIO_VALID_QUESTION
                pass_dominant = pass_avg > max(yes_avg, no_avg)

                if (
                    pass_dominant
                    or (yes_threshold and no_threshold)
                    or (not yes_threshold and not no_threshold)
                ):
                    self.question_pass_masks[question_id] |= 1 << idx
                elif yes_threshold:
                    self.question_yes_masks[question_id] |= 1 << idx
                elif no_threshold:
                    self.question_no_masks[question_id] |= 1 << idx
                else:
                    self.question_pass_masks[question_id] |= 1 << idx
            if (idx + 1) % 200 == 0:
                self._notify(
                    f"Pré-calcul: {idx + 1}/{self.character_count} personnages traités"
                )

        self.question_ids = sorted(question_ids_set)

        # Characters without an explicit answer are treated as "pass".
        for question_id in self.question_ids:
            answered_mask = self.question_answered_masks.get(question_id, 0)
            missing_mask = self.all_mask & ~answered_mask
            if missing_mask:
                self.question_pass_masks[question_id] |= missing_mask
                self.question_yes_masks[question_id] |= missing_mask
                self.question_no_masks[question_id] |= missing_mask

    def _expand_node(
        self,
        node: Node,
        mask: int,
        used_questions: Set[int],
        depth: int,
        partition: Optional[Dict[str, int]] = None,
    ) -> None:
        if depth >= self.max_depth or mask.bit_count() <= 1:
            return

        partition = partition or self._partition(mask, node.question_id)

        branches: List[Tuple[NodeType, int]] = [
            (NodeType.BRANCH_YES, partition["yes_only_mask"]),
            (NodeType.BRANCH_NO, partition["no_only_mask"]),
            (NodeType.BRANCH_PASS, partition["pass_mask"]),
        ]

        for branch_type, child_mask in branches:
            if child_mask == 0 or child_mask.bit_count() <= 1:
                continue

            selection = self._select_best_question(child_mask, used_questions)
            if selection is None:
                continue

            child_question, child_partition = selection
            child_node = Node(
                parent=node,
                depth=node.depth + 1,
                question_id=child_question,
                type=branch_type,
            )

            self.nodes_created += 1
            if self.nodes_created % 50 == 0:
                self._notify(
                    f"Construction de l'arbre: {self.nodes_created} noeuds créés (profondeur {child_node.depth})"
                )

            next_used = set(used_questions)
            next_used.add(child_question)

            self._expand_node(
                node=child_node,
                mask=child_mask,
                used_questions=next_used,
                depth=depth + 1,
                partition=child_partition,
            )

    def _select_best_question(
        self,
        mask: int,
        used_questions: Set[int],
    ) -> Optional[Tuple[int, Dict[str, int]]]:
        if mask.bit_count() <= 1:
            return None

        # Score tuple ordering (higher is better):
        # 1) max min(yes_only, no_only) to push towards 50/50
        # 2) max coverage (yes_only + no_only)
        # 3) min absolute difference |yes_only - no_only|
        # 4) min pass_count
        # 5) deterministic: higher question_id last (as negative)
        best_score: Optional[Tuple[int, int, int, int, int]] = None
        best_question: Optional[int] = None
        best_partition: Optional[Dict[str, int]] = None

        for question_id in self.question_ids:
            if question_id in used_questions:
                continue

            partition = self._partition(mask, question_id)

            yes_only = partition["yes_only_mask"].bit_count()
            no_only = partition["no_only_mask"].bit_count()
            pass_count = partition["pass_mask"].bit_count()

            coverage = yes_only + no_only
            balance = min(yes_only, no_only)
            diff = abs(yes_only - no_only)

            if yes_only == 0 or no_only == 0:
                score = (
                    -1,  # invalid split indicator
                    coverage,  # prefer more coverage
                    -pass_count,  # fewer passes is better
                    -diff,  # smaller diff is better
                    -question_id,
                )
            else:
                score = (
                    balance,  # maximize min(yes_only, no_only)
                    coverage,  # maximize coverage used for the split
                    -diff,  # minimize difference
                    -pass_count,  # minimize pass
                    -question_id,
                )

            if best_score is None or score > best_score:
                best_score = score
                best_question = question_id
                best_partition = partition

        if best_question is None or best_partition is None:
            return None

        return best_question, best_partition

    def _partition(self, mask: int, question_id: int) -> Dict[str, int]:
        yes_mask_raw = self.question_yes_masks.get(question_id, 0) & mask
        no_mask_raw = self.question_no_masks.get(question_id, 0) & mask
        pass_mask_raw = self.question_pass_masks.get(question_id, 0) & mask

        pass_mask = pass_mask_raw

        yes_mask = yes_mask_raw & ~pass_mask_raw
        no_mask = no_mask_raw & ~pass_mask_raw

        yes_mask &= ~no_mask_raw
        no_mask &= ~yes_mask_raw

        remainder = mask & ~(yes_mask | no_mask | pass_mask)
        if remainder:
            pass_mask |= remainder

        yes_only_mask = yes_mask
        no_only_mask = no_mask

        return {
            "yes_mask": yes_mask,
            "no_mask": no_mask,
            "pass_mask": pass_mask,
            "yes_only_mask": yes_only_mask,
            "no_only_mask": no_only_mask,
        }


def create_tree_linear(
    max_depth: int = 3, progress: Optional[Callable[[str], None]] = None
) -> Node:
    builder = LinearTreeBuilder(max_depth=max_depth, progress=progress)
    return builder.build()
