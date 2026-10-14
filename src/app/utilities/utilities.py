from collections import defaultdict

from difflib import SequenceMatcher
from heapq import nlargest
from math import log, sqrt, exp
from random import sample
from re import sub

from django.db import transaction
from django.db.models import Sum, QuerySet

from app.models import Answer, PlayerAnswer, AnswerStat, Item, Question, Report
from admin.utils.tree.settings import MINIMUM_RATIO_VALID_QUESTION
from enum import Enum

import numpy as np
import struct


def float_list_to_bin(floats):
    return struct.pack("f" * len(floats), *floats)


def bin_to_float_list(size, B):
    return struct.unpack(str(size) + "f", B)


def ratio(a: str, b: str) -> float:
    """
    Fonction de calcul de la similarité entre deux chaînes de caractères
    :param a: chaîne a
    :param b: chaîne b
    :return: ratio de similarité
    """
    return SequenceMatcher(lambda x: x == " ", a.lower(), b.lower()).ratio()


def enleve_description_personnage(name: str) -> str:
    """
    Supprime la description du personnage
    Emmanuel Macron (Président de la république) → Emmanuel Macron
    :param name: nom du personnage à traiter
    :return: nom du personnage nettoyé
    """
    return sub(r"\([^)]+\)", "", name)


def enleve_mots_generiques(title: str) -> str:
    """
    Fonction supprimant les mots génériques d'une question
    :param title: la question passée en paramètre
    :return: la question sans les mots génériques
    """
    return (
        title.replace("rapport avec", "")
        .replace("As-tu déjà", "")
        .replace("As-tu", "")
        .replace("Ton personnage", "")
        .replace("Est-ce que ton", "")
        .replace("Votre personnage", "")
        .replace("personnage", "")
        .replace("combat-il", "")
        .replace("est-il", "")
        .replace("a-t-il", "")
        .replace("votre ", "")
        .replace(" un ", "")
    )


def get_closest_questions(title: str) -> list[Question]:
    """
    Fonction de génération de la liste des questions similaires
    :param title: la question
    :return: la liste des 5 questions similaires
    """
    quest: dict[Question, float] = {
        q: r
        for q in Question.objects.iterator()
        if (r := ratio(enleve_mots_generiques(q.title), enleve_mots_generiques(title)))
        > 0.3
    }
    return sorted(quest.keys(), key=lambda elem: quest[elem], reverse=True)[:5]


def get_closest_items(name: str) -> list[Item]:
    """
    Fonction de génération de la liste des éléments similaires
    :param name: nom de personnage
    :return: liste des 7 personnages dont le nom est le plus similaire
    """
    items: dict[Item, float] = {
        i: r
        for i in Item.objects.iterator()
        if (
            r := ratio(
                enleve_description_personnage(i.name),
                enleve_description_personnage(name),
            )
        )
        > 0.4
    }
    return sorted(items.keys(), key=lambda elem: items[elem], reverse=True)[:7]


def get_liste_questions_a_poser() -> QuerySet[Question]:
    """
    Chargement des questions ayant au moins une réponse positive pour pouvoir aller jusqu'à 20
    :return: la liste des questions à poser
    """
    return (
        Question.objects.select_related()
        .annotate(y=Sum("answers_set__yes_count"))
        .filter(y__gte=1)
        .distinct()
    )


def select_liste(question_deja_posees: list[PlayerAnswer]) -> list[int]:
    """
    Récupération d'une liste de question à laquelle on retire les questions déjà posées
    :param question_deja_posees: liste de questions déjà posées
    :return: liste mélangée des id des questions n'étant pas déjà posées
    """
    liste_id_questions_deja_posees: set[int] = {
        tu.question_id for tu in question_deja_posees
    }
    liste_finale: list[int] = [
        elem.id
        for elem in get_liste_questions_a_poser()
        if elem.id not in liste_id_questions_deja_posees
    ]
    return sample(liste_finale, 20 - len(question_deja_posees))


def save_game(item: Item, answers_quest: list[PlayerAnswer], win: bool = False) -> None:
    """
    Fonction de sauvegarde d'un élément et de la liste des réponses associées
    :param item: personnage trouvé
    :param answers_quest: la liste des questions répondues
    :param win: booléen indiquant la victoire ou la défaite de la partie
    """
    for ans_quest in answers_quest:
        answer: Answer = Answer.objects.get_or_create(
            question_id=ans_quest.question_id, item=item
        )[0]
        if ans_quest.choice == "y":
            answer.yes_count += 1
        elif ans_quest.choice == "n":
            answer.no_count += 1
        else:
            answer.pass_count += 1
        answer.save()
    item.games += 1
    if win:
        item.wins += 1
    item.save()


def report_game(item: Item, answers: list[PlayerAnswer]) -> list[Report]:
    """
    Fonction de création du rapport de partie
    :param item: le personnage trouvé
    :param answers: la liste des réponses
    :return: le rapport associé à la partie
    """
    report: list[Report] = []
    for answer in answers:
        given: str = "Passe"
        if answer.choice == "y":
            given = "Oui"
        elif answer.choice == "n":
            given = "Non"

        expected: str = "?"
        try:
            ans: Answer = Answer.objects.get(
                item_id=item.id, question_id=answer.question_id
            )
            if ans.yes_count > ans.no_count:
                expected = "Oui"
            elif ans.yes_count < ans.no_count:
                expected = "Non"
        except Answer.DoesNotExist:
            pass
        report.append(
            Report(expected, given, Question.objects.get(id=answer.question_id).title)
        )
    return report


def count_perso_yes(question: Question, persoCount: int) -> int:
    answers = Answer.objects.filter(question_id=question.id)

    yes_count: int = 0
    no_count: int = 0

    for answer in answers:
        if answer.yes_count == 0 and answer.no_count == 0:
            continue

        if answer.yes_count * 0.75 >= answer.no_count:
            yes_count += 1

        if answer.no_count * 0.75 >= answer.yes_count:
            no_count += 1

    with transaction.atomic():
        for answer in answers:
            answer.yes_tfidf = log(2 - yes_count / persoCount, 2)
            answer.no_tfidf = log(2 - no_count / persoCount, 2)
            answer.save()


def indexation() -> None:
    """
    Fonction de mise à jour du poids des réponses Oui et Non
    Cette fonction doit être appelée avant la création de l'arbre
    """
    answers_dict: dict[int, AnswerStat] = defaultdict(lambda: AnswerStat(0, 0, 0, []))
    answers_list: list[Answer] = []

    documentCount = Item.objects.count()
    answer: Answer
    for answer in Answer.objects.iterator():
        # La réponse a été jouée ou non
        count: int = (
            1 if answer.yes_count + answer.no_count + answer.pass_count > 0 else 0
        )
        # Proportion de oui/non pour cette réponse
        total = answer.yes_count + answer.no_count
        yes_prop: float = answer.yes_count / total if total > 0 else 0
        no_prop: float = answer.no_count / total if total > 0 else 0

        answers_dict[answer.question_id] += AnswerStat(
            count, yes_prop, no_prop, [answer]
        )

    for answer_stat in answers_dict.values():
        # IDF basé sur la rareté moyenne de oui/non
        avg_yes = (
            answer_stat.mostly_yes_count / answer_stat.count
            if answer_stat.count > 0
            else 0
        )
        avg_no = (
            answer_stat.mostly_no_count / answer_stat.count
            if answer_stat.count > 0
            else 0
        )
        yes_idf: float = log(1 / (avg_yes + 1e-10), 2) if avg_yes < 1 else 0
        no_idf: float = log(1 / (avg_no + 1e-10), 2) if avg_no < 1 else 0
        yes_idf = max(0, yes_idf)
        no_idf = max(0, no_idf)

        question = Question.objects.filter(
            id=answer_stat.answers[0].question_id
        ).first()
        question.yes_idf = yes_idf
        question.no_idf = no_idf
        question.save()

        for answer in answer_stat.answers:
            yes_tf: float = answer.yes_count / (
                answer.yes_count + answer.no_count + answer.pass_count
            )
            no_tf: float = answer.no_count / (
                answer.yes_count + answer.no_count + answer.pass_count
            )

            answer.yes_tf = yes_tf
            answer.no_tf = no_tf

            answer.yes_tfidf = yes_tf * yes_idf
            answer.no_tfidf = no_tf * no_idf

            answers_list.append(answer)

    with transaction.atomic():
        Answer.objects.all().delete()
        Answer.objects.bulk_create(answers_list)

        for character in Item.objects.prefetch_related("answers_set").all():
            norme = 0
            for answer in character.answers_set.all():
                norme += (answer.yes_tfidf + answer.no_tfidf) ** 2

            character.vectorSize = norme
            character.save()


def generate_itemdict(answers: list[PlayerAnswer]) -> dict[int, dict[int, float]]:
    """
    Création d'un dictionnaire qui va contenir toutes les réponses de la base
    pour éviter les sous requêtes
    :param answers: liste des réponses du joueur : permet d'affiner le résultat
    :return: le dictionnaire d'item : liste de réponses filtré pour que seuls
        les items ayant une réponse de la liste de questions apparaissent
    """
    itemdict: dict[int, dict[int, float]] = {}
    question_ids = {x.question_id for x in answers}
    if not question_ids:
        return itemdict

    base_answers = Answer.objects.values(
        "item_id",
        "question_id",
        "yes_count",
        "no_count",
        "yes_tfidf",
        "no_tfidf",
    ).filter(question_id__in=question_ids)

    for answer in base_answers.iterator():
        item_id = answer["item_id"]
        question_id = answer["question_id"]
        yes_count = answer["yes_count"]
        no_count = answer["no_count"]
        yes_tfidf = float(answer["yes_tfidf"])
        no_tfidf = float(answer["no_tfidf"])

        isYes: bool = yes_count * MINIMUM_RATIO_VALID_QUESTION > no_count
        isNo: bool = no_count * MINIMUM_RATIO_VALID_QUESTION > yes_count
        isPass: bool = isYes and isNo

        itemdict.setdefault(item_id, {})[question_id] = (
            QuestionAnswer.ANSWER_PASS.value
            if isPass
            else yes_tfidf * QuestionAnswer.ANSWER_YES.value
            if isYes
            else no_tfidf * QuestionAnswer.ANSWER_NO.value
        )

    return itemdict


_ORIGINAL_GENERATE_ITEMDICT = generate_itemdict


class QuestionAnswer(Enum):
    ANSWER_NO = -1
    ANSWER_YES = 1
    ANSWER_PASS = 0


"""TODO
    Le calcul des coefficients TF.IDF n'ont pas de sens dans cette méthode
    de calcul de similarité.

    DF = Nombre de perso pour OUI/NON selon le cas
    IDF = 1/DF

    TF = 1 si en accord avec la base, 0 sinon // Adapter le coeff en fonction de l'ambiguité en cas d'erreur ?

    TF * IDF = 1/DF si en accord avec la base, 0 sinon

    Ensuite, on calcul sommes des scores TF * IDF de chaque réponse du joueur,
    et cela pour chaque personnage de la base.

    On sélectionne ensuite le plus grand score.

    Défaut : Si le joueur loupe une question avec un gros TF.IDF, décale de beaucoup
    la prédiction.
"""


def _score_items_from_answer_rows(
    player_answers: list[PlayerAnswer], depth=20
) -> list[tuple[int, float]]:
    answers_by_question: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    depth_answer_count = 0

    for answer_counter, player_answer in enumerate(player_answers, start=1):
        answer_value = (
            QuestionAnswer.ANSWER_YES.value
            if player_answer.choice == "y"
            else QuestionAnswer.ANSWER_NO.value
            if player_answer.choice == "n"
            else QuestionAnswer.ANSWER_PASS.value
        )
        is_depth_answer = answer_counter <= depth
        if is_depth_answer:
            depth_answer_count += 1
        answers_by_question[player_answer.question_id].append(
            (answer_value, is_depth_answer)
        )

    normalized_by_question = {
        question_id: (
            sum(answer_value for answer_value, _is_depth_answer in answer_values),
            sum(
                1
                for answer_value, _is_depth_answer in answer_values
                if answer_value != QuestionAnswer.ANSWER_PASS.value
            ),
            len(answer_values),
            sum(1 for _answer_value, is_depth_answer in answer_values if is_depth_answer),
        )
        for question_id, answer_values in answers_by_question.items()
    }
    if not normalized_by_question:
        return []

    answer_rows = Answer.objects.values(
        "item_id",
        "question_id",
        "yes_count",
        "no_count",
        "yes_tfidf",
        "no_tfidf",
    ).filter(question_id__in=normalized_by_question.keys())

    scores: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for answer in answer_rows.iterator():
        answer_sum, non_pass_count, answer_count, depth_count = normalized_by_question[
            answer["question_id"]
        ]
        yes_count = answer["yes_count"]
        no_count = answer["no_count"]
        yes_tfidf = float(answer["yes_tfidf"])
        no_tfidf = float(answer["no_tfidf"])

        is_yes = yes_count * MINIMUM_RATIO_VALID_QUESTION > no_count
        is_no = no_count * MINIMUM_RATIO_VALID_QUESTION > yes_count
        is_pass = is_yes and is_no
        database_value = (
            QuestionAnswer.ANSWER_PASS.value
            if is_pass
            else yes_tfidf * QuestionAnswer.ANSWER_YES.value
            if is_yes
            else no_tfidf * QuestionAnswer.ANSWER_NO.value
        )
        abs_database_value = abs(database_value)

        item_score = scores[answer["item_id"]]
        item_score[0] += database_value * answer_sum
        item_score[1] += abs_database_value * non_pass_count
        item_score[2] += abs_database_value * answer_count
        item_score[3] += depth_count

    scored_items: list[tuple[int, float]] = []
    for item_id, (scalar_product, answer_norm, database_norm, known_depth_answers) in scores.items():
        similarity = (
            0
            if scalar_product == 0
            else (scalar_product / sqrt(answer_norm * database_norm))
        )
        no_info_count = depth_answer_count - known_depth_answers
        similarity = (similarity * (20 - no_info_count)) / 20
        if similarity > 0.0:
            scored_items.append((item_id, similarity))

    return scored_items


def search_best_items(
    player_answers: list[PlayerAnswer], depth=20, num_suggestions=3
) -> list[int]:
    """
    This function searches for the most probable items the player might be thinking of.
    :param player_answers: list of player's answers
    :param depth: depth of the search
    :param num_suggestions: number of suggestions to return
    :return: list of item IDs representing suggested items
    """
    if not player_answers:
        return []

    if generate_itemdict is _ORIGINAL_GENERATE_ITEMDICT:
        scored_items = _score_items_from_answer_rows(player_answers, depth=depth)
        if not scored_items:
            return []
        item_ids = [item_id for item_id, _similarity in scored_items]
    else:
        itemdict: dict[int, dict[int, float]] = generate_itemdict(player_answers)
        if not itemdict:
            return []
        item_ids = list(itemdict.keys())

        normalized_answers = [
            (
                answer_counter,
                player_answer.question_id,
                QuestionAnswer.ANSWER_YES.value
                if player_answer.choice == "y"
                else QuestionAnswer.ANSWER_NO.value
                if player_answer.choice == "n"
                else QuestionAnswer.ANSWER_PASS.value,
            )
            for answer_counter, player_answer in enumerate(player_answers, start=1)
        ]

        scored_items = []
        answers_by_question: dict[int, list[tuple[int, bool]]] = defaultdict(list)
        depth_answer_count = 0
        for answer_counter, question_id, answer_value in normalized_answers:
            is_depth_answer = answer_counter <= depth
            if is_depth_answer:
                depth_answer_count += 1
            answers_by_question[question_id].append((answer_value, is_depth_answer))

        for item_id, database_answers in itemdict.items():
            scalarProduct = 0
            answerNorm = 0
            databaseNorm = 0
            known_depth_answers = 0

            for question_id, database_value in database_answers.items():
                answer_values = answers_by_question.get(question_id)
                if not answer_values:
                    continue

                abs_database_value = abs(database_value)
                databaseNorm += abs_database_value * len(answer_values)

                for answer_value, is_depth_answer in answer_values:
                    if is_depth_answer:
                        known_depth_answers += 1
                    if answer_value != QuestionAnswer.ANSWER_PASS.value:
                        answerNorm += abs_database_value

                    scalarProduct += database_value * answer_value

            similarity = (
                0
                if scalarProduct == 0
                else (scalarProduct / sqrt(answerNorm * databaseNorm))
            )
            no_info_count = depth_answer_count - known_depth_answers
            similarity = (similarity * (20 - no_info_count)) / 20
            if similarity > 0.0:
                scored_items.append((item_id, similarity))

    games_lookup = {
        item["id"]: item["games"]
        for item in Item.objects.values("id", "games").filter(id__in=item_ids).iterator()
    }

    if not scored_items:
        return []

    best_items = nlargest(
        num_suggestions + 1,
        scored_items,
        key=lambda x: (x[1], games_lookup.get(x[0], 0)),
    )

    return [item_id for item_id, _similarity in best_items]


def search_best_item(player_answers: list[PlayerAnswer], depth=20) -> int:
    suggestions = search_best_items(player_answers, depth=depth, num_suggestions=1)
    return suggestions[0] if suggestions else -1


def generate_itemdict_no_weight(
    answers: list[PlayerAnswer],
) -> dict[int, list[dict[str, int]]]:
    """
    Création d'un dictionnaire qui va contenir toutes les réponses de la base
    pour éviter les sous requêtes
    :param answers: liste des réponses du joueur : permet d'affiner le résultat
    :return: le dictionnaire d'item : liste de réponses filtré pour que seuls
    les items ayant une réponse de la liste de questions apparaissent
    """
    itemdict: dict[int, dict[int, float]] = {}

    characters = Item.objects.prefetch_related("answers_set").all()

    questionList = [x.question_id for x in answers]

    for character in characters:
        filtered_answers = filter(
            lambda x: x.question_id in questionList, character.answers_set.all()
        )

        answers_dict = {}
        for answer in filtered_answers:
            isYes: bool = (
                answer.yes_count * MINIMUM_RATIO_VALID_QUESTION > answer.no_count
            )
            isNo: bool = (
                answer.no_count * MINIMUM_RATIO_VALID_QUESTION > answer.yes_count
            )
            isPass: bool = isYes and isNo

            answers_dict[answer.question_id] = (
                QuestionAnswer.ANSWER_PASS.value
                if isPass
                else QuestionAnswer.ANSWER_YES.value
                if isYes
                else QuestionAnswer.ANSWER_NO.value
            )

        itemdict[character.id] = answers_dict

    return itemdict


def search_best_item_no_weight(player_answers: list[PlayerAnswer], depth=20) -> int:
    depth = len(player_answers)
    """
    Cette fonction recherche l'item le plus probable auquel pense le joueur
    :param player_choices: liste des réponses du joueur
    :return: l'id du personnage le plus probable auquel pense le joueur
    """
    similarity_check: dict[int, float] = {}
    itemdict: dict[int, dict[int, float]] = generate_itemdict_no_weight(player_answers)

    # Parcourt l'ensemble des items afin de calculer la similarité de chacun avec l'item auquel pense le joueur
    for item_id in itemdict.keys():
        no_info_count = 0

        scalarProduct = 0
        answerNorm = 0
        databaseNorm = 0

        answerCounter = 0

        for player_answer in player_answers:
            answerCounter += 1

            database_answers = itemdict[item_id]

            # Si aucune info sur la question on skip
            if not player_answer.question_id in database_answers:
                if answerCounter <= depth:
                    no_info_count += 1
                continue

            database_value = database_answers[player_answer.question_id]
            databaseNorm += database_value**2

            answer_value = (
                QuestionAnswer.ANSWER_YES.value
                if player_answer.choice == "y"
                else QuestionAnswer.ANSWER_NO.value
                if player_answer.choice == "n"
                else QuestionAnswer.ANSWER_PASS.value
            )
            answerNorm += answer_value**2

            scalarProduct += database_value * answer_value

        similarity_check[item_id] = (
            0
            if scalarProduct == 0
            else (scalarProduct / sqrt(answerNorm * databaseNorm))
        )
        similarity_check[item_id] = (
            (similarity_check[item_id]) * (20 - no_info_count)
        ) / 20

    result_set = sorted(
        similarity_check.items(),
        key=lambda x: (x[1], Item.objects.get(pk=x[0]).games),
        reverse=True,
    )
    # print(result_set[:10])
    result = result_set[0]

    return result[0] if result[1] > 0.0 else -1


def generate_itemdict_fullvector(
    answers: list[PlayerAnswer],
) -> dict[int, list[dict[str, int]]]:
    """
    Création d'un dictionnaire qui va contenir toutes les réponses de la base
    pour éviter les sous requêtes
    :param answers: liste des réponses du joueur : permet d'affiner le résultat
    :return: le dictionnaire d'item : liste de réponses filtré pour que seuls
        les items ayant une réponse de la liste de questions apparaissent
    """
    itemdict: dict[int, list[dict[str, int]]] = defaultdict(list)
    base_answers: QuerySet[Answer] = Answer.objects.values(
        "item_id", "question_id", "yes_tfidf", "no_tfidf"
    )
    base_answers = base_answers.filter(question_id__in=(x.question_id for x in answers))
    answer: dict[str, int]

    for answer in base_answers.iterator():
        itemdict[answer["item_id"]].append(answer)

    return itemdict


def search_best_item_fullvector(player_choices: list[PlayerAnswer], depth=20) -> int:
    """
    Cette fonction recherche l'item le plus probable auquel pense le joueur
    :param player_choices: liste des réponses du joueur
    :return: l'id du personnage le plus probable auquel pense le joueur
    """
    similarity_check: list[tuple[int, float]] = []
    itemdict: dict[int, list[dict[str, int]]] = generate_itemdict_fullvector(
        player_choices
    )

    # Parcourt l'ensemble des items afin de calculer la similarité de chacun avec l'item auquel pense le joueur
    for item_id in itemdict.keys():
        character = Item.objects.prefetch_related("answers_set").get(pk=item_id)
        norm1 = 0
        norm2 = 631625
        prod_scal = 0

        # Parcourt les réponses du joueur afin de calculer la similarité
        for choice in player_choices:
            if choice.choice != "p":
                for answer in itemdict[item_id]:
                    if answer["question_id"] == choice.question_id:
                        if choice.choice == "o":
                            prod_scal += pow(answer["yes_tfidf"], 1)
                            norm1 += 1

                        elif choice.choice == "n":
                            prod_scal += pow(answer["no_tfidf"], 1)
                            norm1 += 1

        if norm1 == 0 or norm2 == 0:
            continue

        similarity = prod_scal
        similarity_check.append((item_id, similarity))

    # Trie les tuples (item, similarité) par similarité décroissante
    similarity_check.sort(key=lambda tup: tup[1], reverse=True)

    if not similarity_check:
        return -1

    result = similarity_check[0]

    return result[0] if result[1] > 0.0 else -1


def build_autoencoder_player_vector(
    player_answers: list[PlayerAnswer],
    questions_to_index: dict[int, int],
    input_size: int,
) -> np.ndarray:
    """
    Build the sparse player vector used by the autoencoder.

    The matrix flattened by datas.flatten_matrix is interleaved by question:
    [q0_yes, q0_no, q1_yes, q1_no, ...]. Keep gameplay, benchmark and
    autoencoder training on that same binary representation.
    """
    player_vec = np.zeros((1, input_size), dtype=np.float32)
    for player_answer in player_answers:
        idx = questions_to_index.get(player_answer.question_id)
        if idx is None:
            continue
        if player_answer.choice == "y":
            player_vec[0][idx * 2] = 1.0
        elif player_answer.choice == "n":
            player_vec[0][idx * 2 + 1] = 1.0
    return player_vec


def _cosine_similarity_vec(a, b):
    # a: (n,) b: (m, n) -> return array of similarities
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b, axis=1)
    # avoid div by zero
    denom = a_norm * b_norm
    denom[denom == 0] = 1e-9
    sims = (b @ a) / denom
    return sims


def search_best_items_autoencoder(
    player_answers: list[PlayerAnswer],
    depth=20,
    num_suggestions=3,
    fallback_to_tfidf: bool = True,
    weights_file: str | None = None,
) -> list[int]:
    """
    Try to compute suggestions using the trained auto-encoder.
    Reads character vectors from the DB (CharacterVector table).
    Falls back to TF-IDF if anything fails.
    """
    try:
        from admin.utils.autoencoder.auto_encoder_utils import get_cached_session, resolve_weights_path
        from admin.utils.autoencoder import datas
        from app.models.AppSettings import AppSettings
        from app.models.CharacterVector import CharacterVector
        import numpy as np

        settings_model, _ = AppSettings.objects.get_or_create(pk=1)
        selected_weights = weights_file or settings_model.ae_weights_path or "auto_encoder_weights.h5"
        weights_path = resolve_weights_path(selected_weights)

        # Always need the session to encode the player's input
        session = get_cached_session(weights_file=weights_path)
        if session is None:
            if fallback_to_tfidf:
                return search_best_items(player_answers, depth=depth, num_suggestions=num_suggestions)
            raise RuntimeError("Autoencoder session unavailable.")

        # Prefer DB vectors (persist across restarts); fall back to in-memory
        rows = list(
            CharacterVector.objects.filter(weights_path=weights_path)
            .values("item_id", "vector")
        )
        if rows:
            item_ids = [r["item_id"] for r in rows]
            encoded_matrix = np.array([r["vector"] for r in rows], dtype=np.float32)
        else:
            encoded_matrix = getattr(session, "encoded_matrix", None)
            if encoded_matrix is None:
                raise RuntimeError("Autoencoder encoded matrix unavailable.")
            inv_map = {v: k for k, v in session.characters_to_index.items()}
            item_ids = [inv_map[i] for i in range(len(inv_map))]

        questions_to_index = datas.get_questions_to_index_table()
        questions_count = datas.get_questions_count()
        input_size = 2 * questions_count

        player_vec = build_autoencoder_player_vector(
            player_answers,
            questions_to_index,
            input_size,
        )

        player_encoded = session.autoEncoder.transform(player_vec)
        player_encoded_flat = player_encoded.flatten()
        encoded_matrix = np.asarray(encoded_matrix)
        if encoded_matrix.ndim != 2:
            raise RuntimeError(
                f"Autoencoder encoded matrix must be 2D, got shape {encoded_matrix.shape}."
            )
        if encoded_matrix.shape[1] != player_encoded_flat.shape[0]:
            raise RuntimeError(
                "Autoencoder vector size mismatch: "
                f"characters use {encoded_matrix.shape[1]} dimensions, "
                f"player vector uses {player_encoded_flat.shape[0]}."
            )

        sims = _cosine_similarity_vec(player_encoded_flat, encoded_matrix)
        idxs = np.argsort(sims)[::-1]
        return [item_ids[i] for i in idxs[: num_suggestions + 1]]
    except Exception:
        if fallback_to_tfidf:
            return search_best_items(player_answers, depth=depth, num_suggestions=num_suggestions)
        raise


def search_best_item_autoencoder(
    player_answers: list[PlayerAnswer],
    depth=20,
    fallback_to_tfidf: bool = True,
    weights_file: str | None = None,
) -> int:
    suggestions = search_best_items_autoencoder(
        player_answers,
        depth=depth,
        num_suggestions=1,
        fallback_to_tfidf=fallback_to_tfidf,
        weights_file=weights_file,
    )
    return suggestions[0] if suggestions else -1
