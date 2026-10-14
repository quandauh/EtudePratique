import imp
from re import I
import MySQLdb
import numpy as np
from functools import lru_cache
from app.models import Question, Item, Answer
from django.db.models import Count
from admin.utils.autoencoder import settings
from admin.utils.tree.settings import MINIMUM_RATIO_VALID_QUESTION
from django.db.models import Count, Q, Case, When, IntegerField


def dbConnect():
    try:
        db = MySQLdb.connect(
            host="localhost",  # Change login according to your db config
            user="devinsa",
            password="devinsa!",
            database="devinsa",
            port=3306,  # 3306 by default
        )
    except Exception as e:
        raise BaseException("CONNEXION BDD", e)

    return db.cursor()


def get_questions_count():
    return (
        Answer.objects.values("question_id")
        .annotate(q_group=Count("question_id"))
        .filter(q_group__gte=settings.MIN_QUESTION_ANSWER)
        .order_by()
        .count()
    )


def get_characters_count():
    return Item.objects.count()


"""
Créé une matrice 3D avec

matrice[i][j][k] : 
- i : indice du personnage
- j : indice de la réponse
- k : 0 pour tfid_yes ; 1 pour tfid_no

Cette représentation facilite le travail humain sur la matrice mais mal adapté aux algos d'IA
"""


def create_character_answers_matrix(
    nb_characters, nb_questions, characters_to_index, questions_to_index
):
    cursor = dbConnect()

    result = np.zeros((nb_characters, nb_questions, 2), dtype=np.float32)

    query = """
            SELECT question_id, item_id, yes_tfidf, no_tfidf 
            FROM app_answer
            ORDER BY item_id
            """
    cursor.execute(query)

    rows = cursor.fetchall()

    # Build index arrays for vectorized scatter assignment
    char_indices = []
    quest_indices = []
    yes_vals = []
    no_vals = []

    for answer in rows:
        question_id = int(answer[0])
        character_id = int(answer[1])
        if question_id not in questions_to_index:
            continue
        char_indices.append(characters_to_index[character_id])
        quest_indices.append(questions_to_index[question_id])
        yes_vals.append(answer[2])
        no_vals.append(answer[3])

    if char_indices:
        ci = np.array(char_indices, dtype=np.intp)
        qi = np.array(quest_indices, dtype=np.intp)
        result[ci, qi, 0] = yes_vals
        result[ci, qi, 1] = no_vals

    return result


def get_characters_to_index_table():
    result = {}
    cursor = dbConnect()
    cursor.execute("SELECT id FROM app_item ORDER BY id")
    res = cursor.fetchall()

    for i in range(len(res)):
        result[int(res[i][0])] = i

    return result


def get_questions_to_index_table():

    result = {}

    cursor = dbConnect()

    cursor.execute(
        f"""
            SELECT question_id FROM app_answer GROUP BY question_id HAVING COUNT(*) >= {settings.MIN_QUESTION_ANSWER} ORDER BY id
        """
    )
    res = cursor.fetchall()

    for i in range(len(res)):
        result[int(res[i][0])] = i

    return result


def flatten_matrix(matrix, nb_characters, input_size):

    result = np.zeros((nb_characters, input_size), dtype=np.float32)
    for i in range(len(matrix)):
        result[i] = matrix[i].flatten()
    return result


def reshape_matrix_to_hr(matrix, nb_characters, nb_questions):
    result = np.zeros((nb_characters, nb_questions, 2), dtype=np.float32)
    for i in range(len(matrix)):
        cpt_ind = 0
        for j in range(0, len(matrix[i]) - 1, 2):
            result[i][cpt_ind][0] = matrix[i][j]
            result[i][cpt_ind][1] = matrix[i][j + 1]
            cpt_ind += 1
    return result


def get_expected_output_matrix(matrix, nb_characters, nb_questions):

    result = np.zeros((nb_characters, nb_questions, 2), dtype=np.float32)

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            x = matrix[i][j]
            tfidf_yes = x[0]
            tfidf_no = x[1]
            treshold = (3 / 4) * tfidf_yes
            if abs(tfidf_yes - tfidf_no) < treshold:
                result[i][j][0] = 1.0
                result[i][j][1] = 1.0
            elif tfidf_yes > tfidf_no:
                result[i][j][0] = 1.0
                result[i][j][1] = 0.0
            elif tfidf_yes == tfidf_no:
                result[i][j][0] = 0.0
                result[i][j][1] = 0.0
            else:
                result[i][j][0] = 0.0
                result[i][j][1] = 1.0

    return result


def create_expected_output_matrix(
    nb_characters,
    nb_questions,
    characters_to_index,
    questions_to_index,
):
    """
    Create the binary answer matrix used by gameplay and benchmark:
    - yes_count * MINIMUM_RATIO_VALID_QUESTION > no_count => yes
    - no_count * MINIMUM_RATIO_VALID_QUESTION > yes_count => no
    - otherwise pass/unknown => both slots stay 0
    """
    result = np.zeros((nb_characters, nb_questions, 2), dtype=np.float32)

    for answer in Answer.objects.values(
        "item_id",
        "question_id",
        "yes_count",
        "no_count",
    ).iterator():
        item_id = answer["item_id"]
        question_id = answer["question_id"]
        if item_id not in characters_to_index or question_id not in questions_to_index:
            continue

        char_index = characters_to_index[item_id]
        question_index = questions_to_index[question_id]
        yes_count = answer["yes_count"]
        no_count = answer["no_count"]

        if yes_count * MINIMUM_RATIO_VALID_QUESTION > no_count:
            result[char_index][question_index][0] = 1.0
        elif no_count * MINIMUM_RATIO_VALID_QUESTION > yes_count:
            result[char_index][question_index][1] = 1.0

    return result


def create_answer_mask_matrix(
    nb_characters,
    nb_questions,
    characters_to_index,
    questions_to_index,
):
    """Mask known answer rows, including explicit pass/ambiguous answers."""
    mask = np.zeros((nb_characters, nb_questions, 2), dtype=np.float32)

    for answer in Answer.objects.values("item_id", "question_id").iterator():
        item_id = answer["item_id"]
        question_id = answer["question_id"]
        if item_id not in characters_to_index or question_id not in questions_to_index:
            continue
        mask[characters_to_index[item_id]][questions_to_index[question_id]][0] = 1.0
        mask[characters_to_index[item_id]][questions_to_index[question_id]][1] = 1.0

    return mask


def get_mask_from_matrix(matrix, nb_characters, nb_questions):

    mask = np.zeros((nb_characters, nb_questions, 2), dtype=np.float32)

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            x = matrix[i][j]
            if x[0] == 0.0 and x[1] == 0.0:
                mask[i][j][0] = 0.0
                mask[i][j][1] = 0.0
            else:
                mask[i][j][0] = 1.0
                mask[i][j][1] = 1.0

    return mask


def get_mask_from_vector(vector):
    size = len(vector)
    result = np.zeros(size, dtype=np.float32)

    for i in range(0, size, 2):
        c1 = vector[i]
        c2 = vector[i + 1]
        if c1 != 0 or c2 != 0:
            result[i] = 1
            result[i + 1] = 1

    return result


@lru_cache(maxsize=1)
def get_questions_by_suggestions_count():
    """
    Fonction appelée par la vue de la page des Suggestions pour obtenir les deux listes de questions, triées
    respectivement par nombres de suggestions oui et non, afin que les questions affichées dans les menus déroulants
    soient triées par nombre de suggestions associées.
    La fonction utilise le cache lru_cache afin de ne pas avoir à refaire cette requête à chaque rafraichissement
    de la page ; le cache est vidé lorsque des suggestions sont validées ou re-calculées, via le code :
    get_questions_by_suggestions_count.cache_clear()
    """
    questions = list(
        Question.objects.annotate(
            suggestion_yes_count=Count(
                Case(
                    When(suggestion__reponse="y", then=1),
                    output_field=IntegerField(),
                )
            ),
            suggestion_no_count=Count(
                Case(
                    When(suggestion__reponse="n", then=1),
                    output_field=IntegerField(),
                )
            ),
        ).filter(Q(suggestion_yes_count__gt=0) | Q(suggestion_no_count__gt=0))
    )

    # Separate questions into two lists based on reponse
    questions_yes = [
        question for question in questions if question.suggestion_yes_count > 0
    ]
    questions_yes = sorted(
        questions_yes, key=lambda q: q.suggestion_yes_count, reverse=True
    )

    questions_no = [
        question for question in questions if question.suggestion_no_count > 0
    ]
    questions_no = sorted(
        questions_no, key=lambda q: q.suggestion_no_count, reverse=True
    )

    return questions_yes, questions_no
