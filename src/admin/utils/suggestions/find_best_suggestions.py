import numpy as np
from app.models import Item, Question, Answer
from django.shortcuts import get_object_or_404


def find_best_suggestions(
    combined_matrix,
    character_idx,
    closest_character_ids,
    questions,
    question_index,
    characters,
    num_questions=5,
):
    """
    Find the best suggestions for the target character based on high TF-IDF questions.
    params:
    - combined_matrix: matrix of distances between characters
    - character_idx: index of the target character chosen by the user
    - closest_character_ids: IDs of the closest characters to the target character
    - questions: list of questions
    - question_index: dictionary mapping questions to their indices
    - characters: list of characters in the dataset
    - num_questions: number of high TF-IDF questions to consider
    """
    # Identify high TF-IDF questions for the target character
    target_yes_tfidf = combined_matrix[character_idx, : len(questions)]
    target_no_tfidf = combined_matrix[character_idx, len(questions) :]
    target_high_tfidf_indices = np.argsort(target_yes_tfidf + target_no_tfidf)[
        -num_questions:
    ]  # Top n high TF-IDF questions

    #### NOT USED BECAUSE IT SEEMS TO FALSIFY SOME SUGGESTIONS
    ## Identify high TF-IDF questions for the closest characters
    # closest_high_tfidf = {}
    # for idx in closest_indices:
    #     char_id = characters[idx]
    #     char_yes_tfidf = combined_matrix[idx, :len(questions)]
    #     char_no_tfidf = combined_matrix[idx, len(questions):]
    #     closest_high_tfidf_indices = np.argsort(char_yes_tfidf + char_no_tfidf)[-num_questions:]  # Top n high TF-IDF questions
    #     closest_high_tfidf[char_id] = closest_high_tfidf_indices

    ## Identify common high TF-IDF questions for the target character and closest characters
    # common_high_tfidf_indices = set(target_high_tfidf_indices)
    # for indices in closest_high_tfidf.values():
    #     common_high_tfidf_indices.intersection_update(indices)

    ## Combine high TF-IDF questions for the target character and closest characters
    # combined_high_tfidf_indices = set(target_high_tfidf_indices)
    # for indices in closest_high_tfidf.values():
    #     combined_high_tfidf_indices.update(indices)
    ######

    # Identify missing values for the combined high TF-IDF questions
    missing_values_combined = {}
    for char_id in closest_character_ids:
        missing_questions = []
        for ques_idx in target_high_tfidf_indices:
            # Check if the question has no answer for this character
            question_answered_before = Answer.objects.filter(
                item_id=char_id, question_id=questions[ques_idx]
            ).values_list("yes_count", "no_count")
            if not question_answered_before.exists():
                missing_questions.append(questions[ques_idx])
        if missing_questions:
            missing_values_combined[char_id] = missing_questions

    imputed_values_combined = {}
    for char_id, missing_questions in missing_values_combined.items():
        imputed_values_combined[char_id] = {}
        for question_id in missing_questions:
            ques_idx = question_index[question_id]
            imputed_values_combined[char_id][question_id] = {
                "yes_tfidf": combined_matrix[character_idx, ques_idx],
                "no_tfidf": combined_matrix[character_idx, len(questions) + ques_idx],
            }

    # Prepare the answers suggested to the user
    proposed_answers_display = {"yes_suggestions": {}, "no_suggestions": {}}
    for char_id, questions in imputed_values_combined.items():
        char_name = Item.objects.get(id=char_id).name
        for question_id, tfidf_values in questions.items():
            question_text = Question.objects.get(id=question_id).title
            # print(f"Question: {question_text}, Character: {char_name}, Yes TF-IDF: {tfidf_values['yes_tfidf']}, No TF-IDF: {tfidf_values['no_tfidf']}")
            # Introduce a threshold to ensure significant difference between yes_tfidf and no_tfidf
            threshold = 0.4  # Adjust this value based on your dataset
            if abs(tfidf_values["yes_tfidf"] - tfidf_values["no_tfidf"]) < threshold:
                suggested_answer = "Uncertain"
            else:
                suggested_answer = (
                    "Yes"
                    if tfidf_values["yes_tfidf"] > tfidf_values["no_tfidf"]
                    else "No"
                )
            key = (question_id, question_text)
            value = [char_id, char_name]
            if suggested_answer == "Yes":
                if key not in proposed_answers_display["yes_suggestions"]:
                    proposed_answers_display["yes_suggestions"][key] = []
                proposed_answers_display["yes_suggestions"][key].append(value)
            else:
                if key not in proposed_answers_display["no_suggestions"]:
                    proposed_answers_display["no_suggestions"][key] = []
                proposed_answers_display["no_suggestions"][key].append(value)

    return proposed_answers_display
