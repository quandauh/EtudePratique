from django.shortcuts import get_object_or_404
from app.models import Item


def find_similarities(
    similarity_matrix, character_idx, answers, num_closest_characters=10
):
    """
    Find the most similar characters to the given character from the similarity matrix.
    """

    similarities = similarity_matrix[character_idx]
    closest_indices = similarities.argsort()[-num_closest_characters:-1][
        ::-1
    ]  # Exclude the character itself and get top 10

    # Object representing the target character
    characters = list(set(answer.item_id for answer in answers))
    # Map back to original character IDs
    closest_character_ids = [characters[idx] for idx in closest_indices]
    # print(f"Comparaison des personnages les plus proches:")
    # for idx in range(len(closest_character_ids)):
    #     classic_char_id = closest_character_ids[idx]
    #     classic_char_name = Item.objects.get(id=classic_char_id).name
    #     classic_similarity = similarities[closest_indices[idx]]
    #     print(f"{idx+1}. {classic_char_name} (classique: {classic_similarity:.2f})")

    return closest_character_ids
