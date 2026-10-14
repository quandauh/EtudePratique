from admin.utils.suggestions.find_keywords import get_characters_by_keywords
from app.models import Answer, Item, Question
from django.db.models import Q


def generate_suggestions_based_on_descriptions(characters_threshold=10):
    """
    Generate suggestions for questions based on keywords in character descriptions.
    Returns:
        dict: A dictionary mapping question IDs to suggested answers.
    """
    # Get all characters and their keywords
    keyword_to_characters = get_characters_by_keywords(characters_threshold)
    # no_suggestions are not used for now because it may be hard to find the questions talking about the opposite of the keyword
    # but to keep the data structure like the one used in the other functions, we keep it
    proposed_answers_display = {'yes_suggestions': {}, 'no_suggestions': {}}

    for keyword, character_ids in keyword_to_characters.items():
        # Find questions containing the keyword
        questions = match_keywords_to_questions([keyword])
        
        for question in questions:
            key = (question.id, question.title)
            # Get the character names
            characters = Item.objects.filter(id__in=character_ids)
            
            characters_without_answers = [
                character for character in characters
                if not Answer.objects.filter(item_id=character.id, question_id=question.id).exists()
            ]

            for character in characters_without_answers:
                value = [character.id, character.name]
                if key not in proposed_answers_display['yes_suggestions']:
                    proposed_answers_display['yes_suggestions'][key] = []
                proposed_answers_display['yes_suggestions'][key].append(value)

    return proposed_answers_display

def match_keywords_to_questions(keywords):
    """
    Match keywords to questions in the database.
    Args:
        keywords (list): A list of keywords to match.
    Returns:
        QuerySet: A queryset of questions matching the keywords.
    """
    query = Q()
    for keyword in keywords:
        query |= Q(title__icontains=keyword)
    
    matched_questions = Question.objects.filter(query)
    return matched_questions