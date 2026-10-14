import re
import spacy
from django.db.models import QuerySet
from app.models import Item


nlp = spacy.load('fr_core_news_sm') # Load the French NLP model to use lemmatization


def extract_keywords_from_name(name):
    """
    Extract keywords from a character's name by removing stop words and punctuation.
    Args:
        name (str): The name of the character.
    Returns:
        list: A list of keywords extracted from the name.
    """
    # Define stop words (you can expand this list)
    stop_words = set(spacy.lang.fr.stop_words.STOP_WORDS) | {"personnage"} | {"perso"} | {"joueur"}
    descriptions = " ".join(re.findall(r'\((.*?)\)', name.lower()))
    doc = nlp(descriptions.lower())

    # Extract keywords: lemmatize, remove stop words, and filter short words
    # special rule for "BD" to be included
    keywords = [
        token.lemma_ for token in doc
        if token.text.lower() == "bd" or (not token.is_stop and not token.is_punct and len(token.text) > 2 and token.lemma_ not in stop_words)
    ]

     # Post-process keywords to fix common issues
    corrected_keywords = []
    for keyword in keywords:
        if keyword == "téler":  # Correct "téler" to "télé"
            corrected_keywords.append("télé")
        else:
            corrected_keywords.append(keyword)
    
    return corrected_keywords


def get_characters_by_keywords(characters_threshold):
    """
    Build a dictionary where each keyword maps to a list of character IDs.
    Returns:
        dict: A dictionary mapping keywords to lists of character IDs.
    """
    characters: QuerySet[Item] = Item.objects.order_by("name")
    keyword_to_characters = {}

    for character in characters:
        keywords = extract_keywords_from_name(character.name)
        print(f"Character: {character.name}, Keywords: {keywords}")
        for keyword in keywords:
            if keyword not in keyword_to_characters:
                keyword_to_characters[keyword] = []
            keyword_to_characters[keyword].append(character.id)
    # We can also only keep the keywords that are appearing in at least X characters
    keyword_to_characters = {k: v for k, v in keyword_to_characters.items() if len(v) > characters_threshold}
    # Sort the keywords by their length
    keyword_to_characters = dict(sorted(keyword_to_characters.items(), key=lambda item: len(item[0]), reverse=True))
    # Sort the character IDs in each list
    for keyword in keyword_to_characters:
        keyword_to_characters[keyword] = sorted(keyword_to_characters[keyword])
    return keyword_to_characters