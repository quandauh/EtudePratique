from django.core.management.base import BaseCommand, CommandParser
from app.models import Item, Question, Suggestion
from django.db.models import Q
import argparse
import json
from collections import defaultdict

import re
import requests


class Command(BaseCommand):
    help = "Fetch all character images URL from Wikipedia"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-a", "--all", action=argparse.BooleanOptionalAction, default=False, help="Fetch image for all items")

    def handle(self, **options: dict) -> None:
        item_list = Item.objects.filter(~Q(image=""), image__isnull=False).all()
        itemCount = len(item_list)
        
        words_count = defaultdict(int)
        questions_words = defaultdict(list)

        questions = Question.objects.all()
        progressCount = 0
        suggestionCount = 0

        for question in questions:

            if any(char.isdigit() for char in question.title):
                continue

            alphanumericTitle = re.sub(r'[ _,(");:|?.\-\']+', ' ', question.title)
            words = alphanumericTitle.split(" ")

            for word in words:
                if len(word) <= 2:
                    continue

                words_count[word] += 1

        for question in questions:

            if any(char.isdigit() for char in question.title):
                continue

            alphanumericTitle = re.sub(r'[ _,(");:|?.\-\']+', ' ', question.title)
            words = alphanumericTitle.split(" ")
    
            sorted_words = sorted(words, key=lambda x : words_count[x])
            minCount = max(words_count[sorted_words[0]], 5)

            for word in words:
                score = words_count[word]
                if score == 0:
                    continue
                elif score <= minCount:
                    questions_words[question.id].append(word)

            print(f"{question.title} : {questions_words[question.id]}")

        for item in item_list:
            progressCount = progressCount + 1

            if progressCount % 10 == 0:
                print(f"Progress : {progressCount} / {itemCount}")

            wikiName = get_wikipedia_name(item.name)

            if wikiName is None:
                continue
            
            json_result = query_wikipedia_api(wikiName)

            print(item.name)

            for question in questions:
                if not question.id in questions_words:
                    continue

                words = questions_words[question.id]
                
                if not words or len(words) > 2:
                    continue

                suggest = True

                for criteria in words:
                    pattern = re.compile(f" {criteria}[;, .)]")

                    if not "parse" in json_result or not pattern.search(json.dumps(json_result["parse"]["text"]["*"]).lower()):
                        suggest = False
                        break
                
                if not suggest:
                    continue

                answer = item.answers_set.filter(question_id=question.id)

                if answer:
                    answer = answer.get()
                    
                    if answer.yes_count * 0.75 >= answer.no_count:
                        continue

                    if answer.no_count * 0.75 >= answer.yes_count:
                        continue

                suggestion = Suggestion(item=item, question=question, reponse='y')
                suggestion.save()
                print(f"\t{question.title} : OUI | ({words})")
                suggestionCount += 1

        print(f"Fin de la récupération des données : {suggestionCount} suggestions trouvées") 
        
        

"""
:param itemName : Convertit un nom dans la base en un slug Wikipédia
:return le slug wikipédia correspondant
"""
def get_wikipedia_name(itemName : str) -> str | None:
    pattern = re.compile(r"^([^(,;:|]*)")
    match = pattern.search(itemName) 

    return match.group(1).strip() if match else None

def get_question_criteria(questionTitle : str) -> re.Match[str] | None:
    pattern = re.compile(r"^(?:Ton|Votre) personnage est-il (?:une |un )?([a-zA-Z-]*) \?$")
    return pattern.search(questionTitle)

"""
:param wikiName : Nom du slug correspondant à la page wikipédia du personnage
:return résultat de l'api wikipédia
"""
def query_wikipedia_api(wikiName : str) -> dict:
    return requests.get(
        url="https://fr.wikipedia.org/w/api.php",
        params={
            "action": "parse",
            "page": wikiName,
            "format": "json",
            "section": 0 
        }
    ).json()