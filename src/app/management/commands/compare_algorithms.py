
from django.core.management.base import BaseCommand, CommandParser
import app.utilities.utilities as utils
from app.models import Item, Question, Answer, Tree, PlayerAnswer
from tqdm import tqdm
import argparse

class Command(BaseCommand):

    help = "Compare différents algorithmes de recherche du personnage"

    def add_arguments(self, parser):
        parser.add_argument("-p", "--profile", action=argparse.BooleanOptionalAction, default=False, help="Profile command")
        parser.add_argument("research_functions", nargs='+', type=str, help="Noms des fonctions de recherche à tester")
        parser.add_argument("-m", "--minPlayed", nargs="?", default=0, help="Nombre mini de parties joues", type=int)

    def handle(self, *args, **options):
        #récupère les noms de fonctions passés en paramètres
        research_functions = options["research_functions"]

        #initialise la SDD de retour
        result = {rsch_fct_name:0 for rsch_fct_name in research_functions}

        rootNode = Tree.objects.get(pk=1)
        item_dict = generate_itemdict()

        for research_function_name in research_functions:
            
            research_function = getattr(utils, research_function_name)

            characters = Item.objects.filter(games__gte=options["minPlayed"]).all()
            for item in tqdm(characters):
                guessed_item_id = fakeGame(item, item_dict, research_function, rootNode)

                if guessed_item_id != item.id:
                    print(f"Found { guessed_item_id } expected { item.id }")

                result[research_function_name] += (guessed_item_id == item.id)

            result[research_function_name] /= len(characters)
        
        print(result)


def fakeGame(character: Item, item_dict, research_function, rootNode):
    answers = []

    questionCount = 0
    tree = rootNode

    while tree is not None:
        question_id = tree.question_id
    
        # Cherche les réponses pour ce perso/cette question
        
        # Choisit une réponse
        if question_id in item_dict[character.id]:
            db_answers = item_dict[character.id][question_id]
            rep = "y" if db_answers[0] > db_answers[1] else "n" if db_answers[1] > db_answers[0] else "p"
            answer = rep
        else:
            answer = "p" #s'il n'y a pas de réponse en base, la question est passée

        answers.append(PlayerAnswer(question_id, answer))

        # Récupère la question suivante
        tree = tree.get_child(answer)

        questionCount += 1

    depthCount = questionCount

    while questionCount < 20:
        question_id = Question.objects.order_by("?").first().id

        # Choisit une réponse
        if question_id in item_dict[character.id]:
            db_answers = item_dict[character.id][question_id]
            rep = "y" if db_answers[0] > db_answers[1] else "n" if db_answers[1] > db_answers[0] else "p"
            answer = rep
        else:
            answer = "p" #s'il n'y a pas de réponse en base, la question est passée

        answers.append(PlayerAnswer(question_id, answer))
        questionCount += 1

    guessed_item_id = research_function(answers, depth=depthCount)
    return guessed_item_id

def generate_itemdict() -> dict[int, list[dict[str, int]]]:
    """
    Création d'un dictionnaire qui va contenir toutes les réponses de la base
    pour éviter les sous requêtes
    :param answers: liste des réponses du joueur : permet d'affiner le résultat
    :return: le dictionnaire d'item : liste de réponses filtré pour que seuls
        les items ayant une réponse de la liste de questions apparaissent
    """
    itemdict: dict[int, dict[int, float]] = {}

    characters = Item.objects.prefetch_related("answers_set").all()
    questionList = [x.question_id for x in Answer.objects.all()]

    for character in characters:
        filtered_answers = filter(lambda x: x.question_id in questionList, character.answers_set.all())
        
        answers_dict = {}
        for answer in filtered_answers:
            answers_dict[answer.question_id] = (answer.yes_tfidf, answer.no_tfidf, 100 if answer.yes_tfidf == answer.no_tfidf else 0)
        
        itemdict[character.id] = answers_dict

    return itemdict