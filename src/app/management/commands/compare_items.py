
from django.core.management.base import BaseCommand, CommandParser
import app.utilities.utilities as utils
import numpy as np
from admin.utils.tree import generation
from app.models import Item, Question, Answer, Tree, PlayerAnswer
from random import choices
from tqdm import tqdm
import cProfile
import argparse

class Command(BaseCommand):

    help = "Compare différents algorithmes de recherche du personnage"

    def add_arguments(self, parser):
        parser.add_argument("character1", nargs="?", help="id du premier personnage à comparer", type=int)
        parser.add_argument("character2", nargs="?", help="ud du second personnage à comparer", type=int)

    def handle(self, *args, **options):
        char1_id = options["character1"]
        char2_id = options["character2"]

        character1 = Item.objects.get(pk=char1_id)
        character2 = Item.objects.get(pk=char2_id)

        answers1 = character1.answers_set.all()
        answers2 = character2.answers_set.all()

        answers1 = list(sorted(list(answers1), key=lambda x : x.question_id))
        answers2 = list(sorted(list(answers2), key=lambda x : x.question_id))

        v1 = list(map(lambda x : x.question_id, list(answers1)))
        v2 = list(map(lambda x : x.question_id, list(answers2)))

        v1A = list(map(lambda x : x.yes_tfidf if x.yes_count * 0.75 > x.no_count else x.no_tfidf if x.no_count * 0.75 > x.yes_count else 0, list(answers1)))
        v2A = list(map(lambda x : x.yes_tfidf if x.yes_count * 0.75 > x.no_count else x.no_tfidf if x.no_count * 0.75 > x.yes_count else 0, list(answers2)))

        inter = set(v1).intersection(set(v2))

        v1 = list(map(lambda x : (x, v1A[v1.index(x)]), inter))
        v2 = list(map(lambda x : (x, v2A[v2.index(x)]), inter))

        v1 = list(map(lambda x : x[1], v1))
        v2 = list(map(lambda x : x[1], v2))

        v1 = np.array(v1)
        v2 = np.array(v2)

        sym = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        print(f"Dimension of answers 1 : {len(answers1)}")
        print(f"Dimension of answers 2 : {len(answers2)}")
        print(f"Union size : {len(inter)}")
        print(f"Similarity : { sym }")

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3