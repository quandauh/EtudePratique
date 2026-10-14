
import csv
import json
from django.core.management.base import BaseCommand, CommandParser
import pandas as pd
from mlxtend.frequent_patterns import *


class Command(BaseCommand):

    help = "Recherche des règles d'association entre les questions"

    def handle(self, *args, **options):

        data = pd.read_csv("../out/list_character.csv", sep=";")
        data = data.drop(columns=list(filter(lambda x : "No" in x, list(data.columns))))
        values = data.iloc[:, 1:]
        values = values.map(hot_encode)

        print(values.head())

        frq_items = fpgrowth(values, min_support = 0.2, use_colnames = True) 
        frq_items.to_csv("../out/frequent_items.csv", sep=";")

        print(frq_items.size)
        print("Frequent itemset done!")

        rules = association_rules(frq_items, metric ="lift", min_threshold=1)
        rules["ant_size"] = rules["antecedents"].apply(lambda x : len(x))
        rules = rules.sort_values(['lift', 'support'], ascending =[False, False])
        rules["antecedents"] = rules["antecedents"].apply(lambda x : list(x)).apply(json.dumps).apply(pd.Series)
        rules.to_csv("../out/patterns_export.csv", sep=";")

        print(rules.head())
        print(rules.size)

        print("Association rules done !")




def hot_encode(x): 
    x = float(x)
    if(x < 1): 
        return False
    if(x >= 1): 
        return True