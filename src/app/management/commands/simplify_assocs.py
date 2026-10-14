
import json
from django.core.management.base import BaseCommand
import pandas as pd
from mlxtend.frequent_patterns import *


class Command(BaseCommand):

    help = "Utilisation des règles d'association pour étendre les données"

    def handle(self, *args, **options):

        data = pd.read_csv("../out/patterns_export.csv", sep=";")
        data["antecedents"] = data["antecedents"].apply(json.loads)
        data["consequents"] = data["consequents"].apply(json.loads)
        data["ant_size"] = data["antecedents"].apply(lambda x : x.size)
        print(data.head())
        print(data[["antecedents", "consequents"]].head())


        
