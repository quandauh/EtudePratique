from asyncio import QueueEmpty
from django.core.management.base import BaseCommand, CommandParser

import argparse
from app import utilities
from admin.utils.visualisation import generate_csv

class Command(BaseCommand):
    help = "Créer les CSV pour la visualisation"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-t", "--tfidf", action=argparse.BooleanOptionalAction, default=False, help="Mettre à jour les TF.IDF avant la visualisation")
        parser.add_argument("-n", "--clusters", nargs="?", default=10, help="Nombre de cluster à rechercher", type=int)

    def handle(self, **options) -> None:
        n_clusters = options["clusters"]

        if options["tfidf"]:
            print("Mise à jour des coeffs TF.IDF...")
            utilities.indexation()

        # Génération complète des CSV avec gestion des vecteurs compressés
        # (incluant stratégie 1 erreur 1 passe pour l'autoencoder)
        generate_csv.generate_all_csv(n_clusters)
        


        