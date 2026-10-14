from django.core.management.base import BaseCommand, CommandParser

from admin.utils.autoencoder import auto_encoder_utils
from admin.utils.autoencoder import datas
from django.db import transaction, connection

from tqdm import tqdm

from admin.tasks import compute_suggestions_task


class Command(BaseCommand):
    help = "Train auto encoder"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-i", "--input", nargs="?", default="auto_encoder_weights.h5", help="Input file for weights", type=str)
        parser.add_argument("-t", "--trigger", nargs="?", default=0.9, help="Which value should trigger suggestion", type=float)
        parser.add_argument("-m", "--minPlayed", nargs="?", default=5, help="Nombre de partie minimum sur les persos", type=int)
        parser.add_argument("-u", "--maxPlayed", nargs="?", default=10, help="Nombre de partie maximum sur les persos", type=int)

    def handle(self, **options) -> None:
        input_file = options["input"]
        min_Played = options["minPlayed"]
        max_Played = options["maxPlayed"]
        triggerValue = options["trigger"]

        compute_suggestions_task.delay(input_file=input_file, min_Played=min_Played, max_Played=max_Played, triggerValue=triggerValue)