from time import time

from django.core.management.base import BaseCommand, CommandParser

from app.utilities import indexation
from admin.utils.tree import generation
from pathlib import Path
from devinsa.settings import BASE_DIR
from admin.utils.generate_tree_json import ecritureData

class Command(BaseCommand):
    help = "Generates the tree from the database"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("depth", nargs="?", default=4, help="The depth of the tree to generate", type=int)

    def handle(self, **options: dict[str, int]) -> None:
        depth = options["depth"]

        self.stdout.write("Calcul des poids TFIDF...")
        
        start = time()
        indexation()
        end = time()

        print_elapsed(start, end)


        self.stdout.write("Calcul de l'abre de décision...")
        start = time()
        rootNode = generation.create_tree(max_depth=depth);
        end = time()

        print_elapsed(start, end)

        self.stdout.write("Sauvegarde de l'arbre calculé...")
        start = time()
        generation.save_tree(rootNode)
        # Création d'un json (après avoir supprimé les fichiers éventuels existant)
        basedir: Path = BASE_DIR / "static/data"
        basedir.mkdir(exist_ok=True)
        tree_files: list[Path] = list(basedir.glob("tree_*"))
        if not Path(f"{basedir}/tree_{depth}.json").exists():
            for f in tree_files:
                f.unlink()
        ecritureData(depth)
        end = time()
        
        print_elapsed(start, end)

        self.stdout.write(self.style.SUCCESS(f"Fin de la génération de l'arbre pour depth={depth}"))
    

def convertMillis(secs: float) -> tuple[int, int, int]:
    millis = int(secs * 1000)
    seconds=(millis // 1000) % 60
    minutes=(millis // (1000 * 60)) % 60
    hours=(millis // (1000 * 60 * 60)) % 24

    return seconds, minutes, hours

def print_elapsed(start: float, end: float):
    (seconds, minutes, hours) = convertMillis(end-start)
    print(f"Temps écoulé : {hours:0=2}:{minutes:0=2}:{seconds:0=2}")