from time import time

from django.core.management.base import BaseCommand, CommandParser

from app.utilities import indexation

class Command(BaseCommand):
    help = "Mettre à jour les coefficients TF.IDF dans la base"

    def handle(self, **options: dict[str, int]) -> None:
        self.stdout.write("Calcul des poids TFIDF...")
        
        start = time()
        indexation()
        end = time()

        print_elapsed(start, end)
    

def convertMillis(secs: float) -> tuple[int, int, int]:
    millis = int(secs * 1000)
    seconds=(millis // 1000) % 60
    minutes=(millis // (1000 * 60)) % 60
    hours=(millis // (1000 * 60 * 60)) % 24

    return seconds, minutes, hours

def print_elapsed(start: float, end: float):
    (seconds, minutes, hours) = convertMillis(end-start)
    print(f"Temps écoulé : {hours:0=2}:{minutes:0=2}:{seconds:0=2}")