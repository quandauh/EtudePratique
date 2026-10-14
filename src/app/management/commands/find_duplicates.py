from django.core.management.base import BaseCommand, CommandParser
from tqdm import tqdm
from app.models import Item, Doublon
from django.db import transaction, connection
from app.utilities.utilities import enleve_description_personnage as strip_desc, ratio

@transaction.atomic
class Command(BaseCommand):
    help = "Find all duplicates characters in database"
    
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-t", "--trigger", nargs="?", default=0.8, help="Which value should trigger suggestion", type=float)

    def handle(self, **options: dict[str, int]) -> None:
        characters = Item.objects.all()
        similarities = {}
        
        trigger = options["trigger"]

        with connection.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE `tmp_doublon`")

        for character1 in tqdm(characters):
            for character2 in characters:

                if character1 == character2:
                    continue
                
                similarity = ratio(
                    strip_desc(character1.name), 
                    strip_desc(character2.name)
                )

                character_tuple = (character1.id, character2.id)
                similarities[tuple(sorted(character_tuple))] = similarity

        for character_tuple, score in similarities.items():
            if score < trigger:
                continue

            doublon = Doublon(item1_id=character_tuple[0], item2_id=character_tuple[1])
            doublon.save()

            
            
