from django.core.management.base import BaseCommand
from app.models import Item

class Command(BaseCommand):
    help = "Generates the tree from the database"

    def handle(self, **options: dict[str, int]) -> None:
        for item in Item.objects.exclude(name__contains="(").all():
            newName = input(f"[{item.name}] : ")
            if newName == "p" or newName == "":
                continue
            elif newName == "d":
                item.delete()
            else:
                item.name = newName
                item.save()