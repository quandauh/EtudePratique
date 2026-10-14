from django.core.management.base import BaseCommand
from app.models import Item

class Command(BaseCommand):
    help = "Rename character with no image found"

    def handle(self, **options: dict[str, int]) -> None:
        for item in Item.objects.exclude(image__contains="https://").all():
            newName = input(f"[{item.name}] : ")
            if newName == "p" or newName == "":
                continue
            elif newName == "d":
                item.delete()
            else:
                item.name = newName
                item.save()