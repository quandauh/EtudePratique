from django.core.management.base import BaseCommand, CommandParser
from app.models import Item
import argparse

import re
import requests

class Command(BaseCommand):
    help = "Fetch all character images URL from Wikipedia"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-a", "--all", action=argparse.BooleanOptionalAction, default=False, help="Fetch image for all items")
        parser.add_argument("-l", "--api", default="https://fr.wikipedia.org/w/api.php", help="Specify API Link to fetch")

    def handle(self, **options: dict) -> None:
        item_list = Item.objects.exclude(image__contains="https://").all() if not options["all"] else Item.objects.all()
        itemCount = len(item_list)

        progressCount = 0
        imageCount = 0

        for item in item_list:
            progressCount = progressCount + 1

            if progressCount % 10 == 0:
                print(f"Progress : {progressCount} / {itemCount}")

            wikiName = get_wikipedia_name(item.name)

            if wikiName is None:
                continue

            json_result = query_wikipedia_api(options["api"], wikiName)
            image_url = extract_thumbnail_url(json_result)

            if image_url:
                print(f"[{progressCount}] Image found for {wikiName}")

                imageCount = imageCount + 1

                # Sauvegarder le personnage le lien vers son image
                item.image = image_url
                item.save()
            else:
                print(f"Image not found for { wikiName }")

        print(f"Fin de la récupération des images : {imageCount} / {itemCount} trouvées")

"""
:param itemName : Convertit un nom dans la base en un slug Wikipédia
:return le slug wikipédia correspondant
"""
def get_wikipedia_name(itemName : str) -> str | None:
    pattern = re.compile(r"^([^(,;:|]*)")
    match = pattern.search(itemName) 
    return match.group(1).strip() if match else None

"""
:param wikiName : Nom du slug correspondant à la page wikipédia du personnage
:return résultat de l'api wikipédia
"""
def query_wikipedia_api(wikiUrl: str, wikiName : str) -> dict | None:
    try:
        response = requests.get(
            url=wikiUrl,
            params={
                "action": "query",
                "prop": "pageimages",
                "titles": wikiName,
                "format": "json",
                "pithumbsize": 200,
                "pilangcode": "FR_fr"
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"Wikipedia API request failed for '{wikiName}': {exc}")
    except ValueError as exc:
        print(f"Wikipedia API returned invalid JSON for '{wikiName}': {exc}")
    return None


def extract_thumbnail_url(json_result: dict | None) -> str | None:
    if not isinstance(json_result, dict):
        return None

    query = json_result.get("query")
    if not isinstance(query, dict):
        return None

    pages = query.get("pages")
    if not isinstance(pages, dict):
        return None

    for page in pages.values():
        if not isinstance(page, dict):
            continue
        thumbnail = page.get("thumbnail")
        if not isinstance(thumbnail, dict):
            continue
        source = thumbnail.get("source")
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            return source

    return None
