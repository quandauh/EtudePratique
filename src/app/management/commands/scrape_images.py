from django.core.management.base import BaseCommand, CommandParser
from app.models import Item
from app.utilities.wikipedia_scraper import WikipediaImageScraper
import argparse
import time


class Command(BaseCommand):
    help = "Scrape character images from Wikipedia pages"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-a", "--all", action=argparse.BooleanOptionalAction, default=False, help="Scrape image for all items")
        parser.add_argument("-l", "--lang", default="fr", help="Wikipedia language (fr, en...)")
        parser.add_argument("-n", "--limit", type=int, default=None, help="Maximum number of items")
        parser.add_argument("-d", "--delay", type=float, default=0.5, help="Seconds to wait between items")
        parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Do not save images")

    def handle(self, **options: dict) -> None:
        if options["all"]:
            items = Item.objects.all()
        else:
            items = Item.objects.exclude(image__contains="https://")

        items = list(items[:options["limit"]])

        scraper = WikipediaImageScraper(lang=options["lang"])
        imageCount = 0

        for index, item in enumerate(items, start=1):
            image = scraper.find_image(item.name)
            progress = f"[{index}/{len(items)}]"

            if image is None:
                print(f"{progress} No image found for {item.name}")
            else:
                print(f"{progress} Image found for {item.name}")
                imageCount += 1

                if not options["dry_run"]:
                    item.image = image
                    item.save()

            time.sleep(options["delay"])

        print(f"Fin du scraping des images : {imageCount} / {len(items)} trouvées")
