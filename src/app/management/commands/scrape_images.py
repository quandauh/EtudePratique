from django.core.management.base import BaseCommand, CommandParser
from app.models import Item
from app.management.commands.fetch_images_url import get_wikipedia_name
from app.utilities.wikipedia_scraper import WikipediaImageScraper, ScrapeResult
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
            result = scrape_item(scraper, item.name)
            progress = f"[{index}/{len(items)}]"

            if result is None:
                print(f"{progress} Page not found for {item.name}")
            elif result.is_disambiguation:
                print(f"{progress} Disambiguation page for {item.name}")
            elif result.image_url is None:
                print(f"{progress} No image for {item.name}")
            else:
                print(f"{progress} Image found for {item.name}")
                imageCount += 1

                if not options["dry_run"]:
                    item.image = result.image_url
                    item.save()

            time.sleep(options["delay"])

        print(f"Fin du scraping des images : {imageCount} / {len(items)} trouvées")


def scrape_item(scraper: WikipediaImageScraper, itemName: str) -> ScrapeResult | None:
    result = scraper.scrape(itemName)
    if result is not None and result.image_url and not result.is_disambiguation:
        return result

    shortName = get_wikipedia_name(itemName)
    if shortName and shortName != itemName:
        return scraper.scrape(shortName) or result

    return result
