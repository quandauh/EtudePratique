from asyncio import QueueEmpty
from django.core.management.base import BaseCommand, CommandParser

from admin.utils.autoencoder import auto_encoder_utils

class Command(BaseCommand):
    help = "Export auto encoder model file"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("output", nargs="?", default="auto_encoder_model.keras", help="Output file for model", type=str)

    def handle(self, **options) -> None:
        filename = options["output"]

        print(f"Exporting auto encoder model to {filename}")

        auto_encoder_utils.export_model(filename)