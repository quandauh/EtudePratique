from django.core.management.base import BaseCommand, CommandParser

from admin.utils.autoencoder import auto_encoder_utils


class Command(BaseCommand):
    help = "Save compressed matrix to DB"

    def add_arguments(self, parser: CommandParser) -> None:
        None
    
    def handle(self, *args, **options):
        # Get the autoencoder session
        auto_encoder_session = auto_encoder_utils.AutoEncoderSession()

        # Save the compressed matrix to DB
        auto_encoder_session.save_compressed_matrix_db()
        self.stdout.write(self.style.SUCCESS("Compressed matrix saved to DB successfully."))