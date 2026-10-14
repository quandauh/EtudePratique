from django.core.management.base import BaseCommand, CommandParser
import keras

from admin.utils.autoencoder import auto_encoder_utils
from app.models import AppSettings


class Command(BaseCommand):
    help = "Train auto encoder"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-n", "--nbepochs", nargs="?", default=100, type=int, help="Iterations count for model training")
        parser.add_argument("-o", "--output", nargs="?", default="auto_encoder_weights.h5", type=str, help="Output file for weights")
        parser.add_argument("-c", "--compression", nargs="?", default=25, type=int, help="Size of compressed vector")

    def handle(self, **options) -> None:
        nbepochs = options["nbepochs"]
        output_file = options["output"]
        compression = options["compression"]

        settings = AppSettings.objects.get(pk=1)
        settings.ae_hidden_size = compression
        settings.save()

        self.stdout.write(f"Entraînement : {nbepochs} époques, compression={compression}, sortie={output_file}")

        session = auto_encoder_utils.AutoEncoderSession()

        class PrintCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None): pass
            def on_epoch_begin(self, epoch, logs=None): pass
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get("loss", 0) if logs else 0
                print(f"  Époque {epoch + 1}/{nbepochs} — loss={loss:.6f}")
            def on_train_end(self, logs=None): pass

        session.train(output_file, nb_epochs=nbepochs, callbacks=[PrintCallback()])

        weights_path = auto_encoder_utils.resolve_weights_path(output_file)
        settings.ae_weights_path = weights_path
        settings.save()

        self.stdout.write("Rechargement du cache et sauvegarde des vecteurs en base...")
        auto_encoder_utils.get_cached_session(force_reload=True)

        self.stdout.write(self.style.SUCCESS(f"Entraînement terminé. Poids sauvegardés dans : {weights_path}"))
