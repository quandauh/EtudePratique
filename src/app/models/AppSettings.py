from django.db.models import CharField, IntegerField

from app.models.SerializableModel import SerializableModel


class AppSettings(SerializableModel):
    ALGORITHM_CHOICES = [
        ("tfidf", "TF-IDF"),
        ("autoencoder", "Auto-encodeur"),
    ]

    ae_hidden_size = IntegerField("Taille couche cachée", default=25)
    algorithm_mode = CharField("Mode algorithme", max_length=20, choices=ALGORITHM_CHOICES, default="tfidf")
    ae_weights_path = CharField("Chemin poids auto-encodeur", max_length=255, default="auto_encoder_weights.h5")

