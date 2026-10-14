from django.db.models import DateTimeField, FloatField, IntegerField, CharField

from app.models.SerializableModel import SerializableModel


class AutoEncoderTrainingRun(SerializableModel):
    created_at = DateTimeField("Date d'entrainement", auto_now_add=True)
    nb_epochs = IntegerField("Nombre d'epochs")
    hidden_size = IntegerField("Taille couche cachée")
    output_file = CharField("Fichier de sortie", max_length=255)
    final_loss = FloatField("Loss finale", null=True, blank=True)
    final_false_positives = IntegerField("False positives finaux", null=True, blank=True)
    final_false_negatives = IntegerField("False negatives finaux", null=True, blank=True)
    total_duration_seconds = FloatField("Durée totale", null=True, blank=True)
    weights_file = CharField("Fichier de poids", max_length=255, null=True, blank=True)
    id = IntegerField("ID", primary_key=True)

    class Meta:
        ordering = ["-created_at"]
