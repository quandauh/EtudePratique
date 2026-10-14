from django.db.models import (
    CASCADE,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKey,
    IntegerField,
    JSONField,
)

from app.models.SerializableModel import SerializableModel


class MoulinetteRun(SerializableModel):
    task_id = CharField("Task Celery", max_length=255, unique=True)
    created_at = DateTimeField("Date d'execution", auto_now_add=True)
    created_by = ForeignKey(
        "auth.User",
        null=True,
        blank=True,
        on_delete=CASCADE,
        related_name="moulinette_runs",
    )
    status = CharField("Statut", max_length=255)
    config = JSONField("Configuration", default=dict)
    result = JSONField("Resultat", default=dict)

    total = IntegerField("Total", default=0)
    processed = IntegerField("Traites", default=0)
    correct = IntegerField("Corrects", default=0)
    incorrect = IntegerField("Incorrects", default=0)
    not_found = IntegerField("Non trouves", default=0)

    accuracy = FloatField("Precision", default=0.0)
    top_k_accuracy = FloatField("Precision top-k", default=0.0)
    top_5_accuracy = FloatField("Precision top 5", default=0.0)
    avg_tree_depth = FloatField("Profondeur moyenne", default=0.0)
    avg_questions = FloatField("Questions moyennes", default=0.0)
    elapsed_seconds = FloatField("Duree", default=0.0)

    class Meta:
        ordering = ["-created_at"]
