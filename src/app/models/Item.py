from django.db.models import CharField, IntegerField, TextField, ManyToManyField, FloatField

from app.models.SerializableModel import SerializableModel


class Item(SerializableModel):
    """
    Un élément (personnage) de la base de données.

    :cvar name: le nom de l'élément
    :cvar games: le nombre de parties jouées
    :cvar wins: le nombre de victoires
    :cvar image: le lien vers l'image du personnage
    """
    name = CharField("Nom", max_length=255, unique=True, db_index=True)
    games = IntegerField("Parties", default=0)
    wins = IntegerField("Victoires", default=0)
    image = TextField("Image", default=None, null=True)
    answers = ManyToManyField("Question", blank=True, through="Answer")
    vectorSize = FloatField("VectorSize", default=0)
    cluster = IntegerField("Cluster", blank=True, null=True, default=None)

    class Meta:
        ordering = ["-games"]
