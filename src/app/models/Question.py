from django.db.models import CharField, ManyToManyField, FloatField

from app.models.SerializableModel import SerializableModel


class Question(SerializableModel):
    """
    Une question de la base de données

    :cvar title: l'intitulé de la question
    :cvar yes_idf: caractère discriminant de la réponse 'oui'
    :cvar no_idf: caractère discriminant de la réponse 'non'
    :cvar answers: réponses correspondantes
    """

    title = CharField("Intitulé", max_length=255, unique=True)

    yes_idf = FloatField("IDF oui", default=0)
    no_idf = FloatField("IDF non", default=0)

    answers = ManyToManyField("Item", blank=True, through="Answer")
