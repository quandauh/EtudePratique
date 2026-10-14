from django.db.models import CASCADE, FloatField, ForeignKey, IntegerField, UniqueConstraint

from app.models.SerializableModel import SerializableModel


class Answer(SerializableModel):
    """
    Une réponse liée à une question et un élément.

    :cvar question: la question concernée
    :cvar item: l'élément concerné
    :cvar yes_count: le nombre de oui
    :cvar no_count: le nombre de non
    :cvar pass_count: le nombre de passes
    :cvar yes_tf: proportion de réponses 'oui'
    :cvar no_tf: proportion de réponses 'non'
    :cvar yes_tfidf: produit yes_tf*yes_idf
    :cvar no_tfidf: produit no_tf*no_idf
    """

    question = ForeignKey("Question", related_name="answers_set", db_column="question_id", on_delete=CASCADE)
    item = ForeignKey("Item", related_name="answers_set", db_column="item_id", on_delete=CASCADE)

    yes_count = IntegerField("Nombre de oui", default=0)
    no_count = IntegerField("Nombre de non", default=0)
    pass_count = IntegerField("Nombre de passes", default=0)
    
    yes_tf = FloatField("TF oui", default=0)
    no_tf = FloatField("TF non", default=0)
    yes_tfidf = FloatField("TF-IDF oui", default=0)
    no_tfidf = FloatField("TF-IDF non", default=0)

    class Meta:
        """
        :cvar constraints: contrainte d'unicité sur question et item
        :cvar ordering: tri par défaut des objets récupérés
        """

        constraints = [UniqueConstraint(fields=["question", "item"], name="answer_question_item")]
        ordering = ["-yes_count", "-no_count", "-pass_count"]
