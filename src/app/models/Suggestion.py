from django.db.models import ForeignKey, CASCADE, CharField

from app.models.SerializableModel import SerializableModel
from app.models import Question


class Suggestion(SerializableModel):
    """_summary_

    Args:
        SerializableModel (_type_): _description_
    """
    question = ForeignKey(Question, on_delete=CASCADE)
    item = ForeignKey("Item", on_delete=CASCADE)
    reponse = CharField("Reponse", max_length=1, default="y") 

    class Meta:
        db_table = "tmp_suggestion"
