from django.db.models import ForeignKey, CASCADE, CharField

from app.models.SerializableModel import SerializableModel


class Doublon(SerializableModel):
    """_summary_

    Args:
        SerializableModel (_type_): _description_
    """
    item1 = ForeignKey("Item", on_delete=CASCADE, related_name="item1")
    item2 = ForeignKey("Item", on_delete=CASCADE, related_name="item2")

    class Meta:
        db_table = "tmp_doublon"
