from django.db import models
from app.models.Item import Item


class CharacterVector(models.Model):
    item = models.OneToOneField(Item, on_delete=models.CASCADE, related_name="character_vector")
    vector = models.JSONField()
    weights_path = models.CharField(max_length=500)
    updated_at = models.DateTimeField(auto_now=True)
