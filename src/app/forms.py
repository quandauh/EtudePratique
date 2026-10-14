from django.forms import ModelForm

from app.models import Item, Question


# Formulaire d'ajout de questions
# https://docs.djangoproject.com/fr/4.2/topics/forms/
class QuestionForm(ModelForm[Question]):
    class Meta:
        model = Question
        fields = ["title"]


# Formulaire d'ajout d'éléments
class ItemForm(ModelForm[Item]):
    class Meta:
        model = Item
        fields = ["name"]
