from typing import Self

from django.db.models import CASCADE, CharField, ForeignKey, IntegerField

from .SerializableModel import SerializableModel


class Tree(SerializableModel):
    """
    La table Tree stockée dans la base, ses entrées sont les nœuds de l'arbre des questions
    L'arbre des questions permet au cours de la partie d'avoir un enchainement rapide des questions
    L'arbre des questions est représenté par la racine de tous les objets de type Tree (les nœuds de l'arbre)
    Dans la suite un objet Tree peut être apparenté à :
        - l'arbre global dans le cas où il s'agit de la racine
        - un nœud simple, ayant 1 père et 3 fils.

    :cvar question: la question du nœud
    :cvar parent: le nœud père
    :cvar choice: le type du nœud
        - "r" pour une racine
        - "y" pour un nœud venant d'une branche "Oui"
        - "n" pour un nœud venant d'une branche "Non"
        - "p" pour un nœud venant d'une branche "Passe"
    :cvar depth: la profondeur du nœud (distance à la racine)
    """

    question = ForeignKey("Question", null=True, on_delete=CASCADE)
    parent = ForeignKey("self", related_name="children", null=True, on_delete=CASCADE)

    choice = CharField("Choix", max_length=1, default="r")
    depth = IntegerField(default=1)
    tid = IntegerField(default=-1)

    def get_child(self, type: str) -> Self | None:
        """
        Récupère une référence sur le fils du type spécifié du nœud
        :return: la référence sur le fils
        """
        try:
            return self.children.get(choice=type)
        except Tree.DoesNotExist:
            return None
        
    class Meta:
        db_table = "tmp_tree"
