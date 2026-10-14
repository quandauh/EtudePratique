from enum import Enum
from app.models import Tree


class NodeType(Enum):
    ROOT = 'r'
    BRANCH_YES = 'y'
    BRANCH_NO = 'n'
    BRANCH_PASS = 'p'


class Node():

    def __init__(self, parent, depth: int, question_id: int, type: NodeType) -> None:
        """Créer un nouveau noeud dans l'arbre de décision

        Args:
            parent (Node): noeud parent
            depth (int): profondeur du noeud dans l'arbre
            question_id (int): identifiant de la question associée
            type (NodeType): type de noeud (root, oui, non, passe)
        """
        self.parent = parent
        self.depth = depth
        self.question_id = question_id
        self.type = type

        self.yes_child = None
        self.no_child = None
        self.pass_child = None

        if parent is not None:
            match type:
                case NodeType.BRANCH_YES:
                    parent.yes_child = self
                case NodeType.BRANCH_NO:
                    parent.no_child = self
                case NodeType.BRANCH_PASS:
                    parent.pass_child = self

    def to_tree_object(self, id: int, parent: Tree) -> Tree:
        """Convertit un objet "Node", qui sert à créer l'arbre en un objet que l'on
        peut stocker en base de donnée.

        Args:
            id (int): id du noeud dans la base (pk)
            parent (Tree): objet du parent dans la base

        Returns:
            Tree: l'objet représentant ce noeud pouvant être stocké en base
        """
        return Tree(id=id, parent=parent, question_id=self.question_id, choice=self.type.value, depth=self.depth, tid=0)

    
        