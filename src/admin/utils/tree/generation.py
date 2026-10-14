from __future__ import annotations

from django.db import connection, transaction

from admin.utils.tree.generation_linear import create_tree_linear
from admin.utils.tree.node import Node
from app.models import Tree
from typing import Callable, Optional



def create_tree(max_depth: int = 13, progress: Optional[Callable[[str], None]] = None) -> Node:
    return create_tree_linear(max_depth=max_depth, progress=progress)
def _count_nodes(node: Optional[Node]) -> int:
    if node is None:
        return 0
    return 1 + _count_nodes(node.no_child) + _count_nodes(node.yes_child) + _count_nodes(node.pass_child)


@transaction.atomic
def save_tree(rootNode: Node, progress: Optional[Callable[[str], None]] = None) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        cursor.execute("TRUNCATE TABLE `tmp_tree`")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    rt: Tree = rootNode.to_tree_object(id=1, parent=None)

    global nodeCounter
    nodeCounter = 2

    rt.save()
    total_nodes = _count_nodes(rootNode)
    saved = 1  # root saved
    if progress is not None:
        progress(f"Sauvegarde de l'arbre: {saved}/{total_nodes}")
    save_childs(rootNode, rt, total_nodes, saved, progress)



@transaction.atomic
def save_childs(node: Node, tree: Tree, total_nodes: int = 0, saved: int = 0, progress: Optional[Callable[[str], None]] = None) -> int:
    global nodeCounter

    if node.no_child is not None:
        newNode = node.no_child.to_tree_object(id=nodeCounter, parent=tree)
        newNode.save()
        nodeCounter += 1
        saved += 1
        if progress is not None and saved % 25 == 0:
            progress(f"Sauvegarde de l'arbre: {saved}/{total_nodes}")
        saved = save_childs(node.no_child, newNode, total_nodes, saved, progress)

    if node.yes_child is not None:
        newNode = node.yes_child.to_tree_object(id=nodeCounter, parent=tree)
        newNode.save()
        nodeCounter += 1
        saved += 1
        if progress is not None and saved % 25 == 0:
            progress(f"Sauvegarde de l'arbre: {saved}/{total_nodes}")
        saved = save_childs(node.yes_child, newNode, total_nodes, saved, progress)

    if node.pass_child is not None:
        newNode = node.pass_child.to_tree_object(id=nodeCounter, parent=tree)
        newNode.save()
        nodeCounter += 1
        saved += 1
        if progress is not None and saved % 25 == 0:
            progress(f"Sauvegarde de l'arbre: {saved}/{total_nodes}")
        saved = save_childs(node.pass_child, newNode, total_nodes, saved, progress)
    if progress is not None and saved == total_nodes:
        progress(f"Sauvegarde de l'arbre: {saved}/{total_nodes}")
    return saved
