## Intégration de l’affichage de progression de génération d’arbre

Le suivi de progression passe par `celery_progress` et par des callbacks remontés depuis la génération et la sauvegarde de l'arbre.

1. **Appels Celery**  
   - `src/admin/tasks.py`: la tâche `create_tree_task` transmet un callback de progression à chaque phase (mise à jour TF-IDF, construction puis sauvegarde), en actualisant en continu la description via `ProgressRecorder`.

2. **Génération linéaire**  
   - `src/admin/utils/tree/generation_linear.py`: la classe `LinearTreeBuilder` accepte un callback optionnel, émet des messages pendant le pré-calcul des réponses, après la sélection de la racine et pendant l'expansion des nœuds.

3. **Sauvegarde de l’arbre**  
   - `src/admin/utils/tree/generation.py`: `create_tree` fait remonter le callback à `create_tree_linear`, et la sauvegarde (`save_tree` / `save_childs`) enregistre le nombre total de nœuds pour émettre des mises à jour régulières afin que la barre de progression de l’interface affiche un message détaillé pendant toute la durée de la tâche.
