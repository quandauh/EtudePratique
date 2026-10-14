# Panorama du projet Devinsa

Devinsa est un jeu Django de type "20 questions" qui combine plusieurs approches pour deviner un personnage à partir des réponses des joueurs.

## Objectif

- Le joueur répond à une suite de questions.
- Le jeu s'appuie sur un arbre de décision calculé hors ligne.
- Si l'arbre ne suffit pas, le moteur TF-IDF et les suggestions complètent la recherche.

## Architecture

- `src/app` : parcours joueur, formulaire d'ajout, calcul du résultat, rapport de partie.
- `src/admin` : administration des données, génération de l'arbre, suggestions, visualisation, benchmark, maintenance.
- `src/devinsa` : configuration Django, Celery, cache Redis, URLs globales.

## Parcours joueur

1. Page d'accueil.
2. Partie guidée par l'arbre de décision.
3. Bascule éventuelle vers des questions aléatoires si l'arbre n'a plus de branche exploitable.
4. Page de devinette.
5. Ajout possible d'un personnage ou d'une question si la base est incomplète.
6. Rapport de partie.

Les vues principales sont définies dans `src/app/urls.py`.

## Moteurs de décision

### Arbre de décision

- Arbre calculé via une commande ou l'interface admin.
- La génération a été refondue autour de `create_tree_linear`.
- La sauvegarde de l'arbre remonte maintenant une progression détaillée.

### Recherche TF-IDF

- Chaque personnage est représenté par un vecteur.
- Les coefficients sont recalculés via les commandes d'administration.
- Le moteur sert aussi à alimenter les suggestions et certaines vues de visualisation.

### Autoencodeur

- Un autoencodeur est entraîné pour produire des suggestions.
- Les runs d'entraînement sont historisés dans `AutoEncoderTrainingRun`.
- Les poids peuvent être archivés et réutilisés depuis l'interface admin.

## Administration

L'admin expose notamment :

- statistiques globales
- génération d'arbre
- visualisation et clustering
- computation et validation des suggestions
- entraînement de l'autoencodeur
- benchmark / moulinette
- maintenance de déploiement

## Travail asynchrone

- Celery exécute les tâches longues.
- Redis sert de broker, de backend de résultat et de cache partagé.
- `celery_progress` alimente les barres de progression affichées dans l'UI.

## Changements majeurs depuis septembre 2025

- clarification du déploiement et des hôtes autorisés
- ajout et stabilisation de la visualisation compressée
- ajout de la page moulinette et du benchmark de succès
- ajout du panneau de maintenance admin
- suivi plus riche des entraînements autoencodeur
- optimisation forte de la génération d'arbre et de son export JSON
- corrections de bugs de navigation, de chemins relatifs et de suggestions

## Où lire quoi

- [`ARBORESCENCE.md`](ARBORESCENCE.md) : structure du projet.
- [`OPTIMISATIONS_GENERATION_ARBRE.md`](OPTIMISATIONS_GENERATION_ARBRE.md) : refonte de l'arbre.
- [`AFFICHAGE_PROGRESSION_ARBRE.md`](AFFICHAGE_PROGRESSION_ARBRE.md) : progression Celery sur l'arbre.
- [`INSTALL_VPS.md`](INSTALL_VPS.md) : installation et exploitation en production.
- [`BONNES PRATIQUES.md`](BONNES_PRATIQUES.md) : règles de style.
