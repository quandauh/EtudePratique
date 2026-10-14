# Optimisations de la génération d'arbre (comparaison avec 1ed53d188c1c4a8461ea0c36ee0ec513792440ac)

Ce document explique ce qui a changé depuis le commit de référence `1ed53d188c1c4a8461ea0c36ee0ec513792440ac` au sujet de la génération de l'arbre de décision, et en quoi ces changements optimisent les performances et la maintenabilité.

Depuis, la génération a aussi gagné un affichage de progression, des correctifs de parcours et des optimisations d'export JSON.

## Résumé

- Avant (commit 1ed53d1) : l'intégralité de l'algorithme de génération vivait dans `src/admin/utils/tree/generation.py` et recalculait, à chaque niveau, la « pertinence » des questions en itérant sur les réponses des personnages filtrés (sous-ensembles yes/no/pass). Cela impliquait de multiples parcours répétés des données en mémoire, avec un coût croissant avec la profondeur et le nombre de nœuds.
- Après (commit 1f03244) : la génération est déléguée à un nouveau composant `LinearTreeBuilder` (`src/admin/utils/tree/generation_linear.py:11`), qui pré‑calcule une fois pour toutes des masques binaires par question (yes/no/pass) et opère ensuite des partitions par simples opérations bit‑à‑bit. Le choix de la « meilleure » question se fait via un score multi‑critères favorisant l'équilibre des splits, réduisant le travail récursif.

## Avant (1ed53d1)

- Code concerné : `src/admin/utils/tree/generation.py` (contenu à cette révision).
- Fonctionnement :
  - `get_questions_sorted_by_pertinence(characters)` parcourt tous les personnages et leurs réponses pour compter yes/no par question, puis calcule un score `yes * no` et trie les questions à chaque appel.
  - À chaque nœud, on splitte la population active (`split_characters_question`) et on relance un tri des questions pour les sous‑listes yes / no (et éventuellement pass).
  - Complexité empirique élevée : on ré‑itère beaucoup sur les mêmes réponses à chaque niveau et pour chaque sous‑arbre.

Conséquences :
- De nombreux parcours de données répétés (sur les réponses) à chaque appel récursif.
- Un coût cumulé important quand le nombre de personnages/questions augmente et/ou quand on pousse la profondeur.

## Après (1f03244)

- Fichier d’entrée simplifié : `src/admin/utils/tree/generation.py:11` délègue la construction à `create_tree_linear` et conserve la logique de persistance en base.
- Nouveau générateur : `src/admin/utils/tree/generation_linear.py:11`.
  - Pré‑calculs uniques (`_precompute_answer_data`) :
    - Construction de masques binaires par question: `question_yes_masks`, `question_no_masks`, `question_pass_masks` et `question_answered_masks` (ex. `src/admin/utils/tree/generation_linear.py:27` à `:31`).
    - Une seule passe sur les personnages et leurs réponses, puis tri de la liste des questions (`:69` à `:116`).
  - Partitions en O(1) bit‑wise :
    - Pour une population courante `mask`, le partitionnement par question est calculé par AND/NOT bit‑à‑bit (`_partition`, `:206` à `:251`).
  - Sélection de question robuste et déterministe :
    - Score multi‑critères (`_select_best_question`, `:167` à `:237`) favorisant d’abord l’équilibre (`min(yes_only, no_only)`), puis la couverture, puis un faible écart, puis un faible « pass » — limitant les splits dégénérés.
  - Expansion récursive qui réutilise les masques :
    - On évite de re‑parcourir toutes les réponses; seules des opérations bit‑wise sont effectuées pour chaque candidate.

## Impact en complexité (haut niveau)

- Avant :
  - À chaque nœud, on recalculait un tri des questions sur le sous‑ensemble de personnages filtré, impliquant de re‑reparcourir des réponses déjà visitées. Schématiquement, cela ressemblait à O(#nœuds × (#persos_sous‑ensemble × #réponses_par_perso)) avec des répétitions.

- Après :
  - Pré‑calcul en O(#persos × #réponses_par_perso) effectué une seule fois.
  - Pour chaque nœud, la sélection de la « meilleure » question parcourt la liste des questions mais ne fait que des opérations sur entiers (masques) et des `bit_count`, bien plus rapides et sans I/O.
  - En pratique, on observe une meilleure scalabilité en mémoire et CPU grâce à la réutilisation des masques et à la réduction drastique des recomputations.

## Effets fonctionnels

- Splits plus équilibrés par conception, réduisant la profondeur effective nécessaire pour séparer les personnages.
- Gestion explicite des réponses « pass » par masque, limitant la pollution des splits oui/non.

## Conclusion

Par rapport à `1ed53d1`, la génération d’arbre a été refondue autour d’un pré‑calcul unique et d’un cœur bit‑wise. Cela supprime les re‑parcours coûteux des réponses à chaque niveau, améliore la qualité des splits, et rend la procédure plus déterministe et rapide.

Les commits plus récents ont surtout ajouté:

- la remontée de progression Celery
- la correction du bug de medoid/branche dans l'affichage de l'arbre
- l'optimisation de l'export JSON avec chargement sparse et filtrage de traversée
- des tests de non-régression sur ces comportements
