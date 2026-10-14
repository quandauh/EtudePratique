# Fonctionnement du projet
Informations spécifiques disponibles dans le dossier "explications" :
- [`PANORAMA_PROJET.md`](explications/PANORAMA_PROJET.md) : panorama global du projet
- [`ARBORESCENCE.md`](explications/ARBORESCENCE.md) : structure du projet.
- [`OPTIMISATIONS_GENERATION_ARBRE.md`](explications/OPTIMISATIONS_GENERATION_ARBRE.md) : refonte de l'arbre.
- [`AFFICHAGE_PROGRESSION_ARBRE.md`](explications/AFFICHAGE_PROGRESSION_ARBRE.md) : progression Celery sur l'arbre.
- [`INSTALL_VPS.md`](explications/INSTALL_VPS.md) : installation et exploitation en production.
- [`BONNES PRATIQUES.md`](explications/BONNES_PRATIQUES.md) : règles de style.

Le jeu fonctionne grâce à ces deux algorithmes, qui se basent sur les réponses de tous les anciens joueurs. 

### Arbre de décision
Un arbre est construit hors-ligne (via la page Arbre d'admin, ou en utilisant la commande `poe m create_tree <profondeur>`) et permet, durant la partie, de décider la question la plus pertinente à poser en fonction des autres réponses fournies par le joueur (voir cours d'apprentissage automatique au S8).


### Moteur de recherche TF-IDF
Chaque personnage de la base de données est représenté par un vecteur de taille 2*nombre_de_questions. A chaque question correspond deux coefficients (un pour 'oui', un pour 'non'), inspirés de la pondération TF-IDF, qui prend à la fois en compte la véracité de la réponse et son caractère discriminant.
Pour un personnage p, une question q et une réponse r ('oui' ou 'non') :


```math
\text{TF-IDF}(p, q, r) = \text{TF}(p, q, r) \times \text{IDF}(q, r) = \frac{|\text{réponses r à q pour p}|}{|\text{réponses à q pour p}|} \times \log \left(\frac{|\text{personnages ayant des réponses à q}|}{|\text{personnages ayant répondu r à q}|} \right)
```
p = personnage
q = question
r = réponse

TF(p, q, r) correspond au pourcentage de r pour la question q pour le personnage p

Exemple : TF("Einstein", "Est-il-fou ?", "Oui") = 0.3 veut dire que 30% des personnes ont répondu oui à la question "Est-il fou ?" pour le personnage "Einstein".

IDF(q, r) correspond à l'inverse de la part de personnage pour lesquels la réponse majoritaire est r à la question q. Le tout est passé dans un log pour amortir l'effet des IDF trop importants dans la pondération.

(cette formule peut être changée dans src/app/utilities/utilities.py -> fonction indexation)

A la fin de la partie, le personnage proposé par le jeu est celui dont le vecteur s'approche le plus du vecteur constitué des réponses du joueur (indépendemment du chemin emprunté dans l'arbre de décision).

# Structuration du projet
- Le projet Django (dans src) est divisé en deux applications : app (le jeu ; dans src/app) et admin (pour visualiser et modifier les données ; dans src/admin). Pour ajouter une page, il faut, dans l'application en question, créer une vue (views.py), ajouter son URL (urls.py) et créer son template html (src/templates). Les différents modèles, correspondant aux différentes tables SQL, se trouvent dans src/app/models.
- Des commandes Django customisées ont été créées à des fins utilitaires (mise à jour des TF-IDF, calcul de l'arbre, chercher des images de personnages sur wikipédia...). On peut les trouver/en ajouter d'autres dans app/management/commands. Pour en utiliser une  : `poe m <nom et arguments de la commande>`
- Concernant la partie admin, certains fichiers csv/json sont générés (dans src/static/data) et servent à stocker la projection/clusters et l'arbre calculés, ainsi que les fichiers CSV disponibles à l'export (page Exporter). Si vous avez le temps, il serait bien de revoir notamment la gestion du JSON correspondant à l'arbre (exploité par la page Arbre pour son affichage), dont le code de génération paraît peu optimisé et coûteux.

Vous pouvez voir l'arborescence complète du projet dans [ARBORESCENCE.md](explications/ARBORESCENCE.md)

# Installation du projet pour le dev

Installation des paquets nécessaires (sur Ubuntu)
 - `sudo add-apt-repository ppa:deadsnakes/ppa` pour avoir la dernière version de python
 - `sudo apt update`
 - `sudo apt install -y curl git python3.11 python3.11-dev python3-pip mariadb-server mariadb-client libmariadb-dev-compat libmariadb-dev redis-server pkg-config build-essential python3-celery`

Démarrage de MariaDB:
`sudo systemctl start mariadb`

Pour vérifier:
`sudo systemctl status mariadb`

Pour exécuter des commandes:
`sudo mariadb -e "commande"`
ou
`sudo mariadb` qui ouvre le terminal directement permettant ensuite d'enchainer les commandes facilement. NE PAS OUBLIER LE ; A CHAQUE FIN DE COMMANDE.

Initialisation de MariaDB
- `CREATE DATABASE devinsa;`
- `CREATE USER 'devinsa'@localhost IDENTIFIED BY 'devinsa!';`
- `GRANT ALL ON devinsa.* TO 'devinsa'@localhost;`
- `FLUSH PRIVILEGES;`
- `sudo mariadb -D devinsa < sauvegarde.sql`

Initialisation de poetry pour gérer les dépendances
- `curl -sSL https://install.python-poetry.org | python3.11 -`
- `poetry self add poethepoet[poetry_plugin]` si poetry est introuvable, relancer son shell
- `poetry config virtualenvs.in-project true`
- `poetry env use 3.11`
- `poetry install`
- `echo 'export PATH="$HOME/.local/bin:$HOME/.local/share/pypoetry/venv/bin:$PATH"' >> ~/.bashrc`
- `source ~/.bashrc` ou relancer son shell
- `poetry run python -m spacy download fr_core_news_sm` pour télécharger le modèle de données utilisé pour extraire les mots clés des descriptions

Lancement de commandes django
- `poe m ...`

Lancement du serveur en mode développement
- `poe m runserver`

Pour utiliser la partie Admin en local et lancer automatiquement les commandes, se déplacer dans le dossier src et faire
- `celery -A devinsa worker --loglevel=info`

# Installation en prod

Instructions dans [INSTALL_VPS.md](explications/INSTALL_VPS.md)

Identifiant de la partie Admin sur devinsa-clone :
user : admin 
password : D3v1ns4-EP

# Choses à améliorer

- Mieux gérer les données notamment au niveau de la récupération des vecteurs personnages, ... pour les avoir sous forme de matrice, ce qui permettrait d'utiliser les algorithmes de scikit et scipy directement au lieu des algos bricolés.
- Mieux gérer le stockage des vecteurs en utilisant une DB spécialisée (ChromaDB par exemple) qui fournissent en plus des méthodes performantes pour la recherche par similarité.
- Revoir les algorithmes de la partie projection / clustering pour éviter l'utilisation de fichiers CSV temporaires et écrire un code propre (actuellement le code est très difficile à comprendre car non commenté, avec des noms de variables compliqués et beaucoup de boucles / conditions imbriqués)
- Revoir la récupération des données pour l'affichage des suggestions (Trouver une solution pour accélérer le chargement de ces pages. Par exemple ajouter de la pagination, un cache, ...)
- Revoir la génération du fichier .json utiliser par la librairie d3.js pour l'affichage de l'arbre de décision. (Actuellement, générer ce fichier est plus long que générer l'arbre et cela est fait en concaténant des string au lieu d'utiliser la librairie json)

# Conseils pour améliorer la qualité du code

- Séparer les grandes fonctionnalités dans des fichiers différents. Si une partie nécessite beaucoup de sous fonctionnalités différentes, les regrouper dans un même package.
- Ecrire un code modulable/réutilisable, éviter les constantes quand c'est possible et ajouter un commentaire expliquant la constante sinon.
- Prendre le temps de comprendre et de remettre au propre les anciens bout de code en cas d'ajout interragissant avec ceux-ci, ça permet de gagner du temps et d'éviter de rajouter du code sur des bases fragiles.
- Donner des noms de variables pertinents (pas de nom style a, b, c) il vaut mieux un nom de variable un peu long qu'un nom court qu'on ne comprend pas.
- Essayer autant que possible de mettre un commentaire décrivant le rôle de chaque fonction, surtout si son nom n'est pas explicite.
- Eviter d'implémenter ses propres fonctions pour des choses qui existent déjà (pandas pour les CSV, json pour le JSON, django-orm pour la BDD, numpy, scipy et scikit-learn pour le calcul)

# Ressources utiles

- Tutoriel officiel pour débuter avec Django (en) : https://docs.djangoproject.com/en/5.0/intro/tutorial01/
- Tutoriels vidéos très complets sur Django (fr) : https://www.youtube.com/watch?v=SjRlmyEVXq4
- Tutoriel sur les services linux (non couvert par le cours de FUS/LDS) (fr) : https://www.malekal.com/creer-service-linux-systemd/
- Tutoriel sur le proxy Nginx (en) : https://nginx.org/en/docs/beginners_guide.html
- Doc du serveur web UWSGI (en) : https://uwsgi-docs.readthedocs.io/en/latest/
- Doc de Celery (gestion de tâches / progressbar) :  https://docs.celeryq.dev/en/stable/
