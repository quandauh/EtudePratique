### Gérer les droits d'accès aux fichiers sur le VPS

Au début vous n'aurez pas de home (et un message d'erreur avec `.bashrc`, faire `mkdir /home-reseau/<user>` et `chown -R <user>:etudiants /home-reseau/<user>`

`sudo usermod -a -G www-data <username>`

`sudo chown www-data:www-data -R /var/www/`

`sudo chmod g+rwx -R /var/www/`

relancer son shell

### Installer les fichiers sur le VPS

`cd /var/www/`

`mkdir devinsa.insa-rennes.fr`

`cd /var/www/devinsa.insa-rennes.fr/`

`sudo apt install git`

`git clone ssh://git@gitlab.insa-rennes.fr:16022/Christian.Raymond/devinsa.git .`
 (ou clone avec HTTP si ça ne fonctionne pas)
`git checkout dev-2025-2026` si c'est la branche de déploiement courante

Si aucune des commandes git ne fonctionne, téléchargez le .tar via gitlab, mettez le dans votre home puis 
`tar -xf archive.tar && cp -r * /var/www/devinsa.insa-rennes.fr && sudo chown -R www-data:www-data /var/www/devinsa.insa-rennes.fr`

### Installer les dépendances

`sudo add-apt-repository ppa:deadsnakes/ppa` (permet d'avoir des versions de Python à jour)

`sudo apt install curl python3.11 python3.11-dev python3-pip nginx mariadb-client mariadb-server libmariadb-dev celery redis pkg-config uwsgi default-libmysqlclient-dev build-essential`

`sudo mysql`


Créer la base de données

```SQL
CREATE DATABASE devinsa;
CREATE USER 'devinsa'@localhost IDENTIFIED BY 'devinsa!';
GRANT ALL ON devinsa.* TO 'devinsa'@localhost;
FLUSH PRIVILEGES;
```

Importer la DB : 

`mariadb -u devinsa -pdevinsa! < sauvegarde.sql`


### Installer Poetry (gestion des dépendances, commandes, venv, ...)

`curl -sSL https://install.python-poetry.org | python3.11 -`

Ajouter poetry à son PATH comme indiqué (la configuration du shell est dans ~/.bashrc)

`exec bash`

`poetry self add poethepoet[poetry_plugin]`

`poetry config virtualenvs.in-project true`

`cd /var/www/devinsa.insa-rennes.fr/`

`poetry env use 3.11`

`poetry install`

`echo 'export PATH="$HOME/.local/bin:$HOME/.local/share/pypoetry/venv/bin:$PATH"' >> ~/.bashrc`

`exec bash`

`poetry poe m migrate`

`cd /etc/nginx/sites-available`

`nano devinsa.insa-rennes.fr`

Copier la config (config_nginx) et garder `devinsa.insa-rennes.fr` comme `server_name` principal.
Prévoir aussi `vps-devinsa.insa-rennes.fr` si le site doit rester accessible via l'ancien nom d'hôte.

Créer le certificat SSL

`sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/nginx-selfsigned.key -out /etc/ssl/certs/nginx-selfsigned.crt`

Activer la config du site

`sudo ln -s /etc/nginx/sites-available/devinsa.insa-rennes.fr /etc/nginx/sites-enabled/devinsa.insa-rennes.fr`


### Créer les services (UWSGI & Celery)

`cd /etc/systemd/system`

copier uwsgi.service et celery.service dans ce dossier

Recharger les services

`sudo systemctl daemon-reload`

Ce service lance le serveur python, si quelque chose ne fonctionne pas il est utile de faire journalctl -u uwsgi -n 50
`sudo service uwsgi start`

`sudo service nginx start`

`sudo service celery start`

Activer le démarrage automatique avec le VPS

`sudo systemctl enable uwsgi`

`sudo systemctl enable celery`

modifier dans `/var/www/devinsa.insa-rennes.fr/src/devinsa/settings.py` le `ALLOWED_HOSTS` pour ajouter `devinsa.insa-rennes.fr` et `vps-devinsa.insa-rennes.fr` si besoin
Puis dans `settings.py`, mettre `DEBUG = False` en production

### Maintenance panel prerequisites

Si l'interface d'administration doit pouvoir lancer `git pull` puis redémarrer le service backend, l'utilisateur du worker Celery
(généralement `www-data`) doit avoir accès au dépôt Git et aux commandes `systemctl` nécessaires.

Le panneau de maintenance gère aussi le redémarrage du worker Celery.

Vérifier le chemin exact de `systemctl` :

`command -v systemctl`

Créer une règle sudoers dédiée :

`sudo visudo -f /etc/sudoers.d/devinsa-maintenance`

Avec le contenu suivant (adapter `/bin/systemctl` si la commande précédente renvoie un autre chemin) :

```sudoers
Defaults:www-data !requiretty
www-data ALL=(root) NOPASSWD: /bin/systemctl daemon-reload, /bin/systemctl restart uwsgi, /bin/systemctl is-active uwsgi, /bin/systemctl restart celery, /bin/systemctl is-active celery, /bin/systemd-run
```

Donner l'accès au dépôt à `www-data` :

`sudo chown -R www-data:www-data /var/www/devinsa.insa-rennes.fr`

Configurer les identifiants Git du remote HTTPS (exemple GitLab) :

`sudo -u www-data -H git config --global credential.helper store`

`sudo -u www-data -H sh -lc 'printf "protocol=https\nhost=gitlab.insa-rennes.fr\nusername=VOTRE_LOGIN_GITLAB\npassword=VOTRE_TOKEN_GITLAB\n\n" | git credential approve'`

Tester le pull :

`sudo -u www-data -H git -C /var/www/devinsa.insa-rennes.fr pull --ff-only`

Tester les commandes systemd sans prompt de mot de passe :

`sudo -u www-data sudo -n /bin/systemctl daemon-reload`

`sudo -u www-data sudo -n /bin/systemctl restart uwsgi`

`sudo -u www-data sudo -n /bin/systemctl is-active uwsgi`

`sudo -u www-data sudo -n /bin/systemctl restart celery`

`sudo -u www-data sudo -n /bin/systemctl is-active celery`

Tester également le lancement de l'unité détachée qui servira au redémarrage du worker depuis l'interface :

`sudo -u www-data sudo -n /bin/systemd-run --unit=devinsa-maintenance-test --collect /bin/true`

Les commandes `is-active` doivent afficher `active`.
