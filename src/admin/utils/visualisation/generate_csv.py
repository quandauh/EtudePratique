"""
    Ce script gère la génération des CSV utilisés pour
    la visualisation des données (vecteurs TF-IDF, clusters,
    projection 2D...), on peut les trouver dans devinsa/src/static/data,
    aux noms écrits dans settings.py.
    Attention si vous modifiez leur emplacement ou leur contenu,
    ils ne sont pas seulement téléchargeables via la page
    'Exporter' mais sont aussi utilisés par le backend d'admin pour certains calculs
"""
import csv
import pandas as pd
from pathlib import Path

from django.db.models import Count
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA, FactorAnalysis

from app.models import Answer, AppSettings, Question, Item
from admin.utils.visualisation import settings, clusters
from admin.utils.autoencoder import auto_encoder_utils


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def generate_questions_csv():
    """Générer le CSV qui contient les données sur les questions
    """
    ensure_parent_dir(settings.QUESTION_CSV_NAME)
    questions = (Question.objects
                    .values("id", "title")
                    .exclude(item=None)
                    .all()
                )

    with open(settings.QUESTION_CSV_NAME, "w", encoding='utf-8') as file:
        file_writer = csv.DictWriter(
                                    file,
                                    delimiter=settings.CSV_DELIMITER,
                                    fieldnames=list(questions[0].keys())
                                    )
        file_writer.writeheader()
        file_writer.writerows(questions)

def generate_answer_csv():
    """Générer le CSV qui contient les données sur les réponses
    """
    ensure_parent_dir(settings.ANSWER_CSV_NAME)
    characters = (Answer.objects
                    .prefetch_related("item", "question")
                    .values("question__id", "question__title", "item__name", "yes_tfidf", "no_tfidf")
                    .annotate(charCount=Count("id"))
                    .order_by()
                    .all()
                  )

    with open(settings.ANSWER_CSV_NAME, "w", encoding='utf-8') as file:
        file_writer = csv.DictWriter(
                                    file,
                                    delimiter=settings.CSV_DELIMITER,
                                    fieldnames=list(characters[0].keys())
                                    )
        file_writer.writeheader()
        file_writer.writerows(characters)

def generate_character_csv():
    """Générer le CSV contenant les données des personnages
    """
    ensure_parent_dir(settings.CHARACTER_CSV_NAME)
    
    # Header of the file containing question Yes-{id}, No-{id} for each question
    header = []

    # create the header based on the question file, containing the pair "question title", "id question"
    with open(settings.QUESTION_CSV_NAME, "r", encoding='utf-8') as file_question:
        file_reader = csv.DictReader(file_question, delimiter=settings.CSV_DELIMITER)

        for question in file_reader:
            header.append(f"Yes-{question['title']} ({question['id']})")
            header.append(f"No-{question['title']} ({question['id']})")

    # using the file_answer.csv file (containing the charac name, TFIDF value, question id and title),
    # generate a dictionnary of {"charac":[TFIDF value]}
    with open(settings.ANSWER_CSV_NAME, "r", encoding='utf-8') as file_answer:
        file_reader = csv.DictReader(file_answer, delimiter=settings.CSV_DELIMITER)
        line = {}
        for answer_n in [row for row in file_reader]:
            if answer_n['item__name'] not in line:
                line[answer_n['item__name']] = [0]*len(header)

            index = header.index(f"Yes-{answer_n['question__title']} ({answer_n['question__id']})")
            line[answer_n['item__name']][index] = answer_n['yes_tfidf']
            line[answer_n['item__name']][index+1] = answer_n['no_tfidf']

    # write all the rows into the character file.
    with open(settings.CHARACTER_CSV_NAME, 'w', encoding="utf-8") as file:
        file_writer = csv.writer(file,
                                delimiter=settings.CSV_DELIMITER,
                                doublequote=True,
                                quoting=csv.QUOTE_ALL
                                )
        file_writer.writerow(["Name"] + header)
        file_writer.writerows([[character.strip()] + tfidf for character, tfidf in line.items()])


def generate_character_compressed_csv(weights_file: str | None = None, retry_count: int = 1) -> bool:
    """Générer le CSV contenant les vecteurs compressés des personnages.
    
    Stratégie "1 erreur 1 passe":
    - 1 passe forward de la génération complète
    - En cas d'erreur (exception), retry une fois (retry_count=1 par défaut)
    
    Args:
        weights_file: Chemin vers les poids de l'autoencoder
        retry_count: Nombre de tentatives en cas d'erreur (défaut: 1)
    
    Retourne:
        True si la génération a réussi, sinon False.
    """
    ensure_parent_dir(settings.CHARACTER_COMPRESSED_CSV_NAME)
    if weights_file is None:
        app_settings, _ = AppSettings.objects.get_or_create(pk=1)
        weights_file = app_settings.ae_weights_path or "auto_encoder_weights.h5"
    weights_file = auto_encoder_utils.resolve_weights_path(weights_file)

    if not Path(weights_file).exists():
        print(f"Autoencoder weights not found: {weights_file}")
        return False

    max_attempts = 1 + retry_count
    for attempt in range(1, max_attempts + 1):
        try:
            session = auto_encoder_utils.AutoEncoderSession()
            session.load(weights_file)
            session.compute()

            if session.encoded_matrix is None:
                raise ValueError("Encoded matrix is empty.")

            encoded_matrix = session.encoded_matrix
            header = [f"v{i}" for i in range(encoded_matrix.shape[1])]

            # Associer les noms de personnages à l'ordre des vecteurs
            id_to_name = dict(Item.objects.values_list("id", "name"))
            names_by_index = ["" for _ in range(session.characters_count)]
            for item_id, idx in session.characters_to_index.items():
                names_by_index[idx] = id_to_name.get(item_id, "")

            with open(settings.CHARACTER_COMPRESSED_CSV_NAME, 'w', encoding="utf-8") as file:
                file_writer = csv.writer(
                    file,
                    delimiter=settings.CSV_DELIMITER,
                    doublequote=True,
                    quoting=csv.QUOTE_ALL
                )
                file_writer.writerow(["Name"] + header)
                file_writer.writerows(
                    [[name] + vector.tolist() for name, vector in zip(names_by_index, encoded_matrix)]
                )

            return True
            
        except Exception as e:
            if attempt < max_attempts:
                print(f"Erreur génération vecteurs compressés (tentative {attempt}/{max_attempts}): {e}")
                print(f"Nouvelle tentative...")
                continue
            else:
                print(f"Erreur génération vecteurs compressés (tentative finale {attempt}/{max_attempts}): {e}")
                print("Echec définitif - vecteurs compressés non générés.")
                return False

def generate_2D_projection_csv():
    """Générer le fichier contenant la projection 2D des personnages
    """
    ensure_parent_dir(settings.VISUALISATION_CSV)
    file_clusters = pd.read_csv(
                                settings.CLUSTERS_CSV,
                                sep=settings.CSV_DELIMITER,
                                header=0,
                                encoding='utf-8'
                                )

    # get the header and remove the Name and Cluster column
    header = file_clusters.columns.to_list()[1:-1]

    # Get all characters
    y = file_clusters.loc[:,'Name'].values
    # Get the tfidf value assign for each charac
    x = file_clusters.loc[:, header].values
    cluster = file_clusters.loc[:,'Clusters'].values # get all the cluster assigned by the previous clusterization

    pca = TSNE(n_components=2, init='pca')
    x_r = pca.fit_transform(x)

    fieldName = ["Axe_X", "Axe_Y", "Name", "Cluster"]
    toWrite = [
        {
            fieldName[0]: x[0],
            fieldName[1]: x[1],
            fieldName[2]: y[i], 
            fieldName[3]: cluster[i]
        } 
        for i, x in enumerate(x_r)
    ]
    
    with open(settings.VISUALISATION_CSV, "w", encoding='utf-8') as fileVisualisation:
        fileWriter = csv.DictWriter(fileVisualisation, fieldnames=fieldName, quoting=csv.QUOTE_MINIMAL, delimiter=settings.CSV_DELIMITER)
        fileWriter.writeheader()
        fileWriter.writerows(toWrite)


def generate_2D_projection_csv_compressed():
    """Générer le fichier contenant la projection 2D des personnages (vecteurs compressés)
    """
    ensure_parent_dir(settings.VISUALISATION_COMPRESSED_CSV)
    file_clusters = pd.read_csv(
        settings.CLUSTERS_COMPRESSED_CSV,
        sep=settings.CSV_DELIMITER,
        header=0,
        encoding='utf-8'
    )

    header = file_clusters.columns.to_list()[1:-1]
    y = file_clusters.loc[:, 'Name'].values
    x = file_clusters.loc[:, header].values
    cluster = file_clusters.loc[:, 'Clusters'].values

    pca = TSNE(n_components=2, init='pca')
    x_r = pca.fit_transform(x)

    fieldName = ["Axe_X", "Axe_Y", "Name", "Cluster"]
    toWrite = [
        {
            fieldName[0]: x[0],
            fieldName[1]: x[1],
            fieldName[2]: y[i],
            fieldName[3]: cluster[i]
        }
        for i, x in enumerate(x_r)
    ]

    with open(settings.VISUALISATION_COMPRESSED_CSV, "w", encoding='utf-8') as fileVisualisation:
        fileWriter = csv.DictWriter(fileVisualisation, fieldnames=fieldName, quoting=csv.QUOTE_MINIMAL, delimiter=settings.CSV_DELIMITER)
        fileWriter.writeheader()
        fileWriter.writerows(toWrite)

def generate_all_csv(
    n_clusters=4,
    n_caracteristic_answers=5,
    clustering_algorithm="kmeans",
    eps=0.2,
    min_samples=8,
    weights_file: str | None = None,
):
    print("Génération du CSV de questions...")
    generate_questions_csv()

    print("Génération du CSV de réponses...")
    generate_answer_csv()

    print("Génération du CSV de personnages...")
    generate_character_csv()

    if clustering_algorithm == "dbscan":
        print(f"Détections de clusters DBSCAN eps={eps}, min_samples={min_samples}...")
    else:
        print(f"Détections de clusters n_clusters = {n_clusters}...")
    clusters.find_clusters(
        n_clusters,
        n_caracteristic_answers,
        algorithm=clustering_algorithm,
        eps=eps,
        min_samples=min_samples,
    )

    print("Génération du CSV des vecteurs projetés en 2D...")
    generate_2D_projection_csv()

    print("Génération des vecteurs compressés...")
    if generate_character_compressed_csv(weights_file=weights_file):
        if clustering_algorithm == "dbscan":
            print(f"Détections de clusters (compressés) DBSCAN eps={eps}, min_samples={min_samples}...")
        else:
            print(f"Détections de clusters (compressés) n_clusters = {n_clusters}...")
        clusters.find_clusters_compressed(
            n_clusters,
            algorithm=clustering_algorithm,
            eps=eps,
            min_samples=min_samples,
        )

        print("Génération du CSV des vecteurs compressés projetés en 2D...")
        generate_2D_projection_csv_compressed()
    else:
        print("Skip compressed visualisation: poids autoencodeur introuvables ou encodage impossible.")

    
