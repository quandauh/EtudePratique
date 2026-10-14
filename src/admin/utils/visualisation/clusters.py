import pandas as pd
from sklearn.cluster import *
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from admin.utils.visualisation import settings
from pathlib import Path
import csv, math
from app.models import Item
import numpy as np
from app.models import Question


DEFAULT_DBSCAN_METRIC = "cosine"
SUPPORTED_DBSCAN_METRICS = {"cosine", "euclidean", "manhattan"}


def normalize_dbscan_metric(metric: str | None) -> str:
    normalized_metric = (metric or DEFAULT_DBSCAN_METRIC).strip().lower()
    aliases = {
        "l2": "euclidean",
        "l1": "manhattan",
        "cityblock": "manhattan",
    }
    normalized_metric = aliases.get(normalized_metric, normalized_metric)
    if normalized_metric not in SUPPORTED_DBSCAN_METRICS:
        return DEFAULT_DBSCAN_METRIC
    return normalized_metric


def load_items_dataframe(dataset_kind: str = "tfidf") -> pd.DataFrame:
    if dataset_kind == "compressed":
        csv_path = settings.CHARACTER_COMPRESSED_CSV_NAME
    else:
        csv_path = settings.CHARACTER_CSV_NAME

    return pd.read_csv(
        csv_path,
        sep=settings.CSV_DELIMITER,
        header=0,
        index_col=0,
        encoding="utf-8",
    )


def get_clustering_fit_vectors(
    df_items: pd.DataFrame,
    dataset_kind: str,
    algorithm: str,
    dbscan_metric: str,
):
    del dataset_kind
    del algorithm
    normalized_metric = normalize_dbscan_metric(dbscan_metric)
    fit_vectors = df_items.to_numpy(dtype=float)
    return fit_vectors, normalized_metric


def estimate_dbscan_eps(
    df_items: pd.DataFrame,
    dataset_kind: str,
    metric: str,
    min_samples: int = 8,
):
    fit_vectors, normalized_metric = get_clustering_fit_vectors(
        df_items,
        dataset_kind,
        "dbscan",
        metric,
    )

    if fit_vectors.shape[0] == 0:
        return {
            "metric": normalized_metric,
            "k": max(1, int(min_samples)),
            "eps": 0.0,
            "q90": 0.0,
            "q95": 0.0,
            "q99": 0.0,
        }

    k = max(1, int(min_samples))
    n_neighbors = min(k, fit_vectors.shape[0])
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=normalized_metric)
    nearest_neighbors.fit(fit_vectors)
    distances, _ = nearest_neighbors.kneighbors(fit_vectors)
    k_distance = np.sort(distances[:, -1])

    q90 = float(np.quantile(k_distance, 0.90))
    q95 = float(np.quantile(k_distance, 0.95))
    q99 = float(np.quantile(k_distance, 0.99))
    return {
        "metric": normalized_metric,
        "k": n_neighbors,
        "eps": q95,
        "q90": q90,
        "q95": q95,
        "q99": q99,
    }


def cluster_items_dataframe(
    df_items: pd.DataFrame,
    question_titles: np.ndarray,
    n_clusters: int,
    n_caracteristic_answers: int,
    algorithm: str = "kmeans",
    eps: float = 0.2,
    min_samples: int = 8,
    dataset_kind: str = "tfidf",
    dbscan_metric: str = DEFAULT_DBSCAN_METRIC,
):
    algorithm = (algorithm or "kmeans").lower()
    fit_vectors, normalized_metric = get_clustering_fit_vectors(
        df_items,
        dataset_kind,
        algorithm,
        dbscan_metric,
    )
    vectors = df_items.to_numpy(dtype=float)

    if vectors.shape[0] == 0:
        empty_answers = np.empty((2 * n_caracteristic_answers, 0), dtype=object)
        return np.array([], dtype=int), np.array([], dtype=int), np.empty((settings.MEDOIDS_NB, 0), dtype=object), empty_answers, []

    if algorithm == "dbscan":
        model = DBSCAN(
            eps=float(eps),
            min_samples=max(1, int(min_samples)),
            metric=normalized_metric,
        ).fit(fit_vectors)
        labels = model.labels_.astype(int)
        cluster_ids = sorted({int(label) for label in labels if int(label) != -1})
    else:
        safe_cluster_count = max(1, min(int(n_clusters), vectors.shape[0]))
        model = KMeans(n_clusters=safe_cluster_count).fit(fit_vectors)
        labels = model.labels_.astype(int)
        cluster_ids = list(range(safe_cluster_count))

    cluster_sizes = np.array([int(np.sum(labels == cluster_id)) for cluster_id in cluster_ids], dtype=int)
    medoids = find_medoids(vectors, labels, df_items.index.values, cluster_ids=cluster_ids)

    if len(cluster_ids) == 0:
        caracteristic_answers = np.empty((2 * n_caracteristic_answers, 0), dtype=object)
    else:
        cluster_centers = np.vstack(
            [vectors[labels == cluster_id].mean(axis=0) for cluster_id in cluster_ids]
        )
        if question_titles.size == 0:
            caracteristic_answers = np.empty((2 * n_caracteristic_answers, 0), dtype=object)
        else:
            caracteristic_answers = find_caracteristic_answers(
                cluster_centers,
                question_titles,
                n_caracteristic_answers,
            )

    return labels, cluster_sizes, medoids, caracteristic_answers, cluster_ids


def find_clusters(
    n_clusters,
    n_caracteristic_answers,
    algorithm="kmeans",
    eps=0.2,
    min_samples=8,
    dbscan_metric=DEFAULT_DBSCAN_METRIC,
):
    """
    Effectue le clustering, génère les 2 CSV associés, et met à jour le champ 'cluster' dans la DB
    """
    try:
        df_items = load_items_dataframe("tfidf")
    
    except FileNotFoundError:
        print(f"Error : The file list_characters.csv does not exist !")
    
    else:
        # Les titres des questions du vecteur
        question_titles = np.array(
            Question.objects.exclude(item=None).all().values_list("title", flat=True)
        )

        labels, clusters_len, medoids, caracteristic_answers, cluster_ids = cluster_items_dataframe(
            df_items,
            question_titles,
            n_clusters,
            n_caracteristic_answers,
            algorithm=algorithm,
            eps=eps,
            min_samples=min_samples,
            dataset_kind="tfidf",
            dbscan_metric=dbscan_metric,
        )

        # Mise à jour du champ cluster dans la DB
        for item, clust in zip(Item.objects.filter(answers_set__isnull=False).distinct().order_by("name"), labels):
            item.cluster = clust
            item.save()
        
        # Générer un CSV contenant les clusters
        clusters_df = df_items.copy()
        clusters_df.insert(len(clusters_df.columns), column="Clusters", value=labels)
        clusters_df.to_csv(settings.CLUSTERS_CSV, sep=settings.CSV_DELIMITER, quoting=csv.QUOTE_MINIMAL)

        if len(cluster_ids) == 0:
            Path(settings.CLUSTERS_DATA_CSV).write_text("", encoding="utf-8")
            return

        # Générer un CSV contenant l'interprétation des clusters
        interpretation_csv = np.empty(
            shape=(1+settings.MEDOIDS_NB+2*n_caracteristic_answers, len(cluster_ids)*2),
            dtype=object
            )
        interpretation_csv.fill("")
        interpretation_csv[0, ::2] = np.vectorize(lambda n: f"{n} personnage{'s' if n > 1 else ''}")(clusters_len)
        interpretation_csv[0, 1::2] = np.array(cluster_ids, dtype=object)
        interpretation_csv[1:1+settings.MEDOIDS_NB, ::2] = medoids
        interpretation_csv[1+settings.MEDOIDS_NB:] = caracteristic_answers
        np.savetxt(settings.CLUSTERS_DATA_CSV, interpretation_csv, delimiter=settings.CSV_DELIMITER, fmt="%s")


def find_clusters_compressed(
    n_clusters,
    algorithm="kmeans",
    eps=0.2,
    min_samples=8,
    dbscan_metric=DEFAULT_DBSCAN_METRIC,
):
    """
    Effectue le clustering sur les vecteurs compressés et génère le CSV associé.
    Ne met pas à jour le champ cluster dans la DB (évite d'écraser les clusters TF-IDF).
    """
    try:
        df_items = load_items_dataframe("compressed")
    except FileNotFoundError:
        print(f"Error : The file {settings.CHARACTER_COMPRESSED_CSV_NAME} does not exist !")
    else:
        labels, _, _, _, _ = cluster_items_dataframe(
            df_items,
            np.array([]),
            n_clusters,
            n_caracteristic_answers=1,
            algorithm=algorithm,
            eps=eps,
            min_samples=min_samples,
            dataset_kind="compressed",
            dbscan_metric=dbscan_metric,
        )

        clusters_df = df_items.copy()
        clusters_df.insert(len(clusters_df.columns), column="Clusters", value=labels)
        clusters_df.to_csv(settings.CLUSTERS_COMPRESSED_CSV, sep=settings.CSV_DELIMITER, quoting=csv.QUOTE_MINIMAL)


def find_medoids(
    vectors: np.ndarray,
    labels: np.ndarray,
    names: np.array,
    n_clusters=None,
    n_medoids=settings.MEDOIDS_NB,
    cluster_ids=None,
):
    """
    Détermine les médoïdes (personnages dont la somme des distances aux autres individus est minimale)
    """
    if cluster_ids is None:
        if n_clusters is None:
            n_clusters = int(np.max(labels)) + 1 if labels.size > 0 else 0
        cluster_ids = list(range(n_clusters))

    medoids = np.empty(shape=(n_medoids, len(cluster_ids)), dtype=object)
    medoids.fill("")

    for output_idx, cluster_idx in enumerate(cluster_ids):
        # Indices des individus appartenant à ce cluster
        indices = np.where(labels == cluster_idx)[0]
        if len(indices) == 0:
            continue
        
        # Calculer la distance entre chaque paire d'individus du cluster
        distances_matrix = pairwise_distances(vectors[indices], metric="euclidean")
        
        # Distance totale de chaque individu aux autres
        total_distances = np.sum(distances_matrix, axis=1)
        
        # Chercher les individus minimisant cette grandeur
        medoid_indices = np.argsort(total_distances)[:n_medoids]
        medoid_names = names[indices][medoid_indices]

        # Les ajouter
        if len(medoid_names) < n_medoids:
            medoid_names = np.pad(medoid_names, (0, n_medoids - len(medoid_names)), constant_values="")
        medoids[:, output_idx] = medoid_names

    return medoids

def find_caracteristic_answers(centroids, question_titles, n_questions):
    """
    Détermine les questions/réponses les plus représentatives du groupe (ie les plus hauts TF-IDF)
    """
    # caracteristic_answers[yes/no, n° de question, n° de cluster, titre/valeur]
    caracteristic_answers = np.empty(shape=(2, n_questions, len(centroids), 2), dtype=object)

    for clust_idx, centroid in enumerate(centroids):
        # Réponses 'OUI'
        yes_tfidf_vector = centroid[::2]
        max_yes_indices = np.argsort(yes_tfidf_vector)[-n_questions:][::-1]
        caracteristic_answers[0, :, clust_idx, 0] = question_titles[max_yes_indices]
        caracteristic_answers[0, :, clust_idx, 1] = np.around(yes_tfidf_vector[max_yes_indices], decimals=3)

        # Réponses 'NON'
        no_tfidf_vector = centroid[1::2]
        max_no_indices = np.argsort(no_tfidf_vector)[-n_questions:][::-1]
        caracteristic_answers[1, :, clust_idx, 0] = question_titles[max_no_indices]
        caracteristic_answers[1, :, clust_idx, 1] = np.around(no_tfidf_vector[max_no_indices], decimals=3)

    return np.reshape(caracteristic_answers, newshape=(2*n_questions, 2*len(centroids)))

def parse_clusters_data_csv():
    """
    Lit le CSV contenant l'interprétation des clusters et rend une liste de clusters au format
    (nom, médoïdes, questions_oui, questions_non), lisible par la vue
    """
    try:
        clusters_data = pd.read_csv(settings.CLUSTERS_DATA_CSV, sep=settings.CSV_DELIMITER,
                                header=None, encoding='utf-8', keep_default_na=False).to_numpy()
        nb_characters = clusters_data[0, ::2]
        cluster_ids_raw = clusters_data[0, 1::2] if clusters_data.shape[1] > 1 else np.array([], dtype=object)
        medoids = clusters_data[1:1+settings.MEDOIDS_NB, ::2]
        questions = clusters_data[1+settings.MEDOIDS_NB:]
        yes_questions, no_questions = np.split(questions, 2, axis=0)

        clusters = []
        for cluster_idx in range(len(nb_characters)):
            nb_characters_in_cluster = nb_characters[cluster_idx]

            cluster_id = cluster_idx
            if cluster_idx < len(cluster_ids_raw):
                raw_value = str(cluster_ids_raw[cluster_idx]).strip()
                if raw_value != "":
                    try:
                        cluster_id = int(raw_value)
                    except ValueError:
                        cluster_id = cluster_idx

            cluster_medoids = medoids[:, cluster_idx]
            cluster_medoids = cluster_medoids[cluster_medoids != ""]

            yes_questions_cluster = yes_questions[:, 2*cluster_idx:2*cluster_idx+2]
            no_questions_cluster = no_questions[:, 2*cluster_idx:2*cluster_idx+2]
            
            clusters.append({"name":cluster_id, "nb_characters": nb_characters_in_cluster, "medoids":cluster_medoids,
                            "questions":zip(yes_questions_cluster, no_questions_cluster)})
        return clusters

    except: # Fichier inexistant, vide ou au format incorrect
        return []
