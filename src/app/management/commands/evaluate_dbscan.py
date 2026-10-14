import argparse
from itertools import product

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from sklearn.metrics import silhouette_score

from admin.utils.visualisation.clusters import (
    DEFAULT_DBSCAN_METRIC,
    cluster_items_dataframe,
    estimate_dbscan_eps,
    load_items_dataframe,
    normalize_dbscan_metric,
    get_clustering_fit_vectors,
)


DATASET_CONFIGS = {
    "tfidf": {"label": "vecteurs TF-IDF"},
    "compressed": {"label": "vecteurs compresses"},
}


def parse_csv_list(raw_value: str, caster):
    return [caster(value.strip()) for value in raw_value.split(",") if value.strip()]


def compute_quality_metrics(df_items: pd.DataFrame, labels: np.ndarray, dataset_kind: str, metric: str):
    fit_vectors, _ = get_clustering_fit_vectors(df_items, dataset_kind, "dbscan", metric)
    non_noise_mask = labels != -1
    clustered_labels = labels[non_noise_mask]
    noise_ratio = float(np.mean(labels == -1))
    unique_clusters = sorted({int(label) for label in labels if label != -1})

    if len(unique_clusters) >= 2 and np.count_nonzero(non_noise_mask) > len(unique_clusters):
        silhouette = float(silhouette_score(fit_vectors[non_noise_mask], clustered_labels))
    else:
        silhouette = float("nan")

    cluster_sizes = [int(np.count_nonzero(labels == cluster_id)) for cluster_id in unique_clusters]
    return {
        "cluster_count": len(unique_clusters),
        "noise_ratio": noise_ratio,
        "silhouette": silhouette,
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
    }


class Command(BaseCommand):
    help = "Evalue plusieurs configurations DBSCAN sur les jeux de donnees de visualisation"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--dataset",
            choices=sorted(DATASET_CONFIGS.keys()),
            default="compressed",
            help="Jeu de donnees a evaluer",
        )
        parser.add_argument(
            "--eps-values",
            default="0.18,0.2,0.22,0.24",
            help="Liste de valeurs eps separees par des virgules",
        )
        parser.add_argument(
            "--min-samples-values",
            default="3,5,8,10",
            help="Liste de valeurs min_samples separees par des virgules",
        )
        parser.add_argument(
            "--metric",
            default=DEFAULT_DBSCAN_METRIC,
            help="Distance DBSCAN a utiliser",
        )
        parser.add_argument(
            "--estimate-only",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Affiche seulement l'estimation d'eps par k-distance",
        )

    def handle(self, **options) -> None:
        dataset_kind = options["dataset"]
        dataset_config = DATASET_CONFIGS[dataset_kind]
        metric = normalize_dbscan_metric(options["metric"])
        df_items = load_items_dataframe(dataset_kind)

        eps_estimate = estimate_dbscan_eps(df_items, dataset_kind, metric)
        self.stdout.write(
            self.style.SUCCESS(
                f"Estimation eps pour {dataset_config['label']} | metric={eps_estimate['metric']} | "
                f"k={eps_estimate['k']} | eps~{eps_estimate['eps']:.4f} | "
                f"q90={eps_estimate['q90']:.4f} | q95={eps_estimate['q95']:.4f} | q99={eps_estimate['q99']:.4f}"
            )
        )

        if options["estimate_only"]:
            return

        eps_values = parse_csv_list(options["eps_values"], float)
        min_samples_values = parse_csv_list(options["min_samples_values"], int)
        if not eps_values or not min_samples_values:
            raise ValueError("Les listes eps et min_samples ne doivent pas etre vides.")

        question_titles = np.array([])
        results = []
        for eps, min_samples in product(eps_values, min_samples_values):
            labels, _, _, _, _ = cluster_items_dataframe(
                df_items,
                question_titles,
                n_clusters=0,
                n_caracteristic_answers=1,
                algorithm="dbscan",
                eps=eps,
                min_samples=min_samples,
                dataset_kind=dataset_kind,
                dbscan_metric=metric,
            )
            metrics = compute_quality_metrics(df_items, labels, dataset_kind, metric)
            results.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    **metrics,
                }
            )

        results.sort(
            key=lambda row: (
                -(-1.0 if np.isnan(row["silhouette"]) else row["silhouette"]),
                row["noise_ratio"],
                -row["cluster_count"],
            )
        )

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Top configurations"))
        for row in results[:10]:
            silhouette = "nan" if np.isnan(row["silhouette"]) else f"{row['silhouette']:.4f}"
            self.stdout.write(
                f"eps={row['eps']:.4f} | min_samples={row['min_samples']} | "
                f"clusters={row['cluster_count']} | noise={row['noise_ratio']:.3f} | "
                f"silhouette={silhouette} | size_range={row['min_cluster_size']}..{row['max_cluster_size']}"
            )
