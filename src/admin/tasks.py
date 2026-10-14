from gc import callbacks
from time import sleep
from celery import shared_task
from celery_progress.backend import ProgressRecorder

from admin.utils.autoencoder import auto_encoder_utils
from admin.utils.autoencoder import datas
from django.conf import settings as django_settings
from django.db import transaction, connection
from django.utils import timezone

from tqdm import tqdm

from pathlib import Path
from devinsa.settings import BASE_DIR

from admin.utils.maintenance import maintenance_append_step, maintenance_run_command, maintenance_write_state
from app.models import Item, Question, Suggestion, Answer, MoulinetteRun
from django.core.cache import cache
from benchmark import utils as benchmark_utils
from app.utilities.utilities import search_best_items, search_best_items_autoencoder
import shutil
from pathlib import Path

@shared_task(bind=True)
def compute_suggestions_task(self, input_file, min_Played, max_Played, triggerValue):
    progress = ProgressRecorder(self)

    session = auto_encoder_utils.AutoEncoderSession()

    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE `tmp_suggestion`")

    session.load(input_file)

    result = session.compute()[1]

    character_to_index = datas.get_characters_to_index_table()
    question_to_index = datas.get_questions_to_index_table()

    count = 0
    total = Item.objects.filter(games__gte=min_Played, games__lte=max_Played).count()
    suggest_count = 0

    for item in tqdm(Item.objects.filter(games__gte=min_Played, games__lte=max_Played).prefetch_related("answers_set").all()):
        
        count += 1
        progress.set_progress(count, total, description=f"Suggestions pour { item.name }")

        if not item.id in character_to_index:
            continue

        char_index = character_to_index[item.id]
        
        for question in Question.objects.all():
            if not question.id in question_to_index:
                continue

            quest_index = question_to_index[question.id]

            yes_value = result[char_index][quest_index][0]
            no_value = result[char_index][quest_index][1]

            answer = Answer.objects.filter(item_id=item.id, question_id=question.id).all()

            if answer and (answer[0].yes_count * 0.75 >= answer[0].no_count or answer[0].no_count * 0.75 >= answer[0].yes_count):
                continue

            if yes_value > triggerValue and no_value <= (1-triggerValue):
                suggest_count += 1
                suggestion = Suggestion(item=item, question=question, reponse='y')
                suggestion.save()
            
            if no_value > triggerValue and yes_value <= (1-triggerValue):
                suggest_count += 1
                suggestion = Suggestion(item=item, question=question, reponse='n')
                suggestion.save()

    print(f"{suggest_count} suggestions trouvées !")
    print(f"Number of characters : { total }")
    print(f"Number of questions : { len(question_to_index) }")

from app.models import AppSettings, AutoEncoderTrainingRun
import keras
from decimal import Decimal
from time import perf_counter

class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self, task, size):
        super().__init__()
        self.task = task
        self.size = size
        self.training_started_at = None
        self.epoch_started_at = None
        self.history = []
        self.last_metrics = {}

    def _set_progress(self, current, total, description="", **extra):
        percent = 0.0
        if total > 0:
            percent = float(round((Decimal(current) / Decimal(total)) * Decimal(100), 2))
        meta = {
            "pending": False,
            "current": current,
            "total": total,
            "percent": percent,
            "description": description,
        }
        meta.update(extra)
        self.task.update_state(state="PROGRESS", meta=meta)

    def _format_metric(self, value):
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    def _format_count(self, value):
        if value is None:
            return "-"
        return str(int(round(float(value))))

    def on_train_begin(self, logs=None):
        self.training_started_at = perf_counter()
        self._set_progress(
            0,
            self.size,
            description="Preparation de l'entrainement...",
            status_line="Preparation de l'entrainement...",
            metrics={},
            history=[],
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started_at = perf_counter()
        current_epoch = epoch + 1
        self._set_progress(
            epoch,
            self.size,
            description=f"Epoch {current_epoch}/{self.size} en cours...",
            status_line=f"Epoch {current_epoch}/{self.size} en cours...",
            epoch=current_epoch,
            metrics=self.last_metrics,
            history=self.history,
        )

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        logs = logs or {}
        duration_seconds = 0.0
        if self.epoch_started_at is not None:
            duration_seconds = perf_counter() - self.epoch_started_at

        metrics = {
            "loss": float(logs["loss"]) if "loss" in logs else None,
            "false_positives": int(round(float(logs["false_positives"]))) if "false_positives" in logs else None,
            "false_negatives": int(round(float(logs["false_negatives"]))) if "false_negatives" in logs else None,
            "duration_seconds": duration_seconds,
        }
        summary = (
            f"Epoch {current_epoch}/{self.size} terminee | "
            f"loss={self._format_metric(metrics['loss'])} | "
            f"false_positives={self._format_count(metrics['false_positives'])} | "
            f"false_negatives={self._format_count(metrics['false_negatives'])} | "
            f"duree={duration_seconds:.2f}s"
        )
        self.history = (self.history + [summary])[-8:]
        self.last_metrics = metrics
        self._set_progress(
            current_epoch,
            self.size,
            description=summary,
            status_line=summary,
            epoch=current_epoch,
            metrics=metrics,
            history=self.history,
        )

    def on_train_end(self, logs=None):
        total_duration_seconds = None
        if self.training_started_at is not None:
            total_duration_seconds = perf_counter() - self.training_started_at
        self.last_metrics = {
            **self.last_metrics,
            "total_duration_seconds": total_duration_seconds,
        }
        self._set_progress(
            self.size,
            self.size,
            description=f"Entrainement termine ({self.size} epochs).",
            status_line=f"Entrainement termine ({self.size} epochs).",
            epoch=self.size,
            metrics=self.last_metrics,
            history=self.history,
        )

@shared_task(bind=True)
def train_auto_encoder_task(self, nbepochs, output_file, compression, run_id):
    settings = AppSettings.objects.get(pk=1)
    settings.ae_hidden_size = compression;
    settings.save()

    session = auto_encoder_utils.AutoEncoderSession()
    callback = CustomCallback(self, nbepochs)
    
    session.train(output_file, nb_epochs=nbepochs, callbacks=[callback])
    weights_path = auto_encoder_utils.resolve_weights_path(output_file)
    archive_dir = auto_encoder_utils.PROJECT_ROOT / "autoencoder_archives"
    archive_dir.mkdir(exist_ok=True)
    output_stem = Path(output_file).stem or "auto_encoder_weights"
    archived_weights = archive_dir / f"{output_stem}_archive_{self.request.id}.h5"
    shutil.copy2(weights_path, archived_weights)
    settings.ae_weights_path = str(archived_weights)
    settings.save()
    run = AutoEncoderTrainingRun.objects.get(pk=run_id)
    run.final_loss = callback.last_metrics.get("loss")
    run.final_false_positives = callback.last_metrics.get("false_positives")
    run.final_false_negatives = callback.last_metrics.get("false_negatives")
    run.total_duration_seconds = callback.last_metrics.get("total_duration_seconds")
    run.weights_file = str(archived_weights)
    run.save()
    auto_encoder_utils.get_cached_session(force_reload=True)
    return {
        "status": "success",
        "epochs": nbepochs,
        "output_file": output_file,
        "compression": compression,
    }

from app import utilities
from admin.utils.visualisation import generate_csv


@shared_task(bind=True)
def compute_visualisation_task(
    self,
    update_tfidf,
    update_csv_clusters_proj,
    n_clusters=4,
    n_caracteristic_answers=5,
    clustering_algorithm="kmeans",
    eps=0.2,
    min_samples=8,
):
    progress = ProgressRecorder(self)
    nb_processes = int(update_tfidf) + int(update_csv_clusters_proj)

    i = 1

    if update_tfidf:
        progress.set_progress(i, nb_processes, 'Mise à jour des TF-IDF')
        utilities.indexation()
        i += 1
    if update_csv_clusters_proj:
        progress.set_progress(i, nb_processes, f'Mise à jour des CSV, des clusters et de la projection')
        settings_model, _ = AppSettings.objects.get_or_create(pk=1)
        generate_csv.generate_all_csv(
            n_clusters,
            n_caracteristic_answers,
            clustering_algorithm=clustering_algorithm,
            eps=eps,
            min_samples=min_samples,
            weights_file=settings_model.ae_weights_path,
        )
        i += 1

from admin.utils.generate_tree_json import ecritureData
from admin.utils.tree import generation

@shared_task(bind=True)
def create_tree_task(self, update_tfidf, depth):
    progress = ProgressRecorder(self)
    nb_processes = int(update_tfidf) + 2

    i = 1

    if update_tfidf:
        progress.set_progress(i, nb_processes, 'Mise à jour des TF-IDF')
        utilities.indexation()
        i += 1

    progress.set_progress(i, nb_processes, f"Calcul de l'abre de décision à la profondeur {depth}...")
    def notify_build(msg: str):
        progress.set_progress(i, nb_processes, msg)
    rootNode = generation.create_tree(max_depth=depth, progress=notify_build)
    progress.set_progress(i, nb_processes, "Calcul de l'arbre terminé")
    i+= 1

    progress.set_progress(i, nb_processes, "Sauvegarde de l'arbre calculé...")
    # Sauvegarde en base
    def notify_save(msg: str):
        progress.set_progress(i, nb_processes, msg)
    generation.save_tree(rootNode, progress=notify_save)
    # Création d'un json (après avoir supprimé les fichiers éventuels existant)
    basedir: Path = BASE_DIR / "static/data"
    basedir.mkdir(exist_ok=True)
    tree_files: list[Path] = list(basedir.glob("tree_*"))
    if not Path(f"{basedir}/tree_{depth}.json").exists():
        for f in tree_files:
            f.unlink()
    ecritureData(depth)
    progress.set_progress(i, nb_processes, "JSON généré")


from app.utilities.utilities import enleve_description_personnage as strip_desc, ratio
from app.models import Doublon

@shared_task(bind=True)
def find_duplicates(self, trigger_value:float):
    progress = ProgressRecorder(self)

    characters = Item.objects.all()
    similarities = {}

    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE `tmp_doublon`")

    for i, character1 in enumerate(characters):
        progress.set_progress(i, len(characters), "Recherche de doublons")
        for character2 in characters:

            if character1 == character2:
                continue
            
            similarity = ratio(
                strip_desc(character1.name), 
                strip_desc(character2.name)
            )

            character_tuple = (character1.id, character2.id)
            similarities[tuple(sorted(character_tuple))] = similarity

    for character_tuple, score in similarities.items():
        if score < trigger_value:
            continue

        doublon = Doublon(item1_id=character_tuple[0], item2_id=character_tuple[1])
        doublon.save()


@shared_task(bind=True)
def run_benchmark_task(
    self,
    num_questions: int = 20,
    min_games: int = 0,
    seed: int | None = 0,
    limit: int | None = None,
    top_k: int = 3,
    item_ids: list[int] | None = None,
    min_ratio: float = benchmark_utils.MINIMUM_RATIO_VALID_QUESTION,
    use_fast_search: bool = False,
    include_autoencoder: bool = False,
    max_mistakes: int = 0,
    max_skips: int = 0,
    created_by_id: int | None = None,
    benchmark_dataset: list[dict] | None = None,
    reused_dataset_from_run_id: int | None = None,
) -> dict[str, float | int]:
    """
    Tâche de benchmark avec stratégie "1 erreur 1 passe":
    - 1 passe forward complet du benchmark
    - En cas d'erreur, retry une fois (fallback)
    """
    progress = ProgressRecorder(self)
    cache_key = f"moulinette:{self.request.id}"
    settings_model, _ = AppSettings.objects.get_or_create(pk=1)
    autoencoder_weights_path = settings_model.ae_weights_path or "auto_encoder_weights.h5"
    config = {
        "num_questions": num_questions,
        "min_games": min_games,
        "seed": seed,
        "limit": limit,
        "top_k": top_k,
        "item_ids": item_ids,
        "min_ratio": min_ratio,
        "use_fast_search": use_fast_search,
        "include_autoencoder": include_autoencoder,
        "max_mistakes": max_mistakes,
        "max_skips": max_skips,
        "autoencoder_weights_path": autoencoder_weights_path,
        "dataset_source": "reused" if benchmark_dataset is not None else "generated",
        "reused_dataset_from_run_id": reused_dataset_from_run_id,
    }

    def persist_result(payload: dict) -> dict:
        MoulinetteRun.objects.update_or_create(
            task_id=str(self.request.id),
            defaults={
                "created_by_id": created_by_id,
                "status": payload.get("status", ""),
                "config": config,
                "result": payload,
                "total": payload.get("total", 0),
                "processed": payload.get("processed", 0),
                "correct": payload.get("correct", 0),
                "incorrect": payload.get("incorrect", 0),
                "not_found": payload.get("not_found", 0),
                "accuracy": payload.get("accuracy", 0.0),
                "top_k_accuracy": payload.get("top_k_accuracy", 0.0),
                "top_5_accuracy": payload.get("top_5_accuracy", 0.0),
                "avg_tree_depth": payload.get("avg_tree_depth", 0.0),
                "avg_questions": payload.get("avg_questions", 0.0),
                "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
            },
        )
        return payload

    # Stratégie "1 erreur 1 passe" : max 2 tentatives (1 + 1 retry)
    max_attempts = 2
    
    for attempt in range(1, max_attempts + 1):
        try:
            cache.set(
                cache_key,
                {
                    "status": f"Initialisation (tentative {attempt}/{max_attempts})" if attempt > 1 else "Initialisation",
                    "total": 0,
                    "processed": 0,
                    "current_item": "",
                    "correct": 0,
                    "incorrect": 0,
                    "not_found": 0,
                    "accuracy": 0.0,
                    "top_k_accuracy": 0.0,
                    "top_5_accuracy": 0.0,
                    "avg_tree_depth": 0.0,
                    "avg_questions": 0.0,
                    "elapsed_seconds": 0.0,
                    "eta_seconds": 0.0,
                    "estimated_total_seconds": 0.0,
                    "average_seconds_per_item": 0.0,
                },
                timeout=60 * 60,
            )

            progress.set_progress(0, 5, description="Preparation des donnees...")

            answer_stats_lookup = benchmark_utils._build_answer_stats_lookup()
            progress.set_progress(1, 5, description="Chargement des reponses...")

            games_lookup = benchmark_utils._build_games_lookup() if use_fast_search else {}
            tree_nodes, tree_children = benchmark_utils._build_tree_index()
            progress.set_progress(2, 5, description="Chargement de l'arbre...")

            eligible_question_ids = list(benchmark_utils.get_liste_questions_a_poser().values_list("id", flat=True))
            progress.set_progress(3, 5, description="Chargement des questions...")

            if benchmark_dataset is not None:
                dataset = benchmark_utils.deserialize_benchmark_dataset(benchmark_dataset)
            else:
                dataset = benchmark_utils.generate_benchmark_dataset(
                    num_questions=num_questions,
                    min_games=min_games,
                    seed=seed,
                    limit=limit,
                    item_ids=item_ids,
                    min_ratio=min_ratio,
                    max_mistakes=max_mistakes,
                    max_skips=max_skips,
                    answer_stats_lookup=answer_stats_lookup,
                    tree_nodes=tree_nodes,
                    tree_children=tree_children,
                    eligible_question_ids=eligible_question_ids,
                )
            serialized_dataset = benchmark_utils.serialize_benchmark_dataset(dataset)
            total_count = len(dataset)
            config["dataset_size"] = total_count
            progress.set_progress(4, 5, description="Dataset pret")
            
            if total_count == 0:
                cache.set(
                    cache_key,
                    {
                        "status": "Aucun personnage a traiter",
                        "total": 0,
                        "processed": 0,
                        "current_item": "",
                        "correct": 0,
                        "incorrect": 0,
                        "not_found": 0,
                        "accuracy": 0.0,
                        "top_k_accuracy": 0.0,
                        "top_5_accuracy": 0.0,
                        "avg_tree_depth": 0.0,
                        "avg_questions": 0.0,
                        "elapsed_seconds": 0.0,
                        "eta_seconds": 0.0,
                        "estimated_total_seconds": 0.0,
                        "average_seconds_per_item": 0.0,
                    },
                    timeout=60 * 60,
                )
                progress.set_progress(1, 1, description="Aucun personnage a traiter")
                return persist_result(cache.get(cache_key, {}))

            progress.set_progress(0, total_count, description="Demarrage du benchmark...")

            algorithms = ["tfidf"]
            if include_autoencoder:
                algorithms.append("autoencoder")
            ranking_limit = max(max(top_k, 1), 5) - 1
            started_at = perf_counter()

            algorithm_stats = {
                "tfidf": {
                    "key": "tfidf",
                    "label": "TF-IDF rapide" if use_fast_search else "TF-IDF jeu",
                    "available": True,
                    "status": "",
                    "total": total_count,
                    "processed": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "not_found": 0,
                    "top_k_hit": 0,
                    "top_5_hit": 0,
                    "total_tree_depth": 0,
                    "total_questions": 0,
                    "duration_seconds": 0.0,
                }
            }
            if include_autoencoder:
                algorithm_stats["autoencoder"] = {
                    "key": "autoencoder",
                    "label": "Vecteurs compresses",
                    "available": True,
                    "status": "",
                    "total": total_count,
                    "processed": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "not_found": 0,
                    "top_k_hit": 0,
                    "top_5_hit": 0,
                    "total_tree_depth": 0,
                    "total_questions": 0,
                    "duration_seconds": 0.0,
                }

            def summarize_algorithm(stats: dict) -> dict:
                processed = stats["processed"]
                return {
                    "key": stats["key"],
                    "label": stats["label"],
                    "available": stats["available"],
                    "status": stats["status"],
                    "total": stats["total"],
                    "processed": processed,
                    "correct": stats["correct"],
                    "incorrect": stats["incorrect"],
                    "not_found": stats["not_found"],
                    "accuracy": stats["correct"] / processed if processed else 0.0,
                    "top_k_accuracy": stats["top_k_hit"] / processed if processed else 0.0,
                    "top_5_accuracy": stats["top_5_hit"] / processed if processed else 0.0,
                    "avg_tree_depth": stats["total_tree_depth"] / processed if processed else 0.0,
                    "avg_questions": stats["total_questions"] / processed if processed else 0.0,
                    "duration_seconds": stats["duration_seconds"],
                }

            def build_payload(status: str, processed: int, current_item: str) -> dict:
                elapsed_seconds = perf_counter() - started_at
                average_seconds_per_item = elapsed_seconds / processed if processed else 0
                eta_seconds = average_seconds_per_item * (total_count - processed)
                estimated_total_seconds = elapsed_seconds + eta_seconds
                algorithm_results = {
                    key: summarize_algorithm(stats)
                    for key, stats in algorithm_stats.items()
                }
                primary = algorithm_results["tfidf"]
                return {
                    "status": status,
                    "total": total_count,
                    "processed": processed,
                    "current_item": current_item,
                    "correct": primary["correct"],
                    "incorrect": primary["incorrect"],
                    "not_found": primary["not_found"],
                    "accuracy": primary["accuracy"],
                    "top_k_accuracy": primary["top_k_accuracy"],
                    "top_5_accuracy": primary["top_5_accuracy"],
                    "avg_tree_depth": primary["avg_tree_depth"],
                    "avg_questions": primary["avg_questions"],
                    "elapsed_seconds": elapsed_seconds,
                    "eta_seconds": eta_seconds,
                    "estimated_total_seconds": estimated_total_seconds,
                    "average_seconds_per_item": average_seconds_per_item,
                    "algorithm_results": algorithm_results,
                    "dataset_source": config["dataset_source"],
                    "dataset_size": total_count,
                    "reused_dataset_from_run_id": reused_dataset_from_run_id,
                }

            def rank_with_algorithm(algorithm: str, answers: list, tree_depth: int) -> list[int]:
                if algorithm == "tfidf" and use_fast_search:
                    return benchmark_utils._search_best_items_fast(
                        player_answers=answers,
                        depth=tree_depth,
                        num_suggestions=ranking_limit,
                        answer_stats_lookup=answer_stats_lookup,
                        games_lookup=games_lookup,
                        min_ratio=min_ratio,
                    )
                if algorithm == "tfidf":
                    return search_best_items(
                        answers,
                        depth=tree_depth,
                        num_suggestions=ranking_limit,
                    )
                return search_best_items_autoencoder(
                    answers,
                    depth=tree_depth,
                    num_suggestions=ranking_limit,
                    fallback_to_tfidf=False,
                    weights_file=autoencoder_weights_path,
                )

            for idx, case in enumerate(dataset, start=1):
                for algorithm in algorithms:
                    stats = algorithm_stats[algorithm]
                    if not stats["available"]:
                        continue

                    algorithm_started_at = perf_counter()
                    try:
                        ranking_suggestions = rank_with_algorithm(
                            algorithm,
                            case.answers,
                            case.tree_depth,
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats["available"] = False
                        stats["status"] = f"Indisponible: {str(exc)[:120]}"
                        stats["duration_seconds"] += perf_counter() - algorithm_started_at
                        continue

                    guessed_item_id = ranking_suggestions[0] if ranking_suggestions else -1
                    stats["processed"] += 1
                    stats["total_tree_depth"] += case.tree_depth
                    stats["total_questions"] += len(case.answers)
                    stats["duration_seconds"] += perf_counter() - algorithm_started_at

                    if guessed_item_id == -1:
                        stats["not_found"] += 1
                    elif guessed_item_id == case.item_id:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1

                    if top_k and top_k > 1:
                        if case.item_id in ranking_suggestions[:top_k]:
                            stats["top_k_hit"] += 1
                    else:
                        if guessed_item_id == case.item_id:
                            stats["top_k_hit"] += 1

                    if case.item_id in ranking_suggestions[:5]:
                        stats["top_5_hit"] += 1

                payload = build_payload(f"Benchmark: {case.item_name}", idx, case.item_name)
                cache.set(cache_key, payload, timeout=60 * 60)
                progress.set_progress(idx, total_count, description=f"Benchmark: {case.item_name}")

            # Succès: retourner les résultats
            payload = build_payload("Benchmark termine", total_count, "")
            payload["dataset"] = serialized_dataset
            cache.set(cache_key, payload, timeout=60 * 60)
            return persist_result(payload)

        except Exception as exc:  # noqa: BLE001
            if attempt < max_attempts:
                # Première erreur: log et retry
                print(f"Erreur lors du benchmark (tentative {attempt}/{max_attempts}): {exc}")
                print("Nouvelle tentative...")
                cache.set(
                    cache_key,
                    {
                        "status": f"Erreur tentative {attempt}, nouvelle tentative en cours...",
                        "total": 0,
                        "processed": 0,
                        "current_item": "",
                        "correct": 0,
                        "incorrect": 0,
                        "not_found": 0,
                        "accuracy": 0.0,
                        "top_k_accuracy": 0.0,
                        "top_5_accuracy": 0.0,
                        "avg_tree_depth": 0.0,
                        "avg_questions": 0.0,
                        "elapsed_seconds": 0.0,
                        "eta_seconds": 0.0,
                        "estimated_total_seconds": 0.0,
                        "average_seconds_per_item": 0.0,
                    },
                    timeout=60 * 60,
                )
                progress.set_progress(attempt, max_attempts, description=f"Erreur - nouvelle tentative ({attempt}/{max_attempts})")
                continue
            else:
                # Erreur définitive après retry
                print(f"Erreur définitive lors du benchmark (tentative finale {attempt}/{max_attempts}): {exc}")
                error_msg = f"Erreur: {str(exc)[:100]}" if str(exc) else "Erreur inconnue"
                cache.set(
                    cache_key,
                    {
                        "status": error_msg,
                        "total": 0,
                        "processed": 0,
                        "current_item": "",
                        "correct": 0,
                        "incorrect": 0,
                        "not_found": 0,
                        "accuracy": 0.0,
                        "top_k_accuracy": 0.0,
                        "top_5_accuracy": 0.0,
                        "avg_tree_depth": 0.0,
                        "avg_questions": 0.0,
                        "elapsed_seconds": 0.0,
                        "eta_seconds": 0.0,
                        "estimated_total_seconds": 0.0,
                        "average_seconds_per_item": 0.0,
                    },
                    timeout=60 * 60,
                )
                progress.set_progress(1, 1, description="Erreur définitive lors du benchmark")
                return persist_result(cache.get(cache_key, {}))


@shared_task(bind=True)
def deploy_backend_task(self, branch: str | None = None) -> dict[str, object]:
    task_id = self.request.id
    repo_dir = Path(django_settings.ADMIN_DEPLOY_REPO_DIR)
    service_name = django_settings.ADMIN_DEPLOY_SERVICE_NAME
    celery_service_name = django_settings.ADMIN_DEPLOY_CELERY_SERVICE_NAME

    maintenance_write_state(
        task_id,
        status="En cours",
        current_step="Preparation du deploiement",
        started_at=timezone.now().isoformat(),
        finished_at="",
        steps=[],
    )

    def build_git_update_command() -> list[str]:
        configured_command = list(django_settings.ADMIN_DEPLOY_GIT_COMMAND)
        if configured_command != ["git", "pull", "--ff-only"]:
            return configured_command

        if branch:
            return ["git", "pull", "--ff-only", "origin", branch]

        branch_result = maintenance_run_command(
            ["git", "branch", "--show-current"],
            cwd=repo_dir,
        )
        current_branch = branch_result.stdout.strip()
        if not current_branch:
            return configured_command
        return ["git", "pull", "--ff-only", "origin", current_branch]

    commands: list[tuple[str, list[str]]] = []
    if branch:
        commands.extend(
            [
                ("Recuperation des branches", ["git", "fetch", "origin"]),
                ("Changement de branche", ["git", "switch", branch]),
            ]
        )
    commands.extend([
        ("Mise a jour Git", build_git_update_command()),
        ("Application des migrations", list(django_settings.ADMIN_DEPLOY_MIGRATE_COMMAND)),
        ("Rechargement systemd", list(django_settings.ADMIN_DEPLOY_RELOAD_COMMAND)),
        ("Redemarrage du backend", list(django_settings.ADMIN_DEPLOY_RESTART_COMMAND)),
    ])

    for label, command in commands:
        maintenance_write_state(task_id, status="En cours", current_step=label)
        result = maintenance_run_command(command, cwd=repo_dir)
        step_status = "success" if result.returncode == 0 else "failed"
        maintenance_append_step(
            task_id,
            label=label,
            command=command,
            status=step_status,
            output=result.stdout,
            return_code=result.returncode,
        )
        if result.returncode != 0:
            return maintenance_write_state(
                task_id,
                status=f"Echec pendant {label}",
                current_step="",
                finished_at=timezone.now().isoformat(),
            )

    healthcheck_command = list(django_settings.ADMIN_DEPLOY_HEALTHCHECK_COMMAND)
    healthcheck_output = ""
    healthcheck_code = 1
    for _ in range(django_settings.ADMIN_DEPLOY_HEALTHCHECK_ATTEMPTS):
        maintenance_write_state(task_id, status="En cours", current_step="Verification du backend")
        result = maintenance_run_command(healthcheck_command, cwd=repo_dir)
        healthcheck_output = result.stdout
        healthcheck_code = result.returncode
        if result.returncode == 0 and result.stdout.strip() == "active":
            maintenance_append_step(
                task_id,
                label="Verification du backend",
                command=healthcheck_command,
                status="success",
                output=result.stdout,
                return_code=result.returncode,
            )
            break
        sleep(django_settings.ADMIN_DEPLOY_HEALTHCHECK_INTERVAL)
    else:
        maintenance_append_step(
            task_id,
            label="Verification du backend",
            command=healthcheck_command,
            status="failed",
            output=healthcheck_output,
            return_code=healthcheck_code,
        )
        return maintenance_write_state(
            task_id,
            status=f"Echec: {service_name} n'est pas actif",
            current_step="",
            finished_at=timezone.now().isoformat(),
        )

    detached_command = list(django_settings.ADMIN_DEPLOY_SYSTEMD_RUN_COMMAND) + [
        f"--unit=devinsa-maintenance-celery-{task_id}",
        "--collect",
        "--property=Type=oneshot",
        f"--working-directory={repo_dir}",
        str(repo_dir / ".venv/bin/python"),
        "src/manage.py",
        "maintenance_restart_celery",
        task_id,
    ]
    maintenance_write_state(
        task_id,
        status="En cours",
        current_step="Preparation du redemarrage du worker Celery",
    )
    detached_result = maintenance_run_command(detached_command, cwd=repo_dir)
    detached_status = "success" if detached_result.returncode == 0 else "failed"
    maintenance_append_step(
        task_id,
        label="Lancement du redemarrage detache de Celery",
        command=detached_command,
        status=detached_status,
        output=detached_result.stdout,
        return_code=detached_result.returncode,
    )
    if detached_result.returncode != 0:
        return maintenance_write_state(
            task_id,
            status="Echec pendant le lancement du redemarrage celery",
            current_step="",
            finished_at=timezone.now().isoformat(),
        )

    return maintenance_write_state(
        task_id,
        status=f"En cours: finalisation du redemarrage de {celery_service_name}",
        current_step="Redemarrage du worker Celery",
    )
