import os
import shlex
from pathlib import Path

from django.conf import settings as django_settings
from django.contrib.auth.decorators import login_required
from django.core.handlers.wsgi import WSGIRequest
from django.core.cache import cache
from django.db import transaction
from django.db.models import QuerySet, F, Avg, Count, Q, Case, When, IntegerField, Sum
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden, HttpResponseNotAllowed, HttpResponseNotFound, FileResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify

from admin.utils.suggestions.compute_distances import compute_distances
from admin.utils.suggestions.find_similarities import find_similarities
from admin.utils.suggestions.find_best_suggestions import find_best_suggestions
from admin.utils.suggestions.generate_suggestions_based_on_description import generate_suggestions_based_on_descriptions
from admin.utils.visualisation.clusters import parse_clusters_data_csv
from app.models import AppSettings, AutoEncoderTrainingRun, Doublon, Item, Answer, Question, Suggestion, MoulinetteRun
from app.forms import ItemForm
from devinsa.settings import BASE_DIR
from subprocess import run, Popen, PIPE, TimeoutExpired
from app.management.commands.fetch_images_url import (
    extract_thumbnail_url,
    get_wikipedia_name,
    query_wikipedia_api,
)
import json
import math
import re
import tempfile
from admin.utils.autoencoder.datas import get_questions_by_suggestions_count

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import random

from admin.utils.maintenance import maintenance_read_state
from admin.utils.system_metrics import collect_system_metrics


def _require_superuser(request: HttpRequest) -> HttpResponse | None:
    if request.user.is_superuser:
        return None
    return HttpResponseForbidden("Superuser access required.")


def _default_autoencoder_output_file() -> str:
    stamp = timezone.now().strftime("%Y%m%d_%H%M%S")
    return f"auto_encoder_weights_{stamp}.h5"


def _normalize_autoencoder_output_file(output_file: str, run_id: int) -> str:
    requested = Path((output_file or "").strip()).name
    stem = slugify(Path(requested).stem) or "auto_encoder_weights"
    suffix = Path(requested).suffix.lower()
    if suffix != ".h5":
        suffix = ".h5"
    stamp = timezone.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{stem}_{stamp}_run_{run_id}{suffix}"


def _normalize_model_path(path: str | None) -> str:
    if not path:
        return ""
    try:
        from admin.utils.autoencoder.auto_encoder_utils import resolve_weights_path

        return str(Path(resolve_weights_path(path)).resolve())
    except Exception:
        return str(Path(path).expanduser())


def _get_autoencoder_defaults(request: HttpRequest) -> dict[str, int | str]:
    return {
        "iterations": request.session.get("autoencoder_iterations", 100),
        "output_file": request.session.get("autoencoder_output_file", _default_autoencoder_output_file()),
        "hidden_size": request.session.get("autoencoder_hidden_size", AppSettings.objects.get(pk=1).ae_hidden_size),
    }


def _get_autoencoder_training_runs() -> list[dict[str, object]]:
    runs = []
    settings_model, _ = AppSettings.objects.get_or_create(pk=1)
    active_model_path = _normalize_model_path(settings_model.ae_weights_path)
    for run in AutoEncoderTrainingRun.objects.all():
        run_model_path = _normalize_model_path(run.weights_file)
        runs.append(
            {
                "id": run.id,
                "created_at": run.created_at,
                "nb_epochs": run.nb_epochs,
                "hidden_size": run.hidden_size,
                "output_file": run.output_file,
                "final_loss_display": "-" if run.final_loss is None else f"{run.final_loss:.4f}",
                "final_false_positives_display": "-" if run.final_false_positives is None else str(run.final_false_positives),
                "final_false_negatives_display": "-" if run.final_false_negatives is None else str(run.final_false_negatives),
                "total_duration_display": "-" if run.total_duration_seconds is None else f"{run.total_duration_seconds:.2f} s",
                "weights_file": run.weights_file,
                "is_active_model": bool(run_model_path) and run_model_path == active_model_path,
            }
        )
    return runs

def _maintenance_context() -> dict[str, object]:
    repo_dir = Path(django_settings.ADMIN_DEPLOY_REPO_DIR)
    current_branch = ""
    branch_options: list[str] = []

    try:
        run(
            ["git", "fetch", "--prune", "--quiet", "origin"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, TimeoutExpired):
        pass

    try:
        current_branch = run(
            ["git", "branch", "--show-current"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        branches_output = run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        remote_branches_output = run(
            ["git", "branch", "--remotes", "--format=%(refname:short)"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        local_branches = [
            branch.strip()
            for branch in branches_output.splitlines()
            if branch.strip() and re.fullmatch(r"[A-Za-z0-9._/-]+", branch.strip())
        ]
        remote_branches = [
            branch.strip().removeprefix("origin/")
            for branch in remote_branches_output.splitlines()
            if branch.strip().startswith("origin/")
            and branch.strip() != "origin/HEAD"
            and re.fullmatch(r"[A-Za-z0-9._/-]+", branch.strip().removeprefix("origin/"))
        ]
        branch_options = list(dict.fromkeys(local_branches + remote_branches))
    except OSError:
        branch_options = []

    if current_branch and current_branch not in branch_options:
        branch_options.insert(0, current_branch)

    return {
        "repo_dir": django_settings.ADMIN_DEPLOY_REPO_DIR,
        "service_name": django_settings.ADMIN_DEPLOY_SERVICE_NAME,
        "celery_service_name": django_settings.ADMIN_DEPLOY_CELERY_SERVICE_NAME,
        "selected_branch": current_branch,
        "branch_options": branch_options,
        "error": "",
        "command_preview": [
            {
                "label": "Mise a jour Git",
                "command": shlex.join(django_settings.ADMIN_DEPLOY_GIT_COMMAND),
            },
            {
                "label": "Application des migrations",
                "command": shlex.join(django_settings.ADMIN_DEPLOY_MIGRATE_COMMAND),
            },
            {
                "label": "Rechargement systemd",
                "command": shlex.join(django_settings.ADMIN_DEPLOY_RELOAD_COMMAND),
            },
            {
                "label": "Redemarrage du backend",
                "command": shlex.join(django_settings.ADMIN_DEPLOY_RESTART_COMMAND),
            },
            {
                "label": "Verification du backend",
                "command": shlex.join(django_settings.ADMIN_DEPLOY_HEALTHCHECK_COMMAND),
            },
            {
                "label": "Redemarrage du worker Celery",
                "command": shlex.join(django_settings.ADMIN_DEPLOY_CELERY_RESTART_COMMAND),
            },
            {
                "label": "Verification du worker Celery",
                "command": shlex.join(django_settings.ADMIN_DEPLOY_CELERY_HEALTHCHECK_COMMAND),
            },
        ],
    }


def _render_task_page(
    request: HttpRequest,
    task_id: str,
    redirection: str,
    *,
    task_title: str = "",
    task_message: str = "",
    task_ui: str = "",
) -> HttpResponse:
    return render(
        request,
        "admin/task.html",
        {
            "task_id": task_id,
            "redirection": redirection,
            "task_title": task_title,
            "task_message": task_message,
            "task_ui": task_ui,
        },
    )


@login_required
def index(request: HttpRequest) -> HttpResponse:
    request.session.pop("perso", None)

    clusters = parse_clusters_data_csv()
    return statistics(request)
    #return render(request, "admin/index.html", context={"clusters": clusters})

@login_required
def autoencodeur(request: HttpRequest) -> HttpResponse:
    request.session.pop("perso", None)

    # Extraire l'interprétation des clusters (médoïdes, questions caractéristiques) du CSV
    clusters = parse_clusters_data_csv()
    return render(
        request,
        "admin/visualisation.html",
        context={
            "clusters": clusters,
            "autoencoder_defaults": _get_autoencoder_defaults(request),
        },
    )

@login_required
def toggle_algorithm_mode(request: HttpRequest) -> HttpResponse:
    wants_json = request.headers.get("x-requested-with") == "XMLHttpRequest"
    selected_mode = None
    if request.method == "POST":
        mode = request.POST.get("mode")
        if mode in ["tfidf", "autoencoder"]:
            settings, _ = AppSettings.objects.get_or_create(pk=1)
            settings.algorithm_mode = mode
            settings.save()
            selected_mode = mode
    if wants_json:
        settings, _ = AppSettings.objects.get_or_create(pk=1)
        return JsonResponse(
            {
                "ok": selected_mode is not None,
                "mode": settings.algorithm_mode,
                "label": "Auto-encodeur" if settings.algorithm_mode == "autoencoder" else "TF-IDF",
            }
        )
    return redirect("admin:admin_index")

@login_required
def tree(request: HttpRequest | WSGIRequest) -> HttpResponse:
    basedir: Path = BASE_DIR / "static/data"
    basedir.mkdir(exist_ok=True)
    tree_files: list[Path] = list(basedir.glob("tree_*"))
    
    if request.method == "GET":
        # S'il existe un arbre, ouvrir le json pour avoir ses informations
        if len(tree_files) > 0:
            depth = int(tree_files[0].stem.split("_")[1])
            with open(tree_files[0], 'r') as file:
                data = json.load(file)
                num_nodes = len(data)
        else:
            # aucun arbre n'a été généré
            depth = 0
            num_nodes = 0

        log2_num_nodes = math.log(num_nodes+1, 2)
        
        return render(request, "admin/tree.html", {"tree_url": f"data/tree_{depth}.json",
                                                   "depth": depth, "num_nodes": num_nodes, "log2_num_nodes": log2_num_nodes})
    elif request.method == "POST":
        depth = request.POST.get("depth", 15)
        update_tfidf = request.POST.get("update_tfidf")

        from admin.tasks import create_tree_task;
        result = create_tree_task.delay(
        update_tfidf = update_tfidf == 'on',
        depth = int(depth)
    )

        return _render_task_page(
            request,
            result.task_id,
            reverse('admin:tree'),
        )
    else:
        return HttpResponseNotFound()

@login_required
def validate_suggestions(request: HttpRequest | WSGIRequest) -> HttpResponse:
    if request.method == "POST":
        if request.session.get("question"):
            # Vider le cache
            get_questions_by_suggestions_count.cache_clear()

            question_id: int = request.session["question"]
            question: Question = Question.objects.get(id=question_id)
            type=None
            for key in request.POST.keys():
                if not key.startswith("checked-"):
                    continue

                suggestionId = int(key.split("-")[1])
                suggestion = Suggestion.objects.get(pk=suggestionId)
                answer = Answer.objects.filter(item=suggestion.item, question=suggestion.question)
                type = suggestion.reponse
                
                if not answer.exists():
                    answer = Answer(item=suggestion.item, question=suggestion.question)
                else:
                    answer = answer.get()

                if request.POST[key] == "y":
                    answer.yes_count += 1
                    answer.save()
                elif request.POST[key] == "n":
                    answer.no_count += 1
                    answer.save()
                
            
            question.suggestion_set.filter(reponse=type).all().delete()
            del request.session["question"]

        return redirect("admin:suggestions")


@login_required
def statistics(request):
    num_characters = Item.objects.count()
    num_questions = Question.objects.count()
    num_answers = Answer.objects.count()
    most_played = Item.objects.order_by('-games')[:5]
    recent_characters = Item.objects.order_by('-id')[:5]
    yes_count = Answer.objects.aggregate(total=Sum('yes_count'))['total'] or 0
    no_count = Answer.objects.aggregate(total=Sum('no_count'))['total'] or 0
    pass_count = Answer.objects.aggregate(total=Sum('pass_count'))['total'] or 0

    admin_messages = {
        'upload_error': request.session.pop('admin_upload_error', None),
        'upload_success': request.session.pop('admin_upload_success', None),
        'reload_error': request.session.pop('admin_reload_error', None),
        'reload_success': request.session.pop('admin_reload_success', None),
    }

    return render(request, "admin/statistics.html", {
        "num_characters": num_characters,
        "num_questions": num_questions,
        "num_answers": num_answers,
        "most_played": most_played,
        "recent_characters": recent_characters,
        "yes_count": yes_count,
        "no_count": no_count,
        "pass_count": pass_count,
        "admin_messages": admin_messages,
        "system_metrics": collect_system_metrics(),
    })


@login_required
def moulinette(request: HttpRequest | WSGIRequest) -> HttpResponse:
    defaults = {
        "num_questions": 20,
        "min_games": 0,
        "seed": 0,
        "limit": "",
        "top_k": 3,
        "min_ratio": 0.75,
        "item_ids": "",
        "use_fast_search": False,
        "include_autoencoder": True,
        "max_mistakes": 0,
        "max_skips": 0,
    }

    if request.method == "GET":
        return render(
            request,
            "admin/moulinette.html",
            {
                "defaults": defaults,
                "moulinette_runs": MoulinetteRun.objects.select_related("created_by")[:20],
            },
        )

    if request.method != "POST":
        return HttpResponseNotAllowed(["GET", "POST"])

    item_ids_raw = (request.POST.get("item_ids") or "").strip()
    item_ids: list[int] | None = None
    if item_ids_raw:
        try:
            item_ids = [int(val.strip()) for val in item_ids_raw.split(",") if val.strip()]
        except ValueError:
            return render(
                request,
                "admin/moulinette.html",
                {
                    "defaults": {**defaults, "item_ids": item_ids_raw},
                    "error": "Invalid item ids format. Use comma-separated integers.",
                    "moulinette_runs": MoulinetteRun.objects.select_related("created_by")[:20],
                },
            )

    def _to_int(value: str, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _to_float(value: str, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    num_questions = _to_int(request.POST.get("num_questions"), defaults["num_questions"])
    min_games = _to_int(request.POST.get("min_games"), defaults["min_games"])
    seed = _to_int(request.POST.get("seed"), defaults["seed"])
    limit_val = request.POST.get("limit")
    limit = _to_int(limit_val, 0) if limit_val else None
    limit = limit if limit and limit > 0 else None
    top_k = _to_int(request.POST.get("top_k"), defaults["top_k"])
    min_ratio = _to_float(request.POST.get("min_ratio"), defaults["min_ratio"])
    use_fast_search = request.POST.get("use_fast_search") == "on"
    include_autoencoder = request.POST.get("include_autoencoder") == "on"
    max_mistakes = _to_int(request.POST.get("max_mistakes"), defaults["max_mistakes"])
    max_skips = _to_int(request.POST.get("max_skips"), defaults["max_skips"])
    settings_model, _ = AppSettings.objects.get_or_create(pk=1)
    autoencoder_weights_path = settings_model.ae_weights_path or "auto_encoder_weights.h5"

    from admin.tasks import run_benchmark_task
    result = run_benchmark_task.delay(
        num_questions=num_questions,
        min_games=min_games,
        seed=seed,
        limit=limit,
        top_k=top_k,
        item_ids=item_ids,
        min_ratio=min_ratio,
        use_fast_search=use_fast_search,
        include_autoencoder=include_autoencoder,
        max_mistakes=max_mistakes,
        max_skips=max_skips,
        created_by_id=request.user.id,
    )

    return render(
        request,
        "admin/moulinette_run.html",
        {
            "task_id": result.task_id,
            "config": {
                "num_questions": num_questions,
                "min_games": min_games,
                "seed": seed,
                "limit": limit,
                "top_k": top_k,
                "min_ratio": min_ratio,
                "item_ids": item_ids_raw,
                "use_fast_search": use_fast_search,
                "include_autoencoder": include_autoencoder,
                "max_mistakes": max_mistakes,
                "max_skips": max_skips,
                "autoencoder_weights_path": autoencoder_weights_path,
                "dataset_source": "generated",
                "reused_dataset_from_run_id": None,
            },
        },
    )


@login_required
def rerun_moulinette_run(request: HttpRequest, run_id: int) -> HttpResponse:
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    previous_run = get_object_or_404(MoulinetteRun, id=run_id)
    dataset = previous_run.result.get("dataset") if previous_run.result else None
    if not dataset:
        return render(
            request,
            "admin/moulinette.html",
            {
                "defaults": {
                    "num_questions": 20,
                    "min_games": 0,
                    "seed": 0,
                    "limit": "",
                    "top_k": 3,
                    "min_ratio": 0.75,
                    "item_ids": "",
                    "use_fast_search": False,
                    "include_autoencoder": True,
                    "max_mistakes": 0,
                    "max_skips": 0,
                },
                "error": "Ce run ne contient pas de dataset rejouable.",
                "moulinette_runs": MoulinetteRun.objects.select_related("created_by")[:20],
            },
        )

    config = {
        "num_questions": 20,
        "min_games": 0,
        "seed": 0,
        "limit": None,
        "top_k": 3,
        "min_ratio": 0.75,
        "item_ids": "",
        "use_fast_search": False,
        "include_autoencoder": True,
        "max_mistakes": 0,
        "max_skips": 0,
        **previous_run.config,
        "dataset_source": "reused",
        "reused_dataset_from_run_id": previous_run.id,
    }
    settings_model, _ = AppSettings.objects.get_or_create(pk=1)
    config["autoencoder_weights_path"] = settings_model.ae_weights_path or "auto_encoder_weights.h5"
    from admin.tasks import run_benchmark_task
    result = run_benchmark_task.delay(
        num_questions=config.get("num_questions", 20),
        min_games=config.get("min_games", 0),
        seed=config.get("seed", 0),
        limit=config.get("limit"),
        top_k=config.get("top_k", 3),
        item_ids=config.get("item_ids"),
        min_ratio=config.get("min_ratio", 0.75),
        use_fast_search=config.get("use_fast_search", False),
        include_autoencoder=config.get("include_autoencoder", True),
        max_mistakes=config.get("max_mistakes", 0),
        max_skips=config.get("max_skips", 0),
        created_by_id=request.user.id,
        benchmark_dataset=dataset,
        reused_dataset_from_run_id=previous_run.id,
    )

    return render(
        request,
        "admin/moulinette_run.html",
        {
            "task_id": result.task_id,
            "config": config,
        },
    )


@login_required
def delete_moulinette_run(request: HttpRequest, run_id: int) -> HttpResponse:
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    MoulinetteRun.objects.filter(id=run_id).delete()
    return redirect("admin:moulinette")


@login_required
def moulinette_status(request: HttpRequest, task_id: str) -> JsonResponse:
    data = cache.get(
        f"moulinette:{task_id}",
        {
            "status": "En attente",
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
            "algorithm_results": {},
            "dataset_source": "",
            "dataset_size": 0,
        },
    )
    return JsonResponse(data)


@login_required
def maintenance(request: HttpRequest | WSGIRequest) -> HttpResponse:
    forbidden = _require_superuser(request)
    if forbidden is not None:
        return forbidden

    if request.method == "GET":
        return render(request, "admin/maintenance.html", _maintenance_context())

    if request.method != "POST":
        return HttpResponseNotAllowed(["GET", "POST"])

    branch = (request.POST.get("branch") or "").strip()
    if branch and not re.fullmatch(r"[A-Za-z0-9._/-]+", branch):
        context = _maintenance_context()
        context["error"] = "Nom de branche invalide."
        context["selected_branch"] = branch
        return render(request, "admin/maintenance.html", context)

    from admin.tasks import deploy_backend_task

    result = deploy_backend_task.delay(branch=branch or None)
    return redirect("admin:maintenance_run", task_id=result.task_id)


@login_required
def maintenance_run(request: HttpRequest, task_id: str) -> HttpResponse:
    forbidden = _require_superuser(request)
    if forbidden is not None:
        return forbidden

    context = _maintenance_context()
    context["task_id"] = task_id
    return render(request, "admin/maintenance_run.html", context)


@login_required
def maintenance_status(request: HttpRequest, task_id: str) -> JsonResponse:
    forbidden = _require_superuser(request)
    if forbidden is not None:
        return JsonResponse({"detail": "Forbidden"}, status=403)

    payload = maintenance_read_state(task_id)
    return JsonResponse(payload)







@login_required
def suggestions(request: HttpRequest | WSGIRequest) -> HttpResponse:
    """_summary_

    Args:
        request (HttpRequest | WSGIRequest): _description_

    Returns:
        HttpResponse: _description_
    """
    """
    La vue commence par récupérer toutes les questions avec leurs suggestions 
    associées, les trier par nombre de suggestions, 
    et les stocker dans la variable questions.
    """
    # Si la méthode de la requête est "POST" => un formulaire a été soumis
    if request.method == "POST":
        question_id: int = request.POST.get("id_question")
        suggestion_type: str = request.POST.get("suggestion_type")
        question: Question = Question.objects.get(id=question_id)

        # stocke l'ID dans la session de l'utilisateur
        request.session["question"] = question.id
        request.session["suggestion_type"] = suggestion_type
        
        # redirige vers la même vue pour afficher les suggestions de cette question.
        return redirect("admin:suggestions")

    questions_yes, questions_no = get_questions_by_suggestions_count()

    # Si la session ne contient pas encore d'ID de question,la 1er question est la question actuelle.
    if not request.session.get("suggestion_type"):
        request.session["suggestion_type"] = "y"  # Par défaut, afficher les suggestions "Oui"

    question = None
    if request.session.get("question"):
        question = Question.objects.get(id=request.session["question"])

    # Séparer les suggestions avec réponses "Oui" et "Non"
    suggestions = None
    if question:
        if request.session["suggestion_type"] == "y":
            suggestions = question.suggestion_set.filter(reponse='y')
        else:
            suggestions = question.suggestion_set.filter(reponse='n')

    return render(
        request,
        "admin/suggestions.html",
        {
            "questions_yes": questions_yes,
            "questions_no": questions_no,
            "current_question": question,
            "suggestions": suggestions,
            "autoencoder_defaults": _get_autoencoder_defaults(request),
            "autoencoder_training_runs": _get_autoencoder_training_runs(),
        },
    )

@login_required
def duplicates(request: HttpRequest | WSGIRequest) -> HttpResponse:
    """_summary_

    Args:
        request (HttpRequest | WSGIRequest): _description_

    Returns:
        HttpResponse: _description_
    """

    with transaction.atomic():
        if request.method == "POST":
            if "validate-item1" in request.POST:
                doublon = Doublon.objects.get(pk=request.POST["validate-item1"])
                perso1: Item = doublon.item1
                perso2: Item = doublon.item2

                perso1.games += perso2.games

                for answer2 in perso2.answers_set.all():
                    answer1 = Answer.objects.filter(item_id=perso1.id, question_id=answer2.question_id)
                    
                    if not answer1.exists():
                        answer1 = Answer(item_id=perso1.id, question_id=answer2.question_id)
                    else:
                        answer1 = answer1.get()

                    answer1.yes_count += answer2.yes_count
                    answer1.no_count += answer2.no_count
                    answer1.pass_count += answer2.pass_count

                    answer1.save()

                perso1.save()
                perso2.delete()
                doublon.delete()
            elif "validate-item2" in request.POST:
                doublon = Doublon.objects.get(pk=request.POST["validate-item2"])
                perso2: Item = doublon.item1
                perso1: Item = doublon.item2

                perso1.games += perso2.games

                for answer2 in perso2.answers_set.all():
                    answer1 = Answer.objects.filter(item_id=perso1.id, question_id=answer2.question_id)
                    
                    if not answer1.exists():
                        answer1 = Answer(item_id=perso1.id, question_id=answer2.question_id)
                    else:
                        answer1 = answer1.get()

                    answer1.yes_count += answer2.yes_count
                    answer1.no_count += answer2.no_count
                    answer1.pass_count += answer2.pass_count

                    answer1.save()

                perso1.save()
                perso2.delete()
                doublon.delete()
            elif "validate-x" in request.POST:
                doublon = Doublon.objects.get(pk=request.POST["validate-x"])
                doublon.delete()
            
            return redirect("admin:duplicates")

    doublons = Doublon.objects.prefetch_related("item1", "item2").all()

    return render(request, "admin/duplicates.html",
                    {"doublons": doublons})


@login_required
def characters(request: HttpRequest | WSGIRequest) -> HttpResponse:
    """
    Page de sélection de personnage
    """
    if request.method == "POST":
        # Rediriger vers la page du personnage choisi 
        item_id: int = request.POST.get("id_item")
        num_questions: int = int(request.POST.get("num_questions", "5"))
        request.session["num_questions"] = num_questions

        return redirect(reverse("admin:character_details", args=[item_id]))

    else:
        # Afficher la page de sélection de personnage
        items: QuerySet[Item] = Item.objects.order_by("name")
        return render(request, "admin/characters.html", {"items": items})

@login_required
def character_details(request: HttpRequest | WSGIRequest, item_id: int) -> HttpResponse:
    """
    Page qui affiche les 5 questions les plus discriminantes pour le personnage courant (pour oui et pour non),
    et permet de modifier son nom/de le supprimer
    """

    num_questions = request.session.get("num_questions", 5)

    current_item: Item = get_object_or_404(Item, pk=item_id)

    # On met le personnage dont on veut les détails au début de la liste déroulante ("items": items dans le render).
    items: QuerySet[Item] = Item.objects.exclude(id=item_id).order_by("name")
    items = [current_item] + list(items)

    n_yes: list[str] = list(
            Answer.objects.filter(item_id=item_id)
            .order_by("-yes_tfidf")
            .exclude(yes_tfidf__exact=0)
            .values_list("question__title","yes_tfidf")[:num_questions])

    n_no: list[str] = list(
        Answer.objects.filter(item_id=item_id)
        .order_by("-no_tfidf")
        .exclude(no_tfidf__exact=0)
        .values_list("question__title","no_tfidf")[:num_questions])
    
    n_ambigous: list[str] = list(
        Answer.objects.filter(item_id=item_id)
        .annotate(ambigous_score=4 * F("yes_tf") * F("no_tf"))
        .order_by("-ambigous_score")
        .exclude(ambigous_score__exact=0)
        .values_list("question__title","ambigous_score", "yes_tf")[:num_questions])

    return render(request, "admin/character_details.html",
                      {"items": items, "current_item": current_item, "n_yes": n_yes, "n_no": n_no,
                       "n_ambigous": n_ambigous, "num_questions": num_questions})

@login_required
def character_delete(request: HttpRequest | WSGIRequest, item_id: int) -> HttpResponse:
    """
    Page de confirmation puis suppresion du personnage courant
    """
    current_item: Item = get_object_or_404(Item, pk=item_id)

    if request.method == "POST":
        # Appliquer le choix de l'admin
        choice: str = request.POST.get("choice")
        if choice == "y":
            current_item.delete()
            return redirect("admin:characters")
        else:
            return redirect(reverse("admin:character_details", args=[item_id]))

    else:
        # Demander confirmation 
        return render(request, "admin/character_delete.html", {"current_item": current_item})
    
@login_required
def character_rename(request: HttpRequest | WSGIRequest, item_id: int) -> HttpResponse:
    """
    Page de renommage du personnage courant
    """
    current_item: Item = get_object_or_404(Item, pk=item_id)

    if request.method == "POST":
        # Renommer le personnage
        form = ItemForm(request.POST, instance=current_item)
        if form.is_valid():
            form.save()
            return redirect(reverse("admin:character_details", args=[item_id]))
        else:
            errors = list(form.errors["name"])
    
    else:
        errors = []
    form = ItemForm(instance=current_item)
    return render(request, "admin/character_rename.html", {"current_item": current_item, "form": form, "errors": errors})

@login_required
def update_visualisation(request):
    """
    Mettre à jour la visualisation (les TF-IDFS, les clusters, la visualisation et les fichiers CSV)
    """
    if request.method != 'POST':
        return HttpResponseNotFound()
    
    update_tfidf = request.POST.get('update_tfidf')
    update_csv_clusters_proj = request.POST.get('update_csv_clusters_proj')
    n_clusters = request.POST.get('n_clusters', '5')
    n_caracteristic_answers = request.POST.get('n_caracteristic_answers', '5')
    clustering_algorithm = (request.POST.get('clustering_algorithm') or 'kmeans').lower()
    eps_raw = request.POST.get('eps', '0.2')
    min_samples_raw = request.POST.get('min_samples', '8')

    if clustering_algorithm not in {'kmeans', 'dbscan'}:
        clustering_algorithm = 'kmeans'

    try:
        eps = float(eps_raw)
        if eps <= 0:
            eps = 0.2
    except (TypeError, ValueError):
        eps = 0.2

    try:
        min_samples = int(min_samples_raw)
        if min_samples < 1:
            min_samples = 8
    except (TypeError, ValueError):
        min_samples = 8

    from admin.tasks import compute_visualisation_task;
    result = compute_visualisation_task.delay(
        update_tfidf = update_tfidf == 'on',
        update_csv_clusters_proj = update_csv_clusters_proj == 'on',
        n_clusters=int(n_clusters),
        n_caracteristic_answers=int(n_caracteristic_answers),
        clustering_algorithm=clustering_algorithm,
        eps=eps,
        min_samples=min_samples,
    )

    print("Launched task ...")

    # Handle the result if needed
    return _render_task_page(
        request,
        result.task_id,
        reverse('admin:autoencodeur'),
        task_title="Mise a jour de la visualisation",
        task_message="La regeneration des fichiers de visualisation est en cours.",
    )

@login_required
def train_auto_encoder(request):
    if request.method == 'POST':
        # Retrieve form data and construct the command
        iterations = int(request.POST.get('input1', '100'))
        hidden_size = int(request.POST.get('input3', '25'))
        requested_output_file = request.POST.get('input2', _default_autoencoder_output_file())

        """trigger_value = request.POST.get('input3', '0.9')
        min_games_played = request.POST.get('input4', '5')"""
        if(AutoEncoderTrainingRun.objects.count() > 0):
            id = AutoEncoderTrainingRun.objects.latest("id").id + 1
        else:
            id = 0  # On met un id temporaire à 0, il sera remplacé par l'id auto-incrémenté à la création de l'objet dans la base de données

        output_file = _normalize_autoencoder_output_file(requested_output_file, id)

        request.session["autoencoder_iterations"] = iterations
        request.session["autoencoder_output_file"] = _default_autoencoder_output_file()
        request.session["autoencoder_hidden_size"] = hidden_size

        # Execute the Python script
        from admin.tasks import train_auto_encoder_task

        run = AutoEncoderTrainingRun.objects.create(
            nb_epochs=iterations,
            hidden_size=hidden_size,
            output_file=output_file,
            id=id
        )

        result = train_auto_encoder_task.delay(
            nbepochs=iterations,
            output_file=output_file, 
            compression=hidden_size,
            run_id=run.id
        )


        return _render_task_page(
            request,
            result.task_id,
            reverse('admin:suggestions'),
            task_title="Entrainement de l'autoencodeur",
            task_message="Les metriques de chaque epoch s'affichent ici pendant l'execution.",
            task_ui="autoencoder",
        )


    # Handle other HTTP methods if needed
    return HttpResponseNotFound()


@login_required
def upload_weights(request: HttpRequest) -> HttpResponse:
    """Allow admin to upload autoencoder weights and store path in AppSettings.ae_weights_path"""
    if request.method == 'POST':
        # Expect file input named 'weights_file' or a path input 'weights_path'
        f = request.FILES.get('weights_file')
        path_input = request.POST.get('weights_path')
        settings_model, _ = AppSettings.objects.get_or_create(pk=1)

        # Save uploaded file to BASE_DIR if provided
        if f:
            # validate extension and size
            filename = os.path.basename(f.name)
            if not filename.lower().endswith('.h5'):
                # invalid file type
                request.session['admin_upload_error'] = 'Type de fichier non supporté (attendu .h5)'
                return redirect('admin:upload_weights')
            max_size = 100 * 1024 * 1024  # 100MB
            if f.size > max_size:
                request.session['admin_upload_error'] = 'Fichier trop gros (max 100MB)'
                return redirect('admin:upload_weights')

            # sanitize filename and store under BASE_DIR / 'models'
            dest_dir = BASE_DIR / "models"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename
            with open(dest, 'wb') as out:
                for chunk in f.chunks():
                    out.write(chunk)
            # store relative path from BASE_DIR
            settings_model.ae_weights_path = str(Path("models") / filename)
            settings_model.save()
            request.session['admin_upload_success'] = f'Fichier {filename} importé avec succès.'
            return redirect('admin:admin_index')

        # Or update path from text input
        if path_input:
            settings_model.ae_weights_path = path_input
            settings_model.save()
            return redirect('admin:admin_index')

    # For GET, show simple upload form
    # show any messages from upload attempt
    error = request.session.pop('admin_upload_error', None)
    success = request.session.pop('admin_upload_success', None)
    return render(request, 'admin/upload_weights.html', {'error': error, 'success': success})



@login_required
def reload_autoencoder(request: HttpRequest) -> HttpResponse:
    """Force reload of cached autoencoder session (useful after uploading new weights)."""
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    from admin.utils.autoencoder.auto_encoder_utils import get_cached_session, get_cached_session_error, resolve_weights_path
    # force reload
    session = get_cached_session(force_reload=True)
    if session is None:
        settings_model, _ = AppSettings.objects.get_or_create(pk=1)
        weights_path = settings_model.ae_weights_path or "auto_encoder_weights.h5"
        error_detail = get_cached_session_error()
        request.session['admin_reload_error'] = (
            "Échec du rechargement du modèle. "
            f"Fichier: {resolve_weights_path(weights_path)}. "
            f"Détail: {error_detail or 'erreur inconnue'}"
        )
    else:
        request.session['admin_reload_success'] = 'Modèle rechargé avec succès.'
    return redirect('admin:admin_index')

@login_required
def delete_autoencoder_training_run(request: HttpRequest, run_id: int) -> HttpResponse:
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    run = get_object_or_404(AutoEncoderTrainingRun, pk=run_id)
    weights_file = run.weights_file
    run.delete()
    weights_file_path = Path(weights_file) if weights_file else None
    if weights_file_path and weights_file_path.exists():
        weights_file_path.unlink()
    return redirect("admin:suggestions")


@login_required
def purge_autoencoder_training_history(request: HttpRequest) -> HttpResponse:
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    settings_model, _ = AppSettings.objects.get_or_create(pk=1)
    active_model_path = _normalize_model_path(settings_model.ae_weights_path)
    weights_files = [
        run.weights_file
        for run in AutoEncoderTrainingRun.objects.all()
        if run.weights_file and _normalize_model_path(run.weights_file) != active_model_path
    ]
    AutoEncoderTrainingRun.objects.all().delete()

    for weights_file in weights_files:
        weights_file_path = Path(weights_file)
        if weights_file_path.exists():
            weights_file_path.unlink()

    return redirect("admin:suggestions")


@login_required
def compute_suggestions(request):
    if request.method == 'POST':
        # Retrieve form data and construct the command
        input_file = request.POST.get('input2', 'auto_encoder_weights.h5')
        triggerValue = request.POST.get('input3', '0.9')
        min_Played= request.POST.get('input4', '5')
        max_Played= request.POST.get('input5', '20')

        # Vider le cache
        get_questions_by_suggestions_count.cache_clear()

        # Execute the Python script
        from admin.tasks import compute_suggestions_task;
        result = compute_suggestions_task.delay(
            input_file=input_file, 
            triggerValue=float(triggerValue),
            min_Played=int(min_Played), 
            max_Played=int(max_Played)
        )

        print("Launched task ...")

        # Handle the result if needed
        return _render_task_page(
            request,
            result.task_id,
            reverse('admin:suggestions'),
        )

    # Handle other HTTP methods if needed
    return HttpResponseNotFound()

@login_required
def find_duplicates(request):
    if request.method == 'POST':
        # Retrieve form data and construct the command
        trigger = request.POST.get('inputTrigger', '0.8')

        from admin.tasks import find_duplicates;
        result = find_duplicates.delay(trigger_value = float(trigger))

        return _render_task_page(
            request,
            result.task_id,
            reverse('admin:duplicates'),
        )
        
    # Handle other HTTP methods if needed
    return HttpResponseNotFound()

@login_required
def update_tfidf_scores(request):
    if request.method != 'POST':
        return HttpResponseNotFound()
    
    import app.management.commands.update_tfidf as update_tfidf_cmd
    cmd = update_tfidf_cmd.Command()
    cmd.handle()

    # Handle the result if needed
    return redirect("admin:admin_index")

@login_required
def export_data(request):
    # Handle the result if needed
    return render(request, "admin/export_data.html")
    
@login_required
def update_character_image(request, item_id):
    if request.method == 'POST':

        image_url = None

        item = get_object_or_404(Item, pk=item_id)
        use_wikipedia_image = request.POST.get('use_wikipedia_image')

        if use_wikipedia_image == 'on':
            wikiName = get_wikipedia_name(item.name)

            if wikiName is not None:
            
                json_result = query_wikipedia_api("https://fr.wikipedia.org/w/api.php", wikiName)
                image_url = extract_thumbnail_url(json_result)

        else:
            image_url = request.POST.get('image_url')

        if image_url is not None:
            item.image = image_url
            item.save()
        else:
            print("Aucune image trouvée")

        return redirect(reverse("admin:character_details", args=[item_id]))

    # Handle other HTTP methods if needed
    return HttpResponseNotFound()

@login_required
def download_backup(request):
    if request.method == "POST":
        try:
            # Exécuter la commande de sauvegarde de la base de données
            process = Popen(["mariadb-dump", "-u", "devinsa", "-pdevinsa!", "devinsa"], stdout=PIPE)
            output, _ = process.communicate()

            # Créer un fichier temporaire pour stocker la sortie de la commande
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(output)

            # Renvoyer le fichier de sauvegarde comme réponse HTTP à télécharger
            response = FileResponse(open(tmp_file.name, 'rb'))
            response['Content-Disposition'] = 'attachment; filename="backup.sql"'
            return response
        except Exception as e:
            return HttpResponse(f"Une erreur s'est produite : {str(e)}", status=500)
    else:
        return HttpResponseNotAllowed(['POST'])



@login_required
def suggestions_alternative(request):
    if (request.method == "GET"):
        return render(request, "admin/select_question.html")
    # On récupère pour chaque personnage sa question avec le plus gros tfidf de oui, 
    # et on les regroupe dans la liste l avec les autres personnages ayant la meme question la plus discriminante

    num_questions = int(request.POST.get("num_questions"))

    num_suggestions = int(request.POST.get("num_suggestions"))
    alpha = float(request.POST.get("alpha"))

    l={}
    for i in Item.objects.all():
        question: str = best_questions(i.id)[0]
        if (question in l):
            l[question].append(i)
        else:
            l[question]=[i]

    # On recupere que au moment ou ils sont plusieurs a avoir la meme question la plus discriminante
    questions = {k[0]: v for k, v in l.items() if len(v) > 1}
    # On prend que le premier groupe
    X=list(questions.items())
    random.shuffle(X)
    questions = dict(X[:num_questions])
    
    # Maintenant pour chaque groupe on va prendre les nb questions les plus pertinentes de chaque personnage
    suggestions={}
    for question,personnages in questions.items():
        for personnage in personnages:
            questions_personnage : list[str]=best_questions(personnage.id,nb=num_suggestions)
            # evidemment on retombe sur la question la plus pertinente donc on l'enleve
            questions_personnage.pop(0)
            for question_personnage,_ in questions_personnage:
                if not(question in suggestions):
                    suggestions[question]=[question_personnage]
                elif not(question_personnage in suggestions[question]):
                    suggestions[question].append(question_personnage)

    
    # suggestions = goodSuggestions(suggestions,alpha=0.5)
    suggestions_par_personnage : dict[str,list[str]]={}
    for question,suggestion_list in suggestions.items():
        # On recupere tous les personnages associés a la question en cours pour voir les recommandations
        for personnage in questions[question]:
            # On regarde si la question a deja été repondue pour ce personnage
            # Je sais qu'on a pas besoin des tfidf mais je sais pas recuperer juste le titre
            # de la question. Désolé.
            reponses = Answer.objects.filter(item_id=personnage.id).values_list("question__title","yes_tfidf")
            reponses = [k for k,_ in reponses]
            reponses.remove(question)
            # print("Pour le perso :",personnage.name)
            for suggestion in suggestion_list:
                # Ici on ajoute les questions non répondues par personnage dans la liste qui correspond
                if not(suggestion in reponses) and goodSuggestions(suggestion,questions[question],alpha=alpha):
                    if (personnage in suggestions_par_personnage):
                        suggestions_par_personnage[personnage].append(suggestion)
                    else:
                        suggestions_par_personnage[personnage]=[suggestion]
    # print(suggestions_par_personnage)
    return render(request, "admin/suggestions_alternative.html", {"questions":questions,"suggestions_par_personnage":suggestions_par_personnage})

def goodSuggestions(question,liste_persos,alpha=0.5):
    yes_count=0
    no_count=0
    try: 
        question_id = Question.objects.get(title=question)
    except:
        print("Big erreur")
    for personnage in liste_persos:
        reponses = Answer.objects.filter(item_id=personnage.id,question=question_id.id).exclude(yes_tfidf__exact=0)
        if (len(reponses)!=0):
            yes_count+=1
        else:
            pass
    return yes_count/len(liste_persos) > alpha

def nameOf(item_id):
    item=get_object_or_404(pk=item_id, klass=Item)
    return item.name

def best_questions(item_id,nb=1):
    '''
    Renvoie une liste comme ceci ('Question posee', tf_idf)
    '''
    n_yes: list[str] = list(
            Answer.objects.filter(item_id=item_id)
            .order_by("-yes_tfidf")
            .exclude(yes_tfidf__exact=0)
            .values_list("question__title","yes_tfidf"))
    if (n_yes != []):
        return n_yes[:nb]
    return ["Aucune question posée a ce joueur dont la réponse est oui",0]

@login_required
def update_database_alternative(request: HttpRequest | WSGIRequest):
    if (request.method == "POST"):
        data = json.loads(request.body)
        
        character = data.get('character')[:-2]
        suggestions = data.get('suggestions')
        type = data.get('validate')

        character_id = Item.objects.get(name=character).id
        for question in suggestions:
            question_id = Question.objects.get(title=question).id
            answer, created = Answer.objects.get_or_create(
                item_id=character_id,
                question_id=question_id,
                defaults={"yes_count": 0, "no_count": 0}
            )

            # Update the counts based on the validation type
            if type == "y":
                answer.yes_count += 1
            elif type == "n":
                answer.no_count += 1

            answer.save()
        print("Updating DB")
    return HttpResponse('')

@login_required
def distance_suggestions(request : HttpRequest | WSGIRequest, item_id : int) -> HttpResponse:
    """This function is used to display the closest characters to a given character based on their answers to questions.

    Args:
        request (HttpRequest | WSGIRequest): Contains the session variables num_questions and num_closest_characters
        item_id (int): The ID of the target character.

    Returns:
        HttpResponse: Renders the distance suggestions page.
    """
    # Retrieve the number of questions and closest characters to display    
    num_questions = request.session.get("num_questions", 5)
    num_closest_characters = request.session.get("num_closest_characters", 10)
    target_character =  item_id
    
    # Every characters and every answer ever given in database
    answers = Answer.objects.all()
    questions = list(set(answer.question_id for answer in answers))
    characters = list(set(answer.item_id for answer in answers))
    # Compute the cosine similarity between characters based on their answers to questions and retrieve the correct indexes to acces the matrix
    similarity_matrix, character_index, question_index, combined_matrix = compute_distances(answers, characters, questions)

    # Retrieve the index of the target character    
    character_idx = character_index[target_character]
    character : Item = get_object_or_404(Item, pk=target_character)
    # Retrieve the list of characters close to the target character
    closest_indices = find_similarities(similarity_matrix, character_idx, answers, num_closest_characters)
    #print("Indices des personnages proches :",closest_indices)
    proposed_answers_display = find_best_suggestions(combined_matrix, character_idx, closest_indices, questions, question_index, characters, num_questions)
    #print("Propositions de questions :",proposed_answers_display)
    items: QuerySet[Item] = Item.objects.exclude(id=item_id).order_by("name")
    items = [character] + list(items)

    return render(request, "admin/distance_suggestions.html", 
                  {"proposed_answers": proposed_answers_display, 
                   "items": items,
                   "num_questions": num_questions, 
                   "num_closest_characters": num_closest_characters
                   })

@login_required
def selection_suggestions(request : HttpRequest | WSGIRequest) -> HttpResponse:
    # Si POST, on redirige vers la la page qui affiche les suggestions
    if request.method == "POST":
        item_id: int = request.POST.get("id_item")
        num_questions: int = int(request.POST.get("num_questions", "5"))
        request.session["num_questions"] = num_questions
        num_closest_characters: int = int(request.POST.get("num_closest_characters", "10"))
        request.session["num_closest_characters"] = num_closest_characters
        return redirect(reverse("admin:distance_suggestions", args=[item_id]))
    else:
        items: QuerySet[Item] = Item.objects.order_by("name")
        return render(request, "admin/selection_suggestions.html", {"items": items})
    
@login_required
def validate_suggestions_alternative(request: HttpRequest | WSGIRequest) -> HttpResponse:
    """Function used to validate suggestions made by alternative methods : similarity between characters or common discriminative questions.
    This function is used to validate the suggestions made by the admin. It updates the answers in the database and clears the session variable.

    Args:
        request (HttpRequest | WSGIRequest): Contains the session variables

    Returns:
        HttpResponse: Redirects to the suggestions page.
    """
    if request.method == "POST":
        try:

            # Parse the JSON data
            data = json.loads(request.body)
            print(data)
            question_data = data.get("question")
            characters_data = data.get("characters", [])
            validate = data.get("validate")
            print(characters_data)
            # Extract question ID and text
            question_id, question_text = eval(question_data)

            # Process each character
            for character_data in characters_data:
                character_id, character_name = eval(character_data)

                # Retrieve or create the Answer object
                answer, created = Answer.objects.get_or_create(
                    item_id=character_id,
                    question_id=question_id,
                    defaults={"yes_count": 0, "no_count": 0}
                )

                # Update the counts based on the validation type
                if validate == "y":
                    answer.yes_count += 1
                elif validate == "n":
                    answer.no_count += 1

                answer.save()

            return JsonResponse({"success": True, "message": "Suggestions validated successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "message": f"Error: {str(e)}"}, status=400)

    return JsonResponse({"success": False, "message": "Invalid request method."}, status=405)


@login_required
def description_based_suggestions(request : HttpRequest | WSGIRequest) -> HttpResponse:
    """
    View to generate and display suggestions based on character descriptions.
    """

    suggestions_save_file = BASE_DIR / "admin/tmp/suggestions.json"
    cache_key = "description_suggestions"  # Key for caching suggestions

    if(request.method == "POST"):
        characters_threshold : int = int(request.POST.get('characters_threshold', 10))
        suggestions = generate_suggestions_based_on_descriptions(characters_threshold)
        
        cache.set(cache_key, suggestions, timeout=900)  # Cache the suggestions for 15 minutes
        # Sauvegarder les suggestions dans un fichier temporaire
        with open(suggestions_save_file, 'w') as f:
            # Convertir les clés des sous-dictionnaires en chaînes de caractères
            suggestions_serializable = {
                suggestion_type: {
                    str(key): value for key, value in suggestions[suggestion_type].items()
                }
                for suggestion_type in suggestions
            }
            json.dump(suggestions_serializable, f)

        return render(request, "admin/description_suggestions.html",
                       {"proposed_answers": suggestions,
                        "characters_threshold": characters_threshold,
                        "method": "description_based"
                        })
    elif(request.method == "GET"):
        # Handle GET request to display the suggestions page 
        characters_threshold = 10

        suggestions = cache.get(cache_key)  # Retrieve cached suggestions
        if suggestions is None:
            # If not cached, try to load from the file
            if os.path.exists(suggestions_save_file):
                with open(suggestions_save_file, 'r') as f:
                    suggestions_serializable = json.load(f)
                    # Reconversion des clés en tuples
                    suggestions = {
                        suggestion_type: {
                            eval(key): value for key, value in suggestions_serializable[suggestion_type].items()
                        }
                        for suggestion_type in suggestions_serializable
                    }
                cache.set(cache_key, suggestions, timeout=900)
            # If not saved in file, generate empty suggestions for now
            else: 
                suggestions = {}

        return render(request, "admin/description_suggestions.html", 
                      {
                          "proposed_answers": suggestions, 
                          "characters_threshold": characters_threshold,
                          "method": "description_based"
                      })

@login_required
def remove_suggestion(request: HttpRequest | WSGIRequest) -> JsonResponse:
    """
    Supprimer une suggestion spécifique du cache et du fichier JSON.
    """
    suggestions_save_file = BASE_DIR / "admin/tmp/suggestions.json"
    cache_key = "description_suggestions"

    if request.method == "POST":
        try:
            # Récupérer les données de la requête
            data = json.loads(request.body)
            question_id = data.get("question_id")

            # Charger les suggestions depuis le cache
            suggestions = cache.get(cache_key)
            if suggestions is None:
                # Si le cache est vide, charger depuis le fichier
                if os.path.exists(suggestions_save_file):
                    with open(suggestions_save_file, 'r') as f:
                        suggestions_serializable = json.load(f)
                        suggestions = {
                            suggestion_type: {
                                eval(key): value for key, value in suggestions_serializable[suggestion_type].items()
                            }
                            for suggestion_type in suggestions_serializable
                        }
                else:
                    return JsonResponse({"success": False, "message": "Aucune suggestion trouvée avec cet id."}, status=404)
            # Supprimer la suggestion
            for suggestion_type in suggestions.keys():
                suggestions[suggestion_type] = {
                    key: value for key, value in suggestions[suggestion_type].items() if int(key[0]) != int(question_id)
                }

            # Mettre à jour le cache
            cache.set(cache_key, suggestions, timeout=900)

            # Sauvegarder les suggestions mises à jour dans le fichier
            with open(suggestions_save_file, 'w') as f:
                suggestions_serializable = {
                    suggestion_type: {
                        str(key): value for key, value in suggestions[suggestion_type].items()
                    }
                    for suggestion_type in suggestions
                }
                json.dump(suggestions_serializable, f)

            return JsonResponse({"success": True, "message": "Suggestion supprimée avec succès."})

        except Exception as e:
            return JsonResponse({"success": False, "message": f"Erreur : {str(e)}"}, status=400)

    return JsonResponse({"success": False, "message": "Méthode non autorisée."}, status=405)
   
