from django.core.handlers.wsgi import WSGIRequest
from django.http import HttpRequest, HttpResponse, HttpResponseNotFound
from django.shortcuts import render, redirect

from app.forms import ItemForm, QuestionForm
from app.models import Answer, Item, Question, Tree, PlayerAnswer, Report
from app.utilities.utilities import search_best_item, search_best_items, select_liste, report_game, save_game, get_closest_questions, \
    get_closest_items, search_best_item_autoencoder, search_best_items_autoencoder


def index(request: HttpRequest) -> HttpResponse:
    """
    Affichage de la page "Index"
    :param request: requête http
    :return: vue "Index"
    """
    request.session.flush()
    request.session.clear_expired()
    return render(request, "app/index.html")


def about(request: HttpRequest) -> HttpResponse:
    """
    Affichage de la page "À propos"
    :param request: requête http
    :return: vue "About"
    """
    request.session.flush()
    return render(request, "app/about.html")

def play(request: HttpRequest | WSGIRequest) -> HttpResponse:
    if request.method == "GET":
        return handle_play_get(request)
    elif request.method == "POST":
        return handle_play_post(request)
    else:
        return HttpResponseNotFound()

def handle_play_get(request: HttpRequest | WSGIRequest) -> HttpResponse:
    
    tree: Tree
    question: Question
    depth: int

    # Début de partie : initialisation des données
    if not request.session.get("tree"):
        # Accès à la racine de l'arbre, utilisation de la clé primaire "1"
        # permet un accès rapide, la recherche par 'choice' est plus lente
        tree = Tree.objects.select_related().get(pk=1)

        request.session["tree"] = tree.id
        request.session["answers"] = []
        request.session["question"] = tree.question_id
        request.session["depth"] = tree.depth
        # Le numéro de la question choisie quand nous serons sur les questions au hasard
        request.session["num_question"] = -1
        request.session["liste_quest_aleatoires"] = []

    question_id: int = request.session.get("question")
    question = Question.objects.get(id=question_id)
    depth = request.session.get("depth")

    return render(request, "app/play.html", {"question": question.title, "depth": depth})

def handle_play_post(request: HttpRequest | WSGIRequest) -> HttpResponse:
    # Partie en cours
    tree_id: int = request.session.get("tree")
    tree = Tree.objects.select_related().get(id=tree_id)
    answers = request.session.get("answers")
    question_id: int = request.session.get("question")
    question = Question.objects.get(id=question_id)
    depth = request.session.get("depth")
    num_question: int = request.session.get("num_question")
    liste_quest_aleatoires = request.session.get("liste_quest_aleatoires")

    if "undo" in request.POST and depth > 1:
        answers.pop()

        if num_question > 0: 
            print("Undo RNG : ")
            print(num_question)
            num_question -= 1
            question = Question.objects.get(id=liste_quest_aleatoires[num_question])
        
        if num_question == 0:
            num_question = -1
            question = tree.question

        # Retourner au parent
        if num_question < 0:
            tree = tree.parent
            question = tree.question

    else:
        if "submitYes" in request.POST:
            print("Yes c bonnnnn")
        choice: str = request.POST.get("choice")
        if choice not in ["y", "n"]:
            choice = "p"
        answers.append(PlayerAnswer(question.id, choice))
        if depth < 20:
            # Récupère la question suivante
            nextTree = tree.get_child(choice)

            if num_question < 0 and nextTree is not None:
                tree = nextTree
                question = tree.question
            else:
                # Questions aléatoires
                if not liste_quest_aleatoires:
                    liste_quest_aleatoires = select_liste(answers)
                    print(liste_quest_aleatoires)
                
                num_question += 1
                question = Question.objects.get(id=liste_quest_aleatoires[num_question])

        else:
            # Fin du jeu
            return redirect("app:thinking")

    print("Save")
    request.session["tree"] = tree.id
    request.session["answers"] = answers
    request.session["question"] = question.id
    request.session["depth"] = tree.depth + num_question + 1
    request.session["num_question"] = num_question
    request.session["liste_quest_aleatoires"] = liste_quest_aleatoires

    return redirect("app:play")

def thinking(request):
    return render(request, "app/thinking.html")

def guess(request: HttpRequest | WSGIRequest) -> HttpResponse:
    """
    Page de réponse du personnage lorsque le joueur termine sa partie.
    Si le bon personnage est trouvé, affiche le résultat de la partie,
    sinon, propose d'ajouter un personnage
    :param request: requête http
    :return: vue "Guess"
    """
    answers: list[PlayerAnswer] = request.session.get("answers", [])

    if request.method == "GET" and len(answers) > 0:
        treeId = request.session.pop("tree", None)
        treeDepth = Tree.objects.get(pk=treeId).depth

        # Choose algorithm according to AppSettings
        from app.models import AppSettings
        settings_model, _ = AppSettings.objects.get_or_create(pk=1)
        if settings_model.algorithm_mode == "autoencoder":
            result_item: int = search_best_item_autoencoder(answers, depth=treeDepth)
        else:
            result_item: int = search_best_item(answers, depth=treeDepth)

        # Vérification si l'utilisateur n'a fait que passer
        if result_item > -1 and request.session.get("depth", None):
            item = Item.objects.get(id=result_item)
            request.session["item"] = item.id
            return render(request, "app/guess.html", {"item": item})

        return redirect("app:character")

    if request.method == "POST":
        choice = request.POST.get("choice")
        treeDepth = request.session.get("depth")
        if choice == "y":
            item_id: int = request.session.get("item")
            item: Item = Item.objects.get(id=item_id)
            save_game(item, answers, True)
            request.session["sentence"] = "Je suis trop fort !\nEssaye de me battre si tu l'oses !"
            return redirect("app:result")
        elif choice and choice.isdigit():
            item = Item.objects.filter(id=int(choice)).first()
            if item:
                save_game(item, answers, True)
                request.session["item"] = item.id
                request.session["sentence"] = "Je suis trop fort !\nEssaye de me battre si tu l'oses !"
                return redirect("app:result")
            return redirect("app:character")
        elif choice == "n":
            # Get suggestions if the user wants to continue
            from app.models import AppSettings
            settings_model, _ = AppSettings.objects.get_or_create(pk=1)
            if settings_model.algorithm_mode == "autoencoder":
                suggestions = search_best_items_autoencoder(answers, depth=treeDepth, num_suggestions=3)
            else:
                suggestions = search_best_items(answers, depth=treeDepth, num_suggestions=3)
            
            if suggestions:
                # Display the suggestions
                items = Item.objects.filter(id__in=suggestions)
                return render(request, "app/continue_guess.html", {"items": items[1:]})
            else:
                return redirect("app:character")
    return redirect("app:index")


def result(request: HttpRequest) -> HttpResponse:
    """
    Affiche la page de fin de partie, avec la consultation du rapport et l'ajout d'une question
    :param request: requête http
    :return: vue "Result"
    """
    item_id: int = request.session.get("item")
    item: Item = Item.objects.filter(id=item_id).first()
    answers: list[PlayerAnswer] = request.session.get("answers")
    sentence: str = request.session.get("sentence")

    # Suppression du contenu de la session pour éviter un retour sur les autres vues
    request.session.pop("closest_questions", None)
    request.session.pop("closest_items", None)
    request.session.pop("depth", None)
    request.session.pop("question", None)

    if item and answers:
        request.session["report"] = report_game(item, answers)
        return render(request, "app/result.html", {"sentence": sentence})
    return redirect("app:index")


def report(request: HttpRequest) -> HttpResponse:
    """
    Renvoie la page affichant le rapport de partie
    :param request: requête http
    :return: vue "Report"
    """
    report_list: list[Report] = request.session.get("report")
    item_id: int = request.session.get("item")
    item: Item = Item.objects.filter(id=item_id).first()

    if report_list and item:
        return render(request, "app/report.html", {"report": report_list, "item": item})
    return redirect("app:index")


def add_question(request: HttpRequest | WSGIRequest) -> HttpResponse:
    """
    Ajout d'une question pour le personnage deviné
    Cherche son similaire au cas où et ajoute la réponse oui perso ↔ question
    puis revient à result
    :param request: requête http
    :return: vue "Question"
    """
    if request.method == "GET":
        if request.session.get("closest_questions"):
            closest_questions: list[Question] = [
                Question.objects.get(id=q_id) for q_id in request.session["closest_questions"]]
            title: str = request.session.get("title")
            return render(request, "app/question.html", {"questions": closest_questions, "title": title})

        # S'il y a demande d'ajout de question, on affiche le formulaire
        item_id: int = request.session.get("item")
        item: Item = Item.objects.filter(id=item_id).first()
        if item:
            return render(request, "app/question.html", {"questionform": QuestionForm(), "name": item.name})
        return redirect("app:index")

    if "question_sent" in request.POST:
        questionform = QuestionForm(request.POST)
        title = questionform.data["title"]
        request.session["title"] = title

        # Si le formulaire est vide, on renvoie la même page
        if title.strip() == "":
            return redirect("app:question")

        # On génère la liste de questions similaires
        closest_questions: list[int] = [qst.id for qst in get_closest_questions(title)]
        if len(closest_questions) > 0:
            request.session["closest_questions"] = closest_questions
            return redirect("app:question")

        # Sinon, on l'insère
        item_id: int = request.session.get("item")
        question_id: int = Question.objects.get_or_create(title=title)[0].id
        answer: Answer = Answer.objects.get_or_create(item_id=item_id, question_id=question_id)[0]
        answer.yes_count += 1
        answer.save()
        return redirect("app:result")

    elif "existing_question" in request.POST and "closest_questions" in request.POST:
        item_id: int = request.session.get("item")
        question_id: int = request.POST.get("closest_questions")
        answer: Answer = Answer.objects.get_or_create(item_id=item_id, question_id=question_id)[0]
        answer.yes_count += 1
        answer.save()
        return redirect("app:result")

    # Ajout final d'une nouvelle question pour le personnage joué.
    elif "new_question" in request.POST:
        title: str = request.session.get("title")
        item_id: int = request.session.get("item")
        question_id: int = Question.objects.get_or_create(title=title)[0].id
        answer: Answer = Answer.objects.get_or_create(item_id=item_id, question_id=question_id)[0]
        answer.yes_count += 1
        answer.save()
        return redirect("app:result")
    return redirect("app:question")


def add_character(request: HttpRequest | WSGIRequest) -> HttpResponse:
    """
    Ajout d'un personnage si l'on n'en a pas trouvé
    Cherche son similaire au cas où et corrige les réponses
    puis revient à result (tu m'as battu)
    :param request: requête http
    :return: vue "Character"
    """
    if request.method == "GET":
        if request.session.get("closest_items"):
            closest_items: list[Item] = [
                Item.objects.get(id=item_id) for item_id in request.session["closest_items"]]
            name: str = request.session.get("name")
            return render(request, "app/character.html", {"items": closest_items, "name": name})

        request.session.pop("depth", None)

        # S'il y a demande d'ajout de personnage, on affiche le formulaire
        if request.session.get("answers") and request.session.get("question", None):
            return render(request, "app/character.html", {"itemform": ItemForm()})
        return redirect("app:index")

    answers: list[PlayerAnswer] = request.session.get("answers")
    request.session["sentence"] = "Bravo, tu m'as battu !"
    if "character_sent" in request.POST:
        itemform = ItemForm(request.POST)
        name = itemform.data["name"]
        request.session["name"] = name

        # Si le formulaire est vide, on renvoie la même page
        if name.strip() == "":
            return redirect("app:character")

        # On génère la liste d'éléments similaires
        closest_items: list[int] = [item.id for item in get_closest_items(name)]
        if len(closest_items) > 0:
            request.session["closest_items"] = closest_items
            return redirect("app:character")

        # Sinon, on l'insère
        item = Item.objects.get_or_create(name=name)[0]
        save_game(item, answers)
        request.session["item"] = item.id
        return redirect("app:result")

    elif "existing_name" in request.POST and "closest_items" in request.POST:
        item_id: int = request.POST.get("closest_items")
        item = Item.objects.get(id=item_id)
        save_game(item, answers)
        request.session["item"] = item.id
        return redirect("app:result")

    # Ajout final d'un nouveau personnage
    elif "new_name" in request.POST:
        name = request.session.get("name")
        item: Item = Item.objects.get_or_create(name=name)[0]
        save_game(item, answers)
        request.session["item"] = item.id
        return redirect("app:result")
    return redirect("app:character")
