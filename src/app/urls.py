from django.urls import path

from app.views import index, play, result, report, about, guess, add_character, add_question, thinking

app_name = "app"

urlpatterns = [
    path("", index, name="index"),
    path("play", play, name="play"),
    path("guess", guess, name="guess"),
    path("result", result, name="result"),
    path("report", report, name="report"),
    path("about", about, name="about"),
    path("character", add_character, name="character"),
    path("question", add_question, name="question"),
    path('thinking/', thinking, name='thinking')
]
