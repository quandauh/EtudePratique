from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from admin.views import statistics, remove_suggestion, update_database_alternative, description_based_suggestions, update_visualisation, export_data, update_tfidf_scores, duplicates, index, tree, suggestions, validate_suggestions, characters, character_details, character_delete, character_rename, train_auto_encoder, compute_suggestions, purge_autoencoder_training_history, find_duplicates, update_character_image, download_backup, suggestions_alternative, distance_suggestions, selection_suggestions, validate_suggestions_alternative,description_based_suggestions, autoencodeur, maintenance, maintenance_run, maintenance_status, moulinette, moulinette_status, delete_moulinette_run, rerun_moulinette_run, delete_autoencoder_training_run, toggle_algorithm_mode, upload_weights, reload_autoencoder

app_name = "admin"

urlpatterns = [
    path("", index, name="admin_index"),
    path("login/", LoginView.as_view(template_name="admin/login.html"), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("tree/", tree, name="tree"),
    path("autoencodeur/", autoencodeur, name="autoencodeur"),
    path("characters/", characters, name="characters"),
    path("characters/<int:item_id>/", character_details, name="character_details"),
    path("characters/<int:item_id>/delete/", character_delete, name="character_delete"),
    path("characters/<int:item_id>/rename/", character_rename, name="character_rename"),
    path("suggestions/", suggestions, name="suggestions"),
    path("validate_suggestions/", validate_suggestions, name="validate_suggestions"),
    path("duplicates/", duplicates, name="duplicates"),
    path("train_auto_encoder/", train_auto_encoder, name="train_auto_encoder"),
    path("train_auto_encoder/history/<int:run_id>/delete/", delete_autoencoder_training_run, name="delete_autoencoder_training_run"),
    path("compute_suggestions/", compute_suggestions, name="compute_suggestions"),
    path("train_auto_encoder/history/purge/", purge_autoencoder_training_history, name="purge_autoencoder_training_history"),
    path("find_duplicates/", find_duplicates, name="find_duplicates"),
    path("update_tfidf/", update_tfidf_scores, name="update_tfidf"),
    path("export_data/", export_data, name="export_data"),
    path("update_visualisation/", update_visualisation, name="update_visualisation"),
    path("update_character_image/<int:item_id>/", update_character_image, name="update_character_image"),
    path('download_backup/', download_backup, name='download_backup'),
    path('suggestions_alternative/', suggestions_alternative, name='suggestions_alternative'),
    path('distance_suggestions/', selection_suggestions, name='selection_suggestions'),
    path('distance_suggestions/<int:item_id>/', distance_suggestions, name='distance_suggestions'),
    path('update_database_alternative', update_database_alternative, name='update_database_alternative'),
    path('validate_suggestions_alternative/', validate_suggestions_alternative, name='validate_suggestions_alternative'),
    path('description_suggestions/', description_based_suggestions, name='description_based_suggestions'),
    path('remove-suggestion/', remove_suggestion, name='remove_suggestion'),
    path("statistics/", statistics, name="statistics"),
    path("maintenance/", maintenance, name="maintenance"),
    path("maintenance/run/<task_id>/", maintenance_run, name="maintenance_run"),
    path("maintenance/status/<task_id>/", maintenance_status, name="maintenance_status"),
    path("moulinette/", moulinette, name="moulinette"),
    path("moulinette/status/<task_id>/", moulinette_status, name="moulinette_status"),
    path("moulinette/history/<int:run_id>/rerun/", rerun_moulinette_run, name="rerun_moulinette_run"),
    path("moulinette/history/<int:run_id>/delete/", delete_moulinette_run, name="delete_moulinette_run"),
    path('toggle_algorithm_mode/', toggle_algorithm_mode, name='toggle_algorithm_mode'),
    path('upload_weights/', upload_weights, name='upload_weights'),
    path('reload_autoencoder/', reload_autoencoder, name='reload_autoencoder'),
]
