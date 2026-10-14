from pathlib import Path
from devinsa.settings import BASE_DIR

_STATIC_DATA = BASE_DIR / "static" / "data"

CSV_DELIMITER=";"
CHARACTER_CSV_NAME=str(_STATIC_DATA / "list_character.csv")
CHARACTER_COMPRESSED_CSV_NAME=str(_STATIC_DATA / "list_character_compressed.csv")
ANSWER_CSV_NAME=str(_STATIC_DATA / "list_answer.csv")
QUESTION_CSV_NAME=str(_STATIC_DATA / "list_question.csv")
CLUSTERS_DATA_CSV=str(_STATIC_DATA / "clusters_data.csv")
CLUSTERS_CSV=str(_STATIC_DATA / "clusters.csv")
VISUALISATION_CSV=str(_STATIC_DATA / "visualisation.csv")
CLUSTERS_COMPRESSED_CSV=str(_STATIC_DATA / "clusters_compressed.csv")
VISUALISATION_COMPRESSED_CSV=str(_STATIC_DATA / "visualisation_compressed.csv")
QUESTION_CSV_NAME_ALL_ATTRIBUTES = str(_STATIC_DATA / "list_question_all_attributes.csv")
ANSWER_CSV_NAME_ALL_ATTRIBUTES = str(_STATIC_DATA / "list_answer_all_attributes.csv")
CHARACTER_CSV_NAME_ALL_ATTRIBUTES = str(_STATIC_DATA / "list_character_all_attributes.csv")
MEDOIDS_NB = 4 # le nombre de médoïdes par clusters
