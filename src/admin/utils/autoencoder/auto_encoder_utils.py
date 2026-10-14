from admin.utils.autoencoder import datas, settings, auto_encoder
from app.models import AppSettings
import numpy as np
from pathlib import Path
from devinsa.settings import BASE_DIR
from threading import Lock

PROJECT_ROOT = BASE_DIR.parent
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "auto_encoder_weights.h5"

_TRAINING_SIMULATION_SEEDS = (0, 1, 2)

# Simple module-level cache so we don't recreate the heavy model on every request.
_CACHED_SESSION = None
_CACHED_SESSION_KEY = None
_CACHED_SESSION_ERROR = ""
_CACHE_LOCK = Lock()


def resolve_weights_path(file: str | Path | None = None) -> str:
    if file is None or str(file).strip() == "":
        return str(DEFAULT_WEIGHTS_PATH)

    file_path = Path(file)
    if not file_path.is_absolute():
        project_path = PROJECT_ROOT / file_path
        base_dir_path = BASE_DIR / file_path
        file_path = base_dir_path if base_dir_path.exists() else project_path

    return str(file_path)


class AutoEncoderSession:
    def __init__(self) -> None:
        self.questions_count = datas.get_questions_count()
        self.characters_count = datas.get_characters_count()

        # 2 * nb_questions: one yes slot and one no slot per question.
        self.inputSize = 2 * self.questions_count

        #  Config de l'auto encoder
        self.autoEncoder = auto_encoder.Autoencoder(
            self.inputSize, AppSettings.objects.get(pk=1).ae_hidden_size
        )

        # Association table for human comprehension
        self.questions_to_index = datas.get_questions_to_index_table()
        self.characters_to_index = datas.get_characters_to_index_table()

        # Create human readable matrix
        self.hr_matrix = datas.create_character_answers_matrix(
            self.characters_count,
            self.questions_count,
            self.characters_to_index,
            self.questions_to_index,
        )

        self.hr_mask = datas.create_answer_mask_matrix(
            self.characters_count,
            self.questions_count,
            self.characters_to_index,
            self.questions_to_index,
        )
        self.hr_expected = datas.create_expected_output_matrix(
            self.characters_count,
            self.questions_count,
            self.characters_to_index,
            self.questions_to_index,
        )

        # Create a matrix usable by the neural network
        self.flat_matrix = datas.flatten_matrix(
            self.hr_matrix, self.characters_count, self.inputSize
        )
        self.flat_mask = datas.flatten_matrix(
            self.hr_mask, self.characters_count, self.inputSize
        )
        self.flat_expected = datas.flatten_matrix(
            self.hr_expected, self.characters_count, self.inputSize
        )
        # The autoencoder search space is binary and interleaved by question.
        # Gameplay only knows yes/no/pass choices, not per-character TF-IDF
        # values, so hidden vectors must be computed from flat_expected rather
        # than the historical TF-IDF flat_matrix.
        self.character_encoding_input_matrix = self.flat_expected

        self.encoded_matrix = None
        self.decoded_matrix = None
        self.decoded_hr_matrix = None

    def load(self, file):
        self.autoEncoder.load_weights(resolve_weights_path(file))

    def _build_training_matrices(self):
        """
        Build denoising examples that match how the game and benchmark call the
        model: sparse 20-answer player vectors in, full character answers out.

        The complete binary character vector is still included for each item so
        encoded character rows and encoded player rows live in the same hidden
        space during nearest-neighbor search.
        """
        X_parts = [self.character_encoding_input_matrix]
        M_parts = [self.flat_mask]
        Y_parts = [self.flat_expected]

        from benchmark import utils as benchmark_utils
        from app.utilities.utilities import build_autoencoder_player_vector

        item_ids = sorted(self.characters_to_index.keys())
        answer_stats_lookup = benchmark_utils._build_answer_stats_lookup()
        tree_nodes, tree_children = benchmark_utils._build_tree_index()
        eligible_question_ids = list(
            benchmark_utils.get_liste_questions_a_poser().values_list("id", flat=True)
        )

        for seed in _TRAINING_SIMULATION_SEEDS:
            dataset = benchmark_utils.generate_benchmark_dataset(
                num_questions=20,
                seed=seed,
                item_ids=item_ids,
                answer_stats_lookup=answer_stats_lookup,
                tree_nodes=tree_nodes,
                tree_children=tree_children,
                eligible_question_ids=eligible_question_ids,
            )
            X_rows = []
            M_rows = []
            Y_rows = []
            for case in dataset:
                char_index = self.characters_to_index.get(case.item_id)
                if char_index is None:
                    continue
                X_rows.append(
                    build_autoencoder_player_vector(
                        case.answers,
                        self.questions_to_index,
                        self.inputSize,
                    )[0]
                )
                M_rows.append(self.flat_mask[char_index])
                Y_rows.append(self.flat_expected[char_index])
            if X_rows:
                X_parts.append(np.asarray(X_rows, dtype=np.float32))
                M_parts.append(np.asarray(M_rows, dtype=np.float32))
                Y_parts.append(np.asarray(Y_rows, dtype=np.float32))

        return (
            np.vstack(X_parts).astype(np.float32, copy=False),
            np.vstack(M_parts).astype(np.float32, copy=False),
            np.vstack(Y_parts).astype(np.float32, copy=False),
        )

    def train(self, file, nb_epochs=10, callbacks=None, batch_size=16, **fit_kwargs):
        X_train, M_train, Y_train = self._build_training_matrices()

        history = self.autoEncoder.fit(
            [X_train, M_train],
            Y_train,
            batch=batch_size,
            nb_epochs=nb_epochs,
            callbacks=callbacks,
            **fit_kwargs,
        )
        self.autoEncoder.save_weights(resolve_weights_path(file))
        return history

    def compute(self):
        self.encoded_matrix = self.autoEncoder.transform(
            self.character_encoding_input_matrix
        )
        self.encoded_hr_matrix = datas.reshape_matrix_to_hr(
            self.encoded_matrix,
            self.characters_count,
            AppSettings.objects.get(pk=1).ae_hidden_size,
        )

        self.decoded_matrix = np.around(
            self.autoEncoder.predict(self.character_encoding_input_matrix), decimals=3
        )
        self.decoded_hr_matrix = datas.reshape_matrix_to_hr(
            self.decoded_matrix, self.characters_count, self.questions_count
        )

        return [self.encoded_hr_matrix, self.decoded_hr_matrix]

    def export_model(self, filename):
        self.autoEncoder.export(filename)


def get_cached_session(force_reload=False, weights_file: str | Path | None = None):
    """Return a cached AutoEncoderSession with weights loaded from AppSettings.ae_weights_path.

    If the configured path changes, or force_reload is True, rebuild/load the session.
    Returns None if loading failed.
    """
    global _CACHED_SESSION, _CACHED_SESSION_KEY, _CACHED_SESSION_ERROR
    with _CACHE_LOCK:
        try:
            settings_model, _ = AppSettings.objects.get_or_create(pk=1)
            if weights_file is None:
                weights_path = settings_model.ae_weights_path or "auto_encoder_weights.h5"
            else:
                weights_path = str(weights_file)
            resolved_weights_path = resolve_weights_path(weights_path)
            cache_key = (resolved_weights_path, settings_model.ae_hidden_size)
        except Exception as exc:
            _CACHED_SESSION_ERROR = str(exc)
            return None

        if _CACHED_SESSION is not None and not force_reload and _CACHED_SESSION_KEY == cache_key:
            return _CACHED_SESSION

        try:
            session = AutoEncoderSession()
            session.load(resolved_weights_path)
            # compute encoded matrix now to have it cached
            session.compute()
            _save_vectors_to_db(session, resolved_weights_path)
            _CACHED_SESSION = session
            _CACHED_SESSION_KEY = cache_key
            _CACHED_SESSION_ERROR = ""
            return _CACHED_SESSION
        except Exception as exc:
            # keep cache untouched on failure
            _CACHED_SESSION_ERROR = str(exc)
            return None


def get_cached_session_error() -> str:
    return _CACHED_SESSION_ERROR


def _save_vectors_to_db(session: "AutoEncoderSession", weights_path: str) -> None:
    """Persist each character's encoded vector to the DB (CharacterVector table)."""
    try:
        from app.models.CharacterVector import CharacterVector
        inv_map = {v: k for k, v in session.characters_to_index.items()}
        to_create = []
        for idx, vector in enumerate(session.encoded_matrix):
            item_id = inv_map.get(idx)
            if item_id is None:
                continue
            to_create.append(
                CharacterVector(item_id=item_id, vector=vector.tolist(), weights_path=weights_path)
            )
        CharacterVector.objects.all().delete()
        CharacterVector.objects.bulk_create(to_create)
    except Exception:
        pass
