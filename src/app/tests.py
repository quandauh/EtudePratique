"""
Tests pour la feature : switch algorithme TF-IDF / Auto-encodeur depuis l'admin.

Ce qu'on teste :
1. Le modèle AppSettings stocke bien algorithm_mode et ae_weights_path
2. toggle_algorithm_mode (POST) change bien le mode en base
3. toggle_algorithm_mode avec une valeur invalide ne change rien
4. Le context processor injecte current_mode dans le template
5. La navbar admin affiche le bon mode en gras
6. La view guess utilise search_best_item_autoencoder quand mode=autoencoder
7. La view guess utilise search_best_item quand mode=tfidf
8. upload_weights : rejet extension non-h5
9. upload_weights : rejet fichier trop gros
"""

from unittest.mock import patch, MagicMock
from django.test import TestCase, Client, RequestFactory
from django.contrib.auth.models import User
from django.urls import reverse

from app.models import AppSettings


class AppSettingsModelTest(TestCase):
    def test_default_values(self):
        s, created = AppSettings.objects.get_or_create(pk=1)
        self.assertEqual(s.algorithm_mode, "tfidf")
        self.assertEqual(s.ae_weights_path, "auto_encoder_weights.h5")

    def test_switch_to_autoencoder(self):
        s, _ = AppSettings.objects.get_or_create(pk=1)
        s.algorithm_mode = "autoencoder"
        s.save()
        s.refresh_from_db()
        self.assertEqual(s.algorithm_mode, "autoencoder")

    def test_switch_back_to_tfidf(self):
        s, _ = AppSettings.objects.get_or_create(pk=1)
        s.algorithm_mode = "autoencoder"
        s.save()
        s.algorithm_mode = "tfidf"
        s.save()
        s.refresh_from_db()
        self.assertEqual(s.algorithm_mode, "tfidf")


class ToggleAlgorithmModeViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_superuser("admin", "admin@test.com", "password")
        self.client.login(username="admin", password="password")
        AppSettings.objects.get_or_create(pk=1)

    def test_switch_to_autoencoder(self):
        url = reverse("admin:toggle_algorithm_mode")
        response = self.client.post(url, {"mode": "autoencoder"})
        self.assertRedirects(response, reverse("admin:admin_index"), fetch_redirect_response=False)
        s = AppSettings.objects.get(pk=1)
        self.assertEqual(s.algorithm_mode, "autoencoder")

    def test_switch_to_tfidf(self):
        s = AppSettings.objects.get(pk=1)
        s.algorithm_mode = "autoencoder"
        s.save()

        url = reverse("admin:toggle_algorithm_mode")
        response = self.client.post(url, {"mode": "tfidf"})
        self.assertRedirects(response, reverse("admin:admin_index"), fetch_redirect_response=False)
        s.refresh_from_db()
        self.assertEqual(s.algorithm_mode, "tfidf")

    def test_invalid_mode_does_not_change(self):
        s = AppSettings.objects.get(pk=1)
        original = s.algorithm_mode

        url = reverse("admin:toggle_algorithm_mode")
        self.client.post(url, {"mode": "invalid_algo"})
        s.refresh_from_db()
        self.assertEqual(s.algorithm_mode, original)

    def test_get_method_redirects(self):
        """GET sur toggle_algorithm_mode doit aussi rediriger (pas de crash)."""
        url = reverse("admin:toggle_algorithm_mode")
        response = self.client.get(url)
        self.assertRedirects(response, reverse("admin:admin_index"), fetch_redirect_response=False)

    def test_requires_login(self):
        self.client.logout()
        url = reverse("admin:toggle_algorithm_mode")
        response = self.client.post(url, {"mode": "autoencoder"})
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])


class ContextProcessorTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_superuser("admin2", "a@test.com", "password")
        self.client.login(username="admin2", password="password")

    def test_current_mode_tfidf_in_context(self):
        s, _ = AppSettings.objects.get_or_create(pk=1)
        s.algorithm_mode = "tfidf"
        s.save()
        response = self.client.get(reverse("admin:admin_index"))
        self.assertEqual(response.context["current_mode"], "tfidf")

    def test_current_mode_autoencoder_in_context(self):
        s, _ = AppSettings.objects.get_or_create(pk=1)
        s.algorithm_mode = "autoencoder"
        s.save()
        response = self.client.get(reverse("admin:admin_index"))
        self.assertEqual(response.context["current_mode"], "autoencoder")

    def test_navbar_highlights_active_mode(self):
        s, _ = AppSettings.objects.get_or_create(pk=1)
        s.algorithm_mode = "autoencoder"
        s.save()
        response = self.client.get(reverse("admin:admin_index"))
        # Le bouton actif doit avoir la classe fw-bold
        self.assertContains(response, 'class="dropdown-item fw-bold">Auto-encodeur')
        self.assertContains(response, 'class="dropdown-item">TF-IDF')


class GuessViewAlgorithmTest(TestCase):
    """Vérifie que la view guess appelle la bonne fonction selon algorithm_mode."""

    def setUp(self):
        self.client = Client()
        AppSettings.objects.get_or_create(pk=1)

    def _set_mode(self, mode):
        s = AppSettings.objects.get(pk=1)
        s.algorithm_mode = mode
        s.save()

    def _make_session_with_answers(self):
        """Simule une session de jeu terminée avec des réponses."""
        from app.models import Tree, Question
        # On a besoin d'un Tree et d'une Question minimaux
        q = Question.objects.create(title="Question test")
        tree = Tree.objects.create(question=q, depth=5)
        session = self.client.session
        session["tree"] = tree.id
        session["answers"] = [{"question_id": q.id, "choice": "y"}]
        session["depth"] = 6
        session.save()
        return tree.id

    @patch("app.views.search_best_item")
    @patch("app.views.search_best_item_autoencoder")
    def test_guess_uses_tfidf_when_mode_tfidf(self, mock_ae, mock_tfidf):
        self._set_mode("tfidf")
        mock_tfidf.return_value = -1  # -1 = pas de perso trouvé → redirect
        self._make_session_with_answers()
        self.client.get(reverse("app:guess"))
        mock_tfidf.assert_called_once()
        mock_ae.assert_not_called()

    @patch("app.views.search_best_item")
    @patch("app.views.search_best_item_autoencoder")
    def test_guess_uses_autoencoder_when_mode_autoencoder(self, mock_ae, mock_tfidf):
        self._set_mode("autoencoder")
        mock_ae.return_value = -1
        self._make_session_with_answers()
        self.client.get(reverse("app:guess"))
        mock_ae.assert_called_once()
        mock_tfidf.assert_not_called()


class UploadWeightsViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_superuser("admin3", "b@test.com", "password")
        self.client.login(username="admin3", password="password")
        AppSettings.objects.get_or_create(pk=1)

    def test_get_shows_form(self):
        response = self.client.get(reverse("admin:upload_weights"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "weights_file")

    def test_rejects_non_h5_file(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        bad_file = SimpleUploadedFile("weights.txt", b"not a real model", content_type="text/plain")
        response = self.client.post(reverse("admin:upload_weights"), {"weights_file": bad_file})
        self.assertRedirects(response, reverse("admin:upload_weights"), fetch_redirect_response=False)
        # Le message d'erreur doit être dans la session
        self.assertIn("admin_upload_error", self.client.session)

    def test_rejects_oversized_file(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        # Créer un faux fichier .h5 trop gros (>100MB)
        huge_data = b"x" * (101 * 1024 * 1024)
        big_file = SimpleUploadedFile("big.h5", huge_data, content_type="application/octet-stream")
        response = self.client.post(reverse("admin:upload_weights"), {"weights_file": big_file})
        self.assertRedirects(response, reverse("admin:upload_weights"), fetch_redirect_response=False)
        self.assertIn("admin_upload_error", self.client.session)

    def test_accepts_path_input(self):
        response = self.client.post(
            reverse("admin:upload_weights"),
            {"weights_path": "my_weights.h5"}
        )
        self.assertRedirects(response, reverse("admin:admin_index"), fetch_redirect_response=False)
        s = AppSettings.objects.get(pk=1)
        self.assertEqual(s.ae_weights_path, "my_weights.h5")
