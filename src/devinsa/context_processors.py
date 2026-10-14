def algorithm_mode(request):
    try:
        from app.models import AppSettings
        settings, _ = AppSettings.objects.get_or_create(pk=1)
        return {"current_mode": settings.algorithm_mode}
    except Exception:
        return {"current_mode": "tfidf"}
