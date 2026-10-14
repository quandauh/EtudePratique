import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "devinsa.settings")

app = Celery("devinsa")

app.config_from_object("django.conf:settings", namespace="CELERY")

app.autodiscover_tasks()

print("test")

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request : {self.request!r}")