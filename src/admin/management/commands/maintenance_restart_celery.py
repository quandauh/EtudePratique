from pathlib import Path
from time import sleep

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from admin.utils.maintenance import maintenance_append_step, maintenance_run_command, maintenance_write_state


class Command(BaseCommand):
    help = "Restart the Celery service from a detached process and report progress for the maintenance UI."

    def add_arguments(self, parser):
        parser.add_argument("task_id")

    def handle(self, *args, **options):
        task_id = options["task_id"]
        repo_dir = Path(settings.ADMIN_DEPLOY_REPO_DIR)
        celery_service_name = settings.ADMIN_DEPLOY_CELERY_SERVICE_NAME

        restart_command = list(settings.ADMIN_DEPLOY_CELERY_RESTART_COMMAND)
        healthcheck_command = list(settings.ADMIN_DEPLOY_CELERY_HEALTHCHECK_COMMAND)

        maintenance_write_state(
            task_id,
            status=f"En cours: redemarrage de {celery_service_name}",
            current_step="Redemarrage du worker Celery",
        )

        restart_result = maintenance_run_command(restart_command, cwd=repo_dir)
        restart_status = "success" if restart_result.returncode == 0 else "failed"
        maintenance_append_step(
            task_id,
            label="Redemarrage du worker Celery",
            command=restart_command,
            status=restart_status,
            output=restart_result.stdout,
            return_code=restart_result.returncode,
        )
        if restart_result.returncode != 0:
            maintenance_write_state(
                task_id,
                status=f"Echec pendant le redemarrage de {celery_service_name}",
                current_step="",
                finished_at=timezone.now().isoformat(),
            )
            return

        healthcheck_output = ""
        healthcheck_code = 1
        for _ in range(settings.ADMIN_DEPLOY_HEALTHCHECK_ATTEMPTS):
            maintenance_write_state(
                task_id,
                status=f"En cours: verification de {celery_service_name}",
                current_step="Verification du worker Celery",
            )
            result = maintenance_run_command(healthcheck_command, cwd=repo_dir)
            healthcheck_output = result.stdout
            healthcheck_code = result.returncode
            if result.returncode == 0 and result.stdout.strip() == "active":
                maintenance_append_step(
                    task_id,
                    label="Verification du worker Celery",
                    command=healthcheck_command,
                    status="success",
                    output=result.stdout,
                    return_code=result.returncode,
                )
                maintenance_write_state(
                    task_id,
                    status=f"Succes: {settings.ADMIN_DEPLOY_SERVICE_NAME} et {celery_service_name} redemarres",
                    current_step="",
                    finished_at=timezone.now().isoformat(),
                )
                return
            sleep(settings.ADMIN_DEPLOY_HEALTHCHECK_INTERVAL)

        maintenance_append_step(
            task_id,
            label="Verification du worker Celery",
            command=healthcheck_command,
            status="failed",
            output=healthcheck_output,
            return_code=healthcheck_code,
        )
        maintenance_write_state(
            task_id,
            status=f"Echec: {celery_service_name} n'est pas actif",
            current_step="",
            finished_at=timezone.now().isoformat(),
        )
