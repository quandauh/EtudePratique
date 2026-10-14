import shlex
import subprocess
from pathlib import Path

from django.conf import settings
from django.core.cache import cache

MAINTENANCE_DEPLOY_CACHE_PREFIX = "maintenance:deploy"


def maintenance_cache_key(task_id: str) -> str:
    return f"{MAINTENANCE_DEPLOY_CACHE_PREFIX}:{task_id}"


def maintenance_default_state(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "status": "En attente",
        "current_step": "",
        "steps": [],
        "started_at": "",
        "finished_at": "",
    }


def maintenance_read_state(task_id: str) -> dict[str, object]:
    return cache.get(maintenance_cache_key(task_id), maintenance_default_state(task_id))


def maintenance_write_state(task_id: str, **updates) -> dict[str, object]:
    state = maintenance_read_state(task_id)
    state.update(updates)
    cache.set(
        maintenance_cache_key(task_id),
        state,
        timeout=settings.ADMIN_DEPLOY_STATUS_TIMEOUT,
    )
    return state


def maintenance_append_step(
    task_id: str,
    *,
    label: str,
    command: list[str],
    status: str,
    output: str,
    return_code: int,
) -> dict[str, object]:
    state = maintenance_read_state(task_id)
    steps = list(state.get("steps", []))
    steps.append(
        {
            "label": label,
            "command": shlex.join(command),
            "status": status,
            "output": output.strip() or "(aucune sortie)",
            "return_code": return_code,
        }
    )
    state["steps"] = steps
    cache.set(
        maintenance_cache_key(task_id),
        state,
        timeout=settings.ADMIN_DEPLOY_STATUS_TIMEOUT,
    )
    return state


def maintenance_run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
