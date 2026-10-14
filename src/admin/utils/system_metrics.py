import os
import shutil
from pathlib import Path
from subprocess import TimeoutExpired, run

from django.conf import settings


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "-"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} %"


def _read_memory() -> dict[str, object]:
    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as file:
            for line in file:
                key, raw_value = line.split(":", 1)
                value = raw_value.strip().split()[0]
                meminfo[key] = int(value) * 1024
    except (FileNotFoundError, OSError, ValueError):
        return {
            "total": "-",
            "available": "-",
            "used": "-",
            "percent": "-",
        }

    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable")
    used = None if total is None or available is None else total - available
    percent = None if total in (None, 0) or used is None else used / total * 100
    return {
        "total": _format_bytes(total),
        "available": _format_bytes(available),
        "used": _format_bytes(used),
        "percent": _format_percent(percent),
    }


def _disk_usage(path: str | Path) -> dict[str, object]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {
            "path": str(path),
            "total": "-",
            "used": "-",
            "free": "-",
            "percent": "-",
        }

    percent = usage.used / usage.total * 100 if usage.total else None
    return {
        "path": str(path),
        "total": _format_bytes(usage.total),
        "used": _format_bytes(usage.used),
        "free": _format_bytes(usage.free),
        "percent": _format_percent(percent),
    }


def _service_status(candidates: list[str]) -> dict[str, str]:
    fallback = {"name": candidates[0] if candidates else "-", "state": "unknown"}
    for service in dict.fromkeys(candidates):
        if not service:
            continue
        try:
            result = run(
                ["systemctl", "is-active", service],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, TimeoutExpired):
            if fallback["state"] == "unknown":
                fallback = {"name": service, "state": "timeout"}
            continue
        state = (result.stdout or result.stderr).strip() or "unknown"
        if result.returncode == 0:
            return {"name": service, "state": state}
        if fallback["state"] == "unknown" and state != "unknown":
            fallback = {"name": service, "state": state}

    return fallback


def _process_snapshot() -> list[dict[str, object]]:
    categories = {
        "Backend": ["uwsgi", "gunicorn", "manage.py runserver"],
        "Celery": ["celery"],
        "MariaDB": ["mariadbd", "mysqld"],
        "Redis/Valkey": ["redis-server", "valkey-server"],
        "Arbre / moulinette": ["create_tree", "moulinette", "compute_visualisation", "compute_suggestions"],
    }
    try:
        result = run(
            ["ps", "-eo", "pid=,pcpu=,pmem=,args=", "--sort=-pcpu"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, cpu, memory, command = parts
        command_lower = command.lower()
        category = next(
            (
                name
                for name, needles in categories.items()
                if any(needle in command_lower for needle in needles)
            ),
            None,
        )
        if category is None:
            continue
        rows.append(
            {
                "category": category,
                "pid": pid,
                "cpu": f"{float(cpu):.1f} %",
                "memory": f"{float(memory):.1f} %",
                "command": command[:140],
            }
        )
        if len(rows) >= 10:
            break
    return rows


def collect_system_metrics() -> dict[str, object]:
    load_average = None
    try:
        load_average = os.getloadavg()
    except OSError:
        pass

    repo_dir = Path(getattr(settings, "ADMIN_DEPLOY_REPO_DIR", settings.BASE_DIR.parent))
    services = [
        {
            "label": "Backend",
            **_service_status([settings.ADMIN_DEPLOY_SERVICE_NAME, "uwsgi", "gunicorn"]),
        },
        {
            "label": "Celery",
            **_service_status([settings.ADMIN_DEPLOY_CELERY_SERVICE_NAME, "celery"]),
        },
        {
            "label": "MariaDB",
            **_service_status(["mariadb", "mysql"]),
        },
        {
            "label": "Redis / Valkey",
            **_service_status(["valkey", "redis", "redis-server"]),
        },
    ]

    return {
        "cpu": {
            "cores": os.cpu_count() or "-",
            "load_1": "-" if load_average is None else f"{load_average[0]:.2f}",
            "load_5": "-" if load_average is None else f"{load_average[1]:.2f}",
            "load_15": "-" if load_average is None else f"{load_average[2]:.2f}",
        },
        "memory": _read_memory(),
        "disk": _disk_usage(repo_dir),
        "services": services,
        "processes": _process_snapshot(),
    }
