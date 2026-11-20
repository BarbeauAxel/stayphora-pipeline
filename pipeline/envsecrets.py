"""Secrets : lus UNIQUEMENT dans l'environnement (ou un fichier .env), jamais affichés."""

from __future__ import annotations

import os
import re
from pathlib import Path

SECRET_NAMES = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "RESEND_API_KEY",
    "RUNPOD_API_KEY",
)

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Charge un .env minimal (KEY=VALUE). Ne remplace jamais une variable déjà définie."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value and key not in os.environ:
            os.environ[key] = value


def get_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"variable d'environnement {name} manquante (voir .env.example)")
    return value


def missing(names: tuple[str, ...]) -> list[str]:
    return [n for n in names if not os.environ.get(n)]


def redact(text: str) -> str:
    """Remplace toute valeur de secret présente dans le texte par ***."""
    for name in SECRET_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return text
