"""Traçabilité du code exécuté : empreinte du dossier pipeline/ et origine (image ou envoi local).

L'empreinte est calculée de la même façon en local (scripts/runpod_job.py, avant l'envoi) et sur
le pod (au démarrage du pipeline) : on peut donc vérifier que le code qui a tourné est bien
celui qui a été envoyé. ORIGIN.json (écrit par runpod_job.py dans l'archive envoyée) donne le
commit git et la présence de modifications non commitées.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ORIGIN_FILE = "ORIGIN.json"


def included(rel: str) -> bool:
    parts = rel.split("/")
    return "__pycache__" not in parts and not rel.endswith(".pyc") and rel != ORIGIN_FILE


def fingerprint(pkg_dir: Path = PKG_DIR) -> dict:
    """SHA-256 sur (chemin relatif POSIX, contenu octet pour octet) de chaque fichier, triés."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(pkg_dir.rglob("*")):
        rel = p.relative_to(pkg_dir).as_posix()
        if not p.is_file() or not included(rel):
            continue
        h.update(rel.encode() + b"\0" + p.read_bytes() + b"\0")
        n += 1
    return {"sha256": h.hexdigest(), "files": n}


def code_info(pkg_dir: Path = PKG_DIR) -> dict:
    fp = fingerprint(pkg_dir)
    origin_path = pkg_dir / ORIGIN_FILE
    if origin_path.is_file():
        origin = json.loads(origin_path.read_text(encoding="utf-8"))
    else:
        origin = {"source": "image-ou-local", "git_commit": None, "git_dirty": None}
    info = {**origin, "sha256": fp["sha256"], "files": fp["files"]}
    if origin.get("sha256_sent"):
        info["matches_sent"] = origin["sha256_sent"] == fp["sha256"]
    return info
