"""Contexte partagé entre les étapes : chemins, configuration, journalisation, exécution."""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .envsecrets import redact

log = logging.getLogger("pipeline")


class StepError(RuntimeError):
    """Échec d'une étape, avec un message lisible pour l'utilisateur."""


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_console_logging(verbose: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:  # console Windows en cp1252 : accents et ─ illisibles sinon
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    root = logging.getLogger("pipeline")
    root.setLevel(logging.DEBUG)
    if not any(getattr(h, "_stayphora_console", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h._stayphora_console = True  # type: ignore[attr-defined]
        h.setLevel(logging.DEBUG if verbose else logging.INFO)
        h.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        root.addHandler(h)


def fmt_cmd(cmd: Sequence[Any]) -> str:
    return redact(" ".join(shlex.quote(str(c)) for c in cmd))


@dataclass
class Context:
    slug: str
    cfg: dict
    root: Path                      # work_dir/<slug>
    input: str | None = None        # chemin ou URL de la vidéo
    dry_run: bool = False
    step_num: int = 0
    step_name: str = ""
    state: Any = None               # pipeline.state.State, renseigné par le runner
    _file_handler: logging.Handler | None = field(default=None, repr=False)

    # ── Chemins ──────────────────────────────────────────────────────────
    def dir(self, name: str, create: bool = True) -> Path:
        p = self.root / name
        if create and not self.dry_run:
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    # ── Journal par étape ────────────────────────────────────────────────
    def begin_step(self, num: int, name: str) -> Path | None:
        self.end_step()
        self.step_num, self.step_name = num, name
        if self.dry_run:
            return None
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.logs_dir / f"{stamp}_{num}_{name}.log"
        h = logging.FileHandler(path, encoding="utf-8")
        h.setLevel(logging.DEBUG)
        h.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(h)
        self._file_handler = h
        return path

    def end_step(self) -> None:
        if self._file_handler is not None:
            log.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    # ── Exécution de commandes ───────────────────────────────────────────
    def run(self, cmd: Sequence[Any], cwd: Path | None = None, env: dict | None = None,
            check: bool = True, on_line: Any = None) -> int:
        """Exécute une commande en recopiant sa sortie dans le journal de l'étape.

        on_line(ligne) -> str | None : si elle renvoie un texte, le processus est arrêté et
        StepError(texte) levée (garde-fou d'entraînement). En --dry-run, affiche seulement la
        commande et renvoie 0.
        """
        cmd = [str(c) for c in cmd]
        prefix = "[dry-run] " if self.dry_run else "$ "
        log.info("%s%s", prefix, fmt_cmd(cmd))
        if self.dry_run:
            return 0
        exe = shutil.which(cmd[0]) or cmd[0]
        try:
            proc = subprocess.Popen(
                [exe, *cmd[1:]], cwd=cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except FileNotFoundError as exc:
            raise StepError(f"commande introuvable : {cmd[0]} ({exc})") from exc
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            # Les barres de progression réécrivent la ligne avec \r : on ne garde que la fin.
            if "\r" in line:
                line = line.rsplit("\r", 1)[-1]
            if line:
                log.debug("  | %s", line)
                stop = on_line(line) if on_line else None
                if stop:
                    log.error("ARRÊT : %s", stop)
                    proc.kill()
                    proc.wait()
                    raise StepError(stop)
        code = proc.wait()
        if check and code != 0:
            raise StepError(f"« {cmd[0]} » a échoué (code {code}). Voir le journal de l'étape dans {self.logs_dir}")
        return code

    def capture(self, cmd: Sequence[Any]) -> str:
        """Exécute une commande courte et renvoie sa sortie (même en dry-run : lecture seule)."""
        cmd = [str(c) for c in cmd]
        exe = shutil.which(cmd[0]) or cmd[0]
        try:
            res = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, encoding="utf-8",
                                 errors="replace")
        except FileNotFoundError as exc:
            raise StepError(f"commande introuvable : {cmd[0]}") from exc
        return res.stdout + res.stderr
