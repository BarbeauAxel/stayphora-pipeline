"""work/<slug>/state.json : étapes terminées, durées et métriques."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class State:
    def __init__(self, path: Path, data: dict[str, Any] | None = None, persist: bool = True):
        self.path = path
        self.persist = persist
        self.data: dict[str, Any] = data or {"version": 1, "created": _now(), "steps": {}}

    @classmethod
    def load(cls, path: Path, persist: bool = True) -> "State":
        if path.is_file():
            return cls(path, json.loads(path.read_text(encoding="utf-8")), persist)
        return cls(path, persist=persist)

    # ── Accès ────────────────────────────────────────────────────────────
    def step(self, num: int) -> dict[str, Any]:
        return self.data["steps"].get(str(num), {})

    def is_done(self, num: int) -> bool:
        return self.step(num).get("status") == "done"

    def metrics(self, num: int) -> dict[str, Any]:
        return self.step(num).get("metrics", {})

    def all_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for _, s in sorted(self.data["steps"].items(), key=lambda kv: int(kv[0])):
            out.update(s.get("metrics", {}))
        return out

    # ── Mise à jour ──────────────────────────────────────────────────────
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def start(self, num: int, name: str) -> None:
        self.data["steps"][str(num)] = {"name": name, "status": "running", "started": _now()}
        self.save()

    def finish(self, num: int, duration_s: float, metrics: dict[str, Any]) -> None:
        s = self.data["steps"][str(num)]
        s.update(status="done", finished=_now(), duration_s=round(duration_s, 1), metrics=metrics)
        self.save()

    def fail(self, num: int, duration_s: float, error: str) -> None:
        s = self.data["steps"].setdefault(str(num), {})
        s.update(status="failed", finished=_now(), duration_s=round(duration_s, 1), error=error)
        self.save()

    def invalidate_from(self, num: int) -> None:
        """Une étape relancée rend obsolètes ses résultats et ceux des étapes suivantes."""
        for k in [k for k in self.data["steps"] if int(k) >= num]:
            del self.data["steps"][k]
        self.save()

    def save(self) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
