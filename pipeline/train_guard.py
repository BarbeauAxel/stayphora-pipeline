"""Garde-fou de l'entraînement gsplat : lit la sortie du trainer ligne à ligne et demande l'arrêt
si l'entraînement diverge, pour ne pas payer 30 000 itérations inutiles.

Signaux (sortie de examples/simple_trainer.py mcmc, gsplat 1.5.3) :
  « Scene scale: X »                       échelle de scène (dès le démarrage)
  « loss=0.564| sh degree=3| : ... 5783/30000 »   perte de l'itération et compteur
  « Step 600: Added 698 GSs. Now having 14664 GSs. »
  « Step 600: Relocated 8294 GSs. »         gaussiennes mortes (opacité ~0) replacées

Essai RunPod du 20/11/2026 (divergence) : échelle 840 ; perte médiane 0,26 vers 500 puis 0,38 à
1 000 et 0,61 à 2 000 ; 57 % puis 71 % des gaussiennes relocalisées à chaque cycle ; plantage
torch.multinomial à 5 800 quand toutes sont mortes.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np

_SCALE = re.compile(r"Scene scale:\s*([0-9.eE+-]+)")
_LOSS = re.compile(r"loss=([0-9.]+)\|.*?(\d+)/(\d+)\s*\[")
_TOTAL = re.compile(r"Step (\d+): Added \d+ GSs\. Now having (\d+) GSs")
_RELOC = re.compile(r"Step (\d+): Relocated (\d+) GSs")


@dataclass
class TrainGuard:
    checkpoints: list[int] = field(default_factory=lambda: [1000, 3000])
    loss_rise_ratio: float = 1.3       # perte (médiane glissante) > 1,3 x son minimum ...
    loss_rise_min: float = 0.03        # ... ET > minimum + 0,03
    max_dead_fraction: float = 0.5     # > 50 % des gaussiennes relocalisées en un cycle
    min_gaussians_ratio: float = 0.5   # nombre de gaussiennes < 50 % de son maximum
    scene_scale_range: tuple[float, float] = (0.5, 10.0)
    window: int = 200                  # nb de valeurs de perte de la médiane glissante
    warmup: int = 300                  # itérations ignorées pour le minimum de perte
    margin: int = 50                   # contrôle à checkpoint + 50 (le .ply partiel est écrit)

    def __post_init__(self) -> None:
        self.scene_scale: float | None = None
        self.step = 0
        self.losses: deque[float] = deque(maxlen=self.window)
        self.min_loss: float | None = None
        self.min_loss_step = 0
        self.total = 0
        self.max_total = 0
        self.last_reloc: tuple[int, int] | None = None   # (étape, nb relocalisées)
        self.pending = sorted(self.checkpoints)
        self.report: dict = {}

    def median_loss(self) -> float | None:
        return float(np.median(self.losses)) if len(self.losses) >= min(50, self.window) else None

    def feed(self, line: str) -> str | None:
        """Ingère une ligne ; renvoie la raison d'arrêt, ou None."""
        m = _SCALE.search(line)
        if m:
            self.scene_scale = float(m.group(1))
            self.report["scene_scale"] = round(self.scene_scale, 4)
            lo, hi = self.scene_scale_range
            if not lo <= self.scene_scale <= hi:
                return (f"échelle de scène gsplat {self.scene_scale:.3g} hors de [{lo}, {hi}] : poses ou "
                        "repère d'entraînement incohérents (cf. 3_sfm/train)")
        m = _TOTAL.search(line)
        if m:
            self.total = int(m.group(2))
            self.max_total = max(self.max_total, self.total)
        m = _RELOC.search(line)
        if m:
            self.last_reloc = (int(m.group(1)), int(m.group(2)))
        m = _LOSS.search(line)
        if m:
            self.step = int(m.group(2))
            self.losses.append(float(m.group(1)))
            med = self.median_loss()
            if med is not None and self.step >= self.warmup and (self.min_loss is None or med < self.min_loss):
                self.min_loss, self.min_loss_step = med, self.step
        if self.pending and self.step >= self.pending[0] + self.margin:
            return self.check(self.pending.pop(0))
        return None

    def check(self, cp: int) -> str | None:
        med = self.median_loss()
        dead = None
        if self.last_reloc and self.total:
            dead = self.last_reloc[1] / self.total
        self.report[f"step_{cp}"] = {
            "loss_median": None if med is None else round(med, 4),
            "loss_min": None if self.min_loss is None else round(self.min_loss, 4),
            "loss_min_step": self.min_loss_step,
            "gaussians": self.total,
            "relocated_fraction": None if dead is None else round(dead, 3),
        }
        reasons = []
        if med is not None and self.min_loss is not None and med > self.loss_rise_ratio * self.min_loss \
                and med - self.min_loss > self.loss_rise_min:
            reasons.append(f"perte remontée à {med:.3f} (minimum {self.min_loss:.3f} à l'itération "
                           f"{self.min_loss_step})")
        if dead is not None and dead > self.max_dead_fraction:
            reasons.append(f"{100 * dead:.0f} % des gaussiennes mortes et relocalisées à l'itération "
                           f"{self.last_reloc[0]}")
        if self.max_total and self.total < self.min_gaussians_ratio * self.max_total:
            reasons.append(f"nombre de gaussiennes effondré : {self.total} (max {self.max_total})")
        if reasons:
            return f"divergence détectée à l'itération {cp} : " + " ; ".join(reasons)
        return None
