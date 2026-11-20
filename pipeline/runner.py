"""Enchaîne les étapes : chronométrage, vérification, state.json, reprise (--from-step)."""

from __future__ import annotations

import logging
import time

from .context import Context, StepError
from .provenance import code_info
from .state import State
from .steps import STEPS, s8_notify

log = logging.getLogger("pipeline")


def step_label(num: int) -> str:
    return f"{num}. {STEPS[num - 1].NAME}"


def check_resumable(ctx: Context, state: State, from_step: int) -> None:
    """Avant de reprendre à l'étape n, les étapes 1..n-1 doivent être terminées et leurs fichiers présents."""
    for num in range(1, from_step):
        mod = STEPS[num - 1]
        if not state.is_done(num):
            raise StepError(f"impossible de reprendre à l'étape {from_step} : l'étape {step_label(num)} "
                            "n'est pas terminée dans state.json")
        absent = [str(p) for p in mod.outputs(ctx) if not p.exists()]
        if absent:
            raise StepError(f"impossible de reprendre à l'étape {from_step} : fichiers de l'étape "
                            f"{step_label(num)} absents : {absent}")


def run_pipeline(ctx: Context, from_step: int = 1, to_step: int | None = None,
                 skip: set[int] | None = None) -> State:
    to_step = to_step or len(STEPS)
    skip = skip or set()
    state = State.load(ctx.root / "state.json", persist=not ctx.dry_run)
    ctx.state = state
    if not ctx.dry_run:
        check_resumable(ctx, state, from_step)
        state.invalidate_from(from_step)
        state.set("slug", ctx.slug)
        code = code_info()
        state.set("code", code)
        log.info("Code : %s (sha256 %s, %d fichiers%s)", code.get("source"), code["sha256"][:12], code["files"],
                 "" if code.get("matches_sent") is not False else " — DIFFÉRENT de l'envoi !")
        if ctx.input and from_step == 1:
            state.set("input", ctx.input if "://" not in ctx.input else ctx.input.split("?")[0])

    for num in range(from_step, to_step + 1):
        mod = STEPS[num - 1]
        if num in skip:
            log.info("── Étape %s : ignorée", step_label(num))
            continue
        logfile = ctx.begin_step(num, mod.NAME)
        log.info("── Étape %s%s", step_label(num), f"  (journal : {logfile})" if logfile else "")
        state.start(num, mod.NAME)
        t0 = time.monotonic()
        try:
            metrics = mod.run(ctx) or {}
            if not ctx.dry_run:
                mod.verify(ctx, metrics)
        except BaseException as exc:
            dt = time.monotonic() - t0
            msg = str(exc) if isinstance(exc, StepError) else f"{type(exc).__name__}: {exc}"
            if isinstance(exc, KeyboardInterrupt):
                msg = "interrompu (Ctrl+C)"
            state.fail(num, dt, msg)
            log.error("Étape %s en échec après %.0f s : %s", step_label(num), dt, msg)
            if not isinstance(exc, StepError):
                log.debug("Trace :", exc_info=True)
            log.error("Reprendre ensuite avec : --from-step %d", num)
            ctx.end_step()
            if mod is not s8_notify and not isinstance(exc, KeyboardInterrupt):
                s8_notify.notify_failure(ctx, state, f"étape {step_label(num)} : {msg}")
            raise
        dt = time.monotonic() - t0
        state.finish(num, dt, metrics)
        log.info("   ok en %.1f s %s", dt, _short(metrics))
        ctx.end_step()
    return state


def _short(metrics: dict) -> str:
    keep = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    return str(keep) if keep else ""
