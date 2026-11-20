"""Étape 8 — notification e-mail via l'API HTTP de Resend."""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path

import requests

from ..context import Context, StepError
from ..envsecrets import get_secret, missing
from ..state import State
from . import s7_tour_entry

NAME = "notify"
log = logging.getLogger("pipeline")
RESEND_URL = "https://api.resend.com/emails"


def outputs(ctx: Context) -> list[Path]:
    return []


def _fmt_s(sec: float | None) -> str:
    if sec is None:
        return "—"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


def _q(t: dict | None) -> str:
    if not t:
        return "—"
    parts = [f"PSNR {t['psnr']:.2f} dB" if t.get("psnr") is not None else None,
             f"SSIM {t['ssim']:.3f}" if t.get("ssim") is not None else None,
             f"LPIPS {t['lpips']:.3f}" if t.get("lpips") is not None else None]
    return ", ".join(p for p in parts if p) + (f" ({t['n']} vues)" if t.get("n") else "")


def quality_lines(ctx: Context, m: dict) -> list[str]:
    lines = []
    if "psnr" in m:
        lines.append("Qualité (vues de test, trainer) : " + _q({k: m.get(k) for k in ("psnr", "ssim", "lpips")}))
    qt = m.get("quality_train")
    if qt:
        lines.append("Contrôle visuel entraînement : " + (f"impossible ({qt['error']})" if "error" in qt else
                     f"test {_q(qt.get('test'))} ; vues d'entraînement {_q(qt.get('train_views'))}"))
    for v in ctx.cfg["export"]["variants"]:
        q = m.get(f"quality_{v}")
        if not q:
            continue
        if "error" in q:
            lines.append(f"Contrôle visuel {v} : impossible ({q['error']})")
            continue
        drop = q.get("psnr_drop_db")
        lines.append(f"Contrôle visuel {v} (.spz relu) : test {_q(q.get('test'))}"
                     + (f" ; baisse {drop:+.2f} dB (max {ctx.cfg['quality']['max_psnr_drop_db'].get(v)})"
                        if drop is not None else ""))
    if m.get("quality_sheet"):
        lines.append(f"Planche 4 vues : pièce jointe {Path(m['quality_sheet']).name}")
    return lines


def attachments(m: dict) -> list[dict]:
    p = Path(m["quality_sheet"]) if m.get("quality_sheet") else None
    if not p or not p.is_file() or p.stat().st_size > 10_000_000:
        return []
    return [{"filename": p.name, "content": base64.b64encode(p.read_bytes()).decode("ascii")}]


def build_report(ctx: Context, state: State, error: str | None = None) -> tuple[str, str]:
    m = state.all_metrics()
    url = f"{ctx.cfg['publish']['tours_url'].rstrip('/')}/{ctx.slug}"
    lines = [f"Visite : {ctx.slug}"]
    if error:
        lines.append(f"ÉCHEC : {error}")
    lines.append("")
    lines.append("Durées :")
    total = 0.0
    for k, s in sorted(state.data["steps"].items(), key=lambda kv: int(kv[0])):
        d = s.get("duration_s")
        total += d or 0
        lines.append(f"  {k}. {s.get('name', '?'):<11} {_fmt_s(d):>12}  {s.get('status', '')}")
    lines.append(f"  total       {_fmt_s(total):>12}")
    lines.append("")
    if "aligned_images" in m:
        lines.append(f"Images alignées exploitables : {m['aligned_images']}/{m.get('frames_total')} "
                     f"({100 * m.get('aligned_ratio', 0):.1f} %, mapper {m.get('mapper')}"
                     + (f", {m['registered_images']} recalées dont {len(m.get('images_rejected') or [])} écartées"
                        if "registered_images" in m else "") + ")")
    ae = m.get("alignment_estimated")
    if ae:
        lines.append(f"Alignement : caméra supposée à {ae['camera_height_assumed_m']} m, échelle "
                     f"{ae['scale_m_per_unit']:.4g} m/unité, verticale à {ae['vertical_angle_deg']}° de -Y COLMAP, "
                     f"roulis caméras médian {ae.get('camera_roll_median_deg')}° (p90 {ae.get('camera_roll_p90_deg')}°), "
                     f"tangage médian {ae.get('camera_pitch_median_deg')}°")
        fc = ae.get("floor_check") or {}
        if "tilt_deg" in fc:
            lines.append(f"Contrôle du sol : plan à {fc['tilt_deg']}° de l'horizontale (cible < {fc['target_deg']}°, "
                         f"alerte > {fc['warn_deg']}°) -> {fc['status']}")
        elif fc:
            lines.append(f"Contrôle du sol : indéterminé ({fc.get('reason')})")
    aa = m.get("alignment_applied")
    if aa and any(v not in (0, 0.0, 1, 1.0) for v in aa["overrides"].values()):
        lines.append(f"Corrections manuelles : {aa['overrides']}")
    if "gaussians_trained" in m:
        lines.append(f"Gaussiennes : {m['gaussians_trained']} entraînées, {m.get('gaussians_cleaned', '?')} après nettoyage")
    lines += quality_lines(ctx, m)
    for v in ctx.cfg["export"]["variants"]:
        if f"{v}_size_mb" in m:
            lines.append(f"{v} : {m.get(f'{v}_splats')} splats, SH {m.get(f'{v}_sh_bands')}, {m[f'{v}_size_mb']} Mo")
    for name, u in (m.get("published_urls") or {}).items():
        lines.append(f"Fichier : {u}")
    if not error:
        lines += ["", f"Lien (actif après ajout dans content/fr/tours.ts + déploiement) : {url}"]
        p = s7_tour_entry.entry_path(ctx)
        if p.is_file():
            lines += ["", "tour-entry.ts :", p.read_text(encoding="utf-8")]
    text = "\n".join(lines)
    subject = f"[Stayphora] {'ÉCHEC' if error else 'Visite prête'} : {ctx.slug}"
    return subject, text


def send(ctx: Context, subject: str, text: str, files: list[dict] | None = None) -> str:
    n = ctx.cfg["notify"]
    body = {
        "from": n["from"],
        "to": [n["to"]],
        "subject": subject,
        "text": text,
        "html": f"<pre style=\"font-family:ui-monospace,monospace;font-size:13px\">{html.escape(text)}</pre>",
    }
    if files:
        body["attachments"] = files
    r = requests.post(RESEND_URL, json=body, timeout=30,
                      headers={"Authorization": f"Bearer {get_secret('RESEND_API_KEY')}"})
    if r.status_code >= 300:
        raise StepError(f"Resend a refusé l'envoi : HTTP {r.status_code} {r.text[:300]}")
    return r.json().get("id", "")


def run(ctx: Context) -> dict:
    state: State = ctx.state
    n = ctx.cfg["notify"]
    if not n["enabled"]:
        log.info("Notification désactivée (notify.enabled = false)")
        return {"email_sent": False}
    subject, text = build_report(ctx, state)
    if ctx.dry_run:
        log.info("[dry-run] e-mail « %s » -> %s", subject, n["to"])
        return {}
    if missing(("RESEND_API_KEY",)):
        raise StepError("variable d'environnement RESEND_API_KEY manquante")
    email_id = send(ctx, subject, text, attachments(state.all_metrics()))
    log.info("E-mail envoyé à %s (id %s)", n["to"], email_id)
    return {"email_sent": True, "email_id": email_id}


def verify(ctx: Context, metrics: dict) -> None:
    if ctx.cfg["notify"]["enabled"] and not metrics.get("email_id"):
        raise StepError("Resend n'a pas renvoyé d'identifiant d'e-mail")


def notify_failure(ctx: Context, state: State, error: str) -> None:
    """Best effort : prévient en cas d'échec (utile sur RunPod, sans surveillance)."""
    if ctx.dry_run or not ctx.cfg["notify"]["enabled"] or missing(("RESEND_API_KEY",)):
        return
    try:
        subject, text = build_report(ctx, state, error=error)
        send(ctx, subject, text)
        log.info("E-mail d'échec envoyé à %s", ctx.cfg["notify"]["to"])
    except Exception as exc:  # noqa: BLE001 — ne jamais masquer l'erreur d'origine
        log.warning("E-mail d'échec non envoyé : %s", exc)
