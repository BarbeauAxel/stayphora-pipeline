#!/usr/bin/env python3
"""Lance un job du pipeline sur un pod GPU RunPod, puis SUPPRIME le pod, même en cas d'erreur.

    python scripts/runpod_job.py --input C:/videos/studio.mp4 --slug studio-lyon
    python scripts/runpod_job.py --test-dataset --slug test-playroom      # dataset public
    python scripts/runpod_job.py --list-gpus                              # identifiants GPU

Étapes : création du pod -> attente SSH -> envoi de la vidéo (scp) -> lancement du pipeline
en arrière-plan sur le pod -> suivi du journal (reconnexion si la connexion tombe) ->
rapatriement des journaux, state.json, tour-entry.ts -> suppression du pod.

Trois filets contre un pod oublié qui facture :
  1. ce script supprime le pod dans un `finally` (erreur, Ctrl+C, fermeture du terminal) ;
  2. sur le pod, un minuteur supprime le pod après --max-hours (défaut 3 h) ;
  3. sur le pod, le job se supprime lui-même 20 min après la fin (le temps de rapatrier).
Les filets 2 et 3 utilisent la clé « pod-scoped » injectée par RunPod (ou --pass-terminate-key).

Prérequis local : pip install runpod ; ssh/scp (OpenSSH, inclus dans Windows 10+) ; une clé SSH.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.config import ConfigError, default_work_dir, load_config, validate_slug  # noqa: E402
from pipeline.envsecrets import load_dotenv, missing, redact  # noqa: E402

DEFAULT_IMAGE = "ghcr.io/barbeauaxel/stayphora-pipeline:0.1.0"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
REMOTE_WORK = "/workspace/work"
# Fichier known_hosts jetable (les pods changent d'hôte à chaque fois). Pas "NUL" : le ssh de
# Git pour Windows le crée comme un vrai fichier dans le dossier courant.
KNOWN_HOSTS = (Path(tempfile.gettempdir()) / "stayphora_runpod_known_hosts").as_posix()


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


# Durées typiques (min) pour l'estimation du coût : démarrage + image (~10) + calcul.
TYPICAL_MINUTES = {"test-dataset": 55, "video": 75}
# Image construite avec CUDA 12.8 : tout pilote >= 12.8 convient (rétrocompatibilité).
DEFAULT_CUDA_VERSIONS = ["12.8", "12.9", "13.0", "13.1", "13.2", "13.3", "13.4"]


def sanitized_pod(pod: dict) -> str:
    """JSON du pod sans les valeurs des variables d'environnement (clés seulement)."""
    import copy
    import json

    p = copy.deepcopy(pod)

    def scrub(o):
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k.lower() in ("env", "environment") and isinstance(v, dict):
                    o[k] = {name: "***" for name in v}
                elif k.lower() in ("env", "environment") and isinstance(v, list):
                    o[k] = [str(x).split("=", 1)[0] + "=***" if isinstance(x, str) else "***" for x in v]
                else:
                    scrub(v)
        elif isinstance(o, list):
            for x in o:
                scrub(x)

    scrub(p)
    return redact(json.dumps(p, indent=1, ensure_ascii=False))


def say(msg: str) -> None:
    print(redact(f"[runpod] {msg}"), flush=True)


# ── RunPod (SDK Python officiel « runpod ») ─────────────────────────────────

class RunPod:
    def __init__(self, api_key: str):
        try:
            import runpod
        except ImportError:
            sys.exit("Le SDK RunPod est requis : pip install runpod")
        runpod.api_key = api_key
        self.sdk = runpod

    def list_gpus(self) -> list[dict]:
        return self.sdk.get_gpus()

    def gpu_info(self, gpu_id: str, cloud: str) -> dict:
        """Prix horaire, disponibilité et versions CUDA (>= 12.8) pour CE cloud.

        Catalogue REST v2 : GET /v2/catalog/gpus/{id}?include=AVAILABILITY&product=POD&cloud=..
        &minCudaVersion=12.8. Attention : sans `cloud`, l'API renvoie la disponibilité SECURE
        (c'est ce que faisait runpod.get_gpu, d'où des « LOW » trompeurs en Community).
        """
        from urllib.parse import quote

        from runpod.api.ctl_commands import run_rest_request

        g = run_rest_request("GET", f"/v2/catalog/gpus/{quote(gpu_id)}?include=AVAILABILITY"
                                    f"&product=POD&cloud={cloud}&minCudaVersion=12.8")
        price = (g.get("price") or {}).get(cloud.lower())
        offered = [v["version"] for v in g.get("cudaVersions") or []
                   if _version_tuple(v.get("version", "0")) >= (12, 8)]
        free = [v for v in offered if any(c.get("version") == v and c.get("available")
                                          for c in g.get("cudaVersions") or [])]
        return {"price": float(price) if price else None,
                "availability": str(g.get("availability") or "NONE").upper(),
                "cuda_versions": offered,
                "cuda_available": free}

    def create(self, **kw) -> str:
        # SDK 2.x (API REST v2) : support_public_ip n'est plus accepté (ValueError) ; demander
        # le port 22/tcp suffit. start_ssh ne concerne que les clés SSH du compte RunPod :
        # l'image démarre son propre sshd avec PUBLIC_KEY (docker/entrypoint.sh).
        args = dict(
            name=kw["name"], image_name=kw["image"], gpu_type_id=kw["gpu"],
            cloud_type=kw["cloud"], gpu_count=1, volume_in_gb=0,
            container_disk_in_gb=kw["disk"], ports="22/tcp", env=kw["env"],
            start_ssh=False, allowed_cuda_versions=kw["cuda_versions"],
        )
        pod = self.sdk.create_pod(**args)
        return pod["id"]

    def get(self, pod_id: str) -> dict | None:
        return self.sdk.get_pod(pod_id)

    def terminate(self, pod_id: str) -> None:
        self.sdk.terminate_pod(pod_id)

    def status(self, pod: dict) -> str:
        return str(pod.get("status") or pod.get("desiredStatus") or "").upper()

    def ssh_endpoint(self, pod_id: str) -> tuple[str, int] | None:
        """IP publique et port mappé sur le 22 du conteneur (API v2, sinon ancien format)."""
        pod = self.get(pod_id) or {}
        status = self.status(pod)
        if status in ("EXITED", "ERROR", "TERMINATED"):
            raise RuntimeError(f"le pod est passé à l'état {status} avant d'être joignable")
        for p in ((pod.get("runtime") or {}).get("ports") or []):
            # v2 : {private, public, type, ip}
            if p.get("private") == 22 and p.get("public") and p.get("ip") and str(p.get("type")).lower() == "tcp":
                return p["ip"], int(p["public"])
            # v1 / GraphQL : {privatePort, publicPort, isIpPublic, ip, type}
            if p.get("privatePort") == 22 and p.get("isIpPublic") and str(p.get("type")).lower() == "tcp":
                return p["ip"], int(p["publicPort"])
        direct = (pod.get("ssh") or {}).get("direct")
        if direct and direct.get("host") and direct.get("port"):
            return direct["host"], int(direct["port"])
        return None


# ── SSH ─────────────────────────────────────────────────────────────────────

class Ssh:
    def __init__(self, host: str, port: int, key: Path):
        self.host, self.port, self.key = host, port, key
        self.opts = ["-i", str(key), "-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                     "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
                     "-o", "LogLevel=ERROR"]

    # Une session SSH n'hérite pas de l'ENV du conteneur (PATH sans /opt/venv/bin, etc.) :
    # docker/entrypoint.sh l'écrit dans /etc/stayphora-env.sh, rechargé avant chaque commande.
    ENV_PREFIX = "[ -f /etc/stayphora-env.sh ] && . /etc/stayphora-env.sh; "

    def run(self, command: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        res = subprocess.run(["ssh", *self.opts, "-p", str(self.port), f"root@{self.host}",
                              self.ENV_PREFIX + command],
                             capture_output=capture, text=capture, encoding="utf-8" if capture else None,
                             errors="replace" if capture else None)
        if check and res.returncode != 0:
            raise RuntimeError(f"commande distante en échec ({res.returncode}) : {command[:120]}")
        return res

    def put(self, local: Path, remote: str) -> None:
        subprocess.run(["scp", *self.opts, "-P", str(self.port), str(local), f"root@{self.host}:{remote}"],
                       check=True)

    def get(self, remote: str, local: Path) -> bool:
        local.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["scp", *self.opts, "-P", str(self.port), "-r", f"root@{self.host}:{remote}",
                            str(local)])
        return r.returncode == 0

    def follow_log(self, path: str, exit_file: str, poll_s: float = 5.0) -> int:
        """Affiche le journal au fur et à mesure et renvoie le code du job dès que `exit_file` existe.

        Lecture par interrogations successives (tail -c +offset), PAS par `tail -f --pid` : dans
        le conteneur, le PID 1 (sleep infinity) ne récolte pas les processus terminés ; le job
        resterait « vivant » (zombie) et le suivi ne finirait jamais. Une coupure SSH ne fait
        que retarder l'interrogation suivante.
        """
        offset, failures = 0, 0
        while True:
            cmd = (f"tail -c +{offset + 1} {path} 2>/dev/null; "
                   f"[ -f {exit_file} ] && cat {exit_file} >&2; true")
            res = subprocess.run(["ssh", *self.opts, "-p", str(self.port), f"root@{self.host}", cmd],
                                 capture_output=True)
            if res.returncode != 0:
                failures += 1
                if failures > 30:
                    raise RuntimeError("connexion SSH perdue trop souvent")
                say("connexion interrompue, nouvel essai dans 15 s…")
                time.sleep(15)
                continue
            failures = 0
            if res.stdout:
                sys.stdout.write(redact(res.stdout.decode("utf-8", "replace")))
                sys.stdout.flush()
                offset += len(res.stdout)
            status = res.stderr.decode("utf-8", "replace").strip()
            if status.isdigit():
                # Dernières lignes écrites juste avant job.exit.
                tail = subprocess.run(["ssh", *self.opts, "-p", str(self.port), f"root@{self.host}",
                                       f"tail -c +{offset + 1} {path} 2>/dev/null; true"], capture_output=True)
                if tail.stdout:
                    sys.stdout.write(redact(tail.stdout.decode("utf-8", "replace")))
                    sys.stdout.flush()
                return int(status)
            time.sleep(poll_s)


# ── Programme ───────────────────────────────────────────────────────────────

def _ports_summary(pod: dict) -> str:
    ports = (pod.get("runtime") or {}).get("ports") or []
    if not ports:
        return "aucun"
    return ", ".join(f"{p.get('private', p.get('privatePort'))}->{p.get('ip') or '?'}:"
                     f"{p.get('public', p.get('publicPort')) or '?'}/{p.get('type')}" for p in ports)


def wait_for_ssh(rp: "RunPod", pod_id: str, pull_timeout_s: int = 20 * 60, port_timeout_s: int = 180,
                 poll_s: int = 10, log_every_s: int = 30, clock=time.monotonic, sleep=time.sleep) -> tuple[str, int]:
    """Attend l'IP publique + port 22 du pod, en journalisant la progression toutes les 30 s.

    API v2 : `status` vaut RUNNING dès qu'une machine est attribuée, AVANT le téléchargement de
    l'image ; le conteneur ne tourne vraiment que lorsque `runtime` est renseigné. Donc :
      - jusqu'à pull_timeout_s pour que le conteneur démarre (téléchargement de l'image) ;
      - puis port_timeout_s pour que le port 22 public apparaisse.
    En cas d'échec, affiche le get_pod complet (variables d'env masquées) avant de lever l'erreur.
    """
    t0 = clock()
    started_at = None
    last_log = -log_every_s
    pod: dict = {}
    while True:
        endpoint = rp.ssh_endpoint(pod_id)  # lève si EXITED / ERROR / TERMINATED
        pod = rp.get(pod_id) or {}
        now = clock()
        runtime = pod.get("runtime")
        if runtime and started_at is None:
            started_at = now
            say(f"[{int(now - t0)} s] conteneur démarré (image téléchargée)")
        if endpoint:
            say(f"[{int(now - t0)} s] SSH disponible : {endpoint[0]}:{endpoint[1]}")
            return endpoint
        if now - last_log >= log_every_s:
            last_log = now
            phase = "conteneur démarré, attente du port SSH" if runtime else "téléchargement de l'image / démarrage"
            uptime = (runtime or {}).get("uptime", (runtime or {}).get("uptimeInSeconds"))
            direct = (pod.get("ssh") or {}).get("direct")
            say(f"[{int(now - t0)} s] statut {rp.status(pod) or '?'} ; runtime "
                f"{'renseigné' if runtime else 'null'}{f' (uptime {uptime} s)' if uptime is not None else ''} ; "
                f"ports {_ports_summary(pod)} ; ssh.direct {'oui' if direct else 'non'} -> {phase}")
        if started_at is None and now - t0 > pull_timeout_s:
            reason = f"le conteneur n'a pas démarré en {pull_timeout_s // 60} min (téléchargement de l'image ?)"
        elif started_at is not None and now - started_at > port_timeout_s:
            reason = (f"conteneur démarré depuis {port_timeout_s // 60} min mais aucun port 22 public "
                      "(pas d'IP publique sur cette machine ?)")
        else:
            sleep(poll_s)
            continue
        say(f"{reason}. get_pod complet (variables d'env masquées) :")
        print(sanitized_pod(pod), flush=True)
        raise RuntimeError(reason)


def local_code_origin() -> dict:
    """Commit git et état du dépôt local, plus l'empreinte du dossier pipeline/ envoyé."""
    from pipeline.provenance import fingerprint

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)

    origin = {"source": "envoi-local", "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "git_commit": None, "git_dirty": None, "sha256_sent": fingerprint(ROOT / "pipeline")["sha256"]}
    try:
        head = git("rev-parse", "HEAD")
    except FileNotFoundError:
        origin["git_note"] = "git non installé"
        return origin
    if head.returncode != 0:
        origin["git_note"] = "pas un dépôt git (ou aucun commit)"
        return origin
    origin["git_commit"] = head.stdout.strip()
    status = git("status", "--porcelain", "--", "pipeline")
    origin["git_dirty"] = bool(status.stdout.strip())
    return origin


def milestone(label: str) -> None:
    """Jalon horodaté (heure locale), pour suivre en parallèle dans la console RunPod."""
    say(f"===== {time.strftime('%H:%M:%S')} — {label}")


def remote_session(endpoint: tuple[str, int], key: Path, a: argparse.Namespace, cfg: dict, slug: str,
                   skip: set[int], is_url: bool, video: Path | None) -> int:
    """Tout ce qui se passe une fois SSH disponible. Identique sur RunPod et avec --local-ssh.

    Les scripts de scripts/ sont ENVOYÉS au pod (et non pris dans /app) : une correction ne
    demande pas de reconstruire l'image.
    """
    import tempfile

    import yaml

    ssh = Ssh(endpoint[0], endpoint[1], key)
    for _ in range(30):
        if ssh.run("true", check=False, capture=True).returncode == 0:
            break
        time.sleep(10)
    else:
        raise RuntimeError("SSH injoignable")
    say(f"SSH ok ({endpoint[0]}:{endpoint[1]})")

    local = Path(cfg["work_dir"] or default_work_dir()) / slug / "remote"
    remote_root = f"{REMOTE_WORK}/{slug}"
    ssh.run("mkdir -p /workspace/input /workspace/scripts")
    for script in ("gpu_smoke_test.sh", "prepare_test_dataset.sh"):
        ssh.put(ROOT / "scripts" / script, f"/workspace/scripts/{script}")
    ssh.run("chmod +x /workspace/scripts/*.sh && sed -i 's/\\r$//' /workspace/scripts/*.sh")

    def fetch_logs() -> None:
        say(f"rapatriement des journaux vers {local}")
        for remote in (f"{remote_root}/logs", f"{remote_root}/state.json", "/workspace/smoke.log",
                       "/workspace/job.log"):
            ssh.get(remote, local)

    # ── Smoke test GPU (avant tout calcul long) ──────────────────────────────
    if not a.no_smoke:
        milestone("smoke test GPU (nvidia-smi, PyTorch, gsplat, COLMAP CUDA)")
        res = ssh.run("bash /workspace/scripts/gpu_smoke_test.sh 2>&1 | tee /workspace/smoke.log; "
                      "exit ${PIPESTATUS[0]}", check=False)
        if res.returncode != 0:
            fetch_logs()
            raise RuntimeError("smoke test GPU en échec : arrêt immédiat (voir smoke.log)")

    # ── Code du pipeline : version LOCALE (phase de test) ────────────────────
    # run_job.sh lance « python -m pipeline » depuis /app : remplacer /app/pipeline suffit, sans
    # reconstruire ni repousser l'image (dont la couche venv pèse ~7 Go).
    sent_sha = None
    if not a.image_code:
        import json
        import tarfile

        from pipeline.provenance import ORIGIN_FILE, included

        origin = local_code_origin()
        sent_sha = origin["sha256_sent"]
        say(f"code local : sha256 {sent_sha[:12]}, commit {origin['git_commit'] or '—'} "
            f"({origin.get('git_note') or ('modifs non commitées' if origin['git_dirty'] else 'propre')})")
        with tempfile.TemporaryDirectory() as td:
            tgz = Path(td) / "pipeline.tgz"
            (Path(td) / ORIGIN_FILE).write_text(json.dumps(origin, indent=2), encoding="utf-8")

            def keep(ti):
                rel = ti.name[len("pipeline/"):] if ti.name.startswith("pipeline/") else ""
                return ti if not rel or included(rel) else None

            with tarfile.open(tgz, "w:gz") as tar:
                tar.add(ROOT / "pipeline", arcname="pipeline", filter=keep)
                tar.add(Path(td) / ORIGIN_FILE, arcname=f"pipeline/{ORIGIN_FILE}")
            ssh.put(tgz, "/workspace/pipeline.tgz")
        ssh.run("rm -rf /app/pipeline && tar -xzf /workspace/pipeline.tgz -C /app "
                "&& cd /app && python -c 'import pipeline.alignment as a; assert hasattr(a, \"floor_plane_check\")'")
        say("code du pipeline : version locale envoyée dans /app/pipeline")

    # ── Configuration + données ──────────────────────────────────────────────
    remote_cfg = "/workspace/config.yaml"
    # La section runpod ne concerne que ce script : on l'enlève (les images <= 0.1.0
    # refusent les clés inconnues).
    pod_cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    pod_cfg.pop("runpod", None)
    with tempfile.TemporaryDirectory() as td:
        tmp_cfg = Path(td) / "config.yaml"
        tmp_cfg.write_text(yaml.safe_dump(pod_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        ssh.put(tmp_cfg, remote_cfg)
    pipeline_args = ["--slug", slug, "--config", remote_cfg, "--work-dir", REMOTE_WORK]
    if a.test_dataset:
        milestone("préparation du dataset playroom")
        ssh.run(f"bash /workspace/scripts/prepare_test_dataset.sh {shlex.quote(slug)}")
        pipeline_args += ["--from-step", "3"]
    elif is_url:
        pipeline_args += ["--input", a.input]
    else:
        milestone(f"envoi de la vidéo ({video.stat().st_size / 1e6:.0f} Mo)")
        remote_video = f"/workspace/input/video{video.suffix.lower()}"
        ssh.put(video, remote_video)
        pipeline_args += ["--input", remote_video]
    if skip:
        pipeline_args += ["--skip", ",".join(map(str, sorted(skip)))]
    pipeline_args += shlex.split(a.pipeline_args)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    milestone("pipeline lancé (les étapes s'affichent ci-dessous)")
    ssh.run("/app/docker/run_job.sh " + " ".join(shlex.quote(x) for x in pipeline_args))
    code = ssh.follow_log("/workspace/job.log", "/workspace/job.exit")
    say(f"pipeline terminé avec le code {code}")

    milestone("rapatriement")
    fetch_logs()
    ssh.get(f"{remote_root}/4_train/guard.json", local)
    # Contrôle visuel : JSON + PNG (photo | rendu, planche 4 vues) ; pas les rendus bruts (renders/).
    for d in ("4_train/check", "5_export/check"):
        ssh.get(f"{remote_root}/{d}/*.json", local / "check")
        ssh.get(f"{remote_root}/{d}/*.png", local / "check")
    if code != 0:
        # Arrêt (garde-fou ou erreur) : .ply partiels des points de contrôle, pour analyse.
        say("rapatriement des .ply partiels de l'entraînement (s'il y en a)")
        ssh.get(f"{remote_root}/4_train/results/ply", local)
    ssh.get(f"{remote_root}/tour-entry.ts", local)
    ssh.get(f"{remote_root}/3_sfm/alignment.json", local)
    if sent_sha:
        import json
        try:
            ran = json.loads((local / "state.json").read_text(encoding="utf-8")).get("code", {})
            ok = ran.get("sha256") == sent_sha
            say(f"traçabilité : code exécuté sha256 {str(ran.get('sha256'))[:12]} "
                f"{'= code envoyé' if ok else '≠ CODE ENVOYÉ (' + sent_sha[:12] + ')'}")
        except (OSError, ValueError) as exc:
            say(f"traçabilité : state.json illisible ({exc})")
    if a.download_results:
        ssh.get(f"{remote_root}/5_export/*.spz", local)
        ssh.get(f"{remote_root}/5_export/cleaned.ply", local)
    return code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", help="vidéo locale (envoyée par scp) ou URL https:// (téléchargée par le pod)")
    src.add_argument("--test-dataset", action="store_true",
                     help="dataset public déjà photographié (Deep Blending « playroom ») : étapes 3 à 7, sans R2")
    p.add_argument("--slug")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--gpu", default=None,
                   help="force UN GPU (sinon : liste ordonnée runpod.gpus de config.yaml)")
    p.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default=None,
                   help="cloud du GPU forcé par --gpu (défaut COMMUNITY)")
    p.add_argument("--max-price", type=float, default=None, help="prix horaire max (défaut : config)")
    p.add_argument("--confirm-gpu", default=None,
                   help="autorise CE GPU marqué « confirm: true » dans la liste (ex. Blackwell)")
    p.add_argument("--image", default=None, help="image Docker (défaut : config)")
    p.add_argument("--disk", type=int, default=None, help="disque du conteneur en Go (défaut : config)")
    p.add_argument("--max-hours", type=float, default=None, help="durée de vie maximale du pod (défaut : config)")
    p.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "stayphora_runpod"))
    p.add_argument("--skip", default="", help="étapes à ignorer, ex. 6,8 (passé au pipeline)")
    p.add_argument("--pipeline-args", default="", help="arguments supplémentaires pour le pipeline")
    p.add_argument("--download-results", action="store_true", help="rapatrie aussi les .spz et cleaned.ply")
    p.add_argument("--pass-terminate-key", action="store_true",
                   help="transmet RUNPOD_API_KEY au pod pour l'auto-suppression (si la clé injectée ne suffit pas)")
    p.add_argument("--keep-pod", action="store_true", help="DÉBOGAGE : ne pas supprimer le pod (facturé !)")
    p.add_argument("--yes", action="store_true", help="ne pas demander de confirmation")
    p.add_argument("--plan", action="store_true", help="affiche GPU, prix et timeout puis quitte (aucun pod)")
    p.add_argument("--list-gpus", action="store_true", help="affiche les GPU disponibles et quitte")
    p.add_argument("--no-smoke", action="store_true", help="ne pas exécuter le smoke test GPU")
    p.add_argument("--image-code", action="store_true",
                   help="utiliser le code du pipeline contenu dans l'image (défaut : envoyer la version locale)")
    p.add_argument("--local-ssh", metavar="HOTE:PORT",
                   help="TEST : session SSH vers un conteneur déjà lancé (aucun appel RunPod, aucun pod)")
    return p.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    rp = None
    if not a.local_ssh:
        if missing(("RUNPOD_API_KEY",)):
            sys.exit("RUNPOD_API_KEY manquante (.env ou variable d'environnement)")
        rp = RunPod(os.environ["RUNPOD_API_KEY"])

    if a.list_gpus:
        for g in sorted(rp.list_gpus(), key=lambda g: g.get("id", "")):
            print(f"{g.get('id'):<40} {g.get('displayName', ''):<20} {g.get('memoryInGb', '?')} Go")
        return 0

    if not a.slug or not (a.input or a.test_dataset):
        sys.exit("--slug et (--input ou --test-dataset) sont obligatoires")
    try:
        slug = validate_slug(a.slug)
        cfg = load_config(a.config)
    except ConfigError as exc:
        sys.exit(str(exc))
    key = Path(a.ssh_key).expanduser()
    pub = key.with_suffix(key.suffix + ".pub")
    if not key.is_file() or not pub.is_file():
        sys.exit(f"clé SSH introuvable : {key} (+ .pub). Créez-la : ssh-keygen -t ed25519")

    is_url = bool(a.input) and a.input.startswith("https://")
    video = None if (a.test_dataset or is_url) else Path(a.input)
    if video is not None and not video.is_file():
        sys.exit(f"vidéo introuvable : {video}")

    rc = cfg["runpod"]
    a.image = a.image or rc["image"]
    a.disk = a.disk or int(rc["disk_gb"])
    a.max_hours = a.max_hours or float(rc["max_hours"])
    max_price = a.max_price or float(rc["max_price_per_hour"])
    candidates = ([{"id": a.gpu, "cloud": a.cloud or "COMMUNITY", "confirm": False}] if a.gpu
                  else [{"id": g["id"], "cloud": g.get("cloud", "COMMUNITY"), "confirm": bool(g.get("confirm"))}
                        for g in rc["gpus"]])

    skip = {int(x) for x in a.skip.split(",") if x.strip()}
    if a.test_dataset:
        skip |= {6}  # jamais de R2 pour le dataset de test
    env = {"PUBLIC_KEY": pub.read_text().strip(), "MAX_POD_HOURS": str(a.max_hours),
           "AUTO_TERMINATE_AFTER_MIN": "20"}
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        if 6 not in skip and os.environ.get(name):
            env[name] = os.environ[name]
    if 8 not in skip and os.environ.get("RESEND_API_KEY"):
        env["RESEND_API_KEY"] = os.environ["RESEND_API_KEY"]
    if a.pass_terminate_key:
        env["STAYPHORA_TERMINATE_KEY"] = os.environ["RUNPOD_API_KEY"]
    if 6 not in skip and missing(("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")):
        sys.exit("secrets R2 manquants : ajoutez-les à .env ou passez --skip 6")

    if a.local_ssh:
        host, _, port = a.local_ssh.rpartition(":")
        milestone(f"TEST LOCAL via SSH {host}:{port} (aucun pod RunPod)")
        try:
            return remote_session((host, int(port)), key, a, cfg, slug, skip, is_url, video)
        except Exception as exc:  # noqa: BLE001
            say(f"ERREUR : {exc}")
            return 1

    # ── Choix du GPU : premier de la liste disponible, CUDA >= 12.8 libre, sous le prix max ──
    typical = TYPICAL_MINUTES["test-dataset" if a.test_dataset else "video"]
    say(f"GPU candidats (ordre de préférence), prix max {max_price:.2f} $/h :")
    eligible = []
    for c in candidates:
        try:
            info = rp.gpu_info(c["id"], c["cloud"])
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"lecture du catalogue GPU impossible pour {c['id']} ({type(exc).__name__}) : "
                     f"{redact(str(exc))[:300]}\nVérifiez les permissions de la clé RunPod (Catalog en lecture).")
        price = info["price"]
        reasons = []
        if info["availability"] == "NONE" or not info["cuda_available"]:
            reasons.append("indisponible")
        if price is None or price > max_price:
            reasons.append(f"prix {price} > {max_price}" if price else "prix inconnu")
        status = "OK" if not reasons else ", ".join(reasons)
        if not reasons and c["confirm"] and a.confirm_gpu != c["id"]:
            status = "OK mais CONFIRMATION REQUISE (--confirm-gpu)"
        say(f"  - {c['id']} ({c['cloud']}) : dispo {info['availability']}, {price} $/h, "
            f"CUDA libres {info['cuda_available'] or '-'} -> {status}")
        if not reasons:
            eligible.append({**c, **info})
    if not eligible:
        say("Aucun GPU de la liste n'est disponible sous le prix max : aucun pod créé.")
        return 3
    # Un GPU marqué confirm: true n'est utilisable que s'il est confirmé explicitement ;
    # on ne descend jamais la liste au-delà de lui.
    usable = []
    for e in eligible:
        if e["confirm"] and a.confirm_gpu != e["id"]:
            if not usable:
                say(f"Premier GPU disponible : {e['id']} ({e['cloud']}) à {e['price']:.2f} $/h — "
                    f"CONFIRMATION REQUISE : relancer avec --confirm-gpu \"{e['id']}\". Aucun pod créé.")
                return 4
            break
        usable.append(e)
    eligible = usable
    chosen = eligible[0]
    price = chosen["price"]
    say(f"GPU retenu : {chosen['id']} ({chosen['cloud']}) à {price:.2f} $/h"
        + (f" ; repli si la création échoue : {', '.join(e['id'] + ' (' + e['cloud'] + ')' for e in eligible[1:])}"
           if len(eligible) > 1 else ""))
    say(f"Image {a.image}, disque {a.disk} Go ; coût estimé ~{price * typical / 60:.2f} $ (~{typical} min) ; "
        f"plafond {price * a.max_hours:.2f} $ (timeout {a.max_hours} h)")
    say(f"Visite « {slug} » ; étapes ignorées : {sorted(skip) or 'aucune'} ; "
        f"secrets transmis au pod : {sorted(k for k in env if k not in ('PUBLIC_KEY', 'MAX_POD_HOURS', 'AUTO_TERMINATE_AFTER_MIN'))}")
    if a.plan:
        say("--plan : aucun pod créé")
        return 0
    if not a.yes and input("Créer le pod (facturé à l'heure) ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
        say("annulé")
        return 1

    pod_id: str | None = None
    terminated = False

    def terminate(*_):
        nonlocal terminated
        if pod_id is None or terminated or a.keep_pod:
            return
        terminated = True
        for attempt in range(1, 6):
            try:
                rp.terminate(pod_id)
                say(f"pod {pod_id} supprimé")
                return
            except Exception as exc:  # noqa: BLE001
                say(f"échec de suppression ({attempt}/5) : {exc}")
                time.sleep(10)
        say(f"!!! LE POD {pod_id} N'A PAS PU ÊTRE SUPPRIMÉ : supprimez-le dans https://console.runpod.io/pods !!!")

    def interrupt(*_):
        raise KeyboardInterrupt

    atexit.register(terminate)
    # SIGBREAK : fermeture de la fenêtre / Ctrl+Pause sous Windows ; SIGHUP : terminal fermé (Linux).
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), interrupt)

    t0 = time.monotonic()
    code = 1
    try:
        # Essaie les GPU éligibles dans l'ordre ; jamais au-delà de la liste.
        for e in eligible:
            try:
                pod_id = rp.create(name=f"stayphora-{slug}", image=a.image, gpu=e["id"], cloud=e["cloud"],
                                   disk=a.disk, env=env, cuda_versions=e["cuda_versions"] or DEFAULT_CUDA_VERSIONS)
                chosen = e
                break
            except Exception as exc:  # noqa: BLE001
                if "no longer any instances available" not in str(exc).lower():
                    raise
                say(f"{e['id']} ({e['cloud']}) : plus de machine libre à la création")
        if pod_id is None:
            raise RuntimeError("aucun GPU de la liste n'a pu être obtenu : aucun pod créé")
        price = chosen["price"]
        say(f"pod créé : {pod_id} sur {chosen['id']} ({chosen['cloud']}, {price:.2f} $/h) "
            "(téléchargement de l'image, quelques minutes)")
        endpoint = wait_for_ssh(rp, pod_id)
        code = remote_session(endpoint, key, a, cfg, slug, skip, is_url, video)
    except KeyboardInterrupt:
        say("interruption : suppression du pod")
        code = 130
    except Exception as exc:  # noqa: BLE001
        say(f"ERREUR : {exc}")
        code = 1
    finally:
        terminate()
        minutes = (time.monotonic() - t0) / 60
        if pod_id is not None:
            say(f"durée totale du pod : {minutes:.0f} min ; coût ≈ {price * minutes / 60:.2f} $ "
                f"({chosen['id']}, {price:.2f} $/h ; montant exact dans la facturation RunPod)")
        else:
            say("aucun pod créé : rien n'est facturé")
    return code


if __name__ == "__main__":
    sys.exit(main())
