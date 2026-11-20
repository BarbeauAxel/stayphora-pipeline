# stayphora-pipeline

Transforme la vidéo d'un logement (filmée au téléphone) en visite 3D Gaussian Splatting
(`.spz`) pour le site Stayphora, et la publie sur Cloudflare R2.

Le calcul tourne sur un GPU loué à l'heure chez RunPod (RTX 4090 par défaut). Le PC local
sert à développer, à tester en `--dry-run` et à lancer les jobs.

> Phase de **test** : la licence des outils n'a pas encore été examinée.

## Le pipeline

| # | Étape | Outil | Contrôle (`verify`) |
|---|-------|-------|---------------------|
| 1 | `fetch` : vidéo locale ou URL `https://` | requests | taille > 1 Mo, pas une page HTML |
| 2 | `frames` : ~400 images nettes | sharp-frames 0.4.0 (`best-n`, `--min-buffer 3`) | ≥ `min_frames` images |
| 3 | `colmap` : poses caméra + alignement | COLMAP 4.2.1 (global mapper, repli incrémental) | ≥ 90 % d'images alignées, sinon `mapper` ; < 70 % : arrêt |
| 4 | `train` : Gaussian Splatting | gsplat 1.5.3, MCMC, 30 000 itérations, ≤ 1 M gaussiennes | .ply valide, nb ≤ `cap_max` |
| 5 | `export` : nettoyage + `.spz` normal et mobile | splat-transform 3.12.0 | ≤ 1 M splats / 16 Mo et ≤ 500 k / 8 Mo |
| 6 | `publish` : envoi sur R2 | boto3 (API S3) | taille identique sur R2 + URL publique en HTTP 200 |
| 7 | `tour_entry` : génère `tour-entry.ts` | — | objet `Tour` complet, `bounds` cohérents |
| 8 | `notify` : e-mail récapitulatif | Resend (HTTP) | identifiant d'e-mail renvoyé |

Dossier de travail : `work_dir/<slug>/` (par défaut `C:/stayphora-work` sous Windows, hors
OneDrive, et `/workspace/work` sur le pod). Il contient :

```
<slug>/
├── state.json            étapes terminées, durées, métriques
├── logs/                 un journal horodaté par étape (sortie complète des outils)
├── 1_input/video.mp4
├── 2_frames/images/
├── 3_sfm/                database.db, database_global.db, global/, incremental/,
│                         model_txt/, alignment.json, undistorted/{images,sparse}
├── 4_train/splats.ply    (+ results/ de gsplat)
├── 5_export/             cleaned.ply, <slug>.spz, <slug>-mobile.spz
└── tour-entry.ts         à coller dans content/fr/tours.ts du site
```

En cas d'échec, l'erreur indique l'étape à reprendre : `--from-step N` repart de là, sans
refaire les étapes précédentes. Si une variable `RESEND_API_KEY` est définie, un e-mail
d'échec est aussi envoyé.

### Choses vérifiées en construisant le pipeline (à connaître)

- **Repère de la scène.** Le viewer du site (`components/tour/engine.ts`) charge le `.spz`
  sans aucune transformation. La scène doit donc être en mètres, avec Y vers le haut, le
  sol à y = 0 et un centrage en x/z, comme `appartement.spz`, qui a été mesuré. COLMAP
  produit une échelle et une orientation arbitraires, donc l'étape 3 calcule un alignement
  ([pipeline/alignment.py](pipeline/alignment.py)) :
  - la verticale vient de l'orientation des caméras : l'axe horizontal de chaque image est
    supposé horizontal (téléphone tenu droit), en paysage comme en portrait, même quand on
    filme le sol ;
  - le sol est placé à un centile bas des points 3D ;
  - l'échelle part du principe que le téléphone est tenu à `camera_height_m` (1,5 m) ;
  - la scène est tournée pour aligner les murs sur X et Z.

  **Images exploitables** : le global_mapper de COLMAP 4 garde « recalées » des images sans
  aucun point 3D et avec une pose absurde (13 sur playroom, jusqu'à 200 000 fois la distance
  médiane). Une image ne compte donc comme alignée que si elle observe au moins
  `colmap.min_image_points3d` (15) points 3D et reste à moins de
  `colmap.max_camera_distance_ratio` (10) fois la distance médiane des caméras. Ce compte sert
  aux seuils de 90 % et 70 %, à l'alignement et à l'entraînement. Les images écartées sont
  listées dans `state.json` → `images_rejected`.

  **Repère d'entraînement** : l'étape 3 réécrit le modèle non distordu directement dans le repère
  du site, multiplié par `train_scale` pour que la distance médiane des caméras vaille 1, comme
  la normalisation de gsplat. Les fichiers sont dans `3_sfm/train/`, en binaire classique, car
  le lecteur TXT du pycolmap de gsplat est cassé et COLMAP 4 lit les poses dans `frames.txt`.
  Deux raisons à ce choix :
  - l'échelle de scène de gsplat est la distance de la caméra **la plus éloignée** ;
  - son bruit MCMC croît avec le carré de l'échelle.

  Dans les unités brutes de COLMAP, avec des poses aberrantes, l'essai RunPod du 20/11/2026 a
  divergé : échelle de scène 840, perte remontée à 0,6, puis plantage `torch.multinomial` à
  l'itération 5 800. L'étape 5 n'applique plus que l'échelle `1/train_scale` (retour en
  mètres) et les corrections manuelles, avec les actions `-s/-r/-t` de splat-transform.

  **Contrôle du modèle binaire** : à chaque exécution, l'étape 3 fait relire le modèle
  d'entraînement par COLMAP (`model_analyzer`, puis `model_converter` vers TXT). Les caméras,
  les poses et les points relus doivent être identiques à ceux écrits ; sinon l'étape s'arrête.

  **Garde-fou d'entraînement** (`train.guard`, [pipeline/train_guard.py](pipeline/train_guard.py)) :
  - dès le démarrage, arrêt si l'échelle de scène de gsplat sort de [0,5 ; 10] ;
  - aux itérations 1 000 et 3 000, arrêt dans trois cas :
    - la perte médiane dépasse 1,3 fois son minimum, avec au moins +0,03 ;
    - plus de 50 % des gaussiennes sont mortes et relocalisées en un cycle ;
    - le nombre de gaussiennes tombe sous 50 % de son maximum.

  En cas d'arrêt, `scripts/runpod_job.py` rapatrie les journaux, `guard.json` et les `.ply`
  partiels (`--ply_steps 1000 3000 30000`), puis supprime le pod. Rejoué sur le journal de
  l'essai du 20/11/2026, le garde-fou l'aurait arrêté au démarrage (échelle 840). Sans ce
  critère, il l'aurait arrêté à l'itération 1 000 (71 % de gaussiennes mortes). Sans ces deux
  critères, il l'aurait arrêté à l'itération 3 000 (perte multipliée par 2,7).
  **Diagnostic** : les valeurs estimées sont dans `state.json` →
  `steps.3.metrics.alignment_estimated` et dans l'e-mail :
  - `camera_height_assumed_m` ;
  - `scale_m_per_unit` ;
  - `vertical_angle_deg` (verticale estimée par rapport à l'axe -Y de COLMAP) ;
  - `camera_roll_median_deg` et `camera_roll_p90_deg` (roulis des caméras : > 15°
    signale un téléphone penché ou une vidéo mal orientée), `camera_pitch_median_deg`,
    `rotated_images` (nombre d'images en paysage ou en portrait) ;
  - `floor_check` : **contrôle indépendant** de la verticale par le nuage de points. C'est
    le plus grand plan quasi horizontal parmi les points entre -0,3 et 0,5 m (RANSAC, puis
    réajustement). Son inclinaison `tilt_deg` vise **< 3°**, et un avertissement apparaît
    **au-delà de 5°**. Sur playroom : 0,43° ;
  - `up_consistency` ;
  - `yaw_deg`.

  **Corrections manuelles** : `alignment.override` dans `config.yaml` :
  - `scale_factor` multiplie l'échelle ;
  - `yaw_deg` ajoute une rotation autour de Y ;
  - `floor_offset_m` décale verticalement.

  Elles sont appliquées après l'estimation, à l'étape 5 : `--from-step 5` suffit, sans
  refaire COLMAP ni l'entraînement. Les valeurs appliquées sont notées dans
  `steps.5.metrics.alignment_applied`.
- **Opacité.** splat-transform v3 filtre l'opacité **linéaire** (0 à 1), même si le .ply la
  stocke en logit. `-V opacity,gt,0.05` est donc correct tel quel. Le convertir en logit
  garderait presque tout. Un test verrouille ce comportement.
- **Décimation.** `--decimate` doit être la dernière action et produire un `.ply`.
  L'export se fait donc en deux passes : `.ply` décimé, puis `.spz`.
- **Taille et harmoniques.** Les scènes actuelles du site sont en SPZ **sans harmoniques**
  (SH 0) : 1 M de splats y font environ 14 Mo. Avec SH 2, 1 M de splats dépasse très
  probablement 16 Mo. L'export réduit alors automatiquement SH 2 → 1 → 0, puis décime si
  c'est encore trop gros. Les choix faits sont notés dans les métriques et dans l'e-mail.
- **gsplat.** `simple_trainer.py` a `data_factor=4` et `normalize_world_space=True` par
  défaut. Le pipeline force `--data_factor 1` et `--no-normalize-world-space` : sans ça, le
  .ply serait exporté dans un repère normalisé inconnu. Avant de lancer l'entraînement,
  le pipeline vérifie que toutes les options figurent dans le `--help` du trainer.

## Contrôle visuel automatique

Activé par `quality.enabled` (`pipeline/quality.py`, `pipeline/render_check.py`). Le rendu
passe par `gsplat.rasterization`, comme à l'entraînement, et se fait sur le pod.

- **Vues de test** : `train.test_every: 8` met de côté 1 image sur 8, qui ne sert pas à
  l'entraînement (c'est la convention de gsplat et de 3DGS). Le trainer mesure PSNR, SSIM et
  LPIPS sur ces vues. Avant, `test_every: 1000` ne gardait qu'une seule vue de test, d'où le
  PSNR de 23 peu représentatif du job 8. Contrepartie : 12,5 % d'images en moins pour
  l'entraînement.
- **Fin de l'étape 4** : le `.ply` entraîné est rendu depuis toutes les vues de test et
  depuis 3 caméras d'entraînement réparties sur la trajectoire. Résultats dans
  `4_train/check/` :
  - `train.json` : métriques moyennes et par vue ;
  - `train_<image>.png` : photo et rendu côte à côte ;
  - `renders/train/` : rendus bruts, qui servent de référence et ne sont pas rapatriés.

  Un écart de plus de 0,5 dB avec le PSNR du trainer est signalé : lecture du `.ply` ou
  caméras à vérifier.
- **Fin de l'étape 5** : chaque `.spz` est relu (splat-transform `.spz` → `.ply`), puis
  ramené dans le repère d'entraînement (positions × k, log-échelles + ln k). Il est rendu
  depuis les mêmes caméras.
  - Le **PSNR de test** est comparé à celui du `.ply` entraîné. Si la baisse dépasse
    `quality.max_psnr_drop_db` (normal 1,5 dB, mobile 4 dB, valeurs **provisoires** à
    calibrer), l'étape échoue et rien n'est publié.
  - Le PSNR du rendu `.spz` contre le rendu entraîné (`psnr_vs_ref`) est aussi noté.
- **Planche 4 vues** : `5_export/check/normal_planche.png` montre 4 vues depuis le point de
  départ de la visite (même calcul que `tour-entry.ts`). Elle est rapatriée et jointe à
  l'e-mail.
- **Si l'outil de contrôle lui-même plante** (CUDA, lecture), l'erreur est notée dans les
  métriques et dans l'e-mail, mais n'arrête pas le job : on ne perd pas un entraînement
  payé. Seul le dépassement de seuil arrête le job.
- **Rapatriement** : les JSON et PNG des deux dossiers `check/` sont copiés dans
  `C:\stayphora-work\<slug>\remote\check\`.

**Diagnostic sans GPU** : `python scripts/inspect_splats.py fichier.ply fichier.spz --out diag`
donne, pour chaque fichier :
- les statistiques : tailles exp(log_scale), opacités, couleurs DC, gaussiennes hors de la pièce ;
- une planche de 4 vues, rendue sur processeur (`pipeline/splat_render.py`, couleur de base
  seulement, environ 10 s par vue).

Une scène d'intérieur vue **de l'extérieur** (vue par défaut de SuperSplat) montre une coque
de grosses gaussiennes floues derrière les murs. Ce n'est pas un défaut d'export : il faut
regarder depuis l'intérieur.

## Installation locale (Windows, pour développer)

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[frames,dev]"
npm install                      # splat-transform 3.12.0 (Node >= 22), pour les tests
winget install Gyan.FFmpeg       # requis par sharp-frames pour l'étape 2 en local
copy .env.example .env           # puis remplir
.venv\Scripts\python -m pytest
```

COLMAP et gsplat ne sont pas nécessaires en local : les étapes 3 et 4 tournent dans
l'image Docker, sur le GPU loué.

## Variables d'environnement

Les secrets sont lus **uniquement** dans l'environnement, ou dans un `.env` exclu de git.
Ils ne sont jamais affichés : toute valeur de secret est remplacée par `***` dans les
journaux et la console.

| Variable | Utilisée par |
|----------|--------------|
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | étape 6 (publication) |
| `RESEND_API_KEY` | étape 8 (e-mail) |
| `RUNPOD_API_KEY` | `scripts/runpod_job.py`, sur le PC local uniquement |
| `STAYPHORA_WORK_DIR` (facultatif) | dossier de travail |

### Créer les clés S3 de R2 (jeton limité au bucket)

1. Tableau de bord Cloudflare → **R2 Object Storage** → **Manage API tokens** (ou
   « Gérer les jetons d'API R2 ») → **Create API token**.
2. Nom : `stayphora-pipeline`. Permissions : **Object Read & Write**.
3. **Specify bucket(s)** → `stayphora-assets` uniquement. Durée de validité : selon votre
   préférence.
4. Valider. Copiez tout de suite l'**Access Key ID** et la **Secret Access Key** dans
   `R2_ACCESS_KEY_ID` et `R2_SECRET_ACCESS_KEY` : la clé secrète ne sera plus affichée.
5. `R2_ACCOUNT_ID` est l'identifiant de compte, visible sur la page R2 ou dans l'endpoint
   `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.

Le jeton ne peut lire et écrire **que** dans `stayphora-assets`. Il ne peut ni gérer les
buckets, ni toucher au reste du compte.

### Resend

Vérifiez le domaine `stayphora.fr` dans Resend, car l'expéditeur par défaut est
`pipeline@stayphora.fr` (`notify.from` dans `config.yaml`). Créez ensuite une clé
**Sending access** limitée à ce domaine.

## Lancer un job en local

```powershell
# Afficher toutes les commandes sans rien exécuter (aucun fichier créé) :
.venv\Scripts\python -m pipeline run --input C:\videos\studio.mp4 --slug studio-lyon --dry-run

# Étapes 1-2 en vrai sur le PC (vérifier la sélection d'images) :
.venv\Scripts\python -m pipeline run --input C:\videos\studio.mp4 --slug studio-lyon --to-step 2

# Reprendre après un échec, sans R2 ni e-mail :
python -m pipeline run --slug studio-lyon --from-step 4 --skip 6,8
```

Options : `--config`, `--from-step N`, `--to-step N`, `--skip 6,8`, `--work-dir`,
`--dry-run` et `-v` (sortie détaillée des outils dans la console).

**Slug** : minuscules, chiffres et tirets (`studio-lyon-3`). Le motif est compatible avec
`lib/tour-routing.ts` du site, en un peu plus strict : pas de tiret au début, à la fin ni
doublé. Les chemins réservés du site (`demo`, `images`, `videos`…) sont refusés.
**Ce slug devient une URL définitive** (`https://visite.stayphora.fr/<slug>`, intégrée en
iframe chez les clients) : choisissez-le bien.

## Image Docker

[Dockerfile](Dockerfile) : construction en trois étapes, sans GPU nécessaire.

- **COLMAP 4.2.1**, compilé depuis le tag de release avec CUDA (architectures 86, 89 et 120).
  Les dépendances et options sont reprises du Dockerfile officiel de ce tag.
  *Pourquoi pas l'image `colmap/colmap` ?* Elle n'est publiée que depuis la branche `main`
  (tags datés, 4.3.0.dev) : ce n'est pas une release, donc pas reproductible.
- **PyTorch 2.7.1 cu128** et **gsplat v1.5.3**, compilés pour les mêmes architectures (pas
  de compilation au premier lancement), avec `examples/requirements.txt`.
- **Node 22** et **splat-transform 3.12.0**, **sharp-frames 0.4.0** (dans un venv à part :
  il exige numpy 2, gsplat numpy < 2), **ffmpeg** et **sshd**.

```powershell
docker build -t ghcr.io/barbeauaxel/stayphora-pipeline:0.1.0 .   # ~1-2 h la 1re fois, ~40 Go de disque
# --build-arg BUILD_JOBS=8 accélère si la VM Docker a >= 16 Go de RAM (défaut 4 : VM de 8 Go)
echo $env:GHCR_TOKEN | docker login ghcr.io -u barbeauaxel --password-stdin   # PAT « write:packages »
docker push ghcr.io/barbeauaxel/stayphora-pipeline:0.1.0
```

Rendez ensuite le paquet **public** : GitHub → Packages → stayphora-pipeline → Package
settings → Change visibility. L'image ne contient que des outils open source et aucun
secret.

## Lancer un job sur RunPod

Prérequis :
- `pip install runpod` (SDK ≥ 2, qui utilise l'API REST v2) ;
- `RUNPOD_API_KEY` dans `.env`. Une clé restreinte suffit : Pods R/W, Catalog R, GraphQL R/W ;
- une clé SSH dédiée : `ssh-keygen -t ed25519 -N "" -f ~/.ssh/stayphora_runpod` (valeur par
  défaut de `--ssh-key`).

```powershell
python scripts\runpod_job.py --test-dataset --slug test-playroom --skip 8 --plan   # GPU retenu, prix, timeout — aucun pod
python scripts\runpod_job.py --test-dataset --slug test-playroom --skip 8 --download-results --yes
python scripts\runpod_job.py --input C:\videos\studio.mp4 --slug studio-lyon
python scripts\runpod_job.py --input C:\videos\studio.mp4 --slug studio-lyon --gpu "NVIDIA GeForce RTX 3090" --cloud COMMUNITY
```

Le script :
1. affiche le prix horaire et **demande confirmation** (sauf avec `--yes`) ;
2. crée le pod (Community Cloud) ;
3. envoie la vidéo par `scp` et lance le pipeline en arrière-plan sur le pod ;
4. affiche le journal en direct, et se reconnecte si la connexion tombe ;
5. rapatrie `logs/`, `state.json`, `tour-entry.ts` et `job.log` dans
   `C:\stayphora-work\<slug>\remote\` (ajoutez `--download-results` pour les `.spz`) ;
6. **supprime le pod**.

**Pas de pod oublié** — trois filets de sécurité :
1. La suppression se fait dans un `finally`, donc aussi en cas d'erreur, de Ctrl+C ou de
   fermeture de la fenêtre. Si elle échoue 5 fois, un message très visible donne
   l'identifiant du pod.
2. Sur le pod, un minuteur supprime le pod au bout de `--max-hours` heures (3 par défaut).
3. Sur le pod, le job se supprime lui-même 20 min après la fin.

Les filets 2 et 3 utilisent la clé « pod-scoped » que RunPod injecte dans chaque pod. Je
n'ai pas pu vérifier dans la doc qu'elle peut supprimer son propre pod. Si les journaux
du premier test montrent un échec, relancez avec `--pass-terminate-key`. Après les
premiers jobs, vérifiez quand même la console RunPod.

Le pod reçoit les secrets R2 et Resend en variables d'environnement, visibles dans la
console RunPod. Les secrets RunPod (`{{ RUNPOD_SECRET_nom }}`) sont une alternative à
étudier après la phase de test.

### Choix du GPU (`runpod` dans `config.yaml`)

`runpod.gpus` est une liste ordonnée de GPU acceptables. Le script retient le premier qui
remplit trois conditions :
- disponible **dans ce cloud** ;
- une version CUDA ≥ 12.8 est libre ;
- prix ≤ `max_price_per_hour`.

Si la création échoue faute de machine, il passe au suivant, sans jamais aller au-delà de la
liste.

Les GPU marqués `confirm: true` (Blackwell, sur lesquels l'image n'a jamais tourné) ne sont
jamais créés sans `--confirm-gpu "<id>"`. L'image n'est compilée que pour les architectures
86, 89 et 120 : **pas** d'A100, H100, H200 ni B200. La liste actuelle a été construite à
partir du catalogue : au moins 24 Go de VRAM, au plus 0,75 $/h, Secure Cloud d'abord.

⚠️ Pour le catalogue, la disponibilité sans cloud précisé correspond à **Secure**. Le script
interroge donc chaque cloud séparément.

### Ce qui se passe sur le pod

1. **Attente** : le journal affiche toutes les 30 s le statut du pod, `runtime` (null tant
   que l'image se télécharge, soit environ 4 min pour notre image), les ports et SSH.
   Attention : avec l'API v2, `RUNNING` arrive **avant** la fin du téléchargement. Le délai
   de 3 min pour obtenir un port 22 public ne part donc que lorsque le conteneur tourne
   vraiment.
2. **Smoke test GPU** (`scripts/gpu_smoke_test.sh`, moins de 2 min) :
   - `nvidia-smi` ;
   - PyTorch CUDA et nom du GPU ;
   - mini rendu gsplat ;
   - COLMAP « with CUDA » et extraction SIFT sur GPU ;
   - splat-transform et sharp-frames.

   Au premier échec : arrêt immédiat, rapatriement de `smoke.log`, suppression du pod.
3. **Code envoyé** : pendant la phase de test, le script envoie la version **locale** de
   `pipeline/` et des scripts de `scripts/` sur le pod. Il n'y a pas besoin de reconstruire
   ni de repousser l'image après une correction. `--image-code` utilise à la place le code
   de l'image.

   **Traçabilité** : `state.json` → `code` enregistre :
   - la source (`envoi-local` ou image) ;
   - le commit git et la présence de modifications non commitées (`git_dirty`) ;
   - le SHA-256 du dossier `pipeline/`, calculé sur le pod, et `matches_sent`, qui le compare
     à l'empreinte calculée en local avant l'envoi.

   Une fois le pipeline stable, on reconstruit l'image (0.2.0) et `--image-code` redevient
   le comportement par défaut.
4. **Pipeline**, puis **rapatriement** dans `C:\stayphora-work\<slug>\remote\`.

Les commandes SSH rechargent `/etc/stayphora-env.sh`, car une session SSH n'hérite pas de
l'environnement du conteneur (`PATH` sans `/opt/venv/bin`). Le suivi du journal interroge
`job.log` et `job.exit` sans s'appuyer sur le processus : le PID 1 du conteneur ne récolte
pas les processus terminés.

### Tester la chaîne SSH en local (gratuit, sans GPU)

```powershell
docker run -d --name stayphora-ssh-test -p 127.0.0.1:2222:22 -e PUBLIC_KEY="$(Get-Content ~/.ssh/stayphora_runpod.pub)" ghcr.io/barbeauaxel/stayphora-pipeline:0.1.0
python scripts\runpod_job.py --local-ssh 127.0.0.1:2222 --test-dataset --slug test-local --skip 8 --no-smoke "--pipeline-args=--dry-run"
docker rm -f stayphora-ssh-test
```

Ce test exerce le **même code** que sur RunPod : envoi des scripts et du code, préparation du
dataset, `run_job.sh`, suivi du journal et rapatriement. Pour tester COLMAP (étape 3) sur le
processeur, utilisez une config avec `colmap.use_gpu: false` et `--pipeline-args=--to-step 3`.
Sur playroom, cela prend environ 2 min, avec 217 images sur 225 alignées.

## Publier une visite : les 2 étapes manuelles restantes

Publier les `.spz` ne suffit pas : le site n'accepte que les slugs déclarés dans
`content/fr/tours.ts` (`cloudflare-worker.ts` → `lib/tour-routing.ts`).

1. **Déclarer la visite** : collez le contenu de `work/<slug>/tour-entry.ts` dans le tableau
   `tours` de `content/fr/tours.ts`, puis remplacez les `TODO` : titre, ville, type,
   description, `alt`. Vérifiez les points de vue avec `/demo/<slug>?debug` en local, et
   ajoutez des pièces à `rooms` si besoin. Ajoutez la **couverture**
   `public/images/tours/<slug>.jpg` et, si vous en avez une, la **vidéo** de survol
   `public/videos/tours/<slug>.webm/.mp4` (ligne `video` à décommenter).
2. **Déployer** : `npm run typecheck`, puis commit et push du repo stayphora (déploiement
   habituel). La visite devient alors accessible sur `https://visite.stayphora.fr/<slug>`.

## Conseils de tournage (ce qui fait la qualité)

- **Lent et régulier** : marchez à environ 0,5 m/s, tournez lentement. Le flou de bougé est
  la première cause d'échec.
- **Lumière** : allumez tout, ouvrez les volets, évitez les contre-jours. Filmez en journée.
- **Téléphone droit, tenu à environ 1,5 m du sol** (à hauteur de poitrine ou des yeux), à
  hauteur constante pendant tout le tournage, orientation verrouillée.
  **C'est ce qui donne l'échelle de la visite** : une vidéo ne contient aucune mesure, donc
  le pipeline suppose que le téléphone était à 1,5 m (`alignment.camera_height_m`). Tenu à
  1,2 m, la scène paraîtra environ 25 % trop grande. Corrigez alors `camera_height_m`, ou
  `alignment.override.scale_factor`. *À reprendre dans le guide de tournage du site :
  « Tenez le téléphone droit, à hauteur de poitrine (environ 1,5 m), et gardez cette
  hauteur pendant toute la vidéo. »*
- **Recouvrement** : chaque zone doit être vue sous plusieurs angles. Faites le tour de
  chaque pièce, puis traversez-la. Filmez les passages entre pièces (portes) lentement,
  dans les deux sens.
- **Éviter** les gros plans sur des murs blancs, les miroirs, les vitres et les écrans
  allumés, les personnes et animaux en mouvement.
- **Réglages** : 4K ou 1080p à 30 i/s, sans stabilisation agressive, sans zoom. Utilisez
  l'objectif principal (×1), pas le grand-angle.
- **Durée** : 2 à 4 min pour un T2. Avec `fps: 10`, cela donne 1 200 à 2 400 candidates
  pour 400 images retenues.

## Coût estimé par job

Ordre de grandeur, à confirmer au premier test : le script affiche le prix réel et la
durée du pod.

| Phase | RTX 4090 |
|-------|----------|
| Démarrage du pod + téléchargement de l'image (~12 Go) | 5-10 min |
| Images (sharp-frames) | 2-4 min |
| COLMAP (400 images, global mapper) | 5-15 min |
| gsplat (30 000 itérations, 1 M gaussiennes) | 25-40 min |
| Export, publication, e-mail | 2-5 min |
| **Total** | **~45-75 min** |

Avec une RTX 4090 en Community Cloud autour de 0,35 à 0,70 $/h selon la disponibilité,
un job coûte **environ 0,30 à 0,90 $**. Le stockage R2 est négligeable (~25 Mo par visite).

## Tests

```powershell
.venv\Scripts\python -m pytest
```

Les tests tournent sur le PC, sans GPU :
- configuration et slugs ;
- lecture des modèles COLMAP ;
- alignement sur une pièce synthétique ;
- **traduction `-r/-s/-t` contre le vrai splat-transform** ;
- filtre d'opacité et décimation réels ;
- `tour-entry.ts` ;
- reprise `--from-step` ;
- dry-run complet ;
- masquage des secrets.

L'entraînement se teste sur le pod avec `--test-dataset` : « playroom » de Deep Blending,
intérieur, 225 photos, étapes 3 à 7, sans R2.

## Points à valider au premier job GPU

- Le contrôle visuel sur GPU : `render_check` a été testé dans l'image avec un rendu CPU à la
  place de gsplat, mais pas encore avec gsplat. Il faut aussi calibrer
  `quality.max_psnr_drop_db` sur les baisses réellement mesurées.

- Les options COLMAP 4.2.1 (`FeatureExtraction.use_gpu`…) viennent du code source au tag
  4.2.1, mais elles n'ont pas encore été exécutées.
- Le SDK `runpod` : champs `runtime.ports` et `allowed_cuda_versions` (repli automatique si
  le paramètre n'existe pas).
- L'auto-suppression du pod avec la clé injectée (voir plus haut).
- La taille réelle du `.spz` à SH 2 pour 1 M de splats.
