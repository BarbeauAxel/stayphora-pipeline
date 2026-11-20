# TODO

Noté le 20/11/2026, après le premier job playroom réussi (RTX 3090, 37 min, ≈ 0,31 $).
Rien de tout cela n'est fait.

## Seuil d'alignement à 80 %

- `colmap.min_ratio` passe de 0,70 à **0,80** dans `config.yaml` et `pipeline/config.py`.
- Le seuil porte sur les images *exploitables* : au moins 15 points 3D et une pose non aberrante.
- Playroom passerait de justesse : 82,2 % au job 8, 83,6 % en local.
- `global_min_ratio` reste à 0,90 : en dessous, on essaie aussi le mapper incrémental.
- Mettre à jour le README et les tests de seuil.

## Image Docker 0.2.0

- **Code actuel dans l'image :**
  - le garde-fou d'entraînement ;
  - le repère d'entraînement ;
  - les images exploitables ;
  - le contrôle du sol ;
  - la traçabilité ;
  - le fichier known_hosts temporaire.
- **Architecture 80 (A100) :** l'ajouter à `CMAKE_CUDA_ARCHITECTURES` (COLMAP) et à `TORCH_CUDA_ARCH_LIST` (gsplat), en plus de 86, 89 et 120. Ensuite, retirer l'exclusion de l'A100 dans la liste des GPU de `config.yaml`.
- **`--image-code` par défaut :** le code de l'image devient le comportement normal de `scripts/runpod_job.py`. Une option du type `--local-code` permettra encore d'envoyer le code local pour un test.
- `DEFAULT_IMAGE` et `runpod.image` passent en `0.2.0`.
- **Vérifications avant publication :**
  - smoke test GPU ;
  - étape 3 en local sur playroom ;
  - le sha256 du dossier `pipeline/` dans l'image correspond au commit.
- Publier sur ghcr.io : l'utilisateur fait lui-même `docker login`.

## Cache de l'image sur un volume réseau RunPod

- **Problème :** le téléchargement de l'image prend 4 à 9 min par pod. C'était environ un quart de la durée facturée du job 8.
- **Piste :** un *network volume* RunPod dans le datacenter le plus utilisé, avec l'image en tarball (`docker save`) ou les couches déjà extraites. Le volume peut aussi garder les datasets et des caches (pip, torch extensions).
- **À évaluer :**
  - le coût du stockage mensuel ;
  - un volume lie le pod à un seul datacenter, ce qui réduit le choix de GPU ;
  - le gain réel par rapport à une image plus légère (multi-stage, sans outils de build).
- Rien à créer sans accord explicite : le volume est facturé même sans pod actif.
