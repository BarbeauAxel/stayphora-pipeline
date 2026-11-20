# syntax=docker/dockerfile:1
#
# Image du pipeline stayphora : COLMAP 4.2.1 (CUDA) + gsplat 1.5.3 + splat-transform 3.12.0
# + sharp-frames 0.4.0 + ffmpeg. Construite sans GPU (les noyaux CUDA sont compilés pour
# les architectures listées), exécutée sur RunPod (RTX 4090 = sm_89).
#
#   docker build -t ghcr.io/barbeauaxel/stayphora-pipeline:0.1.0 .
#
# Choix COLMAP : compilé depuis le tag de release 4.2.1 en reprenant le Dockerfile officiel
# du projet (docker/Dockerfile @ 4.2.1). L'image Docker Hub colmap/colmap n'est publiée que
# depuis la branche main (tags datés, 4.3.0.dev) : pas une release, donc pas reproductible.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

# ─────────────────────────────────────────────────────────────────────────────
# 1. COLMAP (dépendances et options identiques au Dockerfile officiel 4.2.1)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS colmap-builder
ARG COLMAP_VERSION=4.2.1
# 86 = RTX 30xx/A40, 89 = RTX 40xx/L40S, 120 = RTX 50xx
ARG CUDA_ARCHITECTURES="86;89;120"
# Compilations parallèles : ~2 Go de RAM chacune (nvcc). 4 tient dans une VM Docker de 8 Go.
ARG BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libopenimageio-dev openimageio-tools libmetis-dev \
        libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
        qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 qt6-svg-dev \
        libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev libssl-dev \
        libmkl-full-dev \
    && rm -rf /var/lib/apt/lists/*
# Contournement officiel : la config CMake d'openimageio exige ce dossier.
RUN mkdir -p /usr/include/opencv4
# ccache dans un cache BuildKit persistant (--mount=type=cache) : si la compilation est
# interrompue (Ctrl+C, PC éteint), les fichiers déjà compilés sont réutilisés à la reprise.
# Installé dans une couche séparée pour ne pas invalider la couche apt ci-dessus.
# libomp-dev/libtbb-dev : « Recommends » de MKL écartés par --no-install-recommends ; sans eux
# FindBLAS (BLA_VENDOR=Intel10_64lp) échoue : « Did not find BLAS library (required for SuiteSparse) ».
RUN apt-get update && apt-get install -y --no-install-recommends ccache libomp-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CCACHE_DIR=/ccache \
    CCACHE_BASEDIR=/colmap \
    CCACHE_COMPILERCHECK=content \
    CCACHE_NOHASHDIR=true \
    CCACHE_MAXSIZE=5G
RUN git clone --depth 1 --branch ${COLMAP_VERSION} https://github.com/colmap/colmap.git /colmap
RUN --mount=type=cache,id=stayphora-ccache-colmap,target=/ccache \
    cmake -S /colmap -B /colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DBLA_VENDOR=Intel10_64lp \
        `# ONNX (descripteurs appris, inutilisés ici) : l'archive 1.30.0 de GitHub ne se télécharge plus` \
        -DONNX_ENABLED=OFF \
        -DGUI_ENABLED=OFF \
        -DCCACHE_ENABLED=OFF \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    && ninja -C /colmap/build -j ${BUILD_JOBS} install \
    && ccache --show-stats

# ─────────────────────────────────────────────────────────────────────────────
# 2. Python : PyTorch cu128 + gsplat compilé (pas de compilation JIT au premier lancement)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS python-builder
ARG GSPLAT_VERSION=v1.5.3
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0" \
    MAX_JOBS=${BUILD_JOBS} \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential ninja-build python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip setuptools wheel
ENV PATH=/opt/venv/bin:$PATH
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128
# ccache pour les extensions CUDA, dans un cache BuildKit persistant : une compilation
# interrompue reprend où elle en était. gcc/g++ passent par les liens /usr/lib/ccache placés en
# tête du PATH (surtout PAS CC="ccache gcc" : torch le transmet à nvcc -ccbin, qui échoue) ;
# nvcc par PYTORCH_NVCC, lu par torch.utils.cpp_extension.
RUN apt-get update && apt-get install -y --no-install-recommends ccache \
    && rm -rf /var/lib/apt/lists/*
ENV CCACHE_DIR=/ccache \
    CCACHE_BASEDIR=/opt/gsplat \
    CCACHE_COMPILERCHECK=content \
    CCACHE_NOHASHDIR=true \
    CCACHE_MAXSIZE=5G \
    PATH=/usr/lib/ccache:$PATH \
    PYTORCH_NVCC="ccache /usr/local/cuda/bin/nvcc"
# gsplat : dépôt complet (examples/simple_trainer.py est utilisé tel quel), sous-module glm inclus.
RUN git clone --recursive --depth 1 --branch ${GSPLAT_VERSION} \
        https://github.com/nerfstudio-project/gsplat.git /opt/gsplat
RUN --mount=type=cache,id=stayphora-ccache-gsplat,target=/ccache \
    pip install --no-build-isolation /opt/gsplat && ccache --show-stats
RUN --mount=type=cache,id=stayphora-ccache-gsplat,target=/ccache \
    pip install --no-build-isolation -r /opt/gsplat/examples/requirements.txt
# Le pipeline lui-même (+ SDK RunPod pour l'auto-arrêt du pod).
COPY pyproject.toml /app/pyproject.toml
COPY pipeline /app/pipeline
RUN pip install /app "runpod>=1.7"
# sharp-frames dans son propre venv : il tire opencv/numpy 2, incompatible avec
# examples/requirements.txt de gsplat (numpy<2).
RUN python3 -m venv /opt/sharp-frames && /opt/sharp-frames/bin/pip install sharp-frames==0.4.0

# ─────────────────────────────────────────────────────────────────────────────
# 3. Image finale (runtime CUDA : torch embarque ses propres bibliothèques CUDA)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
ARG NODE_VERSION=22.12.0
ARG SPLAT_TRANSFORM_VERSION=3.12.0
ENV DEBIAN_FRONTEND=noninteractive
# Bibliothèques d'exécution de COLMAP (liste du stage runtime officiel 4.2.1)
# + python, ffmpeg, ssh, outils.
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 libc6 libomp5 libopengl0 libmetis5 libceres4t64 \
        libopenimageio2.4t64 libgcc-s1 libgl1 libglew2.2 libgoogle-glog0v6t64 \
        libqt6core6 libqt6gui6 libqt6widgets6 libqt6openglwidgets6 libqt6svg6 \
        libcurl4 libssl3t64 libmkl-locale libmkl-intel-lp64 libmkl-intel-thread libmkl-core \
        python3 python3-venv ffmpeg openssh-server ca-certificates curl xz-utils \
        libgl1 libglib2.0-0 libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# Node 22 (splat-transform 3.x exige node >= 22 ; Ubuntu 24.04 fournit la 18).
RUN curl -fsSL https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz \
        | tar -xJ -C /usr/local --strip-components=1 \
    && npm install -g @playcanvas/splat-transform@${SPLAT_TRANSFORM_VERSION} \
    && npm cache clean --force

COPY --from=colmap-builder /colmap-install/ /usr/local/
COPY --from=python-builder /opt/venv /opt/venv
COPY --from=python-builder /opt/gsplat /opt/gsplat
COPY --from=python-builder /opt/sharp-frames /opt/sharp-frames
RUN ln -s /opt/sharp-frames/bin/sharp-frames /usr/local/bin/sharp-frames

COPY . /app
RUN chmod +x /app/docker/*.sh /app/scripts/*.sh
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STAYPHORA_WORK_DIR=/workspace/work \
    QT_QPA_PLATFORM=offscreen
WORKDIR /app

# Vérifications à la construction (sans GPU) : les binaires répondent.
RUN colmap -h > /dev/null \
    && splat-transform --version \
    && sharp-frames --help > /dev/null \
    && python -m pipeline run --slug build-check --input x.mp4 --dry-run > /dev/null

ENTRYPOINT ["/app/docker/entrypoint.sh"]
