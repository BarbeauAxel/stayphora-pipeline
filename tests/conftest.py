import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def splat_transform_bin() -> str | None:
    local = ROOT / "node_modules" / ".bin" / ("splat-transform.cmd" if platform.system() == "Windows"
                                              else "splat-transform")
    if local.exists():
        return str(local)
    return shutil.which("splat-transform")


@pytest.fixture
def splat_transform():
    exe = splat_transform_bin()
    if not exe:
        pytest.skip("splat-transform non installé (npm install)")
    return exe


GAUSS_COLS = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
              + [f"f_rest_{i}" for i in range(45)]
              + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def write_gaussian_ply(path: Path, xyz: np.ndarray, opacity_linear: np.ndarray | None = None) -> None:
    """PLY 3DGS standard (opacité stockée en logit, comme gsplat)."""
    n = len(xyz)
    d = np.zeros((n, len(GAUSS_COLS)), np.float32)
    d[:, :3] = xyz
    op = np.full(n, 0.5) if opacity_linear is None else opacity_linear
    d[:, GAUSS_COLS.index("opacity")] = np.log(op / (1 - op))
    s = GAUSS_COLS.index("scale_0")
    d[:, s:s + 3] = np.log(0.01)
    d[:, GAUSS_COLS.index("rot_0")] = 1
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
              + "".join(f"property float {c}\n" for c in GAUSS_COLS) + "end_header\n")
    path.write_bytes(header.encode() + d.tobytes())


def run_st(exe: str, *args) -> str:
    res = subprocess.run([exe, *map(str, args)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "RESEND_API_KEY", "RUNPOD_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    os.environ.pop("STAYPHORA_WORK_DIR", None)
