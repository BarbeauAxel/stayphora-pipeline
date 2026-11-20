"""Lecture minimale de fichiers PLY binaires (gaussiennes), sans dépendance externe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class PlyHeader:
    fmt: str
    elements: list[tuple[str, int, list[tuple[str, str]]]]  # (nom, nombre, [(prop, dtype)])
    header_len: int

    def count(self, element: str = "vertex") -> int:
        for name, n, _ in self.elements:
            if name == element:
                return n
        raise KeyError(element)

    def props(self, element: str = "vertex") -> list[str]:
        for name, _, props in self.elements:
            if name == element:
                return [p for p, _ in props]
        raise KeyError(element)


def read_header(path: Path) -> PlyHeader:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} n'est pas un fichier PLY")
        fmt, elements = "", []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: en-tête PLY incomplet")
            parts = line.decode("ascii", "replace").split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append((parts[1], int(parts[2]), []))
            elif parts[0] == "property":
                if parts[1] == "list":
                    raise ValueError("propriétés de type liste non gérées")
                elements[-1][2].append((parts[2], _TYPES[parts[1]]))
            elif parts[0] == "end_header":
                return PlyHeader(fmt, elements, f.tell())


def read_vertex_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    h = read_header(path)
    if h.fmt != "binary_little_endian":
        raise ValueError(f"format PLY non géré : {h.fmt}")
    offset = h.header_len
    for name, n, props in h.elements:
        dtype = np.dtype([(p, "<" + t) for p, t in props])
        if name == "vertex":
            data = np.fromfile(path, dtype=dtype, count=n, offset=offset)
            return {c: np.asarray(data[c], dtype=np.float64) for c in columns}
        offset += n * dtype.itemsize
    raise KeyError("vertex")
