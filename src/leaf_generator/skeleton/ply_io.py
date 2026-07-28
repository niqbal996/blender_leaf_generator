"""Minimal, dependency-free PLY vertex I/O (ASCII and binary_little_endian).

This repo needs to read/write two shapes of "point cloud": a plain xyz(+rgb)
point cloud, and a full 3D Gaussian Splat vertex table (x, y, z, normals, SH
coefficients, opacity, scale, rotation). Both are just "a vertex table with
named columns", so one generic reader/writer covers both instead of two
bespoke ones -- no `plyfile`/`open3d` dependency needed.

Binary support exists specifically to read back the ply gsplat's own
`export_splats` writes (binary_little_endian, for file size -- a trained
splat can be a million+ Gaussians, and ASCII would bloat that badly) without
needing `torch`/`gsplat` imported just to transform an already-trained file
(see `align_plant_skeleton.py`'s alignment-baking step).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

_PLY_TYPE_BY_DTYPE_KIND = {
    ("u", 1): "uchar",
    ("i", 4): "int",
    ("f", 4): "float",
    ("f", 8): "double",
}

_NUMPY_DTYPE_BY_PLY_TYPE = {
    "uchar": np.uint8,
    "uint8": np.uint8,
    "int": np.int32,
    "int32": np.int32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


def write_ply_vertices(path: Union[str, Path], fields: Dict[str, np.ndarray], binary: bool = False) -> None:
    """Write a PLY with a single `element vertex` block, one column per
    entry of `fields` (in insertion order). Each value must be a 1-D array
    of the same length; its numpy dtype determines the PLY property type
    (uchar for uint8, int for int32, float/double for float32/float64).

    `binary=True` writes `binary_little_endian` (compact); the default
    ASCII is human-inspectable and fine for the point-cloud-sized files this
    repo writes directly, but should not be used for a million-Gaussian splat.
    """
    if not fields:
        raise ValueError("fields must contain at least one property")

    names = list(fields.keys())
    columns = [np.asarray(fields[name]) for name in names]
    n = len(columns[0])
    for name, col in zip(names, columns):
        if col.ndim != 1 or len(col) != n:
            raise ValueError(f"field {name!r} must be a 1-D array of length {n}, got shape {col.shape}")

    ply_types = [_ply_type_for(col.dtype) for col in columns]

    with open(Path(path), "wb") as f:
        f.write(b"ply\n")
        f.write(f"format {'binary_little_endian' if binary else 'ascii'} 1.0\n".encode("ascii"))
        f.write(f"element vertex {n}\n".encode("ascii"))
        for name, ply_type in zip(names, ply_types):
            f.write(f"property {ply_type} {name}\n".encode("ascii"))
        f.write(b"end_header\n")

        if binary:
            structured_dtype = np.dtype(
                [(name, col.dtype.newbyteorder("<")) for name, col in zip(names, columns)]
            )
            structured = np.zeros(n, dtype=structured_dtype)
            for name, col in zip(names, columns):
                structured[name] = col
            f.write(structured.tobytes())
        else:
            for row in zip(*columns):
                f.write((" ".join(_format_value(v) for v in row) + "\n").encode("ascii"))


def read_ply_vertices(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Inverse of `write_ply_vertices` -- reads ASCII or binary_little_endian
    PLYs (including ones written by other tools, e.g. gsplat's
    `export_splats`), as long as they have a single `element vertex` block.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    header_end = raw.find(b"end_header\n")
    if header_end == -1:
        raise ValueError(f"no 'end_header' found in {path}")
    header_text = raw[:header_end].decode("ascii")
    body = raw[header_end + len(b"end_header\n") :]

    header_lines = header_text.splitlines()
    if not header_lines or header_lines[0].strip() != "ply":
        raise ValueError(f"not a PLY file: {path}")

    is_binary = False
    n_vertices = None
    props = []  # (ply_type, name)
    in_vertex_element = False
    for line in header_lines[1:]:
        line = line.strip()
        if line.startswith("format"):
            if "ascii" in line:
                is_binary = False
            elif "binary_little_endian" in line:
                is_binary = True
            else:
                raise ValueError(f"only ASCII or binary_little_endian PLY is supported, got: {line!r}")
        elif line.startswith("element"):
            _, elem_name, count = line.split()
            in_vertex_element = elem_name == "vertex"
            if in_vertex_element:
                n_vertices = int(count)
        elif line.startswith("property") and in_vertex_element:
            parts = line.split()
            ply_type, name = parts[1], parts[-1]
            props.append((ply_type, name))

    if n_vertices is None:
        raise ValueError(f"no 'element vertex' found in {path}")

    fields: Dict[str, np.ndarray] = {}
    if is_binary:
        structured_dtype = np.dtype(
            [(name, np.dtype(_NUMPY_DTYPE_BY_PLY_TYPE.get(ply_type, np.float32)).newbyteorder("<")) for ply_type, name in props]
        )
        structured = np.frombuffer(body, dtype=structured_dtype, count=n_vertices)
        for ply_type, name in props:
            fields[name] = np.array(structured[name], dtype=_NUMPY_DTYPE_BY_PLY_TYPE.get(ply_type, np.float32))
    else:
        data_lines = body.decode("ascii").splitlines()
        data_lines = [line for line in data_lines if line.strip()][:n_vertices]
        if len(data_lines) != n_vertices:
            raise ValueError(f"expected {n_vertices} vertex rows, found {len(data_lines)} in {path}")
        parsed = np.array([line.split() for line in data_lines], dtype=np.float64)
        for col_idx, (ply_type, name) in enumerate(props):
            dtype = _NUMPY_DTYPE_BY_PLY_TYPE.get(ply_type, np.float32)
            fields[name] = parsed[:, col_idx].astype(dtype)

    return fields


def _ply_type_for(dtype: np.dtype) -> str:
    key = (dtype.kind, dtype.itemsize)
    if key not in _PLY_TYPE_BY_DTYPE_KIND:
        raise ValueError(f"unsupported dtype for PLY export: {dtype}")
    return _PLY_TYPE_BY_DTYPE_KIND[key]


def _format_value(v) -> str:
    if isinstance(v, np.floating):
        return repr(float(v))
    return str(int(v))
