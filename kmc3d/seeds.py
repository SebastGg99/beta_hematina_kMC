"""Semillas (formas iniciales) en el espacio de índices de la red.

Todas devuelven un array (N, 3) de índices enteros centrado aproximadamente en 0.
`polyhedron` construye cualquier forma limitada por caras (hkl): sirve para semillas
facetadas y para dibujar formas de Wulff o de energía de unión (ver `analysis.py`).
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from .geometry import Cell, reduce_hkl


def box(nx: int, ny: int, nz: int) -> np.ndarray:
    """Paralelepípedo de nx·ny·nz celdas (en la celda real es un paralelepípedo
    con aristas a lo largo de a, b, c)."""
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1)
    pts = g.reshape(-1, 3)
    return pts - np.array([nx // 2, ny // 2, nz // 2])


def cube(n: int) -> np.ndarray:
    """Cubo de n³ celdas."""
    return box(n, n, n)


def sphere(radius: float, cell: Cell = Cell()) -> np.ndarray:
    """Unidades cuyo centro cartesiano está a distancia <= radius del origen.

    `radius` está en las unidades de la celda (Å para β-hematina, 1 para cubos).
    """
    M = cell.matrix
    # caja de índices que contiene la esfera: |i| <= R·|a*| (y análogo para j, k)
    lim = np.ceil(radius * np.linalg.norm(cell.reciprocal, axis=1)).astype(int) + 1
    rng = [np.arange(-L, L + 1) for L in lim]
    g = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    keep = np.linalg.norm(g @ M, axis=1) <= radius
    return g[keep]


def _face_dict(distances: Dict[Tuple[int, int, int], float],
               centrosymmetric: bool) -> Dict[Tuple[int, int, int], float]:
    """Reduce los índices y añade −(hkl) si hay inversión."""
    faces = {}
    for f, D in distances.items():
        r = reduce_hkl(f)
        faces[r] = float(D)
        if centrosymmetric:
            faces.setdefault((-r[0], -r[1], -r[2]), float(D))
    return faces


def polyhedron_vertices(distances: Dict[Tuple[int, int, int], float], cell: Cell = Cell(),
                        centrosymmetric: bool = True) -> np.ndarray:
    """Vértices cartesianos del poliedro {x : n_hkl · x <= D_hkl}.

    Intersecta los planos de tres en tres y se queda con los puntos que cumplen todas
    las desigualdades. Lanza ValueError si la forma no está acotada (las normales no
    rodean el origen) o si no tiene volumen.
    """
    faces = _face_dict(distances, centrosymmetric)
    N = np.array([cell.normal(f) for f in faces])
    D = np.array(list(faces.values()))
    # acotado <=> para toda dirección u alguna normal cumple n·u > 0 (muestreo denso)
    rng = np.random.default_rng(0)
    u = rng.normal(size=(4000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    if np.any((u @ N.T).max(axis=1) <= 1e-9):
        raise ValueError("El poliedro no está acotado con estas caras: añade más caras")
    verts = []
    n = len(N)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                A = N[[i, j, k]]
                if abs(np.linalg.det(A)) < 1e-9:
                    continue
                x = np.linalg.solve(A, D[[i, j, k]])
                if np.all(N @ x <= D + 1e-7 * max(1.0, np.abs(D).max())):
                    verts.append(x)
    if not verts:
        raise ValueError("El poliedro está vacío (revisa las distancias)")
    return np.unique(np.round(np.array(verts), 9), axis=0)


def polyhedron(distances: Dict[Tuple[int, int, int], float], cell: Cell = Cell(),
               centrosymmetric: bool = True, max_units: int = 5_000_000) -> np.ndarray:
    """Unidades dentro de la intersección de semiespacios n_hkl · x <= D_hkl.

    Parámetros:
        distances: {(h, k, l): D} distancia del origen a cada cara (unidades de celda)
        centrosymmetric: añade la cara opuesta (−h−k−l) con la misma distancia
        max_units: tope de la caja de búsqueda (protege la memoria)
    En índices: el plano (hkl) que pasa a distancia D es h·i + k·j + l·k = D / d_hkl,
    así que la condición es lineal en los índices. La caja de búsqueda sale de los
    vértices del poliedro (`polyhedron_vertices`).
    """
    faces = _face_dict(distances, centrosymmetric)
    verts_idx = polyhedron_vertices(distances, cell, centrosymmetric) @ np.linalg.inv(cell.matrix)
    lo = np.floor(verts_idx.min(axis=0)).astype(int) - 1
    hi = np.ceil(verts_idx.max(axis=0)).astype(int) + 1
    if np.prod(hi - lo + 1, dtype=float) > max_units:
        raise ValueError(f"Caja de {np.prod(hi - lo + 1, dtype=float):.3g} sitios > max_units: "
                         "reduce las distancias")
    rng = [np.arange(a, b + 1) for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    keep = np.ones(len(g), dtype=bool)
    for f, D in faces.items():
        m_max = D / cell.d_spacing(f)            # plano máximo permitido
        keep &= (g @ np.asarray(f)) <= m_max + 1e-9
    pts = g[keep]
    if len(pts) == 0:
        raise ValueError("El poliedro no contiene ninguna unidad (distancias demasiado pequeñas)")
    return pts
