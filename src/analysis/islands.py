"""Islas por capa y densidad de escalones a partir de las alturas (O4, O5).

El etiquetado de componentes conexas es 4-conexo y **periódico**, igual que la red
del motor: una isla que cruza el borde cuenta como una sola. Se implementa con
union-find porque `scipy.ndimage.label` no está entre las librerías permitidas (y
además no es periódico).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def label_periodic(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Etiqueta las componentes 4-conexas de `mask` con contorno periódico.

    Parámetros:
        mask: matriz booleana (Lx, Ly)

    Devuelve:
        (labels, n): `labels` vale 0..n−1 en los sitios de la máscara y −1 fuera.
        Las etiquetas se numeran en el orden en que aparece cada componente al
        recorrer la red por filas.
    """
    mask = np.asarray(mask, dtype=bool)
    nx, ny = mask.shape
    labels = np.full(mask.shape, -1, dtype=int)
    if not mask.any():
        return labels, 0

    idx = np.arange(nx * ny).reshape(nx, ny)
    parent = list(range(nx * ny))  # lista de Python: más rápida que numpy en el bucle

    def find(a: int) -> int:
        """Raíz de `a`, con compresión de camino."""
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    # Enlaces entre cada sitio y su vecino +1 en cada eje (np.roll = periodicidad).
    # Con los vecinos +1 de los dos ejes se cubren todos los enlaces de la red.
    for axis in (0, 1):
        neighbor = np.roll(idx, -1, axis=axis)
        both = mask & np.roll(mask, -1, axis=axis)
        for a, b in zip(idx[both].tolist(), neighbor[both].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                # La raíz menor manda: resultado determinista
                parent[max(ra, rb)] = min(ra, rb)

    occupied = idx[mask]
    roots = np.array([find(a) for a in occupied.tolist()])
    # Renumeración consecutiva por orden de primera aparición
    _, first_pos, inverse = np.unique(roots, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first_pos))
    labels[mask] = order[inverse]
    return labels, int(first_pos.size)


def component_sizes(mask: np.ndarray) -> np.ndarray:
    """Tamaños (en sitios) de las componentes periódicas 4-conexas de `mask`."""
    labels, n = label_periodic(mask)
    if n == 0:
        return np.zeros(0, dtype=int)
    return np.bincount(labels[labels >= 0], minlength=n)


def island_statistics(heights: np.ndarray) -> Dict[str, object]:
    """Islas y huecos en cada capa expuesta.

    Para cada nivel z con 0 < cobertura(z) < 1 (z en [min(h), max(h) − 1]):
    - islas = componentes de {h > z} (columnas que llegan por encima del nivel z);
    - huecos = componentes de {h ≤ z}.
    Una capa ya completa o todavía vacía no aporta islas.

    Devuelve:
        n_islands, n_holes: totales sumados sobre las capas expuestas
        island_density: n_islands / (Lx·Ly)
        mean_island_size: tamaño medio en sitios (NaN si no hay islas)
        max_island_fraction: isla más grande / (Lx·Ly) (0 si no hay islas)
        island_sizes: array con el tamaño de cada isla (para histogramas, C2)
        islands_per_level: {z: número de islas en ese nivel}
    """
    h = np.asarray(heights, dtype=int)
    n_sites = h.size
    sizes: List[np.ndarray] = []
    per_level: Dict[int, int] = {}
    n_holes = 0
    for z in range(int(h.min()), int(h.max())):
        above = h > z
        level_sizes = component_sizes(above)
        sizes.append(level_sizes)
        per_level[z] = int(level_sizes.size)
        n_holes += int(component_sizes(~above).size)

    all_sizes = np.concatenate(sizes) if sizes else np.zeros(0, dtype=int)
    n_islands = int(all_sizes.size)
    return {
        "n_islands": n_islands,
        "n_holes": n_holes,
        "island_density": n_islands / n_sites,
        "mean_island_size": float(all_sizes.mean()) if n_islands else float("nan"),
        "max_island_fraction": float(all_sizes.max()) / n_sites if n_islands else 0.0,
        "island_sizes": all_sizes,
        "islands_per_level": per_level,
    }


def step_statistics(heights: np.ndarray) -> Dict[str, float]:
    """Densidad de escalones calculada solo con las alturas (periódica).

    Eje 0 = x y eje 1 = y, la misma convención que `LatticeSOSStatic`
    (`E_pb_over_kT_x` actúa entre vecinos que difieren en el primer índice).

    Devuelve:
        step_density_x / _y: fracción de enlaces en x / y con Δh ≠ 0
        step_density: media de las dos
        step_anisotropy: step_density_x / step_density_y (inf si solo hay escalones
            en x; NaN si no hay escalones)
        mean_abs_dh: ⟨|Δh|⟩ sobre todos los enlaces
        multi_step_fraction: fracción de escalones con |Δh| ≥ 2 (NaN sin escalones)
    """
    h = np.asarray(heights, dtype=int)
    dx = np.abs(np.roll(h, -1, axis=0) - h)
    dy = np.abs(np.roll(h, -1, axis=1) - h)
    sx = float(np.mean(dx != 0))
    sy = float(np.mean(dy != 0))
    all_d = np.concatenate([dx.ravel(), dy.ravel()])
    steps = all_d[all_d != 0]
    if sy > 0:
        anisotropy = sx / sy
    else:
        anisotropy = float("inf") if sx > 0 else float("nan")
    return {
        "step_density_x": sx,
        "step_density_y": sy,
        "step_density": 0.5 * (sx + sy),
        "step_anisotropy": anisotropy,
        "mean_abs_dh": float(all_d.mean()),
        "multi_step_fraction": float(np.mean(steps >= 2)) if steps.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Densidad de islas resuelta por nivel (análisis de transiciones frente a σ)
# ---------------------------------------------------------------------------
def level_island_statistics(heights: np.ndarray,
                            return_sizes: bool = False) -> Dict[str, object]:
    """Islas y huecos de CADA nivel expuesto, con la cobertura de ese nivel.

    Mismas definiciones que `island_statistics`, pero sin sumar sobre los niveles:
    para cada z en [min(h), max(h) − 1], θ(z) = fracción de columnas con h > z,
    islas = componentes periódicas 4-conexas de {h > z} y huecos = las de {h ≤ z}.
    Permite estudiar la densidad de islas frente a la cobertura de su capa, que es la
    forma estándar de la teoría de nucleación (densidad de saturación antes de la
    coalescencia).

    Devuelve arrays alineados (uno por nivel; vacíos si la superficie es plana):
        level, coverage, n_islands, n_holes, mean_island_size (NaN sin islas),
        largest_island_fraction (isla mayor / (Lx·Ly))
    y, con `return_sizes=True`, `island_sizes`: lista con los tamaños de cada nivel.
    """
    h = np.asarray(heights, dtype=int)
    n_sites = h.size
    levels = np.arange(int(h.min()), int(h.max()))
    coverage, n_isl, n_hol, mean_size, largest, sizes = [], [], [], [], [], []
    for z in levels:
        above = h > z
        s = component_sizes(above)
        coverage.append(float(above.mean()))
        n_isl.append(int(s.size))
        n_hol.append(int(component_sizes(~above).size))
        mean_size.append(float(s.mean()) if s.size else float("nan"))
        largest.append(float(s.max()) / n_sites if s.size else 0.0)
        sizes.append(s)
    out: Dict[str, object] = {
        "level": levels,
        "coverage": np.array(coverage, dtype=float),
        "n_islands": np.array(n_isl, dtype=int),
        "n_holes": np.array(n_hol, dtype=int),
        "mean_island_size": np.array(mean_size, dtype=float),
        "largest_island_fraction": np.array(largest, dtype=float),
    }
    if return_sizes:
        out["island_sizes"] = sizes
    return out


def island_density_observables(heights: np.ndarray,
                               level_stats: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    """Densidad de islas descompuesta en «islas por nivel» × «niveles expuestos».

    Con N = Lx·Ly, el número total de islas de `island_statistics` cumple
    N_I = n_I·N = N_capa · N_niv, donde N_niv = h_max − h_min es el número de niveles
    parcialmente cubiertos y N_capa el número medio de islas por nivel. n_I solo no
    distingue «más islas por capa» (nucleación 2D) de «más capas expuestas» (rugoso);
    la descomposición sí.

    Devuelve:
        n_exposed_levels: N_niv (0 si la superficie es plana)
        islands_per_level: N_capa (NaN si es plana)
        island_density_per_level: N_capa / N, islas por sitio y por capa
        island_spacing_per_level: ℓ = (N_capa/N)^(−1/2), separación típica (sitios)
        active_level_coverage: cobertura θ del nivel más próximo a media capa
        island_density_active: islas de ese nivel / N
        hole_density_active: huecos de ese nivel / N

    `level_stats` permite pasar el resultado de `level_island_statistics(heights)` si
    ya se calculó (evita etiquetar la red otra vez).
    """
    h = np.asarray(heights, dtype=int)
    stats = level_island_statistics(h) if level_stats is None else level_stats
    n_levels = int(stats["level"].size)
    nan = float("nan")
    if n_levels == 0:
        return {"n_exposed_levels": 0, "islands_per_level": nan,
                "island_density_per_level": nan, "island_spacing_per_level": nan,
                "active_level_coverage": nan, "island_density_active": nan,
                "hole_density_active": nan}
    per_level = float(stats["n_islands"].sum()) / n_levels
    density = per_level / h.size
    # Nivel "activo": el de cobertura más cercana a 1/2 (la capa que crece en capa a
    # capa). Con empate, argmin toma el más bajo: resultado determinista.
    active = int(np.argmin(np.abs(stats["coverage"] - 0.5)))
    return {
        "n_exposed_levels": n_levels,
        "islands_per_level": per_level,
        "island_density_per_level": density,
        "island_spacing_per_level": float(density ** -0.5),
        "active_level_coverage": float(stats["coverage"][active]),
        "island_density_active": float(stats["n_islands"][active]) / h.size,
        "hole_density_active": float(stats["n_holes"][active]) / h.size,
    }
