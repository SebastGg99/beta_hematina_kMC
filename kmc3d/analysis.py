"""Análisis de cristales 3D: caras, velocidades, forma y superficie de cada cara.

Reutiliza `src.analysis` (observables 2D sobre una matriz de alturas): cada cara del
cristal 3D se convierte en un mapa de alturas `face_height_map` y se analiza con
las mismas funciones que la línea (2+1)D (rugosidad, islas, escalones, correlación).

Posición de una cara (hkl), dos medidas; la distancia al origen (centro de la
semilla) es D = m · d_hkl, en unidades de la celda (Å para β-hematina):
- plano más EXTERNO m = max(h·r) (`face_series`): lo sigue el motor en cada evento,
  pero sube con la rugosidad y con unidades sueltas (sobrestima R de 1.5 a 3 veces
  en caras rugosas);
- plano MEDIO del centro de la cara (`face_mean_series`): a partir de snapshots,
  es la medida robusta para velocidades.
"""

from functools import lru_cache
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from .geometry import Cell, HKL, hkl_label, parse_hkl, reduce_hkl
from .graph import CrystalGraph
from . import seeds


# ---------------------------------------------------------------------------
# Posición y velocidad de las caras
# ---------------------------------------------------------------------------
def face_series(history: pd.DataFrame, graph: CrystalGraph,
                faces: Sequence[Sequence[int]]) -> pd.DataFrame:
    """Historial del motor -> tabla larga (t, N, cara, plano, D).

    `faces` son las `track_faces` del motor (en el mismo orden). Cada cara seguida da
    dos filas por registro: +(hkl) (plano máximo) y −(hkl) (plano mínimo, cambiado
    de signo).
    """
    rows = []
    for f in faces:
        f = reduce_hkl(f)
        d = graph.cell.d_spacing(f)
        tag = "".join(str(x) for x in f)
        neg = (-f[0], -f[1], -f[2])
        for sign, face, col in ((1, f, f"pmax_{tag}"), (-1, neg, f"pmin_{tag}")):
            planes = sign * history[col].to_numpy()
            rows.append(pd.DataFrame({"t": history["t"].to_numpy(), "N": history["N"].to_numpy(),
                                      "eventos": history["eventos"].to_numpy(),
                                      "cara": hkl_label(face), "familia": hkl_label(f),
                                      "plano": planes, "D": planes * d}))
    return pd.concat(rows, ignore_index=True)


def face_growth_rates(series: pd.DataFrame, t_window: Optional[Tuple[float, float]] = None,
                      late_fraction: float = 0.5) -> pd.DataFrame:
    """Velocidad de cada cara: pendiente de D(t) por mínimos cuadrados.

    Por defecto usa la mitad final del tiempo (régimen estacionario). Devuelve una fila
    por cara con R (unidades de celda / unidad de tiempo), R por familia (media de
    ±hkl) y R relativa a la familia más lenta.
    """
    rows = []
    for cara, g in series.groupby("cara", sort=False):
        t = g["t"].to_numpy()
        if t_window is not None:
            m = (t >= t_window[0]) & (t <= t_window[1])
        else:
            m = t >= t.min() + (1.0 - late_fraction) * (t.max() - t.min())
        if m.sum() < 3:
            R = np.nan
        else:
            R = float(np.polyfit(t[m], g["D"].to_numpy()[m], 1)[0])
        rows.append({"cara": cara, "familia": g["familia"].iloc[0], "R": R})
    out = pd.DataFrame(rows)
    fam = out.groupby("familia", sort=False)["R"].mean().rename("R_familia")
    out = out.join(fam, on="familia")
    out["R_rel"] = out["R_familia"] / out["R_familia"].min()
    return out


def face_mean_plane(coords: np.ndarray, hkl: Sequence[int], crop: float = 0.4) -> float:
    """Plano medio de la superficie en el CENTRO de la cara +(hkl).

    Más robusto que el plano más externo (que sube con la rugosidad y con unidades
    sueltas): es la media de la altura de las columnas (base de `face_frame`) en el
    rectángulo central de fracción `crop` de la proyección del cristal.
    """
    B = face_frame(hkl)
    abc = np.rint(np.asarray(coords, float) @ np.linalg.inv(B)).astype(int)
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    a0, b0 = a.min(), b.min()
    H = np.full((a.max() - a0 + 1, b.max() - b0 + 1), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(H, (a - a0, b - b0), c)
    na, nb = H.shape
    pa, pb = int(na * (1 - crop) / 2), int(nb * (1 - crop) / 2)
    sub = H[pa:na - pa, pb:nb - pb]
    sub = sub[sub > np.iinfo(np.int64).min]
    return float(sub.mean())


def face_mean_series(snaps, graph: CrystalGraph, faces: Sequence[Sequence[int]],
                     crop: float = 0.4) -> pd.DataFrame:
    """Como `face_series`, pero con el plano medio del centro de cada cara (±hkl).

    Necesita snapshots frecuentes (`run(snapshot_every=...)`). Compatible con
    `face_growth_rates`.
    """
    rows = []
    for t, coords in snaps:
        for f in faces:
            f = reduce_hkl(f)
            d = graph.cell.d_spacing(f)
            for face in (f, (-f[0], -f[1], -f[2])):
                m = face_mean_plane(coords, face, crop)
                rows.append({"t": t, "N": len(coords), "cara": hkl_label(face),
                             "familia": hkl_label(f), "plano": m, "D": m * d})
    return pd.DataFrame(rows)


def snapshot_face_distances(coords: np.ndarray, graph: CrystalGraph,
                            faces: Iterable[Sequence[int]], q: Optional[float] = None) -> pd.DataFrame:
    """Distancias de las caras ±(hkl) de un snapshot (coordenadas relativas).

    Con `q` (p. ej. 0.999) se usa un cuantil de h·r en vez del máximo: menos sensible
    a unidades sueltas sobre la cara.
    """
    rows = []
    for f in faces:
        f = reduce_hkl(f)
        p = coords @ np.asarray(f)
        d = graph.cell.d_spacing(f)
        top = p.max() if q is None else np.quantile(p, q)
        bot = p.min() if q is None else np.quantile(p, 1 - q)
        rows.append({"cara": hkl_label(f), "D": float(top * d)})
        rows.append({"cara": hkl_label((-f[0], -f[1], -f[2])), "D": float(-bot * d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forma global
# ---------------------------------------------------------------------------
def shape_descriptors(coords: np.ndarray, cell: Cell) -> Dict[str, object]:
    """Dimensiones principales a partir del tensor de inercia (coordenadas reales).

    L1 >= L2 >= L3 son las aristas de la caja con la misma covarianza
    (L = √(12 λ)). `eje_largo_uvw` es la dirección de red más próxima al eje largo
    (entre vectores con componentes de −2 a 2) y `angulo_eje_largo` el ángulo con
    ella, en grados.
    """
    out = _descriptors_from_cartesian(cell.to_cartesian(coords), cell)
    out["N"] = int(len(coords))
    return out


def _descriptors_from_cartesian(X: np.ndarray, cell: Cell) -> Dict[str, object]:
    """Descriptores de forma de una nube de puntos cartesianos (ver shape_descriptors)."""
    X = X - X.mean(axis=0)
    lam, vec = np.linalg.eigh(np.cov(X.T))
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    L = np.sqrt(12.0 * np.clip(lam, 0, None))
    long_axis = vec[:, 0]
    best, best_ang = None, 180.0
    for uvw in product(range(-2, 3), repeat=3):
        # solo vectores primitivos con signo canónico ([001], no [00-2])
        if uvw == (0, 0, 0) or reduce_hkl(uvw) != uvw or next(x for x in uvw if x) < 0:
            continue
        v = np.asarray(uvw, float) @ cell.matrix
        ang = np.degrees(np.arccos(min(1.0, abs(v @ long_axis) / np.linalg.norm(v))))
        if ang < best_ang - 1e-9:
            best, best_ang = uvw, ang
    # cocientes con L = 0 (forma de una sola capa) se dan como inf sin avisos
    with np.errstate(divide="ignore"):
        ratios = {"L1/L2": L[0] / L[1], "L2/L3": L[1] / L[2], "L1/L3": L[0] / L[2]}
    return {"L1": L[0], "L2": L[1], "L3": L[2], **{k: float(v) for k, v in ratios.items()},
            "eje_largo_uvw": best, "angulo_eje_largo": float(best_ang)}


def wulff_shape_descriptors(rates: Dict[Sequence[int], float], cell: Cell,
                            centrosymmetric: bool = True, n_samples: int = 400_000,
                            seed: int = 0) -> Dict[str, object]:
    """Descriptores de forma del poliedro de Wulff CONTINUO (distancias ∝ R).

    No voxeliza: muestrea puntos uniformes dentro del poliedro (rechazo en su caja
    envolvente cartesiana). Así sirve también para formas muy planas o muy alargadas,
    cuya versión voxelizada tendría menos de una celda de espesor.
    """
    P = wulff_samples(rates, cell, centrosymmetric, n_samples, seed)
    out = _descriptors_from_cartesian(P, cell)
    out["n_muestras"] = int(len(P))
    return out


def wulff_samples(rates: Dict[Sequence[int], float], cell: Cell, centrosymmetric: bool = True,
                  n_samples: int = 400_000, seed: int = 0) -> np.ndarray:
    """Puntos cartesianos uniformes dentro del poliedro de Wulff (distancias = R / R_min).

    Muestreo por rechazo en la caja envolvente. Lo usan `wulff_shape_descriptors` y
    las dimensiones por dirección (`directional_extents`).
    """
    rates = {reduce_hkl(f): float(v) for f, v in rates.items()}
    if any(not v > 0 for v in rates.values()):
        raise ValueError("Todas las velocidades deben ser positivas")
    rmin = min(rates.values())
    dist = {f: v / rmin for f, v in rates.items()}
    V = seeds.polyhedron_vertices(dist, cell, centrosymmetric)
    faces = seeds._face_dict(dist, centrosymmetric)
    N = np.array([cell.normal(f) for f in faces])
    D = np.array(list(faces.values()))
    rng = np.random.default_rng(seed)
    lo, hi = V.min(axis=0), V.max(axis=0)
    P = lo + (hi - lo) * rng.random((n_samples, 3))
    return P[np.all(P @ N.T <= D, axis=1)]


def directional_extents(X: np.ndarray, directions: Sequence[np.ndarray]) -> np.ndarray:
    """Dimensión del cristal a lo largo de cada dirección cartesiana: L = √(12 var(X·ê)).

    Es la arista de la caja con la misma varianza (exacta para un paralelepípedo
    alineado con ê). Sirve para comparar el cristal kMC (X = cartesianas de las
    unidades) y el poliedro de Wulff (X = `wulff_samples`) con la misma medida.
    """
    X = np.asarray(X, float)
    out = []
    for e in directions:
        e = np.asarray(e, float)
        out.append(np.sqrt(12.0 * np.var(X @ (e / np.linalg.norm(e)))))
    return np.array(out)


def attachment_energy_rates(graph: CrystalGraph, faces: Iterable[Sequence[int]]) -> Dict[HKL, float]:
    """{hkl: |E_att|}: velocidades relativas del modelo de energía de unión."""
    table = graph.face_table(faces)
    return {(int(r.h), int(r.k), int(r.l)): abs(r.E_att) for r in table.itertuples()}


def morphology_from_rates(rates: Dict[Sequence[int], float], cell: Cell,
                          size: float, centrosymmetric: bool = True,
                          normalize: str = "slowest") -> np.ndarray:
    """Forma cinética de Wulff discreta: distancias de cara ∝ velocidad.

    Parámetros:
        rates: {hkl: R > 0} (o |E_att| para la forma de energía de unión)
        size: escala en unidades de la celda (Å para β-hematina)
        normalize: "slowest" -> la cara más lenta queda a distancia `size`;
            "largest" -> el vértice más lejano queda a distancia `size` (útil si
            las velocidades son muy distintas y la forma sería enorme)
    Devuelve índices (N, 3).
    """
    rates = {reduce_hkl(f): float(v) for f, v in rates.items()}
    if any(not v > 0 for v in rates.values()):
        bad = [hkl_label(f) for f, v in rates.items() if not v > 0]
        raise ValueError(f"Velocidades no positivas en {bad}: la forma de Wulff cinética "
                         "no está definida (quita esas caras o usa más eventos)")
    rmin = min(rates.values())
    dist = {f: v / rmin for f, v in rates.items()}
    if normalize == "largest":
        rmax_vertex = np.linalg.norm(seeds.polyhedron_vertices(dist, cell, centrosymmetric), axis=1).max()
        scale = size / rmax_vertex
    elif normalize == "slowest":
        scale = size
    else:
        raise ValueError("normalize debe ser 'slowest' o 'largest'")
    return seeds.polyhedron({f: scale * d for f, d in dist.items()}, cell, centrosymmetric)


def family_rates(rates_table: pd.DataFrame) -> Dict[HKL, float]:
    """Salida de `face_growth_rates` -> {hkl: R de la familia} (para formas de Wulff)."""
    out = {}
    for fam, R in rates_table.groupby("familia", sort=False)["R_familia"].first().items():
        out[parse_hkl(fam)] = float(R)
    return out


def attachment_energy_morphology(graph: CrystalGraph, faces: Iterable[Sequence[int]],
                                 size: float, normalize: str = "slowest") -> np.ndarray:
    """Forma de Hartman-Perdok: R_hkl ∝ |E_att(hkl)| (ver CrystalGraph.face_table)."""
    return morphology_from_rates(attachment_energy_rates(graph, faces), graph.cell, size,
                                 normalize=normalize)


# ---------------------------------------------------------------------------
# Mapa de alturas de una cara (para reutilizar src.analysis)
# ---------------------------------------------------------------------------
def face_frame(hkl: Sequence[int], search: int = 3) -> np.ndarray:
    """Base entera unimodular (filas u, v, w) con h·u = h·v = 0 y h·w = 1.

    En esa base, r = a·u + b·v + c·w con a, b, c enteros y c = (hkl)·r: c es la
    "altura" sobre la cara y (a, b) la columna. Busca vectores con componentes
    en [−search, search] (suficiente para caras de índices bajos). El resultado se
    guarda en caché (las series por cara la llaman miles de veces).
    """
    return _face_frame_cached(reduce_hkl(hkl), int(search)).copy()


@lru_cache(maxsize=None)
def _face_frame_cached(hkl: HKL, search: int) -> np.ndarray:
    h = np.asarray(hkl)
    cands = [np.array(v) for v in product(range(-search, search + 1), repeat=3) if any(v)]
    inplane = sorted([v for v in cands if v @ h == 0], key=lambda v: (np.abs(v).sum(), tuple(v)))
    w = min((v for v in cands if v @ h == 1), key=lambda v: (np.abs(v).sum(), tuple(v)))
    for i, u in enumerate(inplane):
        for v in inplane[i + 1:]:
            cr = np.cross(u, v)
            if np.array_equal(cr, h) or np.array_equal(cr, -h):
                return np.array([u, v, w])
    raise ValueError(f"No se encontró una base para {hkl_label(h)}; aumenta `search`")


def face_height_map(coords: np.ndarray, hkl: Sequence[int], crop: float = 0.6,
                    band: int = 3) -> np.ndarray:
    """Mapa de alturas de la cara (hkl) del cristal, listo para `src.analysis`.

    1. Pasa las coordenadas a la base de `face_frame`: columnas (a, b), altura c.
    2. La altura de cada columna es el máximo c (la superficie vista desde la solución).
    3. La cara son las columnas con altura >= c_max − band; se toma el rectángulo
       central de su caja envolvente (fracción `crop` en cada dirección) para evitar
       las aristas con otras caras.
    Devuelve alturas enteras >= 0 (se resta el mínimo). Si la cara es demasiado
    pequeña lanza ValueError.
    """
    B = face_frame(hkl)
    abc = np.rint(np.asarray(coords, float) @ np.linalg.inv(B)).astype(int)
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    a0, b0 = a.min(), b.min()
    H = np.full((a.max() - a0 + 1, b.max() - b0 + 1), np.iinfo(np.int32).min, dtype=np.int64)
    np.maximum.at(H, (a - a0, b - b0), c)
    on_face = H >= c.max() - band
    ia, ib = np.nonzero(on_face)
    if len(ia) == 0:
        raise ValueError("Cara vacía")
    lo_a, hi_a, lo_b, hi_b = ia.min(), ia.max(), ib.min(), ib.max()
    pa = int(round((hi_a - lo_a + 1) * (1 - crop) / 2))
    pb = int(round((hi_b - lo_b + 1) * (1 - crop) / 2))
    sub = H[lo_a + pa: hi_a - pa + 1, lo_b + pb: hi_b - pb + 1]
    if sub.size < 9 or min(sub.shape) < 3:
        raise ValueError(f"La cara {hkl_label(hkl)} es demasiado pequeña para un mapa de alturas")
    if np.any(sub == np.iinfo(np.int32).min):
        raise ValueError("Hay columnas vacías dentro del recorte; reduce `crop`")
    return (sub - sub.min()).astype(int)


def face_depth_map(coords: np.ndarray, hkl: Sequence[int]) -> Dict[str, object]:
    """Mapa 2D COMPLETO de la cara (hkl), para visualizarla (sin recortar).

    Usa la base de `face_frame`: cada columna (a, b) es una fila de sitios a lo largo
    de w, y c = (hkl)·r es la altura sobre el plano (hkl). En cada columna se toma la
    unidad más externa (máximo c) y se da su profundidad bajo el plano (hkl) más
    externo del cristal:

        depth(a, b) = c_max − max_c{columna (a, b)}   (0 = capa externa, 1 = una capa
                                                        por debajo, ...)

    A diferencia de `face_height_map` no recorta nada: incluye las aristas y las caras
    vecinas (profundidades grandes), que `viz.plot_face_maps` separa con un umbral.

    Devuelve un dict con:
        depth: array float (na, nb), NaN en las columnas sin ninguna unidad
        a0, b0: índices (a, b) del elemento [0, 0] de `depth`
        frame: la base entera B de `face_frame` (filas u, v, w)
    """
    B = face_frame(hkl)
    abc = np.rint(np.asarray(coords, float) @ np.linalg.inv(B)).astype(int)
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    a0, b0 = int(a.min()), int(b.min())
    vacio = np.iinfo(np.int64).min
    H = np.full((a.max() - a0 + 1, b.max() - b0 + 1), vacio, dtype=np.int64)
    np.maximum.at(H, (a - a0, b - b0), c)
    # Profundidad bajo el plano más externo; las columnas vacías quedan en NaN
    depth = np.where(H == vacio, np.nan, (c.max() - H).astype(float))
    return {"depth": depth, "a0": a0, "b0": b0, "frame": B}


def face_surface_observables(coords: np.ndarray, hkl: Sequence[int], crop: float = 0.6,
                             **kwargs) -> Dict[str, float]:
    """Observables de `src.analysis.surface_observables` sobre la cara (hkl)."""
    from src.analysis import surface_observables
    hmap = face_height_map(coords, hkl, crop=crop, **kwargs)
    out = surface_observables(hmap)
    out["cara"] = hkl_label(hkl)
    out["Lx"], out["Ly"] = hmap.shape
    return out


# ---------------------------------------------------------------------------
# Fragmentos (componentes conexas por el grafo de enlaces)
# ---------------------------------------------------------------------------
def cluster_labels(coords: np.ndarray, graph: CrystalGraph) -> np.ndarray:
    """Etiqueta de componente conexa de cada unidad (0 = la mayor, 1 = la siguiente...).

    Dos unidades están conectadas si son vecinas según algún enlace del grafo. Con
    difusión superficial y sin intercambio con la solución, pequeños fragmentos pueden
    alejarse del cristal (es un gas de red): los descriptores globales de forma deben
    calcularse sobre el fragmento mayor (`largest_cluster`).
    Búsqueda en anchura vectorizada por niveles (numpy), sin scipy.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    n = len(coords)
    if n == 0:
        return np.zeros(0, dtype=int)
    pad = graph.reach
    lo = coords.min(axis=0) - pad
    dims = coords.max(axis=0) - lo + 1 + pad
    flat = np.ravel_multi_index(tuple((coords - lo).T), tuple(dims))
    # índice de unidad de cada sitio de la caja (−1 = vacío)
    unit_at = np.full(int(np.prod(dims)), -1, dtype=np.int64)
    unit_at[flat] = np.arange(n)
    strides = np.array([dims[1] * dims[2], dims[2], 1])
    offs = graph.vectors @ strides
    offs = np.r_[offs, -offs]
    label = np.full(n, -1, dtype=np.int64)
    current = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        label[start] = current
        frontier = np.array([flat[start]])
        while len(frontier):
            cand = (frontier[:, None] + offs[None, :]).ravel()
            u = unit_at[cand]
            u = np.unique(u[u >= 0])
            u = u[label[u] < 0]
            label[u] = current
            frontier = flat[u]
        current += 1
    # renumerar por tamaño decreciente (0 = fragmento mayor)
    sizes = np.bincount(label)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[label]


def largest_cluster(coords: np.ndarray, graph: CrystalGraph) -> np.ndarray:
    """Coordenadas del fragmento conexo mayor (ver `cluster_labels`)."""
    coords = np.asarray(coords)
    return coords[cluster_labels(coords, graph) == 0]


# ---------------------------------------------------------------------------
# Sitios de superficie (análogo 3D de la Fig. 8)
# ---------------------------------------------------------------------------
def attachment_probabilities(engine, by: str = "bonds") -> pd.DataFrame:
    """Probabilidad de que la próxima UNIÓN ocurra en cada tipo de sitio.

    P_c = n_c k+_c / Σ n k+ sobre las clases de unión del motor. `by="bonds"` agrupa
    por número total de vecinos ocupados (0 = adatom aislado ... 2T), `by="class"`
    deja una fila por código de vecindad (distingue tipos de enlace y orientación).
    """
    ct = engine.class_table()
    ct = ct[ct["evento"] == "attach"].copy()
    if ct.empty:
        return ct
    ct["P"] = ct["peso"] / ct["peso"].sum()
    if by == "class":
        return ct.sort_values("P", ascending=False, ignore_index=True)
    g = ct.groupby("n_enlaces").agg(n_sitios=("n_sitios", "sum"), P=("P", "sum"))
    return g.reset_index()


def event_balance(engine) -> Dict[str, float]:
    """Peso total de uniones, despegues y saltos, y el cociente unión/despegue (> 1 crece)."""
    ct = engine.class_table()
    wa = float(ct.loc[ct["evento"] == "attach", "peso"].sum())
    wd = float(ct.loc[ct["evento"] == "detach", "peso"].sum())
    wh = float(ct.loc[ct["evento"] == "hop", "peso"].sum()) if len(ct) else 0.0
    return {"W_attach": wa, "W_detach": wd, "W_hop": wh,
            "W_attach/W_detach": wa / wd if wd > 0 else np.inf}


# ---------------------------------------------------------------------------
# Morfología: caras del poliedro de Wulff y facetas del cristal kMC
# ---------------------------------------------------------------------------
def polyhedron_faces(distances: Dict[Sequence[int], float], cell: Cell,
                     centrosymmetric: bool = True) -> List[Dict[str, object]]:
    """Polígonos de las caras del poliedro {x : n_hkl · x <= D_hkl}.

    Devuelve una lista con un dict por cara de área no nula: `hkl` (con signo),
    `familia` (etiqueta de la cara de `distances` a la que pertenece, igual para ±hkl
    si hay inversión), `vertices` (cartesianos, ordenados en sentido antihorario visto
    desde fuera), `area`, `centroid`, `normal`, `D`. Una cara que el resto de caras
    reduce a una arista o a un vértice no aparece.
    """
    fam_of, dist = {}, {}
    for f, D in distances.items():
        r = reduce_hkl(f)
        fam_of[r], dist[r] = hkl_label(r), float(D)
        if centrosymmetric:
            neg = (-r[0], -r[1], -r[2])
            fam_of.setdefault(neg, hkl_label(r))
            dist.setdefault(neg, float(D))
    V = seeds.polyhedron_vertices(dist, cell, centrosymmetric=False)
    tol = 1e-6 * max(1.0, float(np.abs(V).max()))
    out = []
    for f, D in dist.items():
        n = cell.normal(f)
        P = V[np.abs(V @ n - D) <= tol]
        if len(P) < 3:
            continue
        c = P.mean(axis=0)
        e1 = P[0] - c
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)                        # (e1, e2, n) directa: antihorario desde fuera
        ang = np.arctan2((P - c) @ e2, (P - c) @ e1)
        P = P[np.argsort(ang)]
        area = 0.5 * abs(float(np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0) @ n))
        if area <= tol ** 2:
            continue
        out.append({"hkl": f, "familia": fam_of[f], "vertices": P, "area": area,
                    "centroid": c, "normal": n, "D": D})
    return out


def wulff_faces(rates: Dict[Sequence[int], float], cell: Cell,
                centrosymmetric: bool = True) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    """Caras del poliedro de Wulff (distancias = R / R_min) y área de cada familia.

    Devuelve (caras, tabla). La tabla tiene una fila por familia de `rates`, también
    las que no aparecen (área 0), con el área absoluta (en unidades de la celda², con
    la cara más lenta a distancia 1) y la fracción del área total.
    """
    rates = {reduce_hkl(f): float(v) for f, v in rates.items()}
    if any(not v > 0 for v in rates.values()):
        raise ValueError("Todas las velocidades deben ser positivas")
    rmin = min(rates.values())
    caras = polyhedron_faces({f: v / rmin for f, v in rates.items()}, cell, centrosymmetric)
    area = {hkl_label(f): 0.0 for f in rates}
    for c in caras:
        area[c["familia"]] += c["area"]
    tab = pd.DataFrame({"familia": list(area), "area": list(area.values())})
    tab["fraccion"] = tab["area"] / tab["area"].sum()
    return caras, tab


def surface_normals(coords: np.ndarray, cell: Cell, radius_cells: float = 2.5
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Normal exterior suavizada de cada unidad de superficie del cristal.

    Superficie: unidades con algún vecino vacío entre los 6 de índices ±1 (como
    `viz.surface_mask`). La normal es la suma de los vectores cartesianos hacia los
    sitios VACÍOS dentro de una esfera de radio `radius_cells` · V^(1/3) (V = volumen
    de la celda): apunta hacia la solución y promedia la rugosidad a escala de unas
    pocas celdas, así que en una faceta plana es la normal de la faceta.

    Devuelve (índices de las unidades de superficie en `coords`, normales unitarias).
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    M = cell.matrix
    R = radius_cells * cell.volume ** (1.0 / 3.0)
    lim = np.ceil(R * np.linalg.norm(cell.reciprocal, axis=1)).astype(int)
    rng = [np.arange(-L, L + 1) for L in lim]
    offs = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    offs = offs[(np.linalg.norm(offs @ M, axis=1) <= R) & np.any(offs != 0, axis=1)]
    pad = int(lim.max()) + 1
    lo = coords.min(axis=0) - pad
    occ = np.zeros(tuple(coords.max(axis=0) - lo + 1 + pad), dtype=bool)
    idx = coords - lo
    occ[tuple(idx.T)] = True
    # superficie con los 6 vecinos de índices
    lleno = np.ones(len(coords), dtype=bool)
    for eje in range(3):
        for paso in (-1, 1):
            v = idx.copy()
            v[:, eje] += paso
            lleno &= occ[tuple(v.T)]
    surf = np.flatnonzero(~lleno)
    sidx = idx[surf]
    acc = np.zeros((len(surf), 3))
    for o, oc in zip(offs, offs @ M):
        vacio = ~occ[tuple((sidx + o).T)]
        acc += vacio[:, None] * oc[None, :]
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    return surf, acc / np.where(norm > 0, norm, 1.0)


def facet_labels(coords: np.ndarray, cell: Cell, faces: Sequence[Sequence[int]],
                 radius_cells: float = 2.5, max_angle: float = 20.0,
                 centrosymmetric: bool = True) -> Dict[str, np.ndarray]:
    """Asigna cada unidad de superficie a la cara (hkl) cuya normal es la más próxima.

    Usa `surface_normals`. Una unidad cuya normal se aparta más de `max_angle` grados
    de todas las caras candidatas (±hkl de cada familia de `faces`) queda sin cara
    (−1): aristas, vértices y zonas muy rugosas.

    Devuelve un dict con `surface` (índices en coords), `normals`, `family` (índice en
    `faces` o −1), `sign` (+1 / −1: cara (hkl) o (−h−k−l)) y `cos` (coseno con la cara
    asignada).
    """
    faces = [reduce_hkl(f) for f in faces]
    surf, normals = surface_normals(coords, cell, radius_cells)
    cand, fam, sgn = [], [], []
    for i, f in enumerate(faces):
        for s in ((1, -1) if centrosymmetric else (1,)):
            cand.append(cell.normal(tuple(s * x for x in f)))
            fam.append(i)
            sgn.append(s)
    cosm = normals @ np.array(cand).T
    best = np.argmax(cosm, axis=1)
    cbest = cosm[np.arange(len(best)), best]
    ok = cbest >= np.cos(np.radians(max_angle))
    return {"surface": surf, "normals": normals,
            "family": np.where(ok, np.array(fam)[best], -1),
            "sign": np.where(ok, np.array(sgn)[best], 0), "cos": cbest}


def facet_areas(coords: np.ndarray, cell: Cell, faces: Sequence[Sequence[int]],
                labels: Optional[Dict[str, np.ndarray]] = None, **kwargs) -> pd.DataFrame:
    """Área de cada familia de caras en el cristal kMC (a partir de `facet_labels`).

    Para cada cara con signo, el área es el número de COLUMNAS distintas (base de
    `face_frame`) que ocupan sus unidades de superficie por el área de una columna,
    V / d_hkl. Contar columnas y no unidades evita sumar dos veces los escalones de
    una cara rugosa. Devuelve una fila por familia (con área 0 si no aparece) y la
    fracción del área de superficie asignada a alguna cara.
    """
    faces = [reduce_hkl(f) for f in faces]
    coords = np.asarray(coords)
    lab = labels if labels is not None else facet_labels(coords, cell, faces, **kwargs)
    filas = []
    for i, f in enumerate(faces):
        area = 0.0
        for s in (1, -1):
            sel = lab["surface"][(lab["family"] == i) & (lab["sign"] == s)]
            if len(sel) == 0:
                continue
            hkl = tuple(s * x for x in f)
            B = face_frame(hkl)
            ab = np.rint(coords[sel] @ np.linalg.inv(B)).astype(int)[:, :2]
            n_col = len(np.unique(ab, axis=0))
            area += n_col * cell.volume / cell.d_spacing(hkl)
        filas.append({"familia": hkl_label(f), "area": area})
    tab = pd.DataFrame(filas)
    total = tab["area"].sum()
    tab["fraccion"] = tab["area"] / total if total > 0 else 0.0
    return tab
