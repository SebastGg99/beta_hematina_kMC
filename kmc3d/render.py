"""Vistas 3D de cristales con matplotlib (sin WebGL): facetas, poliedros y espesores.

Complementa a `viz.py` con tres vistas pensadas para LEER LA MORFOLOGÍA:

- `render_crystal`: superficie del cristal kMC dibujada como las caras expuestas de
  cada vóxel (paralelogramos con la celda real), coloreada por FACETA: cada unidad de
  superficie se asigna a la cara (hkl) cuya normal es la más próxima a su normal
  suavizada (`analysis.facet_labels`). El sombreado usa esa normal suavizada, así que
  una faceta plana sale de un solo tono aunque sea de índices altos (con la normal del
  vóxel saldría a rayas). Etiquetas (hkl) sobre las caras visibles y tríada a, b, c.
- `plot_polyhedron`: el poliedro de Wulff continuo (`analysis.wulff_faces`) con los
  mismos colores, sombreado y etiquetas, para comparar lado a lado con el kMC.
- `plot_thickness_maps`: proyecciones de ESPESOR a lo largo de direcciones de red, como
  una imagen de TEM. Cada píxel es una columna de la red a lo largo de la dirección de
  vista (con su geometría real), así que no hay moiré: el valor es el número de
  unidades de la columna por la longitud del vector de red, en Å.

Colores de las familias de caras: los 8 tonos categóricos validados (orden fijo, nunca
cíclico) y gris para el resto ("otras") y las aristas. Con más de 8 familias, las que
pasan de la octava se agrupan en "otras": para cambiar cuáles, ordena `faces`. Las
etiquetas (hkl) sobre cada cara son la codificación secundaria (no se depende solo del
color).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .analysis import face_frame, facet_labels
from .geometry import Cell, hkl_label, reduce_hkl

# Paleta categórica validada (8 tonos, orden fijo) y gris para "otras caras / aristas"
FACE_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                "#6250d6", "#e34948"]
OTHER_COLOR = "#b5b4ad"
EDGE_INK = "#52514e"            # tinta secundaria: aristas del poliedro, tríada
# Rampa secuencial de un tono (azul, claro -> oscuro) para magnitudes (espesor, altura)
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def face_colors(faces: Sequence[Sequence[int]]) -> Dict[str, str]:
    """{etiqueta de familia: color}: las 8 primeras familias en orden, el resto gris."""
    out = {}
    for i, f in enumerate(faces):
        out[hkl_label(reduce_hkl(f))] = FACE_PALETTE[i] if i < len(FACE_PALETTE) else OTHER_COLOR
    return out


def view_vector(elev: float, azim: float) -> np.ndarray:
    """Vector unitario hacia la cámara de `ax.view_init(elev, azim)` (cartesianas)."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _light(elev: float, azim: float) -> np.ndarray:
    """Luz desde arriba y a un lado de la cámara (las caras visibles quedan distintas)."""
    return view_vector(min(elev + 35.0, 85.0), azim + 40.0)


def _shade(rgb: np.ndarray, normals: np.ndarray, light: np.ndarray,
           ambient: float = 0.5) -> np.ndarray:
    """Sombreado de Lambert: color · (ambiente + (1 − ambiente) · max(0, n·L))."""
    k = ambient + (1.0 - ambient) * np.clip(normals @ light, 0.0, 1.0)
    return np.clip(rgb * k[:, None], 0.0, 1.0)


def _equal_box(ax, lo: np.ndarray, hi: np.ndarray, zoom: float = 1.0) -> None:
    """Límites y misma escala en x, y, z."""
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)), zoom=zoom)


def _clean_axes(ax, show_axes: bool) -> None:
    """Ejes 3D discretos: sin paneles grises ni rejilla; o sin ejes."""
    if not show_axes:
        ax.set_axis_off()
        return
    for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
        eje.set_pane_color((1, 1, 1, 0))
        eje.line.set_color("#c3c2b7")
    ax.grid(False)
    ax.locator_params(nbins=4)
    ax.tick_params(labelsize=7, colors="#898781")
    ax.set_xlabel("x", fontsize=8, color="#52514e")
    ax.set_ylabel("y", fontsize=8, color="#52514e")
    ax.set_zlabel("z", fontsize=8, color="#52514e")


def draw_triad(ax, cell: Cell, origin: np.ndarray, length: float,
               labels: Sequence[str] = ("a", "b", "c")) -> None:
    """Flechas de los vectores de red a, b, c (orientación de la celda)."""
    for v, lab in zip(cell.matrix, labels):
        d = length * v / np.linalg.norm(v)
        ax.plot(*np.c_[origin, origin + d], color=EDGE_INK, lw=1.4)
        ax.text(*(origin + 1.18 * d), lab, color=EDGE_INK, fontsize=9, fontweight="bold",
                ha="center", va="center")


def _face_legend(ax, faces: Sequence[Sequence[int]], colors: Dict[str, str]) -> None:
    """Leyenda bajo la vista: familias con color propio + "otras caras y aristas"."""
    handles = [Patch(color=colors[hkl_label(reduce_hkl(f))],
                     label="{" + hkl_label(reduce_hkl(f)).strip("()") + "}")
               for f in faces if colors.get(hkl_label(reduce_hkl(f)), OTHER_COLOR) != OTHER_COLOR]
    handles.append(Patch(color=OTHER_COLOR, label="otras caras y aristas"))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 0.04), ncol=min(4, len(handles)), handlelength=1.0,
              columnspacing=0.8)


def exposed_quads(coords: np.ndarray, cell: Cell) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Caras expuestas de los vóxeles (lado ocupado frente a lado vacío).

    Devuelve (quads, unidad, eje): quads (n, 4, 3) en cartesianas, el índice en
    `coords` de la unidad a la que pertenece cada cara y la dirección de índices
    (0..5 = ±i, ±j, ±k). Solo cuentan los 6 vecinos ±1 de índices.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    lo = coords.min(axis=0) - 1
    occ = np.zeros(tuple(coords.max(axis=0) - lo + 2), dtype=bool)
    idx = coords - lo
    occ[tuple(idx.T)] = True
    M = cell.matrix
    quads, unit, eje = [], [], []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        for s in (-1, 1):
            vec = idx.copy()
            vec[:, k] += s
            sel = np.flatnonzero(~occ[tuple(vec.T)])
            if len(sel) == 0:
                continue
            centro = coords[sel].astype(float)
            centro[:, k] += 0.5 * s
            # 4 esquinas de la cara en orden (−,−), (+,−), (+,+), (−,+)
            esquinas = np.repeat(centro[:, None, :], 4, axis=1)
            for q, (di, dj) in enumerate(((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))):
                esquinas[:, q, i] += di
                esquinas[:, q, j] += dj
            quads.append(esquinas @ M)
            unit.append(sel)
            eje.append(np.full(len(sel), 2 * k + (s > 0)))
    return np.concatenate(quads), np.concatenate(unit), np.concatenate(eje)


def render_crystal(coords: np.ndarray, cell: Cell, faces: Optional[Sequence[Sequence[int]]] = None,
                   ax=None, color_by: str = "facet", colors: Optional[Dict[str, str]] = None,
                   radius_cells: float = 2.5, max_angle: float = 20.0,
                   elev: float = 18.0, azim: float = -60.0, ambient: float = 0.5,
                   label_faces: bool = True, min_label_units: int = 15, legend: bool = True,
                   triad: bool = True, show_axes: bool = False, zoom: float = 1.0,
                   max_units: int = 300_000, title: str = ""):
    """Superficie del cristal kMC coloreada por faceta (o por altura) y sombreada.

    Parámetros:
        coords: (N, 3) índices de red (snapshot de `KMC3D.run`)
        faces: familias (hkl) candidatas para `color_by="facet"` (el orden fija los colores)
        color_by: "facet" (familia de cara), "height" (coordenada z, rampa azul) o un color
        colors: {familia: color}; por defecto `face_colors(faces)`
        radius_cells, max_angle: parámetros de `analysis.facet_labels`
        elev, azim: cámara (`ax.view_init`); la luz se coloca respecto a ella
        label_faces: escribe (hkl) sobre cada cara visible con al menos `min_label_units`
        legend: leyenda de las familias presentes (+ "otras / aristas")
        triad: flechas a, b, c
        show_axes: ejes cartesianos (en Å) en vez de la vista limpia
    Devuelve (ax, info): info tiene las etiquetas de faceta (`facet_labels`) si se usaron.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if len(coords) > max_units:
        raise ValueError(f"{len(coords)} unidades > max_units; usa plot_thickness_maps")
    if ax is None:
        fig = plt.figure(figsize=(5.5, 6.0))
        ax = fig.add_subplot(111, projection="3d")
    quads, unit, _ = exposed_quads(coords, cell)
    X = cell.to_cartesian(coords)
    luz = _light(elev, azim)
    info: Dict[str, object] = {}

    # Normal suavizada de cada unidad de superficie (para sombrear y para la faceta)
    if color_by == "facet":
        if faces is None:
            raise ValueError("color_by='facet' necesita `faces`")
        faces = [reduce_hkl(f) for f in faces]
        colors = colors or face_colors(faces)
        lab = facet_labels(coords, cell, faces, radius_cells=radius_cells, max_angle=max_angle)
        info["labels"] = lab
        fam_u = np.full(len(coords), -1)
        sgn_u = np.zeros(len(coords), dtype=int)
        nrm_u = np.zeros((len(coords), 3))
        fam_u[lab["surface"]] = lab["family"]
        sgn_u[lab["surface"]] = lab["sign"]
        nrm_u[lab["surface"]] = lab["normals"]
        paleta = np.array([to_rgb(colors.get(hkl_label(f), OTHER_COLOR)) for f in faces]
                          + [to_rgb(OTHER_COLOR)])
        rgb = paleta[np.where(fam_u[unit] >= 0, fam_u[unit], len(faces))]
    else:
        from .analysis import surface_normals
        surf, normals = surface_normals(coords, cell, radius_cells)
        nrm_u = np.zeros((len(coords), 3))
        nrm_u[surf] = normals
        if color_by == "height":
            z = X[:, 2]
            t = (z - z.min()) / max(np.ptp(z), 1e-9)
            ramp = LinearSegmentedColormap.from_list("azules", SEQ_BLUES[1:])
            rgb = np.array(ramp(t[unit]))[:, :3]
        else:
            rgb = np.tile(np.array(to_rgb(color_by)), (len(unit), 1))
    fc = _shade(rgb, nrm_u[unit], luz, ambient)

    ax.add_collection3d(Poly3DCollection(quads, facecolors=fc, edgecolors="none",
                                         linewidths=0.0))
    pts = quads.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    _equal_box(ax, lo, hi, zoom)
    ax.view_init(elev=elev, azim=azim)
    _clean_axes(ax, show_axes)

    if color_by == "facet":
        camara = view_vector(elev, azim)
        presentes = []
        for i, f in enumerate(faces):
            for s in (1, -1):
                sel = lab["surface"][(lab["family"] == i) & (lab["sign"] == s)]
                if len(sel) == 0:
                    continue
                if i not in presentes:
                    presentes.append(i)
                hkl = tuple(s * x for x in f)
                n = cell.normal(hkl)
                if label_faces and len(sel) >= min_label_units and n @ camara > 0.25:
                    p = X[sel].mean(axis=0) + 0.04 * np.linalg.norm(hi - lo) * n
                    ax.text(*p, hkl_label(hkl), fontsize=8, ha="center", va="center",
                            color="#0b0b0b", zorder=10,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                      alpha=0.75))
        if legend:
            _face_legend(ax, [faces[i] for i in sorted(presentes)], colors)
    if triad:
        draw_triad(ax, cell, lo + 0.04 * (hi - lo), 0.18 * np.linalg.norm(hi - lo) / np.sqrt(3))
    if title:
        ax.set_title(title, fontsize=10)
    return ax, info


def plot_polyhedron(caras: List[Dict[str, object]], faces: Optional[Sequence[Sequence[int]]] = None,
                    ax=None, colors: Optional[Dict[str, str]] = None, elev: float = 18.0,
                    azim: float = -60.0, ambient: float = 0.5, label_faces: bool = True,
                    min_label_fraction: float = 0.01, cell: Optional[Cell] = None,
                    triad: bool = True, show_axes: bool = False, zoom: float = 1.0,
                    legend: bool = True, title: str = ""):
    """Poliedro (p. ej. de Wulff, `analysis.wulff_faces`) con colores por familia.

    `caras` es la salida de `analysis.polyhedron_faces`/`wulff_faces`. `faces` fija el
    orden de colores (el mismo que en `render_crystal`). Las caras visibles con al menos
    `min_label_fraction` del área total llevan su etiqueta (hkl).
    """
    if ax is None:
        fig = plt.figure(figsize=(5.5, 6.0))
        ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        colors = face_colors(faces) if faces is not None else {}
    luz = _light(elev, azim)
    verts = [c["vertices"] for c in caras]
    rgb = np.array([to_rgb(colors.get(c["familia"], OTHER_COLOR)) for c in caras])
    normals = np.array([c["normal"] for c in caras])
    fc = _shade(rgb, normals, luz, ambient)
    ax.add_collection3d(Poly3DCollection(verts, facecolors=fc, edgecolors=EDGE_INK,
                                         linewidths=0.6))
    if legend and faces is not None:
        presentes = {c["familia"] for c in caras}
        _face_legend(ax, [f for f in faces if hkl_label(reduce_hkl(f)) in presentes], colors)
    pts = np.concatenate(verts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    _equal_box(ax, lo, hi, zoom)
    ax.view_init(elev=elev, azim=azim)
    _clean_axes(ax, show_axes)
    if label_faces:
        camara = view_vector(elev, azim)
        total = sum(c["area"] for c in caras)
        for c in caras:
            if c["normal"] @ camara > 0.25 and c["area"] >= min_label_fraction * total:
                p = c["centroid"] + 0.03 * np.linalg.norm(hi - lo) * c["normal"]
                ax.text(*p, hkl_label(c["hkl"]), fontsize=8, ha="center", va="center",
                        color="#0b0b0b", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    if triad and cell is not None:
        draw_triad(ax, cell, lo + 0.04 * (hi - lo), 0.18 * np.linalg.norm(hi - lo) / np.sqrt(3))
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def lattice_basis_along(t: Sequence[int], search: int = 2) -> np.ndarray:
    """Base entera unimodular (filas u, v, t) con t = [uvw] dado como tercera fila.

    En esa base, r = a·u + b·v + c·t: (a, b) es la COLUMNA de la red a lo largo de t.
    Busca u, v cortos con componentes en [−search, search].
    """
    t = np.asarray(reduce_hkl(t))
    cands = [np.array(v) for v in np.ndindex(*(2 * search + 1,) * 3)]
    cands = [v - search for v in cands if np.any(v - search)]
    cands.sort(key=lambda v: (np.abs(v).sum(), tuple(v)))
    for i, u in enumerate(cands):
        for v in cands[i + 1:]:
            if abs(round(np.linalg.det(np.array([u, v, t])))) == 1:
                return np.array([u, v, t])
    raise ValueError(f"No se encontró base para [{t}]")


def thickness_map(coords: np.ndarray, cell: Cell, direction: Sequence[int],
                  up: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Espesor del cristal a lo largo de la dirección de red `direction` (vista desde +t).

    Cada columna (a, b) de `lattice_basis_along` es una fila de sitios paralela a t: su
    espesor es (unidades en la columna) · |t|. Las esquinas de los píxeles son las de la
    columna proyectada sobre el plano ⟂ t, en una base ortonormal (e1, e2) con e2 a lo
    largo de la proyección de `up` (vector de red; por defecto el eje de red más
    perpendicular a t) y e1 × e2 = t̂ (vista desde fuera, como una imagen).

    Devuelve dict con `thickness` (na, nb) (NaN = columna vacía) y las esquinas X, Y
    (na + 1, nb + 1) para `pcolormesh`.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    B = lattice_basis_along(direction)
    M = cell.matrix
    t_cart = B[2] @ M
    t_hat = t_cart / np.linalg.norm(t_cart)
    if up is None:
        ejes = np.eye(3, dtype=int)
        up = ejes[np.argmin([abs((e @ M) @ t_hat) / np.linalg.norm(e @ M) for e in ejes])]
    up_c = np.asarray(up, float) @ M
    e2 = up_c - (up_c @ t_hat) * t_hat
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, t_hat)
    abc = np.rint(coords @ np.linalg.inv(B)).astype(np.int64)
    a, b = abc[:, 0], abc[:, 1]
    a0, b0 = a.min(), b.min()
    cnt = np.zeros((a.max() - a0 + 1, b.max() - b0 + 1))
    np.add.at(cnt, (a - a0, b - b0), 1.0)
    thick = np.where(cnt > 0, cnt * np.linalg.norm(t_cart), np.nan)
    ga, gb = np.meshgrid(np.arange(cnt.shape[0] + 1) + a0 - 0.5,
                         np.arange(cnt.shape[1] + 1) + b0 - 0.5, indexing="ij")
    P = ga[..., None] * (B[0] @ M) + gb[..., None] * (B[1] @ M)
    return {"thickness": thick, "X": P @ e1, "Y": P @ e2, "basis": B}


def plot_thickness_maps(coords: np.ndarray, cell: Cell,
                        directions: Sequence[Sequence[int]] = ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
                        ups: Optional[Sequence[Optional[Sequence[int]]]] = None, axes=None,
                        same_scale: bool = True, title: str = ""):
    """Proyecciones de espesor (tipo TEM) a lo largo de varias direcciones de red.

    Sustituye a las proyecciones de puntos semitransparentes (`viz.plot_projections`),
    que dan moiré porque los píxeles no coinciden con las columnas de la red. Cada panel
    tiene su barra de color (el espesor a lo largo de cada dirección es distinto); con
    `same_scale` los paneles comparten la extensión en Å, así que los tamaños se pueden
    comparar entre vistas.
    """
    directions = [tuple(d) for d in directions]
    ups = ups or [None] * len(directions)
    if axes is None:
        fig, axes = plt.subplots(1, len(directions), figsize=(3.9 * len(directions), 4.6),
                                 constrained_layout=True)
    axes = np.atleast_1d(axes)
    ramp = LinearSegmentedColormap.from_list("espesor", SEQ_BLUES).with_extremes(bad=(1, 1, 1, 0))
    mapas = [thickness_map(coords, cell, d, u) for d, u in zip(directions, ups)]
    semi = max(max(np.ptp(m["X"]), np.ptp(m["Y"])) for m in mapas) / 2 * 1.04
    for ax, d, m in zip(axes, directions, mapas):
        pc = ax.pcolormesh(m["X"], m["Y"], np.ma.masked_invalid(m["thickness"]), cmap=ramp,
                           shading="flat", rasterized=True, vmin=0)
        ax.set_aspect("equal")
        ax.grid(False)
        if same_scale:
            cx, cy = (m["X"].max() + m["X"].min()) / 2, (m["Y"].max() + m["Y"].min()) / 2
            ax.set_xlim(cx - semi, cx + semi)
            ax.set_ylim(cy - semi, cy + semi)
        ax.set_title("vista según [" + "".join(str(x) for x in d) + "]", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Å", fontsize=8)
        cb = ax.figure.colorbar(pc, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("espesor (Å)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if title:
        axes[0].figure.suptitle(title)
    return axes
