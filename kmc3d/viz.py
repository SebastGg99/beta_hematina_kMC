"""Figuras de cristales 3D (solo matplotlib).

- `plot_voxels`: vóxeles con la forma REAL de la celda (paralelepípedos triclínicos),
  para cristales pequeños (hasta unas decenas de miles de unidades).
- `plot_surface_points`: nube de las unidades de superficie, para cristales grandes.
- `plot_projections`: siluetas en los tres planos cartesianos (barato, cualquier tamaño).
- `plot_face_series`: posición de cada cara frente al tiempo.
- `plot_face_height_map`: mapa de alturas de una cara (ver analysis.face_height_map).
- `plot_face_maps`: modo por caras, un mapa 2D por cara (hkl) visto desde fuera, con
  la geometría real del plano y las terrazas de cada cara.
- `animate_growth_3d`: modo dinámico SIN WebGL, una animación del crecimiento con un
  fotograma por snapshot (color: tiempo de llegada, altura o faceta). En un notebook, `HTML(anim.to_jshtml())` da un reproductor
  (▶, deslizador, velocidad) que funciona en cualquier visor.

`render.py` tiene las vistas de morfología: superficie coloreada por faceta, poliedro de
Wulff y proyecciones de espesor. `viz_interactive.py` tiene la versión con plotly (rotación con el ratón), que necesita
plotly y un visor con WebGL. Este módulo solo necesita matplotlib.

Colores de las familias de caras: paleta fija por orden de aparición (nunca cíclica
dentro de una figura) + marcadores distintos para no depender solo del color.
"""

from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize,
                               to_rgb)

from .analysis import face_depth_map
from .geometry import Cell, hkl_label, reduce_hkl

Snapshot = Tuple[float, np.ndarray]

# Rampa secuencial de un solo tono para magnitudes ordenadas (claro = bajo, oscuro =
# alto). Empieza en un azul medio, no casi blanco, para que las unidades se vean sobre
# el fondo claro. La comparten la animación y la versión con plotly.
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# Paleta categórica fija (orden estable; la de matplotlib/MATLAB que usa el proyecto)
# y marcadores, para que la identidad de cada serie no dependa solo del color
PALETTE = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE",
           "#A2142F", "#555555", "#B8860B"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
_PALETTE, _MARKERS = PALETTE, MARKERS


def _set_equal_3d(ax, X: np.ndarray) -> None:
    """Ejes 3D con la misma escala en x, y, z."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))


def plot_voxels(coords: np.ndarray, cell: Cell = Cell(), ax=None, color_by: str = "height",
                axis: int = 2, cmap: str = "viridis", edgecolor: str = "k",
                linewidth: float = 0.15, max_units: int = 60_000, title: str = ""):
    """Vóxeles con la geometría real de la celda.

    color_by: "height" (posición a lo largo del eje cartesiano `axis`) o un color fijo.
    """
    coords = np.asarray(coords, int)
    if len(coords) > max_units:
        raise ValueError(f"{len(coords)} unidades: usa plot_surface_points o plot_projections")
    if ax is None:
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    lo = coords.min(axis=0)
    shape = coords.max(axis=0) - lo + 1
    filled = np.zeros(shape, dtype=bool)
    filled[tuple((coords - lo).T)] = True
    # esquinas de los vóxeles en índices (semienteros) -> cartesianas con la celda
    gi, gj, gk = np.meshgrid(*(np.arange(n + 1) + l - 0.5 for n, l in zip(shape, lo)), indexing="ij")
    corners = np.stack([gi, gj, gk], -1) @ cell.matrix
    x, y, z = corners[..., 0], corners[..., 1], corners[..., 2]
    if color_by == "height":
        centers = (np.stack(np.meshgrid(*(np.arange(n) + l for n, l in zip(shape, lo)),
                                        indexing="ij"), -1) @ cell.matrix)[..., axis]
        norm = (centers - centers[filled].min()) / max(np.ptp(centers[filled]), 1e-9)
        colors = plt.get_cmap(cmap)(norm)
    else:
        colors = np.empty(shape, dtype=object)
        colors[...] = color_by
    ax.voxels(x, y, z, filled, facecolors=colors, edgecolor=edgecolor, linewidth=linewidth)
    _set_equal_3d(ax, corners.reshape(-1, 3))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    return ax


def plot_surface_points(coords: np.ndarray, cell: Cell = Cell(), ax=None, axis: int = 2,
                        cmap: str = "viridis", s: float = 2.0, title: str = ""):
    """Nube 3D de las unidades de superficie (coloreadas por altura)."""
    coords = np.asarray(coords, int)
    # superficie: unidades con algún vecino de primer orden vacío (red cúbica de índices)
    occ = {tuple(c) for c in coords.tolist()}
    steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    surf = np.array([c for c in coords.tolist()
                     if any((c[0] + d[0], c[1] + d[1], c[2] + d[2]) not in occ for d in steps)])
    X = cell.to_cartesian(surf)
    if ax is None:
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, axis], cmap=cmap, s=s, depthshade=True)
    _set_equal_3d(ax, X)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    return ax


def plot_projections(coords: np.ndarray, cell: Cell = Cell(), axes=None, alpha: float = 0.03,
                     size: float = 2.0, color: str = "k", title: str = ""):
    """Siluetas en los planos xy, xz e yz cartesianos.

    Cada unidad es un punto semitransparente: el gris acumulado es proporcional al
    espesor del cristal en esa dirección (evita el moiré de un histograma 2D cuyos
    bins no coinciden con el espaciado de la red).
    """
    X = cell.to_cartesian(coords)
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pairs = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    span = np.ptp(X, axis=0).max() / 2 + 1.0
    mid = (X.max(axis=0) + X.min(axis=0)) / 2
    for ax, (i, j, li, lj) in zip(axes, pairs):
        ax.scatter(X[:, i], X[:, j], s=size, c=color, alpha=alpha, linewidths=0, rasterized=True)
        ax.set_xlim(mid[i] - span, mid[i] + span)
        ax.set_ylim(mid[j] - span, mid[j] + span)
        ax.set_aspect("equal")
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        ax.set_title(f"proyección {li}{lj}")
    if title:
        axes[0].figure.suptitle(title)
    return axes


def plot_face_series(series: pd.DataFrame, ax=None, x: str = "t", families: Optional[Sequence[str]] = None,
                     title: str = ""):
    """D(t) de cada cara; ±(hkl) de la misma familia con el mismo color.

    `series` es la salida de `analysis.face_series`. Línea continua = +(hkl),
    discontinua = −(hkl).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    fams = list(dict.fromkeys(series["familia"])) if families is None else list(families)
    for i, fam in enumerate(fams):
        g = series[series["familia"] == fam]
        color, marker = _PALETTE[i % len(_PALETTE)], _MARKERS[i % len(_MARKERS)]
        for cara, gc in g.groupby("cara", sort=False):
            plus = cara == fam
            ax.plot(gc[x], gc["D"], color=color, linestyle="-" if plus else "--",
                    linewidth=1.6 if plus else 1.1,
                    marker=marker if plus else None, markevery=max(1, len(gc) // 8), markersize=4,
                    label=f"{{{fam.strip('()')}}}" if plus else None)
    ax.set_xlabel("tiempo" if x == "t" else x)
    ax.set_ylabel("distancia de la cara al centro D")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    if title:
        ax.set_title(title)
    return ax


def plot_face_height_map(hmap: np.ndarray, ax=None, cmap: str = "terrain", title: str = ""):
    """Mapa de alturas de una cara (imagen 2D)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(hmap.T, origin="lower", cmap=cmap, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, label="altura (planos)")
    ax.set_xlabel("columna a")
    ax.set_ylabel("columna b")
    if title:
        ax.set_title(title)
    return ax


def surface_mask(coords: np.ndarray) -> np.ndarray:
    """True para las unidades con algún vecino vacío entre los 6 (±1 en i, j o k).

    Es la misma definición de superficie que `plot_surface_points`, pero vectorizada con
    un array de ocupación. Las unidades interiores no se ven, así que no hace falta
    dibujarlas.
    """
    coords = np.asarray(coords, int)
    lo = coords.min(axis=0) - 1                   # margen de 1 para los vecinos del borde
    occ = np.zeros(coords.max(axis=0) - lo + 2, dtype=bool)
    idx = coords - lo
    occ[tuple(idx.T)] = True
    lleno = np.ones(len(coords), dtype=bool)
    for eje in range(3):
        for paso in (-1, 1):
            vecino = idx.copy()
            vecino[:, eje] += paso
            lleno &= occ[tuple(vecino.T)]
    return ~lleno


def arrival_times(snaps: Sequence[Snapshot]) -> List[np.ndarray]:
    """Tiempo del primer snapshot en el que aparece cada unidad, para cada snapshot.

    Devuelve una lista alineada con `snaps`: para el snapshot k, un array (N_k,) con el
    tiempo de "llegada" de cada una de sus unidades. Una unidad que se desprende y
    vuelve conserva su primera llegada.
    """
    lo = np.min([np.asarray(c).min(axis=0) for _, c in snaps], axis=0)
    hi = np.max([np.asarray(c).max(axis=0) for _, c in snaps], axis=0)
    dims = tuple(int(x) for x in hi - lo + 1)
    first = np.full(int(np.prod(dims)), np.nan)   # tiempo de llegada por sitio (aplanado)
    out = []
    for t, c in snaps:
        key = np.ravel_multi_index(tuple((np.asarray(c, int) - lo).T), dims)
        nuevo = np.isnan(first[key])
        first[key[nuevo]] = t
        out.append(first[key].copy())
    return out


def frame_indices(n: int, max_frames: Optional[int]) -> List[int]:
    """Snapshots que se dibujan: todos, o `max_frames` equiespaciados (incluye el último)."""
    if max_frames is None or n <= max_frames:
        return list(range(n))
    return np.unique(np.rint(np.linspace(0, n - 1, max_frames)).astype(int)).tolist()


def animate_growth_3d(snaps: Union[Sequence[Snapshot], np.ndarray], cell: Cell = Cell(),
                      color_by: str = "arrival", axis: int = 2,
                      faces: Optional[Sequence[Sequence[int]]] = None,
                      max_frames: Optional[int] = None, surface_only: bool = True,
                      s: float = 4.0, cmap=None, elev: float = 15.0, azim: float = -70.0,
                      azim_step: float = 0.0, interval: int = 300,
                      figsize: Tuple[float, float] = (6.0, 6.0), dpi: int = 80,
                      zoom: float = 1.0, title: str = "") -> FuncAnimation:
    """Animación 3D del crecimiento con matplotlib (sin WebGL): un fotograma por snapshot.

    Es el modo dinámico para visores sin WebGL, donde las figuras 3D de plotly no se
    dibujan. En un notebook:

        from IPython.display import HTML
        HTML(anim.to_jshtml())      # reproductor con ▶ / ❚❚, deslizador y velocidad

    `anim.save("x.gif")` o `anim.save("x.mp4")` lo guardan como vídeo.

    Parámetros:
        snaps: lista (t, coords) de `KMC3D.run` o un único array de coordenadas (N, 3)
        cell: celda real (las coordenadas se pasan a cartesianas con `cell.matrix`)
        color_by: "arrival" (tiempo de llegada de cada unidad: se ve qué partes
            crecieron cuándo), "height" (coordenada cartesiana `axis`) o "facet"
            (familia de cara de cada unidad de superficie, con los colores y el
            sombreado de `render.render_crystal`; necesita `faces`)
        faces: familias (hkl) para color_by="facet" (el orden fija los colores)
        max_frames: número máximo de fotogramas (equiespaciados); None = todos
        surface_only: dibuja solo las unidades de superficie (`surface_mask`)
        s: tamaño de los puntos
        elev, azim: cámara inicial (como `ax.view_init`)
        azim_step: grados que gira la cámara en cada fotograma (0 = cámara fija). Un
            giro lento ayuda a percibir la forma 3D.
        interval: milisegundos por fotograma al reproducir
        figsize, dpi: tamaño de cada fotograma (para un cristal alargado en z conviene
            una figura alta, p. ej. (5, 7))
        zoom: acercamiento de la caja 3D dentro de la figura

    Los ejes, la escala de color y la cámara son comunes a todos los fotogramas. La
    figura se cierra antes de devolver la animación, para que el notebook no muestre
    además una imagen estática.
    """
    if isinstance(snaps, np.ndarray):
        snaps = [(0.0, snaps)]
    snaps = [(float(t), np.asarray(c, int)) for t, c in snaps]
    if color_by not in ("arrival", "height", "facet"):
        raise ValueError("color_by debe ser 'arrival', 'height' o 'facet'")
    if color_by == "facet" and faces is None:
        raise ValueError("color_by='facet' necesita `faces`")
    llegada = arrival_times(snaps) if color_by == "arrival" else None
    if color_by == "facet":
        from . import render
        from .analysis import facet_labels
        caras = [reduce_hkl(f) for f in faces]
        colores = render.face_colors(caras)
        paleta = np.array([to_rgb(colores[hkl_label(f)]) for f in caras]
                          + [to_rgb(render.OTHER_COLOR)])
        luz = render._light(elev, azim)

    # Datos de cada fotograma: cartesianas de las unidades visibles y su color
    datos = []
    for k in frame_indices(len(snaps), max_frames):
        t, c = snaps[k]
        if color_by == "facet":
            # superficie = la de facet_labels; color de la familia sombreado con la normal
            lab = facet_labels(c, cell, caras)
            X = cell.to_cartesian(c[lab["surface"]])
            fam = np.where(lab["family"] >= 0, lab["family"], len(caras))
            col = render._shade(paleta[fam], lab["normals"], luz)
        else:
            sel = surface_mask(c) if surface_only else np.ones(len(c), dtype=bool)
            X = cell.to_cartesian(c[sel])
            col = llegada[k][sel] if llegada is not None else X[:, axis]
        datos.append((k, t, len(c), X, col))

    # Ejes y escala de color comunes a todos los fotogramas
    lo = np.min([X.min(axis=0) for *_, X, _ in datos], axis=0)
    hi = np.max([X.max(axis=0) for *_, X, _ in datos], axis=0)
    if color_by != "facet":
        norm = Normalize(min(col.min() for *_, col in datos), max(col.max() for *_, col in datos))
    cmap = cmap or LinearSegmentedColormap.from_list("azules", BLUES)

    # Ejes 3D colocados a mano: add_subplot deja márgenes muy grandes alrededor de la caja
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 0.82, 0.9], projection="3d")
    cax = fig.add_axes([0.84, 0.3, 0.03, 0.4])
    rotulo = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=9)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)), zoom=zoom)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.locator_params(nbins=4)                     # pocas marcas: no se solapan en ejes cortos
    ax.tick_params(labelsize=7)
    if color_by == "facet":
        cax.set_visible(False)
        from matplotlib.patches import Patch
        handles = [Patch(color=colores[hkl_label(f)], label="{" + hkl_label(f).strip("()") + "}")
                   for f in caras if colores[hkl_label(f)] != render.OTHER_COLOR]
        handles.append(Patch(color=render.OTHER_COLOR, label="otras caras y aristas"))
        fig.legend(handles=handles, loc="center right", fontsize=7, frameon=False)
    else:
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                     label="t de llegada" if color_by == "arrival" else "xyz"[axis])
    nube = [None]                                  # scatter del fotograma actual

    def dibujar(i):
        # Sustituye la nube de puntos y actualiza cámara y título
        k, t, n, X, col = datos[i]
        if nube[0] is not None:
            nube[0].remove()
        if color_by == "facet":
            nube[0] = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=col, s=s, linewidths=0,
                                 depthshade=False)
        else:
            nube[0] = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=col, cmap=cmap, norm=norm, s=s,
                                 linewidths=0, depthshade=True)
        ax.view_init(elev=elev, azim=azim + i * azim_step)
        cabecera = f"{title}\n" if title else ""
        rotulo.set_text(f"{cabecera}snapshot {k}/{len(snaps) - 1} · t = {t:.4g} · N = {n}")
        return (nube[0],)

    anim = FuncAnimation(fig, dibujar, frames=len(datos), interval=interval, blit=False)
    plt.close(fig)
    return anim


def face_map_geometry(coords: np.ndarray, hkl: Sequence[int], cell: Cell = Cell()):
    """Mapa de profundidad de la cara (hkl) y las esquinas de sus columnas en el plano.

    Las columnas (a, b) de `analysis.face_depth_map` son paralelogramos de lados
    U = u·A y V = v·A (vectores cartesianos contenidos en el plano (hkl)). Se proyectan
    sobre una base ortonormal del plano, e1 = U/|U| y e2 = n × e1, con n la normal
    exterior: (e1, e2, n) es directa, así que el mapa es la cara VISTA DESDE FUERA.

    Devuelve (depth, X, Y): depth (na, nb) y las esquinas X, Y (na + 1, nb + 1) en las
    unidades de la celda (Å para β-hematina), listas para `pcolormesh`.
    """
    info = face_depth_map(coords, hkl)
    B = info["frame"]
    U, V = B[0] @ cell.matrix, B[1] @ cell.matrix
    n = cell.normal(hkl)
    e1 = U / np.linalg.norm(U)
    e2 = np.cross(n, e1)
    na, nb = info["depth"].shape
    # Esquinas de las columnas: índices semienteros (a ± 1/2, b ± 1/2)
    ga, gb = np.meshgrid(np.arange(na + 1) + info["a0"] - 0.5,
                         np.arange(nb + 1) + info["b0"] - 0.5, indexing="ij")
    P = ga[..., None] * U + gb[..., None] * V
    return info["depth"], P @ e1, P @ e2


def face_relative_height(depth: np.ndarray, band: int = 3) -> np.ndarray:
    """Altura de cada columna respecto a la TERRAZA PRINCIPAL de la cara (capas).

    La terraza principal es el nivel con más columnas entre las `band + 1` capas más
    externas (profundidad <= band). Un átomo suelto por encima de la cara no cambia la
    referencia. Devuelve terraza − profundidad: +1, +2 = islas y adátomos por encima,
    0 = terraza principal, −1, −2 = huecos y terrazas inferiores. NaN en las columnas
    vacías.
    """
    arriba = depth[depth <= band].astype(int)
    terraza = int(np.bincount(arriba).argmax())
    return terraza - depth


def plot_face_maps(coords: np.ndarray, cell: Cell, faces: Sequence[Sequence[int]],
                   band: int = 3, include_opposite: bool = False, ncols: int = 3,
                   same_scale: bool = True, cmap: str = "viridis",
                   other_color: str = "#bdbcb7", axes=None, title: str = ""):
    """Modo por caras: un mapa 2D de cada cara (hkl), vista desde fuera del cristal.

    Cada columna de la cara (ver `analysis.face_depth_map`) se colorea por su altura
    respecto a la terraza principal (`face_relative_height`), en capas: 0 = terraza,
    +1, +2 = islas por encima, −1 … −band = huecos y terrazas inferiores. Las columnas
    más de `band` capas por debajo de la terraza pertenecen a aristas o a otras caras y
    se dibujan en gris (`other_color`); las columnas vacías quedan en blanco. Usa la
    geometría real del plano (Å para β-hematina), así que las caras se pueden comparar
    en tamaño.

    Parámetros:
        faces: caras (hkl) a dibujar
        band: capas por debajo de la terraza que se consideran parte de la cara
        include_opposite: añade también la cara opuesta (−h −k −l) de cada una
        same_scale: misma extensión en todos los paneles (compara tamaños de cara)
        axes: array de ejes ya creado (uno por cara), opcional

    En el título de cada panel: número de columnas de la cara y fracción de ellas que
    está en la terraza principal.
    """
    caras = [reduce_hkl(f) for f in faces]
    if include_opposite:
        caras = [g for f in caras for g in (f, tuple(-x for x in f))]

    mapas, centros, semiancho, tope = [], [], 0.0, 0
    for f in caras:
        depth, X, Y = face_map_geometry(coords, f, cell)
        rel = face_relative_height(depth, band)
        cara = rel >= -band                           # False también en las vacías (NaN)
        # Centro y extensión de la cara (centros de las columnas que pertenecen a ella)
        cx = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
        cy = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
        centros.append((cx[cara].mean(), cy[cara].mean()))
        semiancho = max(semiancho, np.ptp(cx[cara]) / 2, np.ptp(cy[cara]) / 2)
        tope = max(tope, int(np.nanmax(rel)))
        mapas.append((f, X, Y, rel, cara))

    # Colores discretos: un color por nivel (−band … +tope, rampa ordenada de bajo a
    # alto) y gris para "aristas, otras caras" (código −band − 1)
    niveles = np.arange(-band, tope + 1)
    colores = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, len(niveles)))
    lcmap = ListedColormap([other_color] + list(colores))
    lcmap.set_bad((1, 1, 1, 0))                       # columnas vacías: transparentes
    norm = BoundaryNorm(np.arange(-band - 1.5, tope + 1), lcmap.N)

    if axes is None:
        nrows = int(np.ceil(len(caras) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                                 constrained_layout=True, squeeze=False)
    axes = np.atleast_1d(axes).ravel()
    fig = axes[0].figure

    for ax, (f, X, Y, rel, cara), centro in zip(axes, mapas, centros):
        vis = np.where(rel < -band, -band - 1, rel)   # NaN (vacías) se mantiene
        ax.pcolormesh(X, Y, np.ma.masked_invalid(vis), cmap=lcmap, norm=norm,
                      shading="flat", rasterized=True)
        ax.set_aspect("equal")
        ax.grid(False)
        if same_scale:
            m = 1.08 * semiancho + 1.0
            ax.set_xlim(centro[0] - m, centro[0] + m)
            ax.set_ylim(centro[1] - m, centro[1] + m)
        terraza = float(np.mean(rel[cara] == 0))
        ax.set_title(f"{hkl_label(f)}: {int(cara.sum())} columnas, terraza {terraza:.0%}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(mapas):]:
        ax.set_visible(False)

    etiquetas = [f"< −{band} (aristas, otras caras)"] + [
        "0 (terraza principal)" if v == 0 else f"{v:+d}" for v in niveles]
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=lcmap), ax=axes[:len(mapas)].tolist(),
                      ticks=np.arange(-band - 1, tope + 1), shrink=0.5)
    cb.ax.set_yticklabels(etiquetas, fontsize=8)
    cb.set_label("altura respecto a la terraza principal (capas)")
    if title:
        fig.suptitle(title)
    return axes
