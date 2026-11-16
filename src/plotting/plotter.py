"""Plotter unificado para ambas líneas del motor.

Fusiona tres clases que tenían firmas idénticas pero estilos y métodos distintos:
- `src/plotter.py`      -> style="classic"  (Plotter)     + plot_conversion
- `plotter_v2.py`       -> style="v2"       (Plotter_v2)  + métodos de análisis
  (estaba en .descartables pero lo usaban los 4 pRun*.py y 2 notebooks)
- `src/plotter_v3.py`   -> style="academic" (Plotter_v3)

Los métodos de análisis y `plot_conversion` están disponibles con cualquier estilo.

Cambios respecto a los originales:
- Ya no importa `bkl` (el motor dinámico) para anotar tipos: eso acoplaba la línea
  estática a la dinámica. Se usa un `Protocol` con la interfaz mínima.
- El renderizado 3D/GIF de cada estilo reproduce el original; lo comprueba
  `tests/test_plotter.py` comparando PNG con los plotters originales.
- `plt.cm.get_cmap` (eliminado en matplotlib >= 3.9) se sustituye por
  `matplotlib.colormaps[...]`, que devuelve el mismo colormap.
- style="academic" con mode="surface" antes producía una figura vacía sin avisar;
  ahora lanza ValueError.
- `plot_growth_rate_analysis` mostraba `p.E_pb_over_kT` y `p.delta`, que no existen
  en los parámetros estáticos (x/y): ahora muestra el valor escalar o el par x/y.

Estilo nuevo (sin original de referencia):
- style="paper": morfología como en la Fig. 9 de Nagpal et al. (2024),
  `references/modern_kMC/`. Azul = capas completamente llenas (niveles por debajo
  de min(heights)); rojo = capa(s) en crecimiento (todo lo que está por encima).
  Se dibuja solo una losa fina (`base_layers` capas azules bajo el frente de
  crecimiento), sin ejes. `plot_morphology_sequence` compone varios snapshots en
  fila, como la figura del artículo.
- style="relief": variante de "paper" pensada para distinguir islas apiladas. En
  "paper" todo lo que está por encima de min(heights) es del mismo rojo, así que
  una isla sobre otra no se ve. Aquí cada nivel tiene su propio color, en un
  degradado secuencial de claro (capa completa, h_min) a oscuro (nivel más alto),
  con una barra de color discreta en vez de leyenda. La paleta se elige con
  `Plotter(..., palette=...)` o con el argumento `cmap` de cada método: por defecto
  "ylgnbu" (amarillo pálido -> azul marino) y, más sobrias con el mismo contraste,
  "pizarra", "arena", "salvia", "ciruela" y "grises" (ver `RELIEF_PALETTES`);
  `compare_palettes` las muestra lado a lado sobre un snapshot. Los vóxeles
  no tienen aristas, se iluminan con luz direccional, el relieve vertical se
  exagera (`z_scale`) y el borde superior de cada escalón se traza como una línea
  oscura (el contorno de cada isla).
  `plot_surface_2d` da la vista cenital 2D equivalente (disponible con cualquier
  estilo): mismo color por nivel, bisel sombreado en los escalones y curvas de nivel
  en cada borde de isla. `plot_crystal_views` pone la vista 3D y la 2D lado a lado,
  y `crystal_growth_gif(view=...)` / `plot_morphology_sequence(show_map=...)`
  admiten la vista 2D con este estilo.
  Los colores son relativos a h_min = min(heights) de cada snapshot (como la "capa
  completa" de "paper"): en GIF y secuencias el número de niveles de la paleta es
  común a todos los frames, pero h_min cambia, así que la barra se rotula
  "h_min + k" y cada panel indica su h_min.
"""

from typing import Dict, List, Optional, Protocol, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (BoundaryNorm, Colormap, LightSource, LinearSegmentedColormap,
                               ListedColormap, to_rgba)
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra la proyección 3D en mpl antiguos)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

Snapshot = Tuple[float, np.ndarray, float]

STYLES = ("classic", "v2", "academic", "paper", "relief")

# Estilos que solo admiten mode="voxel"
_VOXEL_ONLY_STYLES = ("academic", "paper", "relief")

# ---- Paleta del estilo "paper" (Fig. 9 de Nagpal et al. 2024) ----
# Azul verdoso: capa completamente llena. Rojo oscuro: capa en crecimiento.
# Las aristas usan un tono más oscuro del mismo color (la rejilla fina del artículo).
_PAPER_FULL_FACE = "#2B8A96"
_PAPER_FULL_EDGE = "#14464D"
_PAPER_GROW_FACE = "#8B1212"
_PAPER_GROW_EDGE = "#470707"

# Valores por defecto de cada estilo, tomados de las firmas originales.
_STYLE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "classic": dict(elev=45, cmap="viridis", gif_mode="voxel", gif_interval_ms=100,
                    gif_title_prefix="Crystal growth"),
    "v2": dict(elev=30, cmap="terrain", gif_mode="surface", gif_interval_ms=200,
               gif_title_prefix="Crecimiento cristalino"),
    "academic": dict(elev=45, cmap="hemozoina", gif_mode="voxel", gif_interval_ms=100,
                     gif_title_prefix="Crystal Growth"),
    # cmap no se usa en "paper" (colores fijos por capa)
    "paper": dict(elev=35, cmap=None, gif_mode="voxel", gif_interval_ms=150,
                  gif_title_prefix="Crecimiento cristalino"),
    # cmap: paleta de niveles (h_min toma el extremo claro, el nivel más alto el oscuro)
    "relief": dict(elev=35, cmap="ylgnbu", gif_mode="voxel", gif_interval_ms=150,
                   gif_title_prefix="Crecimiento cristalino"),
}

# ---- Estilo "relief" ----
# Paletas de niveles con nombre: nombre -> (colormap de matplotlib o lista de colores
# de claro a oscuro, tramo (lo, hi) del colormap que se usa). Todas van de claro
# (capa completa, h_min) a oscuro (nivel más alto) con luminosidad monótona, de
# OKLab L ~ 96 a L ~ 31-34: niveles vecinos se distinguen por claridad aunque el tono
# cambie poco, y el contraste total es el mismo en todas. Cambian el matiz y la
# saturación: "ylgnbu" tiene un croma máximo de ~14 (OKLab x100) y las demás de 0 a 6.
# Con 6 niveles, la diferencia mínima entre niveles vecinos es de >= 8 (OKLab x100)
# en todas.
RELIEF_PALETTES: Dict[str, Tuple[object, Tuple[float, float]]] = {
    # Por defecto: amarillo pálido -> verde agua -> azul marino. Se recortan los
    # extremos de YlGnBu (casi blanco, que se confunde con el fondo, y casi negro,
    # que oculta el sombreado). Se eligió tras comparar viridis, cividis, plasma y GnBu.
    "ylgnbu": ("YlGnBu", (0.08, 0.95)),
    # Sobrias (poco saturadas), definidas como degradados entre 6 colores:
    # gris azulado
    "pizarra": (["#F1F2F0", "#CDD5DA", "#9DAEBB", "#6C8599", "#43607A", "#243A50"],
                (0.0, 1.0)),
    # arena -> verde grisáceo -> azul petróleo
    "arena": (["#F4EFE3", "#DDD5BC", "#B3B79E", "#7F978B", "#4F6E73", "#263F4A"],
              (0.0, 1.0)),
    # verdes apagados
    "salvia": (["#F2F1EA", "#D0DAC7", "#A2B9A1", "#73957F", "#4A6F61", "#223D37"],
               (0.0, 1.0)),
    # malvas apagados
    "ciruela": (["#F4F0F1", "#DCCFD8", "#B9A3B9", "#8F7698", "#634F74", "#3B2F4C"],
                (0.0, 1.0)),
    # escala de grises (impresión en blanco y negro)
    "grises": ("Greys", (0.04, 0.85)),
}
# Luz de los vóxeles 3D (coordenadas de datos): alta para que las caras superiores
# conserven su color y las paredes de los escalones queden en sombra.
_RELIEF_LIGHT_3D = dict(azdeg=225, altdeg=55)
# Luz del mapa 2D (convención cartográfica: desde el noroeste de la imagen).
_RELIEF_LIGHT_2D = dict(azdeg=315, altdeg=40)
# Contorno de escalones/islas: negro semitransparente, para no saturar superficies rugosas
_RELIEF_STEP_COLOR = (0.0, 0.0, 0.0, 0.6)
# Exageración vertical por defecto de la vista 3D: con vóxeles cúbicos, unos pocos
# niveles sobre una red de 80x80 apenas se ven
_RELIEF_Z_SCALE = 2.0
# Vistas que admite crystal_growth_gif con el estilo "relief"
_RELIEF_VIEWS = ("3d", "2d", "both")
# Máximo de etiquetas en la barra de color de niveles
_RELIEF_MAX_TICKS = 12


class KMCLike(Protocol):
    """Interfaz mínima que el plotter necesita de un motor (cualquier línea)."""

    lat: object          # con atributo `heights` (np.ndarray 2D)
    t: float
    p: object            # parámetros (se usa `T` y, en análisis, energías/deltas)

    @property
    def conversion_percent(self) -> float: ...


def _get_cmap(name: str):
    """Obtiene un colormap por nombre (reemplazo de plt.cm.get_cmap)."""
    return matplotlib.colormaps[name]


def _column_voxels(heights: np.ndarray, depth: int) -> np.ndarray:
    """Matriz booleana (Lx, Ly, depth) con las columnas llenas hasta su altura."""
    Lx, Ly = heights.shape
    voxels = np.zeros((Lx, Ly, depth), dtype=bool)
    for i in range(Lx):
        for j in range(Ly):
            h = int(heights[i, j])
            if h > 0:
                voxels[i, j, :h] = True
    return voxels


def _paper_window_depth(heights: np.ndarray, base_layers: int) -> int:
    """Número de capas de vóxeles que necesita el estilo 'paper' para `heights`."""
    h = np.asarray(heights)
    return int(h.max()) - int(h.min()) + int(base_layers)


def _paper_voxels(heights: np.ndarray, base_layers: int = 1,
                  depth: Optional[int] = None):
    """Vóxeles y colores del estilo 'paper'.

    Criterio del artículo: una capa está "completamente llena" si todas las columnas
    llegan a ella, es decir, los niveles z < h_full con h_full = min(heights). Lo
    que queda por encima es la capa en crecimiento (rojo).

    La ventana vertical empieza `base_layers` capas por debajo de h_full. Los
    niveles z < 0 se tratan como cristal (sustrato SOS), así que la base azul
    existe aunque la superficie esté en h = 0.

    Parámetros:
        heights: alturas enteras (Lx, Ly)
        base_layers: capas azules que se dibujan bajo el frente (>= 1)
        depth: capas totales de la ventana; None = justo las necesarias. En GIF y
            secuencias se fija al máximo de todos los frames para que la escala
            vertical no cambie.

    Devuelve:
        (filled, facecolors, edgecolors, h_full)
    """
    if base_layers < 1:
        raise ValueError("base_layers debe ser >= 1 (si no, no hay base azul que dibujar).")
    h = np.asarray(heights, dtype=int)
    h_full = int(h.min())
    z0 = h_full - int(base_layers)  # nivel real de la primera capa de vóxeles
    if depth is None:
        depth = _paper_window_depth(h, base_layers)
    z = z0 + np.arange(max(int(depth), 1))

    # Una celda (i, j, k) está llena si su nivel real es menor que la altura de la columna
    filled = z[None, None, :] < h[:, :, None]
    full_layer = filled & (z[None, None, :] < h_full)
    growing = filled & ~full_layer

    facecolors = np.zeros(filled.shape + (4,), dtype=float)
    edgecolors = np.zeros(filled.shape + (4,), dtype=float)
    facecolors[full_layer] = to_rgba(_PAPER_FULL_FACE)
    edgecolors[full_layer] = to_rgba(_PAPER_FULL_EDGE)
    facecolors[growing] = to_rgba(_PAPER_GROW_FACE)
    edgecolors[growing] = to_rgba(_PAPER_GROW_EDGE)
    return filled, facecolors, edgecolors, h_full


def _draw_paper(ax, heights: np.ndarray, base_layers: int = 1,
                depth: Optional[int] = None) -> int:
    """Dibuja una losa estilo 'paper' en `ax` (3D) y devuelve h_full."""
    filled, face, edge, h_full = _paper_voxels(heights, base_layers, depth)
    ax.voxels(filled, facecolors=face, edgecolors=edge, linewidth=0.3)
    Lx, Ly, n_layers = filled.shape
    # Vóxeles cúbicos: la losa se ve fina, como en el artículo
    ax.set_box_aspect((Lx, Ly, n_layers), zoom=1.15)
    ax.set_axis_off()
    return h_full


def _paper_legend_handles(h_full: Optional[int] = None) -> List[Patch]:
    """Leyenda del estilo 'paper' (capa completa / capa en crecimiento).

    Con `h_full` se indica el nivel de la última capa completa (una sola figura);
    en secuencias cambia de panel a panel y se omite.
    """
    full_label = "Capa completa" if h_full is None else f"Capa completa (h_min = {h_full})"
    return [
        Patch(facecolor=_PAPER_FULL_FACE, edgecolor=_PAPER_FULL_EDGE, label=full_label),
        Patch(facecolor=_PAPER_GROW_FACE, edgecolor=_PAPER_GROW_EDGE,
              label="Capa en crecimiento"),
    ]


# ---------------------------------------------------------------------------
# Estilo "relief": color por nivel + contorno de escalones
# ---------------------------------------------------------------------------
def _relief_n_levels(heights: np.ndarray) -> int:
    """Niveles sobre la capa completa en `heights` (>= 1 para que haya degradado)."""
    h = np.asarray(heights)
    return max(int(h.max()) - int(h.min()), 1)


def _resolve_levels_cmap(cmap) -> Tuple[Colormap, Tuple[float, float]]:
    """Convierte una paleta de niveles en (Colormap, tramo usado).

    Acepta el nombre de una paleta de `RELIEF_PALETTES` (sin distinguir mayúsculas),
    el nombre de cualquier colormap de matplotlib (se usa entero, (0, 1)) o un
    objeto `Colormap`. Para que la convención "claro = bajo" se mantenga, un
    colormap de matplotlib debe ir de claro a oscuro (p. ej. "Blues", "bone_r").
    """
    if isinstance(cmap, Colormap):
        return cmap, (0.0, 1.0)
    preset = RELIEF_PALETTES.get(str(cmap).lower())
    if preset is not None:
        spec, value_range = preset
        if isinstance(spec, str):
            return _get_cmap(spec), value_range
        return LinearSegmentedColormap.from_list(f"relief_{str(cmap).lower()}", spec), value_range
    if cmap in matplotlib.colormaps:
        return _get_cmap(cmap), (0.0, 1.0)
    raise ValueError(f"Paleta desconocida {cmap!r}: usa una de {tuple(RELIEF_PALETTES)}, "
                     "un colormap de matplotlib o un objeto Colormap.")


def _relief_level_colors(n_levels: int, cmap) -> np.ndarray:
    """Colores RGBA (n_levels + 1, 4) de los niveles h_full, h_full+1, ..., h_full+n_levels.

    El color 0 es el de la capa completa (extremo claro del tramo) y el último el
    del nivel más alto (extremo oscuro). Fijar `n_levels` en GIF y secuencias hace
    que un mismo nivel relativo tenga siempre el mismo color.
    """
    colormap, (lo, hi) = _resolve_levels_cmap(cmap)
    return colormap(np.linspace(lo, hi, int(n_levels) + 1))


def _relief_voxels(heights: np.ndarray, base_layers: int = 1, depth: Optional[int] = None,
                   n_levels: Optional[int] = None, cmap="ylgnbu"):
    """Vóxeles y colores del estilo 'relief'.

    Usa la ventana de `_paper_voxels` y colorea cada vóxel por su nivel: el vóxel
    de nivel real z es la cara superior de una columna de altura z + 1, así que
    toma el color del nivel k = z + 1 - h_full (recortado a [0, n_levels]). Las
    capas completas y la base (z < h_full) quedan en el color 0. Así la cara
    superior de cada columna tiene el mismo color que su celda en `plot_surface_2d`,
    y en las paredes de los escalones se ve la estratificación por niveles.

    Parámetros:
        heights, base_layers, depth: como en `_paper_voxels`
        n_levels: niveles de la paleta; None = los de `heights`
        cmap: colormap de los niveles

    Devuelve:
        (filled, facecolors, h_full, z0) con z0 el nivel real de la capa k=0 de vóxeles
    """
    filled, _, _, h_full = _paper_voxels(heights, base_layers, depth)
    if n_levels is None:
        n_levels = _relief_n_levels(heights)
    colors = _relief_level_colors(n_levels, cmap)
    z0 = h_full - int(base_layers)
    z = z0 + np.arange(filled.shape[2])
    level = np.clip(z + 1 - h_full, 0, n_levels)
    # Color por capa de la ventana, solo en las celdas llenas (el resto, transparente)
    facecolors = np.where(filled[..., None], colors[level][None, None, :, :], 0.0)
    return filled, facecolors, h_full, z0


def _step_edge_segments(heights: np.ndarray, z0: int = 0) -> np.ndarray:
    """Segmentos 3D del borde superior de cada escalón, en coordenadas de vóxel.

    Entre dos columnas vecinas de alturas distintas hay una pared; su arista
    superior está a la altura de la columna más alta. El vóxel (i, j, k) de
    `ax.voxels` ocupa [i, i+1] x [j, j+1] x [k, k+1], así que el borde entre
    (i, j) e (i+1, j) está en x = i+1 y la cara superior de una columna de altura h
    en k = h - z0. No se dibujan los bordes que cruzan la frontera periódica
    (quedarían fuera de la losa).

    Devuelve:
        array (n_segmentos, 2, 3)
    """
    h = np.asarray(heights, dtype=int)
    segments = []
    # Bordes perpendiculares a x: entre (i, j) e (i+1, j)
    i, j = np.nonzero(h[:-1, :] != h[1:, :])
    z = np.maximum(h[i, j], h[i + 1, j]) - z0
    x = i + 1
    segments.append(np.stack([np.stack([x, j, z], axis=-1),
                              np.stack([x, j + 1, z], axis=-1)], axis=1))
    # Bordes perpendiculares a y: entre (i, j) e (i, j+1)
    i, j = np.nonzero(h[:, :-1] != h[:, 1:])
    z = np.maximum(h[i, j], h[i, j + 1]) - z0
    y = j + 1
    segments.append(np.stack([np.stack([i, y, z], axis=-1),
                              np.stack([i + 1, y, z], axis=-1)], axis=1))
    return np.concatenate(segments, axis=0).astype(float)


class _OnTopLine3DCollection(Line3DCollection):
    """Line3DCollection que mplot3d dibuja después de todos los vóxeles.

    mplot3d ordena los artistas por el valor que devuelve `do_3d_projection()`
    (mayor = más lejos = se dibuja antes). Cada vóxel es un artista y la colección
    de líneas es uno solo que abarca toda la red, así que su profundidad media no
    tiene sentido: se devuelve -inf para que quede siempre encima. Los contornos
    están en caras superiores, que se ven desde arriba; solo una columna más alta
    delante podría taparlos, y con escalones de pocas capas eso es marginal.
    """

    def do_3d_projection(self):
        super().do_3d_projection()
        return -np.inf


def _draw_relief(ax, heights: np.ndarray, base_layers: int = 1,
                 depth: Optional[int] = None, n_levels: Optional[int] = None,
                 cmap="ylgnbu", z_scale: float = _RELIEF_Z_SCALE,
                 step_lines: bool = True, step_linewidth: float = 0.4) -> int:
    """Dibuja una losa estilo 'relief' en `ax` (3D) y devuelve h_full.

    Parámetros:
        heights, base_layers, depth, n_levels, cmap: como en `_relief_voxels`
        z_scale: exageración vertical (1 = vóxeles cúbicos, como en 'paper')
        step_lines: si se traza el contorno superior de cada escalón
        step_linewidth: grosor de ese contorno
    """
    filled, face, h_full, z0 = _relief_voxels(heights, base_layers, depth, n_levels, cmap)
    # Sin aristas de vóxel: la rejilla fina oscurece la superficie y tapa los contornos
    ax.voxels(filled, facecolors=face, edgecolors=None, linewidth=0, shade=True,
              lightsource=LightSource(**_RELIEF_LIGHT_3D))
    if step_lines:
        segments = _step_edge_segments(heights, z0)
        if len(segments):
            ax.add_collection3d(_OnTopLine3DCollection(segments, colors=[_RELIEF_STEP_COLOR],
                                                       linewidths=step_linewidth))
    Lx, Ly, n_layers = filled.shape
    ax.set_box_aspect((Lx, Ly, n_layers * float(z_scale)), zoom=1.15)
    ax.set_axis_off()
    return h_full


def _relief_colorbar(fig, n_levels: int, cmap, h_full: Optional[int] = None,
                     ax=None, cax=None, orientation: str = "vertical", **kwargs):
    """Barra de color discreta (un tramo por nivel) del estilo 'relief'.

    Con `h_full` las etiquetas son alturas absolutas (una sola figura); sin él son
    relativas ("h_min + k"), porque en GIF y secuencias h_min cambia de un frame a
    otro. Con muchos niveles se etiqueta solo uno de cada pocos.

    Parámetros:
        fig: figura donde se añade la barra
        n_levels, cmap: paleta (como en `_relief_level_colors`)
        h_full: altura de la capa completa o None
        ax / cax: ejes de los que se roba espacio / ejes propios de la barra
        orientation, **kwargs: se pasan a `fig.colorbar`
    """
    colors = _relief_level_colors(n_levels, cmap)
    n_colors = len(colors)
    # Un intervalo [k - 0.5, k + 0.5] por nivel k = 0..n_levels
    norm = BoundaryNorm(np.arange(n_colors + 1) - 0.5, n_colors)
    mappable = ScalarMappable(norm=norm, cmap=ListedColormap(colors))
    cbar = fig.colorbar(mappable, ax=ax, cax=cax, orientation=orientation, **kwargs)
    stride = int(np.ceil(n_colors / _RELIEF_MAX_TICKS))
    ticks = np.arange(0, n_colors, stride)
    if h_full is None:
        labels = ["h_min" if k == 0 else f"+{k}" for k in ticks]
        cbar.set_label("Nivel sobre la capa completa (h_min)")
    else:
        labels = [f"{h_full + k}" for k in ticks]
        cbar.set_label(f"Altura (capas); capa completa = {h_full}")
    cbar.set_ticks(ticks, labels=labels)
    cbar.minorticks_off()
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)
    return cbar


def _box_smooth_periodic(field: np.ndarray, width: int) -> np.ndarray:
    """Media móvil periódica de ancho `width` en los dos ejes (sin scipy)."""
    out = np.asarray(field, dtype=float)
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for shift in range(-(width // 2), width - width // 2):
            acc += np.roll(out, shift, axis=axis)
        out = acc / width
    return out


def _hillshade_factor(field: np.ndarray, azdeg: float, altdeg: float,
                      spacing: float, vert_exag: float) -> np.ndarray:
    """Factor de iluminación lambertiana normalizado a 1 en zonas planas.

    `field[i, j]` es la altura en x = i*spacing, y = j*spacing (x a la derecha,
    y hacia arriba al mostrar con origin="lower" y la matriz transpuesta). El
    azimut se mide en sentido horario desde +y (norte), como en LightSource.
    Se implementa aquí en vez de usar `LightSource.hillshade` porque esta
    reescala la intensidad al rango [0, 1] de cada imagen, de modo que el tono de
    las terrazas planas cambiaría de un snapshot a otro.
    """
    gx, gy = np.gradient(vert_exag * field, spacing)
    az, alt = np.radians(azdeg), np.radians(altdeg)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    # Normal de la superficie z = f(x, y): (-gx, -gy, 1) normalizada
    intensity = (-gx * light[0] - gy * light[1] + light[2]) / np.sqrt(gx**2 + gy**2 + 1.0)
    return np.clip(intensity, 0.0, None) / light[2]


def _relief_upsample(shape: Tuple[int, int], upsample: Optional[int]) -> int:
    """Sobremuestreo del mapa 2D: el dado o el que deja ~640 px por lado (1..8)."""
    if upsample is not None:
        return max(int(upsample), 1)
    return int(np.clip(640 // max(shape), 1, 8))


def _relief_map_image(heights: np.ndarray, n_levels: Optional[int] = None,
                      cmap="ylgnbu", upsample: int = 8,
                      shade_strength: float = 0.6) -> np.ndarray:
    """Imagen RGB (Lx*upsample, Ly*upsample, 3) del mapa de relieve.

    Cada celda se colorea por su nivel k = h - h_min (paleta de
    `_relief_level_colors`). Si `shade_strength` > 0 la altura se sobremuestrea,
    se suaviza en una celda de ancho (bisel) y se ilumina: el lado de cada escalón
    que mira a la luz se aclara y el opuesto se oscurece. Las terrazas planas
    conservan su color exacto (factor 1).
    """
    h = np.asarray(heights, dtype=int)
    h_full = int(h.min())
    if n_levels is None:
        n_levels = _relief_n_levels(h)
    palette = _relief_level_colors(n_levels, cmap)
    level_idx = np.clip(h - h_full, 0, n_levels)
    up = max(int(upsample), 1)
    rgb = np.kron(palette[level_idx][..., :3], np.ones((up, up, 1)))
    if shade_strength <= 0:
        return rgb

    bevel = _box_smooth_periodic(np.kron(h, np.ones((up, up))), up)
    # vert_exag = 1: un escalón de una capa sube en una celda de ancho (pendiente de 45°)
    factor = _hillshade_factor(bevel, spacing=1.0 / up, vert_exag=1.0, **_RELIEF_LIGHT_2D)
    factor = 1.0 + shade_strength * (factor - 1.0)
    return np.clip(rgb * factor[..., None], 0.0, 1.0)


def _draw_relief_map(ax, heights: np.ndarray, n_levels: Optional[int] = None,
                     cmap="ylgnbu", upsample: Optional[int] = None,
                     shading: bool = True, contours: bool = True,
                     contour_linewidth: float = 0.35) -> int:
    """Dibuja el mapa de relieve 2D (vista cenital) en `ax` y devuelve h_full.

    Ejes: x = primer índice de `heights`, y = segundo (como en las vistas 3D).
    Las curvas de nivel en h + 0.5 caen exactamente en los bordes de celda, es
    decir, son el contorno de cada isla.

    Parámetros:
        heights, n_levels, cmap: como en `_relief_voxels`
        upsample: píxeles por celda; None = automático (~640 px por lado)
        shading: bisel sombreado en los escalones
        contours: contorno de cada isla
        contour_linewidth: grosor de ese contorno
    """
    h = np.asarray(heights, dtype=int)
    Lx, Ly = h.shape
    h_full, h_max = int(h.min()), int(h.max())
    up = _relief_upsample(h.shape, upsample)
    rgb = _relief_map_image(h, n_levels, cmap, up, shade_strength=0.6 if shading else 0.0)
    # imshow espera (filas = y, columnas = x): se transpone
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=(0, Lx, 0, Ly),
              interpolation="nearest")
    if contours and h_max > h_full:
        # Rejilla sobremuestreada sin suavizar: el nivel h + 0.5 queda en el borde de celda
        h_up = np.kron(h, np.ones((up, up)))
        centers_x = (np.arange(Lx * up) + 0.5) / up
        centers_y = (np.arange(Ly * up) + 0.5) / up
        X, Y = np.meshgrid(centers_x, centers_y, indexing="ij")
        ax.contour(X, Y, h_up, levels=np.arange(h_full, h_max) + 0.5,
                   colors=[_RELIEF_STEP_COLOR], linewidths=contour_linewidth)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return h_full


def _regime_from_sigma(sigma: float) -> str:
    """Régimen cualitativo por umbrales de σ (criterio de plotter_v3)."""
    if sigma < 2.5:
        return "Spiral"
    if sigma < 6.0:
        return "Step"
    return "Rough"


class Plotter:
    """Visualización 3D, animaciones y análisis de una simulación kMC.

    Parámetros:
        kmc: motor de cualquiera de las dos líneas (KMC_BKL_Dynamic / KMC_BKL_Static)
        style: "classic" (Plotter), "v2" (Plotter_v2), "academic" (Plotter_v3),
               "paper" (paleta azul/rojo de Nagpal et al. 2024) o "relief"
               (un color por nivel y contorno de cada escalón)
        palette: paleta de niveles de las vistas por nivel ('relief', vista 2D,
               `plot_crystal_views`): nombre de `RELIEF_PALETTES` ("ylgnbu" por
               defecto, "pizarra", "arena", "salvia", "ciruela", "grises"), un
               colormap de matplotlib o un objeto Colormap. El argumento `cmap` de
               cada método tiene prioridad sobre ella.
    """

    def __init__(self, kmc: KMCLike, style: str = "classic", palette=None):
        if style not in STYLES:
            raise ValueError(f"style debe ser uno de {STYLES}, no {style!r}")
        if palette is not None:
            _resolve_levels_cmap(palette)  # falla aquí si el nombre no existe
        self.kmc = kmc
        self.style = style
        self.palette = palette

    # =====================================================================
    # Helpers comunes
    # =====================================================================
    def _default(self, key: str, value):
        """Devuelve `value` o, si es None, el valor por defecto del estilo."""
        return _STYLE_DEFAULTS[self.style][key] if value is None else value

    def _levels_cmap(self, cmap):
        """Paleta de niveles: `cmap` del método, si no `self.palette`, si no 'ylgnbu'."""
        if cmap is not None:
            return cmap
        if self.palette is not None:
            return self.palette
        return _STYLE_DEFAULTS["relief"]["cmap"]

    def _current_sigma(self) -> float:
        """σ = C/C_eq − 1 si el motor la expone; si no, se reconstruye desde S."""
        if hasattr(self.kmc, "sigma"):
            return float(self.kmc.sigma)
        if hasattr(self.kmc, "supersaturation"):
            return float(np.expm1(self.kmc.supersaturation))
        return 0.0

    def _current_S(self) -> float:
        """Sobresaturación logarítmica interna del motor, S = ln(1 + σ)."""
        if hasattr(self.kmc, "supersaturation"):
            return float(self.kmc.supersaturation)
        sigma = max(self._current_sigma(), -1.0 + 1e-15)
        return float(np.log1p(sigma))

    def _sigma_is_constant(self) -> bool:
        """True si σ no cambia en la corrida (σ fija o concentración constante).

        Solo entonces el σ actual del motor vale para cualquier snapshot; en la
        línea dinámica σ baja al agotarse el reservorio.
        """
        if getattr(self.kmc.p, "fixed_sigma", None) is not None:
            return True
        reservoir_is_dynamic = getattr(self.kmc, "_reservoir_is_dynamic", None)
        return reservoir_is_dynamic is not None and not reservoir_is_dynamic()

    def _paper_title(self, t_sel: float, conv: float) -> str:
        """Título del estilo 'paper': t y conversión del snapshot (+ σ si es constante)."""
        title = f"t = {t_sel:.2f} | conv = {conv:.1f} %"
        if self._sigma_is_constant():
            title += f" | σ = {self._current_sigma():.2f}"
        return title

    def _select_snapshot(self, snapshots: Optional[List[Snapshot]],
                         t_snapshot: Optional[float]) -> Tuple[float, np.ndarray, float]:
        """Elige el snapshot más cercano a t_snapshot, el último, o el estado actual."""
        if snapshots is not None and len(snapshots) > 0 and t_snapshot is not None:
            # Busca el snapshot con tiempo más cercano
            idx = int(np.argmin([abs(t - t_snapshot) for t, _, _ in snapshots]))
            t_sel, heights, conv = snapshots[idx]
            print(f"🧩 Snapshot seleccionado: t={t_sel:.3f} (conv={conv:.2f}%)")
        elif snapshots is not None and len(snapshots) > 0:
            # Toma el último snapshot si no se especifica tiempo
            t_sel, heights, conv = snapshots[-1]
            print(f"🧩 Usando último snapshot disponible: t={t_sel:.3f} (conv={conv:.2f}%)")
        else:
            # Usa el estado actual del cristal
            heights = self.kmc.lat.heights.copy()
            t_sel = self.kmc.t
            conv = self.kmc.conversion_percent
            print(f"🧩 Usando estado actual: t={t_sel:.3f} (conv={conv:.2f}%)")
        return t_sel, heights, conv

    # ---- Estilo académico (de plotter_v3) ----
    @staticmethod
    def _apply_academic_style(ax) -> None:
        """Estética académica: fondo blanco, texto negro, rejilla tenue."""
        fig = ax.figure
        fig.patch.set_facecolor("#FFFFFF")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            except Exception:
                pass
        text_color = "#000000"
        ax.tick_params(colors=text_color, labelcolor=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        try:
            ax.grid(color="#CCCCCC", linestyle="-", linewidth=0.5, alpha=0.5)
        except Exception:
            pass
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.title.set_fontsize(14)
        ax.title.set_fontweight("bold")

    @staticmethod
    def _academic_colors(heights: np.ndarray, voxels: np.ndarray) -> np.ndarray:
        """Colores académicos: cuerpo cian y capa superior roja opaca."""
        Lx, Ly = heights.shape
        colors = np.zeros(voxels.shape + (4,), dtype=float)
        bulk_rgba = list(to_rgba("#00FFFF"))  # cian: cuerpo del cristal
        top_rgba = list(to_rgba("#FF0000"))   # rojo: capa superior
        top_rgba[3] = 1.0
        for i in range(Lx):
            for j in range(Ly):
                h = int(heights[i, j])
                if h > 0:
                    for z in range(h - 1):
                        colors[i, j, z, :] = bulk_rgba
                    colors[i, j, h - 1, :] = top_rgba
        return colors

    # =====================================================================
    # Figura 3D estática
    # =====================================================================
    def plot_crystal_3d(self, mode: str = "voxel", elev: Optional[int] = None, azim: int = 45,
                        cmap: Optional[str] = None, save_path: Optional[str] = None,
                        title: Optional[str] = None,
                        snapshots: Optional[List[Snapshot]] = None,
                        t_snapshot: Optional[float] = None, base_layers: int = 1,
                        z_scale: Optional[float] = None):
        """Visualiza el cristal 3D (superficie continua o cubos discretos).

        Parámetros:
            mode: 'surface' o 'voxel' ('academic', 'paper' y 'relief' solo admiten 'voxel')
            elev, azim: ángulos de cámara (elev por defecto según el estilo)
            cmap: colormap del modo superficie o, en 'relief', de los niveles en
                crecimiento (por defecto según el estilo)
            save_path: ruta opcional para guardar la imagen
            title: título opcional (si es None se genera uno según el estilo)
            snapshots: lista opcional de snapshots generada por run()
            t_snapshot: tiempo del snapshot a mostrar (el más cercano)
            base_layers: solo estilos 'paper' y 'relief', capas de base bajo el
                frente de crecimiento
            z_scale: solo estilo 'relief', exageración vertical (None = 2; 1 = cúbico)
        """
        elev = self._default("elev", elev)
        levels_cmap = self._levels_cmap(cmap)  # solo estilo 'relief'
        cmap = self._default("cmap", cmap)
        if mode not in ("surface", "voxel"):
            raise ValueError("mode debe ser 'surface' o 'voxel'")
        if self.style in _VOXEL_ONLY_STYLES and mode != "voxel":
            raise ValueError(f"El estilo {self.style!r} solo admite mode='voxel'.")

        t_sel, heights, conv = self._select_snapshot(snapshots, t_snapshot)
        Lx, Ly = heights.shape
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")

        # Tamaño de figura de cada plotter original
        figsize = (6, 6) if self.style == "v2" else (7, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if self.style == "academic":
            self._apply_academic_style(ax)
        ax.view_init(elev=elev, azim=azim)

        if self.style == "classic":
            self._draw_3d_classic(fig, ax, heights, X, Y, mode, cmap)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("height")
            if title is None:
                title = f"Crystal at t={t_sel:.2f}, conv={conv:.1f}%"
            ax.set_title(title)
        elif self.style == "v2":
            self._draw_3d_v2(fig, ax, heights, X, Y, mode, cmap)
            ax.set_xlabel("X (lattice units)")
            ax.set_ylabel("Y (lattice units)")
            ax.set_zlabel("Altura (layers)")
            if title is None:
                title = (f"Cristal ({self.kmc.p.T:.0f}K) | t={t_sel:.2f} | "
                         f"σ={self._current_sigma():.2f} | S={self._current_S():.2f}")
            ax.set_title(title)
        elif self.style == "paper":
            fig.patch.set_facecolor("white")
            h_full = _draw_paper(ax, heights, base_layers)
            ax.legend(handles=_paper_legend_handles(h_full), loc="upper left",
                      frameon=False, fontsize=9)
            ax.set_title(self._paper_title(t_sel, conv) if title is None else title)
        elif self.style == "relief":
            fig.patch.set_facecolor("white")
            n_levels = _relief_n_levels(heights)
            h_full = _draw_relief(ax, heights, base_layers, n_levels=n_levels,
                                  cmap=levels_cmap,
                                  z_scale=_RELIEF_Z_SCALE if z_scale is None else z_scale)
            _relief_colorbar(fig, n_levels, levels_cmap, h_full, ax=ax, shrink=0.6, pad=0.02)
            ax.set_title(self._paper_title(t_sel, conv) if title is None else title)
        else:  # academic
            max_h = max(int(np.max(heights)), 1)
            voxels = _column_voxels(heights, max_h)
            colors = self._academic_colors(heights, voxels)
            ax.voxels(voxels, facecolors=colors, edgecolor="#000000", linewidth=0.5)
            ax.set_box_aspect((Lx, Ly, max_h * 0.55))
            if title is None:
                sigma = self._current_sigma()
                title = (f"{_regime_from_sigma(sigma)} Regime (σ={sigma:.2f}) | t={t_sel:.2f} s | "
                         f"conv={conv:.1f}% | S={self._current_S():.2f}")
            ax.set_title(title, pad=15)

        if save_path:
            if self.style in _VOXEL_ONLY_STYLES:
                # Fondo blanco para publicación
                plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white",
                            transparent=False)
            else:
                plt.savefig(save_path, dpi=250, bbox_inches="tight", transparent=True)
            print(f"💾 Imagen guardada en: {save_path}")

        if self.style == "v2":
            plt.tight_layout()
        plt.show()

    # =====================================================================
    # Vista 2D y vista combinada 3D + 2D (paleta del estilo "relief")
    # =====================================================================
    def plot_surface_2d(self, snapshots: Optional[List[Snapshot]] = None,
                        t_snapshot: Optional[float] = None, cmap: Optional[str] = None,
                        shading: bool = True, contours: bool = True,
                        upsample: Optional[int] = None, title: Optional[str] = None,
                        save_path: Optional[str] = None, ax=None):
        """Vista cenital 2D de la superficie, con un color por nivel.

        Usa siempre la paleta del estilo 'relief', sea cual sea `self.style`: cada
        celda toma el color de su nivel (h_min = capa completa, el más claro), los
        escalones llevan un bisel sombreado y cada isla queda rodeada por su curva
        de nivel. Es mucho más rápida que la vista 3D, así que sirve para redes
        grandes o para recorrer muchos snapshots.

        Parámetros:
            snapshots, t_snapshot: como en `plot_crystal_3d` (None = estado actual)
            cmap: colormap de niveles (None = paleta del Plotter, 'ylgnbu')
            shading: bisel sombreado en los escalones
            contours: contorno de cada isla
            upsample: píxeles por celda (None = automático)
            title: título opcional (None = t, conversión y σ si es constante)
            save_path: ruta opcional para guardar la imagen
            ax: ejes 2D donde dibujar; None = figura nueva (y se muestra)

        Devuelve:
            los ejes usados
        """
        cmap = self._levels_cmap(cmap)
        t_sel, heights, conv = self._select_snapshot(snapshots, t_snapshot)
        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            fig.patch.set_facecolor("white")
        fig = ax.figure
        n_levels = _relief_n_levels(heights)
        h_full = _draw_relief_map(ax, heights, n_levels, cmap, upsample, shading, contours)
        _relief_colorbar(fig, n_levels, cmap, h_full, ax=ax, shrink=0.85, pad=0.03)
        ax.set_title(self._paper_title(t_sel, conv) if title is None else title)

        if own_figure:
            if save_path:
                plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
                print(f"💾 Imagen guardada en: {save_path}")
            plt.show()
        return ax

    def plot_crystal_views(self, snapshots: Optional[List[Snapshot]] = None,
                           t_snapshot: Optional[float] = None, elev: Optional[int] = None,
                           azim: int = 45, cmap: Optional[str] = None, base_layers: int = 1,
                           z_scale: Optional[float] = None, contours: bool = True,
                           title: Optional[str] = None, save_path: Optional[str] = None):
        """Vista 3D (estilo 'relief') y vista 2D del mismo snapshot, lado a lado.

        Ambas comparten paleta y barra de color, así que una isla tiene el mismo
        color en las dos. Se usa el renderizado 'relief' sea cual sea `self.style`.

        Parámetros:
            snapshots, t_snapshot: como en `plot_crystal_3d` (None = estado actual)
            elev, azim: cámara de la vista 3D (elev por defecto 35)
            cmap: colormap de niveles (None = paleta del Plotter, 'ylgnbu')
            base_layers: capas de base bajo el frente de crecimiento (vista 3D)
            z_scale: exageración vertical de la vista 3D (None = 2)
            contours: contorno de cada isla en ambas vistas
            title: título general (None = t, conversión y σ si es constante)
            save_path: ruta opcional para guardar la figura
        """
        cmap = self._levels_cmap(cmap)
        elev = _STYLE_DEFAULTS["relief"]["elev"] if elev is None else elev
        z_scale = _RELIEF_Z_SCALE if z_scale is None else z_scale
        t_sel, heights, conv = self._select_snapshot(snapshots, t_snapshot)
        n_levels = _relief_n_levels(heights)

        fig = plt.figure(figsize=(13, 5.8))
        fig.patch.set_facecolor("white")
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)
        ax3d.view_init(elev=elev, azim=azim)
        h_full = _draw_relief(ax3d, heights, base_layers, n_levels=n_levels, cmap=cmap,
                              z_scale=z_scale, step_lines=contours)
        _draw_relief_map(ax2d, heights, n_levels, cmap, contours=contours)
        _relief_colorbar(fig, n_levels, cmap, h_full, ax=[ax3d, ax2d], shrink=0.8, pad=0.02)
        fig.suptitle(self._paper_title(t_sel, conv) if title is None else title, fontsize=13)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
            print(f"💾 Imagen guardada en: {save_path}")
        plt.show()

    def compare_palettes(self, snapshots: Optional[List[Snapshot]] = None,
                         t_snapshot: Optional[float] = None, palettes: Optional[List] = None,
                         n_cols: int = 3, panel_size: float = 4.0,
                         save_path: Optional[str] = None):
        """Vista 2D del mismo snapshot con varias paletas de niveles, para elegir una.

        Parámetros:
            snapshots, t_snapshot: como en `plot_crystal_3d` (None = estado actual)
            palettes: paletas a comparar (nombres de `RELIEF_PALETTES`, colormaps de
                matplotlib u objetos Colormap); None = todas las de `RELIEF_PALETTES`
            n_cols: paneles por fila
            panel_size: lado de cada panel en pulgadas
            save_path: ruta opcional para guardar la figura
        """
        palettes = list(RELIEF_PALETTES) if palettes is None else list(palettes)
        if not palettes:
            raise ValueError("palettes no puede estar vacío.")
        for palette in palettes:
            _resolve_levels_cmap(palette)  # valida todos antes de dibujar
        t_sel, heights, conv = self._select_snapshot(snapshots, t_snapshot)
        n_levels = _relief_n_levels(heights)
        h_full = int(np.min(heights))

        n_cols = max(1, min(int(n_cols), len(palettes)))
        n_rows = int(np.ceil(len(palettes) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                                 figsize=(panel_size * n_cols, (panel_size + 0.9) * n_rows))
        fig.patch.set_facecolor("white")
        for ax, palette in zip(axes.flat, palettes):
            _draw_relief_map(ax, heights, n_levels, palette)
            _relief_colorbar(fig, n_levels, palette, h_full, ax=ax, orientation="horizontal",
                             shrink=0.8, pad=0.18, aspect=30)
            name = palette.name if isinstance(palette, Colormap) else str(palette)
            ax.set_title(name)
        # Paneles sobrantes de la última fila
        for ax in list(axes.flat)[len(palettes):]:
            ax.set_visible(False)
        fig.suptitle(self._paper_title(t_sel, conv), fontsize=13)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
            print(f"💾 Comparación guardada en: {save_path}")
        plt.show()

    @staticmethod
    def _draw_3d_classic(fig, ax, heights, X, Y, mode, cmap) -> None:
        """Renderizado de plotter.py: superficie con cmap o vóxeles azules uniformes."""
        Lx, Ly = heights.shape
        if mode == "surface":
            surf = ax.plot_surface(X, Y, heights, cmap=cmap, linewidth=0, antialiased=True)
            fig.colorbar(surf, shrink=0.5, aspect=10, label="Altura")
        else:
            max_h = int(np.max(heights))
            voxels = _column_voxels(heights, max_h + 1)
            colors = np.zeros(voxels.shape + (4,), dtype=float)
            colors[..., :] = [0.2, 0.3, 0.8, 0.9]  # RGBA (azul translúcido)
            ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
            ax.set_box_aspect((Lx, Ly, max_h))

    @staticmethod
    def _draw_3d_v2(fig, ax, heights, X, Y, mode, cmap) -> None:
        """Renderizado de plotter_v2.py: superficie iluminada o vóxeles con gradiente."""
        Lx, Ly = heights.shape
        if mode == "surface":
            # Superficie continua con iluminación para relieve
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(heights.T, _get_cmap(cmap), vert_exag=0.1)
            ax.plot_surface(X, Y, heights, facecolors=rgb, linewidth=0, antialiased=True,
                            shade=False)
            m = plt.cm.ScalarMappable(cmap=cmap)
            m.set_array(heights)
            fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label="Altura (layers)")
        else:
            max_h = max(int(np.max(heights)), 1)
            voxels = _column_voxels(heights, max_h)
            # Gradiente de color con la altura
            colors = np.zeros(voxels.shape + (4,), dtype=float)
            for z in range(max_h):
                intensity = 0.3 + 0.7 * (z / max(max_h - 1, 1))
                colors[:, :, z, :] = [0.2 * intensity, 0.4 * intensity, 0.8, 0.9]
            # Capa superior de cada columna en verde marino
            top_layer_color = np.array([0.0, 0.35, 0.33, 0.98])
            ii, jj = np.where(heights > 0)
            colors[ii, jj, heights[ii, jj].astype(int) - 1, :] = top_layer_color
            ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.15)
            ax.set_box_aspect((Lx, Ly, max_h * 0.5))

    # =====================================================================
    # Animación GIF
    # =====================================================================
    def crystal_growth_gif(self, snapshots: List[Snapshot], save_path: str,
                           mode: Optional[str] = None, elev: Optional[int] = None,
                           azim: int = 45, cmap: Optional[str] = None, fps: int = 10,
                           interval_ms: Optional[int] = None, dpi: int = 150,
                           title_prefix: Optional[str] = None, repeat: bool = True,
                           every_n: int = 1, base_layers: int = 1, view: str = "3d",
                           z_scale: Optional[float] = None) -> None:
        """Genera un GIF animado del crecimiento a partir de snapshots.

        Parámetros:
            snapshots: lista de tuplas (t, heights, conversion_percent)
            save_path: ruta de salida (debe terminar en .gif)
            mode: 'surface' o 'voxel' (por defecto según el estilo)
            elev, azim, cmap: cámara y colormap (por defecto según el estilo)
            fps, interval_ms, dpi: parámetros de la animación y del guardado
            title_prefix: prefijo del título de cada frame
            repeat: si la animación se repite
            every_n: usa 1 de cada N snapshots
            base_layers: solo estilos 'paper' y 'relief', capas de base bajo el
                frente de crecimiento
            view: solo estilo 'relief': '3d', '2d' (vista cenital, mucho más rápida)
                o 'both' (3D y 2D lado a lado)
            z_scale: solo estilo 'relief', exageración vertical (None = 2)
        """
        mode = self._default("gif_mode", mode)
        elev = self._default("elev", elev)
        levels_cmap = self._levels_cmap(cmap)  # solo estilo 'relief'
        cmap = self._default("cmap", cmap)
        interval_ms = self._default("gif_interval_ms", interval_ms)
        title_prefix = self._default("gif_title_prefix", title_prefix)

        # Validaciones (unión de las de los tres originales)
        if snapshots is None or len(snapshots) == 0:
            raise ValueError("Se requieren snapshots no vacíos para generar el GIF.")
        if every_n < 1:
            raise ValueError("every_n debe ser >= 1.")
        if mode not in ("surface", "voxel"):
            raise ValueError("mode debe ser 'surface' o 'voxel'.")
        if self.style in _VOXEL_ONLY_STYLES and mode != "voxel":
            raise ValueError(f"El estilo {self.style!r} solo admite mode='voxel'.")
        if not save_path.lower().endswith(".gif"):
            raise ValueError("save_path debe terminar en '.gif'.")
        if view not in _RELIEF_VIEWS:
            raise ValueError(f"view debe ser uno de {_RELIEF_VIEWS}.")
        if view != "3d" and self.style != "relief":
            raise ValueError("view='2d'/'both' solo está disponible con style='relief'.")

        # Submuestreo y escala Z global (evita parpadeo entre frames)
        frames_data = snapshots[::every_n]
        if self.style == "relief":
            self._relief_growth_gif(frames_data, save_path, view, elev, azim, levels_cmap, fps,
                                    interval_ms, dpi, title_prefix, repeat, base_layers,
                                    _RELIEF_Z_SCALE if z_scale is None else z_scale)
            return
        _, h0, _ = frames_data[0]
        Lx, Ly = h0.shape
        global_max_h = max(int(max(np.max(h) for _, h, _ in frames_data)), 1)
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")
        # Estilo 'paper': misma altura de ventana en todos los frames (sin saltos de escala)
        paper_depth = max(_paper_window_depth(h, base_layers) for _, h, _ in frames_data)

        figsize = (9, 7) if self.style == "v2" else (7, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if self.style == "academic":
            self._apply_academic_style(ax)

        def _axes_frame() -> None:
            """Limpia y configura el eje común a cada frame."""
            ax.cla()
            if self.style == "paper":
                # Sin ejes: los límites los fija ax.voxels con la ventana común
                fig.patch.set_facecolor("white")
                ax.view_init(elev=elev, azim=azim)
                return
            if self.style == "academic":
                self._apply_academic_style(ax)
            ax.view_init(elev=elev, azim=azim)
            if self.style == "classic":
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("height")
            elif self.style == "v2":
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Altura")
            else:
                ax.set_xlabel("x", labelpad=10)
                ax.set_ylabel("y", labelpad=10)
                ax.set_zlabel("height", labelpad=10)
            ax.set_xlim(0, max(Lx - 1, 1))
            ax.set_ylim(0, max(Ly - 1, 1))
            ax.set_zlim(0, global_max_h)

        def _update(frame_idx: int):
            """Dibuja un frame de la animación."""
            t_sel, heights, conv = frames_data[frame_idx]
            _axes_frame()
            max_h_frame = max(int(np.max(heights)), 1)

            if self.style == "classic":
                if mode == "surface":
                    ax.plot_surface(X, Y, heights, cmap=cmap, linewidth=0, antialiased=True,
                                    vmin=0, vmax=global_max_h)
                else:
                    voxels = _column_voxels(heights, max_h_frame)
                    colors = np.zeros(voxels.shape + (4,), dtype=float)
                    colors[..., :] = [0.2, 0.3, 0.8, 0.9]
                    ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
                    ax.set_box_aspect((Lx, Ly, global_max_h))
                ax.set_title(f"{title_prefix} | t={t_sel:.2f}, conv={conv:.1f}%")

            elif self.style == "v2":
                if mode == "surface":
                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = ls.shade(heights.T, _get_cmap(cmap), vmin=0, vmax=global_max_h,
                                   vert_exag=0.1)
                    ax.plot_surface(X, Y, heights, facecolors=rgb, linewidth=0,
                                    antialiased=True, shade=False)
                else:
                    voxels = _column_voxels(heights, max_h_frame)
                    colors = np.zeros(voxels.shape + (4,), dtype=float)
                    colors[..., :] = [0.2, 0.3, 0.8, 0.85]
                    top_layer_color = np.array([0.0, 0.35, 0.33, 0.98])
                    ii, jj = np.where(heights > 0)
                    colors[ii, jj, heights[ii, jj].astype(int) - 1, :] = top_layer_color
                    ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
                    ax.set_box_aspect((Lx, Ly, global_max_h))
                # Nota: como en el original, σ y S son los del estado final del motor.
                ax.set_title(f"{title_prefix} | t={t_sel:.2f} | σ={self._current_sigma():.2f} | "
                             f"S={self._current_S():.2f}")

            elif self.style == "paper":
                h_full = _draw_paper(ax, heights, base_layers, depth=paper_depth)
                ax.legend(handles=_paper_legend_handles(h_full), loc="upper left",
                          frameon=False, fontsize=9)
                ax.set_title(f"{title_prefix}\n{self._paper_title(t_sel, conv)}")

            else:  # academic
                voxels = _column_voxels(heights, max_h_frame)
                colors = self._academic_colors(heights, voxels)
                ax.voxels(voxels, facecolors=colors, edgecolor="#000000", linewidth=0.5)
                ax.set_box_aspect((Lx, Ly, global_max_h * 0.55))
                regimen = _regime_from_sigma(self._current_sigma())
                ax.set_title(f"{title_prefix} ({regimen} Regime)\nt={t_sel:.2f} s | "
                             f"conv={conv:.1f}%", pad=15)
            return (ax,)

        anim = FuncAnimation(fig, _update, frames=len(frames_data), interval=interval_ms,
                             blit=False, repeat=repeat)
        anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"💾 GIF guardado en: {save_path}")

    def _relief_growth_gif(self, frames_data: List[Snapshot], save_path: str, view: str,
                           elev: int, azim: int, cmap, fps: int, interval_ms: int,
                           dpi: int, title_prefix: str, repeat: bool, base_layers: int,
                           z_scale: float) -> None:
        """GIF del estilo 'relief' con vista 3D, 2D o ambas (ver `crystal_growth_gif`).

        La ventana vertical y el número de niveles de la paleta son comunes a todos
        los frames (sin saltos de escala ni de color); la barra de color es relativa
        a h_min, que se indica en el título de cada frame.
        """
        depth = max(_paper_window_depth(h, base_layers) for _, h, _ in frames_data)
        n_levels = max(_relief_n_levels(h) for _, h, _ in frames_data)

        figsize = {"3d": (7, 6), "2d": (6.5, 6), "both": (12.5, 6)}[view]
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax3d = ax2d = None
        if view == "3d":
            ax3d = fig.add_subplot(111, projection="3d")
        elif view == "2d":
            ax2d = fig.add_subplot(111)
        else:
            ax3d = fig.add_subplot(1, 2, 1, projection="3d")
            ax2d = fig.add_subplot(1, 2, 2)
        # Barra fija en sus propios ejes: no se redibuja ni roba espacio frame a frame
        fig.subplots_adjust(left=0.04, right=0.86, bottom=0.06, top=0.86, wspace=0.12)
        cax = fig.add_axes([0.89, 0.2, 0.02, 0.55])
        _relief_colorbar(fig, n_levels, cmap, cax=cax)

        def _update(frame_idx: int):
            """Dibuja un frame de la animación."""
            t_sel, heights, conv = frames_data[frame_idx]
            h_full = int(np.min(heights))
            if ax3d is not None:
                ax3d.cla()
                ax3d.view_init(elev=elev, azim=azim)
                _draw_relief(ax3d, heights, base_layers, depth=depth, n_levels=n_levels,
                             cmap=cmap, z_scale=z_scale)
            if ax2d is not None:
                ax2d.cla()
                _draw_relief_map(ax2d, heights, n_levels, cmap)
            fig.suptitle(f"{title_prefix}\n{self._paper_title(t_sel, conv)} | h_min = {h_full}",
                         fontsize=12)
            return tuple(a for a in (ax3d, ax2d) if a is not None)

        anim = FuncAnimation(fig, _update, frames=len(frames_data), interval=interval_ms,
                             blit=False, repeat=repeat)
        anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"💾 GIF guardado en: {save_path}")

    # =====================================================================
    # Secuencia de morfologías (estilo Fig. 9 de Nagpal et al. 2024)
    # =====================================================================
    def plot_morphology_sequence(self, snapshots: List[Snapshot],
                                 times: Optional[List[float]] = None, n_panels: int = 3,
                                 base_layers: int = 1, elev: Optional[int] = None,
                                 azim: int = 45, panel_size: Tuple[float, float] = (4.5, 4.0),
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None, show_map: bool = False,
                                 cmap: Optional[str] = None, z_scale: Optional[float] = None):
        """Varios snapshots en fila (estilo Fig. 9 de Nagpal et al. 2024).

        Con style='relief' usa el renderizado por niveles de ese estilo; con
        cualquier otro estilo, la paleta del artículo (azul/rojo, 'paper').

        Parámetros:
            snapshots: lista de tuplas (t, heights, conversion_percent)
            times: tiempos a mostrar (se toma el snapshot más cercano a cada uno);
                None = `n_panels` snapshots equiespaciados, incluidos el primero y el último
            n_panels: número de paneles si `times` es None
            base_layers: capas de base bajo el frente de crecimiento
            elev, azim: cámara (elev por defecto 35, como los estilos 'paper' y 'relief')
            panel_size: tamaño (ancho, alto) de cada panel en pulgadas
            title: título general opcional
            save_path: ruta opcional para guardar la figura
            show_map: solo style='relief', añade debajo una fila con la vista 2D
            cmap: solo style='relief', colormap de niveles (None = paleta del Plotter, 'ylgnbu')
            z_scale: solo style='relief', exageración vertical (None = 2)
        """
        if snapshots is None or len(snapshots) == 0:
            raise ValueError("Se requieren snapshots no vacíos.")
        relief = self.style == "relief"
        if show_map and not relief:
            raise ValueError("show_map=True solo está disponible con style='relief'.")
        elev = _STYLE_DEFAULTS["paper"]["elev"] if elev is None else elev

        # Selección de snapshots
        if times is None:
            n = min(max(int(n_panels), 1), len(snapshots))
            idx = np.unique(np.linspace(0, len(snapshots) - 1, n).round().astype(int))
        else:
            snap_times = np.array([t for t, _, _ in snapshots])
            idx = [int(np.argmin(np.abs(snap_times - t))) for t in times]
        selected = [snapshots[i] for i in idx]

        # Ventana vertical común para que todas las losas tengan el mismo grosor
        depth = max(_paper_window_depth(h, base_layers) for _, h, _ in selected)

        n = len(selected)
        if not relief:
            fig = plt.figure(figsize=(panel_size[0] * n, panel_size[1]))
            fig.patch.set_facecolor("white")
            for k, (t_sel, heights, conv) in enumerate(selected):
                ax = fig.add_subplot(1, n, k + 1, projection="3d")
                ax.view_init(elev=elev, azim=azim)
                _draw_paper(ax, heights, base_layers, depth=depth)
                # Etiqueta de tiempo abajo a la izquierda, como en el artículo
                ax.text2D(0.02, 0.02, f"t = {t_sel:.2f}\nconv = {conv:.1f} %",
                          transform=ax.transAxes, fontsize=11, style="italic")

            fig.legend(handles=_paper_legend_handles(), loc="lower center", ncol=2,
                       frameon=False, fontsize=10)
        else:
            cmap = self._levels_cmap(cmap)
            z_scale = _RELIEF_Z_SCALE if z_scale is None else z_scale
            # Misma paleta en todos los paneles: un nivel relativo, un color
            n_levels = max(_relief_n_levels(h) for _, h, _ in selected)
            n_rows = 2 if show_map else 1
            fig = plt.figure(figsize=(panel_size[0] * n, panel_size[1] * n_rows))
            fig.patch.set_facecolor("white")
            for k, (t_sel, heights, conv) in enumerate(selected):
                ax = fig.add_subplot(n_rows, n, k + 1, projection="3d")
                ax.view_init(elev=elev, azim=azim)
                h_full = _draw_relief(ax, heights, base_layers, depth=depth,
                                      n_levels=n_levels, cmap=cmap, z_scale=z_scale)
                ax.text2D(0.02, 0.02, f"t = {t_sel:.2f}\nconv = {conv:.1f} %\nh_min = {h_full}",
                          transform=ax.transAxes, fontsize=11, style="italic")
                if show_map:
                    ax_map = fig.add_subplot(n_rows, n, n + k + 1)
                    _draw_relief_map(ax_map, heights, n_levels, cmap)
            # Barra horizontal propia, centrada bajo los paneles
            cax = fig.add_axes([0.3, 0.035, 0.4, 0.02 / n_rows])
            _relief_colorbar(fig, n_levels, cmap, cax=cax, orientation="horizontal")

        if title is None and self._sigma_is_constant():
            title = f"σ = {self._current_sigma():.2f}"
        if title:
            fig.suptitle(title, fontsize=13)
        if relief:
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.13 / n_rows + 0.04,
                                top=0.92 if n_rows == 1 else 0.95, wspace=0.08, hspace=0.05)
        else:
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.92, wspace=0.0)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
            print(f"💾 Secuencia guardada en: {save_path}")
        plt.show()

    # =====================================================================
    # Series temporales y análisis
    # =====================================================================
    def plot_conversion(self, snapshots: List[Snapshot], title: str,
                        figsize: Tuple[int, int] = (8, 5), save_path: Optional[str] = None):
        """Conversión vs tiempo a partir de tuplas (t, heights|None, conversion). (plotter.py)"""
        times = [t for t, _, _ in snapshots]
        conversions = [conv for _, _, conv in snapshots]

        plt.figure(figsize=figsize)
        plt.plot(times, conversions, marker="o", linestyle="-", color="blue")
        plt.xlabel("Tiempo")
        plt.ylabel("Conversión (%)")
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", transparent=True)
            print(f"💾 Imagen guardada en: {save_path}")
        plt.show()

    def _param_text(self, scalar: str, x: str, y: str) -> str:
        """Texto de un parámetro escalar (línea dinámica) o del par x/y (estática)."""
        p = self.kmc.p
        if hasattr(p, scalar):
            return f"{getattr(p, scalar):.2f}"
        return f"{getattr(p, x):.2f} / {getattr(p, y):.2f} (x / y)"

    def plot_growth_rate_analysis(self, stats: Dict, sigma: float, face: str,
                                  d_interplanar: float = 3.5e-3,
                                  time_unit_to_min: float = 1.0,
                                  save_path: Optional[str] = None):
        """Velocidad de crecimiento vs tiempo con análisis de régimen (estilo Fig. 7).
        (plotter_v2.py; requiere `stats['height_history']` de la línea estática.)

        Parámetros:
            stats: diccionario retornado por run() con 'height_history'
            sigma: σ de la simulación
            face: cara cristalina ("110" o "101")
            d_interplanar: espaciado interplanar en μm (3.5e-3 para lisozima)
            time_unit_to_min: conversión de unidades de tiempo a minutos
        """
        height_history = np.array(stats["height_history"])
        if len(height_history) == 0:
            print("⚠️ No hay datos de historia de altura")
            return

        t = height_history[:, 0]
        h = height_history[:, 1]
        sigma_actual = height_history[:, 2]

        # Pendiente descartando el 20 % inicial (transitorio)
        skip = max(1, int(0.2 * len(t)))
        if len(t) > skip + 10:
            coeffs = np.polyfit(t[skip:], h[skip:], 1)
            growth_rate_layers = coeffs[0]
            fit_values = np.polyval(coeffs, t[skip:])
        else:
            growth_rate_layers = (h[-1] - h[0]) / (t[-1] - t[0]) if len(t) > 1 else 0
            fit_values = h[skip:]

        # Conversión a μm/min como en el paper
        growth_rate_um_min = growth_rate_layers * d_interplanar / time_unit_to_min

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: altura vs tiempo con ajuste lineal
        ax1 = axes[0, 0]
        ax1.plot(t, h, "b-", alpha=0.7, label="Simulación")
        ax1.plot(t[skip:], fit_values, "r--", linewidth=2,
                 label=f"Ajuste lineal: v={growth_rate_layers:.3f} layers/u.t.")
        ax1.set_xlabel("Tiempo (u.t.)")
        ax1.set_ylabel("Altura promedio (layers)")
        ax1.set_title(f"Crecimiento en cara ({face}) | σ = {sigma:.2f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: verificación de σ constante
        ax2 = axes[0, 1]
        ax2.plot(t, sigma_actual, "g-", linewidth=1.5)
        ax2.axhline(y=sigma, color="r", linestyle="--", alpha=0.5, label="σ objetivo")
        ax2.set_xlabel("Tiempo (u.t.)")
        ax2.set_ylabel("Sigma")
        ax2.set_title("Verificación: σ constante (modo paper)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: velocidad instantánea (derivada suavizada)
        ax3 = axes[1, 0]
        if len(t) > 10:
            window = max(3, len(t) // 20)
            h_smooth = np.convolve(h, np.ones(window) / window, mode="same")
            v_instant = np.gradient(h_smooth, t)
            ax3.plot(t, v_instant, "purple", alpha=0.7, label="Velocidad instantánea")
            ax3.axhline(y=growth_rate_layers, color="r", linestyle="--",
                        label=f"Velocidad media: {growth_rate_layers:.3f}")
        ax3.set_xlabel("Tiempo (u.t.)")
        ax3.set_ylabel("Velocidad (layers/u.t.)")
        ax3.set_title("Velocidad de crecimiento instantánea")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: resumen
        ax4 = axes[1, 1]
        ax4.axis("off")
        if sigma < 2.5:
            regimen, color_reg = "SPIRAL (baja σ)", "yellow"
        elif sigma < 6.0:
            regimen, color_reg = "STEP (media σ)", "orange"
        else:
            regimen, color_reg = "ROUGH (alta σ)", "red"

        info_text = f"""
        PARÁMETROS DE SIMULACIÓN
        ─────────────────────────
        Cara cristalina: ({face})
        Sigma σ: {sigma:.2f}
        S = ln(1 + σ): {np.log1p(max(sigma, -1.0 + 1e-15)):.2f}
        Temperatura: {self.kmc.p.T:.1f} K

        PARÁMETROS DEL MODELO
        ─────────────────────
        E_pb/kT: {self._param_text("E_pb_over_kT", "E_pb_over_kT_x", "E_pb_over_kT_y")}
        φ/kT: {self.kmc.p.phi_over_kT:.2f}
        δ (delta): {self._param_text("delta", "delta_x", "delta_y")}
        K₀⁺: {self.kmc.p.K0_plus:.3f}

        RESULTADOS
        ──────────
        Régimen: {regimen}
        Velocidad: {growth_rate_layers:.4f} layers/u.t.
        Velocidad: {growth_rate_um_min:.4f} μm/min
        Altura final: {h[-1]:.2f} layers
        Rugosidad final: {np.std(self.kmc.lat.heights):.2f}

        Eventos totales: {stats['total_events']:,}
        Adsorciones: {stats['event_counts']['adsorption']:,}
        Desorciones: {stats['event_counts']['desorption']:,}
        Migraciones: {stats['event_counts']['migration']:,}
        """
        ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor=color_reg, alpha=0.3, pad=1))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Análisis guardado en: {save_path}")
        plt.show()

        return {
            "growth_rate_layers": growth_rate_layers,
            "growth_rate_um_min": growth_rate_um_min,
            "regimen": regimen,
            "final_height": h[-1],
            "final_roughness": np.std(self.kmc.lat.heights),
        }

    def plot_adsorption_probabilities(self, stats: Dict, sigma: float, face: str,
                                      save_path: Optional[str] = None):
        """Probabilidades de adsorción por tipo de sitio (estilo Fig. 8). (plotter_v2.py)

        Requiere `stats['adsorption_probs_history']` (motor estático con
        record_adsorption_probs=True).
        """
        prob_hist = stats.get("adsorption_probs_history", [])
        if len(prob_hist) == 0:
            print("⚠️ No hay datos de probabilidades de adsorción "
                  "(¿record_adsorption_probs=False?)")
            return

        times = [p[0] for p in prob_hist]
        sigmas = [p[1] for p in prob_hist]
        p_adatom = [p[2].get(0, 0) for p in prob_hist]
        p_step = [p[2].get(1, 0) for p in prob_hist]
        p_kink = [p[2].get(2, 0) for p in prob_hist]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Izquierda: probabilidades vs tiempo
        ax1.plot(times, p_adatom, "b-", linewidth=2, label="Adatom (i=0)", marker="o", markersize=3)
        ax1.plot(times, p_step, "r-", linewidth=2, label="Step (i=1)", marker="s", markersize=3)
        ax1.plot(times, p_kink, "y-", linewidth=2, label="Kink-like (i≥2)", marker="^", markersize=3)
        ax1.set_xlabel("Tiempo (u.t.)")
        ax1.set_ylabel("Probabilidad relativa")
        ax1.set_title(f"Evolución de probabilidades de adsorción\nCara ({face}), σ={sigma:.2f}")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Derecha: probabilidades vs σ (comparación cualitativa con Fig. 8)
        ax2.plot(sigmas, p_adatom, "b-", linewidth=2, label="Adatom (i=0)", alpha=0.7)
        ax2.plot(sigmas, p_step, "r-", linewidth=2, label="Step (i=1)", alpha=0.7)
        ax2.plot(sigmas, p_kink, "y-", linewidth=2, label="Kink-like (i≥2)", alpha=0.7)
        sigma_range = np.linspace(1, 8, 100)
        # Curvas "teóricas" aproximadas del original (solo guía visual, no del paper)
        p_kink_theory = np.exp(-sigma_range * 0.5) / (1 + np.exp(-sigma_range * 0.5))
        p_adatom_theory = 1 - p_kink_theory - 0.3
        ax2.plot(sigma_range, p_adatom_theory, "b--", alpha=0.3, label="Tendencia esperada")
        ax2.plot(sigma_range, p_kink_theory, "y--", alpha=0.3)
        ax2.set_xlabel("Sigma σ")
        ax2.set_ylabel("Probabilidad relativa")
        ax2.set_title("Probabilidades vs σ (Comparación Figura 8)")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Probabilidades guardadas en: {save_path}")
        plt.show()

    def plot_surface_morphology(self, heights: Optional[np.ndarray] = None,
                                cmap: str = "terrain", show_histogram: bool = True,
                                save_path: Optional[str] = None):
        """Mapa 2D de alturas e histograma con rugosidad. (plotter_v2.py)"""
        if heights is None:
            heights = self.kmc.lat.heights

        fig = plt.figure(figsize=(14, 6) if show_histogram else (8, 7))

        # Panel 1: mapa de alturas
        ax1 = fig.add_subplot(121 if show_histogram else 111)
        im = ax1.imshow(heights, cmap=cmap, origin="lower", interpolation="nearest")
        plt.colorbar(im, ax=ax1, label="Altura (layers)", fraction=0.046)
        ax1.set_title("Morfología Superficial 2D")
        ax1.set_xlabel("X (lattice units)")
        ax1.set_ylabel("Y (lattice units)")

        if show_histogram:
            # Panel 2: histograma y estadísticos
            ax2 = fig.add_subplot(122)
            ax2.hist(heights.flatten(), bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            ax2.axvline(np.mean(heights), color="r", linestyle="--", linewidth=2,
                        label=f"Media: {np.mean(heights):.2f}")
            ax2.axvline(np.median(heights), color="g", linestyle="--", linewidth=2,
                        label=f"Mediana: {np.median(heights):.2f}")
            ax2.set_xlabel("Altura (layers)")
            ax2.set_ylabel("Frecuencia")
            ax2.set_title(f"Distribución de Alturas\nσ_h = {np.std(heights):.2f}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"💾 Morfología guardada en: {save_path}")
        plt.show()

    def plot_multi_sigma_comparison(self, results_list: List[Dict], face: str,
                                    save_path: Optional[str] = None):
        """Velocidad de crecimiento vs σ para varias corridas (Fig. 7 completa). (plotter_v2.py)

        Parámetros:
            results_list: dicts con 'sigma', 'growth_rate_layers' y opcional 'growth_rate_std'
        """
        if not results_list:
            print("⚠️ Lista de resultados vacía")
            return

        results_sorted = sorted(results_list, key=lambda x: x["sigma"])
        sigmas = [r["sigma"] for r in results_sorted]
        rates = [r["growth_rate_layers"] for r in results_sorted]
        rates_err = [r.get("growth_rate_std", 0) for r in results_sorted]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: velocidad vs σ con regiones de régimen
        ax1.errorbar(sigmas, rates, yerr=rates_err, marker="o", markersize=8, linewidth=2,
                     capsize=5, color="darkblue", label=f"Cara ({face}) - Modelo kMC")
        ax1.axvspan(0, 2.5, alpha=0.2, color="yellow", label="Régimen Spiral")
        ax1.axvspan(2.5, 6.0, alpha=0.2, color="orange", label="Régimen Step")
        ax1.axvspan(6.0, 8.5, alpha=0.2, color="red", label="Régimen Rough")
        ax1.set_xlabel("Supersaturación σ", fontsize=12)
        ax1.set_ylabel("Velocidad de crecimiento (layers/u.t.)", fontsize=12)
        ax1.set_title(f"Figura 7 - Crecimiento en cara ({face})\n"
                      "Lysozima HEW (Nagpal et al. 2024)", fontsize=13)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 8.5)

        # Panel 2: log-log para exponentes
        ax2.loglog(sigmas, rates, "o-", color="darkgreen", linewidth=2, markersize=8)
        ax2.set_xlabel("Supersaturación σ (log)", fontsize=12)
        ax2.set_ylabel("Velocidad (log)", fontsize=12)
        ax2.set_title("Análisis de exponentes de crecimiento")
        ax2.grid(True, alpha=0.3, which="both")
        step_mask = [(2.5 <= s <= 6.0) for s in sigmas]
        if sum(step_mask) > 2:
            step_sigmas = np.array([s for s, m in zip(sigmas, step_mask) if m])
            step_rates = np.array([r for r, m in zip(rates, step_mask) if m])
            log_fit = np.polyfit(np.log(step_sigmas), np.log(step_rates), 1)
            ax2.plot(step_sigmas, np.exp(log_fit[1]) * step_sigmas ** log_fit[0], "r--",
                     label=f"Pendiente ≈ {log_fit[0]:.2f}")
            ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Comparación guardada en: {save_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Alias heredados: subclases de una línea para que isinstance() siga funcionando
# ---------------------------------------------------------------------------
class Plotter_v2(Plotter):
    """Alias de `Plotter(kmc, style="v2")` (antes en plotter_v2.py)."""

    def __init__(self, kmc: KMCLike, style: str = "v2"):
        super().__init__(kmc, style=style)


class Plotter_v3(Plotter):
    """Alias de `Plotter(kmc, style="academic")` (antes en plotter_v3.py)."""

    def __init__(self, kmc: KMCLike, style: str = "academic"):
        super().__init__(kmc, style=style)
