"""Visualización dinámica e interactiva de cristales 3D (plotly, necesita WebGL).

`plot_growth_3d` muestra la superficie del cristal en 3D con la celda REAL (triclínica
incluida): se puede rotar, acercar y pasar el cursor por las unidades. Tiene un
fotograma por snapshot, con un deslizador de tiempo y botones ▶ / ❚❚ para ver el
crecimiento.

Las trazas 3D de plotly se dibujan siempre con WebGL. Si el visor no lo tiene (p. ej.
VS Code sin aceleración por hardware: "WebGL is not supported by your browser"), usa
`viz.animate_growth_3d`, que hace lo mismo con matplotlib y sin WebGL.

Es el único módulo de kmc3d que necesita plotly (`import kmc3d` funciona sin él). En
un notebook, `fig.show()` necesita además nbformat.
"""

from typing import Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from .geometry import Cell
from .viz import BLUES, Snapshot, arrival_times, frame_indices, surface_mask

# Escala de color de plotly a partir de la rampa azul de un tono de viz.BLUES
BLUES_PLOTLY = [[float(p), c] for p, c in zip(np.linspace(0.0, 1.0, len(BLUES)), BLUES)]


def plot_growth_3d(snaps: Union[Sequence[Snapshot], np.ndarray], cell: Cell = Cell(),
                   color_by: str = "arrival", axis: int = 2, max_frames: Optional[int] = 15,
                   surface_only: bool = True, marker_size: float = 2.5,
                   colorscale=None, frame_ms: int = 400, title: str = "",
                   height: int = 650) -> go.Figure:
    """Figura 3D interactiva del crecimiento: un fotograma por snapshot.

    Parámetros:
        snaps: lista (t, coords) de `KMC3D.run` o un único array de coordenadas (N, 3)
        cell: celda real (las coordenadas se pasan a cartesianas con `cell.matrix`)
        color_by: "arrival" (tiempo de llegada de cada unidad: se ve qué partes
            crecieron cuándo) o "height" (coordenada cartesiana `axis`)
        max_frames: número máximo de fotogramas (equiespaciados); None = todos
        surface_only: dibuja solo las unidades de superficie (`surface_mask`)
        marker_size: tamaño de cada unidad en píxeles
        colorscale: escala de color de plotly (por defecto, rampa azul de un tono)
        frame_ms: duración de cada fotograma al reproducir

    La escala de color y los ejes son los mismos en todos los fotogramas, de modo que
    los colores y el tamaño del cristal se pueden comparar en el tiempo.
    """
    if isinstance(snaps, np.ndarray):
        snaps = [(0.0, snaps)]
    snaps = [(float(t), np.asarray(c, int)) for t, c in snaps]
    if color_by not in ("arrival", "height"):
        raise ValueError("color_by debe ser 'arrival' o 'height'")
    llegada = arrival_times(snaps) if color_by == "arrival" else None
    marcos = frame_indices(len(snaps), max_frames)

    # Datos de cada fotograma: cartesianas (float32, más compacto en el JSON) y color
    datos = []
    for k in marcos:
        t, c = snaps[k]
        sel = surface_mask(c) if surface_only else np.ones(len(c), dtype=bool)
        X = cell.to_cartesian(c[sel]).astype(np.float32)
        col = llegada[k][sel] if llegada is not None else X[:, axis]
        datos.append((k, t, len(c), X, np.asarray(col, dtype=np.float32)))

    # Ejes y escala de color comunes a todos los fotogramas
    lo = np.min([X.min(axis=0) for *_, X, _ in datos], axis=0)
    hi = np.max([X.max(axis=0) for *_, X, _ in datos], axis=0)
    span = np.maximum(hi - lo, 1e-9)
    cmin = float(min(col.min() for *_, col in datos))
    cmax = float(max(col.max() for *_, col in datos))
    etiqueta = "t de llegada" if color_by == "arrival" else "xyz"[axis]

    def traza(X, col):
        # Una nube de puntos con la escala de color fija
        return go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
            marker=dict(size=marker_size, color=col, colorscale=colorscale or BLUES_PLOTLY,
                        cmin=cmin, cmax=cmax, colorbar=dict(title=etiqueta, thickness=14)),
            hovertemplate=("x = %{x:.1f}<br>y = %{y:.1f}<br>z = %{z:.1f}<br>"
                           + etiqueta + " = %{marker.color:.3g}<extra></extra>"))

    def rotulo(k, t, n):
        # Título de cada fotograma: snapshot, tiempo y número de unidades
        cabecera = f"{title}<br>" if title else ""
        return f"{cabecera}snapshot {k}/{len(snaps) - 1} · t = {t:.4g} · N = {n}"

    frames = [go.Frame(data=[traza(X, col)], name=str(k),
                       layout=go.Layout(title_text=rotulo(k, t, n)))
              for k, t, n, X, col in datos]
    k0, t0, n0, X0, col0 = datos[0]
    fig = go.Figure(data=[traza(X0, col0)], frames=frames)

    ejes = {f"{e}axis": dict(range=[float(lo[i] - 0.02 * span.max()),
                                    float(hi[i] + 0.02 * span.max())], title=e)
            for i, e in enumerate("xyz")}
    fig.update_layout(
        title_text=rotulo(k0, t0, n0), height=height, margin=dict(l=0, r=0, t=70, b=0),
        scene=dict(aspectmode="manual",
                   aspectratio=dict(zip("xyz", (span / span.max()).tolist())), **ejes),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
            pad=dict(t=8, r=8), showactive=False,
            buttons=[
                dict(label="▶", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                      transition=dict(duration=0), fromcurrent=True)]),
                dict(label="❚❚", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0), mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, x=0.1, len=0.9, y=0.0, yanchor="top", pad=dict(t=8),
            currentvalue=dict(prefix="snapshot "),
            steps=[dict(label=str(k), method="animate",
                        args=[[str(k)], dict(mode="immediate", transition=dict(duration=0),
                                             frame=dict(duration=0, redraw=True))])
                   for k, *_ in datos])],
    )
    return fig
