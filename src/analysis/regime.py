"""Observables por snapshot, series temporales, resumen por corrida y barridos de σ.

Une los módulos `moments`, `islands` y `correlation` en el flujo de
`Plan_observables_regimenes.md` §4:

    snaps --observables_timeseries--> DataFrame (una fila por snapshot)
          --summarize_run-----------> dict (β, v, promedios en la ventana final)
    varias corridas --sigma_scan_table--> + pendiente local n_v = d ln v / d ln σ

El código devuelve números, no etiquetas de régimen: los umbrales se calibran con
los barridos (plan §4).
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
import numpy as np
import pandas as pd

from .correlation import correlation_lengths, local_roughness_exponent
from .islands import island_statistics, step_statistics
from .moments import effective_levels, growing_fraction, height_moments

Snapshot = Tuple[float, np.ndarray, float]

# Observables que se promedian en la ventana final del resumen por corrida
LATE_COLUMNS: Tuple[str, ...] = (
    "W", "skewness", "excess_kurtosis", "bimodality", "n_eff_levels", "height_range",
    "growing_fraction", "n_islands", "island_density", "mean_island_size",
    "max_island_fraction", "step_density", "step_density_x", "step_density_y",
    "step_anisotropy", "multi_step_fraction", "xi_x", "xi_y", "xi_r", "alpha_local",
)


def surface_observables(heights: np.ndarray, with_alpha: bool = True) -> Dict[str, float]:
    """Todos los observables escalares de una matriz de alturas.

    Parámetros:
        heights: alturas enteras (Lx, Ly) de cualquiera de las dos líneas
        with_alpha: calcula también α_local (una FFT extra)

    Devuelve un diccionario plano (apto para una fila de DataFrame) con:
    momentos (mean_h, W, skewness, kurtosis, excess_kurtosis, bimodality), capas
    (n_eff_levels, height_range, growing_fraction), islas (n_islands, n_holes,
    island_density, mean_island_size, max_island_fraction), escalones
    (step_density[_x/_y], step_anisotropy, mean_abs_dh, multi_step_fraction) y
    correlación (xi_x, xi_y, xi_r, alpha_local).
    """
    h = np.asarray(heights, dtype=int)
    obs: Dict[str, float] = {}
    obs.update(height_moments(h))
    obs["n_eff_levels"] = effective_levels(h)
    obs["height_range"] = int(h.max() - h.min())
    obs["growing_fraction"] = growing_fraction(h)
    islands = island_statistics(h)
    # Solo los escalares (la lista de tamaños y el dict por nivel no van en la fila)
    for key in ("n_islands", "n_holes", "island_density", "mean_island_size",
                "max_island_fraction"):
        obs[key] = islands[key]
    obs.update(step_statistics(h))
    obs.update(correlation_lengths(h))
    obs["alpha_local"] = local_roughness_exponent(h) if with_alpha else float("nan")
    return obs


def observables_timeseries(snapshots: Sequence[Snapshot], every_n: int = 1,
                           with_alpha: bool = True) -> pd.DataFrame:
    """Serie temporal de observables a partir de snapshots (t, heights, conversion).

    Es el formato que devuelve `run()` en las dos líneas y que guardan `snaps.pkl` y
    `scripts/run_*`. Si la corrida paró por `max_events`, run() rellena los últimos
    snapshots con el estado final: esas filas repetidas se marcan con
    `repeated_state=True` y se excluyen de los ajustes.
    """
    if every_n < 1:
        raise ValueError("every_n debe ser >= 1")
    rows: List[Dict[str, float]] = []
    previous: Optional[np.ndarray] = None
    for t, heights, conversion in list(snapshots)[::every_n]:
        h = np.asarray(heights)
        row = {"t": float(t), "conversion": float(conversion)}
        row["repeated_state"] = previous is not None and np.array_equal(h, previous)
        row.update(surface_observables(h, with_alpha=with_alpha))
        rows.append(row)
        previous = h
    return pd.DataFrame(rows)


def growth_exponent(t: Iterable[float], W: Iterable[float], min_points: int = 3) -> float:
    """β del ajuste log W = β log t + c, con los puntos t > 0 y W > 0.

    NaN si hay menos de `min_points` puntos útiles.
    """
    t = np.asarray(list(t), dtype=float)
    W = np.asarray(list(W), dtype=float)
    sel = (t > 0) & (W > 0) & np.isfinite(t) & np.isfinite(W)
    if np.count_nonzero(sel) < min_points:
        return float("nan")
    return float(np.polyfit(np.log(t[sel]), np.log(W[sel]), 1)[0])


def growth_rate(t: Iterable[float], mean_h: Iterable[float]) -> float:
    """v = d⟨h⟩/dt (capas por unidad de tiempo), por ajuste lineal. NaN con < 2 puntos."""
    t = np.asarray(list(t), dtype=float)
    mean_h = np.asarray(list(mean_h), dtype=float)
    if t.size < 2 or np.ptp(t) == 0:
        return float("nan")
    return float(np.polyfit(t, mean_h, 1)[0])


def summarize_run(ts: pd.DataFrame, late_fraction: float = 0.5,
                  columns: Sequence[str] = LATE_COLUMNS) -> Dict[str, float]:
    """Resumen de una corrida a partir de su serie temporal.

    Devuelve:
        n_snapshots: filas usadas (sin estados repetidos)
        t_final: último tiempo
        beta: exponente de crecimiento de W(t)
        growth_rate: v = d⟨h⟩/dt
        <col>_late: media (ignorando NaN) de cada observable en la fracción final
            `late_fraction` de los snapshots (ventana cuasi-estacionaria). Los valores
            inf (ξ que no decae) se promedian como inf.
    """
    if not 0 < late_fraction <= 1:
        raise ValueError("late_fraction debe estar en (0, 1]")
    data = ts[~ts["repeated_state"]] if "repeated_state" in ts else ts
    out: Dict[str, float] = {
        "n_snapshots": int(len(data)),
        "t_final": float(data["t"].iloc[-1]) if len(data) else float("nan"),
        "beta": growth_exponent(data["t"], data["W"]),
        "growth_rate": growth_rate(data["t"], data["mean_h"]),
    }
    n_late = max(1, int(np.ceil(late_fraction * len(data))))
    late = data.iloc[-n_late:]
    for col in columns:
        values = late[col].to_numpy(dtype=float)
        finite_or_inf = values[~np.isnan(values)]
        out[f"{col}_late"] = float(np.mean(finite_or_inf)) if finite_or_inf.size else float("nan")
    return out


def sigma_scan_table(summary: pd.DataFrame, sigma_col: str = "sigma",
                     v_col: str = "growth_rate") -> pd.DataFrame:
    """Ordena un barrido por σ y añade la pendiente local n_v = d ln v / d ln σ.

    n_v ≈ 2 en el régimen espiral (BCF), grande en nucleación 2D y ≈ 1 en rugoso
    (plan §1). Solo tiene sentido si todas las filas comparten el resto de
    condiciones (K0, time_scale, red...); filtra antes si la carpeta las mezcla.
    Los puntos con σ ≤ 0 o v ≤ 0 quedan con NaN.
    """
    df = summary.sort_values(sigma_col).reset_index(drop=True).copy()
    sigma = df[sigma_col].to_numpy(dtype=float)
    v = df[v_col].to_numpy(dtype=float)
    n_v = np.full(len(df), np.nan)
    ok = (sigma > 0) & (v > 0) & np.isfinite(v)
    if np.count_nonzero(ok) >= 2:
        n_v[ok] = np.gradient(np.log(v[ok]), np.log(sigma[ok]))
    df["n_v"] = n_v
    return df


def _nice_log_ticks(values: np.ndarray) -> List[float]:
    """Ticks legibles para un eje logarítmico de σ: valores 1-2-3-5 dentro del rango."""
    lo, hi = float(np.min(values)), float(np.max(values))
    nice = [m * 10.0 ** e for e in range(-3, 4) for m in (1, 2, 3, 5)]
    ticks = [t for t in nice if lo / 1.05 <= t <= hi * 1.05]
    return ticks if len(ticks) >= 2 else sorted({lo, hi})


def plot_observables_vs_sigma(scan: pd.DataFrame, sigma_col: str = "sigma",
                              title: Optional[str] = None,
                              save_path: Optional[str] = None, show: bool = True):
    """Panel de indicadores de régimen frente a σ (salida de `sigma_scan_table`).

    Paneles: v (log-log), n_v, β, W, índice de bimodalidad b (con las referencias
    1 = dos niveles y 1/3 = gaussiana), niveles efectivos, densidad de islas,
    ξ_r, densidad de escalones y fracción de escalones múltiples.
    """
    panels = [
        ("growth_rate", "v = d⟨h⟩/dt", "loglog"),
        ("n_v", "n_v = d ln v / d ln σ", None),
        ("beta", "β (W ~ t^β)", None),
        ("W_late", "W", None),
        ("bimodality_late", "b = (γ₃²+1)/κ", None),
        ("n_eff_levels_late", "niveles efectivos", None),
        ("island_density_late", "densidad de islas", None),
        ("xi_r_late", "ξ_r (sitios)", None),
        ("step_density_late", "densidad de escalones", None),
        ("multi_step_fraction_late", "fracción |Δh| ≥ 2", None),
    ]
    panels = [p for p in panels if p[0] in scan.columns]
    ncols = 5
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             squeeze=False)
    sigma = scan[sigma_col].to_numpy(dtype=float)
    for ax, (col, label, scale) in zip(axes.ravel(), panels):
        y = scan[col].to_numpy(dtype=float)
        finite = np.isfinite(y)
        ax.plot(sigma[finite], y[finite], "o-", color="#2B8A96")
        if scale == "loglog":
            positive = finite & (y > 0) & (sigma > 0)
            if np.count_nonzero(positive) >= 1:
                ax.set_xscale("log")
                ax.set_yscale("log")
                # Etiquetas legibles: ticks "redondos" dentro del rango, sin etiquetas menores
                ax.xaxis.set_major_locator(FixedLocator(_nice_log_ticks(sigma[positive])))
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_major_formatter(FormatStrFormatter("%g"))
                    axis.set_minor_formatter(NullFormatter())
        if col == "bimodality_late":
            # Referencias exactas: dos niveles (capa a capa) y gaussiana (rugoso)
            ax.axhline(1.0, color="#8B1212", ls="--", lw=1, label="2 niveles")
            ax.axhline(1 / 3, color="gray", ls=":", lw=1, label="gaussiana")
            ax.legend(fontsize=7, frameon=False)
        if col == "n_v":
            for ref in (1.0, 2.0):
                ax.axhline(ref, color="gray", ls=":", lw=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("σ")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"💾 Figura guardada en: {save_path}")
    if show:
        plt.show()
    return fig
