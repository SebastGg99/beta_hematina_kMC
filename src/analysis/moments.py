"""Momentos de la distribución de alturas y estructura de capas (O1, O2).

Todas las funciones reciben la matriz de alturas `heights` (enteros, forma (Lx, Ly))
de cualquiera de las dos líneas y tratan la red completa como la población (sin
corrección muestral). Ver `Plan_observables_regimenes.md` §2-§3.
"""

from typing import Dict, Tuple

import numpy as np

# Por debajo de esta varianza la superficie se considera plana (W = 0)
_FLAT_VARIANCE = 1e-12


def height_moments(heights: np.ndarray) -> Dict[str, float]:
    """Momentos de la distribución de alturas.

    Devuelve:
        mean_h: altura media μ
        variance: μ2 = W²
        W: rugosidad RMS (= np.std(heights) = src.common.observables.roughness)
        skewness: γ3 = μ3 / W³
        kurtosis: κ = μ4 / W⁴ (Pearson; 3 para una gaussiana)
        excess_kurtosis: κ − 3
        bimodality: b = (γ3² + 1) / κ. Vale 1 exactamente si hay dos niveles,
            2/3 con tres niveles equiprobables y → 1/3 para alturas gaussianas.

    Con W = 0 (superficie plana) γ3, κ y b quedan indefinidos: NaN.
    """
    h = np.asarray(heights, dtype=float).ravel()
    mu = float(h.mean())
    d = h - mu
    m2 = float(np.mean(d * d))
    out = {"mean_h": mu, "variance": m2, "W": float(np.sqrt(m2))}
    if m2 <= _FLAT_VARIANCE:
        nan = float("nan")
        out.update(skewness=nan, kurtosis=nan, excess_kurtosis=nan, bimodality=nan)
        return out
    m3 = float(np.mean(d ** 3))
    m4 = float(np.mean(d ** 4))
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2
    out.update(
        skewness=skew,
        kurtosis=kurt,
        excess_kurtosis=kurt - 3.0,
        bimodality=(skew * skew + 1.0) / kurt,
    )
    return out


def height_histogram(heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Niveles presentes y fracción de columnas en cada uno: (levels, p), Σp = 1."""
    levels, counts = np.unique(np.asarray(heights, dtype=int), return_counts=True)
    return levels, counts / counts.sum()


def layer_coverages(heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cobertura de cada capa incompleta.

    Para cada nivel z en [min(h), max(h) − 1], θ(z) = fracción de columnas con h > z
    (las que tienen ocupada la celda z). Todas cumplen 0 < θ(z) < 1: son las capas
    expuestas. Una superficie plana devuelve arrays vacíos.
    """
    h = np.asarray(heights, dtype=int)
    z = np.arange(int(h.min()), int(h.max()))
    theta = np.array([np.mean(h > level) for level in z], dtype=float)
    return z, theta


def effective_levels(heights: np.ndarray) -> float:
    """Número efectivo de niveles expuestos, exp(H) con H la entropía de Shannon.

    Vale 1 si la superficie es plana, entre 1 y 2 si ocupa dos niveles (capa a capa)
    y crece ∝ W si es rugosa. A diferencia de `max − min`, un adátomo aislado apenas
    lo cambia.
    """
    _, p = height_histogram(heights)
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def growing_fraction(heights: np.ndarray) -> float:
    """Fracción de columnas por encima de la última capa completa (h > min(h)).

    Es la "fracción roja" del estilo "paper" del Plotter: en el régimen capa a capa
    oscila de 0 a 1 con cada capa.
    """
    h = np.asarray(heights)
    return float(np.mean(h > h.min()))
