"""Correlación altura-altura y longitudes de correlación (O3).

La red es periódica, así que la autocorrelación calculada con FFT es exacta (no hay
efectos de borde). Convenciones:
- C(r) = ⟨δh(r0 + r) δh(r0)⟩, con δh = h − ⟨h⟩; C(0) = W².
- G(r) = ⟨(h(r0 + r) − h(r0))²⟩ = 2 (W² − C(r)).
- Eje 0 = x, eje 1 = y (misma convención que `LatticeSOSStatic`).
"""

from typing import Dict, Tuple

import numpy as np

_INV_E = float(np.exp(-1.0))
_FLAT_VARIANCE = 1e-12


def autocorrelation(heights: np.ndarray) -> np.ndarray:
    """Autocorrelación periódica C(dx, dy), de la misma forma que `heights`.

    C[0, 0] = W². C[dx, dy] es la correlación para un desplazamiento (dx, dy); por
    periodicidad, dx y Lx − dx son el mismo desplazamiento con signo contrario.
    """
    h = np.asarray(heights, dtype=float)
    dh = h - h.mean()
    spectrum = np.fft.fft2(dh)
    # Teorema de Wiener-Khinchin: C = IFFT(|F|²) / N
    return np.real(np.fft.ifft2(spectrum * np.conj(spectrum))) / h.size


def _first_crossing(c: np.ndarray, level: float) -> float:
    """Primera distancia (interpolada) en que la curva c(r), r = 0, 1, ..., cae por
    debajo de `level`; inf si no cae."""
    below = np.nonzero(c < level)[0]
    if below.size == 0:
        return float("inf")
    r = int(below[0])
    # Interpolación lineal entre r − 1 (por encima) y r (por debajo); c[0] = 1 > level
    return (r - 1) + (c[r - 1] - level) / (c[r - 1] - c[r])


def _radial_profile(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Promedio de `values` (misma forma que la red) por distancia radial entera.

    Usa la distancia de imagen mínima, que es la que corresponde a una red periódica,
    y llega hasta r = min(Lx, Ly) // 2.
    """
    nx, ny = values.shape
    ix = np.minimum(np.arange(nx), nx - np.arange(nx))
    iy = np.minimum(np.arange(ny), ny - np.arange(ny))
    dist = np.sqrt(ix[:, None] ** 2 + iy[None, :] ** 2)
    r_bin = np.rint(dist).astype(int)
    r_max = min(nx, ny) // 2
    keep = r_bin <= r_max
    sums = np.bincount(r_bin[keep], weights=values[keep], minlength=r_max + 1)
    counts = np.bincount(r_bin[keep], minlength=r_max + 1)
    r = np.arange(r_max + 1)
    return r, sums / np.maximum(counts, 1)


def correlation_lengths(heights: np.ndarray) -> Dict[str, float]:
    """Longitudes de correlación ξ_x, ξ_y y ξ_r (radial).

    ξ es la primera distancia a la que C(r)/C(0) cae por debajo de 1/e (con
    interpolación lineal), buscando hasta L/2. Vale:
    - inf si no cae en ese rango: la superficie está correlacionada en toda la red;
    - NaN si W = 0 (superficie plana, correlación indefinida).
    """
    h = np.asarray(heights, dtype=float)
    C = autocorrelation(h)
    if C[0, 0] <= _FLAT_VARIANCE:
        nan = float("nan")
        return {"xi_x": nan, "xi_y": nan, "xi_r": nan}
    c = C / C[0, 0]
    nx, ny = c.shape
    _, c_rad = _radial_profile(c)
    return {
        "xi_x": _first_crossing(c[: nx // 2 + 1, 0], _INV_E),
        "xi_y": _first_crossing(c[0, : ny // 2 + 1], _INV_E),
        "xi_r": _first_crossing(c_rad, _INV_E),
    }


def height_difference_correlation(heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """G(r) = ⟨(h(r0 + r) − h(r0))²⟩ promediada radialmente, r = 0..min(Lx, Ly)//2.

    Satura en 2W² para r ≫ ξ; para r < ξ, G(r) ~ r^(2α).
    """
    C = autocorrelation(heights)
    r, c_rad = _radial_profile(C)
    return r, 2.0 * (C[0, 0] - c_rad)


def local_roughness_exponent(heights: np.ndarray) -> float:
    """Exponente de rugosidad local α (indicativo) a partir de G(r) ~ r^(2α).

    Ajuste log-log para r en [1, r_max], con r_max = ξ_r/2 (al menos 3, como mucho
    L/2; L/4 si ξ_r es infinita). Es solo indicativo: el α de Family-Vicsek requiere
    barridos en el tamaño de red. NaN si W = 0 o si hay menos de dos puntos útiles.
    """
    h = np.asarray(heights)
    xi_r = correlation_lengths(h)["xi_r"]
    if np.isnan(xi_r):
        return float("nan")
    r, G = height_difference_correlation(h)
    half = min(h.shape) // 2
    r_max = half // 2 if np.isinf(xi_r) else int(xi_r / 2)
    r_max = int(min(max(r_max, 3), half))
    sel = (r >= 1) & (r <= r_max) & (G > 0)
    if np.count_nonzero(sel) < 2:
        return float("nan")
    slope = np.polyfit(np.log(r[sel]), np.log(G[sel]), 1)[0]
    return float(slope / 2.0)


def correlation_extrema(heights: np.ndarray) -> Dict[str, float]:
    """Cero y mínimo de la correlación radial normalizada c(r) = C(r)/C(0).

    En nucleación 2D, alturas a una distancia ~ media separación entre islas tienden a
    ser distintas (isla frente a terraza), así que c(r) se hace negativa: su primer
    cero r_0 y su mínimo r_min dan una escala lateral independiente de la de las islas
    (para islas equiespaciadas a distancia ℓ, r_min ≈ ℓ/2). Cuanto más profundo es el
    mínimo (c_min más negativo), más regular es la separación. En una superficie
    rugosa sin escala lateral, c(r) decae sin un mínimo marcado (c_min ≈ 0).

    Como Σ_r C(r) = 0 en una red periódica (se resta la media), c(r) siempre tiene
    alguna parte negativa; por eso se usa el mínimo global de c(r) para 1 ≤ r ≤ L/2.

    Devuelve r_zero (inf si no cruza 0), r_min, c_min; NaN si W = 0.
    """
    C = autocorrelation(heights)
    if C[0, 0] <= _FLAT_VARIANCE:
        nan = float("nan")
        return {"r_zero": nan, "r_min": nan, "c_min": nan}
    r, c = _radial_profile(C / C[0, 0])
    i_min = 1 + int(np.argmin(c[1:]))
    return {
        "r_zero": _first_crossing(c, 0.0),
        "r_min": float(r[i_min]),
        "c_min": float(c[i_min]),
    }
