"""Geometría de la celda unidad: vectores reales, recíprocos y caras (hkl).

El motor trabaja en el espacio de ÍNDICES de la red de Bravais (i, j, k enteros: una
unidad de crecimiento por celda). La geometría real solo entra en:
- el espaciado interplanar d_hkl (para convertir posiciones de planos en distancias);
- las normales cartesianas de las caras y las coordenadas cartesianas (figuras y
  descriptores de forma).

Convenciones:
- `matrix` tiene por FILAS los vectores a, b, c en cartesianas (Å o unidades de red),
  con a sobre x y b en el plano xy (convención cristalográfica habitual).
- `reciprocal` tiene por filas a*, b*, c* SIN el factor 2π: a_i · a*_j = δ_ij.
- Un plano (hkl) contiene los puntos de red r = (i, j, k) con h·i + k·j + l·k = m
  (entero), y planos consecutivos están separados d_hkl = 1 / |h a* + k b* + l c*|.
"""

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Sequence, Tuple

import numpy as np

HKL = Tuple[int, int, int]


def reduce_hkl(hkl: Sequence[int]) -> HKL:
    """Índices de Miller coprimos: (2, 0, 0) -> (1, 0, 0).

    En una red primitiva con una unidad por celda, solo los (hkl) coprimos son
    planos reticulares; (200) no es una cara distinta de (100).
    """
    h, k, l = (int(x) for x in hkl)
    g = gcd(gcd(abs(h), abs(k)), abs(l))
    if g == 0:
        raise ValueError("(hkl) = (000) no define un plano")
    return (h // g, k // g, l // g)


def hkl_label(hkl: Sequence[int]) -> str:
    """Etiqueta cristalográfica: (0, 1, -1) -> '(01-1)'."""
    return "(" + "".join(str(int(x)) for x in hkl) + ")"


def parse_hkl(label: str) -> HKL:
    """'(01-1)' o '[01-1]' -> (0, 1, -1). Solo índices de un dígito."""
    s = label.strip("()[]{} ")
    out, i = [], 0
    while i < len(s):
        if s[i] == "-":
            out.append(-int(s[i + 1]))
            i += 2
        else:
            out.append(int(s[i]))
            i += 1
    if len(out) != 3:
        raise ValueError(f"Etiqueta no válida: {label!r}")
    return tuple(out)


@dataclass(frozen=True)
class Cell:
    """Celda unidad (a, b, c en Å o unidades arbitrarias; ángulos en grados)."""

    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    @property
    def matrix(self) -> np.ndarray:
        """Vectores reales a, b, c como filas (3 x 3), cartesianas."""
        al, be, ga = np.radians([self.alpha, self.beta, self.gamma])
        ca, cb, cg, sg = np.cos(al), np.cos(be), np.cos(ga), np.sin(ga)
        cy = (ca - cb * cg) / sg
        cz2 = 1.0 - cb ** 2 - cy ** 2
        if cz2 <= 0.0:
            raise ValueError("Ángulos de celda incompatibles (volumen no positivo)")
        return np.array([
            [self.a, 0.0, 0.0],
            [self.b * cg, self.b * sg, 0.0],
            [self.c * cb, self.c * cy, self.c * np.sqrt(cz2)],
        ])

    @property
    def reciprocal(self) -> np.ndarray:
        """Vectores recíprocos a*, b*, c* como filas (sin 2π): M · R^T = I."""
        return np.linalg.inv(self.matrix).T

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.matrix)))

    def reciprocal_parameters(self) -> Tuple[float, float, float, float, float, float]:
        """(a*, b*, c*, α*, β*, γ*) con ángulos en grados (para comprobar celdas)."""
        R = self.reciprocal
        lens = np.linalg.norm(R, axis=1)

        def ang(u, v):
            return float(np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))

        return (float(lens[0]), float(lens[1]), float(lens[2]),
                ang(R[1], R[2]), ang(R[0], R[2]), ang(R[0], R[1]))

    def d_spacing(self, hkl: Sequence[int]) -> float:
        """Espaciado interplanar d_hkl (de los índices reducidos)."""
        g = np.asarray(reduce_hkl(hkl), dtype=float) @ self.reciprocal
        return float(1.0 / np.linalg.norm(g))

    def normal(self, hkl: Sequence[int]) -> np.ndarray:
        """Normal cartesiana unitaria de la cara (hkl)."""
        g = np.asarray(hkl, dtype=float) @ self.reciprocal
        return g / np.linalg.norm(g)

    def to_cartesian(self, idx: np.ndarray) -> np.ndarray:
        """Índices de red (N x 3) -> coordenadas cartesianas (N x 3)."""
        return np.asarray(idx, dtype=float) @ self.matrix

    def vector_length(self, uvw: Sequence[int]) -> float:
        """Longitud del vector de red [uvw]."""
        return float(np.linalg.norm(np.asarray(uvw, dtype=float) @ self.matrix))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c,
                "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}


def family(hkl: Sequence[int], centrosymmetric: bool = True) -> Tuple[HKL, ...]:
    """Caras que se tratan como equivalentes: (hkl) y, si hay inversión, (-h-k-l).

    En P-1 (β-hematina) la única simetría es la inversión, así que {hkl} = ±(hkl).
    """
    f = reduce_hkl(hkl)
    if centrosymmetric:
        return (f, (-f[0], -f[1], -f[2]))
    return (f,)


def unique_faces(faces: Iterable[Sequence[int]]) -> Tuple[HKL, ...]:
    """Reduce y elimina repetidos conservando el orden."""
    out = []
    for f in faces:
        r = reduce_hkl(f)
        if r not in out:
            out.append(r)
    return tuple(out)
