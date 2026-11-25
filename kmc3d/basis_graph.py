"""Grafo cristalino con base (varias unidades de crecimiento por celda), como en MONTY.

`CrystalGraph` (graph.py) describe una red de Bravais con UNA unidad por celda. MONTY
(Boerrigter et al., J. Phys. Chem. A 108, 5894, 2004, §2.4) trata cualquier estructura:
cada molécula de la celda es una unidad de crecimiento (GU) propia y el "grafo
cristalino" es la lista de interacciones entre pares de GU, cada una con su energía.
`BasisGraph` es ese grafo:

- una celda (`Cell`), en la práctica la celda CONVENCIONAL de la estructura, con Z
  GU en las posiciones fraccionarias x_s (s = 0..Z−1, la "base");
- enlaces (i, j, Δ, φ): la GU i de la celda R interactúa con la GU j de la celda
  R + Δ con energía φ > 0 (atractiva). Cada enlace implica también el inverso
  (j, i, −Δ), que no se escribe aparte.

MONTY usa como celda de simulación una "celda de capa" del mismo volumen que la celda
unidad y trata cada GU como una entidad única, sin usar la simetría. Aquí se hace lo
mismo con la celda convencional entera, así que no hace falta pasar a la celda primitiva
ni tener en cuenta las extinciones: (002) es la dirección (001) con capas de espesor
d_002 = d_001/2.

Energías: en las unidades que diga `units` ("kcal/mol" para los grafos publicados,
"kT" para simular). `to_kT(T)` convierte con R = 8.314462618/4184 kcal mol⁻¹ K⁻¹.

Energía de red por GU (la de la posición kink, que es neutra en equilibrio):

    E_cr = Σ_b φ_b / Z ,

porque cada enlace b aparece una vez por celda y une dos GU. Con Z = 1 coincide con
`CrystalGraph.lattice_energy` (= Σ_t φ_t).

Capas y energía de unión para una dirección (hkl) (Hartman-Perdok, generalizado a una
base): la altura de la GU (R, s) sobre los planos (hkl) es z = (hkl)·(R + x_s), en
unidades de d_hkl. Una capa de espesor d_hkl es el conjunto de GU con
z ∈ [z0 + m, z0 + m + 1). Con un corte z0 dado:
- E_slice = energía de los enlaces con los dos extremos en la capa, por GU de la capa;
- E_att = energía de los enlaces que unen la capa con las de debajo, por GU de la capa
  (lo que se libera al añadir una capa sobre la superficie);
- E_cut = energía de todos los enlaces que cruzan el plano z0, por celda 2D (la
  energía de la superficie recién cortada).
`best_termination` elige el corte z0 que minimiza E_cut (la superficie plana más
estable), que es el que usa `slab.SlabKMC` para la superficie inicial.
"""

import json
from dataclasses import dataclass
from itertools import product
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .geometry import Cell, hkl_label, reduce_hkl
from .graph import CrystalGraph

# Constante de los gases en kcal/(mol·K): R = 8.314462618 J/(mol·K) / 4184 J/kcal
R_KCAL = 8.314462618 / 4184.0

# Tolerancia para comparar posiciones fraccionarias
_TOL = 1e-6


def kcal_to_kT(x: float, T: float) -> float:
    """Energía en kcal/mol -> unidades de kT a la temperatura T (K)."""
    return float(x) / (R_KCAL * float(T))


def kT_to_kcal(x: float, T: float) -> float:
    """Energía en unidades de kT -> kcal/mol a la temperatura T (K)."""
    return float(x) * R_KCAL * float(T)


@dataclass(frozen=True)
class BasisBond:
    """Interacción entre la GU `i` de la celda R y la GU `j` de la celda R + `shift`."""

    i: int
    j: int
    shift: Tuple[int, int, int]
    energy: float               # φ > 0 (atractiva), en las unidades del grafo
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "i", int(self.i))
        object.__setattr__(self, "j", int(self.j))
        object.__setattr__(self, "shift", tuple(int(x) for x in self.shift))
        object.__setattr__(self, "energy", float(self.energy))
        if self.i == self.j and self.shift == (0, 0, 0):
            raise ValueError("Una GU no puede enlazarse consigo misma")

    def canonical(self) -> Tuple[int, int, Tuple[int, int, int]]:
        """Forma canónica: (i, j, Δ) y (j, i, −Δ) son el mismo enlace."""
        neg = tuple(-x for x in self.shift)
        return min((self.i, self.j, self.shift), (self.j, self.i, neg))


@dataclass(frozen=True)
class Slot:
    """Un vecino de una GU de la subred s: la GU `j` de la celda R + `shift`."""

    j: int
    shift: Tuple[int, int, int]
    energy: float
    bond: int                   # índice del enlace en `BasisGraph.bonds`


class BasisGraph:
    """Celda + base de Z GU + enlaces entre pares de GU (grafo cristalino de MONTY)."""

    def __init__(self, cell: Cell, basis: Sequence[Sequence[float]], bonds: Sequence[BasisBond],
                 units: str = "kT", name: str = "", notes: str = "",
                 basis_labels: Optional[Sequence[str]] = None):
        """
        Parámetros:
            cell: celda (en la práctica la convencional)
            basis: posiciones fraccionarias (Z, 3) de los centros de las GU
            bonds: enlaces `BasisBond` (sin repetir el inverso)
            units: "kT" o "kcal/mol"
            name, notes: descripción (se guardan en el JSON)
            basis_labels: nombre de cada GU (por defecto "M1", "M2", ...)
        """
        self.cell = cell
        self.basis = np.asarray(basis, dtype=float).reshape(-1, 3)
        if len(self.basis) == 0:
            raise ValueError("La base está vacía")
        self.bonds = tuple(bonds)
        if not self.bonds:
            raise ValueError("Hace falta al menos un enlace")
        self.units = str(units)
        self.name = name
        self.notes = notes
        self.basis_labels = (list(basis_labels) if basis_labels is not None
                             else [f"M{s + 1}" for s in range(len(self.basis))])
        if len(self.basis_labels) != len(self.basis):
            raise ValueError("Hace falta una etiqueta por GU de la base")
        nb = len(self.basis)
        seen = set()
        for b in self.bonds:
            if not (0 <= b.i < nb and 0 <= b.j < nb):
                raise ValueError(f"Enlace con una GU fuera de la base: {b}")
            if not b.energy > 0.0:
                raise ValueError(f"Las energías deben ser > 0 (atractivas): {b}")
            key = b.canonical()
            if key in seen:
                raise ValueError(f"Enlace repetido (o su inverso): {b}")
            seen.add(key)
        # Vecinos de cada subred: el enlace (i, j, Δ) da a i el vecino (j, +Δ) y a j el
        # vecino (i, −Δ); si i == j, la misma subred recibe los dos
        self._slots: List[List[Slot]] = [[] for _ in range(nb)]
        for k, b in enumerate(self.bonds):
            self._slots[b.i].append(Slot(b.j, b.shift, b.energy, k))
            self._slots[b.j].append(Slot(b.i, tuple(-x for x in b.shift), b.energy, k))

    # ------------------------------------------------------------------
    # Propiedades básicas
    # ------------------------------------------------------------------
    @property
    def n_basis(self) -> int:
        """Número de GU por celda (Z)."""
        return len(self.basis)

    def slots(self, s: int) -> List[Slot]:
        """Vecinos de una GU de la subred s (en el orden de los enlaces)."""
        return list(self._slots[s])

    def coordination(self, s: int) -> int:
        """Número de vecinos de una GU de la subred s."""
        return len(self._slots[s])

    def full_energy(self, s: int) -> float:
        """Energía de enlace de una GU de la subred s con todos sus vecinos."""
        return float(sum(sl.energy for sl in self._slots[s]))

    @property
    def lattice_energy(self) -> float:
        """E_cr = Σ_b φ_b / Z: energía de red por GU (= energía de la posición kink)."""
        return float(sum(b.energy for b in self.bonds)) / self.n_basis

    @property
    def kink_energy(self) -> float:
        """Energía de la posición kink (media-cristal): igual a `lattice_energy`."""
        return self.lattice_energy

    def bond_vector(self, k: int) -> np.ndarray:
        """Vector cartesiano del enlace k, del centro de la GU i al de la GU j."""
        b = self.bonds[k]
        frac = np.asarray(b.shift, float) + self.basis[b.j] - self.basis[b.i]
        return frac @ self.cell.matrix

    def bond_table(self) -> pd.DataFrame:
        """Una fila por enlace: GU, traslación, longitud entre centros y energía."""
        rows = []
        for k, b in enumerate(self.bonds):
            rows.append({"enlace": b.label or f"b{k + 1}", "i": self.basis_labels[b.i],
                         "j": self.basis_labels[b.j], "shift": b.shift,
                         "longitud": float(np.linalg.norm(self.bond_vector(k))),
                         f"phi ({self.units})": b.energy})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Unidades
    # ------------------------------------------------------------------
    def scaled(self, factor: float, units: Optional[str] = None) -> "BasisGraph":
        """Copia con todas las energías multiplicadas por `factor`."""
        bonds = [BasisBond(b.i, b.j, b.shift, b.energy * factor, b.label) for b in self.bonds]
        return BasisGraph(self.cell, self.basis, bonds, units or self.units, self.name,
                          self.notes, self.basis_labels)

    def to_kT(self, T: float) -> "BasisGraph":
        """Copia con las energías en kT a la temperatura T (K). Necesita units = "kcal/mol"."""
        if self.units == "kT":
            return self
        if self.units != "kcal/mol":
            raise ValueError(f"No sé convertir {self.units!r} a kT")
        return self.scaled(1.0 / (R_KCAL * float(T)), "kT")

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------
    @classmethod
    def from_crystal_graph(cls, g: CrystalGraph) -> "BasisGraph":
        """El mismo cristal de `CrystalGraph` (una GU por celda), como grafo con base."""
        bonds = [BasisBond(0, 0, b.vector, b.energy, b.label) for b in g.bonds]
        return cls(g.cell, [[0.0, 0.0, 0.0]], bonds, units="kT", name=g.name, notes=g.notes)

    def supercell(self, n1: int, n2: int, n3: int) -> "BasisGraph":
        """La misma estructura descrita con una celda n1 × n2 × n3 veces mayor.

        La base nueva tiene Z·n1·n2·n3 GU (las imágenes de cada GU, en el orden
        (p, q, r) y luego s) y cada enlace aparece n1·n2·n3 veces. Sirve para comprobar
        que el motor da lo mismo con base 1 y con base > 1.
        """
        n = np.array([n1, n2, n3], dtype=int)
        if np.any(n < 1):
            raise ValueError("Los factores de la supercelda deben ser >= 1")
        cell = Cell(self.cell.a * n1, self.cell.b * n2, self.cell.c * n3,
                    self.cell.alpha, self.cell.beta, self.cell.gamma)
        images = list(product(range(n1), range(n2), range(n3)))
        index = {}
        basis, labels = [], []
        for p in images:
            for s in range(self.n_basis):
                index[(p, s)] = len(basis)
                basis.append((self.basis[s] + np.array(p)) / n)
                labels.append(f"{self.basis_labels[s]}@{p[0]}{p[1]}{p[2]}")
        bonds = []
        for b in self.bonds:
            for p in images:
                tgt = np.array(p) + np.array(b.shift)
                shift = np.floor_divide(tgt, n)
                q = tuple(int(x) for x in tgt - shift * n)
                bonds.append(BasisBond(index[(p, b.i)], index[(q, b.j)], tuple(int(x) for x in shift),
                                       b.energy, b.label))
        return BasisGraph(cell, basis, bonds, self.units, f"{self.name} ({n1}x{n2}x{n3})",
                          self.notes, labels)

    @classmethod
    def from_centers(cls, cell: Cell, centers: Sequence[Sequence[float]],
                     bonds: Sequence[Dict[str, object]],
                     symops: Optional[Sequence[Tuple[Sequence[Sequence[float]], Sequence[float]]]] = None,
                     units: str = "kcal/mol", name: str = "", notes: str = "",
                     labels: Optional[Sequence[str]] = None) -> "BasisGraph":
        """Grafo a partir de centros moleculares y enlaces dados por coordenadas.

        Pensado para transcribir tablas publicadas, que suelen dar solo las moléculas
        independientes y, para cada enlace, la posición de la molécula vecina.

        Parámetros:
            centers: posiciones fraccionarias de las moléculas independientes
            bonds: lista de dicts {"from": índice en `centers`, "to_frac": posición
                fraccionaria del centro vecino (cualquier celda), "energy": φ,
                "label": opcional}
            symops: operaciones de simetría (W, t) sobre coordenadas fraccionarias,
                x' = W·x + t, incluida la identidad. Si se dan, la base se completa con
                las imágenes de `centers` y cada enlace se replica con cada operación
                (los repetidos se descartan). None = solo la identidad.
            labels: nombre de cada molécula independiente
        """
        ops = [(np.eye(3), np.zeros(3))] if not symops else \
            [(np.asarray(W, float), np.asarray(t, float)) for W, t in symops]
        centers = np.asarray(centers, float).reshape(-1, 3)
        labels = list(labels) if labels is not None else [f"M{i + 1}" for i in range(len(centers))]
        # Base completa: imágenes de los centros, reducidas a [0, 1) y sin repetir
        basis: List[np.ndarray] = []
        blabels: List[str] = []

        def locate(x: np.ndarray) -> Optional[Tuple[int, Tuple[int, int, int]]]:
            """(s, celda) del centro x si está en la base (None si no)."""
            for s, xs in enumerate(basis):
                d = x - xs
                cellv = np.rint(d)
                if np.all(np.abs(d - cellv) < _TOL):
                    return s, tuple(int(v) for v in cellv)
            return None

        for k, (W, t) in enumerate(ops):
            for c, lab in zip(centers, labels):
                x = W @ c + t
                if locate(x) is None:
                    basis.append(x - np.floor(x + _TOL))
                    # la imagen por la operación k (k = 0 suele ser la identidad)
                    blabels.append(lab if k == 0 else f"{lab}_op{k}")
        out: List[BasisBond] = []
        seen = set()
        for W, t in ops:
            for e in bonds:
                xa = W @ centers[int(e["from"])] + t
                xb = W @ np.asarray(e["to_frac"], float) + t
                la, lb = locate(xa), locate(xb)
                if la is None or lb is None:
                    raise ValueError(f"El vecino {e['to_frac']} no coincide con ninguna GU de la base")
                (i, ca), (j, cb) = la, lb
                bond = BasisBond(i, j, tuple(b - a for a, b in zip(ca, cb)), float(e["energy"]),
                                 str(e.get("label", "")))
                key = bond.canonical()
                if key not in seen:
                    seen.add(key)
                    out.append(bond)
        return cls(cell, basis, out, units, name, notes, blabels)

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "notes": self.notes, "units": self.units,
                "cell": self.cell.to_dict(),
                "basis": [{"label": l, "frac": [float(x) for x in xs]}
                          for l, xs in zip(self.basis_labels, self.basis)],
                "bonds": [{"i": b.i, "j": b.j, "shift": list(b.shift), "energy": b.energy,
                           "label": b.label} for b in self.bonds]}

    @classmethod
    def from_dict(cls, d: dict) -> "BasisGraph":
        """Inversa de `to_dict` (acepta también "i"/"j" como etiquetas de la base)."""
        labels = [b["label"] for b in d["basis"]]
        pos = {l: k for k, l in enumerate(labels)}

        def idx(v):
            return pos[v] if isinstance(v, str) else int(v)

        bonds = [BasisBond(idx(b["i"]), idx(b["j"]), tuple(b["shift"]), b["energy"], b.get("label", ""))
                 for b in d["bonds"]]
        return cls(Cell(**d["cell"]), [b["frac"] for b in d["basis"]], bonds,
                   d.get("units", "kT"), d.get("name", ""), d.get("notes", ""), labels)

    def to_json(self, path: Union[str, Path], extra: Optional[dict] = None) -> Path:
        """Guarda el grafo (y campos extra, p. ej. fuente o caras) en un JSON."""
        path = Path(path)
        d = {**(extra or {}), **self.to_dict()}
        path.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        return path

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BasisGraph":
        """Lee un grafo guardado con `to_json` (los campos extra se ignoran)."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Capas (hkl): alturas, cortes y energías de Hartman-Perdok
    # ------------------------------------------------------------------
    def heights(self, hkl: Sequence[int]) -> np.ndarray:
        """Altura (hkl)·x_s de cada GU de la base, en unidades de d_hkl."""
        return self.basis @ np.asarray(hkl, float)

    def d_spacing(self, hkl: Sequence[int]) -> float:
        """Espesor de capa d_hkl SIN reducir los índices: d_002 = d_001 / 2."""
        g = np.asarray(hkl, float) @ self.cell.reciprocal
        return float(1.0 / np.linalg.norm(g))

    def _bond_heights(self, hkl: Sequence[int]) -> List[Tuple[float, float, float]]:
        """(z del extremo inferior, Δz > 0, φ) de cada enlace con Δz != 0, más los de Δz = 0."""
        h = np.asarray(hkl, float)
        zs = self.heights(hkl)
        out = []
        for b in self.bonds:
            za = zs[b.i]
            zb = float(np.dot(h, b.shift)) + zs[b.j]
            lo, dz = (za, zb - za) if zb >= za else (zb, za - zb)
            out.append((float(lo), float(dz), b.energy))
        return out

    @staticmethod
    def _gcd3(hkl: Sequence[int]) -> int:
        """Máximo común divisor de los tres índices (n en (hkl) = n·(h0 k0 l0))."""
        n = gcd(gcd(abs(int(hkl[0])), abs(int(hkl[1]))), abs(int(hkl[2])))
        if n == 0:
            raise ValueError("(hkl) = (000) no define un plano")
        return n

    def cut_candidates(self, hkl: Sequence[int]) -> List[float]:
        """Cortes z0 en [0, n) a mitad de camino entre alturas consecutivas de GU.

        Con (hkl) = n·(h0 k0 l0), las alturas se repiten con periodo n (una celda de la
        red tiene n capas de espesor d_hkl), así que basta mirar z0 en [0, n).
        """
        n = self._gcd3(hkl)
        z = np.unique(np.round(np.mod(self.heights(hkl), n), 9))
        nxt = np.r_[z[1:], z[0] + n]
        return [float(np.mod(0.5 * (a + b), n)) for a, b in zip(z, nxt)]

    def slice_energies(self, hkl: Sequence[int], z0: float) -> Dict[str, float]:
        """E_slice, E_att (por GU de la capa) y E_cut (por celda 2D) con el corte z0.

        La capa es [z0, z0 + 1) en unidades de d_hkl. Con (hkl) = n·(h0 k0 l0), la GU
        (R, s) está a la altura n·c + z_s, con c = (h0 k0 l0)·R entero y z_s = (hkl)·x_s,
        y cada columna de la red de planos tiene una celda por cada c. Los conteos son
        por columna (celda 2D):
        - n_slice: GU con altura en [z0, z0 + 1);
        - E_slice: enlaces con los dos extremos en la capa (por GU de la capa);
        - E_att: enlaces con el extremo superior en la capa y el inferior por debajo de
          z0 (por GU de la capa): la energía que se libera al añadir la capa;
        - E_cut: enlaces que cruzan el plano z0 (extremo inferior < z0 <= superior).
        """
        n = self._gcd3(hkl)

        def count_in(lo: float, hi: float, z: float) -> int:
            """Número de enteros c con n·c + z en [lo, hi) (0 si el intervalo está vacío)."""
            if hi <= lo:
                return 0
            return max(0, int(np.ceil((hi - z) / n - 1e-9) - np.ceil((lo - z) / n - 1e-9)))

        n_sl = sum(count_in(z0, z0 + 1.0, z) for z in self.heights(hkl))
        e_slice = e_att = e_cut = 0.0
        for lo, dz, phi in self._bond_heights(hkl):
            # extremo inferior a la altura n·c + lo y superior a n·c + lo + dz
            e_slice += phi * count_in(z0, z0 + 1.0 - dz, lo)
            if dz > 1e-9:
                e_att += phi * count_in(z0 - dz, min(z0, z0 + 1.0 - dz), lo)
                e_cut += phi * count_in(z0 - dz, z0, lo)
        return {"n_slice": n_sl, "E_slice": e_slice / n_sl if n_sl else np.nan,
                "E_att": e_att / n_sl if n_sl else np.nan, "E_cut": e_cut}

    def best_termination(self, hkl: Sequence[int]) -> float:
        """Corte z0 en [0, n) que minimiza E_cut (empates: mayor E_slice).

        Se descartan los cortes cuya capa no contiene ninguna GU.
        """
        best = None
        for z0 in self.cut_candidates(hkl):
            e = self.slice_energies(hkl, z0)
            if e["n_slice"] == 0:
                continue
            key = (round(e["E_cut"], 9), -round(e["E_slice"], 9))
            if best is None or key < best[0]:
                best = (key, z0)
        if best is None:
            raise ValueError(f"Ninguna capa {hkl_label(hkl)} contiene GU")
        return best[1]

    def face_table(self, faces: Iterable[Sequence[int]]) -> pd.DataFrame:
        """Tabla de caras con el mejor corte: d_hkl, E_slice, E_att (negativa), E_cut."""
        rows = []
        for f in faces:
            f = tuple(int(x) for x in f)
            z0 = self.best_termination(f)
            e = self.slice_energies(f, z0)
            rows.append({"hkl": hkl_label(f), "d_hkl": self.d_spacing(f), "z0": z0,
                         "n_slice": e["n_slice"], "E_slice": e["E_slice"], "E_att": -e["E_att"],
                         "E_cut": e["E_cut"]})
        df = pd.DataFrame(rows)
        if len(df):
            df["R_att_rel"] = df["E_att"].abs() / df["E_att"].abs().replace(0, np.nan).min()
        return df

    def __repr__(self) -> str:
        return (f"BasisGraph({self.name!r}, Z={self.n_basis}, enlaces={len(self.bonds)}, "
                f"E_cr={self.lattice_energy:.4g} {self.units})")


# ---------------------------------------------------------------------------
# Celda de capa (slice cell) para una dirección (hkl)
# ---------------------------------------------------------------------------
def _ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """(g, x, y) con a·x + b·y = g = mcd(a, b) >= 0."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        a, x0, y0 = -a, -x0, -y0
    return a, x0, y0


def _unimodular_completion(h0: Sequence[int]) -> np.ndarray:
    """Matriz entera U (3 × 3, det ±1) con h0 · U = (0, 0, 1), para h0 coprimos.

    Se reduce el vector fila h0 a (0, 0, 1) con operaciones enteras de columna (Euclides),
    aplicándolas también a U (que empieza como la identidad).
    """
    v = [int(x) for x in h0]
    U = np.eye(3, dtype=np.int64)

    def combine(p: int, q: int) -> None:
        """Deja en la columna q el mcd de v[p], v[q] y en la p un cero."""
        a, b = v[p], v[q]
        if a == 0:
            return
        g, x, y = _ext_gcd(b, a)          # b·x + a·y = g
        # nueva col q = x·col q + y·col p ; nueva col p = (a/g)·col q − (b/g)·col p
        cq = x * U[:, q] + y * U[:, p]
        cp = (a // g) * U[:, q] - (b // g) * U[:, p]
        U[:, q], U[:, p] = cq, cp
        v[q], v[p] = g, 0

    combine(0, 2)
    combine(1, 2)
    if v[2] == -1:
        U[:, 2] *= -1
        v[2] = 1
    if v != [0, 0, 1]:
        raise ValueError(f"Los índices {tuple(h0)} no son coprimos")
    return U


def _gauss_reduce(u: np.ndarray, v: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reducción de Lagrange-Gauss de la base 2D (u, v) con la métrica G."""
    def dot(x, y):
        return float(x @ G @ y)

    for _ in range(1000):
        if dot(v, v) < dot(u, u):
            u, v = v, u
        m = int(round(dot(u, v) / dot(u, u)))
        if m == 0:
            break
        v = v - m * u
    return u, v


def slice_cell(hkl: Sequence[int], cell: Cell) -> np.ndarray:
    """Base entera unimodular (filas u, v, w) de la celda de capa de la dirección (hkl).

    u y v están en el plano ((hkl)·u = (hkl)·v = 0) y forman una base reducida de
    Lagrange-Gauss con la métrica real de la celda: los vectores más cortos y con el
    ángulo más cercano a 90° posible, como la celda de capa de MONTY. w cumple
    (hkl)·w = 1 y se elige lo más corto posible (lo más cerca de la normal). det = +1.
    Con índices no coprimos se usa la dirección reducida.
    """
    h0 = np.asarray(reduce_hkl(hkl), dtype=np.int64)
    U = _unimodular_completion(h0)
    G = cell.matrix @ cell.matrix.T
    u, v = _gauss_reduce(U[:, 0].copy(), U[:, 1].copy(), G)
    w = U[:, 2].copy()
    # w más corto: restar la combinación entera de u, v más cercana a su proyección
    A = np.array([[u @ G @ u, u @ G @ v], [u @ G @ v, v @ G @ v]], dtype=float)
    rhs = np.array([u @ G @ w, v @ G @ w], dtype=float)
    p, q = np.linalg.solve(A, rhs)
    best = None
    for dp, dq in product((0, 1), repeat=2):
        cand = w - (int(np.floor(p)) + dp) * u - (int(np.floor(q)) + dq) * v
        length = float(cand @ G @ cand)
        if best is None or length < best[0]:
            best = (length, cand)
    w = best[1]
    B = np.array([u, v, w], dtype=np.int64)
    if round(np.linalg.det(B)) < 0:
        B[1] *= -1
    det = int(round(np.linalg.det(B)))
    assert det == 1 and int(h0 @ B[0]) == 0 and int(h0 @ B[1]) == 0 and int(h0 @ B[2]) == 1
    return B
