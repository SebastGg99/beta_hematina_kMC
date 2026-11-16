"""Grafo cristalino: tipos de enlace entre unidades de crecimiento y sus energías.

Idea tomada de MONTY (Boerrigter et al., J. Phys. Chem. A 108, 5894, 2004): la
estructura se describe como un grafo de interacciones entre unidades de crecimiento,
y la anisotropía sale sola de qué enlaces cruzan cada cara. Aquí la red es de Bravais
con UNA unidad por celda (cubitos en el espacio de índices). Esto cubre:
- cristales de Kossel y cúbicos anisótropos (celda cúbica);
- β-hematina: P-1 con Z = 2 hemos = 1 dímero centrosimétrico por celda, así que el
  dímero es la unidad de crecimiento y la celda triclínica es exacta.

Cada `BondType` es un vector de red v (enteros) con energía φ (en unidades de kT,
positiva = atractiva). Implica dos vecinos, r + v y r − v. Una unidad con todos sus
vecinos tiene energía 2·Σφ / 2 = Σφ por unidad (energía de red E_cr), que coincide
con la de la posición de media-cristal (kink), E_kink = Σφ.

Teoría de Hartman-Perdok para una cara (hkl) (índices coprimos):
- E_slice = Σ φ_t de los enlaces con (hkl)·v_t = 0 (dentro de la capa d_hkl);
- E_att   = Σ φ_t de los enlaces con (hkl)·v_t ≠ 0 (energía liberada al unirse una
  capa, por unidad); E_slice + E_att = E_cr. Se reporta E_att con signo negativo,
  como en la literatura (más negativo = cara más rápida y menos importante);
- cara F (plana) si los enlaces dentro de la capa generan una red 2D. Con una unidad
  por celda cada enlace es una cadena periódica (PBC), así que basta con que los
  vectores de enlace con φ > 0 dentro de la capa tengan rango 2. Si tienen rango 1
  es S (escalonada) y si no hay ninguno es K (con kinks).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .geometry import HKL, Cell, hkl_label, reduce_hkl


@dataclass(frozen=True)
class BondType:
    """Interacción entre una unidad en r y sus vecinas en r ± vector."""

    vector: Tuple[int, int, int]
    energy: float              # φ/kT > 0 (atractiva)
    name: str = ""

    def __post_init__(self):
        v = tuple(int(x) for x in self.vector)
        if v == (0, 0, 0):
            raise ValueError("El vector de enlace no puede ser [000]")
        object.__setattr__(self, "vector", v)
        object.__setattr__(self, "energy", float(self.energy))

    @property
    def label(self) -> str:
        return self.name or "[" + "".join(str(x) for x in self.vector) + "]"


@dataclass
class CrystalGraph:
    """Red de Bravais (una unidad por celda) + tipos de enlace."""

    cell: Cell
    bonds: Tuple[BondType, ...]
    name: str = ""
    notes: str = ""

    def __post_init__(self):
        self.bonds = tuple(self.bonds)
        if not self.bonds:
            raise ValueError("Hace falta al menos un tipo de enlace")
        seen = set()
        for b in self.bonds:
            neg = tuple(-x for x in b.vector)
            if b.vector in seen or neg in seen:
                raise ValueError(f"Enlace repetido (v y -v son el mismo tipo): {b.vector}")
            seen.add(b.vector)
        if len(self.bonds) > 19:
            # el código de vecindad es un entero base 3 en int32: 3^19 < 2^31
            raise ValueError("Como mucho 19 tipos de enlace (38 vecinos)")

    # ------------------------------------------------------------------
    # Propiedades básicas
    # ------------------------------------------------------------------
    @property
    def n_types(self) -> int:
        return len(self.bonds)

    @property
    def vectors(self) -> np.ndarray:
        """Vectores de enlace (T x 3, enteros)."""
        return np.array([b.vector for b in self.bonds], dtype=int)

    @property
    def energies(self) -> np.ndarray:
        """Energías φ_t/kT (T,)."""
        return np.array([b.energy for b in self.bonds], dtype=float)

    @property
    def lattice_energy(self) -> float:
        """E_cr = Σ φ_t: energía de red por unidad (= energía de la posición kink)."""
        return float(self.energies.sum())

    @property
    def kink_energy(self) -> float:
        """Energía de unión en la posición de media-cristal (kink): Σ φ_t."""
        return self.lattice_energy

    @property
    def reach(self) -> int:
        """Máximo |componente| de los vectores de enlace (ancho del borde de la caja)."""
        return int(np.abs(self.vectors).max())

    @property
    def coordination(self) -> int:
        """Número de vecinos de una unidad (2 por tipo de enlace)."""
        return 2 * self.n_types

    # ------------------------------------------------------------------
    # Código de vecindad: cuántos vecinos de cada tipo están ocupados
    # ------------------------------------------------------------------
    @property
    def code_weights(self) -> List[int]:
        """Peso 3^t de cada tipo: código = Σ_t n_t · 3^t con n_t ∈ {0, 1, 2}."""
        return [3 ** t for t in range(self.n_types)]

    @property
    def full_code(self) -> int:
        """Código de una unidad con todos los vecinos ocupados."""
        return sum(2 * w for w in self.code_weights)

    def decode(self, code: int) -> Tuple[int, ...]:
        """Código -> (n_0, ..., n_{T-1}), vecinos ocupados por tipo."""
        counts = []
        for _ in range(self.n_types):
            code, r = divmod(int(code), 3)
            counts.append(r)
        return tuple(counts)

    def encode(self, counts: Sequence[int]) -> int:
        return int(sum(int(n) * w for n, w in zip(counts, self.code_weights)))

    def code_energy(self, code: int) -> float:
        """Energía de enlace E = Σ_t n_t φ_t de un sitio con ese código."""
        return float(np.dot(self.decode(code), self.energies))

    def code_bonds(self, code: int) -> int:
        """Número total de vecinos ocupados del código."""
        return int(sum(self.decode(code)))

    # ------------------------------------------------------------------
    # Hartman-Perdok
    # ------------------------------------------------------------------
    def slice_attachment(self, hkl: Sequence[int]) -> Tuple[float, float]:
        """(E_slice, E_att) en kT por unidad, con E_att >= 0 (energía liberada)."""
        h = np.asarray(reduce_hkl(hkl))
        dots = self.vectors @ h
        e = self.energies
        return float(e[dots == 0].sum()), float(e[dots != 0].sum())

    def face_character(self, hkl: Sequence[int]) -> str:
        """'F', 'S' o 'K' según el rango de los enlaces (φ > 0) dentro de la capa."""
        h = np.asarray(reduce_hkl(hkl))
        inplane = [b.vector for b in self.bonds if b.energy > 0 and np.dot(b.vector, h) == 0]
        if not inplane:
            return "K"
        rank = np.linalg.matrix_rank(np.array(inplane, dtype=float))
        return "F" if rank >= 2 else "S"

    def face_table(self, faces: Iterable[Sequence[int]]) -> pd.DataFrame:
        """Tabla de caras: d_hkl, E_slice, E_att (negativa), carácter y R relativa.

        `R_att_rel` es la velocidad relativa del modelo de energía de unión
        (R ∝ |E_att|), normalizada por la cara más lenta de la lista.
        """
        rows = []
        for f in faces:
            r = reduce_hkl(f)
            es, ea = self.slice_attachment(r)
            rows.append({"hkl": hkl_label(r), "h": r[0], "k": r[1], "l": r[2],
                         "d_hkl": self.cell.d_spacing(r), "E_slice": es, "E_att": -ea,
                         "E_att_frac": ea / self.lattice_energy, "caracter": self.face_character(r)})
        df = pd.DataFrame(rows)
        if len(df):
            df["R_att_rel"] = df["E_att"].abs() / df["E_att"].abs().replace(0, np.nan).min()
        return df

    def bond_table(self) -> pd.DataFrame:
        """Tabla de enlaces: vector, longitud real, energía."""
        return pd.DataFrame([{"enlace": b.label, "vector": b.vector,
                              "longitud": self.cell.vector_length(b.vector),
                              "phi_kT": b.energy} for b in self.bonds])

    # ------------------------------------------------------------------
    # Serialización (metadata JSON)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "notes": self.notes, "cell": self.cell.to_dict(),
                "bonds": [{"vector": list(b.vector), "energy": b.energy, "name": b.name}
                          for b in self.bonds]}

    @classmethod
    def from_dict(cls, d: dict) -> "CrystalGraph":
        return cls(cell=Cell(**d["cell"]),
                   bonds=tuple(BondType(tuple(b["vector"]), b["energy"], b.get("name", ""))
                               for b in d["bonds"]),
                   name=d.get("name", ""), notes=d.get("notes", ""))

    def with_energies(self, energies: Dict[str, float]) -> "CrystalGraph":
        """Copia con energías cambiadas por etiqueta de enlace ({'[001]': 2.5, ...})."""
        new = []
        for b in self.bonds:
            new.append(BondType(b.vector, energies.get(b.label, b.energy), b.name))
        unknown = set(energies) - {b.label for b in self.bonds}
        if unknown:
            raise KeyError(f"Enlaces desconocidos: {sorted(unknown)}")
        return CrystalGraph(self.cell, tuple(new), self.name, self.notes)
