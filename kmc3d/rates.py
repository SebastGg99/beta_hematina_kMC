"""Modelos de tasas para unión (attachment) y despegue (detachment).

Termodinámica (unidades de kT). Δμ = μ_solución − μ_cristal es la fuerza impulsora
(> 0 crece). Con la convención del resto del proyecto, Δμ = S = ln(1 + σ) con
σ = C/C_eq − 1 (`delta_mu_from_sigma`). Unir una unidad a un sitio cuya energía de
enlace es E (suma de φ de los vecinos ocupados) cambia la energía libre en

    ΔG(E) = −[Δμ + (E − E_kink)] ,      E_kink = Σ_t φ_t ,

de modo que en equilibrio (Δμ = 0) la posición kink no gana ni pierde (ΔG = 0).

`SplitRates` (familia general con balance detallado k+/k− = exp(−ΔG)):

    k+(E) = k0 · exp( a·Δμ + b·(E − E_kink) )
    k−(E) = k0 · exp( −(1 − a)·Δμ − (1 − b)·(E − E_kink) )

- a = 1, b = 0 (`gilmer`): k+ = k0 e^{Δμ} igual para todos los sitios y k− depende
  solo de los enlaces. Es el modelo clásico de Gilmer y Bennema para Kossel/SOS.
- a = b = 1/2 (`symmetric`): reparte por igual la fuerza impulsora y el cambio de
  energía entre unión y despegue. Es el tipo de esquema que usan MONTY y
  CrystalGrower; la forma exacta de sus ecuaciones NO se ha verificado aquí (ver
  PLAN_3DkMC.md, decisión D2).
- b > 0 hace la UNIÓN selectiva por sitio, como la tasa adaptativa de Nagpal et al.
  (2024), pero manteniendo el balance detallado.

`AdaptiveRates` reproduce la forma del motor (2+1)D estático del proyecto (Nagpal):
k+ = K0 exp(S + Σ_t n_t δ_t / S), k− = K0 exp(φ0 − E). Sirve para comparar con la
línea 2D; NO cumple balance detallado en general.

Todas las tasas se calculan por CÓDIGO de vecindad (ver `CrystalGraph.decode`) y se
guardan en caché: el motor solo las pide al crear una clase nueva.
"""

from itertools import product
from math import log1p
from typing import Dict, Optional, Sequence, Union

from . import _paths  # noqa: F401  (pone el repositorio en sys.path)
from src.common.numerics import _safe_exp

from .graph import CrystalGraph


def delta_mu_from_sigma(sigma: float) -> float:
    """Δμ/kT = S = ln(1 + σ), misma convención que la línea estática 2D."""
    if sigma <= -1.0:
        raise ValueError("σ debe ser > −1")
    return float(log1p(sigma))


class RateModel:
    """Base: caché de energías por código y de tasas por (tipo, código)."""

    kind = "base"

    def __init__(self, graph: CrystalGraph, delta_mu: float):
        self.graph = graph
        self.delta_mu = float(delta_mu)
        self._energy: Dict[int, float] = {}
        self._att: Dict[int, float] = {}
        self._det: Dict[int, float] = {}

    # --- energía de un sitio -------------------------------------------
    def energy(self, code: int) -> float:
        e = self._energy.get(code)
        if e is None:
            e = self.graph.code_energy(code)
            self._energy[code] = e
        return e

    def delta_g(self, code: int) -> float:
        """ΔG/kT de unir una unidad a un sitio con ese código."""
        return -(self.delta_mu + self.energy(code) - self.graph.kink_energy)

    # --- tasas con caché ------------------------------------------------
    def attach(self, code: int) -> float:
        r = self._att.get(code)
        if r is None:
            r = self._attach(code)
            self._att[code] = r
        return r

    def detach(self, code: int) -> float:
        r = self._det.get(code)
        if r is None:
            r = self._detach(code)
            self._det[code] = r
        return r

    def set_delta_mu(self, delta_mu: float) -> None:
        """Cambia Δμ y vacía la caché de tasas (las energías no cambian)."""
        self.delta_mu = float(delta_mu)
        self._att.clear()
        self._det.clear()

    def _attach(self, code: int) -> float:  # pragma: no cover - abstracto
        raise NotImplementedError

    def _detach(self, code: int) -> float:  # pragma: no cover - abstracto
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"kind": self.kind, "delta_mu": self.delta_mu}


class SplitRates(RateModel):
    """Familia con balance detallado; ver docstring del módulo."""

    kind = "split"

    def __init__(self, graph: CrystalGraph, delta_mu: float, k0: float = 1.0,
                 a_mu: float = 1.0, b_E: float = 0.0):
        super().__init__(graph, delta_mu)
        if not (0.0 <= a_mu <= 1.0 and 0.0 <= b_E <= 1.0):
            raise ValueError("a_mu y b_E deben estar en [0, 1]")
        self.k0 = float(k0)
        self.a_mu = float(a_mu)
        self.b_E = float(b_E)

    @classmethod
    def gilmer(cls, graph: CrystalGraph, delta_mu: float, k0: float = 1.0) -> "SplitRates":
        """Unión igual en todos los sitios (Gilmer-Bennema)."""
        return cls(graph, delta_mu, k0, a_mu=1.0, b_E=0.0)

    @classmethod
    def symmetric(cls, graph: CrystalGraph, delta_mu: float, k0: float = 1.0) -> "SplitRates":
        """Reparto simétrico (tipo MONTY / CrystalGrower)."""
        return cls(graph, delta_mu, k0, a_mu=0.5, b_E=0.5)

    def _attach(self, code: int) -> float:
        x = self.energy(code) - self.graph.kink_energy
        return self.k0 * _safe_exp(self.a_mu * self.delta_mu + self.b_E * x)

    def _detach(self, code: int) -> float:
        x = self.energy(code) - self.graph.kink_energy
        return self.k0 * _safe_exp(-(1.0 - self.a_mu) * self.delta_mu - (1.0 - self.b_E) * x)

    def to_dict(self) -> dict:
        return {**super().to_dict(), "k0": self.k0, "a_mu": self.a_mu, "b_E": self.b_E}


class AdaptiveRates(RateModel):
    """Tasas del motor (2+1)D estático (Nagpal et al. 2024) llevadas a 3D.

    k+ = K0 exp(S + Σ_t n_t δ_t / S),   k− = K0 exp(φ0 − E),   S = Δμ.

    Para una cara (001) de un cristal cúbico con δ_z = 0 y φ0 = φ_2D + φ_z, las tasas
    coinciden con `r_a`/`r_d` de `KMC_BKL_Static` en los sitios encima de cada columna,
    SIEMPRE QUE la convención de conteo de enlaces coincida. El motor 2D cuenta un
    vecino lateral solo si su altura es EXACTAMENTE el nivel del sitio
    (`lateral_neighbors_xy_at_level`, punto abierto del README §8), mientras que aquí
    se cuentan los enlaces geométricos (vecino ocupado en ese nivel). Coinciden si,
    para la unión, ningún vecino es más alto que h + 1 y, para el despegue, ningún
    vecino es más alto que h (test_rates.py lo comprueba). No cumple balance
    detallado: úsese solo para comparar.
    """

    kind = "adaptive"

    def __init__(self, graph: CrystalGraph, delta_mu: float, K0: float = 1.0,
                 phi0: Optional[float] = None, deltas: Optional[Sequence[float]] = None):
        super().__init__(graph, delta_mu)
        self.K0 = float(K0)
        self.phi0 = float(graph.kink_energy if phi0 is None else phi0)
        self.deltas = [0.0] * graph.n_types if deltas is None else [float(d) for d in deltas]
        if len(self.deltas) != graph.n_types:
            raise ValueError("Hace falta un δ por tipo de enlace")

    def _attach(self, code: int) -> float:
        S = self.delta_mu
        S_eff = S if abs(S) > 1e-12 else (1e-12 if S >= 0 else -1e-12)
        bonus = sum(n * d for n, d in zip(self.graph.decode(code), self.deltas))
        return self.K0 * _safe_exp(S + bonus / S_eff)

    def _detach(self, code: int) -> float:
        return self.K0 * _safe_exp(self.phi0 - self.energy(code))

    def to_dict(self) -> dict:
        return {**super().to_dict(), "K0": self.K0, "phi0": self.phi0, "deltas": self.deltas}


class SurfaceDiffusion:
    """Difusión superficial: una unidad salta a un sitio vacío cercano (Fase 4).

    Tasa de un salto de la unidad en i al sitio vacío f = i ± δ (δ un vector de salto):

        k_salto(i → f) = k_d · w_δ · exp(−(E_i − E_kink)),

    con E_i la energía de enlace de la unidad en i (sus vecinos ocupados, ver
    `CrystalGraph.code_energy`) y w_δ un peso por vector de salto (1 por defecto: la
    misma barrera en todas las direcciones). Es la forma de Arrhenius de "romper los
    enlaces del origen", del tipo de Gilmer y Bennema (1972); la forma exacta de sus
    ecuaciones no se ha verificado aquí.

    Balance detallado: el salto inverso f → i tiene tasa k_d · w_δ · exp(−(E_f − E_kink)),
    con E_f la energía en f SIN contar a i (que queda vacío), así que

        k(i → f) / k(f → i) = exp(E_f − E_i) = exp(−ΔE),

    el mismo equilibrio que unión y despegue. No depende de Δμ: no hay intercambio con
    la solución y N se conserva.

    Vectores de salto (`hop_vectors`):
    - "bonds" (por defecto): los vectores de enlace del grafo. Saltar solo entre vecinos
      enlazados es lo más barato, pero en un grafo pequeño (Kossel, 6 vecinos) una unidad
      no puede rodear una arista ni subir o bajar un escalón desde encima de una terraza:
      el transporte entre caras queda bloqueado.
    - "cube18": los 18 vecinos con |δ_i| <= 1 y como mucho dos componentes no nulas
      (caras y aristas del cubo de índices). Permite rodear aristas y cruzar escalones.
      Un salto a una diagonal que no es enlace puede caer en un sitio ocupado (el código
      del origen no lo sabe): ese paso es nulo (rechazo).
    - "cube26": los 26 vecinos con |δ_i| <= 1 (añade las 8 diagonales de vértice, que
      casi nunca son destinos válidos: más rechazo y poca física nueva).
    - una lista de vectores (uno por par ±δ).

    Interpretación de k_d: con tasas de Gilmer, k− = k0 · exp(−(E − E_kink)), así que
    k_d / k0 es el número de saltos POR DIRECCIÓN por cada despegue, igual para todos los
    sitios. Con otras tasas el cociente depende de E (ver el notebook 02).

    La tasa depende solo del código del ORIGEN, así que el motor agrupa los saltos por
    código de origen. El peso de una unidad es

        k_d · exp(−(E_i − E_kink)) · [ Σ_{δ enlace t} (2 − n_t) w_δ + Σ_{δ no enlace} 2 w_δ ],

    donde 2 − n_t es el número de vecinos vacíos de tipo t (sale del propio código) y los
    δ que no son enlaces se cuentan siempre. El destino se elige al ejecutar el salto, con
    probabilidad ∝ w_δ. Una tasa que dependiera también del destino (reparto simétrico)
    haría las clases (origen × destino) y multiplicaría el coste por evento: no está
    implementada.
    """

    kind = "arrhenius"

    def __init__(self, graph: CrystalGraph, k_d: float,
                 hop_vectors: Union[str, Sequence[Sequence[int]]] = "bonds",
                 weights: Optional[Sequence[float]] = None):
        """
        Parámetros:
            graph: grafo cristalino (el mismo que el de las tasas de unión/despegue)
            k_d: tasa de salto por dirección de una unidad con E = E_kink (en 1/k0)
            hop_vectors: "bonds", "cube18", "cube26" o lista de vectores δ (uno por par ±δ)
            weights: peso w_δ de cada vector de salto (en el orden de `self.vectors`);
                por defecto 1
        """
        if not k_d >= 0.0:
            raise ValueError("k_d debe ser >= 0")
        self.graph = graph
        self.k_d = float(k_d)
        bond_vecs = [tuple(int(x) for x in v) for v in graph.vectors]
        if isinstance(hop_vectors, str):
            if hop_vectors == "bonds":
                vecs = list(bond_vecs)
            elif hop_vectors in ("cube18", "cube26"):
                # pares ±δ con componentes en {−1, 0, 1}: los 13 del cubo (cube26) o los 9
                # sin las diagonales de vértice (cube18). Primero los enlaces del grafo que
                # estén en el conjunto, luego el resto con signo canónico.
                max_nz = 3 if hop_vectors == "cube26" else 2
                cubo = [v for v in product((-1, 0, 1), repeat=3)
                        if v != (0, 0, 0) and next(x for x in v if x) > 0
                        and sum(x != 0 for x in v) <= max_nz]
                en_cubo = {c for v in cubo for c in (v, tuple(-x for x in v))}
                vecs = [v for v in bond_vecs if v in en_cubo]
                for v in cubo:
                    if v not in vecs and tuple(-x for x in v) not in vecs:
                        vecs.append(v)
            else:
                raise ValueError(f"hop_vectors desconocido: {hop_vectors!r}")
            self.hop_vectors_spec = hop_vectors
        else:
            vecs = [tuple(int(x) for x in v) for v in hop_vectors]
            self.hop_vectors_spec = [list(v) for v in vecs]
        seen = set()
        for v in vecs:
            if v == (0, 0, 0) or v in seen or tuple(-x for x in v) in seen:
                raise ValueError(f"Vector de salto nulo o repetido: {v}")
            seen.add(v)
        if max(abs(x) for v in vecs for x in v) > graph.reach:
            # la caja deja `margin` = 2·reach + 2 sitios libres: un salto más largo podría
            # salirse de ella
            raise ValueError("Los vectores de salto no pueden ser más largos que los de enlace")
        self.vectors = vecs
        self.weights = [1.0] * len(vecs) if weights is None else [float(w) for w in weights]
        if len(self.weights) != len(vecs):
            raise ValueError("Hace falta un peso por vector de salto")
        if any(not w >= 0.0 for w in self.weights):
            raise ValueError("Los pesos deben ser >= 0")
        # Tipo de enlace de cada vector de salto (−1 si no es un enlace del grafo)
        self.bond_type = []
        for v in vecs:
            neg = tuple(-x for x in v)
            t = next((i for i, b in enumerate(bond_vecs) if b == v or b == neg), -1)
            self.bond_type.append(t)
        # Peso de los vectores que no son enlaces (siempre cuentan) y de cada tipo t
        self._w_extra = 2.0 * sum(w for w, t in zip(self.weights, self.bond_type) if t < 0)
        self._w_type = [0.0] * graph.n_types
        for w, t in zip(self.weights, self.bond_type):
            if t >= 0:
                self._w_type[t] += w
        self._hop: Dict[int, float] = {}
        self._cls: Dict[int, float] = {}

    def hop_rate(self, code: int) -> float:
        """Tasa por dirección con w_δ = 1: k_d · exp(−(E − E_kink))."""
        r = self._hop.get(code)
        if r is None:
            x = self.graph.code_energy(code) - self.graph.kink_energy
            r = self.k_d * _safe_exp(-x)
            self._hop[code] = r
        return r

    def class_rate(self, code: int) -> float:
        """Tasa total de salto de una unidad con ese código (ver docstring de la clase)."""
        r = self._cls.get(code)
        if r is None:
            libres = self._w_extra + sum((2 - n) * w for n, w in
                                         zip(self.graph.decode(code), self._w_type))
            r = self.hop_rate(code) * libres
            self._cls[code] = r
        return r

    def to_dict(self) -> dict:
        return {"kind": self.kind, "k_d": self.k_d, "hop_vectors": self.hop_vectors_spec,
                "weights": self.weights}


def diffusion_from_dict(graph: CrystalGraph, d: Optional[dict]) -> Optional[SurfaceDiffusion]:
    """Reconstruye la difusión desde su `to_dict()` (o un bloque de config). None = sin difusión."""
    if not d:
        return None
    kind = d.get("kind", "arrhenius")
    if kind != "arrhenius":
        raise ValueError(f"Modelo de difusión desconocido: {kind!r}")
    return SurfaceDiffusion(graph, d["k_d"], d.get("hop_vectors", "bonds"), d.get("weights"))


def rates_from_dict(graph: CrystalGraph, d: dict) -> RateModel:
    """Reconstruye un modelo de tasas desde su `to_dict()` (metadata/config JSON)."""
    kind = d.get("kind", "split")
    if kind == "split":
        return SplitRates(graph, d["delta_mu"], d.get("k0", 1.0), d.get("a_mu", 1.0), d.get("b_E", 0.0))
    if kind == "gilmer":
        return SplitRates.gilmer(graph, d["delta_mu"], d.get("k0", 1.0))
    if kind == "symmetric":
        return SplitRates.symmetric(graph, d["delta_mu"], d.get("k0", 1.0))
    if kind == "adaptive":
        return AdaptiveRates(graph, d["delta_mu"], d.get("K0", 1.0), d.get("phi0"), d.get("deltas"))
    raise ValueError(f"Modelo de tasas desconocido: {kind!r}")
