"""Motor kMC de una superficie (hkl) con contorno periódico, como MONTY (modo *slab*).

MONTY (Boerrigter et al., J. Phys. Chem. A 108, 5894, 2004) no simula un cristal
finito: simula UNA cara (hkl). Transforma la celda en una "celda de capa" con dos
vectores u, v en el plano de la cara y un tercero w fuera de él ((hkl)·w = 1), pone
contorno periódico según u y v y deja crecer (o disolverse) la superficie según w.
`SlabKMC` hace lo mismo sobre un `BasisGraph` (cualquier número de GU por celda):

- sitios (a, b, c, s): celda a·u + b·v + c·w de la caja L1 × L2 (periódica en a y b)
  y GU s de la base. La altura de la GU sobre los planos (hkl) es z = c + z_s, con
  z_s ∈ [0, 1) (unidades de d_hkl de los índices reducidos);
- abajo hay un sustrato: las `reach` capas inferiores están congeladas (siempre
  ocupadas, nunca cambian) y por encima hay `depth` capas llenas que sí pueden
  cambiar. La superficie inicial es plana y se corta en el plano de menor energía de
  enlaces rotos (`BasisGraph.best_termination`);
- la caja crece hacia arriba sola (c es el índice más lento, así que basta con añadir
  capas vacías al final). Si la superficie se acerca al sustrato congelado, se añaden
  capas llenas por debajo (reconstrucción completa, poco frecuente).

Eventos y tasas (condición STS de MONTY, §2.3):
- unión en un sitio vacío con al menos un vecino ocupado, también en un hueco
  cerrado (MONTY admite inclusiones). Con `allow_enclosed=False`, un hueco con todos
  los vecinos ocupados no se llena, como en `KMC3D` (decisión D4);
- despegue de una GU con al menos un vecino vacío (una GU totalmente enlazada no se
  despega, para mantener el equilibrio a Δμ = 0);
- con λ1 = λ2 = ½, las reglas de MONTY hacen la dinámica simétrica partícula-hueco:
  crecer (hkl) a +Δμ es lo mismo que disolver (−h−k−l) a −Δμ (notebook 03, §2.3).
  `allow_enclosed=False` rompe esa simetría;
- tasas `MontyRates` (ecs. 11–12 de MONTY), con x = E − E_cr la energía de enlace
  del sitio respecto de la del kink.

Clases (n-fold way): como en `KMC3D`, pero el código de un sitio es la MÁSCARA DE BITS
de sus enlaces (bit k = el vecino k de su subred está ocupado), que es el índice de la
tabla de consulta de MONTY (§2.4). La clave de clase es (código, subred, tipo). El peso
de cada clase está en un árbol de Fenwick (`kmc3d.fenwick.Fenwick`). Con energías de
decenas de kT (las grasas) las tasas de clase abarcan unos 50 órdenes de magnitud: el
árbol se reconstruye antes de elegir si desde la última reconstrucción se restó una
tasa mucho mayor que el total, para que el redondeo no sesgue la elección.

Medidas de MONTY (§2.6): uniones N_growth y despegues N_etch desde el último
`reset_counters()` (fin del equilibrado), fracción de adhesión
S = (N_growth − N_etch)/N_growth (ec. 22) y velocidad de la cara
R = d_hkl (N_growth − N_etch)/(N_slice t) (ec. 20).
"""

from array import array
from math import log
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401  (pone el repositorio en sys.path)
from src.common.numerics import _safe_exp

from .basis_graph import BasisGraph, slice_cell
from .fenwick import Fenwick
from .geometry import hkl_label, reduce_hkl

ATTACH, DETACH = 0, 1
EVENT_NAMES = {ATTACH: "attach", DETACH: "detach"}

# Si desde la última reconstrucción del Fenwick se restó la tasa de una clase mayor que
# _CANCEL veces el peso total, el árbol se reconstruye antes de elegir el evento. Con
# energías de decenas de kT (las grasas de MONTY) las tasas de clase van de 1e-35 a 1e13:
# restar una tasa enorme deja en los nodos del árbol un residuo de redondeo (~1e-16 de
# esa tasa) comparable al total, y la elección se sesga o cae fuera de las clases.
_CANCEL = 1e6


class MontyRates:
    """Probabilidades de unión y despegue de MONTY (ecs. 11 y 12), en unidades de kT.

    Con x = E − E_cr (energía de enlace del sitio menos la del kink) y Δμ/kT:

        k+ = k0 · exp( (1 − λ2)·Δμ + λ1·x )
        k− = k0 · exp( −λ2·Δμ − (1 − λ1)·x )

    que es `rates.SplitRates` con a_mu = 1 − λ2 y b_E = λ1:
    - λ1 = 0: la unión no depende del sitio; λ1 = 1: el despegue no depende del sitio;
    - λ1 = λ2 = 0: modelo de "lluvia aleatoria" de Gilmer y Bennema;
    - λ1 = λ2 = ½: el muestreo eficiente que usa MONTY en las grasas (= `symmetric`).
    El cociente k+/k− = exp(Δμ + x) no depende de λ (balance detallado).

    En MONTY, ΔU − ΔU_i = (1 − U_moth/U_cryst)(E_cr − E_i): una interacción con el
    solvente U_moth ≠ 0 equivale a multiplicar todas las energías del grafo por
    1 − U_moth/U_cryst (`BasisGraph.scaled`). En las grasas se usó U_moth = 0.
    """

    def __init__(self, delta_mu: float, lambda1: float = 0.5, lambda2: float = 0.5,
                 k0: float = 1.0):
        """
        Parámetros:
            delta_mu: fuerza impulsora Δμ/kT (> 0 crece)
            lambda1, lambda2: parámetros λ1, λ2 de MONTY, en [0, 1]
            k0: escala de las tasas (1/τ)
        """
        if not (0.0 <= lambda1 <= 1.0 and 0.0 <= lambda2 <= 1.0):
            raise ValueError("λ1 y λ2 deben estar en [0, 1]")
        self.delta_mu = float(delta_mu)
        self.k0 = float(k0)
        # se guardan a_mu y b_E (los de SplitRates) para dar exactamente las mismas tasas
        self.a_mu = 1.0 - float(lambda2)
        self.b_E = float(lambda1)

    @classmethod
    def from_split(cls, delta_mu: float, a_mu: float, b_E: float, k0: float = 1.0) -> "MontyRates":
        """Las mismas tasas que `SplitRates(graph, delta_mu, k0, a_mu, b_E)`."""
        r = cls(delta_mu, 0.0, 0.0, k0)
        if not (0.0 <= a_mu <= 1.0 and 0.0 <= b_E <= 1.0):
            raise ValueError("a_mu y b_E deben estar en [0, 1]")
        r.a_mu, r.b_E = float(a_mu), float(b_E)
        return r

    @property
    def lambda1(self) -> float:
        return self.b_E

    @property
    def lambda2(self) -> float:
        return 1.0 - self.a_mu

    def attach(self, x: float) -> float:
        """k+ de un sitio con x = E − E_cr."""
        return self.k0 * _safe_exp(self.a_mu * self.delta_mu + self.b_E * x)

    def detach(self, x: float) -> float:
        """k− de una GU con x = E − E_cr."""
        return self.k0 * _safe_exp(-(1.0 - self.a_mu) * self.delta_mu - (1.0 - self.b_E) * x)

    def to_dict(self) -> dict:
        return {"kind": "monty", "delta_mu": self.delta_mu, "lambda1": self.lambda1,
                "lambda2": self.lambda2, "a_mu": self.a_mu, "b_E": self.b_E, "k0": self.k0}


def _int_array(code: str, values) -> array:
    """numpy -> array compacto (acceso escalar rápido desde Python)."""
    out = array(code)
    dt = {"i": np.int32, "q": np.int64}[code]
    out.frombytes(np.ascontiguousarray(values, dtype=dt).tobytes())
    return out


class SlabKMC:
    """kMC BKL de una superficie (hkl) periódica sobre un grafo con base (MONTY)."""

    def __init__(self, graph: BasisGraph, hkl: Sequence[int], rates: MontyRates,
                 size: Tuple[int, int] = (40, 40), rng_seed: Optional[int] = None,
                 depth: Optional[int] = None, allow_enclosed: bool = True,
                 time_mode: str = "bkl", time_scale: float = 1.0,
                 termination: Optional[float] = None, room: int = 8,
                 rebuild_every: int = 20_000, buffer_size: int = 4096):
        """
        Parámetros:
            graph: grafo cristalino con energías en kT (`BasisGraph.to_kT`)
            hkl: dirección de crecimiento (índices de la celda de `graph`)
            rates: tasas `MontyRates`
            size: (L1, L2) celdas de capa en el plano (contorno periódico)
            rng_seed: semilla de np.random.default_rng
            depth: capas llenas (móviles) bajo la superficie inicial; por defecto
                2·reach + 3
            allow_enclosed: True (por defecto, reglas de MONTY): un hueco con todos los
                vecinos ocupados puede llenarse. False: no se llena, como en `KMC3D`
                (decisión D4)
            time_mode: "bkl" (Δt = −ln u / W) o "media" (Δt = 1/W, ec. 16 de MONTY)
            time_scale: multiplica cada Δt
            termination: altura del corte inicial en [0, 1) (por defecto
                `graph.best_termination`)
            room: capas que se añaden cada vez que la caja crece
            rebuild_every: cada cuántos eventos se recalcula el Fenwick desde cero
            buffer_size: números aleatorios que se generan de golpe
        """
        if graph.units != "kT":
            raise ValueError(f"El grafo debe estar en kT (está en {graph.units!r}): usa graph.to_kT(T)")
        if time_mode not in ("bkl", "media"):
            raise ValueError("time_mode debe ser 'bkl' o 'media'")
        self.graph = graph
        self.rates = rates
        self.hkl = tuple(int(x) for x in hkl)
        self.h0 = reduce_hkl(self.hkl)
        self.L1, self.L2 = int(size[0]), int(size[1])
        self.nb = graph.n_basis
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.allow_enclosed = bool(allow_enclosed)
        self.time_mode = time_mode
        self.time_scale = float(time_scale)
        self.room = max(1, int(room))
        self.rebuild_every = int(rebuild_every)
        self._buffer_size = int(buffer_size)
        self.e_cr = graph.lattice_energy

        # --- Celda de capa y base en sus coordenadas -----------------------------
        B = slice_cell(self.h0, graph.cell)                 # filas u, v, w (enteras)
        Binv = np.rint(np.linalg.inv(B)).astype(np.int64)
        assert np.array_equal(B @ Binv, np.eye(3, dtype=np.int64))
        self.frame = B
        xs = graph.basis @ np.linalg.inv(B)                  # (α, β, z) de cada GU
        fl = np.floor(xs + 1e-9).astype(np.int64)
        self.xs = xs - fl                                    # representante en [0, 1)^3
        self.zs = self.xs[:, 2].copy()                       # altura z_s ∈ [0, 1)
        # Vecinos de cada subred en la celda de capa: (j, da, db, dc, energía)
        self._slots: List[List[Tuple[int, int, int, int, float]]] = []
        for s in range(self.nb):
            lst = []
            for sl in graph.slots(s):
                d = np.asarray(sl.shift, dtype=np.int64) @ Binv + fl[sl.j] - fl[s]
                lst.append((sl.j, int(d[0]), int(d[1]), int(d[2]), sl.energy))
            if len(lst) > 62:
                raise ValueError(f"La subred {s} tiene {len(lst)} enlaces: el código de bits "
                                 "admite como mucho 62")
            self._slots.append(lst)
        self.reach = max(1, max(abs(t[3]) for lst in self._slots for t in lst))
        self._check_periodic_images()
        # Bit del vecino inverso: el vecino k de s (j, Δ) ve a s en su vecino (s, −Δ)
        self._rev: List[List[int]] = []
        for s, lst in enumerate(self._slots):
            rev = []
            for (j, da, db, dc, _) in lst:
                back = [k for k, t in enumerate(self._slots[j]) if t[:4] == (s, -da, -db, -dc)]
                assert len(back) == 1, "enlace inverso no encontrado"
                rev.append(back[0])
            self._rev.append(rev)
        self._full = [(1 << len(lst)) - 1 for lst in self._slots]

        # --- Estado inicial: superficie plana -----------------------------------
        self.frozen = self.reach                            # capas congeladas abajo
        self.depth = int(depth) if depth is not None else 2 * self.reach + 3
        z_term = graph.best_termination(self.h0) if termination is None else float(termination)
        self.termination = float(z_term) % 1.0
        c_top = self.frozen + self.depth                    # capa de la superficie inicial
        H = c_top + 2 * self.reach + 2 + self.room
        cidx = np.arange(H)[:, None]
        filled = (cidx + self.zs[None, :]) < (c_top + self.termination)      # (H, nb)
        A = np.broadcast_to(filled[:, None, None, :], (H, self.L1, self.L2, self.nb)).astype(np.uint8)
        self._c_shift = 0                                   # capas añadidas por debajo
        self.n_occ = int(A.sum())
        self._n_ref = self.n_occ                            # N de la superficie inicial
        # Altura de la GU más alta de la superficie inicial (referencia de los mapas)
        zf = np.where(filled, cidx + self.zs[None, :], -np.inf)
        self.z_top0 = float(zf.max())
        self.c_top0 = int(np.flatnonzero(filled.any(axis=1)).max())

        # Estado dinámico
        self.t = 0.0
        self.n_events = 0
        self.n_attach = 0
        self.n_detach = 0
        self.n_bottom_expansions = 0
        self._stage = (0, 0, 0.0, 0)                        # (uniones, despegues, t, eventos)
        self._ubuf: List[float] = []
        self._upos = 0
        self.history: List[tuple] = []
        self._install(A)

    # ------------------------------------------------------------------
    # Geometría de la caja
    # ------------------------------------------------------------------
    def _check_periodic_images(self) -> None:
        """Con la caja L1 × L2, los vecinos de una GU deben ser sitios distintos de ella."""
        for s, lst in enumerate(self._slots):
            seen = set()
            for (j, da, db, dc, _) in lst:
                key = (j, da % self.L1, db % self.L2, dc)
                if key in seen or key == (s, 0, 0, 0):
                    raise ValueError(f"Caja {self.L1}x{self.L2} demasiado pequeña para "
                                     f"{hkl_label(self.hkl)}: dos enlaces caen en el mismo sitio")
                seen.add(key)

    def _neighbor_tables(self) -> None:
        """Desplazamientos planos de los vecinos, por subred y por columna.

        Índice plano: ((c·L1 + a)·L2 + b)·nb + s. Lejos del borde de la caja (en a y b)
        el desplazamiento de un vecino no depende de la columna: todas esas columnas
        comparten la misma lista. Las columnas cercanas al borde tienen la suya (el
        vecino se envuelve por el contorno periódico).
        """
        L1, L2, nb = self.L1, self.L2, self.nb
        A = L1 * L2 * nb
        tabs = []
        for s, lst in enumerate(self._slots):
            rev = self._rev[s]
            ra = max(abs(t[1]) for t in lst)
            rb = max(abs(t[2]) for t in lst)
            interior = [(dc * A + (da * L2 + db) * nb + (j - s), 1 << rev[k])
                        for k, (j, da, db, dc, _) in enumerate(lst)]
            per_col = []
            for a in range(L1):
                for b in range(L2):
                    if ra <= a < L1 - ra and rb <= b < L2 - rb:
                        per_col.append(interior)
                        continue
                    per_col.append([(dc * A + (((a + da) % L1 - a) * L2 + ((b + db) % L2 - b)) * nb
                                     + (j - s), 1 << rev[k])
                                    for k, (j, da, db, dc, _) in enumerate(lst)])
            tabs.append(per_col)
        self._nbr = tabs

    # ------------------------------------------------------------------
    # Construcción (vectorizada) de códigos, clases y Fenwick
    # ------------------------------------------------------------------
    def _codes_from_occupancy(self, A: np.ndarray) -> np.ndarray:
        """Código (máscara de bits) de cada sitio para la ocupación A (H, L1, L2, nb).

        Los vecinos por debajo de la caja cuentan como ocupados (cristal) y los de
        encima como vacíos.
        """
        H = A.shape[0]
        code = np.zeros(A.shape, dtype=np.int64)
        for s, lst in enumerate(self._slots):
            for k, (j, da, db, dc, _) in enumerate(lst):
                X = np.roll(A[:, :, :, j], (-da, -db), axis=(1, 2)).astype(np.int64)   # X[c, a+da, b+db]
                Y = np.zeros_like(X)
                if dc >= 0:
                    Y[:H - dc] = X[dc:]
                else:
                    Y[-dc:] = X[:H + dc]
                    Y[:-dc] = 1                     # por debajo de la caja: cristal
                code[:, :, :, s] += np.int64(1 << k) * Y
        return code

    def _install(self, A: np.ndarray) -> None:
        """(Re)construye todas las estructuras a partir de la ocupación A (H, L1, L2, nb)."""
        H = A.shape[0]
        self.H = H
        self._LL = self.L1 * self.L2
        self._layer = self._LL * self.nb                    # sitios por capa
        self._flim = self.frozen * self._layer              # sitios < _flim: congelados
        self._neighbor_tables()
        code = self._codes_from_occupancy(A)
        full = np.array(self._full, dtype=np.int64)[None, None, None, :]
        occ = A.astype(bool)
        att = (~occ) & (code != 0)
        if not self.allow_enclosed:
            att &= code != full
        det = occ & (code != full)
        elig = att | det
        elig[:self.frozen] = False
        flat = np.flatnonzero(elig.ravel())
        codes = code.ravel()[flat]
        subl = flat % self.nb
        kinds = occ.ravel()[flat].astype(np.int64)          # 1 = ocupado = despegue
        # Clases ordenadas por clave = (código·nb + s)·2 + tipo; miembros en orden de sitio
        order = np.lexsort((kinds, subl, codes))
        sc, ss, sk, sf = codes[order], subl[order], kinds[order], flat[order]
        self._key_of: List[int] = []
        self._kind_of: List[int] = []
        self._rate_of: List[float] = []
        self._members: List[List[int]] = []
        self._cid_of_key: Dict[int, int] = {}
        self._energy_cache: Dict[Tuple[int, int], float] = {}
        cls_of = np.full(A.size, -1, dtype=np.int32)
        pos_of = np.zeros(A.size, dtype=np.int32)
        if len(sf):
            new = np.r_[True, (sc[1:] != sc[:-1]) | (ss[1:] != ss[:-1]) | (sk[1:] != sk[:-1])]
            starts = np.flatnonzero(new)
            ends = np.r_[starts[1:], len(sf)]
            for a, b in zip(starts, ends):
                key = ((int(sc[a]) * self.nb + int(ss[a])) << 1) | int(sk[a])
                cid = self._new_class(key)
                sites = sf[a:b]
                self._members[cid] = sites.tolist()
                cls_of[sites] = cid
                pos_of[sites] = np.arange(b - a, dtype=np.int32)
        self._occ = bytearray(A.astype(np.uint8).ravel().tobytes())
        self._code = _int_array("q", code.ravel())
        self._cls_of = _int_array("i", cls_of)
        self._pos_of = _int_array("i", pos_of)
        self._fw = Fenwick(max(64, 2 * len(self._key_of)))
        self._fw.rebuild(self._exact_weights())
        self._peak = [0.0]          # mayor tasa restada del Fenwick desde la última reconstrucción
        self._make_hot_path()

    def site_energy(self, s: int, code: int) -> float:
        """Energía de enlace (kT) de una GU de la subred s con ese código de bits."""
        key = (s, code)
        e = self._energy_cache.get(key)
        if e is None:
            e = 0.0
            for k, t in enumerate(self._slots[s]):
                if (code >> k) & 1:
                    e += t[4]
            self._energy_cache[key] = e
        return e

    def _new_class(self, key: int) -> int:
        """Registra una clase nueva (sin tocar el Fenwick) y devuelve su id."""
        kind = key & 1
        rest = key >> 1
        s, code = rest % self.nb, rest // self.nb
        x = self.site_energy(s, code) - self.e_cr
        rate = self.rates.detach(x) if kind == DETACH else self.rates.attach(x)
        cid = len(self._key_of)
        self._key_of.append(key)
        self._kind_of.append(kind)
        self._rate_of.append(float(rate))
        self._members.append([])
        self._cid_of_key[key] = cid
        return cid

    def _exact_weights(self) -> List[float]:
        return [len(m) * r for m, r in zip(self._members, self._rate_of)]

    def _rebuild_weights(self, capacity: int = 0) -> None:
        """Fenwick desde los pesos exactos: borra el error de redondeo acumulado."""
        self._fw.rebuild(self._exact_weights(), capacity=capacity)
        self._peak[0] = 0.0

    def _make_hot_path(self) -> None:
        """Funciones internas de un evento con variables locales (más rápido)."""
        occ, code = self._occ, self._code
        cls_of, pos_of = self._cls_of, self._pos_of
        members_all, key_of, rate_of = self._members, self._key_of, self._rate_of
        cid_of_key = self._cid_of_key
        full = self._full
        nb, LL, flim = self.nb, self._LL, self._flim
        allow_enclosed = self.allow_enclosed
        nbr = self._nbr
        fw = self._fw
        fw_add = fw.add
        peak = self._peak

        def create_class(key: int) -> int:
            cid = self._new_class(key)
            if cid >= fw.n:                       # ampliar capacidad del Fenwick
                self._rebuild_weights(capacity=2 * fw.n)
            return cid

        def reclass(i: int) -> None:
            """Mueve el sitio i a la clase que le toca según su ocupación y código."""
            if i < flim:                          # sustrato congelado: nunca elegible
                return
            c = code[i]
            s = i % nb
            if occ[i]:
                key = -1 if c == full[s] else (((c * nb + s) << 1) | 1)
            elif c == 0 or (c == full[s] and not allow_enclosed):
                key = -1
            else:
                key = (c * nb + s) << 1
            old = cls_of[i]
            if old >= 0:
                if key_of[old] == key:
                    return
                m = members_all[old]              # quitar con "swap-and-pop"
                last = m.pop()
                if last != i:
                    p = pos_of[i]
                    m[p] = last
                    pos_of[last] = p
                r = rate_of[old]
                fw_add(old, -r)
                if r > peak[0]:                   # posible cancelación (ver _CANCEL)
                    peak[0] = r
                cls_of[i] = -1
            if key >= 0:
                cid = cid_of_key.get(key)
                if cid is None:
                    cid = create_class(key)
                m = members_all[cid]
                pos_of[i] = len(m)
                m.append(i)
                cls_of[i] = cid
                fw_add(cid, rate_of[cid])

        def flip(i: int, kind: int) -> None:
            """Unión (kind=0) o despegue (kind=1) en i; actualiza los códigos vecinos."""
            q = i // nb
            tab = nbr[i - q * nb][q % LL]
            if kind == ATTACH:
                occ[i] = 1
                for off, w in tab:
                    code[i + off] += w
            else:
                occ[i] = 0
                for off, w in tab:
                    code[i + off] -= w
            reclass(i)
            for off, _ in tab:
                reclass(i + off)

        self._reclass = reclass
        self._flip = flip

    # ------------------------------------------------------------------
    # Un evento
    # ------------------------------------------------------------------
    def _uniforms(self, n: int) -> List[float]:
        p = self._upos
        if p + n > len(self._ubuf):
            self._ubuf = self.rng.random(self._buffer_size).tolist()
            p = 0
        self._upos = p + n
        return self._ubuf[p:p + n]

    def _exact_class(self, u: float) -> int:
        """Selección exacta (lineal) de clase; se usa si el Fenwick acumuló error."""
        w = self._exact_weights()
        total = sum(w)
        self._fw.rebuild(w)
        self._peak[0] = 0.0
        if total <= 0.0:
            return -1
        target, acc = u * total, 0.0
        for cid, wi in enumerate(w):
            acc += wi
            if acc > target and wi > 0.0:
                return cid
        return max(range(len(w)), key=w.__getitem__)

    def step(self) -> bool:
        """Ejecuta un evento BKL. Devuelve False si no hay eventos posibles."""
        fw = self._fw
        if self._peak[0] > _CANCEL * fw.total:  # residuo de redondeo comparable al total
            self._rebuild_weights()
        total = fw.total
        if not total > 0.0:
            self._rebuild_weights()
            total = fw.total
            if not total > 0.0:
                return False
        if self.time_mode == "bkl":
            u1, u2, u3 = self._uniforms(3)
            self.t += -log(1.0 - u1) / total * self.time_scale
        else:
            u2, u3 = self._uniforms(2)
            self.t += self.time_scale / total
        cid = fw.find(u2 * total)
        # Por redondeo, `find` puede caer en una clase vacía o, si el total supera la suma
        # del árbol, en una posición sin clase (cid >= número de clases): selección exacta
        members = self._members[cid] if cid < len(self._members) else None
        if not members:
            cid = self._exact_class(u2)
            if cid < 0:
                return False
            members = self._members[cid]
        i = members[int(u3 * len(members))]
        kind = self._kind_of[cid]
        self._flip(i, kind)
        self.n_events += 1
        if kind == ATTACH:
            self.n_attach += 1
            self.n_occ += 1
            need = i // self._layer + 2 * self.reach + 2 - self.H
            if need > 0:
                self._grow_top(need)
        else:
            self.n_detach += 1
            self.n_occ -= 1
            if i // self._layer < self.frozen + self.reach + 1:
                self._grow_bottom()
        if self.n_events % self.rebuild_every == 0:
            self._rebuild_weights()
        return True

    def _grow_top(self, at_least: int = 1) -> None:
        """Añade max(`room`, `at_least`) capas vacías arriba.

        No hace falta reconstruir nada: c es el índice más lento, así que los sitios
        nuevos van al final de los arrays, y están vacíos y sin vecinos ocupados.
        """
        layers = max(self.room, int(at_least))
        n = layers * self._layer
        self._occ.extend(bytes(n))
        self._code.extend(array("q", bytes(8 * n)))
        self._cls_of.extend(array("i", [-1]) * n)
        self._pos_of.extend(array("i", bytes(4 * n)))
        self.H += layers

    def _grow_bottom(self) -> None:
        """Añade `room` capas llenas abajo y reconstruye (la superficie bajó mucho)."""
        A = np.concatenate([np.ones((self.room, self.L1, self.L2, self.nb), np.uint8),
                            self.occupancy_array()], axis=0)
        self._c_shift += self.room
        self.n_occ += self.room * self._layer
        self._n_ref += self.room * self._layer
        self.n_bottom_expansions += 1
        self._install(A)

    # ------------------------------------------------------------------
    # Bucle principal y etapas
    # ------------------------------------------------------------------
    def record(self) -> None:
        """Añade una fila al historial (t, eventos, uniones, despegues, altura media, W)."""
        self.history.append((self.t, self.n_events, self.n_attach, self.n_detach,
                             self.mean_height(), self._fw.total))

    def run(self, max_events: Optional[int] = None, t_end: Optional[float] = None,
            record_every: int = 0) -> dict:
        """Ejecuta eventos hasta `max_events` (de esta llamada) o hasta `t_end`.

        Devuelve `stats()`. Con `record_every` > 0 añade una fila al historial cada
        tantos eventos.
        """
        if max_events is None and t_end is None:
            raise ValueError("Hace falta al menos una condición de parada")
        n0 = self.n_events
        reason = "max_events"
        step = self.step
        while True:
            if max_events is not None and self.n_events - n0 >= max_events:
                break
            if t_end is not None and self.t >= t_end:
                reason = "t_end"
                break
            if not step():
                reason = "sin_eventos"
                break
            if record_every and self.n_events % record_every == 0:
                self.record()
        return self.stats(stop_reason=reason)

    def reset_counters(self) -> None:
        """Empieza una etapa nueva de medida (p. ej. el muestreo tras el equilibrado)."""
        self._stage = (self.n_attach, self.n_detach, self.t, self.n_events)

    @property
    def n_growth(self) -> int:
        """Uniones desde el último `reset_counters()` (N_growth de MONTY)."""
        return self.n_attach - self._stage[0]

    @property
    def n_etch(self) -> int:
        """Despegues desde el último `reset_counters()` (N_etch de MONTY)."""
        return self.n_detach - self._stage[1]

    def sticking_fraction(self) -> float:
        """S = (N_growth − N_etch) / N_growth (ec. 22 de MONTY); NaN sin uniones."""
        ng = self.n_growth
        return (ng - self.n_etch) / ng if ng > 0 else float("nan")

    def stage_time(self) -> float:
        """Tiempo transcurrido desde el último `reset_counters()`."""
        return self.t - self._stage[2]

    def growth_rate(self) -> float:
        """R = d_hkl (N_growth − N_etch) / (N_slice t) (ec. 20), en Å (o unidades de la
        celda) por unidad de tiempo 1/k0.

        Se usa la capa de la dirección reducida: espesor d_h0 con L1·L2·Z GU, que da lo
        mismo que d_hkl/N_slice para (hkl) no coprimos.
        """
        dt = self.stage_time()
        if not dt > 0:
            return float("nan")
        d = self.graph.cell.d_spacing(self.h0)
        return d * (self.n_growth - self.n_etch) / (self._LL * self.nb * dt)

    def set_delta_mu(self, delta_mu: float) -> None:
        """Cambia Δμ/kT y actualiza las tasas de todas las clases."""
        self.rates.delta_mu = float(delta_mu)
        for cid, key in enumerate(self._key_of):
            kind = key & 1
            rest = key >> 1
            x = self.site_energy(rest % self.nb, rest // self.nb) - self.e_cr
            self._rate_of[cid] = self.rates.detach(x) if kind == DETACH else self.rates.attach(x)
        self._rebuild_weights()

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def occupancy_array(self) -> np.ndarray:
        """Vista numpy (sin copia) de la ocupación, forma (H, L1, L2, Z)."""
        return np.frombuffer(self._occ, dtype=np.uint8).reshape(self.H, self.L1, self.L2, self.nb)

    def mean_height(self) -> float:
        """Capas crecidas desde el principio: (N − N_inicial)/(L1·L2·Z), en capas d_hkl
        de los índices reducidos (negativo si la superficie se disolvió)."""
        return (self.n_occ - self._n_ref) / self._layer

    def height_map(self, integer: bool = False) -> np.ndarray:
        """Altura de la superficie en cada columna (a, b), respecto de la inicial (0).

        Es la altura z = c + z_s de la GU ocupada más alta de la columna (la superficie
        vista desde la solución), en capas d_hkl de los índices reducidos. Con
        `integer=True` se da el índice de capa c (entero), listo para
        `src.analysis.surface_observables`.
        """
        A = self.occupancy_array().astype(bool)
        c = np.arange(self.H, dtype=float)[:, None, None, None]
        z = c + (0.0 if integer else self.zs[None, None, None, :])
        z = np.where(A, z, -np.inf).max(axis=(0, 3))
        ref = (self.c_top0 if integer else self.z_top0) + self._c_shift
        out = z - ref
        return np.rint(out).astype(int) if integer else out

    def roughness(self) -> float:
        """W: desviación típica del mapa de alturas."""
        return float(np.std(self.height_map()))

    def occupied_cartesian(self, top_layers: Optional[int] = None) -> np.ndarray:
        """Coordenadas cartesianas (N, 3) de los centros de las GU ocupadas.

        Con `top_layers`, solo las de las capas que quedan a menos de esa distancia de
        la columna más alta (para dibujar la superficie sin todo el sustrato).
        """
        A = self.occupancy_array()
        idx = np.argwhere(A)
        if top_layers is not None and len(idx):
            idx = idx[idx[:, 0] >= idx[:, 0].max() - int(top_layers)]
        frac_slab = np.c_[idx[:, 1], idx[:, 2], idx[:, 0] - self._c_shift].astype(float) + self.xs[idx[:, 3]]
        return (frac_slab @ self.frame) @ self.graph.cell.matrix

    def class_table(self) -> pd.DataFrame:
        """Una fila por clase no vacía: tipo, subred, enlaces, energía, tasa, sitios."""
        rows = []
        for cid, (key, kind, rate, m) in enumerate(zip(self._key_of, self._kind_of,
                                                         self._rate_of, self._members)):
            if not m:
                continue
            rest = key >> 1
            s, code = rest % self.nb, rest // self.nb
            rows.append({"clase": cid, "evento": EVENT_NAMES[kind], "subred": s,
                         "n_enlaces": bin(code).count("1"), "energia": self.site_energy(s, code),
                         "tasa": rate, "n_sitios": len(m), "peso": rate * len(m)})
        return pd.DataFrame(rows)

    def stats(self, stop_reason: str = "") -> dict:
        """Resumen de la corrida y de la etapa actual."""
        return {"hkl": self.hkl, "size": (self.L1, self.L2), "t": self.t,
                "n_events": self.n_events, "n_attach": self.n_attach, "n_detach": self.n_detach,
                "n_growth": self.n_growth, "n_etch": self.n_etch,
                "S": self.sticking_fraction(), "R": self.growth_rate(),
                "mean_height": self.mean_height(), "H": self.H,
                "n_bottom_expansions": self.n_bottom_expansions, "stop_reason": stop_reason}

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history, columns=["t", "eventos", "n_attach", "n_detach",
                                                   "altura_media", "W_tot"])

    # ------------------------------------------------------------------
    # Comprobación
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Compara el estado incremental con una reconstrucción desde cero.

        Lanza AssertionError si códigos, clases, posiciones o el peso total no
        coinciden. Caro: para tests y depuración.
        """
        A = self.occupancy_array().copy()
        ref = SlabKMC.__new__(SlabKMC)
        ref.__dict__.update({k: v for k, v in self.__dict__.items()
                             if k in ("graph", "rates", "L1", "L2", "nb", "allow_enclosed", "e_cr",
                                      "_slots", "_rev", "_full", "frozen", "reach")})
        ref._install(A)
        mine = np.frombuffer(self._code, dtype=np.int64)
        theirs = np.frombuffer(ref._code, dtype=np.int64)
        # el sustrato congelado nunca se usa: se comparan los códigos por encima
        lim = self._flim
        assert np.array_equal(mine[lim:], theirs[lim:]), "códigos incrementales incorrectos"
        a = {k: sorted(m) for k, m in zip(self._key_of, self._members) if m}
        b = {k: sorted(m) for k, m in zip(ref._key_of, ref._members) if m}
        assert a == b, "clases incrementales incorrectas"
        for cid, m in enumerate(self._members):
            for p, i in enumerate(m):
                assert self._cls_of[i] == cid and self._pos_of[i] == p, "posiciones de clase incorrectas"
        exact = sum(self._exact_weights())
        assert abs(self._fw.total - exact) <= 1e-9 * max(1.0, exact), "peso total desviado"
        assert self.n_occ == int(A.sum()), "N no coincide con la ocupación"
        top = np.flatnonzero(A.reshape(self.H, -1).any(axis=1))
        assert top.max() + 2 * self.reach + 2 <= self.H, "la superficie está demasiado cerca del techo"

    # ------------------------------------------------------------------
    # pickle: las funciones internas no se serializan; se reconstruyen al cargar
    # ------------------------------------------------------------------
    def __getstate__(self):
        skip = ("_reclass", "_flip", "_occ", "_code", "_cls_of", "_pos_of", "_fw", "_key_of",
                "_kind_of", "_rate_of", "_members", "_cid_of_key", "_nbr", "_energy_cache",
                "_peak")
        state = {k: v for k, v in self.__dict__.items() if k not in skip}
        state["_occupancy"] = self.occupancy_array().copy()
        return state

    def __setstate__(self, state):
        A = state.pop("_occupancy")
        self.__dict__.update(state)
        # Al recargar, el orden interno de las clases cambia: la continuación es
        # estadísticamente equivalente pero no idéntica bit a bit a no haber parado.
        self._install(A)
