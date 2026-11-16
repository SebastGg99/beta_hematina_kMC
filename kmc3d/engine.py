"""Motor kMC 3D (BKL, sin rechazo) para un cristal finito que crece en todas sus caras.

Diferencias con los motores (2+1)D del proyecto (`src.static`, `src.dynamic`):
- La configuración es la ocupación 0/1 de TODOS los sitios de una caja 3D (no una
  altura por columna): hay voladizos, huecos y todas las caras crecen a la vez.
- Los vecinos y energías salen de un `CrystalGraph` (enlaces por vector de red), así
  que la anisotropía por cara y orientación es automática.
- Las clases de eventos se actualizan de forma INCREMENTAL: tras cada evento solo
  cambian el sitio y sus 2T vecinos, en vez de reclasificar toda la red. Es lo que
  permite redes 3D grandes en Python puro.

Eventos:
- unión (attach) en un sitio vacío con al menos un vecino ocupado;
- despegue (detach) de una unidad ocupada con al menos un vecino vacío;
- salto (hop, solo con `diffusion`): una unidad con al menos un vecino ocupado salta
  a un vecino vacío (difusión superficial, `rates.SurfaceDiffusion`). N se conserva.

Clases (n-fold way): cada sitio elegible pertenece a la clase (tipo, código), donde
el código cuenta los vecinos ocupados de cada tipo de enlace. Todas las unidades de
una clase tienen la misma tasa. El peso total de cada clase (n_c · r_c) se guarda en
un árbol de Fenwick; un evento cuesta O(T + log K). Con difusión, cada unidad de
superficie está además en una clase de salto (por su código): el peso por unidad ya
incluye todos sus vecinos vacíos, y el destino se elige al ejecutar el salto.

Saltos permitidos (simétricos, para que el salto inverso exista siempre):
- el origen tiene algún vecino ocupado (una unidad suelta no salta: solo se despega);
- el destino f, sin contar el origen, tiene algún vecino ocupado (si no, la unidad
  quedaría suelta: eso sería un despegue, no difusión);
- con `allow_enclosed=False`, el destino no es un hueco cerrado y el origen no queda
  como hueco cerrado (vacante interna) tras el salto.
Si el destino elegido no cumple esto, el paso es NULO (rechazo): el tiempo avanza y
la configuración no cambia. Es exacto (aclarado de un proceso de Poisson) y solo
pasa en una fracción pequeña de los saltos (`n_hop_rejected`).

Algoritmo de un paso (BKL):
1. W = Σ_c n_c r_c; Δt = −ln(u1) / W (× time_scale);
2. clase c con probabilidad n_c r_c / W (Fenwick); sitio uniforme dentro de c;
3. ejecutar, actualizar los códigos de los 2T vecinos y reclasificar 2T + 1 sitios.

Caja: el cristal vive en una caja finita con un margen vacío. Si una unidad se une a
menos de `margin` sitios del borde, la caja se amplía por ese lado y se reconstruyen
las clases (operación vectorizada y poco frecuente). Las coordenadas que devuelve el
motor son RELATIVAS al centro de la semilla y no cambian al ampliar la caja.
"""

from array import array
from math import log
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fenwick import Fenwick
from .geometry import reduce_hkl
from .graph import CrystalGraph
from .rates import RateModel, SurfaceDiffusion

ATTACH, DETACH, HOP = 0, 1, 2
EVENT_NAMES = {ATTACH: "attach", DETACH: "detach", HOP: "hop"}
Snapshot = Tuple[float, np.ndarray]


def _int32_array(values: np.ndarray) -> array:
    """numpy -> array('i') compacto (acceso escalar rápido desde Python)."""
    out = array("i")
    out.frombytes(np.ascontiguousarray(values, dtype=np.int32).tobytes())
    return out


class KMC3D:
    """kMC 3D de unión/despegue sobre un grafo cristalino arbitrario."""

    def __init__(self, graph: CrystalGraph, rates: RateModel, seed_coords: np.ndarray,
                 rng_seed: Optional[int] = None, time_scale: float = 1.0,
                 allow_enclosed: bool = False, margin: Optional[int] = None,
                 room: int = 8, track_faces: Sequence[Sequence[int]] = (),
                 rebuild_every: int = 20_000, buffer_size: int = 4096,
                 diffusion: Optional[SurfaceDiffusion] = None):
        """
        Parámetros:
            graph: red y enlaces
            rates: modelo de tasas construido sobre el MISMO `graph`
            diffusion: difusión superficial (`rates.SurfaceDiffusion`, mismo `graph`);
                None = sin saltos (el motor se comporta exactamente como antes)
            seed_coords: (N, 3) índices de red de la semilla (ver `seeds.py`)
            rng_seed: semilla del generador (np.random.default_rng)
            time_scale: multiplica cada Δt (como en los motores 2D)
            allow_enclosed: si False, un hueco con TODOS los vecinos ocupados no
                recibe unidades (la solución no llega a él); ver PLAN, decisión D4
            margin: distancia mínima (en sitios) entre el cristal y el borde de la caja
            room: espacio vacío inicial alrededor de la semilla (y mínimo al ampliar)
            track_faces: caras (hkl) cuya posición se sigue en cada registro: se
                guarda el plano más externo en +(hkl) y en −(hkl)
            rebuild_every: cada cuántos eventos se recalcula el Fenwick desde cero
            buffer_size: números aleatorios que se generan de golpe
        """
        if rates.graph is not graph:
            raise ValueError("`rates` debe construirse con el mismo `graph`")
        if diffusion is not None and diffusion.graph is not graph:
            raise ValueError("`diffusion` debe construirse con el mismo `graph`")
        self.graph = graph
        self.rates = rates
        self.diffusion = diffusion
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.time_scale = float(time_scale)
        self.allow_enclosed = bool(allow_enclosed)
        self.reach = graph.reach
        self.margin = int(margin) if margin is not None else 2 * self.reach + 2
        self.room = max(int(room), self.margin + 1)
        self.rebuild_every = int(rebuild_every)
        self._buffer_size = int(buffer_size)

        # Estado dinámico
        self.t = 0.0
        self.n_events = 0
        self.n_attach = 0
        self.n_detach = 0
        self.n_hop = 0                 # saltos ejecutados
        self.n_hop_rejected = 0        # pasos nulos (destino no permitido)
        self.n_expansions = 0
        self._ubuf: List[float] = []
        self._upos = 0

        # Semilla: coordenadas únicas y centradas (origen = centro redondeado)
        seed = np.unique(np.asarray(seed_coords, dtype=int).reshape(-1, 3), axis=0)
        if len(seed) == 0:
            raise ValueError("La semilla está vacía")
        center = np.round(seed.mean(axis=0)).astype(int)
        self.origin = center                         # índice absoluto del origen relativo
        seed = seed - center
        lo, hi = seed.min(axis=0), seed.max(axis=0)
        dims = hi - lo + 1 + 2 * self.room
        offset = self.room - lo                      # índice de caja = relativo + offset
        A = np.zeros(tuple(int(d) for d in dims), dtype=np.uint8)
        A[tuple((seed + offset).T)] = 1
        self.n_occ = int(len(seed))
        self.n_seed = int(len(seed))
        self._install(A, offset)

        # Seguimiento de caras: histograma de planos h·r de las unidades ocupadas
        self.track_faces = tuple(reduce_hkl(f) for f in track_faces)
        self._plane_counts: List[Dict[int, int]] = []
        self._plane_max: List[int] = []
        self._plane_min: List[int] = []
        for f in self.track_faces:
            planes = seed @ np.asarray(f)
            vals, cnts = np.unique(planes, return_counts=True)
            self._plane_counts.append({int(v): int(c) for v, c in zip(vals, cnts)})
            self._plane_max.append(int(vals.max()))
            self._plane_min.append(int(vals.min()))
        self.history: List[tuple] = []

    # ------------------------------------------------------------------
    # Construcción (vectorizada) de códigos, clases y Fenwick
    # ------------------------------------------------------------------
    def _install(self, A: np.ndarray, offset: np.ndarray) -> None:
        """(Re)construye todas las estructuras a partir de la ocupación A (caja)."""
        nx, ny, nz = A.shape
        self.dims = (nx, ny, nz)
        self._nz, self._nyz = nz, ny * nz
        self.offset = np.asarray(offset, dtype=int)
        weights = self.graph.code_weights
        self._full = self.graph.full_code

        # Desplazamientos planos de los 2T vecinos y peso de cada uno en el código
        nbr = []
        for v, w in zip(self.graph.vectors, weights):
            off = int((v[0] * ny + v[1]) * nz + v[2])
            nbr.append((off, w))
            nbr.append((-off, w))
        self._nbr = nbr
        self._nbr_offsets = [o for o, _ in nbr]

        # Código de cada sitio: Σ_t (vecinos ocupados del tipo t) · 3^t
        Ai = A.astype(np.int32)
        code = np.zeros(A.shape, dtype=np.int64)
        for v, w in zip(self.graph.vectors, weights):
            sh = tuple(int(x) for x in v)
            # np.roll(A, -v)[r] = A[r + v]; el borde de la caja está vacío, así que
            # el "enrollado" periódico de np.roll no introduce vecinos falsos
            neigh = np.roll(Ai, tuple(-x for x in sh), axis=(0, 1, 2)) + np.roll(Ai, sh, axis=(0, 1, 2))
            code += w * neigh

        occ = A.astype(bool)
        att = (~occ) & (code != 0)
        if not self.allow_enclosed:
            att &= code != self._full
        det = occ & (code != self._full)
        elig = np.flatnonzero((att | det).ravel())
        keys = code.ravel()[elig] * 2 + occ.ravel()[elig]   # clave = código·2 + tipo

        # Clases: una por clave, miembros en orden creciente de sitio (determinista)
        self._key_of: List[int] = []
        self._kind_of: List[int] = []
        self._rate_of: List[float] = []
        self._members: List[List[int]] = []
        self._cid_of_key: Dict[int, int] = {}
        cls_of = np.full(A.size, -1, dtype=np.int32)
        pos_of = np.zeros(A.size, dtype=np.int32)
        order = np.argsort(keys, kind="stable")
        sk, ss = keys[order], elig[order]
        if len(sk):
            starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
            ends = np.r_[starts[1:], len(sk)]
            for a, b in zip(starts, ends):
                cid = self._new_class(int(sk[a]))
                sites = ss[a:b]
                self._members[cid] = sites.tolist()
                cls_of[sites] = cid
                pos_of[sites] = np.arange(b - a, dtype=np.int32)

        # Clases de salto (difusión): una por código de origen, para las unidades con
        # algún vecino ocupado y alguno vacío. Se crean DESPUÉS de las de unión/despegue,
        # así que sin difusión los índices de clase (y la trayectoria) no cambian.
        self._hop_cid_of_code: Dict[int, int] = {}
        self._hcls_of = self._hpos_of = None
        if self.diffusion is not None:
            hop = occ & (code != 0) & (code != self._full)
            hsites = np.flatnonzero(hop.ravel())
            hcodes = code.ravel()[hsites]
            hcls_of = np.full(A.size, -1, dtype=np.int32)
            hpos_of = np.zeros(A.size, dtype=np.int32)
            order = np.argsort(hcodes, kind="stable")
            sc, ss = hcodes[order], hsites[order]
            if len(sc):
                starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
                ends = np.r_[starts[1:], len(sc)]
                for a, b in zip(starts, ends):
                    cid = self._new_hop_class(int(sc[a]))
                    sites = ss[a:b]
                    self._members[cid] = sites.tolist()
                    hcls_of[sites] = cid
                    hpos_of[sites] = np.arange(b - a, dtype=np.int32)
            self._hcls_of = _int32_array(hcls_of)
            self._hpos_of = _int32_array(hpos_of)

        self._occ = bytearray(A.astype(np.uint8).ravel().tobytes())
        self._code = _int32_array(code.ravel())
        self._cls_of = _int32_array(cls_of)
        self._pos_of = _int32_array(pos_of)
        self._fw = Fenwick(max(64, 2 * len(self._key_of)))
        self._fw.rebuild(self._exact_weights())
        self._make_hot_path()

    def _new_class(self, key: int) -> int:
        """Registra una clase nueva (sin tocar el Fenwick) y devuelve su id."""
        kind = key & 1
        code = key >> 1
        rate = self.rates.detach(code) if kind == DETACH else self.rates.attach(code)
        cid = len(self._key_of)
        self._key_of.append(key)
        self._kind_of.append(kind)
        self._rate_of.append(float(rate))
        self._members.append([])
        self._cid_of_key[key] = cid
        return cid

    def _new_hop_class(self, code: int) -> int:
        """Registra una clase de salto (código de origen) y devuelve su id.

        En `_key_of` se guarda el propio código: las clases de salto solo se consultan a
        través de `_hcls_of`/`_hop_cid_of_code`, así que no se confunden con las claves
        código·2 + tipo de unión/despegue.
        """
        cid = len(self._key_of)
        self._key_of.append(code)
        self._kind_of.append(HOP)
        self._rate_of.append(float(self.diffusion.class_rate(code)))
        self._members.append([])
        self._hop_cid_of_code[code] = cid
        return cid

    def _exact_weights(self) -> List[float]:
        return [len(m) * r for m, r in zip(self._members, self._rate_of)]

    def _make_hot_path(self) -> None:
        """Crea las funciones internas de un evento con variables locales (más rápido)."""
        occ, code = self._occ, self._code
        cls_of, pos_of = self._cls_of, self._pos_of
        members_all, key_of, rate_of = self._members, self._key_of, self._rate_of
        cid_of_key = self._cid_of_key
        full = self._full
        allow_enclosed = self.allow_enclosed
        nbr, nbr_off = self._nbr, self._nbr_offsets
        fw = self._fw
        fw_add = fw.add

        def create_class(key: int) -> int:
            cid = self._new_class(key)
            if cid >= fw.n:                       # ampliar capacidad del Fenwick
                fw.rebuild(self._exact_weights(), capacity=2 * fw.n)
            return cid

        def reclass(s: int) -> None:
            """Mueve el sitio s a la clase que le toca según su ocupación y código."""
            c = code[s]
            if occ[s]:
                key = -1 if c == full else (c << 1) | 1
            elif c == 0 or (c == full and not allow_enclosed):
                key = -1
            else:
                key = c << 1
            old = cls_of[s]
            if old >= 0:
                if key_of[old] == key:
                    return
                m = members_all[old]                # quitar con "swap-and-pop"
                last = m.pop()
                if last != s:
                    p = pos_of[s]
                    m[p] = last
                    pos_of[last] = p
                fw_add(old, -rate_of[old])
                cls_of[s] = -1
            if key >= 0:
                cid = cid_of_key.get(key)
                if cid is None:
                    cid = create_class(key)
                m = members_all[cid]
                pos_of[s] = len(m)
                m.append(s)
                cls_of[s] = cid
                fw_add(cid, rate_of[cid])

        def flip(s: int, kind: int) -> None:
            """Ejecuta unión (kind=0) o despegue (kind=1) en s y actualiza vecinos."""
            if kind == ATTACH:
                occ[s] = 1
                for off, w in nbr:
                    code[s + off] += w
            else:
                occ[s] = 0
                for off, w in nbr:
                    code[s + off] -= w
            reclass(s)
            for off in nbr_off:
                reclass(s + off)

        self._reclass = reclass
        self._flip = flip
        self._hop = None
        if self.diffusion is None:
            return                                 # sin difusión: camino original intacto

        # --------------------------------------------------------------
        # Con difusión: misma reclasificación + clase de salto de cada sitio
        # --------------------------------------------------------------
        hcls_of, hpos_of = self._hcls_of, self._hpos_of
        hop_cid_of_code = self._hop_cid_of_code
        # Direcciones de salto: (desplazamiento plano, peso en el código del tipo de enlace
        # o 0 si el vector de salto no es un enlace, w_δ)
        dif = self.diffusion
        cws = self.graph.code_weights
        hop_dirs = []
        for v, t, wt in zip(dif.vectors, dif.bond_type, dif.weights):
            off = int((v[0] * self.dims[1] + v[1]) * self.dims[2] + v[2])
            cw = cws[t] if t >= 0 else 0
            hop_dirs.append((off, cw, wt))
            hop_dirs.append((-off, cw, wt))

        def create_hop_class(c: int) -> int:
            cid = self._new_hop_class(c)
            if cid >= fw.n:
                fw.rebuild(self._exact_weights(), capacity=2 * fw.n)
            return cid

        def reclass_dif(s: int) -> None:
            """Como `reclass`, y además coloca s en su clase de salto (o lo quita)."""
            c = code[s]
            o = occ[s]
            if o:
                key = -1 if c == full else (c << 1) | 1
            elif c == 0 or (c == full and not allow_enclosed):
                key = -1
            else:
                key = c << 1
            old = cls_of[s]
            if old >= 0:
                if key_of[old] == key:
                    # la clave es inyectiva en (ocupación, código): si no cambia, la
                    # clase de salto tampoco
                    return
                m = members_all[old]
                last = m.pop()
                if last != s:
                    p = pos_of[s]
                    m[p] = last
                    pos_of[last] = p
                fw_add(old, -rate_of[old])
                cls_of[s] = -1
            if key >= 0:
                cid = cid_of_key.get(key)
                if cid is None:
                    cid = create_class(key)
                m = members_all[cid]
                pos_of[s] = len(m)
                m.append(s)
                cls_of[s] = cid
                fw_add(cid, rate_of[cid])
            # Salto: unidad con algún vecino ocupado (c != 0) y alguno vacío (c != full)
            hk = c if (o and c != 0 and c != full) else -1
            old = hcls_of[s]
            if old >= 0:
                if key_of[old] == hk:
                    return
                m = members_all[old]
                last = m.pop()
                if last != s:
                    p = hpos_of[s]
                    m[p] = last
                    hpos_of[last] = p
                fw_add(old, -rate_of[old])
                hcls_of[s] = -1
            if hk >= 0:
                cid = hop_cid_of_code.get(hk)
                if cid is None:
                    cid = create_hop_class(hk)
                m = members_all[cid]
                hpos_of[s] = len(m)
                m.append(s)
                hcls_of[s] = cid
                fw_add(cid, rate_of[cid])

        def flip_dif(s: int, kind: int) -> None:
            """Como `flip`, con la reclasificación que incluye los saltos."""
            if kind == ATTACH:
                occ[s] = 1
                for off, w in nbr:
                    code[s + off] += w
            else:
                occ[s] = 0
                for off, w in nbr:
                    code[s + off] -= w
            reclass_dif(s)
            for off in nbr_off:
                reclass_dif(s + off)

        def hop(s: int, u: float) -> int:
            """Salto de la unidad en s a un sitio cercano elegido con prob. ∝ w_δ.

            Candidatos: los vecinos de enlace vacíos (el código lo sabe) y todas las
            direcciones de salto que no son enlaces (pueden estar ocupadas). `u` es
            uniforme en [0, 1). Devuelve el sitio de destino, o −1 si el destino elegido
            no está permitido (paso nulo: no cambia nada).
            """
            tot = 0.0
            for off, cw, wt in hop_dirs:
                if cw == 0 or not occ[s + off]:
                    tot += wt
            if not tot > 0.0:             # (solo con pesos w_δ nulos)
                return -1
            target = u * tot
            acc = 0.0
            elegido = None
            for d in hop_dirs:
                if d[1] == 0 or not occ[s + d[0]]:
                    elegido = d           # último candidato visto (protege del redondeo)
                    acc += d[2]
                    if acc > target:
                        break
            off, cw, _ = elegido
            f = s + off
            if occ[f]:                    # diagonal (no enlace) ocupada
                return -1
            cf = code[f] - cw             # vecinos de f SIN contar el origen s
            if cf == 0:                   # la unidad quedaría suelta: no es difusión
                return -1
            if not allow_enclosed and (cf + cw == full or code[s] + cw == full):
                return -1                 # destino cerrado, o el origen quedaría cerrado
            flip_dif(s, DETACH)
            flip_dif(f, ATTACH)
            return f

        self._reclass = reclass_dif
        self._flip = flip_dif
        self._hop = hop

    # ------------------------------------------------------------------
    # Un evento
    # ------------------------------------------------------------------
    def _uniform(self) -> float:
        """Un número uniforme del bloque (solo lo usan los saltos)."""
        p = self._upos
        if p >= len(self._ubuf):
            self._ubuf = self.rng.random(self._buffer_size).tolist()
            p = 0
        self._upos = p + 1
        return self._ubuf[p]

    def _uniforms3(self) -> Tuple[float, float, float]:
        p = self._upos
        if p + 3 > len(self._ubuf):
            self._ubuf = self.rng.random(self._buffer_size).tolist()
            p = 0
        b = self._ubuf
        self._upos = p + 3
        return b[p], b[p + 1], b[p + 2]

    def _exact_class(self, u: float) -> int:
        """Selección exacta (lineal) de clase; se usa si el Fenwick acumuló error."""
        w = self._exact_weights()
        total = sum(w)
        self._fw.rebuild(w)
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
        total = fw.total
        if not total > 0.0:
            # puede ser error de redondeo: recalcular antes de rendirse
            fw.rebuild(self._exact_weights())
            total = fw.total
            if not total > 0.0:
                return False
        u1, u2, u3 = self._uniforms3()
        self.t += -log(1.0 - u1) / total * self.time_scale
        cid = fw.find(u2 * total)
        members = self._members[cid]
        if not members:
            cid = self._exact_class(u2)
            if cid < 0:
                return False
            members = self._members[cid]
        s = members[int(u3 * len(members))]
        kind = self._kind_of[cid]
        if kind == HOP:
            # Salto: el destino se elige con un número aleatorio más (solo hay saltos
            # con difusión, así que sin ella el consumo del RNG no cambia)
            f = self._hop(s, self._uniform())
            self.n_events += 1
            if f < 0:
                self.n_hop_rejected += 1
            else:
                self.n_hop += 1
                self._track(s, -1)
                self._track(f, +1)
                self._check_margin(f)
            if self.n_events % self.rebuild_every == 0:
                self._fw.rebuild(self._exact_weights())
            return True
        self._flip(s, kind)
        self.n_events += 1
        if kind == ATTACH:
            self.n_attach += 1
            self.n_occ += 1
            self._track(s, +1)
            self._check_margin(s)
        else:
            self.n_detach += 1
            self.n_occ -= 1
            self._track(s, -1)
        if self.n_events % self.rebuild_every == 0:
            # self._fw (no `fw`): si la caja se amplió en este evento, el árbol es nuevo
            self._fw.rebuild(self._exact_weights())
        return True

    def _site_index(self, s: int) -> Tuple[int, int, int]:
        """Sitio plano -> índices (i, j, k) de la caja."""
        i, rem = divmod(s, self._nyz)
        j, k = divmod(rem, self._nz)
        return i, j, k

    def _track(self, s: int, sign: int) -> None:
        """Actualiza los histogramas de planos de las caras seguidas."""
        if not self.track_faces:
            return
        i, j, k = self._site_index(s)
        r = (i - self.offset[0], j - self.offset[1], k - self.offset[2])
        for f, (h, kk, l) in enumerate(self.track_faces):
            p = h * r[0] + kk * r[1] + l * r[2]
            cnt = self._plane_counts[f]
            if sign > 0:
                cnt[p] = cnt.get(p, 0) + 1
                if p > self._plane_max[f]:
                    self._plane_max[f] = p
                if p < self._plane_min[f]:
                    self._plane_min[f] = p
            else:
                c = cnt[p] - 1
                if c:
                    cnt[p] = c
                else:
                    del cnt[p]
                    if not cnt:                  # cristal vacío
                        continue
                    if p == self._plane_max[f]:
                        while self._plane_max[f] not in cnt:
                            self._plane_max[f] -= 1
                    if p == self._plane_min[f]:
                        while self._plane_min[f] not in cnt:
                            self._plane_min[f] += 1

    def _check_margin(self, s: int) -> None:
        """Amplía la caja si la unidad recién unida está cerca del borde."""
        idx = self._site_index(s)
        m = self.margin
        if all(m <= x < n - m for x, n in zip(idx, self.dims)):
            return
        pads = []
        for x, n in zip(idx, self.dims):
            g = max(self.room, n // 2)
            pads.append((g if x < m else 0, g if x >= n - m else 0))
        A = np.pad(self.occupancy_array(), pads)
        self._install(A, self.offset + np.array([p[0] for p in pads]))
        self.n_expansions += 1

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------
    def record(self) -> None:
        """Añade una fila al historial (t, eventos, N, uniones, despegues, saltos, W, caras)."""
        row = (self.t, self.n_events, self.n_occ, self.n_attach, self.n_detach, self.n_hop,
               self._fw.total, *self._plane_max, *self._plane_min)
        self.history.append(row)

    def run(self, t_end: Optional[float] = None, max_events: Optional[int] = None,
            n_max: Optional[int] = None, n_min: int = 0, record_every: int = 1000,
            snapshot_times: Optional[Sequence[float]] = None,
            snapshot_every: Optional[int] = None, final_snapshot: bool = True,
            verbose: bool = False):
        """Ejecuta hasta cumplir alguna condición de parada.

        Parámetros:
            t_end: tiempo final; max_events: eventos de esta llamada
            n_max / n_min: para si el cristal llega a n_max unidades o baja a n_min
            record_every: eventos entre filas del historial
            snapshot_times: tiempos en los que guardar las coordenadas ocupadas
            snapshot_every: además, un snapshot cada tantos eventos
            final_snapshot: guarda siempre el estado final como último snapshot
        Devuelve (snaps, stats): snaps = [(t, coords relativas int32 (N, 3)), ...];
        stats = resumen + `history` como DataFrame (ver `history_frame`).
        """
        if t_end is None and max_events is None and n_max is None:
            raise ValueError("Hace falta al menos una condición de parada")
        times = sorted(snapshot_times) if snapshot_times is not None else []
        next_snap = 0
        snaps: List[Snapshot] = []
        if not self.history:
            self.record()
        if times and times[0] <= self.t:
            while next_snap < len(times) and times[next_snap] <= self.t:
                snaps.append((times[next_snap], self.occupied_coords()))
                next_snap += 1
        n0 = self.n_events
        reason = "max_events"
        while True:
            if t_end is not None and self.t >= t_end:
                reason = "t_end"
                break
            if max_events is not None and self.n_events - n0 >= max_events:
                reason = "max_events"
                break
            if n_max is not None and self.n_occ >= n_max:
                reason = "n_max"
                break
            if self.n_occ <= n_min:
                reason = "n_min"
                break
            if not self.step():
                reason = "sin_eventos"
                break
            if self.n_events % record_every == 0:
                self.record()
                if verbose and self.n_events % (record_every * 50) == 0:
                    print(f"t={self.t:.4g} eventos={self.n_events} N={self.n_occ} "
                          f"caja={self.dims}")
            while next_snap < len(times) and self.t >= times[next_snap]:
                snaps.append((times[next_snap], self.occupied_coords()))
                next_snap += 1
            if snapshot_every and self.n_events % snapshot_every == 0:
                snaps.append((self.t, self.occupied_coords()))
        self.record()
        if final_snapshot and (not snaps or snaps[-1][0] != self.t):
            snaps.append((self.t, self.occupied_coords()))
        return snaps, self.stats(stop_reason=reason)

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def occupancy_array(self) -> np.ndarray:
        """Vista numpy (sin copia) de la ocupación de la caja, forma `dims`."""
        return np.frombuffer(self._occ, dtype=np.uint8).reshape(self.dims)

    def occupied_coords(self) -> np.ndarray:
        """Índices RELATIVOS (al centro de la semilla) de las unidades ocupadas."""
        return (np.argwhere(self.occupancy_array()) - self.offset).astype(np.int32)

    def codes_at(self, coords: np.ndarray) -> np.ndarray:
        """Código de vecindad de los sitios con coordenadas RELATIVAS `coords` (N, 3)."""
        idx = np.asarray(coords, dtype=int).reshape(-1, 3) + self.offset
        code = np.frombuffer(self._code, dtype=np.int32).reshape(self.dims)
        return code[tuple(idx.T)]

    def surface_coords(self) -> np.ndarray:
        """Unidades ocupadas con al menos un vecino vacío (relativas)."""
        code = np.frombuffer(self._code, dtype=np.int32).reshape(self.dims)
        mask = self.occupancy_array().astype(bool) & (code != self._full)
        return (np.argwhere(mask) - self.offset).astype(np.int32)

    def face_positions(self) -> pd.DataFrame:
        """Plano más externo y distancia (unidades de la celda) de cada cara seguida."""
        rows = []
        for f, pmax, pmin in zip(self.track_faces, self._plane_max, self._plane_min):
            d = self.graph.cell.d_spacing(f)
            rows.append({"hkl": f, "plano_max": pmax, "plano_min": pmin,
                         "D_mas": pmax * d, "D_menos": -pmin * d})
        return pd.DataFrame(rows)

    def class_table(self) -> pd.DataFrame:
        """Una fila por clase no vacía: tipo, código, enlaces por tipo, energía, tasa."""
        rows = []
        labels = [b.label for b in self.graph.bonds]
        for cid, (key, kind, rate, m) in enumerate(zip(self._key_of, self._kind_of,
                                                         self._rate_of, self._members)):
            if not m:
                continue
            code = key if kind == HOP else key >> 1
            counts = self.graph.decode(code)
            rows.append({"clase": cid, "evento": EVENT_NAMES[kind],
                         "codigo": code, "n_enlaces": int(sum(counts)),
                         "energia": self.rates.energy(code), "tasa": rate, "n_sitios": len(m),
                         "peso": rate * len(m), **dict(zip(labels, counts))})
        return pd.DataFrame(rows)

    def stats(self, stop_reason: str = "") -> dict:
        """Resumen de la corrida (para guardar en metadata)."""
        return {"t": self.t, "n_events": self.n_events, "n_occ": self.n_occ,
                "n_seed": self.n_seed, "n_attach": self.n_attach, "n_detach": self.n_detach,
                "n_hop": self.n_hop, "n_hop_rejected": self.n_hop_rejected,
                "n_expansions": self.n_expansions, "dims": self.dims,
                "stop_reason": stop_reason, "track_faces": self.track_faces,
                "history": self.history_frame()}

    def history_frame(self) -> pd.DataFrame:
        """Historial como DataFrame, con columnas p+ / p− por cara seguida."""
        cols = ["t", "eventos", "N", "n_attach", "n_detach", "n_hop", "W_tot"]
        lab = ["".join(str(x) for x in f) for f in self.track_faces]
        cols += [f"pmax_{s}" for s in lab] + [f"pmin_{s}" for s in lab]
        return pd.DataFrame(self.history, columns=cols)

    # ------------------------------------------------------------------
    # Cambios de condiciones y comprobaciones
    # ------------------------------------------------------------------
    def set_delta_mu(self, delta_mu: float) -> None:
        """Cambia la sobresaturación (Δμ/kT) y actualiza las tasas de todas las clases."""
        self.rates.set_delta_mu(delta_mu)
        for cid, key in enumerate(self._key_of):
            kind = self._kind_of[cid]
            if kind == HOP:
                continue                          # los saltos no dependen de Δμ
            code = key >> 1
            self._rate_of[cid] = (self.rates.detach(code) if kind == DETACH
                                  else self.rates.attach(code))
        self._fw.rebuild(self._exact_weights())

    def validate(self) -> None:
        """Compara el estado incremental con una reconstrucción desde cero.

        Lanza AssertionError si códigos, clases o el peso total no coinciden. Caro
        (O(tamaño de la caja)): para tests y depuración.
        """
        A = self.occupancy_array().copy()
        ref = KMC3D.__new__(KMC3D)
        ref.graph, ref.rates, ref.allow_enclosed = self.graph, self.rates, self.allow_enclosed
        ref.diffusion = self.diffusion
        ref._install(A, self.offset.copy())
        assert self._code.tolist() == ref._code.tolist(), "códigos incrementales incorrectos"
        # clave (tipo, clave interna): las de salto guardan el código y no deben mezclarse
        mine = {(kd, k): sorted(m) for k, kd, m in zip(self._key_of, self._kind_of, self._members) if m}
        theirs = {(kd, k): sorted(m) for k, kd, m in zip(ref._key_of, ref._kind_of, ref._members) if m}
        assert mine == theirs, "clases incrementales incorrectas"
        # posiciones guardadas = posiciones reales en las listas de miembros
        for cid, (kd, m) in enumerate(zip(self._kind_of, self._members)):
            cls_of, pos_of = ((self._hcls_of, self._hpos_of) if kd == HOP
                              else (self._cls_of, self._pos_of))
            for p, s in enumerate(m):
                assert cls_of[s] == cid and pos_of[s] == p, "posiciones de clase incorrectas"
        exact = sum(self._exact_weights())
        assert abs(self._fw.total - exact) <= 1e-9 * max(1.0, exact), "peso total desviado"
        assert self.n_occ == int(A.sum()), "N no coincide con la ocupación"
        # el cristal debe estar lejos del borde (invariante de la caja)
        occ_idx = np.argwhere(A)
        if len(occ_idx):
            assert occ_idx.min() >= self.reach and all(
                occ_idx[:, a].max() < self.dims[a] - self.reach for a in range(3)), "cristal en el borde"
        for f, (pmax, pmin) in enumerate(zip(self._plane_max, self._plane_min)):
            if not len(occ_idx):              # cristal disuelto: no hay planos
                break
            planes = (occ_idx - self.offset) @ np.asarray(self.track_faces[f])
            assert pmax == planes.max() and pmin == planes.min(), "seguimiento de caras incorrecto"

    # ------------------------------------------------------------------
    # pickle: las funciones internas no se serializan; se reconstruyen al cargar
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items()
                 if k not in ("_reclass", "_flip", "_hop", "_occ", "_code", "_cls_of", "_pos_of",
                              "_hcls_of", "_hpos_of", "_hop_cid_of_code",
                              "_fw", "_key_of", "_kind_of", "_rate_of", "_members",
                              "_cid_of_key", "_nbr", "_nbr_offsets")}
        state["_occupancy"] = self.occupancy_array().copy()
        return state

    def __setstate__(self, state):
        A = state.pop("_occupancy")
        # pickles anteriores a la difusión superficial
        state.setdefault("diffusion", None)
        state.setdefault("n_hop", 0)
        state.setdefault("n_hop_rejected", 0)
        self.__dict__.update(state)
        # Al recargar, el orden interno de las clases cambia: la continuación es
        # estadísticamente equivalente pero no idéntica bit a bit a no haber parado.
        self._install(A, self.offset)
