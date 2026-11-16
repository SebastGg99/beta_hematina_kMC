"""Difusión superficial (saltos): balance detallado, estado incremental, conservación.

Comprueba también que SIN difusión el motor da exactamente la misma trayectoria que
antes de añadir los saltos (firma congelada el 2026-11-11 con numpy 2.5.3).
"""

import hashlib
import math
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import KMC3D, SplitRates, presets, seeds  # noqa: E402
from kmc3d.engine import DETACH, ATTACH, HOP  # noqa: E402
from kmc3d.io import engine_from_config  # noqa: E402
from kmc3d.rates import SurfaceDiffusion, diffusion_from_dict  # noqa: E402


def hop_offsets(kmc):
    """[(desplazamiento plano, peso en el código o 0, w_δ)] de las 2 direcciones de cada δ."""
    dif = kmc.diffusion
    out = []
    for v, t, w in zip(dif.vectors, dif.bond_type, dif.weights):
        off = int((v[0] * kmc.dims[1] + v[1]) * kmc.dims[2] + v[2])
        cw = kmc.graph.code_weights[t] if t >= 0 else 0
        out += [(off, cw, w), (-off, cw, w)]
    return out


def allowed_hops(kmc):
    """Todos los saltos permitidos de la configuración: {(origen, destino): tasa}.

    Implementación independiente (recorre todos los sitios ocupados, no las clases).
    """
    occ, code, full = kmc._occ, kmc._code, kmc._full
    out = {}
    for s in np.flatnonzero(kmc.occupancy_array().ravel()).tolist():
        c = code[s]
        if c == 0 or c == full:
            continue
        for off, cw, w in hop_offsets(kmc):
            f = s + off
            if occ[f]:
                continue
            cf = code[f] - cw
            if cf == 0:
                continue
            if not kmc.allow_enclosed and (cf + cw == full or c + cw == full):
                continue
            out[(s, f)] = kmc.diffusion.hop_rate(c) * w
    return out


def clone(kmc):
    """Copia del motor reconstruida desde la ocupación (sin pasar por pickle)."""
    k2 = KMC3D.__new__(KMC3D)
    k2.graph, k2.rates, k2.diffusion = kmc.graph, kmc.rates, kmc.diffusion
    k2.allow_enclosed = kmc.allow_enclosed
    k2._install(kmc.occupancy_array().copy(), kmc.offset.copy())
    return k2


def signature(kmc, snaps):
    """Firma de una trayectoria: contadores, tiempo y hash de los snapshots."""
    h = hashlib.sha256()
    for t, c in snaps:
        h.update(np.float64(t).tobytes())
        h.update(np.ascontiguousarray(c, dtype=np.int32).tobytes())
    return kmc.t, kmc.n_events, kmc.n_occ, kmc.n_attach, kmc.n_expansions, h.hexdigest()[:16]


CASES = [
    ("cubico_bonds", lambda: presets.anisotropic_cubic(1.0, 1.5, 2.0, phi_xy=0.4), "bonds", False),
    ("cubico_cube26_cerrados", lambda: presets.anisotropic_cubic(1.0, 1.5, 2.0, phi_xy=0.4), "cube26", True),
    ("kossel_cube26", lambda: presets.kossel(1.5), "cube26", False),
    ("beta_hematina", presets.beta_hematin, "bonds", False),
]


class TestHopRates(unittest.TestCase):
    def test_vector_sets(self):
        g = presets.kossel(2.0)
        self.assertEqual(len(SurfaceDiffusion(g, 1.0).vectors), 3)
        d26 = SurfaceDiffusion(g, 1.0, hop_vectors="cube26")
        self.assertEqual(len(d26.vectors), 13)
        self.assertEqual(sorted(d26.bond_type)[-3:], [0, 1, 2])   # los 3 enlaces, una vez
        with self.assertRaises(ValueError):
            SurfaceDiffusion(g, 1.0, hop_vectors=[(2, 0, 0)])        # más largo que los enlaces
        with self.assertRaises(ValueError):
            SurfaceDiffusion(g, 1.0, hop_vectors=[(1, 0, 0), (-1, 0, 0)])  # repetido
        with self.assertRaises(ValueError):
            KMC3D(g, SplitRates.gilmer(g, 1.0), seeds.cube(3),
                  diffusion=SurfaceDiffusion(presets.kossel(2.0), 1.0))  # otro grafo

    def test_gilmer_ratio_is_k_d(self):
        # con tasas de Gilmer, salto por dirección / despegue = k_d / k0 en todos los sitios
        g = presets.beta_hematin()
        dif = SurfaceDiffusion(g, 0.37)
        r = SplitRates.gilmer(g, 1.3, k0=2.0)
        for code in (1, 5, 100, 1000, g.full_code - 1):
            self.assertAlmostEqual(dif.hop_rate(code) / r.detach(code), 0.37 / 2.0)

    def test_dict_roundtrip(self):
        g = presets.kossel(2.0)
        d = SurfaceDiffusion(g, 0.5, hop_vectors="cube26", weights=np.linspace(1, 2, 13))
        d2 = diffusion_from_dict(g, d.to_dict())
        self.assertEqual(d2.vectors, d.vectors)
        self.assertEqual(d2.weights, d.weights)
        self.assertIsNone(diffusion_from_dict(g, None))


class TestDetailedBalance(unittest.TestCase):
    """Cada salto permitido tiene su inverso y k(i→f)/k(f→i) = exp(E_f − E_i)."""

    def test_reverse_hop_exists_with_boltzmann_ratio(self):
        for name, make, vecs, enclosed in CASES:
            with self.subTest(name):
                g = make()
                w = np.linspace(0.5, 1.5, len(SurfaceDiffusion(g, 1.0, vecs).vectors))
                dif = SurfaceDiffusion(g, 0.3, hop_vectors=vecs, weights=w)
                seed = seeds.sphere(30.0, g.cell) if g.n_types == 7 else seeds.cube(6)
                kmc = KMC3D(g, SplitRates.symmetric(g, 0.5), seed, rng_seed=3, diffusion=dif,
                            allow_enclosed=enclosed, room=12)
                kmc.run(max_events=3000)
                hops = allowed_hops(kmc)
                self.assertGreater(len(hops), 20)
                for (s, f), rate in list(hops.items())[:60]:
                    k2 = clone(kmc)
                    k2._flip(s, DETACH)
                    k2._flip(f, ATTACH)
                    back = allowed_hops(k2)
                    self.assertIn((f, s), back)
                    e_i = kmc.rates.energy(kmc._code[s])
                    e_f = k2.rates.energy(k2._code[f])
                    self.assertAlmostEqual(math.log(rate / back[(f, s)]), e_f - e_i, places=9)

    def test_class_weight_matches_candidates(self):
        # el peso de la clase de salto = Σ de las tasas de los candidatos que se sortean
        g = presets.kossel(1.5)
        dif = SurfaceDiffusion(g, 0.7, hop_vectors="cube26")
        kmc = KMC3D(g, SplitRates.gilmer(g, 1.0), seeds.cube(6), rng_seed=2, diffusion=dif)
        kmc.run(max_events=2000)
        occ = kmc._occ
        for cid, (kind, m) in enumerate(zip(kmc._kind_of, kmc._members)):
            if kind != HOP:
                continue
            for s in m[:20]:
                cand = sum(w for off, cw, w in hop_offsets(kmc) if cw == 0 or not occ[s + off])
                self.assertAlmostEqual(kmc._rate_of[cid], dif.hop_rate(kmc._code[s]) * cand)


class TestIncrementalWithHops(unittest.TestCase):
    def test_state_matches_rebuild(self):
        for name, make, vecs, enclosed in CASES:
            with self.subTest(name):
                g = make()
                seed = seeds.sphere(30.0, g.cell) if g.n_types == 7 else seeds.cube(4)
                dmu = 0.3 if enclosed else 1.5
                kmc = KMC3D(g, SplitRates.symmetric(g, dmu), seed, rng_seed=7, room=1,
                            diffusion=SurfaceDiffusion(g, 0.2, hop_vectors=vecs),
                            allow_enclosed=enclosed, track_faces=[(1, 0, 0), (0, 1, 1)])
                for _ in range(4):
                    kmc.run(max_events=2500)
                    kmc.validate()
                self.assertGreater(kmc.n_hop, 0)
                self.assertEqual(kmc.n_occ, kmc.n_seed + kmc.n_attach - kmc.n_detach)

    def test_diffusion_only_conserves_N_and_compacts(self):
        # sin unión ni despegue (k0 ínfimo), una varilla se compacta a N fijo. Algún
        # fragmento pequeño puede alejarse (gas de red): la forma se mide en el mayor.
        from kmc3d.analysis import cluster_labels, largest_cluster, shape_descriptors
        g = presets.kossel(3.0)
        rod = seeds.box(3, 3, 18)
        kmc = KMC3D(g, SplitRates.gilmer(g, 0.0, k0=1e-12), rod, rng_seed=2,
                    diffusion=SurfaceDiffusion(g, 1.0, hop_vectors="cube18"))
        snaps, _ = kmc.run(max_events=100_000)
        kmc.validate()
        self.assertEqual(kmc.n_occ, len(rod))
        self.assertEqual(kmc.n_attach + kmc.n_detach, 0)
        self.assertGreater(kmc.n_hop, 10_000)
        big = largest_cluster(snaps[-1][1], g)
        self.assertGreater(len(big), 0.9 * len(rod))
        self.assertLess(shape_descriptors(big, g.cell)["L1/L2"],
                        0.6 * shape_descriptors(rod, g.cell)["L1/L2"])
        # etiquetas: un solo fragmento en la varilla inicial; dos si se separa una unidad
        self.assertEqual(cluster_labels(rod, g).max(), 0)
        two = np.r_[rod, [[10, 10, 10]]]
        self.assertEqual(sorted(np.bincount(cluster_labels(two, g))), [1, len(rod)])

    def test_determinism_and_pickle(self):
        g = presets.beta_hematin()
        dif = SurfaceDiffusion(g, 0.05)

        def make():
            return KMC3D(g, SplitRates.symmetric(g, 1.5), seeds.sphere(30.0, g.cell),
                         rng_seed=11, diffusion=dif)

        a, b = make(), make()
        sa, _ = a.run(max_events=4000, snapshot_every=1000)
        sb, _ = b.run(max_events=4000, snapshot_every=1000)
        self.assertEqual(signature(a, sa), signature(b, sb))
        c = pickle.loads(pickle.dumps(a))
        c.validate()
        self.assertEqual(c.diffusion.to_dict(), dif.to_dict())
        c.run(max_events=2000)
        c.validate()

    def test_config_block(self):
        cfg = {"graph": {"preset": "kossel", "phi": 2.0},
               "rates": {"kind": "gilmer", "delta_mu": 1.0},
               "seed": {"shape": "cube", "n": 5},
               "diffusion": {"k_d": 0.5, "hop_vectors": "cube26"},
               "engine": {"rng_seed": 1}}
        kmc = engine_from_config(cfg)
        self.assertEqual(len(kmc.diffusion.vectors), 13)
        kmc.run(max_events=3000)
        self.assertGreater(kmc.n_hop, 0)


class TestNoDiffusionUnchanged(unittest.TestCase):
    """Sin difusión la trayectoria es la misma que antes de añadir los saltos."""

    def test_frozen_trajectories(self):
        g = presets.kossel(1.5)
        kmc = KMC3D(g, SplitRates.gilmer(g, 1.5), seeds.cube(4), rng_seed=7, room=1,
                    track_faces=[(1, 0, 0), (1, 1, 1)])
        snaps, _ = kmc.run(max_events=30_000, snapshot_every=5_000)
        t, *rest = signature(kmc, snaps)
        self.assertAlmostEqual(t, 2.085747398241265, places=9)
        self.assertEqual(rest, [30000, 11580, 20758, 12, "ab95b522b515f0a7"])

        g = presets.beta_hematin()
        kmc = KMC3D(g, SplitRates.symmetric(g, 1.0), seeds.sphere(40.0, g.cell), rng_seed=3,
                    track_faces=presets.BETA_HEMATIN_FACES)
        snaps, _ = kmc.run(max_events=40_000, snapshot_every=10_000)
        t, *rest = signature(kmc, snaps)
        self.assertAlmostEqual(t, 45.012341894121626, places=9)
        self.assertEqual(rest, [40000, 9171, 24492, 6, "dbda5cfc9478cb84"])


if __name__ == "__main__":
    unittest.main()
