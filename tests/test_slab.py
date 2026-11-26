"""Motor de superficie SlabKMC (modo slab de MONTY) y tasas MontyRates."""

import hashlib
import pickle
import sys
import unittest
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import monty, presets  # noqa: E402
from kmc3d.basis_graph import BasisGraph, BasisBond  # noqa: E402
from kmc3d.geometry import Cell  # noqa: E402
from kmc3d.rates import SplitRates  # noqa: E402
from kmc3d.slab import MontyRates, SlabKMC  # noqa: E402


def kossel(phi=4.0):
    return BasisGraph.from_crystal_graph(presets.kossel(phi))


def site_frac(k: SlabKMC, idx: np.ndarray) -> np.ndarray:
    """Coordenadas fraccionarias en la celda de capa (a, b, c + x_s) de sitios planos."""
    q, s = np.divmod(idx, k.nb)
    c, col = np.divmod(q, k.L1 * k.L2)
    a, b = np.divmod(col, k.L2)
    return np.c_[a, b, c].astype(float) + k.xs[s], s, c


def brute_force_energies(k: SlabKMC, sites: np.ndarray) -> np.ndarray:
    """Energía de enlace de cada sitio contando sus vecinos ocupados "a mano".

    No usa las tablas de vecinos del motor: parte de los vectores cartesianos de los
    enlaces del grafo, los pasa a la celda de capa y busca la ocupación con el contorno
    periódico. Comprueba la transformación a la celda de capa y los códigos de bits.
    """
    occ = k.occupancy_array().ravel()
    allidx = np.arange(occ.size)
    frac, _, _ = site_frac(k, allidx)
    key = lambda f: (int(round(f[0] * 1e6)) % (k.L1 * 10**6), int(round(f[1] * 1e6)) % (k.L2 * 10**6),
                     int(round(f[2] * 1e6)))
    lookup = {key(f): int(o) for f, o in zip(frac, occ)}
    to_slab = np.linalg.inv(k.frame @ k.graph.cell.matrix)      # cartesiano -> celda de capa
    out = []
    fs, ss, _ = site_frac(k, sites)
    for f, s in zip(fs, ss):
        e = 0.0
        for sl in k.graph.slots(int(s)):
            vec = (np.asarray(sl.shift, float) + k.graph.basis[sl.j] - k.graph.basis[s]) @ k.graph.cell.matrix
            e += sl.energy * lookup.get(key(f + vec @ to_slab), 1)   # fuera de la caja (abajo): cristal
        out.append(e)
    return np.array(out)


class TestMontyRates(unittest.TestCase):
    def test_equals_split_rates(self):
        """MONTY (λ1, λ2) = SplitRates(a_mu = 1 − λ2, b_E = λ1), código a código."""
        for g in (presets.kossel(2.0), presets.beta_hematin()):
            for l1, l2, dmu in ((0.5, 0.5, 1.3), (0.0, 0.0, 0.7), (1.0, 0.0, -0.4), (0.3, 0.8, 2.0)):
                sr = SplitRates(g, dmu, 1.7, a_mu=1 - l2, b_E=l1)
                mr = MontyRates(dmu, l1, l2, k0=1.7)
                for code in range(0, g.full_code + 1, max(1, g.full_code // 200)):
                    x = g.code_energy(code) - g.kink_energy
                    self.assertAlmostEqual(mr.attach(x), sr.attach(code), places=9)
                    self.assertAlmostEqual(mr.detach(x), sr.detach(code), places=9)

    def test_detailed_balance_and_limits(self):
        for l1, l2 in ((0.5, 0.5), (0.0, 0.0), (0.2, 0.9)):
            r = MontyRates(1.1, l1, l2)
            for x in (-6.0, -1.0, 0.0, 2.5):
                self.assertAlmostEqual(log(r.attach(x) / r.detach(x)), 1.1 + x, places=10)
        gil = MontyRates(0.8, 0.0, 0.0)               # lluvia aleatoria: unión uniforme
        self.assertAlmostEqual(gil.attach(-5.0), exp(0.8))
        self.assertAlmostEqual(gil.attach(1.0), exp(0.8))
        with self.assertRaises(ValueError):
            MontyRates(1.0, 1.5, 0.5)


class TestSlabState(unittest.TestCase):
    def test_incremental_matches_rebuild(self):
        """Estado incremental = reconstrucción, con contorno periódico y caja que crece."""
        bh = BasisGraph.from_crystal_graph(presets.beta_hematin())
        cases = [(kossel(), (0, 0, 1), 2.0), (kossel(), (1, 1, 1), 3.0),
                 (bh, (1, 0, 0), 1.5), (bh, (2, -1, -3), 2.5),
                 (bh.supercell(1, 2, 1), (0, 1, 1), 2.0)]
        for g, hkl, dmu in cases:
            k = SlabKMC(g, hkl, MontyRates(dmu), size=(7, 6), rng_seed=4, room=2)
            for _ in range(4):
                k.run(max_events=1500)
                k.validate()
            self.assertEqual(k.n_occ, k._n_ref + k.n_attach - k.n_detach)
        # la regla de KMC3D (un hueco cerrado no se llena) también es coherente
        k = SlabKMC(bh, (2, -1, -3), MontyRates(2.5), size=(7, 6), rng_seed=4, room=2,
                    allow_enclosed=False)
        for _ in range(3):
            k.run(max_events=1500)
            k.validate()

    def test_codes_match_brute_force(self):
        """Energía de cada sitio = la contada a mano desde los vectores de enlace (base > 1)."""
        bh = BasisGraph.from_crystal_graph(presets.beta_hematin())
        for g, hkl in ((bh.supercell(2, 1, 1), (0, 1, 1)), (bh.supercell(1, 1, 2), (2, -1, -3)),
                       (kossel().supercell(1, 2, 1), (1, 1, 1))):
            k = SlabKMC(g, hkl, MontyRates(2.0), size=(5, 4), rng_seed=2)
            k.run(max_events=2000)
            code = np.frombuffer(k._code, dtype=np.int64)
            lo, hi = (k.frozen + k.reach) * k._layer, (k.H - k.reach) * k._layer
            rng = np.random.default_rng(0)
            sites = rng.choice(np.arange(lo, hi), size=300, replace=False)
            mine = np.array([k.site_energy(int(i % k.nb), int(code[i])) for i in sites])
            np.testing.assert_allclose(mine, brute_force_energies(k, sites), atol=1e-12)

    def test_growth_and_etching_expand_box(self):
        k = SlabKMC(kossel(2.0), (1, 1, 1), MontyRates(4.0), size=(5, 5), rng_seed=1, room=2)
        h0 = k.H
        k.run(max_events=6000)
        self.assertGreater(k.H, h0)
        self.assertGreater(k.mean_height(), 1.0)
        k.validate()
        k = SlabKMC(kossel(2.0), (0, 0, 1), MontyRates(-3.0), size=(5, 5), rng_seed=1, room=3)
        k.run(max_events=6000)
        self.assertGreater(k.n_bottom_expansions, 0)
        self.assertLess(k.mean_height(), -2.0)
        k.validate()

    def test_determinism_and_pickle(self):
        def make(seed):
            return SlabKMC(kossel(3.0), (0, 1, 1), MontyRates(1.0), size=(6, 6), rng_seed=seed)

        def sig(k):
            return (k.n_attach, k.n_detach, round(k.t, 12),
                    hashlib.sha256(k.occupancy_array().tobytes()).hexdigest())

        a, b = make(5), make(5)
        a.run(max_events=3000)
        b.run(max_events=3000)
        self.assertEqual(sig(a), sig(b))
        c = make(6)
        c.run(max_events=3000)
        self.assertNotEqual(sig(a), sig(c))
        a2 = pickle.loads(pickle.dumps(a))
        self.assertEqual(sig(a2)[:3], sig(a)[:3])
        np.testing.assert_array_equal(a2.occupancy_array(), a.occupancy_array())
        a2.validate()
        a2.run(max_events=1000)
        a2.validate()

    def test_checks(self):
        with self.assertRaises(ValueError):               # grafo en kcal/mol
            g = BasisGraph(Cell(), [[0, 0, 0]], [BasisBond(0, 0, (1, 0, 0), 1.0),
                                                 BasisBond(0, 0, (0, 1, 0), 1.0),
                                                 BasisBond(0, 0, (0, 0, 1), 1.0)], units="kcal/mol")
            SlabKMC(g, (0, 0, 1), MontyRates(1.0), size=(4, 4))
        with self.assertRaises(ValueError):               # caja demasiado pequeña
            g = BasisGraph(Cell(), [[0, 0, 0]], [BasisBond(0, 0, (1, 0, 0), 1.0),
                                                 BasisBond(0, 0, (2, 0, 0), 1.0),
                                                 BasisBond(0, 0, (0, 0, 1), 1.0),
                                                 BasisBond(0, 0, (0, 1, 0), 1.0)])
            SlabKMC(g, (0, 0, 1), MontyRates(1.0), size=(4, 4))


class TestSlabPhysics(unittest.TestCase):
    def test_flat_vs_rough_faces(self):
        """Kossel φ = 4 kT, Δμ = 1 kT: (001) casi no crece; (111) (cara K) crece ya."""
        S = {}
        for hkl in ((0, 0, 1), (1, 1, 1)):
            k = SlabKMC(kossel(4.0), hkl, MontyRates(1.0), size=(16, 16), rng_seed=3)
            k.run(max_events=5000)
            k.reset_counters()
            k.run(max_events=20000)
            S[hkl] = k.sticking_fraction()
        self.assertLess(S[(0, 0, 1)], 0.1)
        self.assertGreater(S[(1, 1, 1)], 0.4)

    def test_equilibrium(self):
        """Δμ = 0: uniones y despegues se compensan (S ≈ 0) también en una cara rugosa."""
        k = SlabKMC(kossel(2.0), (1, 1, 1), MontyRates(0.0), size=(16, 16), rng_seed=8)
        k.run(max_events=5000)
        k.reset_counters()
        k.run(max_events=40000)
        self.assertLess(abs(k.sticking_fraction()), 0.03)

    def test_particle_hole_symmetry(self):
        """Reglas de MONTY y λ = ½: crecer a +Δμ = disolver a −Δμ (N_g y N_e se intercambian)."""
        ratio = {}
        for dmu in (1.0, -1.0):
            k = SlabKMC(kossel(3.0), (1, 1, 1), MontyRates(dmu), size=(16, 16), rng_seed=11)
            k.run(max_events=5000)
            k.reset_counters()
            k.run(max_events=30000)
            ratio[dmu] = k.n_growth / k.n_etch
        self.assertAlmostEqual(ratio[1.0] * ratio[-1.0], 1.0, delta=0.05)
        self.assertGreater(ratio[1.0], 2.0)               # ≈ 2.5: lejos de 1, la prueba no es trivial

    def test_sticking_point_and_onsets(self):
        r = monty.sticking_point({"graph": "kossel", "phi": 3.0, "hkl": "(011)", "delta_mu": 2.0,
                                  "size": (8, 8), "n_equil": 1000, "n_sample": 3000})
        self.assertEqual(r["hkl"], "(011)")
        self.assertTrue(0.0 < r["S"] <= 1.0)
        self.assertEqual(r["n_growth"] + r["n_etch"], 3000)
        r = monty.sticking_point({"graph": "kossel", "phi": 3.0, "hkl": (0, 0, 1), "T": 300.0,
                                  "delta_mu_kcal": 1.0, "size": (6, 6), "n_equil": 200,
                                  "n_sample": 500, "keep_map": False})
        self.assertAlmostEqual(r["delta_mu"], 1.0 / (monty.R_KCAL * 300.0))
        # umbrales de una curva sintética
        self.assertAlmostEqual(monty.crossing([0, 1, 2, 3], [0, 0, 0.4, 0.8], 0.5), 2.25)
        self.assertTrue(np.isnan(monty.crossing([0, 1], [0, 0.2], 0.5)))

    def test_compare_with_reference(self):
        """Comparación por grupos con la tabla de MONTY, con curvas sintéticas."""
        ref = monty.load_reference()
        sub = ref[ref["grasa"] == "betap_10_12_10"]
        filas = []
        for _, r in sub.iterrows():
            centro = r["dmu_S05"] if np.isfinite(r["dmu_S05"]) else r["dmu_texto"] + 1.0
            for cara in r["caras"].split():
                for x in np.arange(0.0, 60.0, 0.5):     # escalón suave centrado en `centro`
                    filas.append({"hkl": cara, "delta_mu_kcal": x, "S": 1 / (1 + np.exp(-(x - centro) * 3))})
        comp = monty.compare_with_reference(pd.DataFrame(filas), ref, "betap_10_12_10")
        self.assertEqual(len(comp), len(sub))
        self.assertTrue(comp["cumple"].all())
        self.assertEqual(monty.order_agreement(comp), 1.0)

    def test_rounding_with_strong_bonds(self):
        """Energías de decenas de kT (β-16.16.16 de MONTY): tasas de clase de 1e-35 a 1e13.

        Restar del Fenwick la tasa de un adátomo (~1e10) cuando el total es ~1e-3 dejaba
        residuos de redondeo comparables al total: la elección se sesgaba y acababa en
        IndexError (evento 46 440 con esta semilla). Al elegir, el total debe ser exacto.
        """
        g = monty.graph_from_params({"graph": str(monty.DATA_DIR / "beta_16_16_16.json"), "T": 313.0})
        dmu = monty.delta_mu_kT({"delta_mu_kcal": 8.0, "T": 313.0})
        k = SlabKMC(g, (0, -1, 0), MontyRates(dmu), size=(40, 40), rng_seed=1)
        fw, find = k._fw, k._fw.find
        worst = [0.0]

        def checked(target):
            s = sum(k._exact_weights())
            worst[0] = max(worst[0], abs(fw.total - s) / s)
            return find(target)

        fw.find = checked                  # mide el error del total justo antes de elegir
        k.run(max_events=50_000)
        self.assertLess(worst[0], 1e-9)
        k.validate()


if __name__ == "__main__":
    unittest.main()
