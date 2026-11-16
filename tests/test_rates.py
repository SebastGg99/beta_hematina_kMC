"""Modelos de tasas: balance detallado y correspondencia con el motor estático 2D."""

import sys
import unittest
from itertools import product
from math import exp, log
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import KMC3D, presets  # noqa: E402
from kmc3d.rates import (AdaptiveRates, SplitRates, delta_mu_from_sigma,  # noqa: E402
                         rates_from_dict)


class TestSplitRates(unittest.TestCase):
    def test_detailed_balance(self):
        g = presets.beta_hematin()
        for a, b, dmu in ((1.0, 0.0, 0.7), (0.5, 0.5, 1.3), (0.2, 0.8, -0.4)):
            r = SplitRates(g, dmu, k0=2.0, a_mu=a, b_E=b)
            for counts in ((1, 0, 0, 0, 0, 0, 0), (2, 1, 0, 1, 0, 2, 1), (1, 1, 1, 1, 1, 1, 1)):
                code = g.encode(counts)
                # k+/k- = exp(-ΔG), ΔG = -(Δμ + E - E_kink)
                self.assertAlmostEqual(log(r.attach(code) / r.detach(code)), -r.delta_g(code),
                                       places=10)

    def test_kink_is_neutral_at_equilibrium(self):
        g = presets.kossel(2.0)
        r = SplitRates.symmetric(g, 0.0)
        kink = g.encode((1, 1, 1))          # 3 vecinos en Kossel = media-cristal
        self.assertAlmostEqual(r.attach(kink), r.detach(kink))

    def test_gilmer_attach_is_uniform(self):
        g = presets.kossel(2.0)
        r = SplitRates.gilmer(g, 0.8)
        self.assertEqual({round(r.attach(c), 12) for c in range(1, g.full_code)}, {round(exp(0.8), 12)})

    def test_set_delta_mu_clears_cache(self):
        g = presets.kossel(2.0)
        r = SplitRates.gilmer(g, 0.0)
        r.attach(1)
        r.set_delta_mu(1.0)
        self.assertAlmostEqual(r.attach(1), exp(1.0))

    def test_dict_roundtrip(self):
        g = presets.kossel(2.0)
        for r in (SplitRates.symmetric(g, 0.3, k0=3.0), AdaptiveRates(g, 0.5, 2.0, 7.0, [0.1, 0.2, 0.0])):
            r2 = rates_from_dict(g, r.to_dict())
            for code in (1, 5, 13):
                self.assertAlmostEqual(r.attach(code), r2.attach(code))
                self.assertAlmostEqual(r.detach(code), r2.detach(code))

    def test_delta_mu_from_sigma(self):
        self.assertAlmostEqual(delta_mu_from_sigma(np.e - 1), 1.0)
        with self.assertRaises(ValueError):
            delta_mu_from_sigma(-1.0)


class TestAdaptiveVsStatic2D(unittest.TestCase):
    """AdaptiveRates en la cara (001) frente a r_a / r_d de `KMC_BKL_Static`.

    El motor 3D calcula los códigos sobre un cristal hecho de columnas (las alturas de
    una superficie SOS). Donde las dos convenciones de conteo coinciden (ver
    `AdaptiveRates`), las tasas deben ser idénticas; donde no, deben diferir.
    """

    def setUp(self):
        from src.analysis.adsorption import _rate_engine
        from src.static import LatticeSOSStatic

        self.p2d = dict(K0_plus=1.7, E_pb_over_kT_x=1.1, E_pb_over_kT_y=0.6, phi_over_kT=3.2,
                        delta_x=0.3, delta_y=0.1, V=1.0, C_eq=1.0)
        self.sigma = 1.5
        self.eng2d = _rate_engine(self.p2d, self.sigma)
        S = self.eng2d.supersaturation
        phi_z = 2.0
        self.graph = presets.anisotropic_cubic(1.1, 0.6, phi_z)
        self.rates = AdaptiveRates(self.graph, S, K0=1.7, phi0=3.2 + phi_z, deltas=[0.3, 0.1, 0.0])

        rng = np.random.default_rng(4)
        self.L = 10
        self.h = rng.integers(2, 6, size=(self.L, self.L))
        self.lat = LatticeSOSStatic(size=(self.L, self.L), seed=0)
        self.lat.heights = self.h.astype(np.int32)
        cols = [(i, j, k) for i in range(self.L) for j in range(self.L) for k in range(self.h[i, j])]
        self.kmc = KMC3D(self.graph, self.rates, np.array(cols), rng_seed=0)

    def _neighbour_heights(self, i, j):
        return [self.h[i + di, j + dj] for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]

    def test_rates_match_where_conventions_agree(self):
        n_att = n_det = n_diff = 0
        for i, j in product(range(1, self.L - 1), repeat=2):
            hn, h = self._neighbour_heights(i, j), self.h[i, j]
            att_site = np.array([[i, j, h]]) - self.kmc.origin
            det_site = np.array([[i, j, h - 1]]) - self.kmc.origin
            k_att = self.rates.attach(int(self.kmc.codes_at(att_site)[0]))
            k_det = self.rates.detach(int(self.kmc.codes_at(det_site)[0]))
            r_a = self.eng2d.r_a(*self.lat.adsorption_bonds_xy((i, j)))
            r_d = self.eng2d.r_d(*self.lat.desorption_bonds_xy((i, j)))
            if max(hn) <= h + 1:
                self.assertAlmostEqual(k_att, r_a, places=10)
                n_att += 1
            elif abs(k_att - r_a) > 1e-9:
                n_diff += 1
            if max(hn) <= h:
                self.assertAlmostEqual(k_det, r_d, places=10)
                n_det += 1
            elif abs(k_det - r_d) > 1e-9:
                n_diff += 1
        # la comparación no es vacía y la diferencia de convención existe de verdad
        self.assertGreater(n_att, 10)
        self.assertGreater(n_det, 5)
        self.assertGreater(n_diff, 10)


if __name__ == "__main__":
    unittest.main()
