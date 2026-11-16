"""Comprobaciones físicas cualitativas (Fase 1 de PLAN_3DkMC.md).

Son corridas cortas (segundos): verifican el SIGNO y el ORDEN de los efectos, no
valores cuantitativos. La validación cuantitativa está en el notebook 01.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import KMC3D, SplitRates, presets, seeds  # noqa: E402
from kmc3d.analysis import face_growth_rates, face_series, shape_descriptors  # noqa: E402

AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def _face_rates(graph, delta_mu, seed, n_events, rng_seed=1, n_max=None, rates="gilmer"):
    """Velocidades de las familias seguidas (media de ±hkl)."""
    model = getattr(SplitRates, rates)(graph, delta_mu)
    kmc = KMC3D(graph, model, seed, rng_seed=rng_seed, track_faces=AXES)
    snaps, st = kmc.run(max_events=n_events, n_max=n_max, record_every=500)
    R = face_growth_rates(face_series(st["history"], graph, AXES))
    return kmc, snaps, R.groupby("familia")["R_familia"].first()


class TestDrivingForce(unittest.TestCase):
    def test_growth_sign_follows_delta_mu(self):
        g = presets.kossel(2.0)
        for dmu, grows in ((-0.5, False), (1.0, True)):
            kmc = KMC3D(g, SplitRates.symmetric(g, dmu), seeds.cube(10), rng_seed=2)
            kmc.run(max_events=20_000)
            self.assertEqual(kmc.n_occ > kmc.n_seed, grows, msg=f"Δμ = {dmu}")


class TestSmoothVersusRough(unittest.TestCase):
    """Kossel: a Δμ bajo una cara lisa (φ alto) casi no avanza (nucleación 2D) y una
    rugosa (φ bajo, por encima de la transición de rugosidad) avanza ∝ Δμ."""

    def test_nucleation_barrier_only_for_smooth_faces(self):
        dmu = 0.5
        rough = _face_rates(presets.kossel(1.0), dmu, seeds.cube(14), 60_000)[2].mean()
        smooth = _face_rates(presets.kossel(3.0), dmu, seeds.cube(14), 60_000)[2].mean()
        k_plus = np.exp(dmu)
        self.assertGreater(rough / k_plus, 0.2)
        self.assertLess(abs(smooth) / k_plus, 0.05)


class TestAnisotropy(unittest.TestCase):
    def test_needle_along_strongest_bond(self):
        g = presets.anisotropic_cubic(1.2, 1.8, 3.0)
        kmc, snaps, R = _face_rates(g, 1.0, seeds.cube(8), 400_000, n_max=25_000)
        # caras más rápidas = las que cortan los enlaces más fuertes
        self.assertLess(R["(100)"], R["(010)"])
        self.assertLess(R["(010)"], R["(001)"])
        shape = shape_descriptors(snaps[-1][1], g.cell)
        self.assertGreater(shape["L1/L2"], 1.2)
        self.assertEqual(tuple(abs(x) for x in shape["eje_largo_uvw"])[:2], (0, 0))


if __name__ == "__main__":
    unittest.main()
