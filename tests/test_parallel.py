"""Ejecutor paralelo (subprocess) y experimentos de difusión importables."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import experiments as ex, presets, seeds  # noqa: E402
from kmc3d.parallel import run_parallel  # noqa: E402

PEQ = {"L": 6, "growth": 1.3, "n_snaps": 4, "phi": 2.0, "delta_mu": 1.0}


class TestParallel(unittest.TestCase):
    def test_parallel_equals_serial(self):
        # mismas corridas en serie y en 2 procesos: resultados idénticos y en orden
        ps = [dict(PEQ, k_d=kd, rng_seed=3) for kd in (0.0, 1.0)]
        a = run_parallel("kmc3d.experiments:kossel_face_growth", ps, processes=1)
        b = run_parallel("kmc3d.experiments:kossel_face_growth", ps, processes=2)
        for ra, rb, p in zip(a, b, ps):
            self.assertEqual(ra["k_d"], p["k_d"])
            self.assertEqual(ra["eventos"], rb["eventos"])
            np.testing.assert_array_equal(ra["coords"], rb["coords"])
        self.assertEqual(a[0]["frac_saltos"], 0.0)
        self.assertGreater(a[1]["frac_saltos"], 0.0)

    def test_failure_is_reported(self):
        with self.assertRaises(RuntimeError):
            run_parallel("kmc3d.experiments:kossel_face_growth", [{"rates": "no_existe"}] * 2,
                         processes=2)


class TestExperiments(unittest.TestCase):
    def test_bonds_per_unit(self):
        g = presets.kossel(1.0)
        # caja 2×2×2: 12 enlaces / 8 unidades; varilla 1×1×5: 4 / 5
        self.assertAlmostEqual(ex.bonds_per_unit(seeds.box(2, 2, 2), g), 1.5)
        self.assertAlmostEqual(ex.bonds_per_unit(seeds.box(1, 1, 5), g), 0.8)

    def test_relax_rod_conserves_N(self):
        r = ex.relax_rod({"dims": (2, 2, 8), "n_events": 4000, "n_checkpoints": 2})
        self.assertEqual(r["N"], 32)
        self.assertEqual(len(r["serie"]), 3)
        self.assertEqual(len(r["inicial"]), 32)


if __name__ == "__main__":
    unittest.main()
