"""Motor KMC3D: estado incremental, ampliación de la caja, determinismo, E/S."""

import json
import pickle
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import KMC3D, SplitRates, presets, seeds  # noqa: E402
from kmc3d.fenwick import Fenwick  # noqa: E402
from kmc3d.io import engine_from_config, load_run, save_run  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class TestFenwick(unittest.TestCase):
    def test_prefix_and_find(self):
        rnd = random.Random(0)
        w = [0.0] * 50
        fw = Fenwick(8)                    # se amplía con rebuild
        fw.rebuild(w, capacity=50)
        for _ in range(500):
            i, d = rnd.randrange(50), rnd.random()
            w[i] += d
            fw.add(i, d)
        self.assertAlmostEqual(fw.total, sum(w))
        for i in range(50):
            self.assertAlmostEqual(fw.prefix(i), sum(w[:i + 1]))
        for _ in range(200):
            u = rnd.random() * sum(w)
            i = fw.find(u)
            self.assertLessEqual(sum(w[:i]), u + 1e-12)
            self.assertGreater(sum(w[:i + 1]), u - 1e-12)
            self.assertGreater(w[i], 0.0)


class TestIncrementalState(unittest.TestCase):
    """Tras muchos eventos (y ampliaciones de caja) el estado = reconstrucción."""

    def _check(self, graph, rates, seed, n_events, **kw):
        kmc = KMC3D(graph, rates, seed, rng_seed=7, **kw)
        for _ in range(4):
            kmc.run(max_events=n_events // 4)
            kmc.validate()
        # conservación: N = semilla + uniones − despegues = unidades ocupadas
        self.assertEqual(kmc.n_occ, kmc.n_seed + kmc.n_attach - kmc.n_detach)
        self.assertEqual(kmc.n_occ, len(kmc.occupied_coords()))
        return kmc

    def test_kossel_growth_with_expansions(self):
        g = presets.kossel(1.5)
        kmc = self._check(g, SplitRates.gilmer(g, 1.5), seeds.cube(4), 8000, room=1,
                          track_faces=[(1, 0, 0), (1, 1, 1)])
        self.assertGreater(kmc.n_expansions, 0)

    def test_beta_hematin_symmetric(self):
        g = presets.beta_hematin()
        kmc = self._check(g, SplitRates.symmetric(g, 2.0), seeds.sphere(30.0, g.cell), 6000,
                          track_faces=presets.BETA_HEMATIN_FACES)
        self.assertGreater(kmc.n_occ, kmc.n_seed)

    def test_full_dissolution(self):
        # semilla por debajo del tamaño crítico: se disuelve entera y run() para en n_min
        g = presets.beta_hematin()
        kmc = KMC3D(g, SplitRates.symmetric(g, 0.5), seeds.sphere(20.0, g.cell), rng_seed=1,
                    track_faces=[(1, 0, 0)])
        _, st = kmc.run(max_events=50_000)
        self.assertEqual(st["stop_reason"], "n_min")
        kmc.validate()

    def test_allow_enclosed_and_dissolution(self):
        g = presets.anisotropic_cubic(1.0, 1.5, 2.0, phi_xy=0.4)
        self._check(g, SplitRates.symmetric(g, -0.3), seeds.cube(6), 4000, allow_enclosed=True)

    def test_set_delta_mu(self):
        g = presets.kossel(2.0)
        kmc = KMC3D(g, SplitRates.gilmer(g, 1.0), seeds.cube(5), rng_seed=1)
        kmc.run(max_events=500)
        kmc.set_delta_mu(0.2)
        kmc.validate()
        for cid, key in enumerate(kmc._key_of):
            expected = kmc.rates.detach(key >> 1) if key & 1 else kmc.rates.attach(key >> 1)
            self.assertAlmostEqual(kmc._rate_of[cid], expected)


class TestDeterminism(unittest.TestCase):
    def _run(self, rng_seed):
        g = presets.beta_hematin()
        kmc = KMC3D(g, SplitRates.gilmer(g, 1.2), seeds.box(4, 4, 4), rng_seed=rng_seed)
        kmc.run(max_events=3000)
        return kmc.t, kmc.occupied_coords()

    def test_same_seed_same_trajectory(self):
        t1, c1 = self._run(3)
        t2, c2 = self._run(3)
        self.assertEqual(t1, t2)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seed_differs(self):
        t1, c1 = self._run(3)
        t2, c2 = self._run(4)
        self.assertNotEqual(t1, t2)


class TestSerialization(unittest.TestCase):
    def test_pickle_roundtrip(self):
        g = presets.kossel(2.0)
        kmc = KMC3D(g, SplitRates.gilmer(g, 1.0), seeds.cube(5), rng_seed=1, track_faces=[(1, 0, 0)])
        kmc.run(max_events=2000)
        k2 = pickle.loads(pickle.dumps(kmc))
        np.testing.assert_array_equal(np.sort(k2.occupied_coords(), axis=0),
                                      np.sort(kmc.occupied_coords(), axis=0))
        self.assertEqual(k2.t, kmc.t)
        k2.validate()
        k2.run(max_events=500)            # se puede continuar
        k2.validate()

    def test_save_load_run(self):
        g = presets.kossel(2.0)
        kmc = KMC3D(g, SplitRates.symmetric(g, 1.0), seeds.cube(5), rng_seed=1,
                    track_faces=[(1, 0, 0), (0, 0, 1)])
        snaps, stats = kmc.run(max_events=1500, snapshot_every=500)
        with tempfile.TemporaryDirectory() as tmp:
            save_run(Path(tmp) / "run", kmc, snaps, stats, config={"x": 1})
            r = load_run(Path(tmp) / "run")
        self.assertEqual(len(r["snaps"]), len(snaps))
        np.testing.assert_array_equal(r["snaps"][-1][1], snaps[-1][1])
        self.assertEqual(r["meta"]["rates"]["kind"], "split")
        self.assertEqual(len(r["history"]), len(stats["history"]))
        self.assertEqual(r["graph"].lattice_energy, g.lattice_energy)

    def test_example_configs_build(self):
        for path in sorted(CONFIGS.glob("*.json")):
            with self.subTest(config=path.name):
                kmc = engine_from_config(json.loads(path.read_text(encoding="utf-8")))
                kmc.run(max_events=200)
                kmc.validate()


if __name__ == "__main__":
    unittest.main()
