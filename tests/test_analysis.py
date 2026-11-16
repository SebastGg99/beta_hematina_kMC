"""Análisis: base de cara, mapas de alturas, semillas poliédricas, probabilidades."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import KMC3D, SplitRates, presets, seeds  # noqa: E402
from kmc3d.analysis import (attachment_energy_morphology, attachment_probabilities,  # noqa: E402
                            event_balance, face_frame, face_height_map,
                            face_surface_observables, shape_descriptors,
                            snapshot_face_distances)


class TestFaceFrame(unittest.TestCase):
    def test_unimodular_frame(self):
        for hkl in ((1, 0, 0), (0, 1, 1), (0, 1, -1), (1, 1, 0), (1, 1, 1), (1, 0, -1), (2, 1, 0)):
            B = face_frame(hkl)
            h = np.array(hkl)
            self.assertEqual(int(round(abs(np.linalg.det(B)))), 1, msg=str(hkl))
            np.testing.assert_array_equal(B @ h, [0, 0, 1])


class TestHeightMaps(unittest.TestCase):
    def test_flat_box_face(self):
        coords = seeds.box(12, 12, 6)
        hmap = face_height_map(coords, (0, 0, 1), crop=0.8)
        self.assertTrue(np.all(hmap == 0))
        self.assertGreaterEqual(min(hmap.shape), 8)

    def test_step_is_detected(self):
        coords = seeds.box(12, 12, 6)
        top = coords[:, 2].max()
        extra = np.array([[i, j, top + 1] for i in range(-6, 0) for j in range(-6, 6)])
        hmap = face_height_map(np.vstack([coords, extra]), (0, 0, 1), crop=1.0, band=2)
        self.assertEqual(set(np.unique(hmap)), {0, 1})

    def test_reuses_src_analysis(self):
        coords = seeds.box(14, 14, 5)
        obs = face_surface_observables(coords, (0, 0, 1), crop=0.8)
        self.assertAlmostEqual(obs["W"], 0.0)
        self.assertEqual(obs["cara"], "(001)")


class TestShapes(unittest.TestCase):
    def test_polyhedron_cube(self):
        pts = seeds.polyhedron({(1, 0, 0): 3, (0, 1, 0): 3, (0, 0, 1): 3})
        self.assertEqual(len(pts), 7 ** 3)
        d = snapshot_face_distances(pts, presets.kossel(), [(1, 0, 0)])
        self.assertEqual(sorted(d["D"]), [3.0, 3.0])
        from kmc3d.analysis import face_mean_plane
        self.assertAlmostEqual(face_mean_plane(pts, (1, 0, 0)), 3.0)
        self.assertAlmostEqual(face_mean_plane(pts, (-1, 0, 0)), 3.0)

    def test_polyhedron_vertices_and_bounds(self):
        v = seeds.polyhedron_vertices({(1, 0, 0): 2.0, (0, 1, 0): 3.0, (0, 0, 1): 4.0})
        self.assertEqual(len(v), 8)
        np.testing.assert_allclose(np.abs(v).max(axis=0), [2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):           # falta la pareja ±(001): no acotado
            seeds.polyhedron({(1, 0, 0): 2.0, (0, 1, 0): 3.0})

    def test_wulff_normalization_and_rates(self):
        from kmc3d.analysis import family_rates, morphology_from_rates
        from kmc3d.geometry import parse_hkl
        import pandas as pd
        self.assertEqual(parse_hkl("(01-1)"), (0, 1, -1))
        table = pd.DataFrame({"cara": ["(100)", "(-100)", "(010)", "(0-10)", "(001)", "(00-1)"],
                              "familia": ["(100)", "(100)", "(010)", "(010)", "(001)", "(001)"],
                              "R_familia": [1.0, 1.0, 50.0, 50.0, 200.0, 200.0]})
        rates = family_rates(table)
        self.assertEqual(rates[(0, 0, 1)], 200.0)
        pts = morphology_from_rates(rates, presets.kossel().cell, 20.0, normalize="largest")
        self.assertLessEqual(np.abs(pts).max(), 21)
        with self.assertRaises(ValueError):
            morphology_from_rates({(1, 0, 0): 1.0, (0, 1, 0): 0.0, (0, 0, 1): 1.0},
                                  presets.kossel().cell, 5.0)

    def test_continuous_wulff_descriptors(self):
        from kmc3d.analysis import wulff_shape_descriptors
        cell = presets.kossel().cell
        cube = wulff_shape_descriptors({(1, 0, 0): 1.0, (0, 1, 0): 1.0, (0, 0, 1): 1.0}, cell)
        self.assertAlmostEqual(cube["L1/L2"], 1.0, delta=0.02)
        # caja 1 x 2 x 4: L ∝ distancias; funciona aunque la forma sea muy plana
        box = wulff_shape_descriptors({(1, 0, 0): 1.0, (0, 1, 0): 2.0, (0, 0, 1): 400.0}, cell)
        self.assertAlmostEqual(box["L1/L2"], 200.0, delta=4.0)
        self.assertAlmostEqual(box["L2/L3"], 2.0, delta=0.04)
        self.assertEqual(box["eje_largo_uvw"], (0, 0, 1))

    def test_attachment_energy_shape_is_needle(self):
        g = presets.beta_hematin()
        pts = attachment_energy_morphology(g, presets.BETA_HEMATIN_FACES, size=30.0)
        sh = shape_descriptors(pts, g.cell)
        self.assertGreater(sh["L1/L2"], 1.5)
        # eje largo casi paralelo a c (la cara (001) es de las más rápidas)
        self.assertLess(sh["angulo_eje_largo"], 25.0)

    def test_attachment_probabilities_sum_to_one(self):
        g = presets.kossel(2.0)
        kmc = KMC3D(g, SplitRates.symmetric(g, 1.0), seeds.cube(6), rng_seed=1)
        kmc.run(max_events=2000)
        p = attachment_probabilities(kmc)
        self.assertAlmostEqual(float(p["P"].sum()), 1.0)
        pc = attachment_probabilities(kmc, by="class")
        self.assertAlmostEqual(float(pc["P"].sum()), 1.0)
        self.assertGreater(event_balance(kmc)["W_attach/W_detach"], 0.0)


if __name__ == "__main__":
    unittest.main()
