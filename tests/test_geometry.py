"""Geometría de la celda: espaciados, recíproca de β-hematina, índices."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # raíz del repositorio -> kmc3d

from kmc3d.geometry import Cell, family, hkl_label, reduce_hkl, unique_faces  # noqa: E402
from kmc3d.presets import BETA_HEMATIN_CELL  # noqa: E402


class TestCell(unittest.TestCase):
    def test_cubic_spacings(self):
        c = Cell()
        self.assertAlmostEqual(c.d_spacing((1, 0, 0)), 1.0)
        self.assertAlmostEqual(c.d_spacing((1, 1, 0)), 1 / np.sqrt(2))
        self.assertAlmostEqual(c.d_spacing((1, 1, 1)), 1 / np.sqrt(3))
        self.assertAlmostEqual(c.volume, 1.0)

    def test_beta_hematin_reciprocal(self):
        # Solomonov et al., JACS 129, 2615 (2007), nota 26: a*, b*, c* (Å^-1), α*, β*, γ* (°)
        ref = (0.08338, 0.06876, 0.12527, 88.83, 83.05, 82.00)
        got = BETA_HEMATIN_CELL.reciprocal_parameters()
        for g, r in zip(got[:3], ref[:3]):
            self.assertAlmostEqual(g, r, delta=3e-5)
        for g, r in zip(got[3:], ref[3:]):
            self.assertAlmostEqual(g, r, delta=0.1)

    def test_volume_formula(self):
        c = BETA_HEMATIN_CELL
        ca, cb, cg = (np.cos(np.radians(x)) for x in (c.alpha, c.beta, c.gamma))
        v = c.a * c.b * c.c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
        self.assertAlmostEqual(c.volume, v, places=6)

    def test_normal_is_perpendicular_to_plane_vectors(self):
        c = BETA_HEMATIN_CELL
        # [100] y [01-1] están en el plano (011); [1-10] y [001] en (110)
        for hkl, vecs in (((0, 1, 1), ((1, 0, 0), (0, 1, -1))), ((1, 1, 0), ((1, -1, 0), (0, 0, 1)))):
            n = c.normal(hkl)
            for v in vecs:
                self.assertAlmostEqual(float(n @ c.to_cartesian(np.array(v))), 0.0, places=10)

    def test_d_spacing_is_plane_separation(self):
        # la proyección de una traslación con h·r = 1 sobre la normal es d_hkl
        c = BETA_HEMATIN_CELL
        for hkl, r in (((1, 0, 0), (1, 0, 0)), ((0, 1, 1), (0, 1, 0)), ((1, -1, 0), (1, 0, 0))):
            self.assertAlmostEqual(float(c.normal(hkl) @ c.to_cartesian(np.array(r))),
                                   c.d_spacing(hkl), places=10)


class TestIndices(unittest.TestCase):
    def test_reduce_and_label(self):
        self.assertEqual(reduce_hkl((2, 0, -4)), (1, 0, -2))
        self.assertEqual(hkl_label((0, 1, -1)), "(01-1)")

    def test_family_and_unique(self):
        self.assertEqual(set(family((0, 1, 1))), {(0, 1, 1), (0, -1, -1)})
        # (200) se reduce a (100); (-100) es una cara distinta (no se fusiona con +)
        self.assertEqual(unique_faces([(1, 0, 0), (-1, 0, 0), (2, 0, 0), (0, 1, 0)]),
                         ((1, 0, 0), (-1, 0, 0), (0, 1, 0)))


if __name__ == "__main__":
    unittest.main()
