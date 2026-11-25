"""Grafo con base (BasisGraph) y celda de capa: energías, supercelda, simetría, E/S."""

import sys
import tempfile
import unittest
from itertools import product
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import presets  # noqa: E402
from kmc3d.basis_graph import (R_KCAL, BasisBond, BasisGraph, kcal_to_kT,  # noqa: E402
                               kT_to_kcal, slice_cell)
from kmc3d.geometry import Cell  # noqa: E402


def coprime_faces(n: int = 3):
    """Todas las (hkl) coprimas con |índices| <= n, sin repetir ±."""
    out = []
    for f in product(range(-n, n + 1), repeat=3):
        if any(f) and gcd(gcd(abs(f[0]), abs(f[1])), abs(f[2])) == 1:
            if tuple(-x for x in f) not in out:
                out.append(f)
    return out


class TestFromCrystalGraph(unittest.TestCase):
    def test_energies_match_crystal_graph(self):
        for g in (presets.kossel(2.0), presets.beta_hematin(),
                  presets.anisotropic_cubic(1.0, 2.0, 3.0, phi_xy=0.5)):
            bg = BasisGraph.from_crystal_graph(g)
            self.assertAlmostEqual(bg.lattice_energy, g.lattice_energy)
            self.assertAlmostEqual(bg.full_energy(0), 2 * g.lattice_energy)
            faces = coprime_faces(2)
            a, b = g.face_table(faces), bg.face_table(faces)
            np.testing.assert_allclose(a["E_att"], b["E_att"], atol=1e-12)
            np.testing.assert_allclose(a["E_slice"], b["E_slice"], atol=1e-12)
            np.testing.assert_allclose(a["d_hkl"], b["d_hkl"], rtol=1e-12)

    def test_bond_validation(self):
        cell = Cell()
        with self.assertRaises(ValueError):            # repetido como inverso
            BasisGraph(cell, [[0, 0, 0]], [BasisBond(0, 0, (1, 0, 0), 1.0),
                                           BasisBond(0, 0, (-1, 0, 0), 1.0)])
        with self.assertRaises(ValueError):            # energía no positiva
            BasisGraph(cell, [[0, 0, 0]], [BasisBond(0, 0, (1, 0, 0), 0.0)])
        with self.assertRaises(ValueError):            # GU consigo misma
            BasisBond(0, 0, (0, 0, 0), 1.0)


class TestSupercell(unittest.TestCase):
    def test_supercell_preserves_energetics(self):
        bg = BasisGraph.from_crystal_graph(presets.beta_hematin())
        n = (2, 1, 3)
        sc = bg.supercell(*n)
        self.assertEqual(sc.n_basis, 6)
        self.assertEqual(len(sc.bonds), 6 * len(bg.bonds))
        self.assertAlmostEqual(sc.lattice_energy, bg.lattice_energy)
        for s in range(sc.n_basis):
            self.assertAlmostEqual(sc.full_energy(s), bg.full_energy(0))
        # longitudes de enlace: las mismas, cada una repetida 6 veces
        la = sorted(np.linalg.norm(bg.bond_vector(k)) for k in range(len(bg.bonds)) for _ in range(6))
        lb = sorted(np.linalg.norm(sc.bond_vector(k)) for k in range(len(sc.bonds)))
        np.testing.assert_allclose(la, lb, rtol=1e-12)
        # la cara (hkl) del original es (h·n1, k·n2, l·n3) en la supercelda: misma E_att y E_slice
        for f in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, -1, 0), (0, 1, -1)):
            fs = tuple(x * m for x, m in zip(f, n))
            ea = bg.slice_energies(f, bg.best_termination(f))
            eb = sc.slice_energies(fs, sc.best_termination(fs))
            self.assertAlmostEqual(ea["E_att"], eb["E_att"])
            self.assertAlmostEqual(ea["E_slice"], eb["E_slice"])
            self.assertAlmostEqual(bg.d_spacing(f), sc.d_spacing(fs))


class TestFromCenters(unittest.TestCase):
    def test_inversion_pair(self):
        """P-1 con una molécula independiente: la inversión da Z = 2 y enlaces simétricos."""
        cell = Cell(5.0, 6.0, 7.0, 95.0, 100.0, 85.0)
        x = np.array([0.20, 0.10, 0.30])
        inv = (-np.eye(3), np.zeros(3))
        bonds = [{"from": 0, "to_frac": (-x).tolist(), "energy": 3.0, "label": "dímero"},
                 {"from": 0, "to_frac": (x + [1, 0, 0]).tolist(), "energy": 1.5, "label": "a"},
                 {"from": 0, "to_frac": (-x + [0, 1, 0]).tolist(), "energy": 0.7, "label": "c1"}]
        g = BasisGraph.from_centers(cell, [x], bonds, [(np.eye(3), np.zeros(3)), inv])
        self.assertEqual(g.n_basis, 2)
        # El dímero y c1 unen moléculas relacionadas por un centro de inversión: la
        # inversión los deja igual (1 enlace cada uno). La traslación a aparece en las
        # dos subredes (2 enlaces).
        self.assertEqual(len(g.bonds), 1 + 2 + 1)
        self.assertAlmostEqual(g.full_energy(0), 3.0 + 2 * 1.5 + 0.7)
        self.assertAlmostEqual(g.full_energy(0), g.full_energy(1))
        self.assertAlmostEqual(g.lattice_energy, (3.0 + 2 * 1.5 + 0.7) / 2)
        np.testing.assert_allclose(g.basis[1], np.mod(-x, 1.0))

    def test_neighbor_must_exist(self):
        with self.assertRaises(ValueError):
            BasisGraph.from_centers(Cell(), [[0.1, 0.1, 0.1]],
                                    [{"from": 0, "to_frac": [0.5, 0.5, 0.5], "energy": 1.0}])


class TestUnitsAndIO(unittest.TestCase):
    def test_kcal_kT(self):
        self.assertAlmostEqual(kcal_to_kT(R_KCAL * 300.0, 300.0), 1.0)
        self.assertAlmostEqual(kT_to_kcal(kcal_to_kT(7.3, 313.0), 313.0), 7.3)
        g = BasisGraph(Cell(), [[0, 0, 0]], [BasisBond(0, 0, (1, 0, 0), 6.0)], units="kcal/mol")
        gk = g.to_kT(300.0)
        self.assertEqual(gk.units, "kT")
        self.assertAlmostEqual(gk.bonds[0].energy, 6.0 / (R_KCAL * 300.0))

    def test_json_roundtrip(self):
        g = BasisGraph.from_crystal_graph(presets.beta_hematin()).supercell(1, 2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            p = g.to_json(Path(tmp) / "g.json", extra={"fuente": "test"})
            g2 = BasisGraph.from_json(p)
        self.assertEqual(g2.n_basis, g.n_basis)
        self.assertEqual([b.canonical() for b in g2.bonds], [b.canonical() for b in g.bonds])
        np.testing.assert_allclose(g2.basis, g.basis)
        self.assertAlmostEqual(g2.lattice_energy, g.lattice_energy)


class TestSliceCell(unittest.TestCase):
    def test_unimodular_and_in_plane(self):
        cell = presets.BETA_HEMATIN_CELL
        G = cell.matrix @ cell.matrix.T
        faces = coprime_faces(3) + [(3, 1, 7), (3, 0, -5), (6, 1, 1), (1, -1, 4), (6, 0, 2)]
        for f in faces:
            B = slice_cell(f, cell)
            h0 = np.asarray(f) // gcd(gcd(abs(f[0]), abs(f[1])), abs(f[2]))
            self.assertEqual(round(np.linalg.det(B)), 1)
            self.assertEqual(int(h0 @ B[0]), 0)
            self.assertEqual(int(h0 @ B[1]), 0)
            self.assertEqual(int(h0 @ B[2]), 1)
            # base reducida de Lagrange-Gauss: |u·v| <= |u|²/2 y |u| <= |v|
            u, v = B[0], B[1]
            self.assertLessEqual(abs(u @ G @ v), 0.5 * (u @ G @ u) + 1e-9)
            self.assertLessEqual(u @ G @ u, v @ G @ v + 1e-9)


if __name__ == "__main__":
    unittest.main()
