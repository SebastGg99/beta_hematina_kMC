"""Grafo cristalino: códigos de vecindad y teoría de Hartman-Perdok."""

import sys
import unittest
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import presets  # noqa: E402
from kmc3d.graph import BondType, CrystalGraph  # noqa: E402
from kmc3d.geometry import Cell  # noqa: E402


class TestCodes(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        g = presets.beta_hematin()
        for counts in ((0,) * 7, (2,) * 7, (1, 0, 2, 1, 0, 2, 1)):
            self.assertEqual(g.decode(g.encode(counts)), counts)
        self.assertEqual(g.decode(g.full_code), (2,) * 7)

    def test_code_energy(self):
        g = presets.anisotropic_cubic(1.0, 2.0, 3.0)
        self.assertAlmostEqual(g.code_energy(g.encode((1, 2, 0))), 1.0 + 4.0)
        self.assertAlmostEqual(g.code_energy(g.full_code), 2 * g.lattice_energy)
        self.assertAlmostEqual(g.kink_energy, 6.0)

    def test_duplicate_bond_rejected(self):
        with self.assertRaises(ValueError):
            CrystalGraph(Cell(), (BondType((1, 0, 0), 1.0), BondType((-1, 0, 0), 1.0)))

    def test_dict_roundtrip_and_with_energies(self):
        g = presets.beta_hematin()
        g2 = CrystalGraph.from_dict(g.to_dict())
        self.assertEqual([b.vector for b in g2.bonds], [b.vector for b in g.bonds])
        self.assertEqual(g2.cell, g.cell)
        g3 = g.with_energies({"[001]": 3.0})
        self.assertAlmostEqual(g3.lattice_energy, g.lattice_energy + 0.6)
        with self.assertRaises(KeyError):
            g.with_energies({"[111]": 1.0})


class TestHartmanPerdok(unittest.TestCase):
    def test_slice_plus_attachment_is_lattice_energy(self):
        g = presets.beta_hematin()
        for hkl in product(range(-2, 3), repeat=3):
            if hkl == (0, 0, 0):
                continue
            es, ea = g.slice_attachment(hkl)
            self.assertAlmostEqual(es + ea, g.lattice_energy)

    def test_kossel_face_characters(self):
        g = presets.kossel(2.0)
        self.assertEqual(g.face_character((1, 0, 0)), "F")
        self.assertEqual(g.face_character((1, 1, 0)), "S")
        self.assertEqual(g.face_character((1, 1, 1)), "K")
        # E_att(100) = una unión por unidad (la de abajo) = φ
        self.assertAlmostEqual(g.slice_attachment((1, 0, 0))[1], 2.0)

    def test_second_neighbours_make_110_flat(self):
        g = presets.anisotropic_cubic(1.0, 1.0, 1.0, phi_xy=0.5)
        self.assertEqual(g.face_character((1, 1, 0)), "F")
        self.assertEqual(g.face_character((1, 0, 0)), "F")

    def test_beta_hematin_provisional_ordering(self):
        """Las energías provisionales deben ordenar |E_att| como la morfología observada."""
        g = presets.beta_hematin()
        e = {f: g.slice_attachment(f)[1] for f in ((1, 0, 0), (0, 1, 0), (0, 1, 1),
                                                   (0, 0, 1), (0, 1, -1))}
        self.assertLess(e[(1, 0, 0)], e[(0, 1, 0)])
        self.assertLess(e[(0, 1, 0)], e[(0, 1, 1)])
        self.assertLess(e[(0, 1, 1)], e[(0, 0, 1)])
        self.assertLess(e[(0, 0, 1)], e[(0, 1, -1)])
        # todas las caras observadas son F
        for f in e:
            self.assertEqual(g.face_character(f), "F")

    def test_face_table_columns(self):
        t = presets.kossel(1.0).face_table([(1, 0, 0), (1, 1, 0)])
        self.assertEqual(list(t["hkl"]), ["(100)", "(110)"])
        self.assertAlmostEqual(float(t["R_att_rel"].iloc[1]), 2.0)


if __name__ == "__main__":
    unittest.main()
