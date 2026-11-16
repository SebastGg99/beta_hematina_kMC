"""Morfología: caras de poliedros, facetas del cristal kMC, espesores, render y criterios."""

import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d import analysis as an, calibration as cal, presets, render, seeds  # noqa: E402
from kmc3d.geometry import Cell  # noqa: E402

AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


class TestPolyhedronFaces(unittest.TestCase):
    def test_box_areas(self):
        # caja 2 × 4 × 6 (distancias 1, 2, 3): áreas por familia 2·(4·6), 2·(2·6), 2·(2·4)
        cell = Cell()
        caras, tab = an.wulff_faces({(1, 0, 0): 1.0, (0, 1, 0): 2.0, (0, 0, 1): 3.0}, cell)
        self.assertEqual(len(caras), 6)
        area = tab.set_index("familia")["area"]
        self.assertAlmostEqual(area["(100)"], 2 * 4 * 6)
        self.assertAlmostEqual(area["(010)"], 2 * 2 * 6)
        self.assertAlmostEqual(area["(001)"], 2 * 2 * 4)
        self.assertAlmostEqual(tab["fraccion"].sum(), 1.0)
        # vértices ordenados en sentido antihorario visto desde fuera
        for c in caras:
            P = c["vertices"]
            self.assertGreater(np.cross(P[1] - P[0], P[2] - P[1]) @ c["normal"], 0)

    def test_hidden_face_has_zero_area(self):
        # en un cubo, (111) a distancia √3 solo toca un vértice: no aparece
        caras, tab = an.wulff_faces({(1, 0, 0): 1.0, (0, 1, 0): 1.0, (0, 0, 1): 1.0,
                                     (1, 1, 1): np.sqrt(3) * 1.01}, Cell())
        self.assertEqual(tab.set_index("familia")["area"]["(111)"], 0.0)

    def test_beta_hematin_attachment_shape(self):
        g = presets.beta_hematin()
        _, tab = an.wulff_faces(an.attachment_energy_rates(g, presets.BETA_HEMATIN_FACES), g.cell)
        frac = tab.set_index("familia")["fraccion"]
        self.assertEqual(frac.idxmax(), "(100)")
        self.assertEqual(frac["(110)"], 0.0)


class TestFacets(unittest.TestCase):
    def test_box_facets_and_areas(self):
        box = seeds.box(10, 14, 18)
        lab = an.facet_labels(box, Cell(), AXES)
        # el centro de cada cara se asigna a su familia con el signo correcto
        X = box[lab["surface"]]
        centro_z = (X[:, 2] == X[:, 2].max()) & (np.abs(X[:, 0]) <= 1) & (np.abs(X[:, 1]) <= 1)
        self.assertTrue(np.all(lab["family"][centro_z] == 2))
        self.assertTrue(np.all(lab["sign"][centro_z] == 1))

    def test_box_areas_converge(self):
        # Las aristas (una banda del orden de radius_cells) quedan sin cara: el área se
        # subestima en un error relativo ~ 1/L que se reduce a la mitad al duplicar la caja.
        def errores(k):
            a, b, c = 10 * k, 14 * k, 18 * k
            tab = an.facet_areas(seeds.box(a, b, c), Cell(), AXES).set_index("familia")["area"]
            exactas = {"(100)": 2 * b * c, "(010)": 2 * a * c, "(001)": 2 * a * b}
            return np.array([tab[f] / v - 1 for f, v in exactas.items()])
        e1, e2 = errores(1), errores(2)
        self.assertTrue(np.all(e1 < 0) and np.all(e2 < 0))
        self.assertTrue(np.all(np.abs(e2) < 0.2))
        np.testing.assert_allclose(e2 / e1, 0.5, atol=0.05)

    def test_directional_extents_box(self):
        X = Cell().to_cartesian(seeds.box(6, 10, 20))
        L = an.directional_extents(X, np.eye(3))
        # n puntos con paso 1: 12·var = n² − 1
        np.testing.assert_allclose(L, np.sqrt(np.array([6, 10, 20]) ** 2 - 1), rtol=1e-12)


class TestThickness(unittest.TestCase):
    def test_lattice_basis_is_unimodular(self):
        for t in [(0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, -1), (1, 1, 1)]:
            B = render.lattice_basis_along(t)
            self.assertEqual(tuple(B[2]), t)
            self.assertEqual(abs(round(np.linalg.det(B))), 1)

    def test_box_thickness(self):
        cell = presets.BETA_HEMATIN_CELL
        m = render.thickness_map(seeds.box(4, 5, 7), cell, (0, 0, 1))
        th = m["thickness"]
        self.assertEqual(np.count_nonzero(~np.isnan(th)), 4 * 5)
        np.testing.assert_allclose(th[~np.isnan(th)], 7 * cell.c)
        self.assertEqual(m["X"].shape, (th.shape[0] + 1, th.shape[1] + 1))


class TestRenderSmoke(unittest.TestCase):
    def test_render_functions_run(self):
        g = presets.beta_hematin()
        forma = an.attachment_energy_morphology(g, cal.FACES, size=80.0, normalize="largest")
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        _, info = render.render_crystal(forma, g.cell, cal.FACES, ax=ax)
        self.assertIn("labels", info)
        quads, unit, eje = render.exposed_quads(forma, g.cell)
        self.assertEqual(quads.shape[1:], (4, 3))
        caras, _ = an.wulff_faces(an.attachment_energy_rates(g, cal.FACES), g.cell)
        render.plot_polyhedron(caras, cal.FACES, ax=fig.add_subplot(1, 2, 2, projection="3d"),
                               cell=g.cell)
        render.plot_thickness_maps(forma, g.cell)
        plt.close("all")

    def test_face_colors_fold_ninth_family(self):
        col = render.face_colors(cal.FACES)
        self.assertEqual(len(set(list(col.values())[:8])), 8)
        self.assertEqual(col["(10-1)"], render.OTHER_COLOR)


class TestCalibrationCriteria(unittest.TestCase):
    def _metricas(self, **kw):
        m = {"angulo_c": 5.0, "L_c/W": 2.0, "W/T": 3.0, "A(100)": 0.5, "A(010)": 0.25,
             "A(001)": 0.05, "A(011)": 0.1, "A(01-1)": 0.0, "A(110)": 0.02, "A(1-10)": 0.02,
             "A(101)": 0.03, "A(10-1)": 0.03, "deriva_W/T": 0.05, "deriva_L_c/W": -0.05}
        m.update(kw)
        return m

    def test_ideal_shape_passes_all(self):
        ev = cal.evaluate_criteria(self._metricas())
        self.assertEqual(ev["n_criterios"], len(cal.CRITERIOS))
        self.assertAlmostEqual(ev["J"], 0.0)

    def test_each_criterion_can_fail(self):
        casos = {"aguja_c": {"angulo_c": 30.0}, "anchura_espesor": {"W/T": 1.2},
                 "laterales": {"A(110)": 0.4}, "extremos": {"A(101)": 0.3},
                 "sin_0-11": {"A(01-1)": 0.05}, "estacionaria": {"deriva_W/T": 0.4}}
        for crit, cambio in casos.items():
            with self.subTest(crit):
                ev = cal.evaluate_criteria(self._metricas(**cambio))
                self.assertFalse(ev[crit])
                self.assertEqual(ev["n_criterios"], len(cal.CRITERIOS) - 1)
                self.assertGreater(ev["J"], 0.0)

    def test_energies_from_params(self):
        e = cal.energies_from_params(2.0, {"[001]": 3.0})
        self.assertAlmostEqual(e["[001]"], 6.0)
        self.assertAlmostEqual(e["[100]"], 2 * presets.BETA_HEMATIN_ENERGIES_PROVISIONAL["[100]"])


if __name__ == "__main__":
    unittest.main()
