"""Grafos de las grasas de MONTY transcritos de Hollander et al. 2003 (data/monty2004)."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import transcribir_grasas as T  # noqa: E402

from kmc3d import monty  # noqa: E402
from kmc3d.geometry import parse_hkl  # noqa: E402


class TestGrasas(unittest.TestCase):
    def test_json_al_dia(self):
        """Los JSON de data/monty2004 son los que genera scripts/transcribir_grasas.py."""
        for name in T.GRASAS:
            en_disco = json.loads((monty.DATA_DIR / f"{name}.json").read_text())
            generado = json.loads(json.dumps(T.json_dict(name)))
            self.assertEqual(en_disco, generado, name)

    def test_grafos(self):
        """Z, número de enlaces y energía de red de cada grafo (kcal/mol por GU)."""
        esperado = {"beta_16_16_16": (2, 8, 88.25), "betap_10_12_10": (8, 32, 63.30),
                    "betap_16_16_14": (8, 48, 91.61)}
        for name, (Z, nb, ecr) in esperado.items():
            g, meta = monty.load_graph_file(monty.DATA_DIR / f"{name}.json")
            self.assertEqual((g.n_basis, len(g.bonds)), (Z, nb), name)
            self.assertAlmostEqual(g.lattice_energy, ecr, places=2)
            self.assertEqual(g.units, "kcal/mol")
            # las caras de MONTY son las de referencia_figuras.csv
            ref = monty.load_reference()
            caras = " ".join(ref[ref["grasa"] == name]["caras"]).split()
            self.assertEqual(meta["faces"], caras)

    def test_tabla3(self):
        """E_att de la red conexa más fuerte = Tabla 3 de Hollander (±3 %).

        Excepción documentada: (112) de β′-16.16.14, que la Tabla 3 da igual que (110)
        (254.75); el grafo da la misma red que (111) y (113) (196), como agrupa MONTY.
        """
        for name in T.GRASAS:
            df = T.check(name, T.build_graph(name))
            t3 = df[df["origen"] == "tabla3"]
            for _, r in t3.iterrows():
                if name == "betap_16_16_14" and r["hkl"] == "(112)":
                    self.assertAlmostEqual(r["E_att_red_conexa"], 196.03, places=1)
                    continue
                self.assertLess(abs(r["dif_red_pct"]), 3.0, f"{name} {r['hkl']}")

    def test_morfologia_energia_union(self):
        """Formas del poliedro de energía de unión con cortes planos (como la Fig. 1).

        Con los enlaces de la Tabla 2 salen la basal de las tres grasas, las superiores de
        β-16.16.16 que da la Fig. 1 ({2-1-3} y {1-1-1}) y las laterales {101} de
        β′-10.12.10. Las diferencias con la Fig. 1 están en el notebook 03, §3.2.
        """
        esperado = {"beta_16_16_16": {"{001}", "{10-1}", "{1-1-1}", "{2-1-3}"},
                    "betap_10_12_10": {"{001}", "{101}"},
                    "betap_16_16_14": {"{001}", "{111}", "{100}"}}
        for name, formas in esperado.items():
            g = T.build_graph(name)
            ep = T.energias_corte_plano(name, T.caras_candidatas())
            _, tab = T.morfologia_energia_union(g.cell, dict(zip(ep["hkl"], ep["E_att"])))
            vistas = {T.forma(name, parse_hkl(f)) for f in tab[tab["fraccion"] > 0.001]["familia"]}
            self.assertTrue(formas <= vistas, f"{name}: {sorted(vistas)}")
            self.assertGreater(tab["fraccion"].max(), 0.8)       # placas: domina la basal
        # (001) de la red I de β′-10.12.10 se calcula en su capa propia, (002)
        self.assertEqual(T.capa_propia((0, 0, 1), T.GRASAS["betap_10_12_10"]["primitiva"])[1], (0, 0, 2))
        self.assertEqual(T.forma("betap_10_12_10", (1, -1, 2)), "{112}")

    def test_solo_subida(self):
        """run_monty --solo-subida: la zona de subida de cada curva y un punto a cada lado."""
        import pandas as pd
        import run_monty
        df = pd.DataFrame({"hkl": ["(001)"] * 6 + ["(011)"] * 3,
                           "delta_mu_kcal": [0, 1, 2, 3, 4, 5, 0, 1, 2],
                           "S": [0.0, 0.0, 0.3, 0.99, 1.0, 1.0, 0.0, 0.0, 0.0]})
        zona = run_monty.subida(df, "delta_mu_kcal", 0.02, 0.98)
        self.assertEqual(zona, {("(001)", 1.0), ("(001)", 2.0), ("(001)", 3.0)})


if __name__ == "__main__":
    unittest.main()
