"""Transcribe los grafos de enlaces de las tres grasas de MONTY y los comprueba.

Fuente de las energías: F. F. A. Hollander, S. X. M. Boerrigter, J. van de Streek,
P. Bennema y H. Meekes, «Explaining the morphology of crystals of triacylglycerols…»,
*J. Phys. Chem. B* 107, 5680 (2003) (ref. 17 de MONTY), Tablas 1–3, en
`references/papers/Hollander2003/`. Las Tablas 2 (enlaces) y 3 (caras con al menos una
red conexa y su E_att) se copian tal cual abajo (`TABLA2`, `TABLA3`).

Las posiciones de los centros moleculares no están en el artículo. Solo deciden el
corte plano inicial de `SlabKMC` (las alturas de las GU), no la dinámica, que depende
del grafo (quién se enlaza con quién y con qué energía). Su origen:

- β-16.16.16: estructura de tripalmitina, COD 2003033 (van Langevelde et al.,
  *Acta Cryst. B* 55, 114, 1999), centros de masa de los átomos pesados;
- β′-10.12.10: COD 2102198 (van Langevelde et al., *Acta Cryst. B* 56, 1103, 2000);
- β′-16.16.14: no hay estructura publicada. Las posiciones son una reconstrucción
  compatible con la simetría C2 y con la Tabla 2 (ver `POS_16_16_14`).

Uso (desde la raíz del repositorio):

    python scripts/transcribir_grasas.py                 # escribe los JSON y la comprobación
    python scripts/transcribir_grasas.py --solo-comprobar   # solo imprime

Escribe `data/monty2004/{beta_16_16_16,betap_10_12_10,betap_16_16_14}.json` (formato de
`data/monty2004/README.md`) y `data/monty2004/comprobacion_grafos.csv`.

Comprobación (Tabla 3): la E_att de Hollander es la de la red conexa más fuerte de cada
orientación, buscada de forma topológica (como FACELIFT), por celda primitiva:
E_att = Z_prim·(E_cr − E_slice), con E_slice por GU. Aquí se busca igual: cada subred s
de la celda primitiva recibe un desplazamiento entero k_s de capa y se exige que cada
capa sea conexa y periódica en dos direcciones (rango 2, índice 1). También se da la
E_att del mejor corte PLANO (`BasisGraph.best_termination`), que es el que usa `SlabKMC`
para la superficie inicial.
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kmc3d.basis_graph import BasisBond, BasisGraph, slice_cell  # noqa: E402
from kmc3d.geometry import Cell, hkl_label, parse_hkl, reduce_hkl  # noqa: E402
from kmc3d import monty  # noqa: E402

OUT_DIR = ROOT / "data" / "monty2004"
FUENTE = ("Hollander, Boerrigter, van de Streek, Bennema y Meekes, J. Phys. Chem. B 107, "
          "5680 (2003), Tablas 1-3 (ref. 17 de MONTY)")

# ---------------------------------------------------------------------------
# Tabla 2 de Hollander 2003, tal cual: (enlace, "Mi.Mj", [uvw], φ en kcal/mol).
# "Mi.Mj [uvw]" = de la molécula i de la primera celda a la molécula j de la celda
# [uvw]. φ > 0 es la energía de enlace con el signo cambiado.
# ---------------------------------------------------------------------------
TABLA2 = {
    "beta_16_16_16": [("p", "M1.M1", (0, -1, 0), 32.6),
                      ("q1", "M1.M2", (0, 0, 0), 24.0), ("q2", "M1.M2", (0, -1, 0), 28.6),
                      ("r1", "M1.M2", (-1, 1, 0), 24.3), ("r2", "M1.M2", (-1, 0, 0), 29.1),
                      ("s1", "M1.M2", (0, -1, -1), 2.5), ("s2", "M1.M2", (0, 0, -1), 2.8)],
    "betap_10_12_10": [("p", "M1.M1", (0, -1, 0), 26.3), ("q", "M1.M6", (0, -1, 0), 19.5),
                       ("r", "M1.M7", (-1, -1, 0), 16.1), ("s", "M2.M7", (0, 0, 0), 1.4)],
    "betap_16_16_14": [("pA", "M3.M3", (0, -1, 0), 15.7), ("pB", "M8.M8", (0, -1, 0), 17.0),
                       ("q1", "M7.M4", (0, 1, 0), 22.8), ("q2", "M7.M4", (0, 0, 0), 40.8),
                       ("r1", "M8.M7", (0, 1, 0), 20.8), ("r2", "M8.M7", (0, 0, 0), 40.7),
                       ("u", "M2.M6", (-1, 0, 0), 8.84), ("v", "M1.M5", (0, 1, 0), 8.37),
                       ("x", "M2.M6", (-1, -1, 0), 2.70), ("y", "M1.M5", (0, 0, 0), 2.68),
                       ("sA", "M1.M7", (-1, 0, 0), 1.67), ("sB", "M1.M8", (1, 0, -1), 1.17)],
}

# Tabla 3 de Hollander 2003, tal cual (en el marco de su Tabla 2): (cara, {hkl}, E_att en
# kcal/mol con el signo cambiado, número de redes conexas).
TABLA3 = {
    "beta_16_16_16": [("B1", (0, 0, 1), 5.2, 1), ("S1", (1, 0, 1), 52.5, 1), ("S2", (1, 0, 0), 52.6, 1),
                      ("T5", (0, 1, 0), 118.1, 1), ("T6", (0, 1, -1), 118.3, 1),
                      ("T1A", (1, 1, 1), 113.5, 1), ("T1B", (1, 1, 0), 113.7, 2),
                      ("T1C", (1, 1, -1), 122.3, 1), ("T2A", (2, 1, 1), 117.5, 1),
                      ("T2B", (2, 1, 0), 117.7, 1)],
    "betap_10_12_10": [("B1", (0, 0, 2), 5.44, 1), ("S1", (1, 0, 1), 34.88, 3), ("S2", (2, 0, 0), 64.33, 1),
                       ("T1A", (1, -1, 2), 182.46, 3), ("T2A", (2, -1, 1), 182.46, 3)],
    "betap_16_16_14": [("B1", (0, 0, 1), 2.35, 4), ("B2", (0, 0, 2), 5.69, 3), ("S1", (2, 0, 4), 171.44, 3),
                       ("S2", (2, 0, 3), 168.16, 3), ("T1A", (1, 1, 3), 196.17, 4),
                       ("T1B", (1, 1, 2), 254.75, 8), ("T1C", (1, 1, 1), 196.17, 7),
                       ("T1D", (1, 1, 0), 254.75, 3), ("T3A", (3, 1, 6), 278.16, 5),
                       ("T3B", (3, 1, 5), 278.16, 5), ("T3C", (3, 1, 4), 279.22, 2),
                       ("T3D", (3, 1, 3), 297.15, 1), ("T3E", (3, 1, 7), 299.15, 1)],
}

# Operaciones de simetría (W, t) sobre coordenadas fraccionarias, x' = W·x + t
_I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
OPS_P1BAR = [(_I3, [0, 0, 0]), ([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [0, 0, 0])]
# I c 2 a (eje binario según b), en la celda de Hollander; sale de la de COD 2102198
# (I b a 2, a = 57.368, b = 22.783, c = 5.6945 Å) con a_H = b, b_H = c, c_H = a.
_OPS_IC2A_P = [(_I3, [0, 0, 0]), ([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0, 0]),
               ([[1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0.5, 0]),
               ([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0.5, 0])]
OPS_IC2A = _OPS_IC2A_P + [(W, list(np.add(t, 0.5) % 1.0)) for W, t in _OPS_IC2A_P]
# C 1 2 1 (binario según b) con el centrado (½, ½, 0)
_OPS_C2_P = [(_I3, [0, 0, 0]), ([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0, 0])]
OPS_C2 = _OPS_C2_P + [(W, [0.5, 0.5, 0.0]) for W, _ in _OPS_C2_P]


def _red(x: np.ndarray) -> np.ndarray:
    """Coordenadas reducidas a [0, 1) ("primera celda" de Hollander)."""
    x = np.asarray(x, float)
    return x - np.floor(x + 1e-9)


def posiciones_16_16_14(x3: Sequence[float], x8: Sequence[float]) -> Dict[int, np.ndarray]:
    """Centros M1..M8 de β′-16.16.14 a partir de M3 (tipo A) y M8 (tipo B).

    Reconstrucción (no hay estructura publicada): M7 = C(M3) y M4 = C(M8), con C el
    centrado (½, ½, 0), en la misma capa; la otra capa son las imágenes por el binario
    2 y por 2C: M1 = 2(M3), M5 = 2C(M3), M2 = 2C(M8), M6 = 2(M8). Es la única de las
    cuatro asignaciones de 1, 2, 5 y 6 que reproduce la Tabla 3 (las otras dan otro
    grafo). Todo se reduce a la primera celda.
    """
    W2 = np.diag([-1.0, 1.0, -1.0])
    c = np.array([0.5, 0.5, 0.0])
    x3, x8 = np.asarray(x3, float), np.asarray(x8, float)
    return {3: _red(x3), 8: _red(x8), 7: _red(x3 + c), 4: _red(x8 + c),
            1: _red(W2 @ x3), 5: _red(W2 @ x3 + c), 2: _red(W2 @ x8 + c), 6: _red(W2 @ x8)}


# Parámetros libres de la reconstrucción de β′-16.16.14 (centros de M3 y M8). Se eligieron
# con una búsqueda aleatoria (60 000 pruebas) que deja fijo el grafo y pide contactos de
# la capa q, r entre 4 y 9 Å, u, v, x, y entre 5 y 12 Å y s entre capas de 38 a 52 Å, y
# que el corte plano reproduzca la red conexa más fuerte en el mayor número de caras.
# Ninguna posición de esta familia lo consigue en (110), (313) y (317).
POS_16_16_14 = ((0.1866, 0.7516, 0.7556), (0.4922, 0.6257, 0.7987))

# ---------------------------------------------------------------------------
# Definición de cada grasa en el marco del JSON
# ---------------------------------------------------------------------------
_P = posiciones_16_16_14(*POS_16_16_14)

GRASAS = {
    "beta_16_16_16": {
        # Celda de la Tabla 1 (la que usa MONTY). La Tabla 2 y la Tabla 3 están en la celda
        # con a invertido (α = 74.5°, β = 58.3°, γ = 55.7°): con ella, y no con la de la
        # Tabla 1, las caras de MONTY con red conexa (Fig. 2, grupos 5-7) son las de la
        # Tabla 3. Aquí se usa (a, −b, −c) de esa celda, que tiene la métrica de la Tabla 1,
        # es dextrógira y difiere de la Tabla 1 en la inversión (simetría de P-1): los
        # índices del JSON son los de MONTY. Cambio: [uvw] -> [u, −v, −w], (hkl) -> (h, −k, −l).
        "cell": {"a": 14.0, "b": 5.4, "c": 46.8, "alpha": 74.5, "beta": 121.7, "gamma": 124.3},
        "marco": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "centers": [("M1", (0.2094, 0.9348, 0.4268)), ("M2", (0.7906, 0.0652, 0.5732))],
        "symops": OPS_P1BAR,
        "primitiva": _I3,
        "protocolo": "beta",
        # La Tabla 3 de esta grasa coincide (±0.1 kcal/mol) con p = 31.35 y r2 = 28.3 en vez
        # de 32.6 y 29.1 (Tabla 2): se calculó con otras energías. Se usa la Tabla 2.
        "posiciones": ("COD 2003033 (tripalmitina, P-1), centros de masa de los átomos "
                       "pesados; celda de la Tabla 1 = (a+b, −a, −2a−2b+c) de la de COD; "
                       "origen en un centro de inversión, elegido para que los enlaces de la "
                       "Tabla 2 sean contactos (p = 5.4 Å; q y r de 7 a 11 Å, con los más "
                       "fuertes, q2 y r2, más cortos; s1 y s2 a unos 50 Å)"),
    },
    "betap_10_12_10": {
        # Mismo marco que la Tabla 2 (ortorrómbica, sin ambigüedad de signos que importe)
        "cell": {"a": 22.8, "b": 5.7, "c": 57.4, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "marco": _I3,
        "centers": [("M1", (0.1362, 0.2437, 0.7161)), ("M2", (0.8638, 0.2437, 0.2839)),
                    ("M6", (0.3638, 0.7437, 0.7839)), ("M7", (0.8638, 0.7437, 0.7161))],
        "symops": OPS_IC2A,
        "primitiva": [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
        "protocolo": "beta_prime",
        "posiciones": ("COD 2102198 (β′-10.12.10, I b a 2), centros de masa de los átomos "
                       "pesados, con a_H = b, b_H = c, c_H = a de COD. M1, M2, M6 y M7 son las "
                       "moléculas para las que q = 1.6[0-10] y r = 1.7[-1-10] son contactos de "
                       "la capa y s = 2.7[000] un contacto entre capas (las demás GU salen de "
                       "la simetría). Dos etiquetados reproducen la Tabla 3 por igual (s a 25.0 "
                       "o a 29.2 Å); se usa el de 25.0 Å, el único que da a (1-12) y (2-11) la "
                       "misma red conexa, como dice el texto de Hollander (§4.3)"),
    },
    "betap_16_16_14": {
        "cell": {"a": 16.5, "b": 7.5, "c": 81.6, "alpha": 90.0, "beta": 90.3, "gamma": 90.0},
        "marco": _I3,
        "centers": [(f"M{k}", tuple(float(round(v, 4)) for v in _P[k])) for k in range(1, 9)],
        "symops": OPS_C2,
        "primitiva": [[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
        "protocolo": "beta_prime",
        "posiciones": ("RECONSTRUCCIÓN, no hay estructura publicada: M3 y M8 en "
                       f"{POS_16_16_14[0]} y {POS_16_16_14[1]}, el resto por simetría C2 "
                       "(ver posiciones_16_16_14 en scripts/transcribir_grasas.py). Solo "
                       "deciden el corte plano inicial; la topología del grafo es la misma "
                       "para todas las posiciones probadas"),
    },
}


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------
def _labels_index(centers) -> Dict[str, int]:
    return {lab: k for k, (lab, _) in enumerate(centers)}


def bonds_by_position(name: str) -> List[dict]:
    """Enlaces de la Tabla 2 en el formato de `BasisGraph.from_centers` (marco del JSON)."""
    g = GRASAS[name]
    F = np.asarray(g["marco"], float)
    Finv = np.linalg.inv(F)
    idx = _labels_index(g["centers"])
    pos = [np.asarray(x, float) for _, x in g["centers"]]
    out = []
    for lab, par, uvw, phi in TABLA2[name]:
        mi, mj = par.split(".")
        t = np.rint(np.asarray(uvw, float) @ Finv).astype(int)     # [uvw] en el marco nuevo
        to = pos[idx[mj]] + t
        out.append({"from": idx[mi], "to_frac": [float(round(v, 4)) for v in to],
                    "energy": phi, "label": lab})
    return out


def hkl_json(name: str, hkl_h: Sequence[int]) -> Tuple[int, int, int]:
    """Índices de la Tabla 3 (marco de Hollander) -> marco del JSON: h' = F·h."""
    F = np.asarray(GRASAS[name]["marco"], int)
    return tuple(int(v) for v in F @ np.asarray(hkl_h, int))


def json_dict(name: str) -> dict:
    """Contenido del JSON de una grasa (formato de transcripción del README)."""
    g = GRASAS[name]
    ref = monty.load_reference()
    faces = []
    for _, r in ref[ref["grasa"] == name].iterrows():
        faces.extend(str(r["caras"]).split())
    proto = dict(monty.MONTY_PROTOCOLS[g["protocolo"]])
    proto["size"] = list(proto["size"])
    return {
        "name": name,
        "fuente": FUENTE,
        "units": "kcal/mol",
        "cell": g["cell"],
        "ajuste": ("celda de la Tabla 1; los [uvw] de la Tabla 2 se pasan al marco del JSON con "
                   f"la matriz {g['marco']} (filas = vectores nuevos en la base de la Tabla 2)"),
        "posiciones": g["posiciones"],
        "centers": [{"label": lab, "frac": list(x)} for lab, x in g["centers"]],
        "symops": [{"W": W, "t": [float(v) for v in t]} for W, t in g["symops"]],
        "bonds_by_position": bonds_by_position(name),
        "tabla2_hollander": [{"enlace": lab, "moleculas": par, "uvw": list(uvw), "energia": phi}
                             for lab, par, uvw, phi in TABLA2[name]],
        "tabla3_hollander": [{"cara": c, "hkl_hollander": hkl_label(h), "hkl": hkl_label(hkl_json(name, h)),
                              "E_att": e, "redes_conexas": n} for c, h, e, n in TABLA3[name]],
        "celda_primitiva": g["primitiva"],
        "faces": faces,
        "protocolo": proto,
        "notes": ("Grafo de enlaces de Hollander et al. 2003 (CVFF, kcal/mol). Comprobación "
                  "frente a la Tabla 3 en data/monty2004/comprobacion_grafos.csv; generado "
                  "por scripts/transcribir_grasas.py"),
    }


def build_graph(name: str) -> BasisGraph:
    """Grafo de una grasa construido igual que lo hará `monty.load_graph_file`."""
    d = json_dict(name)
    return BasisGraph.from_centers(Cell(**d["cell"]), [c["frac"] for c in d["centers"]],
                                   d["bonds_by_position"], [(o["W"], o["t"]) for o in d["symops"]],
                                   units="kcal/mol", name=name, labels=[c["label"] for c in d["centers"]])


# ---------------------------------------------------------------------------
# Redes conexas (búsqueda topológica, como FACELIFT)
# ---------------------------------------------------------------------------
def to_primitive(g: BasisGraph, Pp) -> BasisGraph:
    """El mismo grafo en una celda primitiva (filas de Pp = vectores primitivos en la
    base de la celda de `g`)."""
    Pp = np.asarray(Pp, float)
    if np.allclose(Pp, np.eye(3)):
        return g
    Pi = np.linalg.inv(Pp)
    M = Pp @ g.cell.matrix
    L = np.linalg.norm(M, axis=1)
    G = M @ M.T
    ang = [math.degrees(math.acos(G[1, 2] / L[1] / L[2])), math.degrees(math.acos(G[0, 2] / L[0] / L[2])),
           math.degrees(math.acos(G[0, 1] / L[0] / L[1]))]
    xp = np.asarray(g.basis) @ Pi                       # coordenadas en la base primitiva
    base, lab, where = [], [], []
    for s, x in enumerate(xp):
        r = _red(x)
        for t, y in enumerate(base):
            if np.allclose(r, y, atol=1e-6):
                where.append((t, np.rint(x - r).astype(int)))
                break
        else:
            base.append(r)
            lab.append(g.basis_labels[s])
            where.append((len(base) - 1, np.rint(x - r).astype(int)))
    bonds, seen = [], set()
    for b in g.bonds:
        (i, ci), (j, cj) = where[b.i], where[b.j]
        tp = np.rint(np.asarray(b.shift, float) @ Pi).astype(int)
        nb = BasisBond(i, j, tuple(int(v) for v in tp + cj - ci), b.energy, b.label)
        if nb.canonical() not in seen:
            seen.add(nb.canonical())
            bonds.append(nb)
    return BasisGraph(Cell(*L, *ang), base, bonds, g.units, g.name, g.notes, lab)


def primitive_hkl(hkl: Sequence[int], Pp) -> Tuple[int, int, int]:
    """Índices en la celda primitiva: h' = Pp·h. Si no son enteros, (hkl) no es un plano
    de la red centrada y se toma 2h (misma orientación, capa propia)."""
    v = np.asarray(Pp, float) @ np.asarray(hkl, float)
    if not np.allclose(v, np.rint(v)):
        v = 2 * v
    return tuple(int(x) for x in np.rint(v))


def _plane_lattice(h) -> np.ndarray:
    """Base entera (2 × 3) del retículo {R : h·R = 0}."""
    return slice_cell(reduce_hkl(h), Cell(1, 1, 1, 90, 90, 90))[:2]


def _layer_ok(g: BasisGraph, h, members: List[int], k) -> bool:
    """¿Las subredes `members`, con los desplazamientos k, forman una capa conexa y
    periódica en dos direcciones (rango 2 e índice 1 en el plano)?"""
    h = np.asarray(h, int)
    adj = {s: [] for s in members}
    for b in g.bonds:
        if b.i in adj and b.j in adj and int(h @ np.array(b.shift)) + k[b.j] - k[b.i] == 0:
            t = np.array(b.shift)
            adj[b.i].append((b.j, t))
            adj[b.j].append((b.i, -t))
    pos = {members[0]: np.zeros(3, int)}
    stack, cyc = [members[0]], []
    while stack:
        i = stack.pop()
        for j, t in adj[i]:
            o = pos[i] + t
            if j not in pos:
                pos[j] = o
                stack.append(j)
            elif np.any(o != pos[j]):
                cyc.append(o - pos[j])
    if len(pos) != len(members) or not cyc:
        return False
    C = np.array(cyc, float)
    if np.linalg.matrix_rank(C, tol=1e-6) != 2:
        return False
    # índice del subretículo de los ciclos dentro del retículo del plano: 1 = una sola red
    L = _plane_lattice(h).astype(float)
    coef = np.rint(np.linalg.lstsq(L.T, C.T, rcond=None)[0].T).astype(int)
    gdet = 0
    for a in range(len(coef)):
        for b in range(a + 1, len(coef)):
            gdet = math.gcd(gdet, abs(int(coef[a, 0] * coef[b, 1] - coef[a, 1] * coef[b, 0])))
    return gdet == 1


def _assignments_tree(g: BasisGraph, h) -> np.ndarray:
    """Desplazamientos k (k_0 = 0) en los que cada subred entra por un enlace de la capa.

    Con n = mcd(h) = 1, una capa conexa contiene todas las subredes y tiene un árbol
    generador de enlaces interiores, así que su k está en este conjunto.
    """
    Z = g.n_basis
    inc = {s: [] for s in range(Z)}
    for b in g.bonds:
        ht = int(np.asarray(h) @ np.array(b.shift))
        inc[b.i].append((b.j, -ht))
        inc[b.j].append((b.i, ht))
    out = set()

    def rec(k):
        if len(k) == Z:
            out.add(tuple(k[s] for s in range(Z)))
            return
        cand = {}
        for i, ki in k.items():
            for j, d in inc[i]:
                if j not in k:
                    cand.setdefault(j, set()).add(ki + d)
        if not cand:
            return
        j = min(cand)
        for v in sorted(cand[j]):
            k[j] = v
            rec(k)
            del k[j]

    rec({0: 0})
    return np.array(sorted(out), dtype=int).reshape(-1, Z)


def _assignments_box(g: BasisGraph, K: int) -> np.ndarray:
    """Todos los k con k_0 = 0 y |k_s| <= K (para n > 1)."""
    Z = g.n_basis
    ks = np.array(list(itertools.product(range(-K, K + 1), repeat=Z - 1)), dtype=int).reshape(-1, Z - 1)
    return np.hstack([np.zeros((len(ks), 1), int), ks])


def strongest_net(g: BasisGraph, hkl: Sequence[int]) -> Tuple[float, Optional[Dict[str, int]]]:
    """E_slice por GU de la red conexa más fuerte de (hkl) y enlaces que rompe (o nan, None).

    `g` debe estar en una celda primitiva. Con n = mcd(hkl) > 1 la capa de grosor d_hkl
    tiene Z/n GU y hay n capas por periodo; cada una debe ser una red conexa.
    """
    h = np.asarray(hkl, int)
    n = math.gcd(math.gcd(abs(int(h[0])), abs(int(h[1]))), abs(int(h[2])))
    Z = g.n_basis
    ks = _assignments_tree(g, h) if n == 1 else _assignments_box(g, max(2, n))
    I = np.array([b.i for b in g.bonds])
    J = np.array([b.j for b in g.bonds])
    HT = np.array([int(h @ np.array(b.shift)) for b in g.bonds])
    PHI = np.array([b.energy for b in g.bonds])
    D = HT[None, :] + ks[:, J] - ks[:, I]             # 0 = enlace dentro de una capa
    es = (PHI[None, :] * (D == 0)).sum(1)
    for idx in np.argsort(-es, kind="stable"):
        k = ks[idx]
        classes: Dict[int, List[int]] = {}
        for s in range(Z):
            classes.setdefault(int(k[s]) % n, []).append(s)
        if n > 1 and (len(classes) != n or any(len(m) != Z // n for m in classes.values())):
            continue
        if all(_layer_ok(g, h, m, k) for m in classes.values()):
            rotos: Dict[str, int] = {}
            for b in np.where(D[idx] != 0)[0]:
                rotos[g.bonds[b].label] = rotos.get(g.bonds[b].label, 0) + 1
            return float(es[idx]) / Z, rotos
    return float("nan"), None


# ---------------------------------------------------------------------------
# Comprobaciones
# ---------------------------------------------------------------------------
def capa_propia(hkl: Sequence[int], Pp) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """(índices primitivos, índices en la celda del JSON) de la capa propia de (hkl).

    Es (hkl) si es un plano de la red centrada y 2·(hkl) si no; p. ej. (001) de la red I de
    β′-10.12.10 da (002), y (100) de la red C de β′-16.16.14 da (200)."""
    hp = primitive_hkl(hkl, Pp)
    hc = tuple(int(v) for v in np.rint(np.linalg.solve(np.asarray(Pp, float), np.asarray(hp, float))))
    return hp, hc


def check(name: str, g: BasisGraph) -> pd.DataFrame:
    """E_att (convención de la Tabla 3) de la red conexa más fuerte y del mejor corte
    plano, para las caras de la Tabla 3 y las de la figura de MONTY."""
    Pp = GRASAS[name]["primitiva"]
    gp = to_primitive(g, Pp)
    zp = gp.n_basis
    filas = []
    t3 = {hkl_json(name, h): (c, h, e, n) for c, h, e, n in TABLA3[name]}
    caras = [("tabla3", hk) for hk in t3] + [("fig_monty", parse_hkl(f)) for f in json_dict(name)["faces"]]
    for origen, hk in caras:
        hp, hc = capa_propia(hk, Pp)
        es, rotos = strongest_net(gp, hp)
        z0 = g.best_termination(hc)
        e_plano = zp * (g.lattice_energy - g.slice_energies(hc, z0)["E_slice"])
        e_red = zp * (g.lattice_energy - es) if np.isfinite(es) else np.nan
        c, h, e, n = t3.get(hk, ("", None, np.nan, np.nan)) if origen == "tabla3" else ("", None, np.nan, np.nan)
        filas.append({"grasa": name, "origen": origen, "hkl": hkl_label(hk),
                      "hkl_hollander": hkl_label(h) if h is not None else "", "cara_tabla3": c,
                      "hkl_primitiva": hkl_label(hp), "E_att_tabla3": e, "redes_tabla3": n,
                      "E_att_red_conexa": round(e_red, 2) if np.isfinite(e_red) else np.nan,
                      "E_att_corte_plano": round(e_plano, 2),
                      "dif_red_pct": round(100 * (e_red - e) / e, 1) if np.isfinite(e_red) and np.isfinite(e) else np.nan,
                      "enlaces_rotos_red": " ".join(f"{k}x{v}" for k, v in sorted(rotos.items())) if rotos else ""})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Morfología de energía de unión: fila superior de la Fig. 1 de MONTY
# ---------------------------------------------------------------------------
# Caras de la fila superior de la Fig. 1 de MONTY, con sus índices (los del JSON). Según el
# pie de la figura, las calculó el módulo de energía de unión de Cerius² (cortes planos,
# con el campo de fuerzas completo), no el análisis de redes conexas de la ref. 17.
FIG1_MONTY = {
    "beta_16_16_16": ["(001)", "(-213)", "(-111)", "(20-3)", "(30-5)"],
    "betap_10_12_10": ["(002)", "(101)", "(112)"],
    "betap_16_16_14": ["(001)", "(110)", "(60-2)"],
}


def equivalentes(name: str, hkl: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Orientaciones equivalentes a (hkl) por el grupo puntual de la grasa, más las
    opuestas (con enlaces de par, (hkl) y (−h−k−l) rompen los mismos enlaces).
    Con x' = W·x + t, el plano h·x = c pasa a (h·W⁻¹)·x' = c'."""
    out = set()
    for W, _ in GRASAS[name]["symops"]:
        Wi = np.rint(np.linalg.inv(np.asarray(W, float))).astype(int)
        v = tuple(int(x) for x in np.asarray(hkl, int) @ Wi)
        out.update({v, tuple(-x for x in v)})
    return sorted(out)


def forma(name: str, hkl: Sequence[int]) -> str:
    """Etiqueta de la forma {hkl} (el representante mayor en orden lexicográfico)."""
    return "{" + hkl_label(max(equivalentes(name, reduce_hkl(hkl))))[1:-1] + "}"


def caras_candidatas(hmax: int = 3, kmax: int = 3, lmax: int = 5) -> List[Tuple[int, int, int]]:
    """Orientaciones (hkl) con |h| <= hmax, |k| <= kmax, |l| <= lmax, índices coprimos y una
    sola de cada par ±(hkl) (el primer índice no nulo, positivo)."""
    out = []
    for v in itertools.product(range(-hmax, hmax + 1), range(-kmax, kmax + 1), range(-lmax, lmax + 1)):
        if v == (0, 0, 0) or math.gcd(math.gcd(abs(v[0]), abs(v[1])), abs(v[2])) != 1:
            continue
        if next(x for x in v if x) > 0:
            out.append(v)
    return out


def energias_corte_plano(name: str, caras: Sequence[Sequence[int]]) -> pd.DataFrame:
    """E_att del mejor corte plano de la capa propia de cada cara (convención de la Tabla 3:
    Z_prim·(E_cr − E_slice)). Es lo que calcula un módulo de energía de unión con capas
    planas, como el de Cerius² que usó la Fig. 1, pero solo con los enlaces de la Tabla 2."""
    g = build_graph(name)
    Pp = GRASAS[name]["primitiva"]
    zp = to_primitive(g, Pp).n_basis
    filas = []
    for hk in caras:
        _, hc = capa_propia(hk, Pp)
        z0 = g.best_termination(hc)
        e = zp * (g.lattice_energy - g.slice_energies(hc, z0)["E_slice"])
        filas.append({"hkl": hkl_label(tuple(hk)), "capa": hkl_label(hc), "E_att": float(e)})
    return pd.DataFrame(filas)


def morfologia_energia_union(cell: Cell, energias: Dict[str, float]):
    """Poliedro de energía de unión (R ∝ E_att) y fracción de área de cada familia ±(hkl).

    `energias` va de la etiqueta (hkl) a E_att > 0. Si varias capas dan la misma
    orientación (p. ej. (001) y (002)), cuenta la de menor E_att. Antes de construir el
    poliedro se descartan las caras que no pueden tocarlo: una cara (n, d) solo corta el
    poliedro de las caras de índices <= 1 si d < max_v v·n sobre sus vértices, y añadir
    caras solo lo encoge. Devuelve (caras del poliedro, tabla) como `analysis.wulff_faces`.
    """
    from kmc3d import analysis, seeds

    def canon(r):
        return r if next(x for x in r if x) > 0 else tuple(-x for x in r)

    rates: Dict[Tuple[int, int, int], float] = {}
    for f, e in energias.items():
        if np.isfinite(e) and e > 0:
            c = canon(reduce_hkl(parse_hkl(f)))
            rates[c] = min(float(e), rates.get(c, np.inf))
    ambas = {**rates, **{tuple(-x for x in f): e for f, e in rates.items()}}
    base = {f: e for f, e in ambas.items() if max(abs(x) for x in f) <= 1}
    try:
        V = seeds.polyhedron_vertices(base, cell, centrosymmetric=False)
    except ValueError:          # con pocas caras de índices bajos el poliedro no se cierra
        V = seeds.polyhedron_vertices(ambas, cell, centrosymmetric=False)
    utiles = {f: e for f, e in rates.items() if e <= (V @ cell.normal(f)).max() * (1 + 1e-9)}
    caras, tabla = analysis.wulff_faces(utiles, cell, centrosymmetric=True)
    tabla["E_att"] = [utiles[parse_hkl(f)] for f in tabla["familia"]]
    return caras, tabla


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-comprobar", action="store_true", help="no escribe archivos")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    tablas = []
    for name in GRASAS:
        d = json_dict(name)
        g = build_graph(name)
        print(f"\n=== {name}: {g}  (E_cr = {g.lattice_energy:.2f} kcal/mol por GU)")
        # un representante de cada enlace de la Tabla 2 (el resto son imágenes por simetría)
        bt = g.bond_table()
        bt["longitud"] = bt["longitud"].round(2)
        print(bt.drop_duplicates(subset=["enlace"]).to_string(index=False))
        df = check(name, g)
        tablas.append(df)
        print(df.drop(columns=["grasa"]).to_string(index=False))
        if not args.solo_comprobar:
            path = OUT_DIR / f"{name}.json"
            path.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
            # el JSON escrito debe dar el mismo grafo
            g2, _ = monty.load_graph_file(path)
            assert g2.n_basis == g.n_basis and len(g2.bonds) == len(g.bonds)
            assert abs(g2.lattice_energy - g.lattice_energy) < 1e-9
            print(f"-> {path.relative_to(ROOT)}")
    if not args.solo_comprobar:
        out = OUT_DIR / "comprobacion_grafos.csv"
        pd.concat(tablas, ignore_index=True).to_csv(out, index=False)
        print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
