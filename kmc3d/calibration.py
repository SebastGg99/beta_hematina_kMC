"""Calibración de las energías de β-hematina contra la morfología observada (Fase 3).

Es el punto 1(b) + 3 de la Fase 3 de PLAN_3DkMC.md: ajustar las energías de enlace
EFECTIVAS a la morfología observada USANDO EL kMC (no solo E_att). No hay velocidades
experimentales en este ajuste (punto 2 de la Fase 3, pendiente): solo forma.

Morfología observada (Solomonov et al. 2007; PLAN §7.2) y cómo se mide aquí, siempre
sobre el cristal kMC final (fragmento conexo mayor):

1. "aguja": alargada según c. Dimensiones L = √(12 var) a lo largo de c, de la anchura
   w (en el plano (100), ⟂ c) y del espesor (normal a (100)); eje largo (tensor de
   inercia) a menos de `ANGULO_MAX` grados de c y L_c / W >= `ALARGAMIENTO_MIN`.
2. "anchura/espesor ≈ 3:1" de las caras laterales: W / T entre `WT_RANGO`.
3. "laterales {100} y {010}": de las caras de la zona [001] ((100), (010), (110),
   (1-10)), las {100} + {010} suman al menos `LATERALES_MIN` del área, y {100} es la
   cara más grande del cristal.
4. "extremos {011} y, en menor medida, {001}": área {011} > área {001} > 0, y las dos
   juntas son al menos `EXTREMOS_MIN` del área de las caras que cortan c ((001), (011),
   (01-1), (101), (10-1)): los extremos no los forman otras caras.
5. "sin {0-11}": fracción de área de {01-1} (= {0-11}) menor que `AUSENTE_MAX`.
6. forma estacionaria (condición de validez, no un dato observado): W/T y L_c/W cambian
   menos de un `DERIVA_MAX` (en logaritmo) entre N/2 y N. Si una cara casi no crece,
   la forma la fija la semilla y no sirve para comparar.

Las áreas salen de `analysis.facet_areas` (cada unidad de superficie se asigna a la
cara más próxima a su normal suavizada). Los umbrales son elecciones de este módulo,
no datos experimentales: el plan solo da "aguja", "≈ 3:1", "{011} > {001}" y "sin
{0-11}". Las energías que salen de aquí siguen siendo EFECTIVAS y PROVISIONALES: no
vienen de ningún cálculo de energías de par (decisión D3 del plan).

Parametrización de las energías (`energies_from_params`):
    φ_t = escala · φ_t^base,   con φ^base = provisionales y algunos enlaces cambiados
    (`overrides`, p. ej. {"[001]": 3.5}).

`run_candidates` corre varios candidatos en paralelo con `parallel.run_parallel` (un
proceso de Python por corrida, lanzado con `subprocess`).
"""

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import analysis as an
from . import presets, seeds
from .engine import KMC3D
from .geometry import Cell
from .graph import CrystalGraph
from .parallel import run_parallel
from .rates import SplitRates, SurfaceDiffusion

# Caras para las métricas; el orden fija también los colores de render.face_colors
FACES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, -1), (1, 1, 0), (1, -1, 0),
         (1, 0, 1), (1, 0, -1))
ZONA_001 = ("(100)", "(010)", "(110)", "(1-10)")

# Umbrales de los criterios (elecciones de este módulo; ver docstring)
ANGULO_MAX = 15.0          # grados entre el eje largo y c
ALARGAMIENTO_MIN = 1.5     # L_c / W
WT_RANGO = (2.0, 4.5)      # W / T alrededor de 3
WT_OBJETIVO = 3.0
LATERALES_MIN = 0.75       # ({100} + {010}) / zona [001]
EXTREMOS_MIN = 0.5         # ({011} + {001}) / caras que cortan c
AUSENTE_MAX = 0.03         # fracción de área de {0-11}
DERIVA_MAX = 0.15          # |Δ ln| de W/T y L_c/W entre N/2 y N

ZONA_EXTREMOS = ("(001)", "(011)", "(01-1)", "(101)", "(10-1)")
CRITERIOS = ("aguja_c", "anchura_espesor", "laterales", "extremos", "sin_0-11", "estacionaria")


def energies_from_params(scale: float = 1.0, overrides: Optional[Dict[str, float]] = None
                         ) -> Dict[str, float]:
    """Energías φ/kT de los 7 enlaces: escala · (provisionales con `overrides`)."""
    e = dict(presets.BETA_HEMATIN_ENERGIES_PROVISIONAL)
    e.update(overrides or {})
    return {k: round(scale * v, 6) for k, v in e.items()}


def crystal_axes(cell: Cell) -> Dict[str, np.ndarray]:
    """Direcciones cartesianas unitarias: c, anchura w (en (100), ⟂ c) y normal de (100)."""
    c_hat = cell.matrix[2] / np.linalg.norm(cell.matrix[2])
    n100 = cell.normal((1, 0, 0))                 # a* es ⟂ a b y c
    w_hat = np.cross(c_hat, n100)
    return {"c": c_hat, "w": w_hat / np.linalg.norm(w_hat), "n100": n100}


def crystal_dimensions(coords: np.ndarray, cell: Cell) -> np.ndarray:
    """(L_c, W, T): dimensiones a lo largo de c, de la anchura w y del espesor (⟂ (100))."""
    ejes = crystal_axes(cell)
    return an.directional_extents(cell.to_cartesian(coords), [ejes["c"], ejes["w"], ejes["n100"]])


def morphology_metrics(coords: np.ndarray, graph: CrystalGraph,
                       faces: Sequence[Sequence[int]] = FACES) -> Dict[str, float]:
    """Dimensiones, eje largo y fracción de área de cada familia (fragmento mayor)."""
    cell = graph.cell
    big = an.largest_cluster(coords, graph)
    X = cell.to_cartesian(big)
    ejes = crystal_axes(cell)
    L_c, W, T = crystal_dimensions(big, cell)
    Xc = X - X.mean(axis=0)
    lam, vec = np.linalg.eigh(np.cov(Xc.T))
    largo = vec[:, np.argmax(lam)]
    ang = float(np.degrees(np.arccos(min(1.0, abs(largo @ ejes["c"])))))
    lab = an.facet_labels(big, cell, faces)
    areas = an.facet_areas(big, cell, faces, labels=lab).set_index("familia")["fraccion"]
    out = {"N": int(len(coords)), "N_mayor": int(len(big)), "L_c": L_c, "W": W, "T": T,
           "L_c/W": L_c / W, "W/T": W / T, "angulo_c": ang,
           "sin_cara": float(np.mean(lab["family"] < 0))}
    out.update({f"A{k}": float(v) for k, v in areas.items()})
    return out


def evaluate_criteria(m: Dict[str, float]) -> Dict[str, object]:
    """Cumplimiento de cada criterio (bool) y una pérdida continua J (0 = todo cumplido).

    J suma los cuadrados de las violaciones relativas de cada criterio, más el error
    logarítmico de W/T respecto de 3. Sirve para ordenar candidatos que cumplen el
    mismo número de criterios.
    """
    a = {f: m.get(f"A{f}", 0.0) for f in ("(100)", "(010)", "(001)", "(011)", "(01-1)",
                                         "(110)", "(1-10)", "(101)", "(10-1)")}
    zona = sum(a[f] for f in ZONA_001)
    lat = (a["(100)"] + a["(010)"]) / zona if zona > 0 else 0.0
    zext = sum(a[f] for f in ZONA_EXTREMOS)
    ext = (a["(011)"] + a["(001)"]) / zext if zext > 0 else 0.0
    deriva = max(abs(m.get("deriva_W/T", 0.0)), abs(m.get("deriva_L_c/W", 0.0)))
    ok = {
        "aguja_c": m["angulo_c"] <= ANGULO_MAX and m["L_c/W"] >= ALARGAMIENTO_MIN,
        "anchura_espesor": WT_RANGO[0] <= m["W/T"] <= WT_RANGO[1],
        "laterales": lat >= LATERALES_MIN and a["(100)"] >= max(a.values()),
        "extremos": a["(011)"] > a["(001)"] > 0.0 and ext >= EXTREMOS_MIN,
        "sin_0-11": a["(01-1)"] < AUSENTE_MAX,
        "estacionaria": deriva < DERIVA_MAX,
    }
    v = [max(0.0, np.log(ALARGAMIENTO_MIN / max(m["L_c/W"], 1e-9))),
         max(0.0, (m["angulo_c"] - ANGULO_MAX) / ANGULO_MAX),
         abs(np.log(m["W/T"] / WT_OBJETIVO)),
         max(0.0, LATERALES_MIN - lat) / LATERALES_MIN,
         max(0.0, a["(001)"] - a["(011)"]) / max(a["(001)"] + a["(011)"], 1e-9),
         max(0.0, EXTREMOS_MIN - ext) / EXTREMOS_MIN,
         max(0.0, a["(01-1)"] - AUSENTE_MAX) / AUSENTE_MAX * 0.2,
         max(0.0, deriva - DERIVA_MAX) / DERIVA_MAX]
    out = {k: bool(x) for k, x in ok.items()}
    out["laterales_frac"] = lat
    out["extremos_frac"] = ext
    out["n_criterios"] = int(sum(ok.values()))
    out["J"] = float(sum(x * x for x in v))
    return out


def run_candidate(params: Dict[str, object]) -> Dict[str, object]:
    """Corre un candidato y devuelve métricas, criterios, velocidades y el cristal final.

    `params` (todas las claves son opcionales salvo `nombre`):
        nombre, scale (1.0), overrides ({}), delta_mu (1.0), rates ("symmetric"),
        k_d (0 = sin difusión), hop_vectors ("bonds"), rng_seed (1),
        seed_radius (70 Å), n_max (40 000), max_events (3 000 000),
        snapshot_every (10 000)
    """
    t0 = time.perf_counter()
    p = dict(params)
    energies = energies_from_params(p.get("scale", 1.0), p.get("overrides"))
    g = presets.beta_hematin(energies)
    dmu = float(p.get("delta_mu", 1.0))
    rates = getattr(SplitRates, p.get("rates", "symmetric"))(g, dmu)
    kd = float(p.get("k_d", 0.0))
    dif = SurfaceDiffusion(g, kd, p.get("hop_vectors", "bonds")) if kd > 0 else None
    seed = seeds.sphere(float(p.get("seed_radius", 70.0)), g.cell)
    kmc = KMC3D(g, rates, seed, rng_seed=int(p.get("rng_seed", 1)), diffusion=dif,
                track_faces=presets.BETA_HEMATIN_FACES)
    snaps, st = kmc.run(n_max=int(p.get("n_max", 40_000)),
                        max_events=int(p.get("max_events", 3_000_000)),
                        record_every=5_000, snapshot_every=int(p.get("snapshot_every", 10_000)))
    final = snaps[-1][1]
    m = morphology_metrics(final, g)
    # deriva de la forma entre N/2 y N (estacionariedad)
    k_mid = int(np.argmin([abs(len(c) - len(final) / 2) for _, c in snaps]))
    Lc2, W2, T2 = crystal_dimensions(an.largest_cluster(snaps[k_mid][1], g), g.cell)
    m["deriva_W/T"] = float(np.log(m["W/T"] / (W2 / T2)))
    m["deriva_L_c/W"] = float(np.log(m["L_c/W"] / (Lc2 / W2)))
    # espesor inicial (semilla) para detectar formas no estacionarias
    T0 = an.directional_extents(g.cell.to_cartesian(seed), [crystal_axes(g.cell)["n100"]])[0]
    m["T/T_semilla"] = m["T"] / T0
    R = an.face_growth_rates(an.face_mean_series(snaps, g, presets.BETA_HEMATIN_FACES))
    fam = R.groupby("familia", sort=False)["R_familia"].first()
    out = {**{k: v for k, v in p.items() if k != "overrides"},
           "overrides": dict(p.get("overrides") or {}), "energias": energies,
           "eventos": kmc.n_events, "saltos": kmc.n_hop, "t": kmc.t,
           "parada": st["stop_reason"], "segundos": time.perf_counter() - t0,
           **m, **evaluate_criteria(m),
           **{f"R{k}": float(v) for k, v in fam.items()},
           "coords": final.astype(np.int32)}
    return out


def run_candidates(candidates: Sequence[Dict[str, object]], processes: Optional[int] = None,
                   verbose: bool = True) -> List[Dict[str, object]]:
    """Corre `run_candidate` para cada candidato en paralelo (`parallel.run_parallel`).

    Devuelve los resultados en el mismo orden que `candidates`. Con processes=1 corre en
    serie en este proceso (útil para depurar).
    """
    rep = (lambda i, r: _report(r)) if verbose else None
    return run_parallel("kmc3d.calibration:run_candidate", candidates, processes, report=rep)


def _report(r: Dict[str, object]) -> None:
    """Una línea de progreso por candidato terminado."""
    print(f"{r['nombre']:<30} {r['n_criterios']}/{len(CRITERIOS)}  J = {r['J']:.3f}  N = {r['N']}  "
          f"{r['segundos']:.0f} s", flush=True)


def results_frame(results: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Resultados -> tabla (sin las coordenadas), ordenada por criterios y J."""
    filas = [{k: v for k, v in r.items() if k not in ("coords", "energias")} for r in results]
    df = pd.DataFrame(filas)
    return df.sort_values(["n_criterios", "J"], ascending=[False, True], ignore_index=True)
