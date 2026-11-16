"""Experimentos de difusión superficial del notebook 02, como funciones importables.

Cada función recibe un diccionario de parámetros y devuelve un diccionario de
resultados, para poder lanzarlas en paralelo con `parallel.run_parallel`, que necesita
funciones de nivel de módulo:

    run_parallel("kmc3d.experiments:kossel_face_growth", [{"k_d": 0}, {"k_d": 3}])

- `kossel_face_growth`: cubo de Kossel que crece con o sin difusión. Mide la velocidad
  media de las caras {100}, la rugosidad de las caras y el reparto de eventos.
- `relax_rod`: varilla que solo difunde (sin unión ni despegue). Mide cómo se compacta.
"""

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import analysis as an
from . import presets, seeds
from .engine import KMC3D
from .graph import CrystalGraph
from .rates import SplitRates, SurfaceDiffusion

CUBE_FACES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def bonds_per_unit(coords: np.ndarray, graph: CrystalGraph) -> float:
    """Enlaces del grafo por unidad (cada enlace cuenta una vez): mide lo compacto que es."""
    X = np.asarray(coords, dtype=np.int64)
    if len(X) == 0:
        return 0.0
    lo = X.min(axis=0) - graph.reach
    span = X.max(axis=0) + graph.reach - lo + 1
    # código entero único por sitio para buscar vecinos con np.isin
    def enc(Y):
        Y = Y - lo
        return (Y[:, 0] * span[1] + Y[:, 1]) * span[2] + Y[:, 2]
    keys = enc(X)
    n = sum(int(np.isin(enc(X + np.asarray(b.vector)), keys).sum()) for b in graph.bonds)
    return n / len(X)


def kossel_face_growth(params: Dict[str, object]) -> Dict[str, object]:
    """Cubo de Kossel que crece; velocidad y rugosidad de las caras {100}.

    `params` (todas opcionales):
        phi (2.0), delta_mu (0.5), rates ("gilmer"), k_d (0 = sin difusión),
        hop_vectors ("bonds"), L (16: lado del cubo semilla), growth (2.0: N final / N0),
        n_snaps (20), rng_seed (1)
    La velocidad R (capas / unidad de tiempo) es la pendiente del plano medio de cada
    cara en la mitad final (`analysis.face_mean_series` + `face_growth_rates`), promediada
    sobre las 6 caras. Los snapshots se toman a N equiespaciados, no por número de
    eventos, porque con difusión la mayoría de los eventos son saltos.
    """
    t0 = time.perf_counter()
    phi = float(params.get("phi", 2.0))
    dmu = float(params.get("delta_mu", 0.5))
    g = presets.kossel(phi)
    rates = getattr(SplitRates, params.get("rates", "gilmer"))(g, dmu)
    kd = float(params.get("k_d", 0.0))
    dif = SurfaceDiffusion(g, kd, params.get("hop_vectors", "bonds")) if kd > 0 else None
    L = int(params.get("L", 16))
    kmc = KMC3D(g, rates, seeds.cube(L), rng_seed=int(params.get("rng_seed", 1)), diffusion=dif)
    n0 = kmc.n_occ
    n_fin = int(round(float(params.get("growth", 2.0)) * n0))
    objetivos = np.linspace(n0, n_fin, int(params.get("n_snaps", 20)) + 1)[1:]
    snaps = [(kmc.t, kmc.occupied_coords())]
    parada = "n_max"
    for nt in objetivos:
        _, st = kmc.run(n_max=int(nt), max_events=int(params.get("max_events", 20_000_000)),
                        final_snapshot=False)
        snaps.append((kmc.t, kmc.occupied_coords()))
        if st["stop_reason"] != "n_max":
            parada = st["stop_reason"]
            break
    R = an.face_growth_rates(an.face_mean_series(snaps, g, CUBE_FACES))
    # rugosidad: desviación típica de las alturas del centro de cada cara (final)
    W = []
    for f in CUBE_FACES:
        for s in (1, -1):
            try:
                W.append(float(np.std(an.face_height_map(snaps[-1][1], tuple(s * x for x in f)))))
            except ValueError:
                pass
    k_plus = float(rates._attach(0))   # k+ de Gilmer: igual en todos los sitios
    n_ev = max(kmc.n_events, 1)
    return {**params, "R": float(R["R"].mean()), "R_sd": float(R["R"].std()),
            "R/k+": float(R["R"].mean()) / k_plus, "W": float(np.mean(W)) if W else np.nan,
            "N0": n0, "N": kmc.n_occ, "t": kmc.t, "eventos": kmc.n_events,
            "frac_saltos": kmc.n_hop / n_ev, "frac_rechazos": kmc.n_hop_rejected / n_ev,
            "uniones": kmc.n_attach, "despegues": kmc.n_detach, "parada": parada,
            "segundos": time.perf_counter() - t0,
            "coords": snaps[-1][1].astype(np.int32)}


def relax_rod(params: Dict[str, object]) -> Dict[str, object]:
    """Varilla de Kossel que solo difunde: relajación de la forma a N constante.

    `params` (todas opcionales):
        phi (2.5), k_d (1.0), hop_vectors ("cube18"), dims ((4, 4, 24)),
        n_events (2 000 000), n_checkpoints (25), rng_seed (1)
    Sin unión ni despegue (k0 = 1e-12). En cada punto de control mide, sobre el fragmento
    conexo mayor: L1/L2 (`shape_descriptors`), enlaces por unidad y número de fragmentos.
    """
    t0 = time.perf_counter()
    phi = float(params.get("phi", 2.5))
    g = presets.kossel(phi)
    dims = tuple(params.get("dims", (4, 4, 24)))
    rod = seeds.box(*dims)
    kmc = KMC3D(g, SplitRates.gilmer(g, 0.0, k0=1e-12), rod, rng_seed=int(params.get("rng_seed", 1)),
                diffusion=SurfaceDiffusion(g, float(params.get("k_d", 1.0)),
                                           params.get("hop_vectors", "cube18")))
    n_ev = int(params.get("n_events", 2_000_000))
    n_cp = int(params.get("n_checkpoints", 25))
    filas = []
    snaps = []

    def medir():
        c = kmc.occupied_coords()
        lab = an.cluster_labels(c, g)
        big = c[lab == 0]
        filas.append({"eventos": kmc.n_events, "t": kmc.t, "saltos": kmc.n_hop,
                      "L1/L2": an.shape_descriptors(big, g.cell)["L1/L2"],
                      "enlaces_por_unidad": bonds_per_unit(big, g),
                      "fragmentos": int(lab.max()) + 1, "frac_mayor": len(big) / len(c)})
        return c

    snaps.append(medir())
    for _ in range(n_cp):
        kmc.run(max_events=n_ev // n_cp, final_snapshot=False)
        c = medir()
    snaps.append(c)
    return {**params, "serie": pd.DataFrame(filas), "inicial": snaps[0].astype(np.int32),
            "final": snaps[-1].astype(np.int32), "N": kmc.n_occ,
            "frac_rechazos": kmc.n_hop_rejected / max(kmc.n_events, 1),
            "segundos": time.perf_counter() - t0}
