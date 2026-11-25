"""Validación con MONTY (Boerrigter et al. 2004): curvas de fracción de adhesión S(Δμ).

Funciones importables para lanzar con `parallel.run_parallel` y para analizar y dibujar
los resultados, como en las Figs. 2–4 del artículo:

    from kmc3d.parallel import run_parallel
    from kmc3d import monty
    tareas = monty.sticking_grid({"graph": "kossel", "phi": 4.0}, [(0, 0, 1), (1, 1, 1)],
                                 delta_mu=[0.5, 1, 2, 4], seeds=[1, 2])
    df = monty.results_frame(run_parallel("kmc3d.monty:sticking_point", tareas))
    monty.onset_table(df)

Grafos (`params["graph"]`):
- "kossel" (con `phi` en kT), "beta_hematin" (energías provisionales de
  `presets.beta_hematin`, en kT: solo como grafo de prueba);
- la ruta de un JSON de grafo (`load_graph_file`), con energías en kcal/mol y la
  temperatura `T` (K) para pasarlas a kT;
- un dict de `BasisGraph.to_dict()`.
Opcional: `supercell` = (n1, n2, n3) describe el mismo cristal con una base mayor.

Fuerza impulsora: `delta_mu` en kT, o `delta_mu_kcal` en kcal/mol con `T` (como en los
ejes de MONTY). En el resultado van las dos.

Protocolo de MONTY (§3.1): `n_equil` eventos de equilibrado, `reset_counters()` y
`n_sample` eventos de muestreo, en los que se cuentan N_growth y N_etch. Las reglas
STS son las de MONTY (`allow_enclosed=True`) salvo que la tarea diga otra cosa.
"""

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import presets
from .basis_graph import BasisGraph, R_KCAL, kcal_to_kT
from .geometry import Cell, hkl_label, parse_hkl
from .slab import MontyRates, SlabKMC

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "monty2004"

# Protocolos de simulación de MONTY para las grasas (§3.1 del artículo)
MONTY_PROTOCOLS = {
    "beta": {"size": (40, 40), "n_equil": 100_000, "n_sample": 100_000, "T": 313.0,
             "lambda1": 0.5, "lambda2": 0.5},
    "beta_prime": {"size": (50, 50), "n_equil": 200_000, "n_sample": 1_000_000, "T": 300.0,
                   "lambda1": 0.5, "lambda2": 0.5},
}


# ---------------------------------------------------------------------------
# Grafos
# ---------------------------------------------------------------------------
def load_graph_file(path: Union[str, Path]) -> Tuple[BasisGraph, dict]:
    """Lee un grafo de un JSON y devuelve (grafo en sus unidades, metadatos).

    Admite dos formatos:
    - el de `BasisGraph.to_dict` ("basis" + "bonds" con i, j, shift, energy);
    - el de transcripción ("centers" + "bonds_by_position" + "symops" opcional), que
      se pasa a `BasisGraph.from_centers`: cada enlace da la molécula independiente de
      origen ("from") y la posición fraccionaria del centro vecino ("to_frac").
    Los metadatos son el resto de campos del JSON (fuente, T, caras, protocolo...).
    """
    d = json.loads(Path(path).read_text())
    if "bonds_by_position" in d:
        ops = d.get("symops")
        g = BasisGraph.from_centers(Cell(**d["cell"]), [c["frac"] for c in d["centers"]],
                                    d["bonds_by_position"],
                                    [(o["W"], o["t"]) for o in ops] if ops else None,
                                    units=d.get("units", "kcal/mol"), name=d.get("name", ""),
                                    notes=d.get("notes", ""),
                                    labels=[c.get("label", f"M{k + 1}") for k, c in enumerate(d["centers"])])
        skip = ("cell", "centers", "bonds_by_position", "symops", "units", "name", "notes")
    else:
        g = BasisGraph.from_dict(d)
        skip = ("cell", "basis", "bonds", "units", "name", "notes")
    meta = {k: v for k, v in d.items() if k not in skip}
    return g, meta


def graph_from_params(params: Dict[str, object]) -> BasisGraph:
    """Grafo en kT a partir de los parámetros de una tarea (ver docstring del módulo)."""
    spec = params.get("graph", "kossel")
    if isinstance(spec, dict):
        g = BasisGraph.from_dict(spec)
    elif spec == "kossel":
        g = BasisGraph.from_crystal_graph(presets.kossel(float(params.get("phi", 2.0))))
    elif spec == "beta_hematin":
        g = BasisGraph.from_crystal_graph(presets.beta_hematin(params.get("energies")))
    else:
        g, _ = load_graph_file(spec)
    if g.units != "kT":
        if "T" not in params:
            raise ValueError(f"El grafo está en {g.units}: hace falta la temperatura T (K)")
        g = g.to_kT(float(params["T"]))
    sc = params.get("supercell")
    if sc:
        g = g.supercell(*[int(x) for x in sc])
    return g


def _as_hkl(f: Union[str, Sequence[int]]) -> Tuple[int, int, int]:
    """'(1-11)' o (1, -1, 1) -> (1, -1, 1)."""
    return parse_hkl(f) if isinstance(f, str) else tuple(int(x) for x in f)


# ---------------------------------------------------------------------------
# Un punto de la curva S(Δμ)
# ---------------------------------------------------------------------------
def delta_mu_kT(params: Dict[str, object]) -> float:
    """Δμ/kT de una tarea: `delta_mu` (kT) o `delta_mu_kcal` (kcal/mol) con `T`."""
    if "delta_mu_kcal" in params:
        return kcal_to_kT(float(params["delta_mu_kcal"]), float(params["T"]))
    return float(params.get("delta_mu", 0.0))


def sticking_point(params: Dict[str, object]) -> Dict[str, object]:
    """Equilibrado + muestreo de una superficie (hkl): S, R y rugosidad.

    `params` (además del grafo y Δμ, ver docstring del módulo):
        hkl: cara, como tupla o etiqueta '(1-11)'
        size ((40, 40)), n_equil (100 000), n_sample (100 000), rng_seed (1),
        lambda1 (0.5), lambda2 (0.5), time_mode ("bkl"), depth (None),
        allow_enclosed (True: reglas STS de MONTY, un hueco cerrado puede llenarse;
        False: no se llena, como en `KMC3D`; ver notebook 03, §2.3),
        keep_map (True: guarda el mapa de alturas final)
    Devuelve los parámetros más S, R (ec. 20, en unidades de la celda por 1/k0),
    N_growth, N_etch, capas crecidas en el muestreo, W (rugosidad final), segundos y
    eventos por segundo.
    """
    t0 = time.perf_counter()
    g = graph_from_params(params)
    hkl = _as_hkl(params["hkl"])
    dmu = delta_mu_kT(params)
    rates = MontyRates(dmu, float(params.get("lambda1", 0.5)), float(params.get("lambda2", 0.5)))
    size = tuple(int(x) for x in params.get("size", (40, 40)))
    kmc = SlabKMC(g, hkl, rates, size=size, rng_seed=int(params.get("rng_seed", 1)),
                  depth=params.get("depth"), time_mode=str(params.get("time_mode", "bkl")),
                  allow_enclosed=bool(params.get("allow_enclosed", True)))
    kmc.run(max_events=int(params.get("n_equil", 100_000)))
    kmc.reset_counters()
    h0 = kmc.mean_height()
    st = kmc.run(max_events=int(params.get("n_sample", 100_000)))
    seg = time.perf_counter() - t0
    out = {**{k: v for k, v in params.items() if k != "graph" or isinstance(v, str)},
           "hkl": hkl_label(hkl), "delta_mu": dmu,
           "delta_mu_kcal": dmu * R_KCAL * float(params["T"]) if "T" in params else np.nan,
           "S": st["S"], "R": st["R"], "n_growth": st["n_growth"], "n_etch": st["n_etch"],
           "capas": kmc.mean_height() - h0, "W": kmc.roughness(), "H": kmc.H,
           "Z": g.n_basis, "E_cr": g.lattice_energy, "segundos": seg,
           "eventos_por_s": kmc.n_events / seg if seg > 0 else np.nan}
    if params.get("keep_map", True):
        out["mapa"] = kmc.height_map().astype(np.float32)
    return out


def sticking_grid(base: Dict[str, object], faces: Iterable[Union[str, Sequence[int]]],
                  delta_mu: Optional[Sequence[float]] = None,
                  delta_mu_kcal: Optional[Sequence[float]] = None,
                  seeds: Sequence[int] = (1,)) -> List[Dict[str, object]]:
    """Lista de tareas (cara × Δμ × semilla) con los parámetros comunes `base`.

    Se da `delta_mu` (kT) o `delta_mu_kcal` (kcal/mol, necesita `T` en `base`).
    """
    if (delta_mu is None) == (delta_mu_kcal is None):
        raise ValueError("Da delta_mu o delta_mu_kcal (uno de los dos)")
    key, vals = ("delta_mu", delta_mu) if delta_mu is not None else ("delta_mu_kcal", delta_mu_kcal)
    tasks = []
    for f in faces:
        for v in vals:
            for s in seeds:
                tasks.append({**base, "hkl": hkl_label(_as_hkl(f)), key: float(v), "rng_seed": int(s)})
    return tasks


def results_frame(results: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Resultados de `sticking_point` como DataFrame (sin los mapas de alturas)."""
    return pd.DataFrame([{k: v for k, v in r.items() if k != "mapa"} for r in results])


# ---------------------------------------------------------------------------
# Análisis de las curvas
# ---------------------------------------------------------------------------
def mean_curves(df: pd.DataFrame, x: str = "delta_mu") -> pd.DataFrame:
    """S medio, su error típico y el número de semillas por (cara, Δμ)."""
    g = df.groupby(["hkl", x])["S"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"].fillna(0.0) / np.sqrt(out["count"])
    return out.rename(columns={"mean": "S"})


def crossing(xs: Sequence[float], ys: Sequence[float], level: float) -> float:
    """Primer x donde la curva (x creciente) llega a `level`, por interpolación lineal.

    NaN si no llega dentro del intervalo; si ya empieza por encima, el primer x.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if not len(xs):
        return float("nan")
    if ys[0] >= level:
        return float(xs[0])
    for k in range(1, len(xs)):
        if ys[k] >= level:
            x0, x1, y0, y1 = xs[k - 1], xs[k], ys[k - 1], ys[k]
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def onset_table(df: pd.DataFrame, x: str = "delta_mu", levels: Sequence[float] = (0.1, 0.5, 0.9)
                ) -> pd.DataFrame:
    """Δμ en que la curva media de cada cara alcanza S = 0.1, 0.5 y 0.9."""
    mc = mean_curves(df, x)
    rows = []
    for hkl, sub in mc.groupby("hkl", sort=False):
        row = {"hkl": hkl}
        for lv in levels:
            row[f"{x}@S={lv:g}"] = crossing(sub[x], sub["S"], lv)
        rows.append(row)
    return pd.DataFrame(rows)


def refine_points(df: pd.DataFrame, x: str = "delta_mu", lo: float = 0.02, hi: float = 0.98,
                  max_new: int = 3) -> Dict[str, List[float]]:
    """Δμ nuevos por cara: puntos medios de los intervalos donde la curva sube.

    Se refinan los intervalos [x_k, x_{k+1}] de la curva media cuyo S está entre `lo` y
    `hi` en algún extremo (o los cruza), empezando por los de mayor salto de S.
    """
    mc = mean_curves(df, x)
    out = {}
    for hkl, sub in mc.groupby("hkl", sort=False):
        sub = sub.sort_values(x)
        xs, ys = sub[x].to_numpy(), sub["S"].to_numpy()
        cand = []
        for k in range(len(xs) - 1):
            a, b = ys[k], ys[k + 1]
            if (lo < a < hi) or (lo < b < hi) or (a <= lo and b >= hi):
                cand.append((abs(b - a), 0.5 * (xs[k] + xs[k + 1])))
        cand.sort(reverse=True)
        out[hkl] = [float(v) for _, v in cand[:max_new]]
    return out


def group_faces(onsets: pd.DataFrame, column: str, rel_tol: float = 0.12,
                abs_tol: float = 0.0) -> pd.DataFrame:
    """Agrupa caras con umbral parecido (como los grupos de curvas de las Figs. 2–4).

    Ordena por `column` y abre un grupo nuevo cuando el salto respecto de la cara
    anterior supera max(abs_tol, rel_tol·valor). Devuelve la tabla con la columna "grupo".
    """
    t = onsets.sort_values(column).reset_index(drop=True).copy()
    grupo, prev, gid = [], None, 0
    for v in t[column]:
        if prev is not None and np.isfinite(v) and np.isfinite(prev) \
                and v - prev > max(abs_tol, rel_tol * abs(v)):
            gid += 1
        grupo.append(gid)
        prev = v
    t["grupo"] = grupo
    return t


# ---------------------------------------------------------------------------
# Referencia (Figs. 2–4 de MONTY) y figuras
# ---------------------------------------------------------------------------
def load_reference(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Tabla de referencia leída de las Figs. 2–4 y del texto del artículo.

    Una fila por grupo de caras: grasa, figura, caras (separadas por espacios), Δμ en
    kcal/mol al arrancar (umbral del texto) y con S = 0.5 (leído en la figura), y la
    fuente de cada valor. Ver `data/monty2004/README.md`.
    """
    path = Path(path) if path is not None else DATA_DIR / "referencia_figuras.csv"
    return pd.read_csv(path, comment="#")


def compare_with_reference(df: pd.DataFrame, reference: pd.DataFrame, fat: str,
                           x: str = "delta_mu_kcal", rel_tol: float = 0.2,
                           abs_tol: float = 1.0) -> pd.DataFrame:
    """Compara los umbrales simulados de cada grupo de caras con los de MONTY.

    Para cada grupo de `reference` (filas de `load_reference` con grasa = `fat`):
    - si hay valor digitalizado en S = 0.5, se compara con la media de Δμ(S = 0.5) de
      las caras simuladas del grupo;
    - si no (las basales, con solo el umbral del texto), se compara Δμ(S = 0.1) con
      `dmu_texto`.
    Un grupo cumple si |sim − ref| <= max(`abs_tol`, `rel_tol`·ref). Las caras del
    grupo que no se simularon se ignoran; si no hay ninguna, el grupo queda sin veredicto.
    """
    on = onset_table(df, x)
    rows = []
    for _, r in reference[reference["grasa"] == fat].iterrows():
        caras = str(r["caras"]).split()
        sub = on[on["hkl"].isin(caras)]
        usa_s05 = np.isfinite(r["dmu_S05"])
        col = f"{x}@S={0.5 if usa_s05 else 0.1:g}"
        ref = float(r["dmu_S05"] if usa_s05 else r["dmu_texto"])
        vals = sub[col].to_numpy(float) if len(sub) else np.array([])
        sim = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        tol = max(abs_tol, rel_tol * abs(ref))
        rows.append({"grupo": int(r["grupo"]), "caras": r["caras"], "medida": "S=0.5" if usa_s05 else "S=0.1 vs texto",
                     "ref": ref, "sim": sim, "sim_rango": float(np.nanmax(vals) - np.nanmin(vals))
                     if np.isfinite(vals).sum() > 1 else 0.0,
                     "n_caras": int(np.isfinite(vals).sum()), "diferencia": sim - ref,
                     "relativa": (sim - ref) / ref if ref else np.nan,
                     "cumple": bool(abs(sim - ref) <= tol) if np.isfinite(sim) else None})
    return pd.DataFrame(rows)


def order_agreement(comparison: pd.DataFrame) -> float:
    """Fracción de pares de grupos que la simulación ordena igual que MONTY.

    Solo cuenta los pares con valor simulado y cuyos valores de referencia se
    distinguen (difieren más del 5 %).
    """
    c = comparison[np.isfinite(comparison["sim"])].reset_index(drop=True)
    buenos = total = 0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            ri, rj = c.loc[i, "ref"], c.loc[j, "ref"]
            if abs(ri - rj) <= 0.05 * max(abs(ri), abs(rj)):
                continue
            total += 1
            buenos += (ri - rj) * (c.loc[i, "sim"] - c.loc[j, "sim"]) > 0
    return buenos / total if total else float("nan")


def plot_sticking(df: pd.DataFrame, ax=None, x: str = "delta_mu", faces: Optional[Sequence[str]] = None,
                  label: bool = True, errorbars: bool = True, color: Optional[str] = None,
                  xlabel: Optional[str] = None, **line_kw):
    """Curvas S(Δμ) por cara (media de las semillas), con el estilo de las Figs. 2–4.

    `faces` fija qué caras y en qué orden se dibujan. Devuelve el eje.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.6))
    mc = mean_curves(df, x)
    sel = list(faces) if faces is not None else list(dict.fromkeys(mc["hkl"]))
    for hkl in sel:
        sub = mc[mc["hkl"] == hkl].sort_values(x)
        if not len(sub):
            continue
        kw = {"lw": 1.4, **line_kw}
        if color is not None:
            kw["color"] = color
        if errorbars and (sub["count"] > 1).any():
            ax.errorbar(sub[x], sub["S"], yerr=sub["sem"], capsize=0, label=hkl if label else None,
                        marker=".", ms=3, **kw)
        else:
            ax.plot(sub[x], sub["S"], label=hkl if label else None, marker=".", ms=3, **kw)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel or ("Δμ (kcal/mol)" if x == "delta_mu_kcal" else "Δμ / kT"))
    ax.set_ylabel("fracción de adhesión S")
    return ax
