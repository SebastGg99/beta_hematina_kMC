"""Probabilidad de adsorción por clase de sitio frente a la sobresaturación.

Reproduce la Fig. 8 de Nagpal et al. (2024) (`references/modern_kMC/`) con el motor
de la línea ESTÁTICA. Para una superficie `heights` y una σ dadas:

    P_k = W_a,k / W_a ,    W_a,k = Σ_{sitios de clase k} r_a(ix, iy) ,

donde (ix, iy) son los enlaces laterales de adsorción de cada sitio (vecinos con
altura EXACTAMENTE h+1, convención de `LatticeSOSStatic.adsorption_bonds_xy`) y
r_a es la tasa del propio motor (`KMC_BKL_Static.r_a`, con S = ln(1+σ)). Este
módulo NO reimplementa la física: construye un motor mínimo y le pide r_a.

Clases (i = ix + iy enlaces laterales):
    0 = adatom, 1 = step (edge), 2 = kink, 3 = hang, 4 = vacancy.

Convenciones para las tres curvas de la figura:
- "lump"    (por defecto): i ≥ 2 se suma a "kink" y se normaliza sobre todos los
            sitios. Es lo que registra el motor en `adsorption_probs_history`
            (`_adsorption_probabilities_3class`).
- "exclude": solo i = 0, 1, 2, normalizando sobre esas tres clases (los sitios
            hang/vacancy no cuentan). Es lo que hacía el motor `_v2` que generó
            `results/probabilidades_vs_sigma_75_126_418.png`.

Fuentes de datos admitidas (para corridas antiguas y futuras):
- historial del motor (`stats['adsorption_probs_history']`), solo "lump";
- snapshots `(t, heights, conversion)` (`snaps.pkl` o `run()` en memoria);
- carpetas de corrida de `scripts/run_sigma_scan.py` / `results/` (metadata.json);
- cachés `alturas.npz` + `diseno.json` de los barridos a espesor controlado
  (`notebooks/metricas_regimen_sigma.ipynb`).

Solo línea estática: la dinámica usa otra red y otra tasa de adsorción.
"""

from dataclasses import fields, replace
from pathlib import Path
import json
import re
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = Union[str, Path]
Window = Optional[Tuple[float, float]]

# Clases que se dibujan y sus etiquetas (las del artículo)
SITE_CLASSES: Tuple[int, ...] = (0, 1, 2)
SITE_LABELS: Dict[int, str] = {0: "adatom", 1: "step", 2: "kink"}
CONVENTIONS: Tuple[str, ...] = ("lump", "exclude")
# Colores de la Fig. 8 del artículo (paleta por defecto de MATLAB) y un marcador
# distinto por clase para que la identidad no dependa solo del color.
SITE_COLORS: Dict[int, str] = {0: "#0072BD", 1: "#D95319", 2: "#EDB120"}
SITE_MARKERS: Dict[int, str] = {0: "o", 1: "s", 2: "^"}
_MAX_BONDS_AXIS = 2   # en una red cuadrada hay como mucho 2 vecinos por eje


# ---------------------------------------------------------------------------
# Enlaces de adsorción y tasas
# ---------------------------------------------------------------------------
def adsorption_bond_maps(heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Enlaces laterales de adsorción (ix, iy) de cada sitio, vectorizado.

    Misma convención que `LatticeSOSStatic.adsorption_bonds_xy` (red periódica):
    un vecino cuenta si su altura es EXACTAMENTE h+1; ix cuenta los vecinos a lo
    largo del eje 0 (índice i) e iy los del eje 1 (índice j).
    """
    h = np.asarray(heights)
    if h.ndim != 2:
        raise ValueError(f"heights debe ser 2D, no {h.shape}")
    target = h + 1
    # Vecinos (i-1, j) y (i+1, j): eje 0 -> ix
    ix = (np.roll(h, 1, axis=0) == target).astype(np.int64)
    ix += np.roll(h, -1, axis=0) == target
    # Vecinos (i, j-1) y (i, j+1): eje 1 -> iy
    iy = (np.roll(h, 1, axis=1) == target).astype(np.int64)
    iy += np.roll(h, -1, axis=1) == target
    return ix, iy


def site_class_counts(heights: np.ndarray) -> np.ndarray:
    """M_i: número de sitios con i = 0..4 enlaces laterales de adsorción."""
    ix, iy = adsorption_bond_maps(heights)
    return np.bincount((ix + iy).ravel(), minlength=5)[:5]


def static_params(params, sigma: Optional[float] = None):
    """Devuelve un `KMCParamsStatic` (desde dataclass o dict) con `fixed_sigma=sigma`.

    Las claves que no son campos del dataclass se ignoran, así que sirve tanto el
    bloque `params` de metadata.json como el de diseno.json.
    """
    from ..static import KMCParamsStatic   # import local: analysis no depende del motor

    if isinstance(params, KMCParamsStatic):
        p = params
    elif isinstance(params, Mapping):
        names = {f.name for f in fields(KMCParamsStatic)}
        p = KMCParamsStatic(**{k: v for k, v in params.items() if k in names})
    else:
        raise TypeError(f"params debe ser KMCParamsStatic o dict, no {type(params)}")
    if sigma is not None:
        p = replace(p, fixed_sigma=float(sigma))
    if p.fixed_sigma is None:
        raise ValueError("Hace falta σ: pásala como `sigma` o en params.fixed_sigma")
    return p


def _rate_engine(params, sigma: Optional[float] = None):
    """Motor estático mínimo (red 3x3, concentración constante) para pedirle r_a.

    Con concentración constante N_bulk = N0, así que el factor N_bulk/N0 de r_a es 1.
    El factor de solvente de adsorción multiplica a todas las clases y se cancela en
    P_k. No se ejecuta ningún evento.
    """
    from ..static import KMC_BKL_Static, LatticeSOSStatic

    p = static_params(params, sigma)
    return KMC_BKL_Static(lattice=LatticeSOSStatic(size=(3, 3), seed=0), params=p,
                          N_bulk0=1, constant_concentration=True,
                          record_adsorption_probs=False)


def adsorption_rate_table(params, sigma: Optional[float] = None) -> np.ndarray:
    """Tabla R[ix, iy] = r_a(ix, iy) del motor estático (ix, iy = 0..2)."""
    eng = _rate_engine(params, sigma)
    n = _MAX_BONDS_AXIS + 1
    return np.array([[eng.r_a(ix, iy) for iy in range(n)] for ix in range(n)], dtype=float)


def effective_sigma(params, sigma: Optional[float] = None) -> Tuple[float, float]:
    """(σ, S) que usa realmente el motor (σ se recorta vía S_floor/S_ceil).

    Si el motor no recorta, devuelve la σ nominal exacta: el paso por
    expm1(log1p(σ)) del motor deja errores de redondeo (0.2 -> 0.20000000000000018)
    que estropean las agrupaciones por σ.
    """
    eng = _rate_engine(params, sigma)
    sig_eff, S = float(eng.sigma), float(eng.supersaturation)
    nominal = float(eng.p.fixed_sigma)
    if np.isclose(sig_eff, nominal, rtol=1e-9, atol=1e-12):
        return nominal, float(np.log1p(nominal))
    return sig_eff, S


# ---------------------------------------------------------------------------
# Probabilidades a partir de una superficie
# ---------------------------------------------------------------------------
def adsorption_probabilities(heights: np.ndarray, rates: np.ndarray,
                             convention: str = "lump") -> Dict[str, float]:
    """P_adatom, P_step, P_kink de una superficie dada la tabla de tasas.

    Parámetros:
        heights: alturas enteras (Lx, Ly)
        rates: tabla R[ix, iy] de `adsorption_rate_table`
        convention: "lump" (i ≥ 2 -> kink, como el motor) o "exclude" (solo i ≤ 2)

    Devuelve {"p0", "p1", "p2", "Wa", "M0".."M4"}; Wa es la tasa total de
    adsorción por unidad de K0+ x factores comunes (solo útil para comparar).
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention debe ser una de {CONVENTIONS}")
    ix, iy = adsorption_bond_maps(heights)
    i_tot = ix + iy
    r_site = np.asarray(rates, dtype=float)[ix, iy]   # tasa de cada sitio
    w = np.array([r_site[i_tot == 0].sum(), r_site[i_tot == 1].sum(),
                  r_site[i_tot >= 2].sum() if convention == "lump"
                  else r_site[i_tot == 2].sum()])
    Wa = float(w.sum())
    p = w / Wa if Wa > 0.0 else np.zeros(3)
    out = {f"p{k}": float(p[k]) for k in SITE_CLASSES}
    out["Wa"] = Wa
    counts = np.bincount(i_tot.ravel(), minlength=5)[:5]
    out.update({f"M{i}": int(counts[i]) for i in range(5)})
    return out


def equal_sites_reference(params, sigmas: Iterable[float],
                          convention: str = "exclude") -> pd.DataFrame:
    """Referencia sin morfología: P_k con el mismo número de sitios en cada clase.

    P_k = r̄_k / Σ_j r̄_j, con r̄_k la tasa media de los pares (ix, iy) con
    ix + iy = k. Aísla el efecto del factor exp(i·δ/S) de r_a: la diferencia con la
    curva simulada es el efecto de la morfología (los M_i). Solo informativa: con
    "lump" las clases 3 y 4 se suman a kink con el mismo peso.
    """
    rows = []
    for s in sigmas:
        R = adsorption_rate_table(params, s)
        # tasas medias por número total de enlaces i = ix + iy
        r_by_i = {i: np.mean([R[a, i - a] for a in range(3) if 0 <= i - a <= 2])
                  for i in range(5)}
        w = np.array([r_by_i[0], r_by_i[1],
                      sum(r_by_i[i] for i in range(2, 5)) if convention == "lump"
                      else r_by_i[2]])
        sig, S = effective_sigma(params, s)
        rows.append({"sigma": sig, "S": S, **{f"p{k}": w[k] / w.sum() for k in SITE_CLASSES}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Series a partir de snapshots o del historial del motor
# ---------------------------------------------------------------------------
def _snapshot_heights(snaps) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Normaliza snapshots: lista de (t, heights, conv) o array (n, Lx, Ly)."""
    if isinstance(snaps, np.ndarray) and snaps.ndim == 3:
        return np.full(len(snaps), np.nan), [h for h in snaps]
    times, heights = [], []
    for item in snaps:
        times.append(float(item[0]))
        heights.append(np.asarray(item[1]))
    return np.array(times), heights


def probabilities_from_snapshots(snaps, params, sigma: Optional[float] = None,
                                 convention: str = "lump") -> pd.DataFrame:
    """Una fila por snapshot con t, ⟨h⟩, espesor depositado dh y P_k.

    `snaps` puede ser la lista que devuelve `run()` / guarda `snaps.pkl`, o un array
    (n, Lx, Ly) como los de `alturas.npz` (entonces t = NaN). dh se mide respecto
    al PRIMER snapshot. Los snapshots repetidos al final (corrida cortada por
    max_events) se marcan con `repeated_state=True`.
    """
    times, heights = _snapshot_heights(snaps)
    rates = adsorption_rate_table(params, sigma)
    sig, S = effective_sigma(params, sigma)
    h0 = float(np.mean(heights[0])) if heights else 0.0
    rows, previous = [], None
    for k, (t, h) in enumerate(zip(times, heights)):
        row = {"k": k, "t": t, "sigma": sig, "S": S, "mean_h": float(np.mean(h)),
               "dh": float(np.mean(h)) - h0,
               "repeated_state": previous is not None and np.array_equal(h, previous)}
        row.update(adsorption_probabilities(h, rates, convention))
        rows.append(row)
        previous = h
    return pd.DataFrame(rows)


def history_frame(stats: Mapping) -> pd.DataFrame:
    """`adsorption_probs_history` (+ ⟨h⟩ de `height_history`) como DataFrame.

    Una fila por evento: t, sigma, p0, p1, p2 y, si los dos historiales tienen la
    misma longitud (se registran juntos en cada evento), mean_h y dh (respecto al
    primer evento). Convención "lump" (la del motor).
    """
    hist = stats.get("adsorption_probs_history", [])
    if len(hist) == 0:
        raise ValueError("stats sin 'adsorption_probs_history' (¿record_adsorption_probs=False?). "
                         "Usa probabilities_from_snapshots.")
    df = pd.DataFrame({
        "t": np.fromiter((e[0] for e in hist), float, len(hist)),
        "sigma": np.fromiter((e[1] for e in hist), float, len(hist)),
        **{f"p{k}": np.fromiter((e[2].get(k, 0.0) for e in hist), float, len(hist))
           for k in SITE_CLASSES},
    })
    hh = stats.get("height_history", [])
    if len(hh) == len(hist):
        df["mean_h"] = np.fromiter((e[1] for e in hh), float, len(hh))
        df["dh"] = df["mean_h"] - df["mean_h"].iloc[0]
    return df


def _window_mask(df: pd.DataFrame, h_window: Window, tail_frac: Optional[float]) -> np.ndarray:
    """Filas dentro de la ventana: dh ∈ (h_min, h_max] o la fracción final del tiempo."""
    mask = np.ones(len(df), dtype=bool)
    if h_window is not None:
        if "dh" not in df:
            raise ValueError("No hay espesor (dh) para aplicar h_window")
        mask &= (df["dh"] > h_window[0]).to_numpy() & (df["dh"] <= h_window[1]).to_numpy()
    if tail_frac is not None:
        t = df["t"].to_numpy()
        if np.all(np.isfinite(t)) and len(t):
            mask &= t >= t[0] + (1.0 - tail_frac) * (t[-1] - t[0])
        else:   # sin tiempos (alturas.npz): fracción final de los snapshots
            mask &= np.arange(len(df)) >= int(np.floor((1.0 - tail_frac) * len(df)))
    if "repeated_state" in df:
        mask &= ~df["repeated_state"].to_numpy()
    return mask


def window_average(df: pd.DataFrame, h_window: Window = None,
                   tail_frac: Optional[float] = None,
                   weighting: str = "uniform") -> Dict[str, float]:
    """Promedio de p0..p2 en una ventana de espesor (dh) y/o de tiempo final.

    weighting:
        "uniform": cada fila pesa igual (snapshots equiespaciados en dh o eventos)
        "time":    cada estado pesa lo que dura, t_{k+1} - t_k (historial por
                   evento; el último estado pesa 0 porque no se sabe cuánto dura)
    Devuelve p_k, sd_k (desviación típica ponderada: fluctuación, no error
    estándar), n (filas usadas) y el rango de dh/t cubierto.
    """
    mask = _window_mask(df, h_window, tail_frac)
    if weighting == "time":
        t = df["t"].to_numpy()
        w = np.diff(t, append=t[-1]) if len(t) else np.array([])
    elif weighting == "uniform":
        w = np.ones(len(df))
    else:
        raise ValueError("weighting debe ser 'uniform' o 'time'")
    w = np.where(mask, w, 0.0)
    out: Dict[str, float] = {"n": int(mask.sum())}
    if w.sum() <= 0.0:
        out.update({f"{c}{k}": np.nan for c in ("p", "sd") for k in SITE_CLASSES})
    else:
        for k in SITE_CLASSES:
            x = df[f"p{k}"].to_numpy()
            m = float(np.sum(w * x) / w.sum())
            out[f"p{k}"] = m
            out[f"sd{k}"] = float(np.sqrt(np.sum(w * (x - m) ** 2) / w.sum()))
    for col in ("dh", "t"):
        if col in df and mask.any():
            out[f"{col}_min"] = float(df.loc[mask, col].min())
            out[f"{col}_max"] = float(df.loc[mask, col].max())
    return out


# ---------------------------------------------------------------------------
# Carpetas de corrida (results/, scripts/run_sigma_scan.py)
# ---------------------------------------------------------------------------
def _is_anisotropic(params: Mapping) -> bool:
    return (not np.isclose(params.get("delta_x", 0), params.get("delta_y", 0))
            or not np.isclose(params.get("E_pb_over_kT_x", 0), params.get("E_pb_over_kT_y", 0)))


def find_run_dirs(paths: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    """Carpetas de corrida (con metadata.json) dentro de `paths`, a cualquier profundidad.

    Una carpeta de corrida debe tener metadata.json y snaps.pkl o stats.pkl; así se
    descartan otros metadata.json (p. ej. resúmenes de un barrido).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found: List[Path] = []
    for p in map(Path, paths):
        for meta in sorted(p.rglob("metadata.json")):
            d = meta.parent
            if (d / "snaps.pkl").exists() or (d / "stats.pkl").exists():
                found.append(d)
    return found


def run_probabilities(run_dir: PathLike, source: str = "auto", convention: str = "lump",
                      h_window: Window = None, tail_frac: Optional[float] = 0.5,
                      weighting: Optional[str] = None) -> Dict[str, object]:
    """Probabilidades promediadas de UNA carpeta de corrida de la línea estática.

    Parámetros:
        source: "history" (stats.pkl, por evento), "snapshots" (snaps.pkl) o "auto"
                (historial si existe y convention == "lump"; si no, snapshots)
        h_window: ventana de espesor depositado (dh_min, dh_max]; recomendado para
                  comparar corridas con distinta velocidad o duración
        tail_frac: fracción final del tiempo que se promedia (None = todo)
        weighting: por defecto "time" con historial y "uniform" con snapshots

    Devuelve una fila (dict) con la identificación de la corrida, σ, S, los
    parámetros que distinguen carpetas y el promedio de `window_average`.
    """
    from ..common.io import load_json, load_pickle

    run_dir = Path(run_dir)
    meta = load_json(run_dir / "metadata.json")
    if meta.get("engine", "static") != "static":
        raise ValueError(f"{run_dir}: solo se admite la línea estática")
    params = dict(meta["params"])
    sigma = meta.get("fixed_sigma", params.get("fixed_sigma"))

    stats = None
    if source in ("auto", "history"):
        stats = load_pickle(run_dir / "stats.pkl")
        has_hist = len(stats.get("adsorption_probs_history", [])) > 0
        if source == "history" and convention != "lump":
            raise ValueError("El historial del motor solo tiene la convención 'lump'")
        if source == "history" and not has_hist:
            raise ValueError(f"{run_dir}: stats.pkl sin historial de probabilidades")
        source = "history" if (has_hist and convention == "lump") else "snapshots"

    if source == "history":
        df = history_frame(stats)
        weighting = weighting or "time"
    elif source == "snapshots":
        df = probabilities_from_snapshots(load_pickle(run_dir / "snaps.pkl"), params,
                                          sigma, convention)
        weighting = weighting or "uniform"
    else:
        raise ValueError("source debe ser 'auto', 'history' o 'snapshots'")

    sig, S = effective_sigma(params, sigma)
    row: Dict[str, object] = {
        "folder": run_dir.parent.name, "run_id": run_dir.name, "run_dir": str(run_dir),
        "sigma": sig, "S": S,
        "anisotropic": _is_anisotropic(params), "K0_plus": params.get("K0_plus"),
        "delta_x": params.get("delta_x"), "delta_y": params.get("delta_y"),
        "time_scale": meta.get("time_scale"), "size": "x".join(map(str, meta.get("size", []))),
        "rng_seed": meta.get("rng_seed"), "source": source, "convention": convention,
        "weighting": weighting, "dh_final": float(df["dh"].iloc[-1]) if "dh" in df else np.nan,
    }
    row.update(window_average(df, h_window, tail_frac, weighting))
    return row


def scan_run_folders(paths: Union[PathLike, Sequence[PathLike]], **kwargs) -> pd.DataFrame:
    """`run_probabilities` sobre todas las corridas de `paths` (una fila por corrida).

    `paths` puede mezclar carpetas de corrida y carpetas que las contienen
    (p. ej. `results/outputs_iso_2`). Las corridas ilegibles se saltan con aviso.
    """
    rows = []
    for d in find_run_dirs(paths):
        try:
            rows.append(run_probabilities(d, **kwargs))
        except (OSError, ValueError, KeyError) as exc:
            warnings.warn(f"Se omite {d}: {exc}")
    if not rows:
        warnings.warn(f"No se encontraron corridas en {paths}")
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["folder", "sigma"], ignore_index=True)


# ---------------------------------------------------------------------------
# Cachés alturas.npz + diseno.json (barridos a espesor controlado)
# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"^sigma_(?P<sigma>[-+0-9.eE]+)_seed_(?P<seed>-?\d+)$")


def scan_heights_cache(cache_dir: PathLike, convention: str = "lump",
                       params: Optional[Mapping] = None) -> pd.DataFrame:
    """Una fila por (σ, semilla, snapshot) de un caché `alturas.npz` + `diseno.json`.

    Las claves del npz son `sigma_{σ}_seed_{semilla}` con arrays (n_snaps, L, L);
    los parámetros salen de `diseno.json["params"]` salvo que se pasen en `params`.
    """
    cache_dir = Path(cache_dir)
    design = json.loads((cache_dir / "diseno.json").read_text())
    params = dict(params if params is not None else design["params"])
    frames = []
    with np.load(cache_dir / "alturas.npz") as z:
        for key in z.files:
            m = _KEY_RE.match(key)
            if m is None:
                warnings.warn(f"Clave no reconocida en alturas.npz: {key}")
                continue
            df = probabilities_from_snapshots(z[key], params, float(m["sigma"]), convention)
            df.insert(0, "seed", int(m["seed"]))
            df.insert(0, "cache", cache_dir.name)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Sin tiempos en el npz: el índice k y dh (≈ k·dh_snap) son el eje de avance
    return out.sort_values(["sigma", "seed", "k"], ignore_index=True)


def average_cache_window(snap_df: pd.DataFrame, h_window: Window = (2.0, 6.0)) -> pd.DataFrame:
    """Promedia cada (σ, semilla) de `scan_heights_cache` en la ventana de espesor."""
    rows = []
    for (cache, sigma, seed), g in snap_df.groupby(["cache", "sigma", "seed"], sort=True):
        row = {"cache": cache, "sigma": sigma, "S": float(g["S"].iloc[0]), "seed": seed}
        row.update(window_average(g, h_window=h_window, weighting="uniform"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Resumen frente a σ y figura
# ---------------------------------------------------------------------------
def summarize_vs_sigma(df: pd.DataFrame, by: Sequence[str] = ("sigma",)) -> pd.DataFrame:
    """Media y dispersión ENTRE réplicas (semillas o corridas) para cada σ.

    Espera columnas p0..p2 (una fila por réplica). Devuelve p_k, err_k (desviación
    típica entre réplicas; 0 si solo hay una) y n_rep.
    """
    by = list(by)
    g = df.dropna(subset=[f"p{k}" for k in SITE_CLASSES]).groupby(by, sort=True)
    out = g[[f"p{k}" for k in SITE_CLASSES]].mean()
    err = g[[f"p{k}" for k in SITE_CLASSES]].std(ddof=1).fillna(0.0)
    err.columns = [f"err{k}" for k in SITE_CLASSES]
    out = out.join(err)
    out["n_rep"] = g.size()
    if "S" in df:
        out["S"] = g["S"].first()
    return out.reset_index()


def plot_site_probabilities(summary: pd.DataFrame, x: str = "sigma", ax=None,
                            errors: bool = True, markers: bool = True,
                            reference: Optional[pd.DataFrame] = None,
                            reference_label: Optional[str] = None,
                            label_suffix: str = "", line_kw: Optional[dict] = None,
                            xlim: Optional[Tuple[float, float]] = None,
                            frame_like_paper: bool = True):
    """Figura tipo Fig. 8: P_adatom, P_step y P_kink frente a σ (o S).

    Parámetros:
        summary: tabla con columnas `x`, p0..p2 y opcionalmente err0..err2
        x: "sigma" (σ = C/C_eq − 1, eje del artículo) o "S" (= ln(1+σ))
        reference: tabla con `x` y p0..p2 que se dibuja en trazo discontinuo
                   (p. ej. `equal_sites_reference` o una curva digitalizada)
        reference_label: texto de la leyenda para el trazo discontinuo
        line_kw: argumentos extra para `ax.plot` (p. ej. {"alpha": 0.5})
        frame_like_paper: marcas hacia dentro en los cuatro lados, como el artículo
    Devuelve el eje.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))
    s = summary.sort_values(x)
    kw = {"linewidth": 2.0, "markersize": 5.5}
    kw.update(line_kw or {})
    for k in SITE_CLASSES:
        label = f"{SITE_LABELS[k]} (i = {k}){label_suffix}"
        ax.plot(s[x], s[f"p{k}"], color=SITE_COLORS[k],
                marker=SITE_MARKERS[k] if markers else None, label=label, **kw)
        if errors and f"err{k}" in s and np.any(s[f"err{k}"] > 0):
            ax.fill_between(s[x], s[f"p{k}"] - s[f"err{k}"], s[f"p{k}"] + s[f"err{k}"],
                            color=SITE_COLORS[k], alpha=0.18, linewidth=0)
    if reference is not None:
        r = reference.sort_values(x)
        for k in SITE_CLASSES:
            ax.plot(r[x], r[f"p{k}"], color=SITE_COLORS[k], linestyle="--",
                    linewidth=1.3, alpha=0.8)
        if reference_label:
            # entrada neutra en la leyenda: el color ya identifica la clase
            ax.plot([], [], color="0.35", linestyle="--", linewidth=1.3, label=reference_label)
    ax.set_xlabel("Sobresaturación σ" if x == "sigma" else "S = ln(1 + σ)")
    ax.set_ylabel("Probabilidad de adsorción")
    ax.set_ylim(0, 1)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if frame_like_paper:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    ax.legend(loc="best", frameon=False)
    return ax
