"""Parámetros de orden y transiciones de régimen al aumentar la sobresaturación σ.

Reúne en un solo flujo todos los observables de `src/analysis/` (momentos, capas,
islas, escalones, correlación, clases de sitio y probabilidades de adsorción) y les
añade lo que necesita un estudio de transiciones:

1. **Referencias de deposición aleatoria (RD)**, la cota sin relajación a igual espesor:
   W_RD = √h, b_RD = (1 + 1/h)/(3 + 1/h), f≥2_RD (Skellam) y n_eff_RD.
2. **Parámetros de orden por corrida** en una ventana de espesor depositado:
   b_½ (bimodalidad a media capa), f≥2/f≥2_RD, islas por nivel N_capa, niveles
   expuestos N_niv, amplitud de las oscilaciones capa a capa Ψ_osc, β, densidad de
   islas de saturación n_sat, P_adatom/P_step/P_kink, etc.
3. **Fronteras**: cruce de cada parámetro de orden con un umbral (por réplica y sobre la
   curva media), punto de cambio más rápido (sin umbral) y clasificación por régimen.

Flujo (todo funciona con cualquier lista de `RunData`, ver `runs.py`):

    runs --run_timeseries / level_islands_table--> una fila por snapshot / por nivel
         --summarize_runs--------------------------> una fila por corrida (ventana)
         --aggregate_replicas----------------------> una fila por (barrido, σ)
         --transition_table / classify_regime------> fronteras y régimen
    analyze_transitions() hace todo lo anterior, con caché opcional en disco.

Comparación a igual espesor ⟨h⟩, no a igual tiempo: la velocidad v(σ) del modelo tiene
la anomalía del término δ/S (README §8.4) y cambia mucho con σ. Los umbrales son
convenciones explícitas (`DEFAULT_CRITERIA`), no resultados: el código devuelve números.
"""

from dataclasses import dataclass, field
import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
import numpy as np
import pandas as pd

from .adsorption import adsorption_probabilities, adsorption_rate_table, site_class_counts
from .correlation import correlation_extrema
from .islands import island_density_observables, level_island_statistics
from .regime import _nice_log_ticks, growth_exponent, growth_rate, surface_observables
from .runs import RunData, assign_scans, runs_inventory

PathLike = Union[str, Path]
Window = Tuple[float, float]

# Versión del cálculo: si cambia, las cachés de disco se recalculan solas
ANALYSIS_VERSION = "1"

# Referencias exactas del índice de bimodalidad de Sarle b = (γ3² + 1)/κ
B_SARLE = 5.0 / 9.0      # distribución uniforme: frontera convencional bimodal / no bimodal
B_GAUSS = 1.0 / 3.0      # alturas gaussianas
_EPS = 1e-9              # tolerancia para comparar espesores nominales (múltiplos de 0.1)


# ---------------------------------------------------------------------------
# 1. Referencias de deposición aleatoria (RD)
# ---------------------------------------------------------------------------
def _log_scaled_bessel_i(nu: int, x: float) -> float:
    """log(e^(−x) I_ν(x)) por la serie de potencias, sumada en logaritmos.

    I_ν(x) = Σ_m (x/2)^(2m+ν) / (m! (m+ν)!). El término máximo está en m ≈ x/2, así
    que basta con m ≲ x + 10√x + 50. Estable para los espesores de interés (x ≲ 10³).
    """
    if x <= 0.0:
        return 0.0 if nu == 0 else -np.inf
    m_max = int(x + 10.0 * np.sqrt(x) + 50)
    m = np.arange(m_max + 1)
    # log(k!) acumulado hasta m_max + ν
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, m_max + nu + 1)))])
    log_terms = (2 * m + nu) * np.log(x / 2.0) - log_fact[m] - log_fact[m + nu]
    top = log_terms.max()
    return float(top + np.log(np.exp(log_terms - top).sum()) - x)


def rd_multi_step_fraction(h) -> np.ndarray:
    """f≥2 de la deposición aleatoria con espesor medio h (array o escalar).

    En RD cada columna recibe Poisson(h) átomos de forma independiente, así que la
    diferencia entre vecinos sigue una Skellam(h, h): P(Δ = k) = e^(−2h) I_k(2h).
    f≥2 = P(|Δ| ≥ 2 | Δ ≠ 0) = 1 − 2 P(Δ = 1) / (1 − P(Δ = 0)). NaN si h ≤ 0.
    """
    out = []
    for hi in np.atleast_1d(np.asarray(h, dtype=float)):
        if not np.isfinite(hi) or hi <= 0:
            out.append(np.nan)
            continue
        p0 = np.exp(_log_scaled_bessel_i(0, 2.0 * hi))
        p1 = np.exp(_log_scaled_bessel_i(1, 2.0 * hi))
        out.append(1.0 - 2.0 * p1 / (1.0 - p0) if p0 < 1.0 else np.nan)
    return np.array(out)


def rd_bimodality(h) -> np.ndarray:
    """b de la RD: Poisson(h) tiene γ3 = h^(−1/2) y κ = 3 + 1/h, así que
    b = (1 + 1/h) / (3 + 1/h). Tiende a 1/3 (gaussiana) para h grande. NaN si h ≤ 0."""
    h = np.asarray(h, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h > 0, (1.0 + 1.0 / h) / (3.0 + 1.0 / h), np.nan)


def rd_effective_levels(h) -> np.ndarray:
    """n_eff = exp(H) de la RD, con H la entropía de Shannon de Poisson(h)."""
    out = []
    for hi in np.atleast_1d(np.asarray(h, dtype=float)):
        if not np.isfinite(hi) or hi < 0:
            out.append(np.nan)
            continue
        if hi == 0:
            out.append(1.0)
            continue
        k = np.arange(int(hi + 12.0 * np.sqrt(hi) + 30) + 1)
        log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, k[-1] + 1)))])
        log_p = k * np.log(hi) - hi - log_fact[k]
        p = np.exp(log_p)
        out.append(float(np.exp(-np.sum(p * log_p))))
    return np.array(out)


def random_deposition_reference(dh) -> pd.DataFrame:
    """Tabla de referencias RD (W, b, f≥2, n_eff) para cada espesor depositado `dh`."""
    dh = np.atleast_1d(np.asarray(dh, dtype=float))
    with np.errstate(invalid="ignore"):
        W = np.where(dh >= 0, np.sqrt(np.clip(dh, 0, None)), np.nan)
    return pd.DataFrame({"W_RD": W, "b_RD": rd_bimodality(dh),
                         "f2_RD": rd_multi_step_fraction(dh),
                         "n_eff_RD": rd_effective_levels(dh)})


# ---------------------------------------------------------------------------
# 2. Oscilaciones capa a capa (análogo de las oscilaciones RHEED)
# ---------------------------------------------------------------------------
def _ideal_width_amplitude(n: int = 200_000) -> float:
    """Amplitud del armónico de periodo 1 capa de W en capa a capa ideal.

    Con dos niveles y cobertura θ de la capa que crece, W(θ) = √(θ(1 − θ)). Su
    primer armónico vale 2|∫₀¹ √(θ(1−θ)) cos(2πθ) dθ| = J₁(π)/2 ≈ 0.1423 (integral
    numérica por punto medio: así no hace falta ninguna función de Bessel).
    """
    theta = (np.arange(n) + 0.5) / n
    w = np.sqrt(theta * (1.0 - theta))
    return float(np.hypot(2 * np.mean(w * np.cos(2 * np.pi * theta)),
                          2 * np.mean(w * np.sin(2 * np.pi * theta))))


# Límite continuo (muestreo infinitamente fino). `summarize_run_window` NO normaliza con
# este valor, sino con la referencia ideal evaluada en los mismos snapshots que los datos:
# con 10 snapshots por capa el armónico discreto de √(θ(1−θ)) vale ≈ 0.169, un 18 % más.
W_OSC_IDEAL = _ideal_width_amplitude()


def ideal_layer_width(dh) -> np.ndarray:
    """W del capa a capa ideal tras depositar `dh` capas sobre una superficie plana:
    dos niveles, cobertura θ = dh mod 1 de la capa que crece y W = √(θ(1 − θ))."""
    theta = np.mod(np.asarray(dh, dtype=float), 1.0)
    return np.sqrt(theta * (1.0 - theta))


def layer_oscillation(dh, x, window: Optional[Window] = None,
                      min_per_layer: float = 4.0) -> Dict[str, float]:
    """Amplitud del armónico de periodo una capa de la serie x(dh).

    Ajuste por mínimos cuadrados x = a + b·dh + c·cos(2π dh) + d·sin(2π dh) sobre los
    puntos con dh en `window` = (dh_min, dh_max]. La tendencia lineal absorbe el
    crecimiento lento (p. ej. W ∝ √dh en rugoso); c y d miden la oscilación por capa.

    Devuelve amplitude = √(c² + d²), phase = atan2(d, c) y n (puntos usados). NaN si
    hay menos de una capa completa o menos de `min_per_layer` puntos por capa (con
    snapshots a tiempo fijo y Δdh ~ 1 capa la oscilación no se puede resolver).
    """
    dh = np.asarray(dh, dtype=float)
    x = np.asarray(x, dtype=float)
    sel = np.isfinite(dh) & np.isfinite(x)
    if window is not None:
        sel &= (dh > window[0] + _EPS) & (dh <= window[1] + _EPS)
    n = int(sel.sum())
    nan = {"amplitude": np.nan, "phase": np.nan, "n": n}
    if n < 5:
        return nan
    span = float(np.ptp(dh[sel]))
    if span < 1.0 - 1e-6 or n / span < min_per_layer:
        return nan
    d = dh[sel]
    A = np.column_stack([np.ones(n), d, np.cos(2 * np.pi * d), np.sin(2 * np.pi * d)])
    coef, *_ = np.linalg.lstsq(A, x[sel], rcond=None)
    return {"amplitude": float(np.hypot(coef[2], coef[3])),
            "phase": float(np.arctan2(coef[3], coef[2])), "n": n}


# ---------------------------------------------------------------------------
# 3. Observables por snapshot y por nivel
# ---------------------------------------------------------------------------
def snapshot_observables(heights: np.ndarray, with_alpha: bool = False,
                         rates: Optional[np.ndarray] = None, convention: str = "lump",
                         level_stats: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    """Todos los observables escalares de una superficie.

    Es `surface_observables` (momentos, capas, islas, escalones, ξ) más:
    - descomposición de la densidad de islas (`island_density_observables`);
    - cero y mínimo de la correlación radial (`correlation_extrema`);
    - fracción de sitios con i = 0..4 enlaces laterales de adsorción, `site_frac_i`
      (0 terraza, 1 borde de escalón, 2 kink, 3 hang, 4 vacante; solo geometría);
    - si se pasa la tabla de tasas `rates` (`adsorption_rate_table`, línea estática),
      las probabilidades de adsorción por clase p0 (adatom), p1 (step), p2 (kink).
    """
    h = np.asarray(heights, dtype=int)
    obs = surface_observables(h, with_alpha=with_alpha)
    obs.update(island_density_observables(h, level_stats=level_stats))
    obs.update(correlation_extrema(h))
    counts = site_class_counts(h)
    obs.update({f"site_frac_{i}": float(counts[i]) / h.size for i in range(5)})
    if rates is not None:
        probs = adsorption_probabilities(h, rates, convention)
        obs.update({f"p{k}": probs[f"p{k}"] for k in (0, 1, 2)})
    return obs


def _rate_table_or_none(run: RunData, with_adsorption: bool) -> Optional[np.ndarray]:
    """Tabla de tasas r_a del motor estático para la corrida, o None si no aplica."""
    if not with_adsorption or run.engine != "static" or not run.params:
        return None
    try:
        return adsorption_rate_table(run.params, run.sigma)
    except (TypeError, ValueError) as exc:
        warnings.warn(f"{run.label}: sin probabilidades de adsorción ({exc})")
        return None


def _thickness_axis(run: RunData) -> np.ndarray:
    """Espesor nominal de cada snapshot: `dh_objetivo` si la fuente lo guarda (barridos a
    espesor controlado, múltiplos exactos de Δh) y si no el espesor medido."""
    info = run.snapshot_info
    if info is not None and "dh_objetivo" in info:
        return info["dh_objetivo"].to_numpy(dtype=float)
    return run.dh


def _level_rows(run: RunData, k: int, dh: float, dh_grid: float,
                stats: Dict[str, object], repeated: bool) -> List[Dict[str, object]]:
    """Filas de la tabla por nivel de un snapshot (salida de `level_island_statistics`)."""
    n_sites = run.shape[0] * run.shape[1]
    return [{
        "source": run.source, "run_id": run.run_id, "seed": run.seed, "sigma": run.sigma,
        "k": k, "dh": float(dh), "dh_grid": float(dh_grid),
        "level": int(stats["level"][j]), "coverage": float(stats["coverage"][j]),
        "n_islands": int(stats["n_islands"][j]), "n_holes": int(stats["n_holes"][j]),
        "island_density": stats["n_islands"][j] / n_sites,
        "hole_density": stats["n_holes"][j] / n_sites,
        "mean_island_size": float(stats["mean_island_size"][j]),
        "largest_island_fraction": float(stats["largest_island_fraction"][j]),
        "repeated_state": bool(repeated),
    } for j in range(stats["level"].size)]


def run_timeseries(run: RunData, with_alpha: bool = False, with_adsorption: bool = True,
                   convention: str = "lump", every_n: int = 1,
                   half_layer: Window = (0.4, 0.6),
                   return_levels: bool = False):
    """Una fila por snapshot con todos los observables, las referencias RD y los
    indicadores derivados.

    Columnas de identificación: source, run_id, seed, sigma, S, k, t, mean_h, dh
    (espesor medido), dh_grid (espesor nominal), phase (= dh_grid mod 1, cobertura de
    la capa que crece), half_layer (phase en `half_layer`), repeated_state.

    Derivadas: f2 (= multi_step_fraction con NaN → 0: sin escalones no hay escalones
    múltiples), f2_over_RD, W_over_RD.

    Con `return_levels=True` devuelve también la tabla por nivel (`level_islands_table`)
    sin repetir el etiquetado de islas.
    """
    if every_n < 1:
        raise ValueError("every_n debe ser >= 1")
    rates = _rate_table_or_none(run, with_adsorption)
    dh, dh_grid = run.dh, _thickness_axis(run)
    mean_h = run.mean_h
    rd = random_deposition_reference(dh)
    info = run.snapshot_info
    extra_cols = [c for c in (info.columns if info is not None else [])
                  if c not in ("k", "t", "dh_objetivo")]

    rows, level_rows, previous = [], [], None
    for k in range(0, run.n_snapshots, every_n):
        h = np.asarray(run.heights[k], dtype=int)
        phase = float(np.round(dh_grid[k] % 1.0, 6)) % 1.0
        row = {
            "source": run.source, "run_id": run.run_id, "seed": run.seed,
            "sigma": run.sigma, "S": run.S, "k": k, "t": float(run.t[k]),
            "mean_h": float(mean_h[k]), "dh": float(dh[k]), "dh_grid": float(dh_grid[k]),
            "phase": phase,
            "half_layer": bool(half_layer[0] - _EPS <= phase <= half_layer[1] + _EPS),
            "repeated_state": previous is not None and np.array_equal(h, previous),
        }
        for c in extra_cols:
            row[c] = info[c].iloc[k]
        stats = level_island_statistics(h)
        row.update(snapshot_observables(h, with_alpha=with_alpha, rates=rates,
                                        convention=convention, level_stats=stats))
        row.update(rd.iloc[k].to_dict())
        rows.append(row)
        if return_levels:
            level_rows.extend(_level_rows(run, k, dh[k], dh_grid[k], stats,
                                          row["repeated_state"]))
        previous = h

    ts = pd.DataFrame(rows)
    ts["f2"] = ts["multi_step_fraction"].fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts["f2_over_RD"] = ts["f2"] / ts["f2_RD"]
        ts["W_over_RD"] = ts["W"] / ts["W_RD"]
    if return_levels:
        return ts, pd.DataFrame(level_rows)
    return ts


def level_islands_table(run: RunData, every_n: int = 1) -> pd.DataFrame:
    """Una fila por (snapshot, nivel expuesto): cobertura, islas, huecos y tamaños.

    Es la base de la densidad de islas frente a la cobertura de su capa, n(θ).
    """
    dh, dh_grid = run.dh, _thickness_axis(run)
    rows, previous = [], None
    for k in range(0, run.n_snapshots, every_n):
        h = np.asarray(run.heights[k], dtype=int)
        repeated = previous is not None and np.array_equal(h, previous)
        rows.extend(_level_rows(run, k, dh[k], dh_grid[k], level_island_statistics(h), repeated))
        previous = h
    return pd.DataFrame(rows)


def _run_fingerprint(run: RunData) -> Dict[str, object]:
    """Huella de los datos de una corrida para validar cachés (SHA-1 de las alturas)."""
    digest = hashlib.sha1(np.ascontiguousarray(run.heights, dtype=np.int64).tobytes())
    return {"source": run.source, "run_id": run.run_id, "shape": list(run.heights.shape),
            "sha1": digest.hexdigest()}


def compute_tables(runs: Sequence[RunData], cache_dir: Optional[PathLike] = None,
                   recompute: bool = False, verbose: bool = True,
                   **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Series por snapshot y tabla por nivel de todas las corridas, con caché opcional.

    Parámetros:
        cache_dir: carpeta donde guardar/leer `timeseries.csv`, `levels.csv` y
            `manifest.json`. La caché solo se reutiliza si coinciden la versión del
            cálculo, las opciones (`kwargs`) y la huella de cada corrida.
        kwargs: opciones de `run_timeseries` (with_alpha, with_adsorption, convention,
            every_n, half_layer)
    """
    settings = {"version": ANALYSIS_VERSION,
                **{k: list(v) if isinstance(v, tuple) else v for k, v in sorted(kwargs.items())}}
    manifest = {"settings": settings, "runs": [_run_fingerprint(r) for r in runs]}
    if cache_dir is not None and not recompute:
        cache = Path(cache_dir)
        try:
            if json.loads((cache / "manifest.json").read_text()) == manifest:
                ts = pd.read_csv(cache / "timeseries.csv")
                levels = pd.read_csv(cache / "levels.csv")
                if verbose:
                    print(f"Caché de observables leída: {cache} ({len(runs)} corridas)")
                return ts, levels
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    series, level_tables = [], []
    for i, run in enumerate(runs, start=1):
        ts, lv = run_timeseries(run, return_levels=True, **kwargs)
        series.append(ts)
        level_tables.append(lv)
        if verbose and (i % 10 == 0 or i == len(runs)):
            print(f"  observables: {i}/{len(runs)} corridas")
    ts = pd.concat(series, ignore_index=True)
    levels = pd.concat(level_tables, ignore_index=True)
    if cache_dir is not None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        ts.to_csv(cache / "timeseries.csv", index=False)
        levels.to_csv(cache / "levels.csv", index=False)
        (cache / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if verbose:
            print(f"Caché de observables guardada en {cache}")
    return ts, levels


# ---------------------------------------------------------------------------
# 4. Resumen por corrida en una ventana de espesor
# ---------------------------------------------------------------------------
# Observables por snapshot que se promedian en la ventana (los que existan)
WINDOW_COLUMNS: Tuple[str, ...] = (
    "W", "W_RD", "skewness", "kurtosis", "bimodality", "b_RD", "n_eff_levels", "n_eff_RD",
    "height_range", "growing_fraction",
    "n_islands", "island_density", "n_exposed_levels", "islands_per_level",
    "island_density_per_level", "island_density_active", "hole_density_active",
    "mean_island_size", "max_island_fraction",
    "step_density", "step_density_x", "step_density_y", "step_anisotropy", "mean_abs_dh",
    "f2", "f2_RD",
    "xi_x", "xi_y", "xi_r", "r_zero", "r_min", "c_min", "alpha_local",
    "site_frac_0", "site_frac_1", "site_frac_2", "site_frac_3", "site_frac_4",
    "p0", "p1", "p2",
)
# Columnas que pueden valer inf (ξ que no decae dentro de L/2, c(r) que no cruza 0)
_INF_COLUMNS = ("xi_x", "xi_y", "xi_r", "r_zero")


def _in_window(dh_grid: pd.Series, window: Window) -> np.ndarray:
    """dh_grid ∈ (window[0], window[1]] con tolerancia de redondeo."""
    d = dh_grid.to_numpy(dtype=float)
    return (d > window[0] + _EPS) & (d <= window[1] + _EPS)


def _finite_mean(values) -> float:
    """Media de los valores finitos (NaN si no hay ninguno)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def summarize_run_window(ts: pd.DataFrame, window: Window = (2.0, 6.0),
                         beta_range: Window = (1.0, 6.0),
                         columns: Sequence[str] = WINDOW_COLUMNS) -> Dict[str, float]:
    """Parámetros de orden de UNA corrida (su serie de `run_timeseries`).

    Promedios en la ventana de espesor `window` = (dh_min, dh_max] (sin estados
    repetidos):
        <col>: media de cada observable de `columns` (solo valores finitos);
        <col>_inf_frac: fracción de snapshots con ξ (o r_0) infinito.
    Parámetros de orden derivados:
        b_half: b promediado solo en los snapshots a media capa (`half_layer`). Con ⟨h⟩
            entero una superficie capa a capa está casi plana y b no discrimina; a media
            capa el capa a capa ideal da b = 1 exactamente.
        f2_rel: ⟨f≥2⟩ / ⟨f≥2_RD⟩ en la ventana (0 = sin escalones múltiples, 1 = RD).
        W_rel: ⟨W⟩ / ⟨W_RD⟩.
        psi_osc: amplitud de la oscilación de W con periodo una capa, normalizada por la
            del capa a capa ideal con el mismo muestreo (`ideal_layer_width`): 1 = capa
            a capa perfecto, 0 = sin oscilación.
        step_osc: amplitud de la oscilación de la densidad de escalones, relativa a su
            media (análogo de la intensidad RHEED).
        beta_h: exponente de W ∝ dh^β (dh en `beta_range`, W > 0).
        beta_t, growth_rate: W ∝ t^β y v = d⟨h⟩/dt en la ventana (si hay tiempos).
        des_per_ads, mig_per_ads, inc_per_ads: eventos de cada tipo por adsorción en la
            ventana (si la fuente guarda los contadores).
    """
    data = ts[~ts["repeated_state"].astype(bool)] if "repeated_state" in ts else ts
    v = data[_in_window(data["dh_grid"], window)]
    out: Dict[str, float] = {
        "n_window": int(len(v)),
        "dh_window_min": float(v["dh"].min()) if len(v) else np.nan,
        "dh_window_max": float(v["dh"].max()) if len(v) else np.nan,
        "dh_final": float(data["dh"].iloc[-1]) if len(data) else np.nan,
    }
    for col in columns:
        if col in v:
            out[col] = _finite_mean(v[col])
            if col in _INF_COLUMNS:
                vals = v[col].to_numpy(dtype=float)
                out[f"{col}_inf_frac"] = float(np.mean(np.isinf(vals))) if vals.size else np.nan

    half = v[v["half_layer"].astype(bool)] if "half_layer" in v else v.iloc[0:0]
    out["n_half"] = int(len(half))
    out["b_half"] = _finite_mean(half["bimodality"]) if len(half) else np.nan
    out["b_RD_half"] = _finite_mean(half["b_RD"]) if len(half) else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f2_rel"] = out.get("f2", np.nan) / out.get("f2_RD", np.nan)
        out["W_rel"] = out.get("W", np.nan) / out.get("W_RD", np.nan)

    # Ψ_osc: amplitud de W relativa a la del capa a capa ideal MUESTREADO IGUAL (mismos
    # dh), para que valga 1 exactamente en el capa a capa ideal con cualquier muestreo
    osc_W = layer_oscillation(data["dh_grid"], data["W"], window)
    osc_ideal = layer_oscillation(data["dh_grid"], ideal_layer_width(data["dh_grid"]), window)
    out["psi_osc"] = (osc_W["amplitude"] / osc_ideal["amplitude"]
                      if osc_ideal["amplitude"] and np.isfinite(osc_ideal["amplitude"]) else np.nan)
    osc_s = layer_oscillation(data["dh_grid"], data["step_density"], window)
    mean_s = out.get("step_density", np.nan)
    out["step_osc"] = osc_s["amplitude"] / mean_s if mean_s and np.isfinite(mean_s) else np.nan

    b = data[(data["dh_grid"] >= beta_range[0] - _EPS) & (data["dh_grid"] <= beta_range[1] + _EPS)]
    out["beta_h"] = growth_exponent(b["dh"], b["W"])
    has_t = len(v) >= 2 and np.all(np.isfinite(v["t"].to_numpy(dtype=float)))
    out["beta_t"] = growth_exponent(b["t"], b["W"]) if has_t else np.nan
    out["growth_rate"] = growth_rate(v["t"], v["mean_h"]) if has_t else np.nan

    # Reparto de eventos en la ventana (contadores acumulados del motor)
    if {"adsorption", "desorption", "migration", "incorporation"} <= set(v.columns) and len(v) >= 2:
        delta = {c: float(v[c].iloc[-1] - v[c].iloc[0])
                 for c in ("adsorption", "desorption", "migration", "incorporation")}
        ads = delta["adsorption"]
        for c, name in (("desorption", "des_per_ads"), ("migration", "mig_per_ads"),
                        ("incorporation", "inc_per_ads")):
            out[name] = delta[c] / ads if ads > 0 else np.nan
    return out


def saturation_island_density(levels: pd.DataFrame, coverage: Window = (0.1, 0.3),
                              window: Window = (2.0, 6.0)) -> pd.DataFrame:
    """Densidad de islas de saturación por corrida, n_sat.

    Media de la densidad de islas de cada nivel (islas / N) sobre los niveles con
    cobertura θ ∈ [coverage] y snapshots en la ventana de espesor. Es la medida
    estándar de la teoría de nucleación: antes de la coalescencia (θ ≲ 0.3), el número
    de islas de una capa ya no crece y fija la separación entre centros,
    ℓ_sat = n_sat^(−1/2). Devuelve una fila por (source, run_id).
    """
    lv = levels[~levels["repeated_state"].astype(bool)] if "repeated_state" in levels else levels
    sel = (lv["coverage"] >= coverage[0]) & (lv["coverage"] <= coverage[1])
    sel &= _in_window(lv["dh_grid"], window)
    g = lv[sel].groupby(["source", "run_id"], sort=False)
    out = g.agg(n_sat=("island_density", "mean"), n_sat_records=("island_density", "size"))
    out["ell_sat"] = out["n_sat"] ** -0.5
    return out.reset_index()


def summarize_runs(ts: pd.DataFrame, levels: Optional[pd.DataFrame] = None,
                   inventory: Optional[pd.DataFrame] = None, window: Window = (2.0, 6.0),
                   beta_range: Window = (1.0, 6.0),
                   coverage: Window = (0.1, 0.3)) -> pd.DataFrame:
    """`summarize_run_window` para todas las corridas de `ts` (una fila por corrida).

    Si se pasa `levels`, añade n_sat y ℓ_sat; si se pasa `inventory` (con `scan`, de
    `assign_scans`), añade el barrido y las condiciones de cada corrida.
    """
    rows = []
    for (source, run_id), g in ts.groupby(["source", "run_id"], sort=False):
        row = {"source": source, "run_id": run_id, "seed": g["seed"].iloc[0],
               "sigma": float(g["sigma"].iloc[0]), "S": float(g["S"].iloc[0])}
        row.update(summarize_run_window(g.sort_values("k"), window, beta_range))
        rows.append(row)
    per_run = pd.DataFrame(rows)
    if levels is not None and len(levels):
        per_run = per_run.merge(saturation_island_density(levels, coverage, window),
                                on=["source", "run_id"], how="left")
    if inventory is not None:
        keep = [c for c in inventory.columns if c not in per_run.columns or c in ("source", "run_id")]
        per_run = per_run.merge(inventory[keep], on=["source", "run_id"], how="left")
    return per_run


# ---------------------------------------------------------------------------
# 5. Promedio entre réplicas
# ---------------------------------------------------------------------------
_ID_COLUMNS = ("source", "run_id", "seed", "scan", "kind", "engine", "init_mode")


def aggregate_replicas(per_run: pd.DataFrame, by: Sequence[str] = ("scan", "sigma"),
                       columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Media, error estándar y desviación típica ENTRE réplicas para cada grupo.

    Devuelve una fila por grupo con <col> (media), <col>_sem, <col>_std y n_rep. Las
    réplicas son las corridas del grupo (semillas, o corridas con otra escala de tiempo
    pero la misma física). Con una sola réplica, sem y std quedan NaN.
    """
    by = list(by)
    if columns is None:
        columns = [c for c in per_run.columns
                   if c not in by and c not in _ID_COLUMNS
                   and pd.api.types.is_numeric_dtype(per_run[c])]
    rows = []
    for key, g in per_run.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n_rep"] = int(len(g))
        for c in columns:
            vals = g[c].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            row[c] = float(finite.mean()) if finite.size else np.nan
            row[f"{c}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
            row[f"{c}_sem"] = row[f"{c}_std"] / np.sqrt(finite.size) if finite.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Fronteras y clasificación
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    """Umbral de un parámetro de orden que marca una frontera.

    column: columna del resumen por corrida; threshold: valor del umbral;
    rising: True si el parámetro SUBE al cruzar la frontera al aumentar σ;
    boundary: frontera que marca ("1" o "2"); label: texto para tablas.
    """

    column: str
    threshold: float
    rising: bool
    boundary: str
    label: str = ""


# Umbrales por defecto (convenciones, ver notebooks/transiciones_sobresaturacion.ipynb):
# - frontera 1 (un centro → nucleación 2D): más de dos islas por nivel expuesto;
# - frontera 2 (capa a capa → rugoso): b_½ deja de ser bimodal (Sarle 5/9), aparecen
#   escalones múltiples (10 % de la cota RD) y la oscilación por capa cae a la mitad.
DEFAULT_CRITERIA: Tuple[Criterion, ...] = (
    Criterion("islands_per_level", 2.0, True, "1", "N_capa = 2"),
    Criterion("b_half", B_SARLE, False, "2", "b_½ = 5/9"),
    Criterion("f2_rel", 0.10, True, "2", "f≥2 = 10 % RD"),
    Criterion("psi_osc", 0.50, False, "2", "Ψ_osc = 1/2"),
)


def crossing_sigma(sigmas, y, threshold: float, rising: bool) -> float:
    """σ del PRIMER cruce de y(σ) con el umbral, interpolando linealmente en ln σ.

    `rising=True` busca el paso de y < umbral a y ≥ umbral (al revés si es False). Los
    puntos con y no finita se ignoran. NaN si no cruza dentro del barrido (o si ya
    empieza al otro lado del umbral).
    """
    s = np.asarray(sigmas, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y) & (s > 0)
    order = np.argsort(s[ok])
    s, y = np.log(s[ok][order]), y[ok][order]
    side = (y >= threshold) if rising else (y <= threshold)
    for i in range(1, len(y)):
        if side[i] and not side[i - 1]:
            x = s[i - 1] + (threshold - y[i - 1]) * (s[i] - s[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(x))
    return float("nan")


def steepest_change(sigmas, y, rising: Optional[bool] = None,
                    log_y: bool = False) -> Dict[str, float]:
    """σ* donde y(σ) cambia más deprisa (máximo de |dy/d ln σ|), sin umbral.

    Derivada por diferencias finitas centradas en ln σ (válidas con espaciado no
    uniforme) y refinamiento parabólico del máximo con sus dos vecinos. `rising`
    restringe el signo (True: el máximo de dy/d ln σ; False: el mínimo). Con
    `log_y=True` se deriva ln y (para magnitudes que cambian en órdenes de magnitud).
    Devuelve sigma_star y slope (la derivada en el máximo); NaN con < 3 puntos.
    """
    s = np.asarray(sigmas, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y) & (s > 0)
    if log_y:
        ok &= y > 0
    if np.count_nonzero(ok) < 3:
        return {"sigma_star": np.nan, "slope": np.nan}
    order = np.argsort(s[ok])
    x = np.log(s[ok][order])
    yy = np.log(y[ok][order]) if log_y else y[ok][order]
    g = np.gradient(yy, x)
    score = g if rising is True else (-g if rising is False else np.abs(g))
    i = int(np.argmax(score))
    x_star = x[i]
    if 0 < i < len(x) - 1:
        # Vértice de la parábola que pasa por (x_{i-1}, x_i, x_{i+1}) del score
        a, b, _ = np.polyfit(x[i - 1:i + 2], score[i - 1:i + 2], 2)
        if a < 0:
            x_star = float(np.clip(-b / (2 * a), x[i - 1], x[i + 1]))
    return {"sigma_star": float(np.exp(x_star)), "slope": float(g[i])}


def transition_table(per_run: pd.DataFrame,
                     criteria: Sequence[Criterion] = DEFAULT_CRITERIA,
                     sigma_col: str = "sigma", replica_col: Optional[str] = "seed") -> pd.DataFrame:
    """σ* de cada criterio para UN barrido (filtra antes por `scan`).

    Columnas:
        sigma_star_curve: cruce de la curva media (media de réplicas en cada σ);
        sigma_star_rep_mean / _rep_sem / sigma_star_reps: cruce de cada réplica (solo
            si cada réplica cubre el barrido completo, p. ej. una semilla por σ);
        sigma_star_steepest: punto de cambio más rápido de la curva media (sin umbral);
        y_low / y_high: valor del parámetro en el σ más bajo y más alto del barrido.
    """
    rows = []
    for c in criteria:
        if c.column not in per_run:
            continue
        mean = per_run.groupby(sigma_col)[c.column].mean()
        reps = []
        if replica_col is not None and replica_col in per_run:
            for _, g in per_run.groupby(replica_col):
                # Una réplica es una curva y(σ): exige σ únicos y al menos 3 puntos
                if g[sigma_col].is_unique and g[sigma_col].nunique() >= 3:
                    reps.append(crossing_sigma(g[sigma_col], g[c.column], c.threshold, c.rising))
        reps_arr = np.array(reps, dtype=float)
        finite = reps_arr[np.isfinite(reps_arr)]
        steep = steepest_change(mean.index, mean.values, rising=c.rising)
        rows.append({
            "boundary": c.boundary, "criterion": c.label or c.column, "column": c.column,
            "threshold": c.threshold,
            "sigma_star_curve": crossing_sigma(mean.index, mean.values, c.threshold, c.rising),
            "sigma_star_rep_mean": float(finite.mean()) if finite.size else np.nan,
            "sigma_star_rep_sem": (float(finite.std(ddof=1) / np.sqrt(finite.size))
                                   if finite.size > 1 else np.nan),
            "sigma_star_reps": ", ".join(f"{x:.3g}" for x in reps_arr),
            "sigma_star_steepest": steep["sigma_star"],
            "y_low": float(mean.iloc[0]) if len(mean) else np.nan,
            "y_high": float(mean.iloc[-1]) if len(mean) else np.nan,
        })
    return pd.DataFrame(rows)


def boundaries(transitions: pd.DataFrame, use: str = "sigma_star_curve") -> Dict[str, float]:
    """σ₁ y σ₂ de una tabla de `transition_table`: media de los criterios de cada frontera."""
    out = {}
    for b in ("1", "2"):
        vals = transitions.loc[transitions["boundary"] == b, use].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"sigma_{b}"] = float(vals.mean()) if vals.size else np.nan
    return out


REGIME_LABELS: Dict[str, str] = {
    "I": "I · un centro dominante",
    "II": "II · nucleación 2D, capa a capa",
    "III": "III · rugoso",
    "≠": "transición (b_½ y f≥2 discrepan)",
    "?": "sin datos suficientes",
}


def classify_regime(islands_per_level: float, b_half: float, f2_rel: float,
                    n_threshold: float = 2.0, b_threshold: float = B_SARLE,
                    f2_threshold: float = 0.10) -> str:
    """Régimen de una corrida o de una media por σ con tres parámetros de orden.

    - b_½ ≥ b_threshold y f≥2/f≥2_RD < f2_threshold (superficie de dos niveles):
      "I" si hay menos de `n_threshold` islas por nivel, "II" si no;
    - b_½ < b_threshold y f≥2/f≥2_RD ≥ f2_threshold: "III";
    - si b_½ y f≥2 no coinciden: "≠" (zona de transición, no un error);
    - "?" si falta alguno de los valores.
    """
    vals = (islands_per_level, b_half, f2_rel)
    if any(v is None or not np.isfinite(v) for v in vals):
        return "?"
    smooth_b = b_half >= b_threshold
    smooth_f = f2_rel < f2_threshold
    if smooth_b and smooth_f:
        return "I" if islands_per_level < n_threshold else "II"
    if not smooth_b and not smooth_f:
        return "III"
    return "≠"


def add_regimes(df: pd.DataFrame, col: str = "regime") -> pd.DataFrame:
    """Añade la columna de régimen (`classify_regime`) a una tabla por corrida o por σ."""
    out = df.copy()
    out[col] = [classify_regime(*x) for x in
                out[["islands_per_level", "b_half", "f2_rel"]].to_numpy(dtype=float)]
    return out


# Catálogo de los parámetros de orden del resumen por corrida: qué miden, con qué
# función se calculan, qué frontera marcan y qué valor se espera en cada régimen
# (firmas de Plan_observables_regimenes.md §1; no son umbrales calibrados).
_CATALOG: Tuple[Tuple[str, ...], ...] = (
    # (columna, símbolo, qué mide, función de origen, frontera, I, II, III)
    ("islands_per_level", "N_capa", "islas por nivel expuesto",
     "islands.island_density_observables", "1", "≈ 1", "2–7", "crece por más niveles"),
    ("n_sat", "n_sat", "islas/sitio de una capa con θ ∈ [0.1, 0.3] (antes de coalescer)",
     "islands.level_island_statistics → saturation_island_density", "1",
     "1/N (limitada por L)", "crece con σ", "sigue creciendo"),
    ("ell_sat", "ℓ_sat", "separación entre centros, n_sat^(−1/2) (sitios)",
     "saturation_island_density", "1", "≳ L", "< L", "pocos sitios"),
    ("island_density", "n_I", "islas por sitio sumadas sobre todos los niveles",
     "islands.island_statistics", "1 y 2", "≈ 1/N", "crece", "crece (más niveles)"),
    ("n_exposed_levels", "N_niv", "niveles parcialmente cubiertos (h_max − h_min)",
     "islands.island_density_observables", "2", "≈ 1", "1–3", "≳ 5"),
    ("b_half", "b_½", "bimodalidad de Sarle a media capa",
     "moments.height_moments + summarize_run_window", "2", "≈ 1", "≥ 5/9", "≈ b_RD (≤ 1/3 con fosos)"),
    ("psi_osc", "Ψ_osc", "amplitud de la oscilación de W por capa (1 = capa a capa ideal)",
     "layer_oscillation", "2", "≈ 1", "1 → 1/2 (se amortigua)", "≈ 0"),
    ("step_osc", "A_esc", "oscilación relativa de la densidad de escalones (análogo RHEED)",
     "layer_oscillation", "2", "grande", "decrece", "≈ 0"),
    ("f2_rel", "f≥2/f≥2_RD", "escalones de ≥ 2 capas relativos a la deposición aleatoria",
     "islands.step_statistics + rd_multi_step_fraction", "2", "0", "< 0.1", "→ 0.7–0.8"),
    ("beta_h", "β", "exponente de crecimiento W ∝ ⟨h⟩^β", "regime.growth_exponent", "2",
     "≈ 0", "≈ 0–0.1", "→ 1/2 (RD)"),
    ("W_rel", "W/W_RD", "rugosidad relativa a la deposición aleatoria", "rd_width", "2",
     "pequeña", "pequeña", "→ 0.7"),
    ("n_eff_levels", "n_eff", "niveles efectivos exp(H)", "moments.effective_levels", "2",
     "≤ 2", "≤ 2", "crece ∝ W"),
    ("skewness", "γ₃", "asimetría de la distribución de alturas", "moments.height_moments",
     "control", "oscila con θ", "oscila con θ", "< 0 con fosos"),
    ("kurtosis", "κ", "curtosis de Pearson", "moments.height_moments", "control",
     "—", "—", "→ 3 (gaussiana)"),
    ("xi_r", "ξ_r", "longitud de correlación radial (1/e)", "correlation.correlation_lengths",
     "1 y 2", "∼ tamaño de la isla", "∼ radio de isla", "1–2 sitios"),
    ("c_min", "c_min", "profundidad de la anticorrelación de alturas",
     "correlation.correlation_extrema", "2", "marcada", "marcada", "≈ 0"),
    ("step_density", "ρ_esc", "fracción de enlaces con Δh ≠ 0", "islands.step_statistics",
     "2", "baja", "oscila", "alta"),
    ("step_anisotropy", "ρ_x/ρ_y", "anisotropía de escalones (líneas aniso)",
     "islands.step_statistics", "—", "—", "—", "—"),
    ("site_frac_2", "m_kink", "fracción de sitios kink (2 enlaces laterales)",
     "adsorption.site_class_counts", "—", "—", "—", "—"),
    ("p0", "P_adatom", "probabilidad de adsorción en terraza (Fig. 8)",
     "adsorption.adsorption_probabilities", "Fig. 8", "baja", "baja", "crece"),
    ("p1", "P_step", "probabilidad de adsorción en borde de escalón",
     "adsorption.adsorption_probabilities", "Fig. 8", "baja", "crece", "crece"),
    ("p2", "P_kink", "probabilidad de adsorción en kink",
     "adsorption.adsorption_probabilities", "Fig. 8", "domina", "domina", "decrece"),
    ("growth_rate", "v", "velocidad d⟨h⟩/dt en la ventana (anomalía δ/S, README §8.4)",
     "regime.growth_rate", "—", "—", "—", "—"),
    ("des_per_ads", "n_des/n_ads", "desorciones por adsorción (cercanía al equilibrio)",
     "summarize_run_window", "—", "—", "—", "—"),
    ("mig_per_ads", "n_mig/n_ads", "migraciones por adsorción (relajación superficial)",
     "summarize_run_window", "—", "—", "—", "—"),
)


def order_parameter_catalog() -> pd.DataFrame:
    """Tabla con los parámetros de orden del resumen por corrida (ver `_CATALOG`)."""
    return pd.DataFrame(list(_CATALOG), columns=[
        "columna", "símbolo", "qué mide", "función", "frontera", "I (un centro)",
        "II (nucleación 2D)", "III (rugoso)"])


# ---------------------------------------------------------------------------
# 7. Flujo completo
# ---------------------------------------------------------------------------
@dataclass
class TransitionAnalysis:
    """Resultado de `analyze_transitions` (todas las tablas son DataFrames).

    inventory: una fila por corrida (condiciones y `scan`)
    timeseries: una fila por snapshot (todos los observables)
    levels: una fila por (snapshot, nivel expuesto)
    per_run: parámetros de orden por corrida en la ventana, con régimen
    per_sigma: media entre réplicas por (scan, σ), con régimen
    transitions: σ* de cada criterio y barrido
    settings: ventanas y opciones usadas
    """

    inventory: pd.DataFrame
    timeseries: pd.DataFrame
    levels: pd.DataFrame
    per_run: pd.DataFrame
    per_sigma: pd.DataFrame
    transitions: pd.DataFrame
    settings: Dict[str, object] = field(default_factory=dict)

    def scan(self, name: str) -> Dict[str, pd.DataFrame]:
        """Las tablas de un solo barrido."""
        return {k: getattr(self, k)[getattr(self, k)["scan"] == name]
                for k in ("per_run", "per_sigma", "transitions")}


def analyze_transitions(runs: Sequence[RunData], window: Window = (2.0, 6.0),
                        beta_range: Window = (1.0, 6.0), coverage: Window = (0.1, 0.3),
                        criteria: Sequence[Criterion] = DEFAULT_CRITERIA,
                        cache_dir: Optional[PathLike] = None, recompute: bool = False,
                        min_sigmas: int = 3, verbose: bool = True,
                        **timeseries_kwargs) -> TransitionAnalysis:
    """Análisis completo de transiciones para cualquier conjunto de corridas.

    1. inventario y agrupación por barrido (`assign_scans`: misma física y red);
    2. observables por snapshot y por nivel (`compute_tables`, con caché);
    3. parámetros de orden por corrida en la ventana (`summarize_runs`);
    4. media entre réplicas por (barrido, σ) (`aggregate_replicas`);
    5. fronteras de cada barrido con al menos `min_sigmas` valores de σ
       (`transition_table`) y régimen de cada corrida y de cada σ (`add_regimes`).
    """
    inventory = assign_scans(runs_inventory(runs))
    ts, levels = compute_tables(runs, cache_dir=cache_dir, recompute=recompute,
                                verbose=verbose, **timeseries_kwargs)
    per_run = add_regimes(summarize_runs(ts, levels, inventory, window, beta_range, coverage))
    per_sigma = add_regimes(aggregate_replicas(per_run, by=("scan", "sigma")))
    tables = []
    for name, g in per_run.groupby("scan", sort=True):
        if g["sigma"].nunique() >= min_sigmas:
            tables.append(transition_table(g, criteria).assign(scan=name))
    transitions = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    settings = {"window": window, "beta_range": beta_range, "coverage": coverage,
                "criteria": [c.__dict__ for c in criteria], **timeseries_kwargs}
    return TransitionAnalysis(inventory, ts, levels, per_run, per_sigma, transitions, settings)


# ---------------------------------------------------------------------------
# 8. Ayudas de figura (paleta validada del proyecto)
# ---------------------------------------------------------------------------
SERIES_COLORS: Tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")   # azul, naranja, aqua
REFERENCE_COLOR = "#8a8984"      # referencias teóricas (recesivas)
INK_SECONDARY = "#52514e"        # texto secundario
REGIME_FILLS = ("#f3f2ee", "#ffffff", "#ecebe7")   # bandas I, II, III (neutras)
# σ es una magnitud ordenada: rampa secuencial de un solo tono (claro = σ bajo)
SIGMA_CMAP = LinearSegmentedColormap.from_list(
    "rampa_sigma", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def sigma_color(sigma: float, sigmas: Sequence[float]) -> Tuple[float, ...]:
    """Color de la rampa secuencial para σ, en escala logarítmica dentro de `sigmas`."""
    lo, hi = np.log(min(sigmas)), np.log(max(sigmas))
    return SIGMA_CMAP(0.0 if hi == lo else (np.log(sigma) - lo) / (hi - lo))


def format_sigma_axis(ax, sigmas: Sequence[float], label: str = "σ") -> None:
    """Eje x logarítmico de σ con ticks 1-2-3-5 legibles."""
    sig = np.asarray([s for s in sigmas if s > 0], dtype=float)
    ax.set_xscale("log")
    ax.set_xlim(sig.min() / 1.15, sig.max() * 1.15)
    ax.xaxis.set_major_locator(FixedLocator(_nice_log_ticks(sig)))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(label)


def shade_regimes(ax, sigma_1: float, sigma_2: float, sigmas: Sequence[float],
                  labels: Sequence[str] = ("I", "II", "III")) -> None:
    """Bandas neutras de los regímenes I, II, III y líneas en σ₁, σ₂.

    Llamar antes de dibujar los datos para que las bandas queden debajo.
    """
    lo, hi = min(sigmas) / 1.15, max(sigmas) * 1.15
    edges = [lo, sigma_1 if np.isfinite(sigma_1) else lo, sigma_2 if np.isfinite(sigma_2) else hi, hi]
    for x0, x1, color, text in zip(edges[:-1], edges[1:], REGIME_FILLS, labels):
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, color=color, zorder=-2, lw=0)
        ax.annotate(text, (np.sqrt(x0 * x1), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY)
    for x in (sigma_1, sigma_2):
        if np.isfinite(x):
            ax.axvline(x, color=REFERENCE_COLOR, lw=0.8, ls=":", zorder=-1)


def reference_line(ax, y: float, text: str, below: bool = False) -> None:
    """Línea horizontal de referencia con su etiqueta directa a la derecha."""
    ax.axhline(y, color=REFERENCE_COLOR, lw=0.8, ls="--", zorder=0)
    ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-2, -3 if below else 3), textcoords="offset points", ha="right",
                va="top" if below else "bottom", color=INK_SECONDARY, fontsize=8)


def plot_order_parameter(ax, per_sigma: pd.DataFrame, column: str, err: str = "sem",
                         color: str = SERIES_COLORS[0], label: Optional[str] = None,
                         sigma_col: str = "sigma", **kwargs) -> None:
    """Media entre réplicas de un parámetro de orden frente a σ, con barras de error.

    `err` = "sem" (error estándar) o "std" (dispersión entre réplicas); sin réplicas
    se dibuja sin barras.
    """
    d = per_sigma.sort_values(sigma_col)
    y = d[column].to_numpy(dtype=float)
    e = d.get(f"{column}_{err}")
    yerr = None if e is None or not np.any(np.isfinite(e)) else np.nan_to_num(e.to_numpy(float))
    style = {"fmt": "o-", "ms": 4, "lw": 1.4, "capsize": 2, "color": color}
    style.update(kwargs)
    ax.errorbar(d[sigma_col], y, yerr=yerr, label=label, **style)
