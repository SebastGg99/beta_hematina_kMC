"""Carga unificada de corridas para el análisis de transiciones frente a σ.

Todas las fuentes de datos del repositorio se convierten en un mismo objeto `RunData`
(alturas apiladas + tiempos + condiciones), de modo que el resto del análisis
(`src/analysis/transitions.py`) no depende del formato en que se guardó la corrida:

| Formato                                        | Función               | Origen                                   |
|------------------------------------------------|-----------------------|------------------------------------------|
| carpeta con `metadata.json` + `snaps.pkl`      | `load_run_folder`     | `results/`, `scripts/run_sigma_scan.py`  |
| caché `alturas.npz` + `diseno.json` (+ CSV)    | `load_heights_cache`  | `notebooks/metricas_regimen_sigma.ipynb` |
| snapshots en memoria `(t, heights, conv)`      | `from_snapshots`      | `run()` de cualquiera de las dos líneas  |
| cualquier mezcla de carpetas                   | `discover_runs`       | búsqueda recursiva                        |

No se modifica ningún archivo de entrada: solo se lee.
"""

from dataclasses import dataclass, field
import json
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Columnas que fijan la estadística de la morfología a igual espesor depositado.
# K0_plus, K_inc_plus y time_scale NO están: en la línea estática solo reescalan el
# tiempo (las probabilidades de los eventos que cambian `heights` no dependen de ellos;
# ver notebooks/metricas_regimen_sigma.ipynb §3). El tamaño de red sí cuenta, porque
# fija qué cabe en la red (frontera "un centro").
PHYSICS_COLUMNS: Tuple[str, ...] = (
    "engine", "E_pb_over_kT_x", "E_pb_over_kT_y", "phi_over_kT", "delta_x", "delta_y",
    "init_mode", "size_x", "size_y",
)
# Condiciones que solo cambian la escala de tiempo (se guardan, pero no agrupan)
KINETIC_COLUMNS: Tuple[str, ...] = ("K0_plus", "K_inc_plus", "time_scale")

# Claves de alturas.npz: sigma_<σ>_seed_<semilla> (mismo formato que adsorption.py)
_CACHE_KEY = re.compile(r"^sigma_(?P<sigma>[-+0-9.eE]+)_seed_(?P<seed>-?\d+)$")
# Columnas por snapshot que se copian del CSV de una caché (si existe)
_CACHE_INFO_COLUMNS = ("k", "dh_objetivo", "t", "eventos", "adsorption", "desorption",
                       "migration", "incorporation")


@dataclass
class RunData:
    """Una corrida lista para el análisis: alturas por snapshot y sus condiciones.

    Atributos:
        run_id: identificador de la corrida dentro de su fuente
        source: nombre de la fuente (carpeta de results/ o carpeta de la caché)
        sigma: sobresaturación σ = C/C_eq − 1 que usa el motor (fija en toda la corrida)
        heights: alturas enteras, forma (n_snapshots, Lx, Ly)
        t: tiempo de cada snapshot (NaN si la fuente no lo guarda)
        seed: semilla del RNG del motor (réplica), si se conoce
        engine: "static" o "dynamic"
        params: parámetros del modelo (dict), necesarios para las probabilidades de sitio
        conditions: condiciones de la corrida (PHYSICS_COLUMNS + KINETIC_COLUMNS + kind)
        snapshot_info: columnas extra por snapshot (p. ej. dh_objetivo, contadores de
            eventos), una fila por snapshot en el mismo orden que `heights`
        path: carpeta de origen
    """

    run_id: str
    source: str
    sigma: float
    heights: np.ndarray
    t: np.ndarray
    seed: Optional[int] = None
    engine: str = "static"
    params: Dict[str, object] = field(default_factory=dict)
    conditions: Dict[str, object] = field(default_factory=dict)
    snapshot_info: Optional[pd.DataFrame] = None
    path: Optional[str] = None

    def __post_init__(self) -> None:
        # σ con 10 cifras significativas: algunos metadata.json antiguos guardan σ con
        # ruido de coma flotante (0.6000000000000001), que partiría un mismo σ en dos
        if np.isfinite(self.sigma):
            self.sigma = float(f"{float(self.sigma):.10g}")
        # Normaliza tipos y comprueba que las formas son coherentes
        self.heights = np.asarray(self.heights)
        if self.heights.ndim != 3:
            raise ValueError(f"{self.run_id}: heights debe ser (n, Lx, Ly), no {self.heights.shape}")
        self.t = np.asarray(self.t, dtype=float)
        if self.t.shape != (self.heights.shape[0],):
            raise ValueError(f"{self.run_id}: t tiene {self.t.size} valores para "
                             f"{self.heights.shape[0]} snapshots")
        if self.snapshot_info is not None and len(self.snapshot_info) != self.heights.shape[0]:
            raise ValueError(f"{self.run_id}: snapshot_info no tiene una fila por snapshot")

    @property
    def n_snapshots(self) -> int:
        """Número de snapshots."""
        return int(self.heights.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        """Tamaño de la red (Lx, Ly)."""
        return int(self.heights.shape[1]), int(self.heights.shape[2])

    @property
    def S(self) -> float:
        """S = ln(1 + σ), la variable que usa el motor en sus tasas."""
        return float(np.log1p(self.sigma))

    @property
    def mean_h(self) -> np.ndarray:
        """⟨h⟩ de cada snapshot."""
        return self.heights.reshape(self.n_snapshots, -1).mean(axis=1)

    @property
    def dh(self) -> np.ndarray:
        """Espesor depositado respecto al primer snapshot, ⟨h⟩_k − ⟨h⟩_0 (capas)."""
        m = self.mean_h
        return m - m[0]

    @property
    def label(self) -> str:
        """Etiqueta legible fuente/corrida."""
        return f"{self.source}/{self.run_id}"


# ---------------------------------------------------------------------------
# Condiciones
# ---------------------------------------------------------------------------
def _conditions(params: Mapping, engine: str, size: Sequence[int], init_mode: Optional[str],
                time_scale: Optional[float], kind: str) -> Dict[str, object]:
    """Diccionario de condiciones con las claves de PHYSICS_COLUMNS y KINETIC_COLUMNS.

    Las corridas antiguas isotrópicas pueden tener `E_pb_over_kT`/`delta` sin sufijo:
    se copian a x e y.
    """
    def pick(*keys):
        for k in keys:
            if k in params and params[k] is not None:
                return float(params[k])
        return np.nan

    return {
        "kind": kind,
        "engine": engine,
        "E_pb_over_kT_x": pick("E_pb_over_kT_x", "E_pb_over_kT"),
        "E_pb_over_kT_y": pick("E_pb_over_kT_y", "E_pb_over_kT"),
        "phi_over_kT": pick("phi_over_kT"),
        "delta_x": pick("delta_x", "delta"),
        "delta_y": pick("delta_y", "delta"),
        "init_mode": init_mode if init_mode is not None else "desconocido",
        "size_x": int(size[0]),
        "size_y": int(size[1]),
        "K0_plus": pick("K0_plus"),
        "K_inc_plus": pick("K_inc_plus"),
        "time_scale": float(time_scale) if time_scale is not None else np.nan,
    }


def _sigma_from(engine: str, fixed_sigma) -> float:
    """σ de la corrida a partir de `fixed_sigma`.

    En la línea estática `fixed_sigma` es σ. En la dinámica es S = ln(1 + σ)
    (`src/dynamic/params.py`), así que σ = e^S − 1. Sin σ fija (reservorio que se
    agota) no hay una σ única: NaN.
    """
    if fixed_sigma is None:
        return float("nan")
    return float(np.expm1(float(fixed_sigma))) if engine == "dynamic" else float(fixed_sigma)


# ---------------------------------------------------------------------------
# Cargadores
# ---------------------------------------------------------------------------
def _effective_configs(run_dir: Path, configs_dir: Optional[PathLike]) -> Dict[str, Dict]:
    """Configs reconstruidos `scripts/configs/results/<carpeta>.json` por run_id.

    Dan el modo de inicio EFECTIVO de las corridas antiguas de results/, que no siempre
    coincide con metadata.json (README §4.2). Devuelve {} si no existe.
    """
    if configs_dir is None:
        from ..common.io import find_repo_root

        root = find_repo_root(run_dir)
        if root is None:
            return {}
        configs_dir = root / "scripts" / "configs" / "results"
    cfg_path = Path(configs_dir) / f"{run_dir.parent.name}.json"
    if not cfg_path.exists():
        return {}
    cfg = json.loads(cfg_path.read_text())
    return {r["run_id"]: {**r, "engine": cfg.get("engine", "static")} for r in cfg.get("runs", [])}


def load_run_folder(run_dir: PathLike, configs_dir: Optional[PathLike] = None) -> RunData:
    """Carga una carpeta de corrida (`metadata.json` + `snaps.pkl`).

    Parámetros:
        run_dir: carpeta de la corrida
        configs_dir: carpeta de configs reconstruidos (por defecto
            `scripts/configs/results/` del repositorio); de ahí sale el modo de
            inicio efectivo si existe
    """
    from ..common.io import load_json, load_pickle   # import local: E/S del proyecto

    run_dir = Path(run_dir)
    meta = load_json(run_dir / "metadata.json")
    snaps = load_pickle(run_dir / "snaps.pkl")
    run_id = meta.get("run_id", run_dir.name)
    eff = _effective_configs(run_dir, configs_dir).get(run_id, {})

    engine = meta.get("engine", eff.get("engine", "static"))
    params = dict(meta.get("params", {}))
    fixed = meta.get("fixed_sigma", params.get("fixed_sigma"))
    heights = np.stack([np.asarray(item[1]) for item in snaps])
    times = np.array([float(item[0]) for item in snaps])
    conversion = np.array([float(item[2]) if len(item) > 2 else np.nan for item in snaps])
    size = meta.get("size", list(heights.shape[1:]))
    return RunData(
        run_id=run_id,
        source=run_dir.parent.name,
        sigma=_sigma_from(engine, fixed),
        heights=heights,
        t=times,
        seed=meta.get("rng_seed"),
        engine=engine,
        params=params,
        # Inicio efectivo (config reconstruido) si existe; si no, el de metadata.json
        conditions=_conditions(params, engine, size, eff.get("init_mode", meta.get("init_mode")),
                               meta.get("time_scale"), kind="results"),
        snapshot_info=pd.DataFrame({"conversion": conversion}),
        path=str(run_dir),
    )


def load_heights_cache(cache_dir: PathLike) -> List[RunData]:
    """Carga una caché de barrido a espesor controlado: una `RunData` por (σ, semilla).

    Lee `alturas.npz` (claves `sigma_<σ>_seed_<semilla>`, arrays (n, L, L)) y
    `diseno.json` (parámetros, L, time_scale, init_mode). Si existe
    `serie_espesor.csv`, toma de ahí el tiempo, el espesor objetivo `dh_objetivo` y
    los contadores de eventos de cada snapshot.
    """
    cache_dir = Path(cache_dir)
    design = json.loads((cache_dir / "diseno.json").read_text())
    params = dict(design.get("params", {}))
    engine = design.get("engine", "static")
    serie_path = cache_dir / "serie_espesor.csv"
    serie = pd.read_csv(serie_path) if serie_path.exists() else None
    seed_col = None
    if serie is not None:
        seed_col = "semilla" if "semilla" in serie else ("seed" if "seed" in serie else None)

    runs: List[RunData] = []
    with np.load(cache_dir / "alturas.npz") as z:
        for key in z.files:
            m = _CACHE_KEY.match(key)
            if m is None:
                warnings.warn(f"{cache_dir.name}: clave no reconocida en alturas.npz: {key}")
                continue
            sigma, seed = float(m["sigma"]), int(m["seed"])
            heights = z[key]
            n = heights.shape[0]
            info = None
            times = np.full(n, np.nan)
            if serie is not None and seed_col is not None:
                rows = serie[np.isclose(serie["sigma"], sigma) & (serie[seed_col] == seed)]
                rows = rows.sort_values("k") if "k" in rows else rows
                if len(rows) == n:
                    cols = [c for c in _CACHE_INFO_COLUMNS if c in rows]
                    info = rows[cols].reset_index(drop=True)
                    if "t" in info:
                        times = info["t"].to_numpy(dtype=float)
                else:
                    warnings.warn(f"{cache_dir.name}/{key}: el CSV no tiene una fila por "
                                  f"snapshot ({len(rows)} != {n}); se ignora")
            size = (design.get("L", heights.shape[1]), design.get("L", heights.shape[2]))
            runs.append(RunData(
                run_id=key,
                source=cache_dir.name,
                sigma=sigma,   # las claves del npz guardan σ (no S) en las dos líneas
                heights=heights,
                t=times,
                seed=seed,
                engine=engine,
                params=params,
                conditions=_conditions(params, engine, size, design.get("init_mode"),
                                       design.get("time_scale"), kind="cache"),
                snapshot_info=info,
                path=str(cache_dir),
            ))
    # Orden estable: σ y luego semilla
    return sorted(runs, key=lambda r: (r.sigma, r.seed if r.seed is not None else -1))


def from_snapshots(snaps, sigma: float, run_id: str = "corrida", source: str = "memoria",
                   seed: Optional[int] = None, engine: str = "static",
                   params: Optional[Mapping] = None, init_mode: Optional[str] = None,
                   time_scale: Optional[float] = None) -> RunData:
    """`RunData` a partir de snapshots en memoria.

    Parámetros:
        snaps: lista `(t, heights, conversion)` que devuelve `run()`, o array (n, Lx, Ly)
        sigma: σ de la corrida (σ = C/C_eq − 1, también en la línea dinámica)
        params: parámetros del modelo (dict o dataclass) para las probabilidades de sitio
    """
    if isinstance(snaps, np.ndarray) and snaps.ndim == 3:
        heights, times = snaps, np.full(snaps.shape[0], np.nan)
    else:
        heights = np.stack([np.asarray(item[1]) for item in snaps])
        times = np.array([float(item[0]) for item in snaps])
    if params is not None and not isinstance(params, Mapping):
        params = dict(vars(params))   # dataclass de parámetros -> dict
    params = dict(params or {})
    return RunData(run_id=run_id, source=source, sigma=float(sigma), heights=heights, t=times,
                   seed=seed, engine=engine, params=params,
                   conditions=_conditions(params, engine, heights.shape[1:], init_mode,
                                          time_scale, kind="memoria"))


def discover_runs(paths: Union[PathLike, Sequence[PathLike]],
                  configs_dir: Optional[PathLike] = None) -> List[RunData]:
    """Busca y carga todas las corridas bajo `paths` (a cualquier profundidad).

    Reconoce cachés (`alturas.npz` + `diseno.json`) y carpetas de corrida
    (`metadata.json` + `snaps.pkl`). Las que no se pueden leer se saltan con aviso.
    """
    from .adsorption import find_run_dirs   # misma búsqueda que el análisis de sitios

    if isinstance(paths, (str, Path)):
        paths = [paths]
    runs: List[RunData] = []
    seen = set()
    for base in map(Path, paths):
        # Cachés de barridos a espesor controlado
        for design in sorted(base.rglob("diseno.json")) if base.is_dir() else []:
            cache = design.parent
            if (cache / "alturas.npz").exists() and cache.resolve() not in seen:
                seen.add(cache.resolve())
                try:
                    runs.extend(load_heights_cache(cache))
                except (OSError, ValueError, KeyError) as exc:
                    warnings.warn(f"Se omite la caché {cache}: {exc}")
        # Carpetas de corrida con snapshots
        for run_dir in find_run_dirs(base):
            if (run_dir / "snaps.pkl").exists() and run_dir.resolve() not in seen:
                seen.add(run_dir.resolve())
                try:
                    runs.append(load_run_folder(run_dir, configs_dir))
                except (OSError, ValueError, KeyError, EOFError) as exc:
                    warnings.warn(f"Se omite {run_dir}: {exc}")
    if not runs:
        warnings.warn(f"No se encontraron corridas en {list(map(str, paths))}")
    return runs


# ---------------------------------------------------------------------------
# Inventario y agrupación por condiciones
# ---------------------------------------------------------------------------
def runs_inventory(runs: Iterable[RunData]) -> pd.DataFrame:
    """Una fila por corrida: identificación, σ, snapshots, espesor alcanzado y condiciones."""
    rows = []
    for r in runs:
        dh = r.dh
        rows.append({
            "source": r.source, "run_id": r.run_id, "seed": r.seed, "sigma": r.sigma,
            "S": r.S, "n_snapshots": r.n_snapshots, "dh_final": float(dh[-1]),
            # Paso medio de espesor entre snapshots: decide qué análisis por capa caben
            "dh_step": float(np.mean(np.diff(dh))) if r.n_snapshots > 1 else np.nan,
            "t_final": float(r.t[-1]) if np.isfinite(r.t[-1]) else np.nan,
            **r.conditions,
        })
    return pd.DataFrame(rows)


def _fmt(value) -> str:
    """Formato compacto para etiquetas de grupo."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def assign_scans(inventory: pd.DataFrame,
                 columns: Sequence[str] = PHYSICS_COLUMNS) -> pd.DataFrame:
    """Añade `scan`: etiqueta del barrido de σ al que pertenece cada corrida.

    Dos corridas están en el mismo barrido si comparten todas las `columns` (por
    defecto, la física y el tamaño de red). La etiqueta es legible, por ejemplo
    `iso · L=35 · flat` o `aniso δ=0.3/1.78 · L=80 · flat`.
    """
    inv = inventory.copy()
    labels = []
    for _, row in inv.iterrows():
        iso = (np.isclose(row["delta_x"], row["delta_y"])
               and np.isclose(row["E_pb_over_kT_x"], row["E_pb_over_kT_y"]))
        kind = "iso" if iso else f"aniso δ={_fmt(row['delta_x'])}/{_fmt(row['delta_y'])}"
        size = (f"L={row['size_x']}" if row["size_x"] == row["size_y"]
                else f"{row['size_x']}x{row['size_y']}")
        labels.append(f"{kind} · {size} · {row['init_mode']}"
                      + ("" if row["engine"] == "static" else f" · {row['engine']}"))
    inv["scan"] = labels
    # Si dos grupos de condiciones distintos dieran la misma etiqueta, se numeran
    key = inv[list(columns)].astype(str).agg("|".join, axis=1)
    for label, grp in inv.groupby("scan"):
        groups = key[grp.index].unique()
        if len(groups) > 1:
            for i, g in enumerate(groups):
                inv.loc[grp.index[key[grp.index] == g], "scan"] = f"{label} ({i + 1})"
    return inv
