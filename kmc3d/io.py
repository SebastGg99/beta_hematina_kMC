"""Guardado y carga de corridas 3D, y construcción del motor desde un config JSON.

Formato de una carpeta de corrida (análogo al de `scripts/run_sigma_scan.py`):
    metadata.json   grafo, tasas, parámetros del motor, condiciones de parada, git
    snaps.npz       coordenadas de cada snapshot (concatenadas) + tiempos + offsets
    history.csv     historial del motor (t, eventos, N, caras seguidas, ...)
    kmc.pkl         (opcional) motor completo para continuar la corrida

Reutiliza `src.common.io` (save_json, load_json, git_commit, find_repo_root).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from src.common.io import find_repo_root, git_commit, load_json, save_json, save_pickle

from . import presets, seeds
from .engine import KMC3D
from .geometry import Cell
from .graph import CrystalGraph
from .rates import delta_mu_from_sigma, diffusion_from_dict, rates_from_dict

PathLike = Union[str, Path]


def save_snapshots(snaps: List[Tuple[float, np.ndarray]], path: PathLike) -> Path:
    """Guarda [(t, coords (N, 3))] en un .npz compacto (coordenadas int32 concatenadas)."""
    path = Path(path)
    times = np.array([t for t, _ in snaps], dtype=float)
    sizes = np.array([len(c) for _, c in snaps], dtype=np.int64)
    coords = (np.concatenate([c for _, c in snaps]).astype(np.int32) if snaps
              else np.zeros((0, 3), np.int32))
    np.savez_compressed(path, times=times, sizes=sizes, coords=coords)
    return path


def load_snapshots(path: PathLike) -> List[Tuple[float, np.ndarray]]:
    """Inversa de `save_snapshots`."""
    with np.load(path) as z:
        times, sizes, coords = z["times"], z["sizes"], z["coords"]
    edges = np.r_[0, np.cumsum(sizes)]
    return [(float(t), coords[a:b]) for t, a, b in zip(times, edges[:-1], edges[1:])]


def save_run(run_dir: PathLike, kmc: KMC3D, snaps, stats: dict, config: Optional[dict] = None,
             save_engine: bool = False) -> Path:
    """Guarda una corrida completa en `run_dir` (ver formato en el docstring)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_snapshots(snaps, run_dir / "snaps.npz")
    stats["history"].to_csv(run_dir / "history.csv", index=False)
    meta = {
        "engine": "kmc3d.KMC3D",
        "graph": kmc.graph.to_dict(),
        "rates": kmc.rates.to_dict(),
        "diffusion": kmc.diffusion.to_dict() if kmc.diffusion is not None else None,
        "engine_params": {"rng_seed": kmc.rng_seed, "time_scale": kmc.time_scale,
                          "allow_enclosed": kmc.allow_enclosed, "margin": kmc.margin,
                          "room": kmc.room, "track_faces": [list(f) for f in kmc.track_faces]},
        "stats": {k: (list(v) if isinstance(v, tuple) else v) for k, v in stats.items()
                  if k != "history"},
        "config": config,
        "git_commit": git_commit(find_repo_root(run_dir) or _paths.REPO_ROOT),
    }
    meta["stats"]["track_faces"] = [list(f) for f in kmc.track_faces]
    save_json(_jsonable(meta), run_dir / "metadata.json")
    if save_engine:
        save_pickle(kmc, run_dir / "kmc.pkl")
    return run_dir


def load_run(run_dir: PathLike) -> Dict[str, object]:
    """Carga metadata, snapshots e historial de una carpeta de corrida."""
    run_dir = Path(run_dir)
    meta = load_json(run_dir / "metadata.json")
    return {"meta": meta, "graph": CrystalGraph.from_dict(meta["graph"]),
            "snaps": load_snapshots(run_dir / "snaps.npz"),
            "history": pd.read_csv(run_dir / "history.csv")}


def _jsonable(obj):
    """Convierte tipos numpy/tuplas a tipos JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Config JSON -> motor
# ---------------------------------------------------------------------------
def graph_from_config(cfg: dict) -> CrystalGraph:
    """Bloque "graph" del config: {"preset": ..., parámetros} o un grafo completo.

    Presets: "kossel" {phi}, "anisotropic_cubic" {phi_x, phi_y, phi_z, phi_xy, ...},
    "beta_hematin" {energies, scale}. Sin "preset" se espera el formato de
    `CrystalGraph.to_dict()`.
    """
    g = dict(cfg)
    preset = g.pop("preset", None)
    if preset is None:
        return CrystalGraph.from_dict(g)
    if preset == "kossel":
        return presets.kossel(**g)
    if preset == "anisotropic_cubic":
        if "cell" in g:
            g["cell"] = Cell(**g["cell"])
        return presets.anisotropic_cubic(**g)
    if preset == "beta_hematin":
        return presets.beta_hematin(**g)
    raise ValueError(f"Preset desconocido: {preset!r}")


def seed_from_config(cfg: dict, graph: CrystalGraph) -> np.ndarray:
    """Bloque "seed": {"shape": "cube"|"box"|"sphere"|"polyhedron", ...}."""
    s = dict(cfg)
    shape = s.pop("shape")
    if shape == "cube":
        return seeds.cube(**s)
    if shape == "box":
        return seeds.box(**s)
    if shape == "sphere":
        return seeds.sphere(s["radius"], graph.cell)
    if shape == "polyhedron":
        dist = {tuple(int(x) for x in k.strip("()").split(",")): v for k, v in s["distances"].items()}
        return seeds.polyhedron(dist, graph.cell, s.get("centrosymmetric", True))
    raise ValueError(f"Semilla desconocida: {shape!r}")


def engine_from_config(cfg: dict) -> KMC3D:
    """Construye el motor desde un config (ver configs/*.json).

    El bloque "rates" acepta "delta_mu" o "sigma" (Δμ = ln(1 + σ)). El bloque opcional
    "diffusion" ({"k_d": ..., "hop_vectors": "bonds" | "cube26" | [[u, v, w], ...],
    "weights": [...]}) activa la difusión superficial (`rates.SurfaceDiffusion`).
    """
    graph = graph_from_config(cfg["graph"])
    rcfg = dict(cfg["rates"])
    if "sigma" in rcfg:
        rcfg["delta_mu"] = delta_mu_from_sigma(rcfg.pop("sigma"))
    rates = rates_from_dict(graph, rcfg)
    ecfg = dict(cfg.get("engine", {}))
    if "track_faces" in ecfg:
        ecfg["track_faces"] = [tuple(f) for f in ecfg["track_faces"]]
    diffusion = diffusion_from_dict(graph, cfg.get("diffusion"))
    return KMC3D(graph, rates, seed_from_config(cfg["seed"], graph), diffusion=diffusion, **ecfg)
