"""Ejecuta una o varias corridas 3D a partir de un config JSON.

Uso (desde la raíz del repositorio):
    python scripts/run_3d.py --config configs/kossel_smooth.json
    python scripts/run_3d.py --config ... --delta-mu 0.5 1.0 2.0 --seeds 1 2 3

Cada corrida se guarda en <out>/<nombre>_dmu_<Δμ>_seed_<semilla>/ con el formato
de `kmc3d.io.save_run` (metadata.json, snaps.npz, history.csv).

El config tiene los bloques "graph", "rates", "seed", "engine" y "run"
(ver configs/ y kmc3d.io.engine_from_config).
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

# raíz del repositorio en sys.path para importar kmc3d (y con él src/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kmc3d.io import engine_from_config, save_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config JSON")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto outputs/)")
    p.add_argument("--delta-mu", type=float, nargs="*", default=None,
                   help="barrido de Δμ/kT (sustituye rates.delta_mu / rates.sigma)")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="semillas del RNG (sustituye engine.rng_seed)")
    p.add_argument("--save-engine", action="store_true", help="guarda también kmc.pkl")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg0 = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "outputs"
    name = cfg0.get("name", Path(args.config).stem)
    dmus = args.delta_mu if args.delta_mu else [None]
    rng_seeds = args.seeds if args.seeds else [cfg0.get("engine", {}).get("rng_seed")]

    for dmu in dmus:
        for seed in rng_seeds:
            cfg = copy.deepcopy(cfg0)
            if dmu is not None:
                cfg["rates"].pop("sigma", None)
                cfg["rates"]["delta_mu"] = dmu
            cfg.setdefault("engine", {})["rng_seed"] = seed
            kmc = engine_from_config(cfg)
            tag = f"{name}_dmu_{kmc.rates.delta_mu:g}_seed_{seed}"
            t0 = time.perf_counter()
            snaps, stats = kmc.run(**cfg.get("run", {}))
            dt = time.perf_counter() - t0
            stats["wall_seconds"] = dt
            save_run(out / tag, kmc, snaps, stats, config=cfg, save_engine=args.save_engine)
            print(f"{tag}: {stats['n_events']} eventos, N = {stats['n_occ']}, "
                  f"parada = {stats['stop_reason']}, {dt:.1f} s "
                  f"({stats['n_events'] / max(dt, 1e-9):.0f} eventos/s)")


if __name__ == "__main__":
    main()
