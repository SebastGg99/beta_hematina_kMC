"""Barridos S(Δμ) por cara con el motor de superficie (validación con MONTY).

Uso (desde la raíz del repositorio):

    # grasa transcrita: protocolo de MONTY (caja, eventos, T y λ) y caras de su figura
    python scripts/run_monty.py --grafo data/monty2004/beta_16_16_16.json \\
        --dmu-kcal 0 20 0.5 --semillas 1 2 3

    # grafo de prueba en kT
    python scripts/run_monty.py --grafo kossel --phi 4 --caras "(001)" "(111)" \\
        --dmu 0 6 0.25 --tam 40 40 --equil 100000 --muestreo 100000

Cada punto (cara, Δμ, semilla) es una llamada a `kmc3d.monty.sticking_point`, lanzada en
paralelo con `kmc3d.parallel.run_parallel`. Los resultados se añaden a un CSV
(`--salida`, por defecto outputs/03_validacion_monty/barridos/<grafo>.csv). Los puntos
que ya están en el CSV no se repiten, así que un barrido interrumpido se reanuda
lanzando la misma orden. Los mapas de alturas finales van a un .pkl al lado del CSV.

Con un JSON de grasa, los valores por defecto (caras, T, caja, eventos, λ) salen de sus
campos "faces" y "protocolo" (ver data/monty2004/README.md).
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kmc3d import monty  # noqa: E402
from kmc3d.geometry import hkl_label, parse_hkl  # noqa: E402
from kmc3d.parallel import run_parallel  # noqa: E402


def rango(v):
    """[ini, fin, paso] -> valores (incluido fin); [a, b, c, ...] con != 3 valores -> tal cual."""
    if len(v) == 3 and v[2] > 0 and v[1] > v[0]:
        n = int(round((v[1] - v[0]) / v[2]))
        return [round(v[0] + k * v[2], 10) for k in range(n + 1)]
    return list(v)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grafo", required=True, help='"kossel", "beta_hematin" o ruta a un JSON de grafo')
    p.add_argument("--phi", type=float, default=None, help="φ/kT de Kossel")
    p.add_argument("--T", type=float, default=None, help="temperatura (K) para pasar kcal/mol a kT")
    p.add_argument("--caras", nargs="*", default=None, help='caras, p. ej. "(001)" "(1-11)"')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dmu", type=float, nargs="+", help="Δμ/kT: lista o ini fin paso")
    g.add_argument("--dmu-kcal", type=float, nargs="+", help="Δμ en kcal/mol: lista o ini fin paso")
    p.add_argument("--semillas", type=int, nargs="+", default=[1])
    p.add_argument("--tam", type=int, nargs=2, default=None, help="L1 L2 (celdas de capa)")
    p.add_argument("--equil", type=int, default=None, help="eventos de equilibrado")
    p.add_argument("--muestreo", type=int, default=None, help="eventos de muestreo")
    p.add_argument("--lambdas", type=float, nargs=2, default=None, help="λ1 λ2")
    p.add_argument("--huecos-cerrados-fijos", action="store_true",
                   help="un hueco cerrado no se llena (regla de KMC3D); por defecto, reglas de MONTY")
    p.add_argument("--procesos", type=int, default=None)
    p.add_argument("--salida", default=None, help="CSV de resultados (se añade)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    base = {"graph": args.grafo}
    proto, faces = {}, None
    if args.grafo.endswith(".json"):
        base["graph"] = str(Path(args.grafo).resolve())
        _, meta = monty.load_graph_file(args.grafo)
        proto = dict(meta.get("protocolo", {}))
        faces = meta.get("faces")
    if args.phi is not None:
        base["phi"] = args.phi
    T = args.T if args.T is not None else proto.get("T")
    if T is not None:
        base["T"] = float(T)
    base["size"] = tuple(args.tam) if args.tam else tuple(proto.get("size", (40, 40)))
    base["n_equil"] = args.equil if args.equil is not None else int(proto.get("n_equil", 100_000))
    base["n_sample"] = args.muestreo if args.muestreo is not None else int(proto.get("n_sample", 100_000))
    l1, l2 = args.lambdas if args.lambdas else (proto.get("lambda1", 0.5), proto.get("lambda2", 0.5))
    base["lambda1"], base["lambda2"] = float(l1), float(l2)
    base["allow_enclosed"] = not args.huecos_cerrados_fijos
    caras = args.caras or faces
    if not caras:
        raise SystemExit("Faltan las caras (--caras o el campo 'faces' del JSON)")
    caras = [hkl_label(parse_hkl(c)) if isinstance(c, str) else hkl_label(c) for c in caras]

    if args.dmu_kcal is not None:
        dcol, vals = "delta_mu_kcal", rango(args.dmu_kcal)
        if "T" not in base:
            raise SystemExit("Con --dmu-kcal hace falta la temperatura (--T o el protocolo del JSON)")
    else:
        dcol, vals = "delta_mu", rango(args.dmu)
    tareas = monty.sticking_grid(base, caras, **{dcol: vals}, seeds=args.semillas)

    nombre = Path(args.grafo).stem if args.grafo.endswith(".json") else args.grafo
    salida = Path(args.salida) if args.salida else ROOT / "outputs" / "03_validacion_monty" / "barridos" / f"{nombre}.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    hechos = set()
    if salida.exists():
        prev = pd.read_csv(salida)
        hechos = {(r["hkl"], round(float(r[dcol]), 9), int(r["rng_seed"])) for _, r in prev.iterrows()}
    pend = [t for t in tareas if (t["hkl"], round(t[dcol], 9), t["rng_seed"]) not in hechos]
    print(f"{len(tareas)} puntos, {len(tareas) - len(pend)} ya hechos, {len(pend)} pendientes -> {salida}")
    if not pend:
        return

    t0 = time.perf_counter()
    mapas_f = salida.with_suffix(".mapas.pkl")
    mapas = pickle.loads(mapas_f.read_bytes()) if mapas_f.exists() else {}

    def guardar(i, r):
        # cada resultado se escribe al terminar: un barrido interrumpido no pierde nada
        fila = {k: v for k, v in r.items() if k != "mapa"}
        fila["size"] = "x".join(str(x) for x in fila["size"])
        pd.DataFrame([fila]).to_csv(salida, mode="a", header=not salida.exists(), index=False)
        if "mapa" in r:
            mapas[(r["hkl"], round(float(r[dcol]), 9), int(r["rng_seed"]))] = r["mapa"]
        print(f"  {r['hkl']:>9} {dcol} = {r[dcol]:7.3f}  semilla {r['rng_seed']}: S = {r['S']:.3f}  "
              f"({r['segundos']:.0f} s, {r['eventos_por_s']:.0f} ev/s)", flush=True)

    run_parallel("kmc3d.monty:sticking_point", pend, processes=args.procesos, report=guardar)
    mapas_f.write_bytes(pickle.dumps(mapas))
    print(f"Hecho en {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
