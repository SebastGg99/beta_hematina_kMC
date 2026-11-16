"""Ejecución de corridas independientes en paralelo con `subprocess` (biblioteca estándar).

Cada tarea es una llamada `funcion(params)` a una función importable de nivel de módulo,
indicada como "paquete.modulo:funcion". Se lanza en un proceso de Python aparte; los
parámetros y el resultado viajan en archivos pickle temporales.

Por qué no `multiprocessing`: necesita semáforos POSIX, que no existen en algunos
entornos aislados (`PermissionError` en `SemLock`), y "fork" da problemas dentro de
Jupyter. Lanzar procesos con `subprocess` funciona en ambos casos.

Las corridas son independientes y deterministas (cada una lleva su `rng_seed`), así que
el resultado no depende del número de procesos ni del orden de terminación.
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Código que ejecuta cada proceso hijo: importa la función, lee params y guarda el resultado
_CHILD = ("import importlib, pickle, sys\n"
          "mod, fun = sys.argv[1].split(':')\n"
          "f = getattr(importlib.import_module(mod), fun)\n"
          "p = pickle.load(open(sys.argv[2], 'rb'))\n"
          "pickle.dump(f(p), open(sys.argv[3], 'wb'))\n")


def default_processes(n_tasks: int) -> int:
    """Procesos por defecto: todos los núcleos menos 2, sin pasar del número de tareas."""
    return max(1, min(n_tasks, (os.cpu_count() or 2) - 2))


def run_parallel(target: str, params_list: Sequence[Dict[str, object]],
                 processes: Optional[int] = None,
                 report: Optional[Callable[[int, object], None]] = None,
                 poll: float = 0.2) -> List[object]:
    """Ejecuta `target(params)` para cada elemento de `params_list`.

    Parámetros:
        target: "paquete.modulo:funcion" (debe poder importarse desde la raíz del repositorio)
        params_list: un diccionario de parámetros por tarea (debe poder serializarse con pickle)
        processes: procesos simultáneos (por defecto `default_processes`); con 1 se
            ejecuta en serie en este mismo proceso, sin subprocesos (útil para depurar)
        report: función opcional report(i, resultado) llamada al terminar cada tarea
    Devuelve los resultados en el mismo orden que `params_list`. Si una tarea falla, se
    detienen las demás y se lanza RuntimeError con su stderr.
    """
    mod, fun = target.split(":")
    if processes == 1:
        import importlib
        f = getattr(importlib.import_module(mod), fun)
        out = []
        for i, p in enumerate(params_list):
            out.append(f(dict(p)))
            if report:
                report(i, out[-1])
        return out

    processes = processes or default_processes(len(params_list))
    raiz_3d = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    # el hijo ve kmc3d y src (raíz del repositorio) y los mismos paquetes que este proceso
    env["PYTHONPATH"] = os.pathsep.join([raiz_3d] + [p for p in sys.path if p])
    out: List[object] = [None] * len(params_list)
    with tempfile.TemporaryDirectory(prefix="kmc3d_par_") as tmp:
        pendientes = list(range(len(params_list)))
        activos: Dict[int, subprocess.Popen] = {}
        while pendientes or activos:
            # lanzar hasta `processes` tareas a la vez
            while pendientes and len(activos) < processes:
                i = pendientes.pop(0)
                fin, fout = Path(tmp) / f"in_{i}.pkl", Path(tmp) / f"out_{i}.pkl"
                with open(fin, "wb") as fh:
                    pickle.dump(dict(params_list[i]), fh)
                activos[i] = subprocess.Popen(
                    [sys.executable, "-c", _CHILD, target, str(fin), str(fout)],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            # recoger las terminadas
            for i, proc in list(activos.items()):
                if proc.poll() is None:
                    continue
                err = proc.stderr.read().decode(errors="replace")
                proc.stderr.close()
                del activos[i]
                if proc.returncode != 0:
                    for q in activos.values():
                        q.kill()
                        q.wait()
                        q.stderr.close()
                    raise RuntimeError(f"La tarea {i} ({target}) falló:\n{err}")
                with open(Path(tmp) / f"out_{i}.pkl", "rb") as fh:
                    out[i] = pickle.load(fh)
                if report:
                    report(i, out[i])
            time.sleep(poll)
    return out
