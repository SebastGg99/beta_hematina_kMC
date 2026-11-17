# Código reutilizado

## De dónde viene `src/`

`kmc3d` se desarrolló dentro de
[kMC_Malaria](https://github.com/SebastGg99/kMC_Malaria), junto al modelo (2+1)D, y
reutiliza parte de ese código. Al trasladarlo a este repositorio (2026-11-16) se copió
el paquete `src/` **sin modificar**, tal como estaba en el árbol de trabajo ese día
(commit `5fbe643` más cambios sin confirmar en `src/analysis/` y `src/plotting/`).

Se copió entero porque cualquier `import src.*` ejecuta `src/__init__.py`, que carga
también `src.dynamic` y `src.plotting`, aunque `kmc3d` no los use.

## Qué usa `kmc3d`

| Módulo de `src` | Dónde se usa |
|---|---|
| `src.common.numerics._safe_exp` | `kmc3d/rates.py`: todas las tasas |
| `src.common.io` (`save_json`, `load_json`, `save_pickle`, `git_commit`, `find_repo_root`) | `kmc3d/io.py` |
| `src.analysis.surface_observables` | `kmc3d/analysis.py`: observables de la superficie de una cara |
| `src.static`, `src.analysis.adsorption` | `tests/test_rates.py`: correspondencia con `KMC_BKL_Static` |

`kmc3d/_paths.py` añade la raíz del repositorio a `sys.path` la primera vez que se
importa `kmc3d`, para que `src` se encuentre sin instalar nada.

## Mantenimiento

- **No se edita `src/` aquí.** Sus reglas (tests golden, apariencia de los estilos del
  `Plotter`) viven en kMC_Malaria y no se pueden comprobar en este repositorio.
- Si hace falta un cambio, se hace en kMC_Malaria y se vuelve a copiar `src/` entero.
  La copia no se actualiza sola.
- `src/` no aparece en la [referencia de la API](api.rst); su documentación está en
  kMC_Malaria.
