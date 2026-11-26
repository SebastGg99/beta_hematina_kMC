# beta_hematina_kMC: kinetic Monte Carlo 3D de crecimiento cristalino

Motor kMC (BKL, sin rechazo) para un **cristal finito que crece por todas sus caras**.
La anisotropía sale de un **grafo de enlaces** sobre una red de Bravais con una unidad
de crecimiento por celda, con cualquier celda (triclínica incluida). Las ideas vienen de
MONTY y CrystalGrower. Diseño, ecuaciones, fases y decisiones abiertas:
[`PLAN_3DkMC.md`](PLAN_3DkMC.md). Fundamentos teóricos:
[`teoria/teoria_3DkMC.pdf`](teoria/teoria_3DkMC.pdf).

Reutiliza `src.common` (numérica, E/S) y `src.analysis` (observables de superficie) del
motor (2+1)D, y usa `src.static` para un test de correspondencia con ese modelo. Ver
[Código reutilizado](#código-reutilizado).

## Código reutilizado

Este proyecto se desarrolló en la carpeta `3DkMC/` del repositorio
[kMC_Malaria](https://github.com/SebastGg99/kMC_Malaria) y se trasladó aquí el
2026-11-16 con sus archivos sin cambios, salvo las rutas:

- El contenido de `3DkMC/` está en la raíz de este repositorio.
- `src/` es una **copia sin modificar** del `src/` de kMC_Malaria, tal como estaba en el
  árbol de trabajo el 2026-11-16 (commit `5fbe643` más cambios sin confirmar en
  `src/analysis/` y `src/plotting/`). Se copia entero porque cualquier `import src.*`
  ejecuta `src/__init__.py`, que carga también `src.dynamic` y `src.plotting`.
- `kmc3d/_paths.py` busca ahora `src/` junto a `kmc3d/` (antes, dos niveles más arriba).
  Fuera de esto solo cambian docstrings y comentarios con rutas.

Si `src/` cambia en kMC_Malaria, aquí hay que volver a copiarlo a mano. Lo que usa
`kmc3d`:

| Módulo de `src` | Dónde se usa |
|---|---|
| `src.common.numerics._safe_exp` | `kmc3d/rates.py` |
| `src.common.io` (`save_json`, `load_json`, `save_pickle`, `git_commit`, `find_repo_root`) | `kmc3d/io.py` |
| `src.analysis.surface_observables` | `kmc3d/analysis.py` (observables de una cara) |
| `src.static`, `src.analysis.adsorption` | `tests/test_rates.py` (correspondencia con `KMC_BKL_Static`) |

## Uso rápido

```python
import sys; sys.path.insert(0, "<ruta>/beta_hematina_kMC")   # o abrir desde la raíz
from kmc3d import KMC3D, SplitRates, presets, seeds
from kmc3d import analysis as an, viz

g = presets.beta_hematin()                     # celda P-1 real + energías provisionales
kmc = KMC3D(g, SplitRates.symmetric(g, delta_mu=1.0), seeds.sphere(70.0, g.cell),
            rng_seed=1, track_faces=presets.BETA_HEMATIN_FACES)
snaps, stats = kmc.run(n_max=40_000, max_events=2_000_000, snapshot_every=10_000)

R = an.face_growth_rates(an.face_mean_series(snaps, g, presets.BETA_HEMATIN_FACES))
forma = an.shape_descriptors(snaps[-1][1], g.cell)
```

Visualización (`kmc3d.render`, matplotlib, sin WebGL):

```python
from kmc3d import render
render.render_crystal(snaps[-1][1], g.cell, presets.BETA_HEMATIN_FACES)   # superficie por faceta
caras, areas = an.wulff_faces(an.family_rates(R), g.cell)                 # Wulff cinético
render.plot_polyhedron(caras, presets.BETA_HEMATIN_FACES, cell=g.cell)
render.plot_thickness_maps(snaps[-1][1], g.cell)                           # espesor tipo TEM
anim = viz.animate_growth_3d(snaps, g.cell, color_by="facet", faces=presets.BETA_HEMATIN_FACES)
```

Difusión superficial (opcional; sin ella las trayectorias no cambian):

```python
from kmc3d.rates import SurfaceDiffusion
dif = SurfaceDiffusion(g, k_d=0.05, hop_vectors="bonds")   # o "cube18", "cube26"
kmc = KMC3D(g, SplitRates.symmetric(g, 1.0), seeds.sphere(70.0, g.cell), rng_seed=1, diffusion=dif)
```

La tasa de salto es k_d·exp[−(E_origen − E_kink)] por dirección, con balance detallado
respecto de la energía de enlace. Los destinos no válidos se rechazan (paso nulo). En un
config JSON va como bloque `"diffusion": {"k_d": ..., "hop_vectors": ...}`.

Calibración con la morfología y corridas en paralelo:

```python
from kmc3d import calibration as cal
res = cal.run_candidates([{"nombre": "a", "delta_mu": 1.5, "overrides": {"[001]": 3.0}},
                          {"nombre": "b", "delta_mu": 1.5, "scale": 1.2}])
cal.results_frame(res)        # métricas de forma, 6 criterios y pérdida J
```

`kmc3d.parallel.run_parallel("modulo:funcion", lista_de_params)` corre cualquier función
importable en varios procesos (con `subprocess`; funciona dentro de Jupyter).

Otros grafos: `presets.kossel(phi)`, `presets.anisotropic_cubic(phi_x, phi_y, phi_z,
phi_xy=..., ...)`, `presets.beta_hematin(energies={"[001]": 3.5}, scale=1.3)`, o un
`CrystalGraph(cell, [BondType((u, v, w), phi), ...])` propio.

Tasas (`kmc3d.rates`):
- `SplitRates.gilmer` y `SplitRates.symmetric`, o `SplitRates(a_mu, b_E)` en general.
  Todas cumplen balance detallado.
- `AdaptiveRates`: la forma del motor estático 2D (Nagpal), solo para comparar.

## Desde la línea de comandos

```bash
python scripts/run_3d.py --config configs/beta_hematin.json
python scripts/run_3d.py --config configs/kossel_smooth.json --delta-mu 0.5 1 2 --seeds 1 2 3
```

Cada corrida se guarda en `outputs/<nombre>_dmu_<Δμ>_seed_<s>/` con
`metadata.json` (grafo, tasas, parámetros, git commit), `snaps.npz` e `history.csv`.
Se cargan con `kmc3d.io.load_run`. El §8 del notebook 01 las resume.

Para las curvas de fracción de adhesión de MONTY (una superficie periódica por cara):

```bash
python scripts/run_monty.py --grafo kossel --phi 4 --caras "(001)" "(111)" \
    --dmu 0 6 0.25 --semillas 1 2 3
```

Los resultados se añaden a `outputs/03_validacion_monty/barridos/<grafo>.csv`, y un
barrido interrumpido se reanuda con la misma orden (ver `data/monty2004/README.md` para
las grasas). Para añadir semillas solo donde la curva sube, sin repetir las mesetas
S ≈ 0 y S ≈ 1, se añade `--solo-subida 0.02 0.98`.

## Contenido

| Ruta | Qué es |
|---|---|
| `kmc3d/geometry.py` | `Cell` (matriz real y recíproca, d_hkl, normales), índices de Miller |
| `kmc3d/graph.py` | `BondType`, `CrystalGraph`: códigos de vecindad, E_slice/E_att, caras F/S/K |
| `kmc3d/presets.py` | Kossel, cúbico anisótropo, β-hematina (celda, caras, energías provisionales) |
| `kmc3d/rates.py` | `SplitRates`, `AdaptiveRates`, `SurfaceDiffusion`, `delta_mu_from_sigma` |
| `kmc3d/fenwick.py` | árbol de Fenwick para elegir la clase de evento en O(log K) |
| `kmc3d/engine.py` | `KMC3D`: motor BKL incremental, caja que se amplía sola, seguimiento de caras |
| `kmc3d/seeds.py` | semillas: cubo, caja, esfera, poliedro (hkl) |
| `kmc3d/analysis.py` | posición y velocidad de caras, forma, Wulff cinético (caras y áreas), facetas y áreas del cristal kMC, fragmentos, mapas de alturas por cara, probabilidad de unión |
| `kmc3d/render.py` | superficie coloreada por faceta, poliedros de Wulff, proyecciones de espesor |
| `kmc3d/viz.py` | vóxeles con la celda real, proyecciones, D(t), mapas de alturas, animación |
| `kmc3d/calibration.py` | calibración de β-hematina con la morfología (criterios, corridas, tabla) |
| `kmc3d/parallel.py` | corridas independientes en paralelo con `subprocess` |
| `kmc3d/experiments.py` | experimentos de difusión del notebook 02 (Kossel, varilla) |
| `kmc3d/io.py` | guardar/cargar corridas, config JSON -> motor |
| `kmc3d/basis_graph.py` | `BasisGraph`: grafo con Z unidades por celda (formato de MONTY), kcal/mol ↔ kT, simetría, energías de capa por corte, celda de capa (`slice_cell`) |
| `kmc3d/slab.py` | `SlabKMC`: superficie (hkl) con contorno periódico (modo *slab* de MONTY); `MontyRates` (λ1, λ2) |
| `kmc3d/monty.py` | validación con MONTY: curvas S(Δμ) por cara, umbrales, comparación con la referencia |
| `configs/` | `kossel_smooth`, `anisotropic_needle`, `beta_hematin` |
| `scripts/run_3d.py` | corridas y barridos desde un config |
| `scripts/run_monty.py` | barridos S(Δμ) por cara con `SlabKMC` (CSV reanudable) |
| `scripts/digitalizar_monty.py` | digitaliza las Figs. 2–4 de MONTY y comprueba la referencia |
| `scripts/transcribir_grasas.py` | transcribe los grafos de las grasas de MONTY (Tablas 2 y 3 de Hollander et al. 2003) a `data/monty2004/*.json` y los comprueba |
| `data/monty2004/` | referencia digitalizada de MONTY y grafos de las grasas (Hollander et al. 2003) |
| `notebooks/01_primeros_resultados.ipynb` | validación (Kossel), anisotropía, β-hematina |
| `notebooks/02_difusion_visualizacion_calibracion.ipynb` | difusión, visualizaciones nuevas, calibración de β-hematina |
| `notebooks/03_validacion_monty.ipynb` | validación con MONTY (Boerrigter et al. 2004): metodología y grasas |
| `teoria/` | documento teórico (`teoria_3DkMC.tex`/`.pdf`) y `make_figures.py` |
| `outputs/` | figuras y cachés de los notebooks |
| `src/` | copia del motor (2+1)D de kMC_Malaria (ver [Código reutilizado](#código-reutilizado)) |
| `docs/` | documentación con Sphinx (ver [Documentación](#documentación)) |
| `tests/` | 100 tests, unos 25 s |

## Documentación

`docs/` tiene la documentación con Sphinx: páginas en Markdown (MyST), tema de Read the
Docs y la referencia de la API generada desde los docstrings (autodoc + autosummary).
Sigue la estructura del [tutorial de Sphinx](https://www.sphinx-doc.org/en/master/tutorial/index.html).

```bash
pip install -r docs/requirements.txt
cd docs
make html        # docs/build/html/index.html
make doctest     # comprueba los ejemplos >>> de la guía de uso
```

`docs/build/` y `docs/source/generated/` se generan al compilar y no se versionan. El
logo se regenera con `python docs/make_logo.py`.

## Tests

```bash
python -m unittest discover -s tests      # desde la raíz del repositorio
```

Cubren la geometría (recíproca de β-hematina frente a Solomonov et al. 2007), el grafo
(Hartman-Perdok, Kossel F/S/K) y las tasas (balance detallado, correspondencia con
`KMC_BKL_Static`). Del motor comprueban que el estado incremental coincide con una
reconstrucción, la conservación, el determinismo y el pickle. También hay comprobaciones
físicas: signo de Δμ, caras lisas frente a rugosas y aguja con enlaces anisótropos. La
difusión tiene balance detallado comprobado por enumeración, y sin difusión las
trayectorias son idénticas a las de antes (`test_diffusion`). También hay tests de las
áreas de Wulff y de facetas, de los criterios de calibración y del ejecutor paralelo. El motor de
superficie (`test_slab`, `test_basis_graph`) tiene sus propios tests: `MontyRates` frente
a `SplitRates`, estado incremental frente a reconstrucción, energías de sitio contadas a
mano con base > 1, supercelda y celda de capa. `test_grasas` comprueba los grafos de las
grasas de MONTY frente a la Tabla 3 de Hollander et al. (2003) y su morfología de energía
de unión (Fig. 1 de MONTY).

## Rendimiento

Python puro: unos 80 000 eventos/s en Kossel (6 vecinos) y unos 31 000 en β-hematina
(14 vecinos). El coste por evento no depende del tamaño del cristal.

Con difusión, la mayoría de los eventos son saltos y una parte se rechaza. En
β-hematina con tasas simétricas, la tasa de salto de un adátomo débil es enorme frente a
su despegue (E_kink ≈ 10 kT). Con k_d = 0.1 se necesitan 8 veces más eventos y 12 veces
más tiempo (notebook 02, §4.7).
