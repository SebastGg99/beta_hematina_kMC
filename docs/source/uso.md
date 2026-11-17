# Guía de uso

Los ejemplos con `>>>` de esta página se ejecutan con `make doctest`, así que su
salida es la real (ver [](instalacion.md)). Los demás bloques son ejemplos de uso
que tardan más y no se comprueban.

## Primeros pasos: un cristal de Kossel

Un **grafo cristalino** (`CrystalGraph`) reúne la celda y los tipos de enlace, cada uno
con un vector de red y una energía $\varphi$ en unidades de $kT$. El más simple es el
cristal de Kossel: celda cúbica y 3 tipos de enlace iguales, 6 vecinos.

```{doctest}
>>> from kmc3d import presets
>>> g = presets.kossel(phi=2.0)
>>> g.coordination, g.lattice_energy, g.kink_energy
(6, 6.0, 6.0)
```

La energía de red por unidad y la de la posición de media-cristal (*kink*) coinciden:
$E_{cr} = E_{kink} = \sum_t \varphi_t$. El grafo también da el carácter de cada cara
según Hartman-Perdok y el reparto de $E_{cr}$ en $E_{slice}$ y $E_{att}$ (ver
[](modelo.md)):

```{doctest}
>>> [g.face_character(f) for f in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]]
['F', 'S', 'K']
>>> g.slice_attachment((1, 0, 0))     # (E_slice, E_att) en kT
(4.0, 2.0)
```

Las **tasas** de unión y despegue dependen del código del sitio, que cuenta los
vecinos ocupados de cada tipo. En un kink (un vecino de cada tipo) y con
$\Delta\mu = 0$, unión y despegue se igualan:

```{doctest}
>>> from kmc3d import SplitRates
>>> r = SplitRates.symmetric(g, delta_mu=0.0)
>>> kink = g.encode([1, 1, 1])
>>> r.attach(kink), r.detach(kink)
(1.0, 1.0)
```

Para simular hace falta además una **semilla** (`kmc3d.seeds`) y el motor `KMC3D`. Con
$\Delta\mu = 1$ un cubo de lado 8 crece:

```{doctest}
>>> from kmc3d import KMC3D, seeds
>>> kmc = KMC3D(g, SplitRates.gilmer(g, delta_mu=1.0), seeds.cube(8), rng_seed=1)
>>> snaps, stats = kmc.run(max_events=5_000, snapshot_every=5_000)
>>> stats["n_seed"], stats["n_occ"], stats["stop_reason"]
(512, 952, 'max_events')
```

`run` devuelve los *snapshots* (lista de `(t, coordenadas)`, con las coordenadas
enteras de las unidades ocupadas) y un diccionario de estadísticas, que incluye el
historial como `DataFrame` en `stats["history"]`. Con la misma `rng_seed` la
trayectoria es siempre la misma.

## β-hematina

```python
from kmc3d import KMC3D, SplitRates, presets, seeds
from kmc3d import analysis as an

g = presets.beta_hematin()                     # celda P-1 real + energías provisionales
kmc = KMC3D(g, SplitRates.symmetric(g, delta_mu=1.0), seeds.sphere(70.0, g.cell),
            rng_seed=1, track_faces=presets.BETA_HEMATIN_FACES)
snaps, stats = kmc.run(n_max=40_000, max_events=2_000_000, snapshot_every=10_000)

R = an.face_growth_rates(an.face_mean_series(snaps, g, presets.BETA_HEMATIN_FACES))
forma = an.shape_descriptors(snaps[-1][1], g.cell)
```

`presets.beta_hematin(energies={"[001]": 3.5}, scale=1.3)` cambia energías concretas
o todas a la vez (equivale a cambiar la temperatura efectiva). Otros grafos:
`presets.anisotropic_cubic(phi_x, phi_y, phi_z, ...)` o un
`CrystalGraph(cell, [BondType((u, v, w), phi), ...])` propio.

## Modelos de tasas

- `SplitRates.gilmer` y `SplitRates.symmetric`, o `SplitRates(a_mu, b_E)` en general.
  Todas cumplen balance detallado.
- `AdaptiveRates`: la forma del motor estático 2D (Nagpal et al.), solo para comparar
  con la línea (2+1)D. No cumple balance detallado.

## Visualización

`kmc3d.render` y `kmc3d.viz` dibujan con matplotlib y no necesitan WebGL:

```python
from IPython.display import HTML
from kmc3d import render, viz

render.render_crystal(snaps[-1][1], g.cell, presets.BETA_HEMATIN_FACES)   # superficie por faceta
caras, areas = an.wulff_faces(an.family_rates(R), g.cell)                 # Wulff cinético
render.plot_polyhedron(caras, presets.BETA_HEMATIN_FACES, cell=g.cell)
render.plot_thickness_maps(snaps[-1][1], g.cell)                           # espesor tipo TEM
anim = viz.animate_growth_3d(snaps, g.cell, color_by="facet", faces=presets.BETA_HEMATIN_FACES)
HTML(anim.to_jshtml())                                                     # animación en el notebook
```

`kmc3d.viz_interactive` usa plotly 3D, que necesita WebGL y no se ve en todos los
visores de notebooks.

## Difusión superficial

Es opcional. Sin ella, las trayectorias no cambian:

```python
from kmc3d.rates import SurfaceDiffusion

dif = SurfaceDiffusion(g, k_d=0.05, hop_vectors="bonds")   # o "cube18", "cube26"
kmc = KMC3D(g, SplitRates.symmetric(g, 1.0), seeds.sphere(70.0, g.cell), rng_seed=1,
            diffusion=dif)
```

## Calibración y corridas en paralelo

```python
from kmc3d import calibration as cal

res = cal.run_candidates([{"nombre": "a", "delta_mu": 1.5, "overrides": {"[001]": 3.0}},
                          {"nombre": "b", "delta_mu": 1.5, "scale": 1.2}])
cal.results_frame(res)        # métricas de forma, 6 criterios y pérdida J
```

`kmc3d.parallel.run_parallel("modulo:funcion", lista_de_params)` corre cualquier
función importable en varios procesos (con `subprocess`; funciona dentro de Jupyter).

## Desde la línea de comandos

```bash
python scripts/run_3d.py --config configs/beta_hematin.json
python scripts/run_3d.py --config configs/kossel_smooth.json --delta-mu 0.5 1 2 --seeds 1 2 3
```

Opciones: `--delta-mu` hace un barrido de $\Delta\mu/kT$, `--seeds` repite con varias
semillas del RNG, `--out` cambia la carpeta de salida (por defecto `outputs/`) y
`--save-engine` guarda también el motor (`kmc.pkl`).

### Formato de los configs

Los configs de `configs/` son JSON con estos bloques (los lee
`kmc3d.io.engine_from_config`):

| Bloque | Contenido | Ejemplo |
|---|---|---|
| `graph` | preset y sus parámetros | `{"preset": "kossel", "phi": 2.0}` |
| `rates` | modelo de tasas, `delta_mu` o `sigma` ($\Delta\mu = \ln(1+\sigma)$) y `k0` | `{"kind": "gilmer", "delta_mu": 1.0, "k0": 1.0}` |
| `seed` | forma de la semilla | `{"shape": "sphere", "radius": 40.0}` |
| `engine` | opciones de `KMC3D` | `{"rng_seed": 1, "track_faces": [[1, 0, 0], ...]}` |
| `run` | condiciones de parada y registro | `{"max_events": 300000, "n_max": 60000, ...}` |
| `diffusion` (opcional) | difusión superficial | `{"k_d": 0.05, "hop_vectors": "bonds"}` |

Hay tres de ejemplo: `kossel_smooth` (validación), `anisotropic_needle` (cubitos con
$\varphi_z > \varphi_y > \varphi_x$) y `beta_hematin` (energías provisionales).

### Salida de una corrida

Cada corrida se guarda en `outputs/<nombre>_dmu_<Δμ>_seed_<s>/` con:

- `metadata.json`: grafo, tasas, parámetros y commit de git;
- `snaps.npz`: los snapshots;
- `history.csv`: el historial.

Se cargan con `kmc3d.io.load_run`. El §8 del notebook 01 las resume.
