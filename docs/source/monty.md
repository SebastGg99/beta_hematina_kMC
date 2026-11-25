# Validación con MONTY

MONTY (S. X. M. Boerrigter et al., *J. Phys. Chem. A* **108**, 5894 (2004)) es el
programa de Monte Carlo de crecimiento cristalino del que vienen las ideas de `kmc3d`:
el grafo cristalino, el reparto de Δμ y de la energía entre unión y despegue, y el BKL
sin rechazo. El artículo valida MONTY con las curvas de **fracción de adhesión**
$S(\Delta\mu)$ por cara de tres grasas: β-16.16.16, β′-10.12.10 y β′-16.16.14. Esta
página explica cómo `kmc3d` reproduce ese protocolo y qué se ha comprobado. Los
resultados están en `notebooks/03_validacion_monty.ipynb`.

## Qué hacía falta

`KMC3D` simula un cristal finito con una unidad de crecimiento (GU) por celda. MONTY
simula **una superficie (hkl) con contorno periódico**, y las grasas tienen **varias
moléculas por celda** (Z = 2, 8 y 4). Por eso hay tres módulos nuevos. Ninguno modifica
el motor, las tasas, los grafos ni β-hematina, así que las trayectorias congeladas de
`TestNoDiffusionUnchanged` quedan iguales por construcción.

| Módulo | Contenido |
|---|---|
| `kmc3d.basis_graph` | `BasisGraph`: celda, base de Z GU y enlaces (s, s′, celda, φ), con energías en kT o kcal/mol. Completa la base con la simetría (`from_centers`), calcula las energías de capa de cada corte (`slice_energies`, `best_termination`, `face_table`) y la celda de capa (`slice_cell`) |
| `kmc3d.slab` | `SlabKMC`: superficie (hkl) periódica, BKL con clases (subred, código), donde el código es la máscara de bits de los enlaces del sitio (la tabla de consulta de MONTY). `MontyRates(λ1, λ2)` |
| `kmc3d.monty` | `sticking_point` (un punto de una curva, para `run_parallel`), umbrales por cara (`onset_table`), referencia digitalizada (`load_reference`) y comparación por grupos (`compare_with_reference`, `order_agreement`) |

Desde la terminal, `scripts/run_monty.py` lanza los barridos en paralelo y los guarda en
un CSV que se puede reanudar.

## Correspondencia MONTY ↔ kmc3d

Con $U_{moth} = 0$ y $\Delta V = 0$ (lo que usa el artículo para las grasas), las
probabilidades de las ecs. 11 y 12 son, con $x = E - E_{cr}$ (la energía de enlace del
sitio menos la de red por GU):

$$
k^+ = k_0\, e^{(1-\lambda_2)\Delta\mu + \lambda_1 x}, \qquad
k^- = k_0\, e^{-\lambda_2\Delta\mu - (1-\lambda_1) x},
$$

es decir, `SplitRates(a_mu = 1 − λ2, b_E = λ1)`. Cumple el balance detallado,
$k^+/k^- = e^{\Delta\mu + x}$. Con λ1 = λ2 = 0 es Gilmer (lluvia aleatoria) y con
λ1 = λ2 = ½, que es lo que usa MONTY para las grasas, es `SplitRates.symmetric`.

| MONTY | kmc3d |
|---|---|
| $U^{cryst}$, energía de red por GU | $E_{cr} = \sum_b \varphi_b / Z$ (`BasisGraph.lattice_energy`) |
| $U_{moth} \neq 0$ | equivale a multiplicar todas las φ por $1 - U_{moth}/U^{cryst}$ (`BasisGraph.scaled`) |
| STS: unión con al menos un enlace (también en un hueco cerrado: el texto admite inclusiones); una GU totalmente enlazada no se despega «para mantener el equilibrio en Δμ = 0» | `allow_enclosed=True`, el valor por defecto de `SlabKMC` (ver {ref}`simetria-particula-hueco`). `KMC3D` usa `False` (decisión D4) |
| $\Delta t = \tau/Q$ (ec. 16) | $\Delta t = -\ln u / W$ (`time_mode="media"` da la ec. 16); $S$ no depende de $t$ |
| $S = (N_g - N_e)/N_g$ (ec. 22) | `SlabKMC.sticking_fraction()` tras `reset_counters()` |
| $R_{hkl} = d_{hkl}(N_g - N_e)/(N_{slice}\,t)$ (ec. 20) | `SlabKMC.growth_rate()` |
| celda de capa (u, v en el plano) | `slice_cell`: u, v con $h\cdot u = h\cdot v = 0$, reducidas por Lagrange-Gauss con la métrica real; w con $h\cdot w = 1$, matriz unimodular |
| Δμ en kcal/mol | Δμ/kT = Δμ/(RT) (`basis_graph.kcal_to_kT`) |

(simetria-particula-hueco)=
### Simetría partícula-hueco

Con λ1 = λ2 = ½ y la misma energía total $E_{full} = 2E_{cr}$ en todas las subredes (se
cumple si todas las GU son equivalentes por simetría), el cambio ocupado ↔ vacío
lleva $x \to -x$ y una unión a $\Delta\mu$ a un despegue a $-\Delta\mu$ con la misma
tasa. También lleva una regla STS a la otra: «no se une una GU sin enlaces» ↔ «no se
despega una GU con todos los enlaces». El cristal pasa al otro lado de la superficie,
así que **crecer (hkl) a +Δμ es, evento a evento, lo mismo que disolver (−h−k−l) a
−Δμ**:

$$
R_{(hkl)}(\Delta\mu) = -R_{(\bar h\bar k\bar l)}(-\Delta\mu).
$$

Es lo que quiere decir MONTY con «para mantener el equilibrio en Δμ = 0, también se
prohíbe despegar una GU totalmente enlazada» (§2.3 del artículo): con las dos reglas,
Δμ = 0 es exactamente la coexistencia. La regla de `KMC3D` de no llenar los huecos
cerrados (`allow_enclosed=False`, decisión D4) no tiene pareja y rompe la simetría; el
texto de MONTY admite inclusiones y no prohíbe llenarlas. Por eso las reglas de MONTY
corresponden a `allow_enclosed=True`, que es lo que usan por defecto `SlabKMC` y
`monty.sticking_point` (decisión del autor, 2026-11-24). `KMC3D` no cambia.

Supuestos sobre lo que el artículo no especifica:
- el fondo del *slab* son capas congeladas (al menos el alcance de los enlaces) que
  cuentan como vecinos;
- la superficie inicial es plana, cortada donde se rompen menos enlaces
  (`best_termination`);
- si la superficie se disuelve, la caja se amplía por abajo.

## Uso

```python
from kmc3d import BasisGraph, SlabKMC, MontyRates, presets

g = BasisGraph.from_crystal_graph(presets.kossel(4.0))   # base 1, en kT
k = SlabKMC(g, (1, 1, 1), MontyRates(delta_mu=1.0), size=(40, 40), rng_seed=1)
k.run(max_events=100_000)      # equilibrado
k.reset_counters()
k.run(max_events=100_000)      # muestreo
k.sticking_fraction(), k.growth_rate(), k.roughness()
```

Para barrer varias caras y Δμ en paralelo:

```python
from kmc3d import monty
from kmc3d.parallel import run_parallel

tareas = monty.sticking_grid({"graph": "kossel", "phi": 4.0, "size": (40, 40)},
                             ["(001)", "(011)", "(111)"], delta_mu=[0.5, 1, 2, 4],
                             seeds=[1, 2, 3])
df = monty.results_frame(run_parallel("kmc3d.monty:sticking_point", tareas))
monty.onset_table(df)          # Δμ en S = 0.1, 0.5 y 0.9 por cara
```

## Parte A: metodología

Sin datos de las grasas, con Kossel, β-hematina (como grafo de prueba) y dos cristales
de juguete con dos GU por celda. Detalles y tablas en el notebook 03, §2.

| Prueba | Resultado |
|---|---|
| Tasas | `MontyRates(λ1, λ2)` = `SplitRates(1 − λ2, λ1)` en los 2187 códigos de β-hematina; balance detallado exacto |
| Equilibrio | A Δμ = 0, $S \to 0$ como $1/N$ en la cara lisa (001) y como $1/\sqrt N$ en las rugosas |
| Partícula-hueco | Con las reglas de MONTY, crecer (hkl) a +Δμ y disolver (−h−k−l) a −Δμ dan la misma velocidad: 0.2 % en Kossel y $\lvert z\rvert \le 1$ en un P1 polar. Si los huecos cerrados no se llenan, se rompe (P1: hasta z = −12) |
| Kossel | (001) tiene umbral de nucleación 2D que sube con φ; (011) y (111) crecen desde Δμ = 0, cerca del límite de kinks $1 - e^{-\Delta\mu}$ |
| Energía de borde | El γ exacto de Onsager queda entre las estimaciones mononuclear y de *birth-and-spread* (φ = 3, 4, 5) |
| Superficie frente a cristal finito | El cristal de `KMC3D` crece más despacio por tamaño finito: 65–78 % del slab con L = 12, 98–99 % con L = 48 |
| Supercelda | Δμ(S = ½) igual con base 1 y base 2 dentro de un 1 % (Kossel y β-hematina) |
| Caras opuestas | Coinciden en un P-1 (≤ 0.8 % en S = ½). En un P1 polar se separan un 5 % en S = ½, pero no en el umbral |

```{figure} ../../outputs/03_validacion_monty/kossel_S_vs_dmu.png
:width: 100%
:alt: Curvas de fracción de adhesión de Kossel para cuatro valores de φ

Kossel: $S(\Delta\mu)$ de (001), (011) y (111) con el protocolo de MONTY para
β-16.16.16 (40 × 40, $10^5 + 10^5$ eventos, λ = ½). Línea gris: límite de kinks.
```

```{figure} ../../outputs/03_validacion_monty/particula_hueco.png
:width: 90%
:alt: Velocidad de crecimiento frente a velocidad de disolución de la cara opuesta

Simetría partícula-hueco. Arriba, Kossel: las líneas (crecimiento) y los símbolos
(disolución de la cara opuesta a −Δμ) coinciden con λ = ½ y no con Gilmer. Abajo, P1
polar: el cociente es 1 con las reglas de MONTY y se aleja de 1 si los huecos cerrados
no se llenan.
```

Dos consecuencias para las grasas:
- **Regla de los huecos cerrados.** Se usa la de MONTY (`allow_enclosed=True`), por
  defecto en `SlabKMC`. En Kossel casi no cambia nada respecto a la de `KMC3D`; en el P1
  polar cambia S hasta 0.14.
- **GU no equivalentes.** Si una estructura tiene GU no equivalentes por simetría
  (Z′ > 1) con energías totales distintas, Δμ = 0 con $E_{cr} = \sum\varphi/Z$ deja de
  ser la coexistencia a T > 0.

## Parte B: las tres grasas

### Referencia

`scripts/digitalizar_monty.py` calibra las Figs. 2–4 por píxeles y lista dónde cruzan
las curvas las filas de S = 0.1, 0.5 y 0.9. Los valores, asignados a mano a cada grupo
de caras, están en `data/monty2004/referencia_figuras.csv` junto con los umbrales que da
el texto.

```{figure} ../../outputs/03_validacion_monty/referencia_digitalizada.png
:width: 100%
:alt: Figuras 2 a 4 de MONTY con los valores de referencia digitalizados encima

Figs. 2–4 de MONTY con la referencia digitalizada encima (círculos: S = 0.1, 0.5 y 0.9
de cada grupo).
```

### Estado

Las energías de par de las grasas (kcal/mol, campo de fuerzas CVFF) no están en el
artículo de MONTY sino en sus refs. 16 y 17 (Hollander et al., *J. Phys. Chem. B*
**103**, 8301 (1999) y **107**, 5680 (2003)). Cuando estén transcritas en
`data/monty2004/<grasa>.json` (formato en `data/monty2004/README.md`), el §3.2 del
notebook 03 comprueba el grafo, dibuja la morfología de energía de unión (fila superior
de la Fig. 1) y compara las curvas con la referencia.

Criterios de aceptación:
- **obligatorios** (cualitativos):
  - los mismos grupos de caras y el mismo orden de umbrales;
  - las caras rugosas tienen $S > 1/2$ por debajo de unos 2 kcal/mol;
  - las basales quedan muy por encima del resto;
  - las caras opuestas coinciden en β-16.16.16;
  - los cocientes lateral : basal (13 : 60 y 12 : 43) y superior : lateral (7 : 13 y
    2 : 12) tienen el mismo orden;
- **objetivo** (semicuantitativo): Δμ en $S = 1/2$ de cada grupo dentro de ±20 %, o
  de ±1 kcal/mol en los grupos cercanos a 0.
