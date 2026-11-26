# Plan de implementación: kMC 3D (`3DkMC/`)

Autor del proyecto: sebastgg99@gmail.com. Documento creado el 2026-11-09.

> **Nota (2026-11-16):** este plan se escribió cuando el proyecto vivía en
> `kMC_Malaria/3DkMC/`. En el repositorio beta_hematina_kMC, el contenido de `3DkMC/`
> está en la raíz y `src/` es una copia del motor (2+1)D de kMC_Malaria. Donde el plan
> dice `3DkMC/...`, léase la raíz de este repositorio (ver «Código reutilizado» en el
> README).

## 0. Estado actual (resumen)

| Fase | Contenido | Estado |
|---|---|---|
| F1 | Núcleo: grafo de enlaces, tasas, motor BKL 3D incremental, tests | **hecho** |
| F2 | Análisis por cara, forma de energía de unión, mapas de alturas, figuras, E/S, modo *slab* | **hecho** (*slab*: `kmc3d/slab.py`, §11) |
| F3 | β-hematina: celda real, energías provisionales, calibración | celda y presets hechos; **calibración con la morfología hecha** (§7.5); falta la cinética (AFM) |
| F4 | Física adicional: difusión superficial, reservorio finito, aditivos | **difusión hecha**; reservorio, aditivos y tasas adaptativas pendientes |
| F5 | Rendimiento (réplicas en paralelo, compilación) | **réplicas en paralelo hechas** (`parallel.py`); compilación pendiente |
| F6 | Unidades moleculares, base > 1, grafo desde CIF | **base > 1 hecha en el *slab*** (`kmc3d/basis_graph.py`, §11); en `KMC3D` y grafo desde CIF, pendiente |

Los resultados de la Fase 1 y la primera exploración de β-hematina están en
`notebooks/01_primeros_resultados.ipynb` y en el §7.4 de este documento. La difusión,
las visualizaciones nuevas (`render.py`) y la calibración de β-hematina con la
morfología están en `notebooks/02_difusion_visualizacion_calibracion.ipynb` y en el
§7.5. La validación con MONTY (superficie periódica, base > 1, curvas de fracción de
adhesión de las tres grasas) está en `notebooks/03_validacion_monty.ipynb` y en el §11.
Los grafos de las grasas se transcribieron y se barrieron el 2026-11-25 (§11.4). Los 19
grupos de caras quedan dentro del objetivo, con dos discrepancias abiertas.

---

## 1. Objetivo

Pasar del modelo actual **(2+1)D** (una altura por columna sobre una red cuadrada
periódica, una sola cara) a un kMC **3D de verdad**:

- un cristal finito que crece (o se disuelve) **por todas sus caras a la vez**;
- con **anisotropía** y **energías distintas por cara y orientación**, que salgan de
  la estructura cristalina (como en β-hematina) y no se impongan a mano;
- empezando con **cubitos** (una unidad de crecimiento por celda) y dejando el camino
  abierto para unidades más realistas.

No se toca el código existente de `src/`: el motor 3D vive en `3DkMC/kmc3d/` y
**reutiliza** de `src/` lo que sirve (numérica, E/S, observables de superficie y,
para comparar, el motor estático 2D).

## 2. Qué cambia respecto al modelo (2+1)D

| Aspecto | `src/static`, `src/dynamic` (2+1)D | `3DkMC` (3D) |
|---|---|---|
| Configuración | altura `h[i, j]` por columna | ocupación 0/1 de cada sitio de una caja 3D |
| Geometría | una cara (001), contorno periódico | cristal finito, todas las caras, caja que se amplía sola |
| Voladizos, huecos | imposibles (SOS) | permitidos |
| Vecinos | 4 laterales (x, y) + el de abajo implícito | los que defina el grafo de enlaces (6 en Kossel, 14 en β-hematina) |
| Anisotropía | `E_pb_x`, `E_pb_y`, `δx`, `δy` | una energía por tipo de enlace (vector de red); las caras salen solas |
| Celda | cuadrada | cualquier celda de Bravais (triclínica incluida) |
| Clasificación de eventos | se reconstruye entera en cada paso | **incremental**: solo el sitio y sus 2T vecinos |
| Elección de clase | lineal | árbol de Fenwick, O(log K) |
| Conteo de enlaces | vecino con altura **exactamente** igual al nivel (`==`, punto abierto del README §8) | geométrico: vecino ocupado en ese nivel |

La última fila es importante: **las dos líneas no cuentan igual los enlaces**. En sitios
donde ningún vecino es más alto (despegue) o más de un nivel más alto (unión) las tasas
coinciden exactamente (`tests/test_rates.py::TestAdaptiveVsStatic2D`); en el resto, el
3D cuenta los enlaces que hay geométricamente y el 2D no. No se ha cambiado la física
del 2D (regla del proyecto): la diferencia queda documentada (decisión D6).

## 3. MONTY, CrystalGrower y este enfoque

| | MONTY (Boerrigter et al. 2004) | CrystalGrower (Anderson et al. 2017) | `3DkMC` |
|---|---|---|---|
| Qué se simula | **una superficie (hkl)** con contorno periódico; las velocidades de cada cara se combinan después en una forma de Wulff cinética | el **cristal entero** | el **cristal entero** (`KMC3D`) y una **superficie (hkl)** periódica (`SlabKMC`, §11) |
| Estructura | grafo cristalino desde la estructura real, cualquier orientación | estructura real, unidades con varias "particiones" | grafo de enlaces sobre una red de Bravais con **una** unidad por celda en `KMC3D`; base > 1 en `SlabKMC` (`BasisGraph`, §11) |
| Energías | campo de fuerzas (pares de unidades) | ΔU por tipo de sitio | φ por tipo de enlace, en kT (provisionales para β-hematina) |
| Tasas | reparto de Δμ y ΔU entre unión y despegue con (λ1, λ2): es `SplitRates(a_mu = 1 − λ2, b_E = λ1)` (verificado sobre las ecs. 11–12, §11) | probabilidades de unión/despegue por energía de sitio | familia `SplitRates` con balance detallado (Gilmer, simétrica, intermedias) + `AdaptiveRates` (Nagpal, como el 2D) |
| Difusión superficial | sí | sí | no (F4) |

Lo que tomamos de MONTY: **el grafo cristalino** como descripción de las
interacciones y el análisis de **caras F** (cadenas periódicas de enlaces, PBC) para
saber qué caras son lisas. Lo que no copiamos: la restricción a una superficie, el
campo de fuerzas y la detección general de redes conexas (con una unidad por celda
basta un criterio de rango, §4.3).

## 4. Modelo

### 4.1 Red y unidades de crecimiento

- Red de Bravais con celda `Cell(a, b, c, α, β, γ)` (`kmc3d/geometry.py`). El motor
  trabaja con **índices enteros** (i, j, k); la geometría real solo entra en d_hkl,
  normales y coordenadas cartesianas.
- Una unidad de crecimiento por celda ("cubito" en el espacio de índices). Para
  β-hematina es **exacto en la topología**: P-1 con Z = 2 hemos = un dímero
  centrosimétrico por celda (§7).

### 4.2 Grafo de enlaces y energía de un sitio

Cada tipo de enlace t es un vector de red v_t con energía φ_t > 0 (en kT), y une r con
r ± v_t (`kmc3d/graph.py`). El estado de un sitio es su **código**

  código = Σ_t n_t · 3^t ,   n_t ∈ {0, 1, 2} = vecinos ocupados del tipo t,

y su energía de enlace es E = Σ_t n_t φ_t. La energía de red por unidad y la de la
posición de media-cristal (kink) coinciden: **E_cr = E_kink = Σ_t φ_t**.

### 4.3 Hartman-Perdok: E_slice, E_att y carácter de cara

Para una cara (hkl) (índices coprimos), un enlace está en la capa d_hkl si
(hkl)·v_t = 0:

  E_slice(hkl) = Σ_{t: hkl·v_t = 0} φ_t ,   E_att(hkl) = Σ_{t: hkl·v_t ≠ 0} φ_t ,
  E_slice + E_att = E_cr.

Modelo de energía de unión: R_hkl ∝ |E_att| (Hartman y Bennema 1980), que da la forma
"de energía de unión" (`analysis.attachment_energy_morphology`).

Carácter: con una unidad por celda, cada tipo de enlace es una cadena periódica, así
que la cara es **F** si los enlaces (φ > 0) de la capa tienen rango 2, **S** si tienen
rango 1 y **K** si no hay ninguno. Comprobado con Kossel: {100} F, {110} S, {111} K.

### 4.4 Tasas

Unir una unidad a un sitio de energía E cambia la energía libre en
ΔG = −[Δμ + (E − E_kink)], con Δμ = ln(1 + σ) (la convención de la línea estática).

**`SplitRates(a, b)`** (`kmc3d/rates.py`), con balance detallado k+/k− = e^{−ΔG}:

  k+ = k0 · exp(a·Δμ + b·(E − E_kink)) ,   k− = k0 · exp(−(1−a)·Δμ − (1−b)·(E − E_kink)).

- `gilmer` (a = 1, b = 0): la unión no depende del sitio (Gilmer y Bennema 1972).
- `symmetric` (a = b = ½): reparto simétrico, del tipo de MONTY/CrystalGrower.
- b > 0: la unión es selectiva por sitio, como la tasa adaptativa de Nagpal, pero sin
  romper el balance detallado.

**`AdaptiveRates`**: k+ = K0 exp(S + Σ n_t δ_t / S), k− = K0 exp(φ0 − E), la forma del
motor estático 2D. Sirve para comparar con la línea (2+1)D y **no** cumple balance
detallado.

Consecuencias que el modelo reproduce y que los tests comprueban:
- Δμ < 0 disuelve y Δμ > 0 hace crecer;
- un cristal finito necesita Δμ > Δμ_c(L) (Gibbs-Thomson): una semilla pequeña se
  disuelve incluso con Δμ > 0. En un cubo de Kossel, L* ≈ 2φ/Δμ;
- las caras lisas (φ alto) no avanzan a Δμ bajo (barrera de nucleación 2D), mientras
  que las rugosas avanzan ∝ Δμ.

### 4.5 Algoritmo (`kmc3d/engine.py`, clase `KMC3D`)

- **Eventos:** unión en un sitio vacío con al menos un vecino, y despegue de una unidad
  con al menos un vecino vacío. Por defecto, un hueco con todos los vecinos ocupados
  no recibe unidades (`allow_enclosed=False`, D4).
- **Clases n-fold way:** clave = código·2 + tipo de evento. Todos los sitios de una
  clase tienen la misma tasa, y sus miembros se guardan en listas con borrado
  *swap-and-pop*.
- **Fenwick:** el peso n_c·r_c de cada clase está en un árbol de Fenwick. Elegir la
  clase cuesta O(log K), y el árbol se recalcula desde cero cada 20 000 eventos para
  evitar la deriva de redondeo.
- **Incremental:** un evento actualiza ±3^t los códigos de los 2T vecinos y reclasifica
  2T + 1 sitios. El coste por evento es O(T + log K), independiente del tamaño.
- **Caja:** el cristal vive en una caja con margen vacío. Si una unidad se une cerca
  del borde, la caja se amplía por ese lado (reconstrucción vectorizada con numpy,
  poco frecuente). Las coordenadas que devuelve el motor son relativas al centro de
  la semilla.
- **Tiempo:** Δt = −ln(u)/W·`time_scale`. Los números aleatorios se generan en bloques
  con `np.random.default_rng(seed)`, así que una misma semilla da la misma trayectoria.
- **Comprobación:** `validate()` compara el estado incremental con una reconstrucción
  desde cero. Los tests la llaman tras miles de eventos y ampliaciones de caja.

**Rendimiento medido** (Python puro, este portátil): ~75–85 mil eventos/s en Kossel
(6 vecinos) y ~31 mil eventos/s en β-hematina (14 vecinos). Un cristal de 10^5
unidades sale en minutos; 10^7 requiere la Fase 5.

### 4.6 Observables (`kmc3d/analysis.py`)

- Posición de cada cara, con dos medidas y distancia D = m·d_hkl:
  - el plano más externo m = max(h·r), seguido en el motor con un histograma de planos
    (O(1) por evento);
  - el plano medio del centro de la cara (`face_mean_series`), a partir de snapshots.
    Es el robusto (§9).
- Velocidad de cara: pendiente de D(t) en la mitad final (`face_growth_rates`).
- Forma: dimensiones principales del tensor de inercia, relaciones de aspecto y eje
  largo como dirección de red (`shape_descriptors`). Forma de Wulff cinética (con las
  R medidas) y de energía de unión, voxelizadas (`morphology_from_rates`) o como
  poliedro continuo (`wulff_shape_descriptors`).
- Superficie de una cara: mapa de alturas en una base entera unimodular de la cara
  (`face_frame`, `face_height_map`), analizado con **los mismos observables de
  `src.analysis`** que la línea 2D (rugosidad, islas, escalones, correlación).
- Probabilidad de unión por tipo de sitio (análogo 3D de la Fig. 8 de Nagpal):
  `attachment_probabilities`.

## 5. Arquitectura y reutilización

```
3DkMC/
├── PLAN_3DkMC.md            este documento
├── README.md                uso rápido
├── kmc3d/
│   ├── _paths.py            pone la raíz del repo en sys.path (para importar src)
│   ├── geometry.py          Cell, d_hkl, normales, índices de Miller
│   ├── graph.py             BondType, CrystalGraph, Hartman-Perdok
│   ├── presets.py           kossel, anisotropic_cubic, beta_hematin (+ celda y caras)
│   ├── rates.py             SplitRates (gilmer, symmetric), AdaptiveRates
│   ├── fenwick.py           árbol de Fenwick
│   ├── engine.py            KMC3D (motor BKL incremental)
│   ├── seeds.py             cubo, caja, esfera, poliedro (hkl)
│   ├── analysis.py          caras, forma, mapas de alturas, probabilidades
│   ├── viz.py               vóxeles triclínicos, proyecciones, D(t), mapas
│   └── io.py                save_run / load_run, config JSON -> motor
├── configs/                 kossel_smooth, anisotropic_needle, beta_hematin
├── scripts/run_3d.py        corridas y barridos desde un config
├── notebooks/01_primeros_resultados.ipynb
└── tests/                   47 tests, unos 5 s (python -m unittest discover -s tests)
```

| Se reutiliza de `src/` | Dónde |
|---|---|
| `src.common.numerics._safe_exp` | todas las tasas (`rates.py`) |
| `src.common.io.save_json`, `load_json`, `save_pickle`, `git_commit`, `find_repo_root` | `io.py` |
| `src.analysis.surface_observables` y el resto de observables 2D | `analysis.face_surface_observables` |
| `src.static.KMC_BKL_Static`, `LatticeSOSStatic`, `src.analysis.adsorption._rate_engine` | test de correspondencia con el 2D |
| Convención Δμ = S = ln(1 + σ), formato de carpetas de corrida (metadata + snaps + historial), configs JSON | `rates.delta_mu_from_sigma`, `io.py`, `scripts/run_3d.py` |

## 6. Fases

### F1: núcleo (hecho)

Entregables: `geometry`, `graph`, `rates`, `fenwick`, `engine`, `seeds`, `presets` y
sus tests.

| Criterio de validación | Resultado |
|---|---|
| Estado incremental = reconstrucción tras eventos y ampliaciones | OK (`test_engine`) |
| Conservación N = semilla + uniones − despegues | OK |
| Determinismo con la misma semilla | OK |
| Balance detallado de `SplitRates`; kink neutro a Δμ = 0 | OK (`test_rates`) |
| `AdaptiveRates` = `KMC_BKL_Static` donde coinciden las convenciones | OK |
| Kossel: {100} F, {110} S, {111} K | OK (`test_graph`) |
| Signo del crecimiento con Δμ; disolución completa bajo el tamaño crítico | OK (`test_physics`, `test_engine`) |
| Caras lisas sin avance a Δμ bajo, rugosas ∝ Δμ | OK (`test_physics`; curvas R(Δμ) en el notebook 01) |
| Tamaño crítico de un cubo de Kossel, L* = 2φ/Δμ | OK: la frontera crece/se disuelve sigue L* (notebook 01, §3.1) |
| Cubitos anisótropos: R(100) < R(010) < R(001), aguja según z | OK. Con el plano medio, R_rel = 1 : 1.66 : 3.04 frente a E_att 1 : 1.5 : 2.5 a Δμ = 1: la cinética amplifica la anisotropía, porque la cara lenta es lisa |

### F2: análisis por cara (hecho)

Hecho: `analysis.py`, `viz.py`, `io.py`, `scripts/run_3d.py`, configs y el notebook 01.
El **modo *slab*** está hecho en `kmc3d/slab.py` (`SlabKMC`, §11), con la celda de capa
de MONTY (`basis_graph.slice_cell`) en vez de `face_frame`.

Diseño original del **modo *slab***. Es una caja con contorno periódico en las dos direcciones
de una cara (hkl) (base de `face_frame`) y crecimiento solo hacia +w. Es el equivalente
3D de MONTY y del motor 2D. Sirve para:
- curvas R_hkl(Δμ) por cara sin efectos de aristas ni de tamaño finito;
- comparar con la línea 2D en (001), con la misma convención de conteo (D6).

Criterio: en (001) Kossel con `AdaptiveRates` y sin voladizos, ⟨h⟩(t) y W(t)
estadísticamente iguales a `KMC_BKL_Static` en los sitios comparables. **Pendiente**:
`SlabKMC` usa `MontyRates`; la comparación con `KMC_BKL_Static` no se ha hecho.

### F3: β-hematina (celda hecha; calibración con la morfología hecha, §7.5)

Hecho el 2026-11-11 (notebook 02, §4; `kmc3d/calibration.py`): opción (b) solo con la
morfología (puntos 1, 3 y 4). Falta la parte cinética (punto 2) y, sobre todo, la
opción (a).

1. Fijar el origen de las energías (D3). Opciones, de mejor a peor:
   (a) energías de par calculadas con un campo de fuerzas o DFT para cada vector de
   enlace; (b) ajuste a la morfología observada y a las velocidades AFM de la cara
   (100); (c) mantener las provisionales.
2. Calibrar Δμ y k0 con datos experimentales: velocidad de escalón y de cara en (100),
   relación de aspecto de las agujas.
3. Criterios: aguja según c; {100} y {010} laterales con anchura/espesor ≈ 3:1;
   extremos {011} > {001}; sin {0-11}.
4. Barridos de Δμ con varias semillas (`scripts/run_3d.py --delta-mu ... --seeds ...`).

### F4: física adicional

- **Difusión superficial: hecha** (2026-11-11; `rates.SurfaceDiffusion`, evento HOP en
  `engine.py`, `tests/test_diffusion.py`, notebook 02 §2).
  - Tasa por dirección k_d·w_δ·exp[−(E_i − E_kink)] (barrera = energía de enlace del
    origen, como Gilmer-Bennema), con balance detallado respecto de la energía de
    enlace. Las clases van por código de origen.
  - El destino se elige dentro de la clase y, si no es válido, el evento se rechaza
    (paso nulo, exacto). No es válido si está ocupado, si no tiene otro vecino
    ocupado, si cierra un hueco o si cruza una diagonal ocupada.
  - Vectores de salto `"bonds"`, `"cube18"` o `"cube26"` (D9).
  - Sin difusión, las trayectorias son idénticas bit a bit a las de antes.
  - Kossel: la difusión acelera sobre todo las caras lisas (φ = 3, Δμ = 1: R/k+ de 0.031
    a 0.157 con k_d = 10, ×5; φ = 2, Δμ = 0.5: ×2.2). Con `"bonds"` una caja perfecta
    no puede cambiar (se rechazan todos los saltos); con `"cube18"` una varilla de
    4×4×24 se compacta (L1/L2 de 6.2 a 1.0). Sin reservorio, con φ bajo, el cristal se
    evapora en un gas de red.
  - En β-hematina calibrada (§7.5), la difusión débil (k_d ≤ 0.1) no cambia la
    morfología y multiplica el coste por ~10.
  - Pendiente: comparar con el `r_m` del motor 2D (necesita el modo *slab*).
- **Reservorio finito** (como `src/dynamic`): Δμ(t) a partir de N_bulk(t). La API ya
  está (`set_delta_mu`); falta acoplarla en `run` sin recalcular todas las clases en
  cada evento (por ejemplo, actualizando cada n eventos o separando la parte de Δμ).
- **Aditivos** (cloroquina y otras quinolinas): bloqueo de sitios o reducción de k+ en
  caras concretas. Según Solomonov et al. (2007) se adsorben en {001} y {011}, los
  extremos de la aguja. Implementación: especie adsorbida en sitios de superficie, con
  sus propios eventos.
- **Tasas adaptativas en 3D** con δ por tipo de enlace (ya disponibles en
  `AdaptiveRates`) para trasladar el estudio de regímenes de la línea 2D.

### F5: rendimiento

- **Réplicas en paralelo: hechas** con `subprocess` (`kmc3d/parallel.py`).
  `multiprocessing` falla en el sandbox (`PermissionError` en `SemLock`) y "fork" da
  problemas en Jupyter.
- **Compilación** del bucle de eventos (numba o una extensión en C). Ganancia esperada
  de 50 a 100×, pero numba **no** está en las librerías permitidas: requiere el visto
  bueno del autor.
- Snapshots solo de la superficie para cristales grandes.

### F6: unidades más realistas

- **Base > 1**: hecha para el *slab* (`BasisGraph` + `SlabKMC`, §11), con el código de
  vecindad como máscara de bits de los enlaces (tabla de consulta de MONTY). Falta en
  `KMC3D` (cristal finito), que obliga a tocar `engine.py` y sus trayectorias congeladas.
  Diseño original: (varias unidades por celda: hemo en vez de dímero), con un grafo cuyos
  nodos son (celda, posición en la base). El criterio de cara F pasa a ser el de redes
  conexas de Hartman-Perdok/Grimbergen, y el rango ya no basta.
- **Grafo desde un CIF** y energías de par desde un campo de fuerzas (como MONTY).
- **Orientación de la unidad** (unidades no centrosimétricas, aditivos con
  orientación).

## 7. β-hematina

### 7.1 Celda y unidad

- P-1, a = 12.196 Å, b = 14.684 Å, c = 8.040 Å, α = 90.22°, β = 96.80°, γ = 97.92°
  (Pagola et al. 2000).
- La recíproca calculada coincide con la de la nota 26 de Solomonov et al. (2007),
  dentro de 3·10⁻⁵ Å⁻¹ y 0.1° (`test_geometry`).
- Z = 2 hemos forman un dímero centrosimétrico (Fe–O–propionato) por celda, que es la
  unidad de crecimiento.
- Los dímeros se unen por puentes de hidrógeno O–H···O=C en cadenas según [10-1]
  (a − c).

### 7.2 Morfología observada (objetivo)

- Agujas o láminas alargadas según **c**.
- Caras laterales **{100}** (la más desarrollada) y **{010}**, con anchura/espesor ≈ 3:1.
- Extremos **{011}** y, en menor medida, **{001}**.
- La {0-11} no aparece.

### 7.3 Energías efectivas provisionales (kT)

Se eligieron **solo** para que |E_att| ordene las caras como se observa. No salen de
ningún cálculo.

| Enlace | Longitud (Å) | φ/kT | Interpretación |
|---|---|---|---|
| [001] | 8.04 | 2.4 | traslación más corta, apilamiento de porfirinas: eje de la aguja |
| [10-1] | 15.38 | 1.0 | cadenas de puentes de H |
| [01-1] | 16.77 | 1.1 | |
| [010] | 14.68 | 0.9 | |
| [011] | 16.71 | 0.5 | |
| [100] | 12.20 | 0.4 | |
| [101] | 13.79 | 0.3 | |

E_cr = 6.6 kT. Teoría de energía de unión (`CrystalGraph.face_table`):

| Cara | d_hkl (Å) | E_att (kT) | R_att relativa | Carácter |
|---|---|---|---|---|
| (100) | 11.99 | −1.7 | 1.00 | F |
| (010) | 14.54 | −2.5 | 1.47 | F |
| (110), (1-10) | 8.68, 9.96 | −4.2 | 2.47 | S |
| (101) | 6.30 | −4.7 | 2.76 | F |
| (011) | 6.94 | −5.1 | 3.00 | F |
| (001) | 7.98 | −5.3 | 3.12 | F |
| (10-1) | 7.05 | −5.4 | 3.18 | F |
| (01-1) | 7.06 | −5.7 | 3.35 | F |

### 7.4 Primera exploración kMC (notebook 01, §5)

Semilla esférica de 70 Å, crecimiento hasta N = 40 000, Δμ ∈ {0.75, 1, 1.5, 2.5} y
tasas Gilmer o simétricas.

- **Con las energías provisionales, el kMC NO da una aguja.** Sale una lámina
  (L2/L3 ≈ 2 a Δμ ≤ 1), con {100} grande y poco alargada (L1/L2 ≈ 1.1–1.3). El eje
  largo está a unos 20° de c. En cambio, la forma de energía de unión sí es una aguja
  según c (L1/L2 ≈ 2).
- El motivo es que la cinética no sigue a |E_att|. La {010} avanza más de 3 veces más
  rápido que la {100} (E_att predice 1.5), y las caras S {110}/{1-10} y la {101} son
  lentas. Estas caras recortan la aguja.
- **La anisotropía cae con Δμ** (rugosidad cinética): a Δμ = 2.5 casi todas las caras
  avanzan a la misma velocidad.
- **Sensibilidad (Δμ = 1, simétricas).** Con [001] = 3.5 sale L1/L2 ≈ 2 según c. Si
  además se multiplican todas las energías por 1.3, sale una aguja según c con
  L1/L2 ≈ 6. En este último caso, la (100) casi no crece y el espesor es el de la
  semilla: la forma no es estacionaria.

Conclusión para F3: calibrar las energías con la morfología obliga a usar el kMC, no
solo E_att, y la escala global de las energías (temperatura efectiva) importa tanto
como su reparto.

### 7.5 Calibración con la morfología (notebook 02, §4)

Método (`kmc3d/calibration.py`):
- tasas simétricas, semilla esférica de 70 Å, crecimiento hasta N = 40 000;
- medidas sobre el fragmento mayor: dimensiones √(12·var) según c, la anchura w y el
  espesor (normal a (100)), y fracción de área por familia con `analysis.facet_areas`;
- 6 criterios (aguja según c, 2 ≤ W/T ≤ 4.5, laterales {100}+{010} con {100} mayor,
  extremos {011} > {001} y ≥ 50 % de las caras que cortan c, {0-11} < 3 %, forma
  estacionaria entre N/2 y N) y una pérdida J para desempatar.

Resultados:
- **Energías provisionales (Δμ = 1.5): 2 de 6.** Sale una lámina con el eje largo a
  43° de c y extremos (101).
- **Mecanismo.**
  - (101) tiene en su plano [10-1] y [010]: es una cara F lenta.
  - (011) solo es F si [100] es fuerte.
  - [01-1] corta (010), (101), (001) y (01-1), pero no (100) ni (011): sube W/T y
    elimina {0-11}.
  - Ningún cambio de un solo enlace cumple los 6 criterios.
- **Rejilla de 270 corridas** (Δμ, [001], [01-1], [100], [101]): 7 cumplen 6/6 con la
  semilla 1. Solo uno cumple con las 5 semillas:

  | | Δμ | [001] | [10-1] | [100] | [010] | [011] | [01-1] | [101] |
  |---|---|---|---|---|---|---|---|---|
  | provisional | — | 2.4 | 1.0 | 0.4 | 0.9 | 0.5 | 1.1 | 0.3 |
  | calibrada (efectiva) | 1.5 | 3.0 | 1.0 | **1.6** | 0.9 | 0.5 | **2.3** | 0.3 |

  Con N = 40 000: W/T = 2.16 ± 0.10, L_c/W = 1.81, eje a ~10° de c,
  ({011}+{001})/extremos = 0.58 y deriva de W/T 0.07.
- **Tamaño.** Con N = 120 000 (3 semillas) se mantienen los 6 criterios: W/T = 2.43,
  L_c/W = 1.84–1.95 y ({011}+{001})/extremos = 0.62–0.64. W/T sigue subiendo despacio.
- **E_att con las energías calibradas** ordena las caras como pide la observación:
  (100) < (010) < (011) < (110) < (001) < (01-1) < (101) < (10-1).
- **Difusión débil** (k_d ≤ 0.1): 7 de 8 corridas siguen cumpliendo los 6 criterios.
- **El Wulff cinético con las R de plano medio no reproduce el cristal kMC**: da
  (110)/(1-10) grandes y ninguna (010). Hipótesis: en caras no desarrolladas, el plano
  medio mide una arista redondeada y subestima R. Por eso la calibración usa las facetas
  del cristal. Queda pendiente revisar esa medida (riesgo "posición de cara", §9).

Límites:
- Los criterios son desigualdades cualitativas: hay una familia de soluciones. Otros
  candidatos con [01-1] = 2.9–3.5 y [100] = 1.2–1.6 cumplen 5–6 criterios según la
  semilla.
- W/T no llega a 3 en ningún candidato robusto: subir [01-1] lo acerca, pero la (100)
  deja de crecer y la forma deja de ser estacionaria.
- N = 120 000 son unos 55 nm, lejos de las micras reales.
- Las energías siguen siendo **efectivas y provisionales**. La predicción comprobable
  es que las interacciones según [100] y [01-1] son más fuertes que en el conjunto
  provisional, comparables o mayores que la cadena de puentes de H [10-1]. Lo puede
  comprobar un cálculo con campo de fuerzas (opción (a) de D3).

## 8. Decisiones abiertas para el autor

| | Decisión | Opciones | Por defecto ahora |
|---|---|---|---|
| D1 | Unidad de crecimiento de β-hematina | dímero (una por celda) / hemo (base 2, F6) | dímero |
| D2 | Modelo de tasas | Gilmer / simétrico (tipo MONTY) / intermedio / adaptativo (Nagpal) | Gilmer para validar Kossel; simétrico para β-hematina |
| D3 | Origen de las energías de enlace | campo de fuerzas / ajuste a morfología y AFM / provisionales | provisionales (§7.3) |
| D4 | ¿Se llenan los huecos cerrados? | no (la solución no llega) / sí | no en `KMC3D` (`allow_enclosed=False`); sí en `SlabKMC`, como MONTY (decidido el 2026-11-24, §11.2) |
| D5 | Fuerza impulsora | Δμ = ln(1 + σ) constante / reservorio finito (F4) | constante |
| D6 | Conteo de enlaces al comparar con el 2D | geométrico (3D) / "altura exactamente igual" (2D) | cada línea con la suya; el 2D no se toca |
| D7 | Escala de tiempo | k0 = 1 (tiempo en 1/k0) / calibrada con velocidades AFM | k0 = 1 |
| D8 | Unidades sueltas (sin vecinos) | se despegan con su tasa / se eliminan | se despegan con su tasa |
| D9 | Vectores de salto de la difusión | enlaces / + diagonales de cara (cube18) / + diagonales de cubo (cube26) | `"bonds"`. Con solo enlaces una unidad no puede pasar de una cara a la vecina: las formas no se relajan (notebook 02, §2.3). Para relajar formas, `"cube18"` |
| D10 | Barrera de salto | energía del origen (Gilmer-Bennema) / simétrica (origen y destino a medias) / con barrera extra | origen. Con tasas simétricas y E_kink ≈ 10 kT (β-hematina) los adátomos débiles saltan mucho más de lo que se despegan, y el coste se dispara (§7.5) |

## 9. Riesgos y limitaciones

- **Velocidad:** en Python puro los cristales simulables tienen de 10^5 a 10^6
  unidades, decenas de unidades de lado, muy lejos de las micras reales. Las
  relaciones de aspecto de un cristal pequeño están afectadas por aristas y por el
  tamaño crítico.
- **Rugosidad cinética:** a Δμ alto todas las caras se vuelven rugosas y la
  anisotropía se comprime. Hay que comparar a Δμ comparables a los experimentales.
- **Posición de cara:** el plano más externo (el que sigue el motor) sube con la
  rugosidad y con unidades sueltas, y sobrestima R de 1.5 a 3 veces en caras rugosas.
  Para velocidades se usa el plano medio del centro de la cara (`face_mean_series`, a
  partir de snapshots). Las formas de Wulff se calculan sobre el poliedro continuo
  (`wulff_shape_descriptors`).
- **No es SOS:** en una cara rugosa hay más de un sitio de unión por columna (los lados
  de los salientes). Con tasas de Gilmer cada sitio recibe k+, así que R/k+ supera la
  ley de Wilson-Frenkel de SOS (notebook 01, §3.2). Es una propiedad del modelo de gas
  de red 3D, no un error, pero hay que tenerla en cuenta al comparar con la línea 2D.
  Las tasas simétricas penalizan los sitios poco coordinados y la reducen.
- **Semilla:** si una cara casi no crece (lisa a Δμ bajo), su espesor queda fijado por
  la semilla y la forma no es estacionaria. Hay que comprobarlo antes de comparar
  relaciones de aspecto.
- **Difusión superficial:** el coste por unidad incorporada sube mucho con k_d (saltos
  y rechazos). Sin un reservorio, con φ bajo y difusión el cristal se fragmenta: es el
  gas de red en volumen infinito. Las formas se miden sobre el fragmento mayor
  (`analysis.largest_cluster`).
- **Energías provisionales:** cualquier conclusión cuantitativa sobre β-hematina
  depende de D3.
- **Observables 2D sobre recortes de cara:** `src.analysis` supone contorno periódico.
  En un recorte no periódico, los observables con vecinos (escalones, islas,
  correlación) tienen un error de borde O(1/L).

## 10. Referencias

- A. B. Bortz, M. H. Kalos, J. L. Lebowitz, *J. Comput. Phys.* **17**, 10 (1975).
  Algoritmo n-fold way.
- G. H. Gilmer, P. Bennema, *J. Appl. Phys.* **43**, 1347 (1972). kMC de Kossel con
  difusión superficial.
- P. Hartman, W. G. Perdok, *Acta Cryst.* **8**, 49 (1955). Teoría PBC, caras F/S/K.
- P. Hartman, P. Bennema, *J. Cryst. Growth* **49**, 145 (1980). Energía de unión y
  velocidad de cara.
- R. F. P. Grimbergen, H. Meekes, P. Bennema, C. S. Strom, L. J. P. Vogels,
  *Acta Cryst. A* **54**, 491 (1998). Redes conexas y rugosidad.
- S. X. M. Boerrigter et al., *J. Phys. Chem. A* **108**, 5894 (2004). MONTY.
- M. W. Anderson et al., *Nature* **544**, 456 (2017). CrystalGrower.
- S. Pagola, P. W. Stephens, D. S. Bohle, A. D. Kosar, S. K. Madsen, *Nature* **404**,
  307 (2000). Estructura de β-hematina.
- I. Solomonov et al., *J. Am. Chem. Soc.* **129**, 2615 (2007). Morfología de
  β-hematina, energías de unión y efecto de las quinolinas.
- K. N. Olafson, M. A. Ketchum, J. D. Rimer, P. G. Vekilov, *PNAS* **112**, 4946 (2015).
  Crecimiento por capas de hematina en (100) (AFM) e inhibición por cloroquina:
  candidato para calibrar en F3.
- Nagpal et al. (2024), `references/modern_kMC/`. Tasas adaptativas del modelo 2D.
- F. F. A. Hollander et al., *J. Phys. Chem. B* **103**, 8301 (1999) y **107**, 5680
  (2003). Grafos cristalinos y análisis de redes conexas de las grasas (refs. 16 y 17
  de MONTY). Los grafos de la validación salen de la de 2003 (§11.4,
  `references/papers/Hollander2003/`).

## 11. Validación con MONTY (2026-11-24)

Objetivo: reproducir las curvas de fracción de adhesión S(Δμ) por cara de las tres
grasas de MONTY (Figs. 2–4 de Boerrigter et al. 2004) sin tocar nada de lo anterior.
Resultados en `notebooks/03_validacion_monty.ipynb` (salidas en
`outputs/03_validacion_monty/`) y resumen en `docs/source/monty.md`.

### 11.1 Código nuevo

| Archivo | Contenido |
|---|---|
| `kmc3d/basis_graph.py` | `BasisGraph`: grafo con Z GU por celda (formato `.cfg` de MONTY), kcal/mol ↔ kT, base y enlaces desde la simetría (`from_centers`), energías de capa por corte (`slice_energies`, `best_termination`, `face_table`), celda de capa unimodular y reducida (`slice_cell`) |
| `kmc3d/slab.py` | `SlabKMC`: superficie (hkl) periódica (modo *slab*, F2), clases (subred, código) con el código como máscara de bits de los enlaces, caja que crece por arriba y por abajo; `MontyRates(λ1, λ2)` |
| `kmc3d/monty.py` | `sticking_point` y `sticking_grid` (para `run_parallel`), umbrales, referencia y comparación por grupos |
| `scripts/run_monty.py`, `scripts/digitalizar_monty.py` | barridos reanudables; digitalización de las Figs. 2–4 |
| `data/monty2004/` | `referencia_figuras.csv` y grafos de las grasas (`<grasa>.json`, `comprobacion_grafos.csv`) |
| `scripts/transcribir_grasas.py` | transcripción de las Tablas 2 y 3 de Hollander et al. 2003 y comprobación con redes conexas |
| `tests/test_basis_graph.py`, `tests/test_slab.py`, `tests/test_grasas.py` | 26 tests nuevos (100 en total) |

No cambia `engine.py`, `rates.py`, `graph.py`, `seeds.py`, `presets.py` ni `src/`:
`TestNoDiffusionUnchanged` sigue pasando.

### 11.2 Correspondencia con MONTY

- Tasas (ecs. 11–12 con U_moth = 0, ΔV = 0): `SplitRates(a_mu = 1 − λ2, b_E = λ1)`
  sobre x = E − E_cr, con E_cr = Σφ/Z.
- Reglas STS (§2.3 del artículo): se une con al menos un enlace (también en un hueco
  cerrado: el texto admite inclusiones) y no se despega una GU totalmente enlazada.
  Equivalen a `allow_enclosed=True`. Con λ = ½ y la misma energía total en todas las
  subredes, estas reglas hacen la dinámica **simétrica partícula-hueco**: crecer (hkl) a
  +Δμ es lo mismo que disolver (−h−k−l) a −Δμ. Comprobado en el notebook 03, §2.3.
- **Decisión del autor (2026-11-24): el slab sigue a MONTY.** `SlabKMC` y
  `monty.sticking_point` usan `allow_enclosed=True` por defecto. `KMC3D` sigue con
  `False` (D4). La regla de `KMC3D` rompe la simetría: en Kossel casi no se nota, pero
  en un P1 polar de juguete cambia S hasta 0.14.

### 11.3 Parte A: metodología (hecha)

Resumen; los números están en el notebook 03, §4.
- `MontyRates` coincide con `SplitRates` y el balance detallado es exacto.
- S(Δμ = 0) → 0 como 1/N (cara lisa) o 1/√N (rugosa).
- La simetría partícula-hueco se cumple al 0.2 % en Kossel y con |z| ≤ 1 en el P1.
- Kossel se comporta según Hartman-Perdok:
  - umbral de (001) en Δμ(S = 0.1) = 0.24–1.85 kT para φ = 2–5;
  - (011) y (111) crecen desde Δμ = 0, cerca del límite de kinks 1 − e^(−Δμ).
- El γ de Onsager queda entre las dos estimaciones de nucleación 2D.
- El slab coincide con el cristal finito de `KMC3D` cuando este es grande: 98–99 % con
  L = 48, frente a 65–78 % con L = 12.
- La supercelda da las mismas curvas dentro de un 1 %.
- Las caras opuestas coinciden en un P-1. En un P1 polar se separan un 5 % en S = ½,
  pero no en el umbral: con enlaces de par, las caras opuestas rompen los mismos
  enlaces. La prueba de MONTY comprueba el código más que la física.

### 11.4 Parte B: grasas (barridos hechos el 2026-11-25)

- Referencia: las Figs. 2–4 digitalizadas (`data/monty2004/referencia_figuras.csv`).
- Grafos: transcritos de Hollander et al. 2003 (ref. 17, Tablas 1–3) con
  `scripts/transcribir_grasas.py` a `data/monty2004/<grasa>.json`. Decisiones y erratas
  del artículo en `data/monty2004/README.md`.
  - β-16.16.16, β′-10.12.10 y β′-16.16.14: Z = 2, 8 y 8 GU por celda; 8, 32 y 48
    enlaces; E_cr = 88.25, 63.30 y 91.61 kcal/mol.
  - La E_att de la red conexa más fuerte reproduce la Tabla 3:
    - β′-10.12.10: las 5 caras entre −1.8 % y +2.9 %;
    - β′-16.16.14: 12 de 13 caras dentro del 0.4 % (la (112) es una errata probable);
    - β-16.16.16: entre +0.2 % y +2.9 % (la Tabla 3 se calculó con otras energías).
  - Morfología de energía de unión (fila superior de la Fig. 1; notebook 03, §3.2).
    MONTY la calculó con Cerius², con cortes planos y el campo de fuerzas completo.
    Aquí se calcula solo con los enlaces de la Tabla 2, de dos formas: con el mejor
    corte plano (|h|, |k| ≤ 3, |l| ≤ 5) y con las redes conexas. Las tres formas salen
    con el mismo aspecto que en la Fig. 1: placa alargada, aguja y hexágono.
    - β-16.16.16: las superiores coinciden ({2-1-3} y {1-1-1}). Las laterales salen
      {10-1} (52.6 kcal/mol) en vez de {20-3} + {30-5} (111.3); {10-1} es la que dan
      las redes conexas y las simulaciones.
    - β′-10.12.10: la basal {002} y las laterales {101} coinciden. Las superiores
      {011} y {110} tienen la misma E_att que {112} (179.2).
    - β′-16.16.14: salen {111} (196.0) en vez de {110} (199.1) y {100} o {203} en vez
      de {60-2} (301).
    - La versión con redes conexas da las formas de la Tabla 3 de Hollander.
- Corrección numérica en `SlabKMC`. No cambia la física.
  - Problema: con las grasas, las tasas de las clases van de 1e-35 a 1e13. Al restar
    tasas grandes, el total del árbol de Fenwick perdía precisión (cancelación
    catastrófica), hasta elegir una clase vacía (`IndexError`).
  - Solución: el árbol se reconstruye desde los pesos exactos cuando la mayor tasa
    retirada supera 10⁶ veces el total.
  - Test: `test_slab.TestSlabPhysics.test_rounding_with_strong_bonds`.
  - `KMC3D` (`engine.py`, congelado) actualiza su árbol de la misma forma y no se tocó.
    Si la clase elegida está vacía, ya recurre a la selección exacta, pero no corrige
    la pérdida de precisión del total ni un índice fuera de rango. Solo importa si las
    tasas abarcan muchos órdenes de magnitud.
- Barridos con el protocolo del artículo (`outputs/03_validacion_monty/barridos/<grasa>.csv`,
  notebook 03, §3.3):
  - la semilla 1 cubre toda la rejilla de Δμ;
  - las semillas 2 y 3 cubren la zona de subida de cada curva
    (`run_monty.py --solo-subida 0.02 0.98`), que decide los cruces;
  - entre semillas, Δμ(S = ½) cambia con una desviación típica de 0.002–0.03 kcal/mol
    (mediana por grasa) y de 0.23 como máximo, en una basal.

  Δμ en kcal/mol en S = ½, media de las 3 semillas; para las basales, en S = 0.1,
  comparado con el umbral del texto:

  | Grasa | Grupo: caras | MONTY | kmc3d |
  |---|---|---|---|
  | β-16.16.16 | 1: (1-11) | 0.95 | 0.93 |
  | | 2: (-110) (1-10) | 2.25 | 2.05 |
  | | 3: (-111) (1-1-1) | 4.53 | 4.28 |
  | | 4: (2-1-3) (2-1-2) | 9.34 | 9.44 |
  | | 5: (0-10) (0-11) (010) (2-1-1) (2-10) (0-1-1) | 9.59 | 9.60 |
  | | 6: (100) (10-1) (-101) (20-3) (30-5) | 15.80 | 15.81 |
  | | 7: (001), S = 0.1 | 60 | 59.6 |
  | β′-10.12.10 | 1: (011) (013) (110) (112) (310) (610) (611) | 0.35 | 0.60 (0.46–0.70) |
  | | 2: (211) | 4.97 | 4.91 |
  | | 3: (101) (200) (201) | 14.64 | 14.38 |
  | | 4: (002) | 46.07 | 41.41 |
  | β′-16.16.14 | 1: (020) (0-21) (0-22) (0-11) | 0.54 | 0.48 |
  | | 2: (310) (313) (314) (315) (316) (317) | 2.20 | 1.85 |
  | | 3: (111) (112) (113) (1-14) | 3.17 | 3.77 (2.80–4.10) |
  | | 4: (110) (11-1) | 4.46 | 4.09 |
  | | 5: (205) (204) | 5.57 | 5.80 |
  | | 6: (203) | 5.86 | 6.22 |
  | | 7: (100) (202) (201) (30-1) | 6.90 | 6.80 |
  | | 8: (001), S = 0.1 | 62 | 62.2 |

- Criterios:
  - objetivo (±20 % o ±1 kcal/mol): lo cumplen los 19 grupos;
  - orden de los grupos: el mismo en las tres grasas;
  - caras rugosas con S > ½ por debajo de 2 kcal/mol y basales muy por encima: sí;
  - caras opuestas de β-16.16.16: coinciden ((-110)/(1-10): 2.06/2.05;
    (-111)/(1-1-1): 4.28/4.29);
  - cocientes, con S = 0.1: kmc3d da 0.242, 0.327, 0.557–0.568 y 0.278 para
    13 : 60, 12 : 43, 7 : 13 y 2 : 12. Las figuras de MONTY dan 0.242, 0.299,
    0.546–0.569 y 0.280, en el mismo orden. Con los números del texto, 2 : 12 pasa delante
    de 13 : 60, porque el texto da 2 kcal/mol para (211) y su Fig. 3 da 3.7;
  - mismos grupos: no del todo en β′-16.16.14 (ver abajo).
- Lo que no cuadra:
  - **β′-16.16.14, grupos 2 a 4.** Entre 1 y 5 kcal/mol, MONTY dibuja cuatro curvas
    (S = ½ en 1.91, ~2.35, 3.17 y 4.46) y kmc3d tres:
    - (310)–(317), todas iguales, en 1.85;
    - (1-14), sola, en 2.80;
    - (111), (112) y (113) en 4.09–4.10, junto con (110) y (11-1), cuando MONTY las
      etiqueta con (1-14).

    Estas cinco caras están en la zona [1-10] y sus mejores cortes planos tienen casi la
    misma energía (E_att 196–199 kcal/mol), aunque la red conexa más fuerte de (110) da
    254.7. La causa no está resuelta. La topología la fijan la Tabla 2 y la simetría
    (reproduce la Tabla 3), y las posiciones reconstruidas solo deciden el corte
    inicial. MONTY pudo usar otras energías, como en β-16.16.16, u otro grafo. No es
    ruido: con 3 semillas, (111) y (110) quedan a 0.01 kcal/mol y (1-14) a 1.3.
  - **β′-10.12.10, (002):** −10 % (41.4 frente a 46.1 en S = ½; 40.1 frente a 44.6
    en S = 0.1, y el texto da 43). Las otras dos basales coinciden dentro del 1 %. El
    etiquetado elegido (punto 4 del README de los datos) solo cambia a qué GU de la capa
    inferior va el enlace s, y en (002) cada GU tiene igualmente un enlace s hacia
    abajo, así que no debería influir. No es ruido: entre semillas, Δμ(S = ½) tiene una
    desviación típica de 0.13 kcal/mol.
- Sensibilidad en β-16.16.16. Con las energías compatibles con la Tabla 3 (p = 31.35,
  r2 = 28.3), los grupos 1–3 empeoran: 0.80, 1.85 y 3.88, frente a 0.95, 2.25 y 4.53 en
  MONTY y 0.93, 2.08 y 4.30 con la Tabla 2. Los grupos 4–6 casi no cambian. MONTY usó,
  con toda probabilidad, las energías de la Tabla 2, que son las de los JSON.
- Pendiente: la causa de las dos discrepancias.
- Protocolos del artículo:
  - β-16.16.16: 40 × 40, 10⁵ + 10⁵ eventos, 313 K;
  - β′: 50 × 50, 2·10⁵ + 10⁶ eventos, 300 K;
  - en todos, λ1 = λ2 = ½.
- Criterios de aceptación:
  - obligatorios: mismos grupos y mismo orden de umbrales; las caras rugosas con
    S > ½ por debajo de unos 2 kcal/mol; las basales muy por encima; las caras opuestas
    coinciden en β-16.16.16; los cocientes 13 : 60, 12 : 43, 7 : 13 y 2 : 12 tienen el
    mismo orden;
  - objetivo: Δμ(S = ½) por grupo dentro de ±20 %, o de ±1 kcal/mol cerca de 0.
- Regla de los huecos cerrados: la de MONTY (`allow_enclosed=True`), ya por defecto en
  el slab (§11.2).
