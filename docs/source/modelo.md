# Modelo

Resumen del modelo que implementa `kmc3d`. La deducción completa, los supuestos y las
figuras de comprobación están en el documento teórico
{download}`teoria_3DkMC.pdf <../../teoria/teoria_3DkMC.pdf>`, y las decisiones de
diseño en el [plan](plan.md).

## Red y unidades de crecimiento

El cristal vive en una red de Bravais con celda `Cell(a, b, c, α, β, γ)`
(`kmc3d.geometry`). El motor trabaja con **índices enteros** $(i, j, k)$: cada sitio
de la red es una unidad de crecimiento, ocupada o vacía. La geometría real solo entra
en el espaciado interplanar $d_{hkl}$, las normales de las caras y las coordenadas
cartesianas de las figuras.

Hay una unidad de crecimiento por celda («cubito» en el espacio de índices). Para
β-hematina esto es exacto en la topología: la celda P-1 contiene $Z = 2$ hemos, que
forman un dímero centrosimétrico (ver [](beta_hematina.md)).

## Grafo de enlaces y energía de un sitio

Cada tipo de enlace $t$ es un vector de red $\mathbf{v}_t$ con energía
$\varphi_t > 0$ (en $kT$), y une el sitio $\mathbf{r}$ con $\mathbf{r} \pm
\mathbf{v}_t$ (`kmc3d.graph`). El estado de un sitio es su **código**

$$
\text{código} = \sum_t n_t \, 3^t, \qquad n_t \in \{0, 1, 2\},
$$

donde $n_t$ es el número de vecinos ocupados del tipo $t$. Su energía de enlace es

$$
E = \sum_t n_t \, \varphi_t .
$$

La energía de red por unidad y la de la posición de media-cristal (*kink*) coinciden:

$$
E_{cr} = E_{kink} = \sum_t \varphi_t .
$$

## Teoría de Hartman-Perdok

Para una cara $(hkl)$ (índices coprimos), un enlace está dentro de la capa $d_{hkl}$
si $(hkl) \cdot \mathbf{v}_t = 0$. Eso reparte la energía de red en

$$
E_{slice}(hkl) = \sum_{t:\ hkl \cdot \mathbf{v}_t = 0} \varphi_t, \qquad
E_{att}(hkl) = \sum_{t:\ hkl \cdot \mathbf{v}_t \neq 0} \varphi_t, \qquad
E_{slice} + E_{att} = E_{cr}.
$$

El modelo de energía de unión supone $R_{hkl} \propto |E_{att}|$ (Hartman y Bennema,
1980) y da la forma «de energía de unión»
(`analysis.attachment_energy_morphology`).

Carácter de la cara: con una unidad por celda, cada tipo de enlace forma una cadena
periódica. La cara es **F** (plana) si los enlaces de la capa con $\varphi > 0$
tienen rango 2, **S** (escalonada) si tienen rango 1 y **K** (con *kinks*) si no hay
ninguno. En Kossel: {100} F, {110} S, {111} K.

## Tasas

Unir una unidad a un sitio de energía $E$ cambia la energía libre en

$$
\Delta G = -\left[\Delta\mu + (E - E_{kink})\right], \qquad \Delta\mu = \ln(1 + \sigma),
$$

con la misma convención de sobresaturación $\sigma$ que la línea estática 2D.

**`SplitRates(a, b)`** (`kmc3d.rates`) reparte $\Delta\mu$ y $E - E_{kink}$ entre la
unión y el despegue:

$$
k^+ = k_0 \, e^{\,a\,\Delta\mu + b\,(E - E_{kink})}, \qquad
k^- = k_0 \, e^{-(1-a)\,\Delta\mu - (1-b)\,(E - E_{kink})},
$$

de modo que siempre se cumple el balance detallado, $k^+/k^- = e^{-\Delta G}$.

- `gilmer` ($a = 1$, $b = 0$): la unión no depende del sitio (Gilmer y Bennema, 1972).
- `symmetric` ($a = b = 1/2$): reparto simétrico, del tipo de MONTY y CrystalGrower.
- $b > 0$ en general: la unión es selectiva por sitio sin romper el balance detallado.

**`AdaptiveRates`** reproduce la forma del motor estático 2D (Nagpal et al.),
$k^+ = K_0 \exp(S + \sum_t n_t \delta_t / S)$ y $k^- = K_0 \exp(\varphi_0 - E)$. Sirve
solo para comparar con la línea (2+1)D y **no** cumple balance detallado.

## Difusión superficial (opcional)

Con `SurfaceDiffusion`, una unidad de superficie en $i$ puede saltar a un sitio vacío
$f = i \pm \boldsymbol{\delta}$ con tasa

$$
k_{salto}(i \to f) = k_d \, w_{\delta} \, e^{-(E_i - E_{kink})},
$$

donde $E_i$ es la energía de enlace en el origen y $w_\delta$ un peso por vector de
salto. El salto inverso usa la energía en $f$ sin contar a $i$, así que
$k(i \to f)/k(f \to i) = e^{-\Delta E}$: el mismo equilibrio que unión y despegue. No
depende de $\Delta\mu$ y conserva el número de unidades.

Los vectores de salto pueden ser los de enlace (`"bonds"`, por defecto), los 18
vecinos de cara y arista del cubo de índices (`"cube18"`), los 26 (`"cube26"`) o una
lista propia. Si el destino elegido no es válido (ocupado, sin otro vecino, hueco
cerrado), el paso es nulo: avanza el tiempo y la configuración no cambia. Sin
difusión, las trayectorias son idénticas bit a bit a las del motor sin saltos.

## Algoritmo (BKL, `kmc3d.engine.KMC3D`)

**Eventos.** Unión en un sitio vacío con al menos un vecino ocupado, despegue de una
unidad con al menos un vecino vacío y, con difusión, salto. Por defecto un hueco con
todos los vecinos ocupados no recibe unidades (`allow_enclosed=False`).

**Clases (*n-fold way*).** Todos los sitios con el mismo código y tipo de evento
tienen la misma tasa $r_c$ y forman una clase. El peso $n_c r_c$ de cada clase se
guarda en un árbol de Fenwick, así que elegir la clase cuesta $O(\log K)$. El árbol se
recalcula desde cero cada 20 000 eventos para evitar la deriva de redondeo.

**Un paso.**

1. $W = \sum_c n_c r_c$ y $\Delta t = -\ln(u_1)/W$ (por `time_scale`).
2. Se elige la clase $c$ con probabilidad $n_c r_c / W$ y un sitio uniforme dentro de
   ella.
3. Se ejecuta el evento, se actualizan los códigos de los $2T$ vecinos ($\pm 3^t$) y
   se reclasifican $2T + 1$ sitios.

El coste por evento es $O(T + \log K)$ e **independiente del tamaño del cristal**.

**Caja.** El cristal vive en una caja con margen vacío. Si una unidad se une cerca del
borde, la caja se amplía por ese lado. Las coordenadas devueltas son relativas al
centro de la semilla.

**Reproducibilidad.** Los números aleatorios salen de `np.random.default_rng(seed)`
en bloques: la misma semilla da la misma trayectoria. `validate()` compara el estado
incremental con una reconstrucción desde cero; los tests la llaman tras miles de
eventos.

**Rendimiento** (Python puro): unos 80 000 eventos/s en Kossel (6 vecinos) y unos
31 000 en β-hematina (14 vecinos).

## Observables (`kmc3d.analysis`)

- **Posición de cada cara**, como distancia $D = m \, d_{hkl}$: el plano más externo,
  que sigue el motor, o el plano medio del centro de la cara (`face_mean_series`), que
  es la medida robusta. La velocidad $R_{hkl}$ es la pendiente de $D(t)$ en la mitad
  final (`face_growth_rates`).
- **Forma**: dimensiones principales del tensor de inercia, relaciones de aspecto y
  eje largo (`shape_descriptors`); forma de Wulff cinética con las $R$ medidas
  (`wulff_faces`, `morphology_from_rates`); facetas y áreas del cristal kMC
  (`facet_areas`).
- **Superficie de una cara**: mapa de alturas en una base entera de la cara
  (`face_height_map`), analizado con los mismos observables 2D que el modelo (2+1)D
  (`face_surface_observables`, ver [](codigo_reutilizado.md)).
- **Probabilidad de unión por tipo de sitio**, el análogo 3D de la Fig. 8 de Nagpal
  et al. (`attachment_probabilities`).

## Validación física

El modelo reproduce, y los tests comprueban, que $\Delta\mu < 0$ disuelve y
$\Delta\mu > 0$ hace crecer. También que un cristal finito necesita superar un tamaño
crítico (Gibbs-Thomson): en un cubo de Kossel, $L^* \approx 2\varphi/\Delta\mu$.

```{figure} ../../outputs/01_primeros_resultados/kossel_tamano_critico.png
:width: 75%
:alt: Mapa de cubos de Kossel que crecen o se disuelven según su lado inicial y Δμ

Cubos de Kossel con $\varphi/kT = 2$: los que crecen (círculos) y los que se disuelven
(cruces) quedan separados por la curva $L^* = 2\varphi/\Delta\mu$ (notebook 01, §3.1).
```

Las caras lisas ($\varphi$ alto) casi no avanzan a $\Delta\mu$ bajo, porque necesitan
nuclear una isla 2D, mientras que las rugosas avanzan desde el principio.

```{figure} ../../outputs/01_primeros_resultados/kossel_R_vs_dmu.png
:width: 90%
:alt: Velocidad de las caras {100} de Kossel frente a Δμ para tres energías de enlace

Velocidad de {100} en Kossel con tasas de Gilmer, medida con el plano medio y con el
plano externo, frente a la ley de Wilson-Frenkel del modelo SOS (notebook 01, §3.2).
En caras rugosas el 3D supera a Wilson-Frenkel porque hay más de un sitio de unión
por columna: es una propiedad del gas de red 3D, no un error.
```

Con enlaces anisótropos ($\varphi_z > \varphi_y > \varphi_x$), los cubitos crecen como
una aguja según $z$, con $R(100) < R(010) < R(001)$.

```{figure} ../../outputs/01_primeros_resultados/cubitos_anisotropos.png
:width: 90%
:alt: Distancia de cada cara al centro frente al tiempo y cristal anisótropo resultante

Cubitos anisótropos: plano medio de cada cara frente al tiempo y cristal final
(notebook 01, §4).
```

## Limitaciones conocidas

- En Python puro los cristales simulables tienen de $10^5$ a $10^6$ unidades, decenas
  de nanómetros, lejos de las micras reales.
- A $\Delta\mu$ alto todas las caras se vuelven rugosas y la anisotropía se comprime.
- El plano más externo sobrestima $R$ en caras rugosas; por eso se usa el plano
  medio.
- Si una cara casi no crece, su espesor queda fijado por la semilla y la forma no es
  estacionaria.

La lista completa está en el §9 del [plan](plan.md).
