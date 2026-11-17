# β-hematina

:::{warning}
Todas las energías de enlace de esta página son **efectivas y provisionales**. Las de
`kmc3d/presets.py` se eligieron solo para ordenar las caras, y las calibradas solo
reproducen cualitativamente la morfología. Ninguna sale de un cálculo con campo de
fuerzas. No deben presentarse como resultados cuantitativos sobre β-hematina.
:::

## Celda y unidad de crecimiento

- Triclínica P-1, $a = 12.196$ Å, $b = 14.684$ Å, $c = 8.040$ Å, $\alpha = 90.22°$,
  $\beta = 96.80°$, $\gamma = 97.92°$ (Pagola et al., 2000). Está en
  `presets.BETA_HEMATIN_CELL`.
- La red recíproca calculada coincide con la de Solomonov et al. (2007), dentro de
  $3 \cdot 10^{-5}$ Å⁻¹ y 0.1° (`tests/test_geometry.py`).
- $Z = 2$ hemos forman un dímero centrosimétrico (Fe–O–propionato) por celda: el
  dímero es la unidad de crecimiento y una unidad por celda es exacta para la
  topología.
- Los dímeros se unen por puentes de hidrógeno O–H···O=C en cadenas según
  $[10\bar{1}]$.

## Morfología observada (objetivo)

- Agujas o láminas alargadas según **c**.
- Caras laterales **{100}** (la más desarrollada) y **{010}**, con anchura/espesor
  ≈ 3:1.
- Extremos **{011}** y, en menor medida, **{001}**.
- La $\{0\bar{1}1\}$ no aparece.

## Energías provisionales

Se eligieron **solo** para que $|E_{att}|$ ordene las caras como se observa
(`presets.BETA_HEMATIN_ENERGIES_PROVISIONAL`):

| Enlace | Longitud (Å) | $\varphi/kT$ | Interpretación |
|---|---|---|---|
| [001] | 8.04 | 2.4 | traslación más corta, apilamiento de porfirinas: eje de la aguja |
| [10-1] | 15.38 | 1.0 | cadenas de puentes de H |
| [01-1] | 16.77 | 1.1 | |
| [010] | 14.68 | 0.9 | |
| [011] | 16.71 | 0.5 | |
| [100] | 12.20 | 0.4 | |
| [101] | 13.79 | 0.3 | |

Con ellas, $E_{cr} = 6.6\,kT$ y la teoría de energía de unión ordena las caras como
(100) < (010) < (110) < (101) < (011) < (001) < (10-1) < (01-1) en $|E_{att}|$.

## Primera exploración (notebook 01, §5)

Semilla esférica de 70 Å, crecimiento hasta $N = 40\,000$ unidades, cuatro valores de
$\Delta\mu$ y los dos modelos de tasas.

- **Con las energías provisionales, el kMC no da una aguja**, sino una lámina con
  {100} grande y poco alargada. La forma de energía de unión, en cambio, sí es una
  aguja según c.
- La cinética no sigue a $|E_{att}|$: la {010} avanza más de 3 veces más rápido que la
  {100} (la energía de unión predice 1.5), y las caras S {110} y la {101} son lentas y
  recortan la aguja.
- La anisotropía cae con $\Delta\mu$ (rugosidad cinética).

```{figure} ../../outputs/01_primeros_resultados/bh_forma_vs_dmu.png
:width: 90%
:alt: Alargamiento y aplanamiento del cristal kMC de β-hematina frente a Δμ

Alargamiento $L_1/L_2$ y aplanamiento $L_2/L_3$ del cristal kMC con las energías
provisionales, para tasas simétricas y de Gilmer, frente a la forma de energía de
unión (línea discontinua).
```

Conclusión: para calibrar las energías con la morfología hay que usar el kMC, no
solo $E_{att}$, y la escala global de las energías (temperatura efectiva) importa
tanto como su reparto.

## Calibración con la morfología (notebook 02, §4)

`kmc3d.calibration` busca energías que cumplan **6 criterios** cualitativos: aguja
según c, $2 \le W/T \le 4.5$, laterales {100}+{010} con {100} mayor, extremos
{011} > {001}, $\{0\bar{1}1\}$ < 3 % y forma estacionaria entre $N/2$ y $N$. Una
pérdida $J$ sirve para desempatar. Las medidas se hacen sobre el fragmento mayor, con
tasas simétricas y semilla esférica de 70 Å.

- Con las energías provisionales ($\Delta\mu = 1.5$) se cumplen **2 de 6**.
- Ningún cambio de un solo enlace cumple los 6.
- De una rejilla de 270 corridas, 7 cumplen los 6 con la semilla 1 y solo una con las
  5 semillas:

| | $\Delta\mu$ | [001] | [10-1] | [100] | [010] | [011] | [01-1] | [101] |
|---|---|---|---|---|---|---|---|---|
| provisional | — | 2.4 | 1.0 | 0.4 | 0.9 | 0.5 | 1.1 | 0.3 |
| calibrada (efectiva) | 1.5 | 3.0 | 1.0 | **1.6** | 0.9 | 0.5 | **2.3** | 0.3 |

Con $N = 40\,000$: $W/T = 2.16 \pm 0.10$, $L_c/W = 1.81$ y eje largo a unos 10° de c.
Con $N = 120\,000$ (3 semillas) se mantienen los 6 criterios. Con difusión débil
($k_d \le 0.1$), 7 de 8 corridas siguen cumpliéndolos, a un coste unas 10 veces mayor.

```{figure} ../../outputs/02_difusion_visualizacion_calibracion/bh_calibrado_render.png
:width: 100%
:alt: Cristal kMC calibrado de β-hematina visto desde dos ángulos y su forma de Wulff

Cristal kMC con las energías calibradas ($N = 120\,000$) desde dos ángulos, coloreado
por familia de caras, y forma de Wulff con las velocidades medidas.
```

**Límites.**

- Los criterios son desigualdades cualitativas: hay una familia de soluciones, no una
  única.
- $W/T$ no llega a 3 en ningún candidato robusto.
- $N = 120\,000$ son unos 55 nm, lejos de las micras reales.
- El Wulff cinético con las $R$ de plano medio no reproduce el cristal kMC (da
  {110} grandes y ninguna {010}); por eso la calibración usa las facetas del cristal.

La predicción comprobable es que las interacciones según [100] y [01-1] son más
fuertes que en el conjunto provisional, comparables o mayores que la cadena de puentes
de H [10-1]. Lo puede comprobar un cálculo con campo de fuerzas (opción (a) de la
decisión D3 del [plan](plan.md)).

## Pendiente

- Origen de las energías: campo de fuerzas o DFT por vector de enlace (D3).
- Calibrar $\Delta\mu$ y $k_0$ con velocidades AFM de la cara (100) (Olafson et al.,
  2015).
- Aditivos (cloroquina y otras quinolinas), que según Solomonov et al. (2007) se
  adsorben en {001} y {011}.
