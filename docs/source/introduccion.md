# Introducción

## Qué es este proyecto

`beta_hematina_kMC` es un motor **kinetic Monte Carlo (kMC) 3D** para estudiar cómo
crece un cristal **finito** con una estructura cristalina real. El caso de interés es
la **β-hematina**, el análogo sintético de la hemozoína: el cristal en el que el
parásito de la malaria encierra el hemo libre.

El paquete se llama `kmc3d`. Simula un cristal que crece o se disuelve **por todas sus
caras a la vez**. Cada cara avanza a su propia velocidad porque la energía de las
unidades de su superficie depende de qué enlaces cortan esa cara. Así la forma del
cristal (aguja, lámina, prisma) sale de la estructura y no hay que imponerla.

## Objetivo

El proyecto nació como extensión del modelo **(2+1)D** de
[kMC_Malaria](https://github.com/SebastGg99/kMC_Malaria), que describe una sola
cara con una altura por columna (aproximación *Solid-On-Solid*, SOS). Los objetivos
del 3D son:

- un cristal finito que crezca o se disuelva por todas sus caras;
- anisotropía y energías distintas por cara y orientación, que salgan de la
  estructura cristalina y no se impongan a mano;
- empezar con «cubitos» (una unidad de crecimiento por celda) y dejar el camino
  abierto a unidades más realistas.

## Qué cambia respecto al modelo (2+1)D

| Aspecto | Modelo (2+1)D (`src/static`, `src/dynamic`) | `kmc3d` (3D) |
|---|---|---|
| Configuración | altura `h[i, j]` por columna | ocupación 0/1 de cada sitio de una caja 3D |
| Geometría | una cara (001), contorno periódico | cristal finito, todas las caras, caja que se amplía sola |
| Voladizos, huecos | imposibles (SOS) | permitidos |
| Vecinos | 4 laterales + el de abajo | los que defina el grafo (6 en Kossel, 14 en β-hematina) |
| Anisotropía | `E_pb_x`, `E_pb_y`, `δx`, `δy` | una energía por tipo de enlace; las caras salen solas |
| Celda | cuadrada | cualquier celda de Bravais (triclínica incluida) |
| Clasificación de eventos | se reconstruye entera en cada paso | **incremental**: solo el sitio y sus vecinos |
| Elección de clase | lineal | árbol de Fenwick, $O(\log K)$ |
| Conteo de enlaces | vecino con altura **exactamente** igual | geométrico: vecino ocupado |

La última fila importa al comparar resultados: **las dos líneas no cuentan igual los
enlaces**. Coinciden en los sitios donde ningún vecino es más alto (despegue) o donde
los vecinos son más de un nivel más altos (unión), y lo comprueba un test
(`tests/test_rates.py`). En el resto, el 3D cuenta los enlaces que hay geométricamente
y el 2D no. Es la decisión abierta D6 del [plan](plan.md).

## MONTY, CrystalGrower y este enfoque

| | MONTY (Boerrigter et al. 2004) | CrystalGrower (Anderson et al. 2017) | `kmc3d` |
|---|---|---|---|
| Qué se simula | una superficie (hkl) periódica; las velocidades se combinan en un Wulff cinético | el cristal entero | el cristal entero |
| Estructura | grafo cristalino desde la estructura real | estructura real, unidades con varias «particiones» | grafo de enlaces sobre una red de Bravais con una unidad por celda |
| Energías | campo de fuerzas | $\Delta U$ por tipo de sitio | $\varphi$ por tipo de enlace, en $kT$ |
| Tasas | reparto simétrico de $\Delta\mu$ y $\Delta U$ | probabilidades por energía de sitio | familia `SplitRates` con balance detallado + `AdaptiveRates` |
| Difusión superficial | sí | sí | sí, opcional (`SurfaceDiffusion`) |

De MONTY se toma el **grafo cristalino** como descripción de las interacciones y el
análisis de **caras F** (cadenas periódicas de enlaces) para saber qué caras son
lisas. No se copia la restricción a una superficie ni el campo de fuerzas.

## Estado actual

| Fase | Contenido | Estado |
|---|---|---|
| F1 | Núcleo: grafo, tasas, motor BKL 3D incremental, tests | hecho |
| F2 | Análisis por cara, forma de energía de unión, mapas de alturas, figuras, E/S | hecho (falta el modo *slab*) |
| F3 | β-hematina: celda real, energías, calibración | celda y calibración con la morfología hechas; falta la cinética (AFM) |
| F4 | Difusión superficial, reservorio finito, aditivos | difusión hecha; el resto pendiente |
| F5 | Rendimiento (réplicas en paralelo, compilación) | paralelo hecho; compilación pendiente |
| F6 | Unidades moleculares, base > 1, grafo desde un CIF | pendiente |

El detalle de cada fase está en el [plan de implementación](plan.md) y los
fundamentos teóricos en el documento
{download}`teoria_3DkMC.pdf <../../teoria/teoria_3DkMC.pdf>`.

## Origen del repositorio

El proyecto se desarrolló en la carpeta `3DkMC/` de
[kMC_Malaria](https://github.com/SebastGg99/kMC_Malaria) y se trasladó a este
repositorio el 2026-11-16, con una copia del motor 2D en `src/` (ver
[](codigo_reutilizado.md)).
