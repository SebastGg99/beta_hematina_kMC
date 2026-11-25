# Pruebas

## Cómo correrlas

Desde la raíz del repositorio (94 tests, unos 25 s):

```bash
python -m unittest discover -s tests
```

Los ejemplos de la [guía de uso](uso.md) se comprueban aparte, con `make doctest`
desde `docs/`.

## Qué cubren

| Archivo | Tests | Qué comprueba |
|---|---|---|
| `test_geometry.py` | 7 | Espaciados, recíproca de β-hematina frente a Solomonov et al. (2007), índices de Miller |
| `test_graph.py` | 9 | Códigos de vecindad y Hartman-Perdok (Kossel: {100} F, {110} S, {111} K) |
| `test_rates.py` | 7 | Balance detallado de `SplitRates` y correspondencia de `AdaptiveRates` con el motor estático 2D (`KMC_BKL_Static`) |
| `test_engine.py` | 11 | Estado incremental frente a reconstrucción, ampliación de la caja, disolución completa, determinismo, pickle y E/S |
| `test_physics.py` | 3 | Signo de $\Delta\mu$, caras lisas frente a rugosas, aguja con enlaces anisótropos |
| `test_diffusion.py` | 10 | Difusión: balance detallado por enumeración, estado incremental, conservación de $N$ y trayectorias congeladas sin difusión |
| `test_analysis.py` | 10 | Base de cara, mapas de alturas, semillas poliédricas, probabilidades de unión |
| `test_morphology.py` | 13 | Caras de poliedros, áreas de Wulff, facetas del cristal kMC, espesores, render y criterios de calibración |
| `test_parallel.py` | 4 | Ejecutor paralelo con `subprocess` y experimentos importables |
| `test_basis_graph.py` | 8 | `BasisGraph`: E_slice, E_att y d_hkl iguales a los de `CrystalGraph` con base 1, supercelda, simetría (`from_centers`), kcal/mol ↔ kT, JSON y celda de capa unimodular y reducida |
| `test_slab.py` | 12 | `SlabKMC` y `MontyRates`: tasas iguales a `SplitRates(1 − λ2, λ1)`, balance detallado, estado incremental frente a reconstrucción (contorno periódico, caja que crece por arriba y por abajo, las dos reglas de huecos cerrados), energías de sitio contadas a mano con base > 1, determinismo, pickle, caras lisas frente a rugosas, equilibrio, simetría crecimiento/disolución y comparación con la referencia de MONTY |

## Reproducibilidad

`test_diffusion.py::TestNoDiffusionUnchanged` guarda una **firma** (contadores, tiempo
y hash de los snapshots) de dos trayectorias sin difusión, una de Kossel y otra de
β-hematina, congelada el 2026-11-11 con numpy 2.5.3. Comprueba que añadir la difusión no cambió nada cuando no se usa.

Cualquier cambio en el motor, las tasas, el grafo o las semillas debe pasarla. Por
eso no se debe cambiar el orden en que se consumen los números aleatorios (por
ejemplo, introduciendo `sorted()` o `set()` nuevos, o recorriendo diccionarios en otro
orden). Con otra versión de numpy la firma puede cambiar aunque el código sea el
mismo.
