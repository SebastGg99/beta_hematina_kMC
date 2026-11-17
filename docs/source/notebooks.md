# Notebooks

Los resultados del proyecto están en dos notebooks de `notebooks/`. Cada uno guarda
sus figuras (y, el 02, sus cachés) en `outputs/<nombre_del_notebook>/`, y son las
figuras que usan esta documentación, el plan y el documento teórico.

Para abrirlos basta con lanzar Jupyter desde cualquier carpeta del repositorio: la
primera celda sube desde la carpeta actual hasta encontrar `kmc3d/` y la añade a
`sys.path`. Hace falta `ipykernel` (ver [](instalacion.md)).

:::{note}
Las animaciones usan matplotlib (`viz.animate_growth_3d` + `HTML(anim.to_jshtml())`)
porque las figuras 3D de plotly necesitan WebGL, que no todos los visores de
notebooks tienen.
:::

## 01: primeros resultados

`notebooks/01_primeros_resultados.ipynb` valida el motor y hace la primera
exploración de β-hematina.

| Sección | Contenido |
|---|---|
| 2 | Grafos de enlaces y teoría de energía de unión |
| 3 | Validación con Kossel: tamaño crítico (Gibbs-Thomson), caras lisas frente a rugosas |
| 4 | Anisotropía con cubitos |
| 5 | β-hematina: barrido en $\Delta\mu$ con los dos modelos de tasas, sensibilidad a las energías, visualización dinámica y por caras |
| 6 | Superficie de una cara con los observables 2D de `src.analysis` |
| 7 | Probabilidad de unión por tipo de sitio |
| 8 | Analizar corridas guardadas (`kmc3d.io.load_run`) |
| 9 | Conclusiones |

Salidas en `outputs/01_primeros_resultados/`: curvas $R(\Delta\mu)$ y tamaño crítico
de Kossel, cubitos anisótropos, formas de β-hematina frente a $\Delta\mu$, mapas de
alturas por cara, probabilidades de unión, una tabla del barrido
(`bh_barrido.csv`) y una vista 3D interactiva (`bh_crecimiento_3d.html`, con plotly).

## 02: difusión, visualización y calibración

`notebooks/02_difusion_visualizacion_calibracion.ipynb` añade la difusión
superficial, las visualizaciones de `kmc3d.render` y la calibración de β-hematina.

| Sección | Contenido |
|---|---|
| 0–1 | Revisión de pendientes y configuración |
| 2 | Difusión superficial: tasas, caras {100} de Kossel frente a $k_d$, relajación de una varilla |
| 3 | Visualizaciones nuevas (superficie por faceta, Wulff, espesores) |
| 4 | Calibración de β-hematina: método, sensibilidad, rejilla, robustez frente a la semilla, tamaño, cristal calibrado y efecto de la difusión |
| 5 | Conclusiones |

Las corridas largas se guardan en `outputs/02_difusion_visualizacion_calibracion/cache/`
(archivos `.pkl`), así que volver a ejecutar el notebook no las repite.

```{figure} ../../outputs/02_difusion_visualizacion_calibracion/kossel_R_W_vs_kd.png
:width: 95%
:alt: Velocidad y rugosidad de las caras {100} de Kossel frente a la tasa de difusión

Kossel: velocidad y rugosidad de las caras {100} frente a $k_d/k_0$ (notebook 02,
§2.2). La difusión acelera sobre todo las caras lisas ($\varphi = 3$).
```
