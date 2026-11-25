# Datos para la validación con MONTY (Boerrigter et al. 2004)

Referencia: S. X. M. Boerrigter, G. P. H. Josten, J. van de Streek, F. F. A. Hollander,
J. Los, H. M. Cuppen, P. Bennema y H. Meekes, «MONTY: Monte Carlo crystal growth on any
crystal structure in any crystallographic orientation; application to fats»,
*J. Phys. Chem. A* **108**, 5894–5902 (2004). El texto (OCR) y las figuras están en
`references/papers/MONTY2004/`, que no se versiona.

Se usa desde `kmc3d/monty.py`, `scripts/run_monty.py`, `scripts/digitalizar_monty.py` y
`notebooks/03_validacion_monty.ipynb`.

## Archivos

| Archivo | Contenido | Estado |
|---|---|---|
| `referencia_figuras.csv` | Umbrales de las Figs. 2–4 por grupo de caras | hecho |
| `beta_16_16_16.json` | Grafo de β-16.16.16 (P-1, Z = 2) | pendiente: refs. 16 y 17 |
| `betap_10_12_10.json` | Grafo de β′-10.12.10 (centrada I, Z = 8) | pendiente: refs. 16 y 17 |
| `betap_16_16_14.json` | Grafo de β′-16.16.14 (C2, Z = 4) | pendiente: refs. 16 y 17 |

### `referencia_figuras.csv`

Una fila por grupo de curvas (curvas que el artículo dibuja juntas o casi juntas). Δμ
está en kcal/mol.

| Columna | Significado |
|---|---|
| `grasa`, `figura`, `T_K` | grasa, figura del artículo y temperatura de la simulación |
| `grupo`, `caras` | número del grupo y caras (hkl) que lo forman, separadas por espacios |
| `tipo` | superior, lateral o basal, como las clasifica el texto (§3.2) |
| `dmu_texto` | umbral («empiezan a crecer a partir de…») que da el texto, si lo da |
| `dmu_S01`, `dmu_S05`, `dmu_S09` | Δμ en que el grupo cruza S = 0.1, 0.5 y 0.9 en la figura |
| `dmu_err` | margen de lectura: anchura del haz de curvas o error de digitalización |
| `confianza` | alta, media (curvas muy juntas o tapadas por etiquetas) o texto (solo `dmu_texto`) |

Cómo se obtuvo:
1. `scripts/digitalizar_monty.py` calibra cada figura (`images/page_5_{0,1,2}.jpg`) por
   píxeles. El marco de los ejes da S = 0 y S = 1, y los centros de las etiquetas de los
   ticks dan la escala de Δμ (0–20, 0–50 y 0–10 kcal/mol). Después lista dónde cruza
   cada fila de S un píxel oscuro.
2. Esos cruces se asignaron a mano a cada grupo comparando con la figura. Las etiquetas
   de texto de las curvas también cruzan algunas filas, sobre todo en la Fig. 4.
3. El script dibuja la tabla sobre las figuras originales
   (`outputs/03_validacion_monty/referencia_digitalizada.png`) para comprobarla a ojo.

La resolución es de unos 0.02 kcal/mol por píxel en la Fig. 2, 0.06 en la Fig. 3 y 0.012
en la Fig. 4. La incertidumbre real es mayor: los grupos son haces de curvas y el trazo
tiene varios píxeles de grosor. Para las caras basales, (001) y (002), el valor de
referencia es el que da el texto, porque la curva queda fuera del eje (Figs. 2 y 4) o
casi.

## Dónde están los grafos de las grasas (búsqueda del 2026-11-24)

| Fuente | Qué tiene | Acceso |
|---|---|---|
| Hollander et al., *J. Phys. Chem. B* **103**, 8301 (1999), DOI 10.1021/jp991634z (ref. 16 de MONTY) | redes conexas de las β-n.n.n (β-16.16.16) | ACS, de pago |
| Hollander et al., *J. Phys. Chem. B* **107**, 5680 (2003), DOI 10.1021/jp021353i (ref. 17) | las tres grasas: β-16.16.16, β′-10.12.10 y β′-16.16.14 | ACS, de pago |
| Repositorio de Radboud: <https://repository.ubn.ru.nl/bitstream/handle/2066/60273/1/60273.pdf> («Crystal Growth and Morphology», del mismo grupo; habla de redes conexas y de 16.16.16) | probablemente parte de los datos | abierto para navegadores; responde 403 a clientes automáticos |
| Tesis de F. F. A. Hollander, «Growth and Morphology of Triacylglycerol Crystals» (Nijmegen, 2001) | los grafos completos, probablemente | solo resumen en <https://vsc.science.ru.nl/theses/frank.html> |
| MONTY (<https://vsc.science.ru.nl/deij/monty.html>) | el programa y sus `.cfg` | licencia académica, pidiéndola a H. Meekes (Radboud) |
| Estructuras: β-16.16.16 en van Langevelde et al., *Acta Cryst. B* **55**, 114 (1999); β′-10.12.10 en *Acta Cryst. B* (2000), «Structure of CnCn+2Cn-type (n = even) β′-triacylglycerols» | celdas y coordenadas (CIF) | IUCr / CSD |

Con solo las estructuras habría que calcular las energías de par con un campo de fuerzas
(MONTY usó CVFF en Cerius²), que está fuera del alcance actual del proyecto (D3).

## Formato de los grafos de las grasas (pendiente de transcribir)

Las energías de enlace de las grasas no están en el artículo de MONTY. Están en sus
refs. 16 y 17: F. F. A. Hollander et al., *J. Phys. Chem. B* **103**, 8301 (1999) y
**107**, 5680 (2003). Las calcularon con el campo de fuerzas CVFF (Cerius²), en kcal/mol.
Las estructuras son las de las refs. 35–37 del artículo.

`kmc3d.monty.load_graph_file` lee el formato de transcripción, pensado para copiar las
tablas publicadas: las moléculas independientes, las operaciones de simetría y, para cada
enlace, la posición del centro de la molécula vecina. `BasisGraph.from_centers` completa
la base y los enlaces con la simetría y descarta los repetidos.

```json
{
  "name": "beta_16_16_16",
  "fuente": "Hollander et al. 2003, tabla N, p. NNNN",
  "units": "kcal/mol",
  "cell": {"a": 0.0, "b": 0.0, "c": 0.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
  "ajuste": "celda convencional de la ref. 35",
  "centers": [{"label": "M1", "frac": [0.0, 0.0, 0.0]}],
  "symops": [{"W": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0]},
             {"W": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], "t": [0, 0, 0]}],
  "bonds_by_position": [
    {"from": 0, "to_frac": [0.0, 0.0, 0.0], "energy": 0.0, "label": "a"}
  ],
  "faces": ["(001)", "(1-11)"],
  "protocolo": {"T": 313.0, "size": [40, 40], "n_equil": 100000, "n_sample": 100000,
                "lambda1": 0.5, "lambda2": 0.5},
  "notes": "decisiones de transcripción"
}
```

- `energy` es φ > 0 (la energía de enlace con el signo cambiado). Cada enlace se escribe
  una sola vez.
- `symops` incluye la identidad. En grupos centrados se añaden las traslaciones de
  centrado como operaciones (W = identidad, t = (½, ½, ½) en I, o (½, ½, 0) en C).
- `faces` son las caras de la figura correspondiente, con los índices en la misma celda
  que `cell`. Hay que confirmar el ajuste con la ref. 17. Por ejemplo, (201) y (610) no
  son planos reticulares de una red I, así que MONTY usa la celda convencional como celda
  de capa.
- `protocolo` da los valores por defecto de `scripts/run_monty.py`
  (`kmc3d.monty.MONTY_PROTOCOLS`).
- Al transcribir, también se pueden dar `"basis"` y `"bonds"` ya completos (formato de
  `BasisGraph.to_dict`).

Comprobaciones antes de usarlos (notebook 03, §3.2):
- la energía de red por molécula (`lattice_energy`) y las longitudes de enlace
  (`bond_table`) coinciden con las publicadas;
- E_slice y E_att por cara (`face_table`) coinciden con las tablas de las refs. 16 y 17,
  si las traen;
- la morfología de energía de unión coincide con la fila superior de la Fig. 1, al
  menos en el orden de las caras.
