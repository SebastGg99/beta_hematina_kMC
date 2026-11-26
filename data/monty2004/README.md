# Datos para la validación con MONTY (Boerrigter et al. 2004)

Referencia: S. X. M. Boerrigter, G. P. H. Josten, J. van de Streek, F. F. A. Hollander,
J. Los, H. M. Cuppen, P. Bennema y H. Meekes, «MONTY: Monte Carlo crystal growth on any
crystal structure in any crystallographic orientation; application to fats»,
*J. Phys. Chem. A* **108**, 5894–5902 (2004). El texto (OCR) y las figuras están en
`references/papers/MONTY2004/`, que no se versiona.

Se usa desde `kmc3d/monty.py`, `scripts/run_monty.py`, `scripts/digitalizar_monty.py`,
`scripts/transcribir_grasas.py` y `notebooks/03_validacion_monty.ipynb`.

## Archivos

| Archivo | Contenido | Estado |
|---|---|---|
| `referencia_figuras.csv` | Umbrales de las Figs. 2–4 por grupo de caras | hecho |
| `beta_16_16_16.json` | Grafo de β-16.16.16 (P-1, Z = 2) | hecho (2026-11-25), Hollander 2003 |
| `betap_10_12_10.json` | Grafo de β′-10.12.10 (I c 2 a, Z = 8) | hecho (2026-11-25), Hollander 2003 |
| `betap_16_16_14.json` | Grafo de β′-16.16.14 (C2, Z = 8; 4 por celda primitiva) | hecho (2026-11-25), Hollander 2003; posiciones reconstruidas |
| `comprobacion_grafos.csv` | E_att de la red conexa más fuerte y del mejor corte plano, frente a la Tabla 3 de Hollander | hecho (2026-11-25) |

Los cuatro últimos los escribe `scripts/transcribir_grasas.py`, que contiene las Tablas 2
y 3 de Hollander tal cual. `tests/test_grasas.py` comprueba que los JSON están al día con
el script y que reproducen la Tabla 3. Detalles en «Transcripción (2026-11-25)», al final.

Los barridos S(Δμ) con estos grafos están en `outputs/03_validacion_monty/barridos/`: la
semilla 1 en toda la rejilla de Δμ y las semillas 2 y 3 en la zona de subida de cada curva
(`run_monty.py --solo-subida 0.02 0.98`). Su comparación con MONTY está en el notebook 03,
§3.3, y en `PLAN_3DkMC.md`, §11.4, con las dos discrepancias abiertas.

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

Resuelto el 2026-11-25: la ref. 17 (Hollander et al. 2003) está en
`references/papers/Hollander2003/` y trae las tres grasas, así que no ha hecho falta la
ref. 16. La tabla queda como registro de la búsqueda.

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

## Formato de los grafos de las grasas

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
  que `cell`. Por ejemplo, (201) y (610) no son planos reticulares de una red I, así que
  MONTY usa la celda convencional como celda de capa.
- `protocolo` da los valores por defecto de `scripts/run_monty.py`
  (`kmc3d.monty.MONTY_PROTOCOLS`).
- Al transcribir, también se pueden dar `"basis"` y `"bonds"` ya completos (formato de
  `BasisGraph.to_dict`).

Comprobaciones antes de usarlos (notebook 03, §3.2):
- la energía de red por molécula (`lattice_energy`) y las longitudes de enlace
  (`bond_table`) coinciden con las publicadas;
- E_att por cara coincide con la Tabla 3 de la ref. 17. Ojo: `face_table` usa cortes
  planos y la Tabla 3, redes conexas (ver abajo y `comprobacion_grafos.csv`);
- la morfología de energía de unión coincide con la fila superior de la Fig. 1, al
  menos en el orden de las caras.

## Transcripción (2026-11-25)

Fuente: F. F. A. Hollander, S. X. M. Boerrigter, J. van de Streek, P. Bennema y
H. Meekes, *J. Phys. Chem. B* **107**, 5680 (2003) (ref. 17 de MONTY), Tablas 1–3.

### Qué se copió

- **Tabla 2**, tal cual, en `scripts/transcribir_grasas.py` (`TABLA2`) y en el campo
  `tabla2_hollander` de cada JSON. «Mi.Mj [uvw]» es el enlace de la molécula i de la
  primera celda a la molécula j de la celda [uvw]. Hay 7 enlaces en β-16.16.16 (p, q1,
  q2, r1, r2, s1, s2), 4 en β′-10.12.10 (p, q, r, s) y 12 en β′-16.16.14 (pA, pB, q1,
  q2, r1, r2, u, v, x, y, sA, sB). La simetría completa el resto. Resultado: 8, 32 y 48
  enlaces por celda, y E_cr = 88.25, 63.30 y 91.61 kcal/mol por GU.
- **Tabla 1**: las celdas.
- **Tabla 3** (caras con al menos una red conexa, su E_att y el número de redes): solo
  para comprobar (`TABLA3` y el campo `tabla3_hollander`).
- `faces`: las caras de las Figs. 2–4 de MONTY, tal como están en
  `referencia_figuras.csv`, con los índices de MONTY.

### Posiciones de los centros

El artículo no da coordenadas. En `SlabKMC` las posiciones solo deciden las alturas de
las GU y, con ellas, el **corte plano inicial** de la superficie. La dinámica depende
solo del grafo: quién se enlaza con quién y con qué energía.

| Grasa | Origen de las posiciones |
|---|---|
| β-16.16.16 | COD 2003033 (tripalmitina, P-1; van Langevelde et al., *Acta Cryst. B* 55, 114, 1999). Centros de masa de los átomos pesados, en la celda (a+b, −a, −2a−2b+c) de COD |
| β′-10.12.10 | COD 2102198 (I b a 2; *Acta Cryst. B* 56, 1103, 2000), con a_H = b, b_H = c, c_H = a de COD (grupo I c 2 a) |
| β′-16.16.14 | **Reconstrucción**: no hay estructura publicada. M3 y M8 son parámetros libres y el resto sale de la simetría C2 (`posiciones_16_16_14`). Se eligieron para que los contactos sean razonables (q, r entre 4 y 9 Å; s de 38 a 52 Å) y el corte plano reproduzca la red conexa en el mayor número de caras. La topología del grafo es la misma para todas las posiciones probadas |

Los CIF de COD no se guardan en el repositorio: basta con los números de COD.

### Convención de la Tabla 3

Al comprobarla se vio que E_att es **Z_prim·(E_cr − E_slice)** de la **red conexa más
fuerte**. E_slice es por GU y Z_prim es el número de GU por celda primitiva (2, 4 y 4),
de modo que E_att es la suma de los enlaces rotos por celda primitiva de la capa. Las
redes conexas son topológicas (como en FACELIFT), no cortes planos. El script las busca
así: cada subred de la celda primitiva recibe un desplazamiento entero de capa, y cada
capa debe ser conexa y periódica en dos direcciones (rango 2 e índice 1). Con
(hkl) = n·(h₀k₀l₀) hay n capas por periodo. `comprobacion_grafos.csv` da esa E_att y la
del mejor corte plano (`best_termination`, el que usa `SlabKMC`) para las caras de la
Tabla 3 y de las figuras de MONTY, con los enlaces que rompe cada red.

Resultado:
- β′-10.12.10: las 5 caras entre −1.8 % y +2.9 %;
- β′-16.16.14: 12 de 13 caras dentro del 0.4 %; la (112) no (punto 3 de abajo);
- β-16.16.16: las 10 caras entre +0.2 % y +2.9 % (punto 2).

### Morfología de energía de unión (Fig. 1 de MONTY)

La fila superior de la Fig. 1 se calculó con el módulo de energía de unión de Cerius²
(cortes planos y el campo de fuerzas completo), no con las redes conexas de la ref. 17.
`transcribir_grasas.py` la reproduce, solo con los enlaces de la Tabla 2, con
`energias_corte_plano`, `morfologia_energia_union` y `forma`. El notebook 03, §3.2, la
dibuja y la compara de dos maneras: con el mejor corte plano y con las redes conexas.

| Grasa | Fig. 1 (Cerius²) | Corte plano, Tabla 2 |
|---|---|---|
| β-16.16.16 | {001}, {2-1-3}, {1-1-1}, {20-3} + {30-5} | {001}, {10-1}, {1-1-1}, {2-1-3} |
| β′-10.12.10 | {002}, {101}, {112} | {002}, {101}, {011} + {110} (la misma E_att que {112}) |
| β′-16.16.14 | {001}, {110}, {60-2} | {001}, {100}, {111} |

### Discrepancias del artículo y decisiones

1. **Marco de β-16.16.16.** La Tabla 1 da α = 74.5°, β = 121.7° y γ = 124.3°, pero las
   Tablas 2 y 3 están en la celda con **a invertido** (β = 58.3°, γ = 55.7°). Solo en
   esa celda las caras de MONTY con red conexa (grupos 5 a 7 de la Fig. 2) son las de la
   Tabla 3. MONTY usa los índices de la Tabla 1. El JSON usa la celda (a, −b, −c) de la
   de las Tablas 2 y 3, que tiene la métrica de la Tabla 1, es dextrógira y se diferencia
   de la Tabla 1 en una inversión, que es simetría de P-1. Así, los índices del JSON son
   los de MONTY. Cambio: [uvw] → [u, −v, −w] y (hkl) → (h, −k, −l).
2. **E_att de la Tabla 3 de β-16.16.16.** Las 10 caras quedan entre 0.1 y 3.4 kcal/mol
   por debajo de lo que da la Tabla 2. Con p = 31.35 y r2 = 28.3 kcal/mol, en vez de
   32.6 y 29.1, coinciden todas dentro de ±0.1. La Tabla 3 se calculó con otras
   energías, quizá las de la ref. 16. El JSON usa la Tabla 2.
3. **(112) de β′-16.16.14.** La Tabla 3 le da 254.75, el mismo valor que a (110). El
   grafo le da 196.03, la misma red que (111) y (113) (196.17 en la Tabla 3). La Fig. 4
   de MONTY pone (112) en el grupo de (111) y (113), no en el de (110), así que lo más
   probable es una errata de la Tabla 3.
4. **Etiquetado de β′-10.12.10.** Dos asignaciones de M1, M2, M6 y M7 reproducen la
   Tabla 3 por igual (el enlace s a 25.0 o a 29.2 Å). Se usa la de 25.0 Å, la única que
   da a (1-12) y (2-11) la misma red conexa, como dice el texto (§4.3: «the bond
   topologies of the most stable connected nets of these two faces are exactly
   identical»).
5. **Redes conexas que la Tabla 3 no lista.** En β′-10.12.10, el grafo (con cualquiera
   de los dos etiquetados) tiene redes conexas en (011), (110) y (310), con E_att de 179
   a 195, y la Tabla 3 solo da (1-12) y (2-11) como caras superiores. En β-16.16.16,
   MONTY llama «no conexas» a sus grupos 1 a 3 de la Fig. 2, que son las formas T1 de la
   Tabla 3 (conexas solo a través de los enlaces s, débiles).
6. **Corte inicial de β′-16.16.14.** Con las posiciones reconstruidas, el mejor corte
   plano no es la red conexa en (110), (313) y (317). En esas caras, la superficie
   inicial de `SlabKMC` no es la capa F, y es el equilibrado el que la relaja.
7. **Índices que no son planos de la red centrada.** En β′-16.16.14, (100), (201),
   (203), (205) y (30-1) tienen h + k impar; en β′-10.12.10, (201) y (610) tienen
   h + k + l impar. `SlabKMC` simula la orientación con la celda convencional: la capa
   propia es la de 2h. Las caras con índices no coprimos, como (020), (0-22), (202),
   (204) y (002), se simulan con los reducidos, así que (0-22) y (0-11) son la misma
   simulación.
