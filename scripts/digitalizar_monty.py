"""Digitaliza las Figs. 2-4 de MONTY (Boerrigter et al. 2004) y comprueba la referencia.

Uso (desde la raíz del repositorio):

    python scripts/digitalizar_monty.py            # cruces por nivel de S + figura de control
    python scripts/digitalizar_monty.py --sin-figura

1. Calibra cada imagen por píxeles: el marco de los ejes da S = 0 (abajo) y S = 1
   (arriba), y las etiquetas de los ticks del eje x dan Δμ (0, 10, 20, ... kcal/mol).
2. En las filas de S = 0.03, 0.1, 0.3, 0.5, 0.7 y 0.9 busca los píxeles oscuros (las
   curvas, y también las etiquetas de texto que cruzan esa fila) y escribe su Δμ.
3. Dibuja sobre cada figura los valores de `data/monty2004/referencia_figuras.csv`
   (asignados a mano a cada grupo a partir de esos cruces) para comprobarlos a ojo.

La figura de control va a outputs/03_validacion_monty/referencia_digitalizada.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "references" / "papers" / "MONTY2004" / "images"
REF = ROOT / "data" / "monty2004" / "referencia_figuras.csv"
OUT = ROOT / "outputs" / "03_validacion_monty"

# Calibración de cada figura: (imagen, fila de S = 1, fila de S = 0, columna de Δμ = 0,
# píxeles por kcal/mol). Las filas y columnas del marco son las que tienen una línea
# oscura en más del 60 % de la imagen; la escala sale de las etiquetas de los ticks
# (centros de "0" y "20" en la Fig. 2, de "0" y "50" en la Fig. 3, de "0" y "10" en la 4).
CALIBRACION = {
    2: ("page_5_0.jpg", 23, 763, 112.0, (993.5 - 112.5) / 20),
    3: ("page_5_1.jpg", 19, 719, 99.0, (869.0 - 99.0) / 50),
    4: ("page_5_2.jpg", 27, 736, 116.0, (923.0 - 116.0) / 10),
}
NIVELES = (0.03, 0.1, 0.3, 0.5, 0.7, 0.9)


def leer_imagen(nombre: str) -> np.ndarray:
    """Imagen en escala de grises (0 = negro) como array float."""
    from PIL import Image          # viene con matplotlib
    return np.asarray(Image.open(IMAGES / nombre).convert("L"), dtype=float)


def cruces(im: np.ndarray, top: int, bot: int, x0: float, sx: float, S: float,
           umbral: float = 120.0) -> list:
    """Δμ (kcal/mol) de los tramos oscuros de la fila de S (curvas y texto)."""
    y = int(round(bot - S * (bot - top)))
    fila = im[y - 1:y + 2, int(x0) + 3:-3].min(axis=0) < umbral
    cols = np.flatnonzero(fila) + int(x0) + 3
    if not len(cols):
        return []
    grupos = np.split(cols, np.flatnonzero(np.diff(cols) > 2) + 1)
    return [round(float(((g[0] + g[-1]) / 2 - x0) / sx), 2) for g in grupos]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sin-figura", action="store_true", help="no dibuja la figura de control")
    args = ap.parse_args(argv)

    for fig, (nombre, top, bot, x0, sx) in CALIBRACION.items():
        im = leer_imagen(nombre)
        print(f"Fig. {fig} ({nombre}): {sx:.2f} px por kcal/mol")
        for S in NIVELES:
            print(f"  S = {S:4.2f}: {cruces(im, top, bot, x0, sx, S)}")

    if args.sin_figura:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = pd.read_csv(REF, comment="#")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, (nfig, (nombre, top, bot, x0, sx)) in zip(axes, CALIBRACION.items()):
        ax.imshow(leer_imagen(nombre), cmap="gray")
        for _, r in ref[ref.figura == nfig].iterrows():
            for S, col in ((0.1, "dmu_S01"), (0.5, "dmu_S05"), (0.9, "dmu_S09")):
                if np.isfinite(r[col]):
                    xp, yp = x0 + r[col] * sx, bot - S * (bot - top)
                    ax.plot(xp, yp, "o", ms=8, mfc="none", mec="#e34948", mew=1.6)
                    if S == 0.5:
                        ax.annotate(str(r["grupo"]), (xp, yp), color="#e34948", fontsize=11,
                                    xytext=(5, -12), textcoords="offset points")
        ax.set_title(f"Fig. {nfig} de MONTY: grupos de referencia (S = 0.1, 0.5, 0.9)", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "referencia_digitalizada.png", dpi=110, bbox_inches="tight")
    print(f"Figura de control: {OUT / 'referencia_digitalizada.png'}")


if __name__ == "__main__":
    sys.exit(main())
