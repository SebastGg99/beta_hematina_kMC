"""Genera el logo y el favicon de la documentación (docs/source/_static/).

El logo es el poliedro de Hartman-Perdok de β-hematina (velocidad de cada cara
proporcional a |E_att|) con las energías efectivas calibradas con la morfología
(PLAN_3DkMC.md §7.5), dibujado con kmc3d.render. Es solo ilustrativo: esas energías
son provisionales. No hace falta simular nada, así que tarda unos segundos.

Uso (desde la raíz del repositorio):
    python docs/make_logo.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # sin ventana: solo se guardan archivos
import matplotlib.pyplot as plt           # noqa: E402

# Raíz del repositorio en sys.path para importar kmc3d (y con él src/)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kmc3d import analysis as an, presets, render  # noqa: E402

STATIC = ROOT / "docs" / "source" / "_static"

# Energías calibradas (efectivas) de PLAN_3DkMC.md §7.5: solo cambian [001], [100] y
# [01-1] respecto de las provisionales de kmc3d/presets.py
CALIBRADAS = {"[001]": 3.0, "[100]": 1.6, "[01-1]": 2.3}


def main() -> None:
    g = presets.beta_hematin(energies=CALIBRADAS)
    caras_obs = presets.BETA_HEMATIN_FACES
    # Velocidad de cada cara ∝ |E_att| (slice_attachment devuelve E_att >= 0)
    rates = {f: g.slice_attachment(f)[1] for f in caras_obs}
    caras, _ = an.wulff_faces(rates, g.cell)

    STATIC.mkdir(parents=True, exist_ok=True)
    for nombre, tam, dpi in [("logo.png", (2.4, 3.0), 150), ("favicon.png", (1.0, 1.0), 64)]:
        fig = plt.figure(figsize=tam)
        ax = fig.add_subplot(111, projection="3d")
        # Sin etiquetas, leyenda ni flechas a, b, c: solo la forma coloreada por familia
        render.plot_polyhedron(caras, caras_obs, ax=ax, cell=g.cell, label_faces=False,
                               triad=False, legend=False, elev=22.0, azim=-55.0, zoom=1.0)
        fig.subplots_adjust(0, 0, 1, 1)
        # recorte ajustado al dibujo y fondo transparente (se ve bien en el tema claro y oscuro)
        fig.savefig(STATIC / nombre, dpi=dpi, transparent=True, bbox_inches="tight",
                    pad_inches=0.02)
        plt.close(fig)
        print("guardado", STATIC / nombre)


if __name__ == "__main__":
    main()
