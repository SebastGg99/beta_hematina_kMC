"""Acceso al paquete `src` del repositorio para reutilizar su código.

En beta_hematina_kMC, `kmc3d/` y `src/` están los dos en la raíz del repositorio.
`src/` es una copia sin modificar del `src/` del repositorio kMC_Malaria (motor 2D);
ver la sección «Código reutilizado» del README. Para reutilizar `src.common`,
`src.analysis` y `src.static` sin instalar nada, se añade la raíz del repositorio a
`sys.path` la primera vez que se importa kmc3d.
"""

import sys
from pathlib import Path

# kmc3d/_paths.py -> parents[0] = kmc3d, [1] = raíz del repositorio (contiene src/)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_3D_DIR = PACKAGE_DIR.parent
# Dentro de kMC_Malaria/3DkMC/ la raíz era PROJECT_3D_DIR.parent; aquí las dos coinciden.
REPO_ROOT = PROJECT_3D_DIR


def ensure_repo_on_path() -> Path:
    """Añade la raíz del repositorio a sys.path (si hace falta) y la devuelve."""
    if not (REPO_ROOT / "src" / "__init__.py").exists():
        raise ImportError(f"No se encuentra src/ en {REPO_ROOT}: kmc3d necesita la copia "
                          "de src/ (de kMC_Malaria) en la raíz del repositorio")
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


ensure_repo_on_path()
