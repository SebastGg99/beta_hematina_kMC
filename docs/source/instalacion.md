# Instalación

## Requisitos

- Python 3 (verificado con Python 3.14).
- numpy, pandas y matplotlib, con las versiones de `requirements.txt`:

  ```text
  numpy==2.5.3
  pandas==3.0.6
  matplotlib==3.11.2
  ```

  Las firmas congeladas de `tests/test_diffusion.py` (trayectorias sin difusión) solo
  valen con numpy 2.5.3.
- Opcionales: `ipykernel` para abrir los notebooks y `plotly` solo para
  `kmc3d/viz_interactive.py` (vistas 3D con WebGL).

## Pasos

1. Clonar el repositorio:

   ```bash
   git clone git@github.com:SebastGg99/beta_hematina_kMC.git
   cd beta_hematina_kMC
   ```

2. Crear un entorno virtual e instalar las dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Comprobar que todo funciona (74 tests, unos 20 s):

   ```bash
   python -m unittest discover -s tests
   ```

## Cómo se importa `kmc3d`

`kmc3d` **no es un paquete instalable** (no hay `pyproject.toml`). Para importarlo, la
raíz del repositorio tiene que estar en `sys.path`:

```python
import sys
sys.path.insert(0, "/ruta/a/beta_hematina_kMC")   # innecesario si se trabaja desde la raíz

from kmc3d import KMC3D, SplitRates, presets, seeds
```

Al importarse, `kmc3d` (a través de `kmc3d/_paths.py`) añade también la raíz para
encontrar la copia de `src/`, de la que reutiliza funciones (ver
[](codigo_reutilizado.md)). Los notebooks buscan la raíz solos, subiendo desde la
carpeta actual hasta encontrar `kmc3d/`, y los scripts y tests la añaden con
`Path(__file__).parents[1]`.

## Corridas en paralelo

`kmc3d.parallel.run_parallel` lanza cada tarea en un proceso aparte con
`subprocess`. No usa `multiprocessing`, que falla en algunos entornos restringidos
(`PermissionError` en `SemLock`) y da problemas con «fork» en Jupyter.

## Construir esta documentación

La documentación usa Sphinx con páginas en Markdown (MyST) y el tema de Read the Docs.
Desde la raíz:

```bash
pip install -r docs/requirements.txt
cd docs
make html        # resultado en docs/build/html/index.html
make doctest     # ejecuta los ejemplos >>> de la guía de uso
```

En Windows se usa `make.bat html`. El logo se regenera con `python docs/make_logo.py`.

La versión publicada, <https://sebastgg99.github.io/beta_hematina_kMC/>, la compila
`.github/workflows/docs.yml` en GitHub Actions en cada push a `main`. Usa Python 3.14,
los dos archivos de requisitos, `-W` y `make doctest`, y solo publica si todo pasa.
Requisito, una sola vez: en GitHub, *Settings → Pages → Source* = «GitHub Actions».
