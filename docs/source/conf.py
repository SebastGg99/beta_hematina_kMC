# Configuración de Sphinx para la documentación de beta_hematina_kMC.
#
# Generado con sphinx-quickstart (fuentes en docs/source, salida en docs/build) y
# completado siguiendo el tutorial de Sphinx:
# https://www.sphinx-doc.org/en/master/tutorial/index.html
# Lista completa de opciones: https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Compilar (desde docs/):  make html      -> build/html/index.html
# Ejemplos ejecutables:    make doctest
#
# Necesita los paquetes de ../requirements.txt (para importar kmc3d) y los de
# docs/requirements.txt (Sphinx, MyST y el tema).

import re
import sys
from pathlib import Path

# -- Rutas -------------------------------------------------------------------
# Como en el tutorial («Automatic documentation generation from code»): la raíz del
# repositorio va en sys.path para que autodoc pueda importar kmc3d. Al importarse,
# kmc3d añade también la raíz para encontrar la copia de src/.
ROOT = Path(__file__).resolve().parents[2]      # docs/source/conf.py -> raíz del repo
sys.path.insert(0, str(ROOT))

# -- Información del proyecto ------------------------------------------------
project = "beta_hematina_kMC"
copyright = "2026, Sebastian Gaviria Giraldo - Hernan Salinas - German Torres"
author = "Sebastian Gaviria Giraldo - Hernan Salinas - German Torres"
release = "0.1.0"

# -- Configuración general ---------------------------------------------------
extensions = [
    "sphinx.ext.duration",      # tutorial: tiempo de compilación de cada página
    "sphinx.ext.doctest",       # tutorial: ejemplos >>> que se comprueban con `make doctest`
    "sphinx.ext.autodoc",       # tutorial: documentación a partir de los docstrings
    "sphinx.ext.autosummary",   # tutorial: tablas resumen y una página por módulo
    "sphinx.ext.intersphinx",   # enlaces a la documentación de Python, numpy, etc.
    "sphinx.ext.mathjax",       # ecuaciones en HTML
    "sphinx.ext.viewcode",      # enlace al código fuente junto a cada objeto
    "myst_parser",              # páginas en Markdown (igual que en kMC_Malaria)
]

# Páginas en reStructuredText y en Markdown (MyST)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

language = "es"

# En los docstrings, `nombre` (comillas simples invertidas) se interpreta como un
# objeto de Python: si existe se enlaza y, si no, se muestra como código.
default_role = "py:obj"

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",     # $...$ y $$...$$ para ecuaciones
    "amsmath",        # entornos de LaTeX (align, ...)
    "colon_fence",    # :::{note} ... ::: para avisos
    "deflist",        # listas de definición
]
myst_heading_anchors = 3   # anclas automáticas para enlazar títulos de las páginas .md

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True            # genera una página por módulo (generated/)
autodoc_default_options = {
    "members": True,                   # documentar las funciones y clases del módulo
    "member-order": "bysource",        # en el orden del código fuente
    "show-inheritance": True,
    # kmc3d/__init__.py reexporta en __all__ los nombres principales (KMC3D, Cell...).
    # Se ignora __all__ para que cada objeto se documente una sola vez, en su submódulo.
    "ignore-module-all": True,
}
# Tipos en la firma: los docstrings ya tienen su propia sección «Parámetros:», y con
# "description" saldría una segunda lista de parámetros solo con los tipos
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
python_maximum_signature_line_length = 88   # firmas largas: un parámetro por línea
# plotly es opcional (solo kmc3d/viz_interactive.py): se simula si no está instalado
autodoc_mock_imports = ["plotly"]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- doctest -----------------------------------------------------------------
# Se ejecuta antes de cada grupo de ejemplos (make doctest)
doctest_global_setup = """
import numpy as np
"""

# -- Salida HTML -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"        # el mismo tema que la documentación de kMC_Malaria
html_static_path = ["_static"]
html_logo = "_static/logo.png"         # generado con docs/make_logo.py
html_favicon = "_static/favicon.png"
html_title = "beta_hematina_kMC"
html_theme_options = {
    "navigation_depth": 3,
}


# -- Docstrings de kmc3d -> reStructuredText ---------------------------------
# Los docstrings del paquete están escritos en un estilo cercano a Markdown:
#   - listas «- texto» o «1. texto» sin línea en blanco antes ni después;
#   - secciones «Parámetros:» seguidas de entradas sangradas «nombre: texto»;
#   - fórmulas y código sangrados tras una línea que acaba en «:»;
#   - |E_att| o configs/*.json, que en reST son marcado (sustitución, énfasis).
# reST los interpretaría mal y daría avisos. En vez de cambiar el código, el evento
# autodoc-process-docstring los adapta aquí, solo para la documentación.

_ITEM = re.compile(r"^(\s*)(?:-|\d+\.)\s+\S")                  # «- texto», «1. texto»
_CAMPO = re.compile(r"^([^:]{1,40}?):\s+(\S.*)$")             # «nombre: texto»
_CODIGO = re.compile(r"(``.+?``|`[^`]+`)")                    # fragmentos `código`


def _sangria(linea):
    """Número de espacios al principio de la línea."""
    return len(linea) - len(linea.lstrip(" "))


def _escapar(texto):
    """Escapa | y * fuera de los fragmentos `código` (en reST son marcado)."""
    partes = _CODIGO.split(texto)
    return "".join(p if p.startswith("`") else p.replace("*", r"\*").replace("|", r"\|")
                   for p in partes)


def _siguiente_no_vacia(lines, i):
    """Primera línea no vacía después de la posición i ('' si no hay)."""
    return next((l for l in lines[i + 1:] if l.strip()), "")


def _docstring_a_rst(app, what, name, obj, options, lines):
    """Convierte in situ (lista `lines`) un docstring de kmc3d a reST válido."""
    out = []
    modo = None        # None, "literal", "seccion" o "lista"
    base = 0           # sangría de referencia del bloque actual

    def cerrar_bloque():
        """Línea en blanco antes de empezar o después de terminar un bloque."""
        if out and out[-1].strip():
            out.append("")

    for i, linea in enumerate(lines):
        vacia = not linea.strip()
        sangria = _sangria(linea)

        # --- Bloque literal: se copia tal cual hasta que la sangría vuelve atrás
        if modo == "literal":
            if vacia or sangria > base:
                out.append(linea)
                continue
            modo = None
            cerrar_bloque()

        # --- Sección «Parámetros:»: entradas sangradas -> lista con el nombre en negrita
        if modo == "seccion":
            if not vacia and sangria > 0:
                texto = linea.strip()
                if sangria <= base:                       # entrada nueva
                    m = _CAMPO.match(texto)
                    if m:
                        out.append(f"- **{_escapar(m.group(1))}**: {_escapar(m.group(2))}")
                    else:
                        out.append(f"- {_escapar(texto)}")
                else:                                     # continuación de la entrada
                    out.append("  " + _escapar(texto))
                continue
            modo = None
            cerrar_bloque()

        # --- Lista con viñetas o numerada: continúa mientras haya elementos o sangría
        if modo == "lista":
            if vacia:
                modo = None
            elif _ITEM.match(linea) or sangria > base:
                out.append(_escapar(linea))
                continue
            else:
                modo = None
                cerrar_bloque()

        if vacia:
            out.append(linea)
            continue

        siguiente = _siguiente_no_vacia(lines, i)
        fin = linea.rstrip()
        termina_en_dos_puntos = fin.endswith(":") and not fin.endswith("::")
        sigue_blanco = i + 1 < len(lines) and not lines[i + 1].strip()

        # «Título:» seguido directamente de entradas sangradas -> sección en negrita
        if (termina_en_dos_puntos and not sigue_blanco and sangria == 0
                and _sangria(siguiente) > 0 and not _ITEM.match(siguiente)):
            cerrar_bloque()
            out.append(f"**{_escapar(linea.strip())}**")
            out.append("")
            modo, base = "seccion", _sangria(siguiente)
            continue

        # Línea que acaba en «:» + línea en blanco + bloque más sangrado -> literal
        if (termina_en_dos_puntos and sigue_blanco and _sangria(siguiente) > sangria
                and not _ITEM.match(siguiente)):
            out.append(_escapar(fin) + ":")             # «:» + «:» = «::» (bloque literal)
            modo, base = "literal", sangria
            continue

        # Primer elemento de una lista
        if _ITEM.match(linea):
            cerrar_bloque()
            out.append(_escapar(linea))
            modo, base = "lista", sangria
            continue

        # Línea sangrada en medio de un párrafo (p. ej. una fórmula): se une al párrafo
        if sangria > 0 and out and out[-1].strip():
            out.append(_escapar(linea.strip()))
            continue

        out.append(_escapar(linea))

    lines[:] = out


def _sin_object(app, name, obj, options, bases):
    """Quita `object` de las clases base mostradas («Bases: object» no aporta nada)."""
    bases[:] = [b for b in bases if b is not object]


def setup(app):
    """Registra los ajustes de autodoc (Sphinx llama a setup al leer conf.py)."""
    app.connect("autodoc-process-docstring", _docstring_a_rst)
    app.connect("autodoc-process-bases", _sin_object)
