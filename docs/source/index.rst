.. Página principal de la documentación de beta_hematina_kMC.
   Creada con sphinx-quickstart y adaptada siguiendo el tutorial de Sphinx.
   Debe contener al menos la directiva ``toctree`` raíz.

beta_hematina_kMC
=================

**Kinetic Monte Carlo 3D del crecimiento de un cristal finito de β-hematina**
(hemozoína sintética, el biomineral que forma el parásito de la malaria).

El motor ``kmc3d`` sigue el algoritmo BKL (*n-fold way*, sin rechazo) para un cristal
que crece o se disuelve **por todas sus caras a la vez**. La anisotropía no se impone a
mano: sale de un **grafo de enlaces** sobre la red de Bravais real del cristal (celda
triclínica P-1 en β-hematina), con una unidad de crecimiento por celda. Las ideas
vienen de MONTY y CrystalGrower; la implementación es propia, en Python puro (numpy,
pandas y matplotlib).

.. warning::

   Las energías de enlace de β-hematina son **efectivas y provisionales**, tanto las
   de ``kmc3d/presets.py`` como las calibradas con la morfología. Sirven para explorar
   el modelo, no para sacar conclusiones cuantitativas sobre β-hematina (ver
   :doc:`beta_hematina`).

El logo es la forma de energía de unión de β-hematina (velocidad de cada cara
proporcional a :math:`|E_{att}|`) con las energías calibradas, dibujada con
``kmc3d.render`` (``docs/make_logo.py``).

.. toctree::
   :maxdepth: 2
   :caption: Guía

   introduccion
   instalacion
   uso

.. toctree::
   :maxdepth: 2
   :caption: Modelo y resultados

   modelo
   beta_hematina
   notebooks

.. toctree::
   :maxdepth: 2
   :caption: Desarrollo

   pruebas
   codigo_reutilizado
   api

.. toctree::
   :maxdepth: 1
   :caption: Anexo

   plan

Índices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
