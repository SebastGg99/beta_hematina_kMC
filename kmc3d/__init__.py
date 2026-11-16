"""kmc3d: kinetic Monte Carlo 3D de crecimiento cristalino (todas las caras).

Uso mínimo (desde un notebook o script):

    import sys; sys.path.insert(0, "<ruta>/beta_hematina_kMC")
    from kmc3d import presets, seeds, SplitRates, KMC3D
    g = presets.kossel(phi=2.0)
    kmc = KMC3D(g, SplitRates.gilmer(g, delta_mu=1.0), seeds.cube(8), rng_seed=1,
                track_faces=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    snaps, stats = kmc.run(max_events=50_000, snapshot_every=10_000)

Ver PLAN_3DkMC.md (diseño y fases) y README.md (uso).
"""

from . import _paths  # noqa: F401  (añade la raíz del repositorio a sys.path)
from . import presets, seeds
from .engine import KMC3D
from .geometry import Cell, family, hkl_label, reduce_hkl
from .graph import BondType, CrystalGraph
from .rates import AdaptiveRates, RateModel, SplitRates, delta_mu_from_sigma, rates_from_dict

__all__ = [
    "KMC3D", "Cell", "BondType", "CrystalGraph", "RateModel", "SplitRates", "AdaptiveRates",
    "delta_mu_from_sigma", "rates_from_dict", "reduce_hkl", "hkl_label", "family",
    "presets", "seeds",
]
