"""Observables de superficie para estudiar la transición entre regímenes.

Funciones puras sobre la matriz de alturas `heights`, válidas para las dos líneas del
motor y para los `snaps.pkl` guardados en `results/`. Diseño y justificación en
`Plan_observables_regimenes.md`. El estudio de transiciones frente a σ (carga unificada
de corridas, parámetros de orden por ventana de espesor, fronteras) está en `runs.py`
y `transitions.py`; se usa en `notebooks/transiciones_sobresaturacion.ipynb`.

No se exporta desde `src/__init__.py` para no añadir nombres a `from src import *`
(que usan los notebooks antiguos). Uso: `from src.analysis import ...`.
"""

from .adsorption import (
    SITE_CLASSES,
    SITE_LABELS,
    adsorption_bond_maps,
    adsorption_probabilities,
    adsorption_rate_table,
    average_cache_window,
    equal_sites_reference,
    history_frame,
    plot_site_probabilities,
    probabilities_from_snapshots,
    run_probabilities,
    scan_heights_cache,
    scan_run_folders,
    site_class_counts,
    summarize_vs_sigma,
    window_average,
)
from .correlation import (
    autocorrelation,
    correlation_extrema,
    correlation_lengths,
    height_difference_correlation,
    local_roughness_exponent,
)
from .islands import (
    component_sizes,
    island_density_observables,
    island_statistics,
    label_periodic,
    level_island_statistics,
    step_statistics,
)
from .moments import (
    effective_levels,
    growing_fraction,
    height_histogram,
    height_moments,
    layer_coverages,
)
from .regime import (
    LATE_COLUMNS,
    growth_exponent,
    growth_rate,
    observables_timeseries,
    plot_observables_vs_sigma,
    sigma_scan_table,
    summarize_run,
    surface_observables,
)
from .runs import (
    PHYSICS_COLUMNS,
    RunData,
    assign_scans,
    discover_runs,
    from_snapshots,
    load_heights_cache,
    load_run_folder,
    runs_inventory,
)
from .transitions import (
    B_GAUSS,
    B_SARLE,
    DEFAULT_CRITERIA,
    REGIME_LABELS,
    Criterion,
    TransitionAnalysis,
    add_regimes,
    aggregate_replicas,
    analyze_transitions,
    boundaries,
    classify_regime,
    compute_tables,
    crossing_sigma,
    ideal_layer_width,
    layer_oscillation,
    level_islands_table,
    order_parameter_catalog,
    random_deposition_reference,
    rd_bimodality,
    rd_effective_levels,
    rd_multi_step_fraction,
    run_timeseries,
    saturation_island_density,
    snapshot_observables,
    steepest_change,
    summarize_run_window,
    summarize_runs,
    transition_table,
)

__all__ = [
    # momentos y capas
    "height_moments", "height_histogram", "layer_coverages", "effective_levels",
    "growing_fraction",
    # islas y escalones
    "label_periodic", "component_sizes", "island_statistics", "step_statistics",
    "level_island_statistics", "island_density_observables",
    # correlación
    "autocorrelation", "correlation_lengths", "height_difference_correlation",
    "local_roughness_exponent", "correlation_extrema",
    # series, resúmenes y barridos
    "surface_observables", "observables_timeseries", "growth_exponent", "growth_rate",
    "summarize_run", "sigma_scan_table", "plot_observables_vs_sigma", "LATE_COLUMNS",
    # probabilidad de adsorción por clase de sitio (Fig. 8 de Nagpal et al. 2024)
    "SITE_CLASSES", "SITE_LABELS", "adsorption_bond_maps", "site_class_counts",
    "adsorption_rate_table", "adsorption_probabilities", "equal_sites_reference",
    "probabilities_from_snapshots", "history_frame", "window_average",
    "run_probabilities", "scan_run_folders", "scan_heights_cache",
    "average_cache_window", "summarize_vs_sigma", "plot_site_probabilities",
    # carga unificada de corridas (runs.py)
    "RunData", "load_run_folder", "load_heights_cache", "from_snapshots", "discover_runs",
    "runs_inventory", "assign_scans", "PHYSICS_COLUMNS",
    # parámetros de orden y transiciones frente a σ (transitions.py)
    "B_SARLE", "B_GAUSS", "rd_multi_step_fraction", "rd_bimodality", "rd_effective_levels",
    "random_deposition_reference", "ideal_layer_width", "layer_oscillation",
    "snapshot_observables", "run_timeseries", "level_islands_table", "compute_tables",
    "summarize_run_window", "saturation_island_density", "summarize_runs",
    "aggregate_replicas", "Criterion", "DEFAULT_CRITERIA", "crossing_sigma",
    "steepest_change", "transition_table", "boundaries", "REGIME_LABELS",
    "classify_regime", "add_regimes", "order_parameter_catalog", "TransitionAnalysis",
    "analyze_transitions",
]
