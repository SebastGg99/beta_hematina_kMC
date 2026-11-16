"""Grafos cristalinos predefinidos: Kossel, cúbico anisótropo y β-hematina.

β-HEMATINA (hemozoína sintética)
--------------------------------
Celda (Pagola et al., Nature 404, 307, 2000; valores citados en la nota 26 de
Solomonov et al., JACS 129, 2615, 2007): triclínica P-1,
    a = 12.196 Å, b = 14.684 Å, c = 8.040 Å, α = 90.22°, β = 96.80°, γ = 97.92°,
y recíproca a* = 0.08338, b* = 0.06876, c* = 0.12527 Å⁻¹, α* = 88.83°,
β* = 83.05°, γ* = 82.00° (comprobado en tests/test_geometry.py).

Z = 2 hemos forman UN dímero centrosimétrico por celda (enlace Fe–O–propionato), así
que la unidad de crecimiento natural es el dímero y la red de Bravais con una unidad
por celda es exacta para la topología. Según Solomonov et al. (2007), los dímeros se
unen por pares de puentes de hidrógeno O–H···O=C en cadenas paralelas a a − c, es
decir, al vector [10-1].

Morfología observada (misma referencia): agujas o láminas alargadas según c,
limitadas por caras laterales {100} y {010} (anchura/espesor ≈ 3:1) y rematadas por
{011} y, en menor medida, {001}. La {0-11} no aparece (muy corrugada). La (001) tiene
la energía de unión más negativa de las caras expresadas (crece más rápido).

ENERGÍAS: no hay aquí valores calculados con un campo de fuerzas. Las de
`BETA_HEMATIN_ENERGIES_PROVISIONAL` son EFECTIVAS (en solución, unidades de kT) y se
eligieron solo para que la teoría de energía de unión ordene las caras como se
observa: |E_att| (100) < (010) < (011) < (001) < (01-1). Calibrarlas es la Fase 3 de
PLAN_3DkMC.md.
"""

from typing import Dict, Optional

from .geometry import Cell, parse_hkl
from .graph import BondType, CrystalGraph

# ---------------------------------------------------------------------------
# Celdas
# ---------------------------------------------------------------------------
CUBIC_CELL = Cell(1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
BETA_HEMATIN_CELL = Cell(12.196, 14.684, 8.040, 90.22, 96.80, 97.92)

# Caras de interés para β-hematina (observadas + candidatas de baja energía)
BETA_HEMATIN_FACES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, -1),
                      (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0))
BETA_HEMATIN_OBSERVED = {"laterales": ((1, 0, 0), (0, 1, 0)),
                         "extremos": ((0, 1, 1), (0, 0, 1)),
                         "ausente": ((0, 1, -1),),
                         "eje_aguja": (0, 0, 1)}

# Energías efectivas φ/kT PROVISIONALES (ver docstring): una por vector de enlace
BETA_HEMATIN_ENERGIES_PROVISIONAL: Dict[str, float] = {
    "[001]": 2.4,    # traslación más corta (8.04 Å): apilamiento de porfirinas, eje de la aguja
    "[10-1]": 1.0,   # cadenas de puentes de H O–H···O=C (a − c)
    "[100]": 0.4,
    "[010]": 0.9,
    "[011]": 0.5,
    "[01-1]": 1.1,
    "[101]": 0.3,
}


def kossel(phi: float = 2.0) -> CrystalGraph:
    """Cristal de Kossel: red cúbica simple, 6 vecinos, misma energía φ/kT."""
    return CrystalGraph(CUBIC_CELL, (BondType((1, 0, 0), phi), BondType((0, 1, 0), phi),
                                     BondType((0, 0, 1), phi)),
                        name=f"Kossel (φ/kT = {phi:g})")


def anisotropic_cubic(phi_x: float, phi_y: float, phi_z: float, phi_xy: float = 0.0,
                      phi_xz: float = 0.0, phi_yz: float = 0.0,
                      cell: Optional[Cell] = None) -> CrystalGraph:
    """Cubitos con energía distinta por eje y, opcionalmente, segundos vecinos.

    phi_xy añade los enlaces diagonales [110] y [1-10] (análogo para xz, yz). Con
    segundos vecinos las caras {110} pasan a ser F y la forma se redondea.
    `cell` permite deformar los cubitos (p. ej. una celda ortorrómbica).
    """
    bonds = [BondType((1, 0, 0), phi_x), BondType((0, 1, 0), phi_y), BondType((0, 0, 1), phi_z)]
    for phi, (u, v) in ((phi_xy, ((1, 1, 0), (1, -1, 0))), (phi_xz, ((1, 0, 1), (1, 0, -1))),
                        (phi_yz, ((0, 1, 1), (0, 1, -1)))):
        if phi > 0.0:
            bonds += [BondType(u, phi), BondType(v, phi)]
    return CrystalGraph(cell or CUBIC_CELL, tuple(bonds),
                        name=f"cúbico anisótropo (φx, φy, φz) = ({phi_x:g}, {phi_y:g}, {phi_z:g})")


def beta_hematin(energies: Optional[Dict[str, float]] = None, scale: float = 1.0) -> CrystalGraph:
    """β-hematina: celda triclínica real + enlaces efectivos por vector de red.

    Parámetros:
        energies: {etiqueta: φ/kT} para sustituir las provisionales (p. ej. '[001]')
        scale: factor común (equivale a cambiar la temperatura efectiva)
    """
    e = dict(BETA_HEMATIN_ENERGIES_PROVISIONAL)
    if energies:
        unknown = set(energies) - set(e)
        if unknown:
            raise KeyError(f"Enlaces desconocidos: {sorted(unknown)}; disponibles {sorted(e)}")
        e.update(energies)
    bonds = tuple(BondType(tuple(int(x) for x in _parse_uvw(lbl)), scale * phi, lbl)
                  for lbl, phi in e.items())
    return CrystalGraph(BETA_HEMATIN_CELL, bonds, name="β-hematina (energías provisionales)",
                        notes="Celda: Pagola et al. 2000. Energías efectivas provisionales: "
                              "ver kmc3d/presets.py y PLAN_3DkMC.md (Fase 3).")


def _parse_uvw(label: str):
    """'[01-1]' -> (0, 1, -1)."""
    return parse_hkl(label)
