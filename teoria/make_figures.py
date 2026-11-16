"""Figuras de datos del documento teórico `teoria/teoria_3DkMC.tex`.

Todas se calculan con el propio paquete `kmc3d` (no hay números copiados a mano):
- celda_bh_enlaces.pdf:   celda triclínica de β-hematina y sus 7 vectores de enlace
- tasas_vs_enlaces.pdf:   k+ y k− frente al número de enlaces (Kossel) y balance detallado
- gibbs_thomson.pdf:      energía libre de un cubo de Kossel y tamaño crítico L*
- clases_estado.pdf:      clases n-fold way de un estado real (sitios, tasa y peso)

Uso (desde la raíz del repositorio):
    python teoria/make_figures.py
Las figuras se escriben en teoria/figuras/. Las figuras de resultados
(notebook 01) se copian de outputs/01_primeros_resultados/.
"""

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # sin ventana: solo se guardan archivos
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

# raíz del repositorio en sys.path para importar kmc3d (y con él src/)
TEORIA = Path(__file__).resolve().parent
ROOT_3D = TEORIA.parent
sys.path.insert(0, str(ROOT_3D))

from kmc3d import KMC3D, SplitRates, presets, seeds  # noqa: E402

FIG = TEORIA / "figuras"
FIG.mkdir(exist_ok=True)

# Estilo común: serif, tamaños pensados para \linewidth de un artículo a 11 pt
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
})
# Paleta categórica fija del proyecto (kmc3d.viz.PALETTE) y marcadores por serie
PAL = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
MARK = ["o", "s", "^", "D", "v", "P", "X"]


def fig_celda_bh() -> None:
    """Celda de β-hematina (aristas) y los 7 tipos de enlace ±v_t, grosor ∝ φ."""
    g = presets.beta_hematin()
    M = g.cell.matrix
    fig = plt.figure(figsize=(6.6, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0], projection="3d")
    # aristas del paralelepípedo: pares de vértices que difieren en un solo índice
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    for p in corners:
        for q in corners:
            if np.abs(p - q).sum() == 1 and tuple(p) < tuple(q):
                P, Q = p @ M, q @ M
                ax.plot(*zip(P, Q), color="#999999", lw=0.8)
    # vectores de la celda a, b, c desde el origen
    for v, lab in zip(M, ("a", "b", "c")):
        ax.quiver(0, 0, 0, *v, color="k", lw=1.2, arrow_length_ratio=0.08)
        ax.text(*(1.08 * v), rf"$\mathbf{{{lab}}}$", fontsize=9)
    # enlaces ±v_t desde una unidad situada en el centro de la celda
    c0 = 0.5 * M.sum(axis=0)
    phis = g.energies
    for t, b in enumerate(g.bonds):
        v = np.asarray(b.vector) @ M
        for s in (1, -1):
            ax.plot(*zip(c0, c0 + s * v), color=PAL[t], lw=0.6 + 1.3 * b.energy)
        ax.scatter(*(c0 + v), color=PAL[t], s=8)
        ax.scatter(*(c0 - v), color=PAL[t], s=8)
    ax.scatter(*c0, color="k", s=25, zorder=5)
    ax.set_box_aspect(np.ptp(np.vstack([c0 + np.array([b.vector for b in g.bonds]) @ M,
                                        c0 - np.array([b.vector for b in g.bonds]) @ M]),
                             axis=0))
    ax.set_xlabel("x (Å)", labelpad=-6)
    ax.set_ylabel("y (Å)", labelpad=-6)
    ax.set_zlabel("z (Å)", labelpad=-6)
    ax.tick_params(pad=-2, labelsize=6)
    ax.view_init(elev=18, azim=-62)
    ax.set_title("(a) celda P$\\bar{1}$ y 14 vecinos", pad=0)

    # (b) energía y longitud de cada tipo de enlace
    ax2 = fig.add_subplot(gs[1])
    lengths = [g.cell.vector_length(b.vector) for b in g.bonds]
    order = np.argsort(lengths)
    xs = np.arange(len(order))
    ax2.bar(xs, phis[order], color=[PAL[i] for i in order], edgecolor="k", lw=0.4)
    for x, i in zip(xs, order):
        ax2.text(x, phis[i] + 0.05, f"{lengths[i]:.1f} Å", ha="center", fontsize=7, rotation=90,
                 va="bottom")
    ax2.set_xticks(xs, [g.bonds[i].label for i in order], rotation=45)
    ax2.set_ylabel(r"$\varphi_t / kT$ (provisional)")
    ax2.set_ylim(0, 3.3)
    ax2.set_title("(b) energía por tipo (ordenados por longitud)")
    fig.tight_layout()
    fig.savefig(FIG / "celda_bh_enlaces.pdf")
    plt.close(fig)


def fig_tasas() -> None:
    """k+ y k− de SplitRates frente a n (Kossel, φ = 2, Δμ = 1) y ln(k+/k−)."""
    phi, dmu = 2.0, 1.0
    g = presets.kossel(phi)
    modelos = [("Gilmer ($a=1,\\ b=0$)", SplitRates.gilmer(g, dmu)),
               ("simétrico ($a=b=\\frac{1}{2}$)", SplitRates.symmetric(g, dmu)),
               ("intermedio ($a=1,\\ b=\\frac{1}{4}$)", SplitRates(g, dmu, a_mu=1.0, b_E=0.25))]
    # un código representativo por número de enlaces n (en Kossel solo importa n)
    ns = np.arange(0, 7)
    codes = [g.encode(tuple(min(2, max(0, n - 2 * t)) for t in range(3))) for n in ns]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for i, (lab, r) in enumerate(modelos):
        kp = [r.attach(c) for c in codes]
        km = [r.detach(c) for c in codes]
        axes[0].plot(ns, kp, color=PAL[i], marker=MARK[i], lw=1.6, label=f"$k^+$ {lab}")
        axes[0].plot(ns, km, color=PAL[i], marker=MARK[i], lw=1.0, ls="--", mfc="none")
        axes[1].plot(ns, np.log(np.array(kp) / np.array(km)), color=PAL[i], marker=MARK[i],
                     lw=1.4, ms=6 - 1.5 * i, label=lab)
    axes[0].set_yscale("log")
    axes[0].axvline(3, color="#888888", lw=0.8, ls=":")
    axes[0].text(3.1, 150, "kink ($n=3$)", fontsize=7, color="#555555")
    axes[0].set_xlabel("vecinos ocupados $n$")
    axes[0].set_ylabel("tasa / $k_0$")
    axes[0].set_title("(a) $k^+$ (continua) y $k^-$ (discontinua)")
    nn = np.linspace(0, 6, 50)
    axes[1].plot(nn, dmu + (nn - 3) * phi, color="k", lw=0.8, ls=":",
                 label=r"$-\Delta G=\Delta\mu+(E-E_{\rm kink})$")
    axes[1].set_xlabel("vecinos ocupados $n$")
    axes[1].set_ylabel(r"$\ln(k^+/k^-)$")
    axes[1].set_title("(b) balance detallado: igual para todos")
    axes[1].legend(fontsize=6.5, frameon=False)
    fig.suptitle(rf"Kossel, $\varphi/kT={phi:g}$, $\Delta\mu/kT={dmu:g}$", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "tasas_vs_enlaces.pdf")
    plt.close(fig)


def fig_gibbs_thomson() -> None:
    """G(L) = −Δμ L³ + 3φ L² (cubo de Kossel) y el máximo en L* = 2φ/Δμ."""
    phi = 2.0
    L = np.linspace(0, 16, 400)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for i, dmu in enumerate((0.5, 0.75, 1.0)):
        G = -dmu * L**3 + 3 * phi * L**2
        Ls = 2 * phi / dmu
        ax.plot(L, G, color=PAL[i], lw=1.6, label=rf"$\Delta\mu={dmu:g}$, $L^*={Ls:.2g}$")
        ax.plot(Ls, -dmu * Ls**3 + 3 * phi * Ls**2, marker=MARK[i], color=PAL[i], ms=6)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_ylim(-60, 140)
    ax.set_xlabel("lado del cubo $L$ (celdas)")
    ax.set_ylabel(r"$G(L)/kT$")
    ax.set_title(r"Kossel, $\varphi/kT=2$")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "gibbs_thomson.pdf")
    plt.close(fig)


def fig_clases() -> None:
    """Clases n-fold way de un estado real: Kossel φ = 2, Δμ = 1, tasas simétricas."""
    g = presets.kossel(2.0)
    kmc = KMC3D(g, SplitRates.symmetric(g, 1.0), seeds.cube(10), rng_seed=1)
    kmc.run(max_events=20_000)
    ct = kmc.class_table()
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharex=True)
    for j, (ev, lab) in enumerate((("attach", "unión"), ("detach", "despegue"))):
        sub = ct[ct["evento"] == ev].groupby("n_enlaces").agg(
            n_sitios=("n_sitios", "sum"), tasa=("tasa", "first"), peso=("peso", "sum"))
        off = -0.2 + 0.4 * j
        kw = dict(width=0.38, color=PAL[j], edgecolor="k", lw=0.4, label=lab)
        axes[0].bar(sub.index + off, sub["n_sitios"], **kw)
        axes[1].bar(sub.index + off, sub["tasa"], **kw)
        axes[2].bar(sub.index + off, sub["peso"] / ct["peso"].sum(), **kw)
    axes[0].set_ylabel("sitios $n_c$")
    axes[1].set_ylabel("tasa $r_c / k_0$")
    axes[1].set_yscale("log")
    axes[2].set_ylabel(r"peso $n_c r_c / W$")
    for ax, t in zip(axes, ("(a) sitios por clase", "(b) tasa de la clase", "(c) probabilidad")):
        ax.set_xlabel("vecinos ocupados $n$")
        ax.set_title(t)
        ax.set_xticks(range(0, 7))
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Estado tras 20 000 eventos (Kossel, $\\varphi=2$, $\\Delta\\mu=1$, simétricas, "
                 f"N = {kmc.n_occ}, {len(ct)} clases no vacías)", y=1.03, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "clases_estado.pdf")
    plt.close(fig)


def copy_notebook_figures() -> None:
    """Copia las figuras de resultados del notebook 01 (para que la carpeta sea autónoma)."""
    src = ROOT_3D / "outputs" / "01_primeros_resultados"
    for name in ("kossel_tamano_critico", "kossel_R_vs_dmu", "cubitos_anisotropos",
                 "forma_energia_union_bh", "bh_R_por_cara", "bh_forma_vs_dmu",
                 "bh_voxeles", "bh_probabilidad_union",
                 "bh_mapa_alturas_100"):
        shutil.copy2(src / f"{name}.png", FIG / f"{name}.png")


if __name__ == "__main__":
    fig_celda_bh()
    fig_tasas()
    fig_gibbs_thomson()
    fig_clases()
    copy_notebook_figures()
    print("Figuras en", FIG)
