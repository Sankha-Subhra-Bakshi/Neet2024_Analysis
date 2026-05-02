#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import AutoMinorLocator


# ==========================================================
# OUTPUT
# ==========================================================
FIG3_OUT = "fig3"


# ==========================================================
# STYLE
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",

    "font.size": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,

    "figure.dpi": 130,
    "savefig.dpi": 400,

    "axes.linewidth": 1.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,

    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "0.75",

    "axes.spines.top": True,
    "axes.spines.right": True,
})


# ==========================================================
# COLORS
# ==========================================================
COL_THIN = "#2a9d8f"
COL_FAT = "#d62828"
COL_SHADE = "#f4a261"
COL_GREY = "0.35"


# ==========================================================
# COMMON PARAMETERS
# ==========================================================
Q = 180
d = 4
M_MAX = d * Q          # 720
HIGH_CUTOFF = 620


# ==========================================================
# MODEL
# ==========================================================
def model_distribution(Q0, L, sigma_q, sigma_r):
    """
    Model:

        G(q) = exp[-(q-Q0)^2/(2 sigma_q^2)]

        F(r|q) = exp[-(r-L q)^2/(2 sigma_r^2)]

        M = 5r - q

        P(M) = sum_{q,r} delta_{M,5r-q} G(q) F(r|q)

    Here q is the number of attempted questions and r is the number
    of correct answers among those attempted questions.
    """

    m_values = np.arange(-Q, d * Q + 1, dtype=int)
    P = np.zeros_like(m_values, dtype=float)

    q_values = np.arange(0, Q + 1, dtype=int)

    G = np.exp(-0.5 * ((q_values - Q0) / sigma_q) ** 2)
    G /= np.sum(G)

    for q in q_values:
        r_values = np.arange(0, q + 1, dtype=int)

        if len(r_values) == 0:
            continue

        center = L * q

        F = np.exp(-0.5 * ((r_values - center) / sigma_r) ** 2)
        F_sum = np.sum(F)

        if F_sum == 0:
            continue

        F /= F_sum

        M_values = (d + 1) * r_values - q
        indices = M_values - m_values[0]

        P[indices] += G[q] * F

    P /= np.sum(P)

    return m_values, P


# ==========================================================
# PARAMETERS FOR COMPARISON
# ==========================================================
Q0_low = 54
L_low = 0.70
sigma_q_low = 0.4 * Q
sigma_r_low = 0.4 * Q

Q0_high = 144
L_high = 0.70
sigma_q_high = 0.4 * Q
sigma_r_high = 0.4 * Q


# ==========================================================
# COMPUTE DISTRIBUTIONS
# ==========================================================
m_low, P_low = model_distribution(
    Q0=Q0_low,
    L=L_low,
    sigma_q=sigma_q_low,
    sigma_r=sigma_r_low,
)

m_high, P_high = model_distribution(
    Q0=Q0_high,
    L=L_high,
    sigma_q=sigma_q_high,
    sigma_r=sigma_r_high,
)


# Smooth only for visual envelope
P_low_env = gaussian_filter1d(P_low, sigma=2.0)
P_high_env = gaussian_filter1d(P_high, sigma=2.0)


# Avoid zeros on log scale
P_floor = 1e-6
P_low_plot = np.maximum(P_low_env, P_floor)
P_high_plot = np.maximum(P_high_env, P_floor)


# Tail probabilities
tail_low = np.sum(P_low[m_low >= HIGH_CUTOFF])
tail_high = np.sum(P_high[m_high >= HIGH_CUTOFF])

print("Model figure tail probabilities:")
print(f"Harder-paper model P(M >= {HIGH_CUTOFF}) = {tail_low:.4e}")
print(f"Easy-paper model   P(M >= {HIGH_CUTOFF}) = {tail_high:.4e}")

if tail_low > 0:
    print(f"Tail ratio = {tail_high / tail_low:.2f}")


# ==========================================================
# PLOT
# ==========================================================
fig, ax = plt.subplots(figsize=(9.6, 6.2))


# Tail region
ax.axvspan(
    HIGH_CUTOFF,
    M_MAX,
    color=COL_SHADE,
    alpha=0.12,
    zorder=0,
)

ax.axvline(
    HIGH_CUTOFF,
    color=COL_GREY,
    lw=1.4,
    ls="--",
    alpha=0.85,
    zorder=1,
)


# Main curves
ax.semilogy(
    m_low,
    P_low_plot,
    color=COL_THIN,
    lw=3.0,
    label=(
        rf"Fewer attempts: "
        rf"$Q_0={Q0_low:.0f},\ \mathcal{{L}}={L_low:.2f}$"
    ),
    zorder=3,
)

ax.semilogy(
    m_high,
    P_high_plot,
    color=COL_FAT,
    lw=3.0,
    label=(
        rf"More attempts: "
        rf"$Q_0={Q0_high:.0f},\ \mathcal{{L}}={L_high:.2f}$"
    ),
    zorder=4,
)


# Tail fill
mask_low = m_low >= HIGH_CUTOFF
mask_high = m_high >= HIGH_CUTOFF

ax.fill_between(
    m_low[mask_low],
    P_floor,
    P_low_plot[mask_low],
    color=COL_THIN,
    alpha=0.18,
    zorder=2,
)

ax.fill_between(
    m_high[mask_high],
    P_floor,
    P_high_plot[mask_high],
    color=COL_FAT,
    alpha=0.18,
    zorder=2,
)


# Tail probability box
tail_text = (
    rf"$P(M\geq {HIGH_CUTOFF})$" "\n"
    rf"$Q_0=54$:  ${tail_low:.2e}$" "\n"
    rf"$Q_0=144$: ${tail_high:.2e}$"
)

ax.text(
    0.055,
    0.34,
    tail_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc="white",
        ec="0.75",
        alpha=0.96,
    ),
    zorder=10,
)


# Annotations
x_annot = 645

y_fat = P_high_plot[np.argmin(np.abs(m_high - x_annot))]
y_thin = P_low_plot[np.argmin(np.abs(m_low - x_annot))]

ax.annotate(
    "Easy question paper",
    xy=(x_annot, y_fat),
    xytext=(430, 4e-5),
    arrowprops=dict(
        arrowstyle="->",
        lw=1.6,
        color=COL_FAT,
        shrinkA=0,
        shrinkB=5,
    ),
    color=COL_FAT,
    fontsize=15,
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.30",
        fc="white",
        ec=COL_FAT,
        alpha=0.95,
    ),
    zorder=10,
)

ax.annotate(
    "Harder paper",
    xy=(x_annot, y_thin),
    xytext=(430, 2e-6),
    arrowprops=dict(
        arrowstyle="->",
        lw=1.6,
        color=COL_THIN,
        shrinkA=0,
        shrinkB=5,
    ),
    color=COL_THIN,
    fontsize=15,
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.30",
        fc="white",
        ec=COL_THIN,
        alpha=0.95,
    ),
    zorder=10,
)


# Tail label
ax.text(
    HIGH_CUTOFF + 7,
    2e-6,
    "selection tail",
    fontsize=14,
    rotation=90,
    va="bottom",
    ha="left",
    color=COL_GREY,
)


# Axes
ax.set_xlabel("Marks")
ax.set_ylabel(r"Probability $P(M)$")

ax.set_xlim(-Q, M_MAX)
ax.set_ylim(P_floor, 8e-3)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which="both", top=True, right=True)

ax.legend(
    loc="lower left",
    bbox_to_anchor=(0.5, 0.8),
    fontsize=14,
)

fig.tight_layout()

fig.savefig(FIG3_OUT + ".png")
fig.savefig(FIG3_OUT + ".pdf")

print(f"Saved {FIG3_OUT}.png and {FIG3_OUT}.pdf")