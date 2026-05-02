#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import AutoMinorLocator, LogLocator


# ==========================================================
# INPUT FILE
# ==========================================================
# The code will try these names in order.
DATA_FILE_CANDIDATES = [
    "fit_result.dat",
    "fit results.dat",
    "fit_result(2).dat",
    "fit_results.dat",
]

DATA_FILE = None
for fname in DATA_FILE_CANDIDATES:
    if os.path.exists(fname):
        DATA_FILE = fname
        break

if DATA_FILE is None:
    raise FileNotFoundError(
        "Could not find fit_result data file. "
        "Put fit_result.dat in this directory or edit DATA_FILE_CANDIDATES."
    )


# ==========================================================
# OUTPUT
# ==========================================================
FIG1_OUT = "fig1_neet_fit"
FIG2_OUT = "fig2_tail_probability"
FIG3_OUT = "fig3_model_fat_tail"


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
COL_RAW = "#8ecae6"
COL_ENV = "#023047"
COL_FIT = "#fb5607"

COL_TAIL_DATA = "#005f99"
COL_TAIL_FIT = "#ff7f0e"

COL_THIN = "#2a9d8f"
COL_FAT = "#d62828"
COL_SHADE = "#f4a261"

COL_GREY = "0.35"


# ==========================================================
# LOAD DATA
# ==========================================================
data = np.loadtxt(DATA_FILE, comments="#")

marks = data[:, 0].astype(int)
data_count = data[:, 1].astype(float)
fit_count = data[:, 2].astype(float)

N_total = np.sum(data_count)

print(f"Loaded: {DATA_FILE}")
print(f"Total count = {N_total:.0f}")
print(f"Marks range = {marks.min()} to {marks.max()}")


# ==========================================================
# COMMON PARAMETERS
# ==========================================================
Q = 180
d = 4
M_MAX = d * Q
HIGH_CUTOFF = 620

XMIN = -5
XMAX = 725


def beautify_linear_axes(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True)


def round_to_nearest_10000(x):
    return int(np.round(x / 10000.0) * 10000)


# ==========================================================
# SMOOTHING FOR VISUAL ENVELOPE
# ==========================================================
data_env = gaussian_filter1d(data_count, sigma=2.0)
fit_env = gaussian_filter1d(fit_count, sigma=1.0)


# ==========================================================
# FIGURE 1: DATA + FIT
# ==========================================================
fig, ax = plt.subplots(figsize=(10.0, 6.2))

ax.axvspan(
    HIGH_CUTOFF,
    XMAX,
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

ax.plot(
    marks,
    data_count,
    color=COL_RAW,
    lw=0.9,
    alpha=0.65,
    label="NEET 2024 data",
    zorder=2,
)
ax.plot(
    marks,
    data_env,
    color=COL_ENV,
    lw=2.5,
    label="Smoothed data",
    zorder=4,
)

ax.plot(
    marks,
    fit_env,
    color=COL_FIT,
    lw=3.3,
    label="Minimal model fit",
    zorder=5,
)

ax.text(
    0.045,
    0.80,
    r"$M=5r-q$" "\n"
    r"$Q=180,\quad M_{\max}=720$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=17,
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc="white",
        ec="0.75",
        alpha=0.94,
    ),
)

ax.text(
    HIGH_CUTOFF + 8,
    0.55 * np.max(data_count),
    "high-score tail",
    fontsize=14,
    rotation=90,
    va="center",
    ha="left",
    color=COL_GREY,
)

ax.set_xlabel("Marks")
ax.set_ylabel("Number of candidates")

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(0, 1.08 * np.max(data_count))

beautify_linear_axes(ax)

# Centered legend above the plot
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    bbox_transform=ax.transAxes,
    ncol=3,
    columnspacing=1.4,
    handlelength=2.2,
)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG1_OUT + ".png")
fig.savefig(FIG1_OUT + ".pdf")
plt.close(fig)

print(f"Saved {FIG1_OUT}.png and {FIG1_OUT}.pdf")


# ==========================================================
# FIGURE 2: TAIL PROBABILITY
# ==========================================================
tail_data = np.cumsum(data_count[::-1])[::-1]
tail_fit = np.cumsum(fit_count[::-1])[::-1]

tail_data_env = gaussian_filter1d(tail_data, sigma=1.0)
tail_fit_env = gaussian_filter1d(tail_fit, sigma=1.0)

# Number of students above M_c
mask_cut = marks >= HIGH_CUTOFF
N_above_Mc = np.sum(data_count[mask_cut])
N_above_Mc_rounded = round_to_nearest_10000(N_above_Mc)

print(f"Students above {HIGH_CUTOFF} = {N_above_Mc:.0f}")
print(f"Rounded to nearest 10000 = {N_above_Mc_rounded}")

fig, ax = plt.subplots(figsize=(9.4, 6.2))

ax.axvspan(
    HIGH_CUTOFF,
    XMAX,
    color=COL_SHADE,
    alpha=0.13,
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

skip = 10

ax.semilogy(
    marks[::skip],
    tail_data_env[::skip],
    linestyle="None",
    marker="o",
    markersize=5.0,
    markerfacecolor="white",
    markeredgecolor=COL_TAIL_DATA,
    markeredgewidth=1.2,
    alpha=0.85,
    label="Data tail",
    zorder=4,
)

ax.semilogy(
    marks,
    tail_fit_env,
    color=COL_TAIL_FIT,
    lw=3.1,
    linestyle="-",
    label="Model tail",
    zorder=3,
)
# Actual y-value near Mc for arrow target
idx_mc = np.argmin(np.abs(marks - HIGH_CUTOFF))
y_mc = tail_data_env[idx_mc]

# Lower M_c label
ax.text(
    HIGH_CUTOFF + 8,
    1.0e3,
    r"$M_c=620$",
    fontsize=15,
    rotation=90,
    va="bottom",
    ha="left",
    color=COL_GREY,
)

# Arrow annotation
annotation_text = (
    f"{N_above_Mc_rounded:,} students\n"
    f"above {HIGH_CUTOFF} marks limit!"
)

ax.annotate(
    annotation_text,
    xy=(HIGH_CUTOFF, y_mc),
    xytext=(315, 1.2e5),
    textcoords="data",
    arrowprops=dict(
        arrowstyle="->",
        lw=1.8,
        color=COL_TAIL_DATA,
        shrinkA=0,
        shrinkB=5,
    ),
    color=COL_TAIL_DATA,
    fontsize=17,
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc="white",
        ec=COL_TAIL_DATA,
        alpha=0.95,
    ),
)


ax.set_xlabel("Marks")
ax.set_ylabel(r"Number with score $\geq M$")

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(40, 3.0e6)

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which="both", top=True, right=True)

ax.legend(loc="lower left", bbox_to_anchor=(0.04, 0.1))

fig.tight_layout()
fig.savefig(FIG2_OUT + ".png")
fig.savefig(FIG2_OUT + ".pdf")
plt.close(fig)

print(f"Saved {FIG2_OUT}.png and {FIG2_OUT}.pdf")


# ==========================================================
# MODEL FOR FIGURE 3
# ==========================================================
def model_distribution(Q0, L, sigma_q, sigma_r):
    """
    Your model:
        G(q) = exp[-(q-Q0)^2/(2 sigma_q^2)]
        F(r|q) = exp[-(r-Lq)^2/(2 sigma_r^2)]
        P(M) = sum_{q,r} delta_{M,5r-q} G(q)F(r|q)
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
# FIGURE 3: REAL MODEL, SAME PERFORMANCE, DIFFERENT ATTEMPTS
# Log-y version
# ==========================================================
Q0_low = 54
L_low = 0.70
sigma_q_low = 0.4 * Q
sigma_r_low = 0.4 * Q

Q0_high = 144
L_high = 0.70
sigma_q_high = 0.4 * Q
sigma_r_high = 0.4 * Q

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

tail_low = np.sum(P_low[m_low >= HIGH_CUTOFF])
tail_high = np.sum(P_high[m_high >= HIGH_CUTOFF])

print("\nModel figure tail probabilities:")
print(f"Thin-tail model P(M >= {HIGH_CUTOFF}) = {tail_low:.4e}")
print(f"Fat-tail model  P(M >= {HIGH_CUTOFF}) = {tail_high:.4e}")
if tail_low > 0:
    print(f"Tail ratio = {tail_high / tail_low:.2f}")

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

# Curves
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

# Tail fill on log plot: fill from floor
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

# Simple annotation, no big crossing arrows
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

# Selection tail label
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
plt.close(fig)

print(f"Saved {FIG3_OUT}.png and {FIG3_OUT}.pdf")