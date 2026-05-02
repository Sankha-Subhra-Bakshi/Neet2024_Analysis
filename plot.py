#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import AutoMinorLocator, LogLocator


# ==========================================================
# INPUT FILES
# ==========================================================
COMPONENT_FILE = "fit_components.dat"
PARAM_FILE = "fit_parameters.dat"

# ==========================================================
# OUTPUT FILES
# ==========================================================
OUTPUT_STACKED_FIG = "fit_stacked_components_replot.png"
OUTPUT_STACKED_FIG_PDF = "fit_stacked_components_replot.pdf"

OUTPUT_TAIL_FIG = "fit_tail_replot.png"
OUTPUT_TAIL_FIG_PDF = "fit_tail_replot.pdf"


# ==========================================================
# CONSTANTS
# ==========================================================
Q = 180
M_MAX = 720
HIGH_CUTOFF = 620

XMIN = -5
XMAX = 725


# ==========================================================
# PLOT STYLE
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
    "legend.fontsize": 12,

    "figure.dpi": 130,
    "savefig.dpi": 400,

    "axes.linewidth": 1.35,
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
# Stacked-plot colors, inspired by old historical stacked-area plots
COLORS_COMPONENTS = [
    "#1f78b4",  # blue
    "#33a02c",  # green
    "#ffd92f",  # yellow
    "#ff7f00",  # orange
    "#e31a1c",  # red
]

COL_DATA_RAW = "#9bd3f5"
COL_DATA_SMOOTH = "r"
COL_TOTAL_FIT = "#111111"
COL_TAIL_SHADE = "#f4a261"
COL_GREY = "0.35"

COL_TAIL_DATA = "#005f99"
COL_TAIL_FIT = "#ff7f0e"


# ==========================================================
# READ COMPONENT DATA
# ==========================================================
arr = np.loadtxt(COMPONENT_FILE, comments="#")

marks = arr[:, 0].astype(int)

# File format:
# mark component_1 component_2 ... component_K total_fit_count data_count
component_counts = arr[:, 1:-2].T
fit_counts = arr[:, -2]
data_counts = arr[:, -1]

K_COMPONENTS = component_counts.shape[0]
N_total = np.sum(data_counts)

print(f"Loaded {COMPONENT_FILE}")
print(f"  K_COMPONENTS = {K_COMPONENTS}")
print(f"  Total candidates = {N_total:.0f}")
print(f"  Marks range = {marks.min()} to {marks.max()}")


# ==========================================================
# READ PARAMETERS
# ==========================================================
# Expected rows:
# sector weight Q0 L sigma_q sigma_r mean_mark
params = np.loadtxt(PARAM_FILE, comments="#")

if params.ndim == 1:
    params = params.reshape(1, -1)

# Make sure order follows sector number
params = params[np.argsort(params[:, 0])]

sector_id = ['Not-serious', 'Semi-serious', 'Serious']
weights = params[:, 1]
Q0s = params[:, 2]
Ls = params[:, 3]
sigma_qs = params[:, 4]
sigma_rs = params[:, 5]
mean_marks = params[:, 6]

print(f"Loaded {PARAM_FILE}")
for i in range(K_COMPONENTS):
    print(
        f"  sector {sector_id[i]}: "
        f"weight={weights[i]:.3f}, "
        f"Q0={Q0s[i]:.2f}, "
        f"L={Ls[i]:.3f}, "
        f"mean_mark={mean_marks[i]:.2f}"
    )


# ==========================================================
# SMOOTHING FOR DISPLAY ONLY
# ==========================================================
smooth_sigma_components = 4.0
smooth_sigma_data = 5.0
smooth_sigma_fit = 3.0

component_smooth = np.array([
    gaussian_filter1d(component_counts[k], sigma=smooth_sigma_components)
    for k in range(K_COMPONENTS)
])

data_smooth = gaussian_filter1d(data_counts, sigma=smooth_sigma_data)
fit_smooth = gaussian_filter1d(fit_counts, sigma=smooth_sigma_fit)


# ==========================================================
# FIGURE 1: STACKED COMPONENTS
# ==========================================================
fig, ax = plt.subplots(figsize=(10.4, 6.4))

# High-score tail shading
ax.axvspan(
    HIGH_CUTOFF,
    XMAX,
    color=COL_TAIL_SHADE,
    alpha=0.12,
    zorder=0,
)

ax.axvline(
    HIGH_CUTOFF,
    color=COL_GREY,
    lw=1.4,
    ls="--",
    alpha=0.85,
    zorder=2,
)

# Component labels
sector_labels = []
for k in range(K_COMPONENTS):
    sector_labels.append(
        rf"sector {k+1}: "
        rf"$Q_0={Q0s[k]:.0f},\ \mathcal{{L}}={Ls[k]:.2f}$"
    )

# Stacked components
ax.stackplot(
    marks,
    component_smooth,
    labels=sector_labels,
    colors=COLORS_COMPONENTS[:K_COMPONENTS],
    alpha=0.72,
    linewidth=0.7,
    edgecolor="0.25",
    zorder=1,
)

# Total fit
ax.plot(
    marks,
    fit_smooth,
    color=COL_TOTAL_FIT,
    lw=2.8,
    label="total fit",
    zorder=5,
)

# Raw data
ax.plot(
    marks,
    data_counts,
    color=COL_DATA_RAW,
    lw=0.65,
    alpha=0.45,
    label="raw data",
    zorder=3,
)

# Smoothed data
ax.plot(
    marks,
    data_smooth,
    color=COL_DATA_SMOOTH,
    lw=2.5,
    label="smoothed data",
    zorder=6,
)

# Equation box
ax.text(
    0.035,
    0.79,
    r"$M=5r-q$" "\n"
    r"$Q=180,\quad M_{\max}=720$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=16,
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc="white",
        ec="0.75",
        alpha=0.94,
    ),
    zorder=10,
)

# Tail label
ax.text(
    HIGH_CUTOFF + 8,
    0.48 * np.max(data_counts),
    "high-score tail",
    fontsize=13,
    rotation=90,
    va="center",
    ha="left",
    color=COL_GREY,
)

ax.set_xlabel("Marks")
ax.set_ylabel("Number of candidates")

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(0, 1.08 * np.max(data_counts))

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which="both", top=True, right=True)

# Legend placement similar to your latest style
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper right",
    bbox_to_anchor=(0.70, 0.98),
    ncol=1,
    frameon=True,
    framealpha=0.96,
    borderpad=0.7,
)

fig.tight_layout()
fig.savefig(OUTPUT_STACKED_FIG)
fig.savefig(OUTPUT_STACKED_FIG_PDF)
plt.close(fig)

print(f"Saved {OUTPUT_STACKED_FIG}")
print(f"Saved {OUTPUT_STACKED_FIG_PDF}")


# ==========================================================
# FIGURE 2: TAIL / SURVIVAL PLOT
# ==========================================================
tail_data = np.cumsum(data_counts[::-1])[::-1]
tail_fit = np.cumsum(fit_counts[::-1])[::-1]

tail_data_smooth = gaussian_filter1d(tail_data, sigma=1.0)
tail_fit_smooth = gaussian_filter1d(tail_fit, sigma=1.0)

fig, ax = plt.subplots(figsize=(9.4, 6.2))

ax.axvspan(
    HIGH_CUTOFF,
    XMAX,
    color=COL_TAIL_SHADE,
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

skip = 5

ax.semilogy(
    marks[::skip],
    tail_data_smooth[::skip],
    linestyle="None",
    marker="o",
    markersize=4.0,
    markerfacecolor="white",
    markeredgecolor=COL_TAIL_DATA,
    markeredgewidth=1.2,
    alpha=0.90,
    label="data tail",
    zorder=4,
)

ax.semilogy(
    marks,
    tail_fit_smooth,
    color=COL_TAIL_FIT,
    lw=3.0,
    label="model tail",
    zorder=3,
)

idx_mc = np.argmin(np.abs(marks - HIGH_CUTOFF))
N_above_Mc = tail_data[idx_mc]
N_above_Mc_rounded = int(np.round(N_above_Mc / 10000.0) * 10000)

ax.annotate(
    f"{N_above_Mc_rounded:,} students\nabove {HIGH_CUTOFF} marks limit!",
    xy=(HIGH_CUTOFF, tail_data_smooth[idx_mc]),
    xytext=(315, 1.2e5),
    arrowprops=dict(
        arrowstyle="->",
        lw=1.8,
        color=COL_TAIL_DATA,
        shrinkA=0,
        shrinkB=5,
    ),
    color=COL_TAIL_DATA,
    fontsize=16,
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc="white",
        ec=COL_TAIL_DATA,
        alpha=0.95,
    ),
)

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

ax.set_xlabel("Marks")
ax.set_ylabel(r"Number with score $\geq M$")

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(40, 3.0e6)

ax.yaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which="both", top=True, right=True)

ax.legend(
    loc="lower left",
    bbox_to_anchor=(0.04, 0.08),
)

fig.tight_layout()
fig.savefig(OUTPUT_TAIL_FIG)
fig.savefig(OUTPUT_TAIL_FIG_PDF)
plt.close(fig)

print(f"Saved {OUTPUT_TAIL_FIG}")
print(f"Saved {OUTPUT_TAIL_FIG_PDF}")


# ==========================================================
# PRINT USEFUL NUMBERS
# ==========================================================
print("\nUseful tail numbers:")
for cut in [500, 550, 600, 620, 650, 680, 700]:
    idx = np.argmin(np.abs(marks - cut))
    print(
        f"M >= {cut:3d}: "
        f"data = {tail_data[idx]:10.0f}, "
        f"fit = {tail_fit[idx]:10.0f}, "
        f"data fraction = {tail_data[idx] / N_total:.4e}, "
        f"fit fraction = {tail_fit[idx] / N_total:.4e}"
    )

print("\nDone.")