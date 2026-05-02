#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import AutoMinorLocator


# ==========================================================
# USER SETTINGS
# ==========================================================
DATA_FILE = "data.dat"

Q = 180
d = 4

K_COMPONENTS = 3
CONDITION_ON_OBSERVED_SUPPORT = True

OUTPUT_FIT_DATA = "fit_result.dat"
OUTPUT_PARAM_DATA = "fit_parameters.dat"
OUTPUT_COMPONENT_DATA = "fit_components.dat"

OUTPUT_STACKED_FIG = "fit_stacked_components.png"
OUTPUT_STACKED_FIG_PDF = "fit_stacked_components.pdf"

OUTPUT_TAIL_FIG = "fit_tail.png"
OUTPUT_TAIL_FIG_PDF = "fit_tail.pdf"

N_STARTS = 12
RANDOM_SEED = 12345


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


# Colors inspired by old stacked historical-emission plots
COLORS_COMPONENTS = [
    "#1f78b4",  # blue
    "#33a02c",  # green
    "#ffd92f",  # yellow
    "#ff7f00",  # orange
    "#e31a1c",  # red
]

COL_DATA_RAW = "#9bd3f5"
COL_DATA_SMOOTH = "k",  # blue
COL_TOTAL_FIT = "r",  # red
COL_TAIL_SHADE = "#f4a261"
COL_GREY = "0.35"


# ==========================================================
# READ DATA
# ==========================================================
data = np.loadtxt(DATA_FILE, comments="#")

marks_data = data[:, 0].astype(int)
counts_data = data[:, 1].astype(float)

N_total = np.sum(counts_data)

print("Loaded data:")
print(f"  Number of mark bins = {len(marks_data)}")
print(f"  Total students/count = {N_total:.0f}")
print(f"  Min mark = {marks_data.min()}")
print(f"  Max mark = {marks_data.max()}")


# ==========================================================
# FULL MARK SUPPORT
# ==========================================================
m_min_model = -Q
m_max_model = d * Q

marks_model = np.arange(m_min_model, m_max_model + 1, dtype=int)
n_marks_model = len(marks_model)

mark_to_index = {m: i for i, m in enumerate(marks_model)}
observed_indices = np.array([mark_to_index[m] for m in marks_data], dtype=int)


# ==========================================================
# MODEL P(M)
# ==========================================================
def one_component_distribution(Q0, L, sigma_q, sigma_r):
    """
    One effective attempt--correctness sector.

    G(q)    ~ truncated Gaussian centered at Q0
    F(r|q)  ~ truncated Gaussian centered at L*q
    M       = (d+1)r - q
    """

    q_values = np.arange(0, Q + 1, dtype=int)

    G = np.exp(-0.5 * ((q_values - Q0) / sigma_q) ** 2)
    G_sum = np.sum(G)
    if G_sum <= 0:
        raise RuntimeError("Bad G(q) normalization.")
    G /= G_sum

    P = np.zeros(n_marks_model, dtype=float)

    for q in q_values:
        r_values = np.arange(0, q + 1, dtype=int)

        center = L * q
        F = np.exp(-0.5 * ((r_values - center) / sigma_r) ** 2)

        F_sum = np.sum(F)
        if F_sum <= 0:
            continue

        F /= F_sum

        M_values = (d + 1) * r_values - q
        indices = M_values - m_min_model

        P[indices] += G[q] * F

    P_sum = np.sum(P)
    if P_sum <= 0:
        raise RuntimeError("Bad P(M) normalization.")

    P /= P_sum
    return P


def softmax(a):
    a = np.asarray(a)
    a = a - np.max(a)
    e = np.exp(a)
    return e / np.sum(e)


def unpack_params(params, K):
    comp_params = []
    idx = 0

    for _ in range(K):
        Q0 = params[idx]
        L = params[idx + 1]
        sigma_q = params[idx + 2]
        sigma_r = params[idx + 3]
        idx += 4

        comp_params.append((Q0, L, sigma_q, sigma_r))

    weight_logits = params[idx:idx + K]
    weights = softmax(weight_logits)

    return comp_params, weights


def component_distributions(params, K):
    """
    Returns:
        P_components_full[k, m] : contribution of component k to P(M)
        P_total_full[m]         : total P(M)

    If CONDITION_ON_OBSERVED_SUPPORT is True, all components are rescaled by the
    same factor so that the total probability over observed mark bins is 1.
    """

    comp_params, weights = unpack_params(params, K)

    P_components = []

    for w, (Q0, L, sigma_q, sigma_r) in zip(weights, comp_params):
        Pk = one_component_distribution(Q0, L, sigma_q, sigma_r)
        P_components.append(w * Pk)

    P_components = np.array(P_components)
    P_total = np.sum(P_components, axis=0)

    P_total_sum = np.sum(P_total)
    if P_total_sum <= 0:
        raise RuntimeError("Bad mixture normalization.")

    P_components /= P_total_sum
    P_total /= P_total_sum

    if CONDITION_ON_OBSERVED_SUPPORT:
        obs_total = np.sum(P_total[observed_indices])
        P_components /= obs_total
        P_total /= obs_total

    return P_components, P_total


def mixture_distribution(params, K):
    _, P_total = component_distributions(params, K)
    return P_total


# ==========================================================
# NEGATIVE LOG-LIKELIHOOD
# ==========================================================
def negative_log_likelihood(params, K):
    P = mixture_distribution(params, K)
    P_obs = P[observed_indices]

    eps = 1e-300
    P_obs = np.maximum(P_obs, eps)

    mu = N_total * P_obs

    # Poisson negative log-likelihood, ignoring the constant log(n!)
    nll = np.sum(mu - counts_data * np.log(mu))

    if not np.isfinite(nll):
        return 1e100

    return nll


# ==========================================================
# INITIALIZATION
# ==========================================================
def initial_params(K):
    params = []

    if K == 1:
        guesses = [(100.0, 0.55, 40.0, 25.0)]
    elif K == 3:
        # More physically separated initial guesses.
        guesses = [
            (70.0, 0.40, 20.0, 10.0),
            (100.0, 0.60, 20.0, 10.0),
            (130.0, 0.80, 45.0, 25.0),
        ]
    else:
        Q0_guesses = np.linspace(50, 160, K)
        L_guesses = np.linspace(0.35, 0.85, K)
        guesses = []

        for Q0, L in zip(Q0_guesses, L_guesses):
            guesses.append((Q0, L, 35.0, 18.0))

    for Q0, L, sigma_q, sigma_r in guesses:
        params.extend([Q0, L, sigma_q, sigma_r])

    # Equal weights initially
    params.extend([0.0] * K)

    return np.array(params, dtype=float)


def parameter_bounds(K):
    bounds = []

    for _ in range(K):
        bounds.append((0.0, Q))        # Q0
        bounds.append((0.02, 0.999))   # L
        bounds.append((1.0, 120.0))    # sigma_q
        bounds.append((1.0, 120.0))    # sigma_r

    for _ in range(K):
        bounds.append((-8.0, 8.0))     # weight logits

    return bounds


# ==========================================================
# FIT
# ==========================================================
best_result = None
rng = np.random.default_rng(RANDOM_SEED)
bounds = parameter_bounds(K_COMPONENTS)

lower_bounds = np.array([b[0] for b in bounds])
upper_bounds = np.array([b[1] for b in bounds])

for start in range(N_STARTS):
    p0 = initial_params(K_COMPONENTS)

    if start > 0:
        for k in range(K_COMPONENTS):
            base = 4 * k

            p0[base + 0] += rng.normal(0, 22)       # Q0
            p0[base + 1] += rng.normal(0, 0.12)     # L
            p0[base + 2] *= np.exp(rng.normal(0, 0.45))
            p0[base + 3] *= np.exp(rng.normal(0, 0.45))

        # Randomize weights slightly
        weight_start = 4 * K_COMPONENTS
        p0[weight_start:weight_start + K_COMPONENTS] += rng.normal(
            0, 1.0, size=K_COMPONENTS
        )

        p0 = np.clip(p0, lower_bounds, upper_bounds)

    print(f"\nStarting fit {start + 1}/{N_STARTS}...")

    result = minimize(
        negative_log_likelihood,
        p0,
        args=(K_COMPONENTS,),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2500,
            "ftol": 1e-9,
        },
    )

    print("  success:", result.success)
    print("  nll:", result.fun)

    if best_result is None or result.fun < best_result.fun:
        best_result = result


print("\nBest fit:")
print("  success:", best_result.success)
print("  message:", best_result.message)
print("  nll:", best_result.fun)


# ==========================================================
# EXTRACT BEST FIT
# ==========================================================
P_components_full, P_best_full = component_distributions(
    best_result.x,
    K_COMPONENTS,
)

P_best_obs = P_best_full[observed_indices]
fit_counts = N_total * P_best_obs

component_counts = N_total * P_components_full[:, observed_indices]

comp_params, weights = unpack_params(best_result.x, K_COMPONENTS)


# ==========================================================
# SORT COMPONENTS BY MEAN MARK
# ==========================================================
component_mean_marks = []

for k in range(K_COMPONENTS):
    Pk = P_components_full[k]
    mean_k = np.sum(marks_model * Pk) / np.sum(Pk)
    component_mean_marks.append(mean_k)

component_mean_marks = np.array(component_mean_marks)
order = np.argsort(component_mean_marks)

component_counts = component_counts[order]
P_components_full = P_components_full[order]
component_mean_marks = component_mean_marks[order]

comp_params_sorted = [comp_params[i] for i in order]
weights_sorted = weights[order]


print("\nBest-fit effective sectors, sorted by mean score:")
for kk, old_index in enumerate(order):
    Q0, L, sigma_q, sigma_r = comp_params[old_index]
    w = weights[old_index]
    print(
        f"Sector {kk+1}: "
        f"weight={w:.6f}, "
        f"Q0={Q0:.6f}, "
        f"L={L:.6f}, "
        f"sigma_q={sigma_q:.6f}, "
        f"sigma_r={sigma_r:.6f}, "
        f"mean_mark={component_mean_marks[kk]:.3f}"
    )


# ==========================================================
# SAVE FIT DATA
# ==========================================================
out = np.column_stack([
    marks_data,
    counts_data,
    fit_counts,
    counts_data - fit_counts,
])

np.savetxt(
    OUTPUT_FIT_DATA,
    out,
    fmt=["%d", "%.10e", "%.10e", "%.10e"],
    header="mark data_count fit_count residual"
)

component_out = np.column_stack([
    marks_data,
    *[component_counts[k] for k in range(K_COMPONENTS)],
    fit_counts,
    counts_data,
])

header_cols = ["mark"]
header_cols += [f"component_{k+1}_count" for k in range(K_COMPONENTS)]
header_cols += ["total_fit_count", "data_count"]

np.savetxt(
    OUTPUT_COMPONENT_DATA,
    component_out,
    fmt="%.10e",
    header=" ".join(header_cols)
)

with open(OUTPUT_PARAM_DATA, "w") as f:
    f.write("# Best-fit parameters\n")
    f.write(f"# K_COMPONENTS = {K_COMPONENTS}\n")
    f.write(f"# N_total = {N_total:.0f}\n")
    f.write(f"# negative_log_likelihood = {best_result.fun:.12e}\n")
    f.write("# Sorted by mean score\n")
    f.write("# sector weight Q0 L sigma_q sigma_r mean_mark\n")

    for k in range(K_COMPONENTS):
        Q0, L, sigma_q, sigma_r = comp_params_sorted[k]
        w = weights_sorted[k]
        f.write(
            f"{k+1:d} {w:.12e} {Q0:.12e} {L:.12e} "
            f"{sigma_q:.12e} {sigma_r:.12e} "
            f"{component_mean_marks[k]:.12e}\n"
        )

print(f"\nSaved fit data to: {OUTPUT_FIT_DATA}")
print(f"Saved component data to: {OUTPUT_COMPONENT_DATA}")
print(f"Saved fit parameters to: {OUTPUT_PARAM_DATA}")


# ==========================================================
# STACKED COMPONENT PLOT
# ==========================================================
HIGH_CUTOFF = 620
XMIN = -5
XMAX = 725

# Smooth only for visualization
smooth_sigma_components = 4.0
smooth_sigma_data = 5.0
smooth_sigma_fit = 3.0

component_counts_smooth = np.array([
    gaussian_filter1d(component_counts[k], sigma=smooth_sigma_components)
    for k in range(K_COMPONENTS)
])

data_smooth = gaussian_filter1d(counts_data, sigma=smooth_sigma_data)
fit_smooth = gaussian_filter1d(fit_counts, sigma=smooth_sigma_fit)

fig, ax = plt.subplots(figsize=(10.4, 6.4))

# High-score region
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

# Stacked effective sectors
sector_labels = []
for k in range(K_COMPONENTS):
    Q0, L, sigma_q, sigma_r = comp_params_sorted[k]
    sector_labels.append(
        rf"sector {k+1}: $Q_0={Q0:.0f},\ \mathcal{{L}}={L:.2f}$"
    )

ax.stackplot(
    marks_data,
    component_counts_smooth,
    labels=sector_labels,
    colors=COLORS_COMPONENTS[:K_COMPONENTS],
    alpha=0.72,
    linewidth=0.7,
    edgecolor="0.25",
    zorder=1,
)

# Total fit line
ax.plot(
    marks_data,
    fit_smooth,
    color=COL_TOTAL_FIT,
    lw=2.8,
    label="total fit",
    zorder=5,
)

# Raw data and smoothed data envelope
ax.plot(
    marks_data,
    counts_data,
    color=COL_DATA_RAW,
    lw=0.65,
    alpha=0.45,
    label="raw data",
    zorder=3,
)

ax.plot(
    marks_data,
    data_smooth,
    color=COL_DATA_SMOOTH,
    lw=2.5,
    label="smoothed data",
    zorder=6,
)

# Text box
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

ax.text(
    HIGH_CUTOFF + 8,
    0.48 * np.max(counts_data),
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
ax.set_ylim(0, 1.08 * np.max(counts_data))

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which="both", top=True, right=True)

# Put legend in an old-style boxed block
handles, labels = ax.get_legend_handles_labels()

# Reorder legend: sectors first, then data/fit
ax.legend(
    handles,
    labels,
    loc="upper right",
    bbox_to_anchor=(0.7, 0.98),
    ncol=1,
    frameon=True,
    framealpha=0.96,
    borderpad=0.7,
)

fig.tight_layout()
fig.savefig(OUTPUT_STACKED_FIG)
fig.savefig(OUTPUT_STACKED_FIG_PDF)
plt.close(fig)

print(f"Saved stacked component plot to: {OUTPUT_STACKED_FIG}")
print(f"Saved stacked component plot to: {OUTPUT_STACKED_FIG_PDF}")


# ==========================================================
# TAIL / SURVIVAL PLOT
# ==========================================================
tail_data = np.cumsum(counts_data[::-1])[::-1]
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
    marks_data[::skip],
    tail_data_smooth[::skip],
    linestyle="None",
    marker="o",
    markersize=4.0,
    markerfacecolor="white",
    markeredgecolor="#005f99",
    markeredgewidth=1.2,
    alpha=0.90,
    label="data tail",
    zorder=4,
)

ax.semilogy(
    marks_data,
    tail_fit_smooth,
    color="#ff7f0e",
    lw=3.0,
    label="model tail",
    zorder=3,
)

idx_mc = np.argmin(np.abs(marks_data - HIGH_CUTOFF))
N_above_Mc = tail_data[idx_mc]
N_above_Mc_rounded = int(np.round(N_above_Mc / 10000.0) * 10000)

ax.annotate(
    f"{N_above_Mc_rounded:,} students\nabove {HIGH_CUTOFF} marks limit!",
    xy=(HIGH_CUTOFF, tail_data_smooth[idx_mc]),
    xytext=(315, 1.2e5),
    arrowprops=dict(
        arrowstyle="->",
        lw=1.8,
        color="#005f99",
        shrinkA=0,
        shrinkB=5,
    ),
    color="#005f99",
    fontsize=16,
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc="white",
        ec="#005f99",
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

print(f"Saved tail plot to: {OUTPUT_TAIL_FIG}")
print(f"Saved tail plot to: {OUTPUT_TAIL_FIG_PDF}")


# ==========================================================
# MODULO-5 DIAGNOSTIC
# ==========================================================
print("\nModulo-5 diagnostic:")
for mod in range(5):
    mask = (marks_data % 5) == mod
    print(
        f"  M mod 5 = {mod}: "
        f"data count = {counts_data[mask].sum():.0f}, "
        f"fit count = {fit_counts[mask].sum():.0f}"
    )

print("\nDone.")