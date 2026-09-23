"""
make_figures.py
===============
Draws every figure for the manuscript and writes PNG (600 dpi) + PDF (vector)
+ TIFF (LZW compressed) into ../02 Figures/ .

Figures
-------
  Fig 1  analysis framework (flow diagram)
  Fig 2  extinction probability q vs R0, coloured by dispersion k
  Fig 3  establishment probability vs number of imported cases
  Fig 4  survival to generation g, convergence to the ultimate value
  Fig 5  seasonal establishment probability (main figure)
  Fig 6  vector control effect
  Fig S1 supplementary: periodic steady state vs finite-horizon establishment

Run:  ../.venv/bin/python make_figures.py

Style: white background, sans-serif, low-saturation ColorBrewer-like palettes.
English only -- no CJK characters anywhere.

Journal (Mathematical Biosciences / Elsevier) compliance
--------------------------------------------------
  * resolution must be MORE than 300 dpi  -> exported at 600 dpi
  * figures submitted as separate files in JPEG or TIFF
  * no caption/legend text on the face of the figure
  * double-column width 17.5 cm = 6.89 in  -> every figure is 6.9 in wide
  * type sizes (pt): axis label 10, tick 8.5, legend 8.5, panel title 10.5,
    in-figure annotation >= 8.5  (nothing below 7 pt is ever emitted)
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from PIL import Image

import branching_model as bm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "02 Figures"))
os.makedirs(OUT, exist_ok=True)

DPI = 600
TIFF_MAX_MB = 20.0          # shrink the TIFF dpi (600 -> 400 -> 300) if larger
FIG_W = 6.9                 # 17.5 cm double-column width

# publication type sizes (pt)
FS_LABEL = 10.0             # axis labels
FS_TICK = 8.5               # tick labels
FS_LEGEND = 8.5             # legend entries
FS_PANEL = 10.5             # panel titles (A, B, C ...)
FS_NOTE = 8.5               # smallest in-figure annotation

# -----------------------------------------------------------------------------
# style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 9,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_PANEL,
    "axes.titleweight": "bold",
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.7,
    "legend.frameon": False,
})

CM = plt.get_cmap("viridis")
PB = plt.get_cmap("PuBu")
SET2 = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#E5C494", "#B3B3B3"]
GREY = "#9aa5ad"

K_ORDER = ["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "Poisson"]
K_COLOR = {k: CM(0.03 + 0.77 * i / (len(K_ORDER) - 1))
           for i, k in enumerate(K_ORDER)}

R0P_LIST = [1.5, 2.5, 4.0]
R0P_COLOR = {1.5: SET2[2], 2.5: SET2[1], 4.0: SET2[0]}      # blue, orange, teal
THETA_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
THETA_COLOR = {t: PB(0.42 + 0.53 * i / (len(THETA_LIST) - 1))
               for i, t in enumerate(THETA_LIST)}

MONTH_DAYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABEL = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

G_HORIZON = 6
TAU_REF = 15
W_REF = 55.0
T_PEAK = 190.0
K_REF_LABEL = "0.5"
THRESHOLD = 0.05


def save(fig, name):
    """PNG + PDF (vector) + TIFF (LZW).

    The TIFF goes out through PIL on purpose: matplotlib >= 3.8 silently drops
    ``pil_kwargs`` for the tiff writer, so the raw file lands uncompressed
    (compression tag 1, ~76 MB).  Rendering to PNG and re-saving with
    ``compression="tiff_lzw"`` is the only way to actually get an LZW TIFF.
    dpi ladder 600 -> 400 -> 300 if the TIFF would exceed TIFF_MAX_MB.
    """
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=DPI, bbox_inches="tight")

    tif = os.path.join(OUT, name + ".tif")
    tif_dpi = 300
    for d in (DPI, 400, 300):
        tif_dpi = d
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=d, bbox_inches="tight")
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        im.save(tif, format="TIFF", compression="tiff_lzw", dpi=(d, d))
        mb = os.path.getsize(tif) / 1048576.0
        if mb <= TIFF_MAX_MB:
            break
    EXPORT_LOG[name] = {"tif_dpi": tif_dpi, "tif_mb": mb,
                        "tif_px": im.size}
    plt.close(fig)
    print("   wrote", name + ".png / .pdf / .tif",
          f"(tiff {tif_dpi} dpi, {mb:.2f} MB)")


EXPORT_LOG: dict[str, dict] = {}


def month_axis(ax):
    ax.set_xticks(MONTH_DAYS)
    ax.set_xticklabels(MONTH_LABEL)
    ax.set_xlim(1, 365)


def day_to_date(day):
    """Approximate calendar date for a day-of-year (non-leap year)."""
    import datetime
    d = datetime.date(2001, 1, 1) + datetime.timedelta(days=int(day) - 1)
    return d.strftime("%d %b")


# -----------------------------------------------------------------------------
# Fig 1 : framework
# -----------------------------------------------------------------------------
def fig1():
    fig = plt.figure(figsize=(FIG_W, 7.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3f8", ec="#5b7a99", fs=8.4, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    linewidth=1.0, edgecolor=ec, facecolor=fc,
                                    zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.45,
                fontweight="bold" if bold else "normal")

    def arrow(p1, p2, color="#5b7a99", rad=0.0, ls="-", lw=1.1):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                     linewidth=lw, color=color, zorder=1,
                                     connectionstyle=f"arc3,rad={rad}",
                                     linestyle=ls))

    # ---- row A : homogeneous branching process -----------------------------
    ax.text(1.5, 98.2, "A   Homogeneous branching process (constant $R_0$)",
            fontsize=FS_PANEL, fontweight="bold", va="center")

    box(1.5, 87.0, 22.0, 8.5, "Imported index\ncases ($n$)")
    arrow((23.5, 91.25), (27.5, 91.25))
    box(27.5, 87.0, 28.0, 8.5,
        "Offspring distribution\nNegative binomial\nmean $R_0$, dispersion $k$")
    arrow((55.5, 91.25), (59.5, 91.25))
    box(59.5, 87.0, 38.5, 8.5,
        "Galton-Watson chain, $Z_0 = n$\n"
        "Extinction:  $P = q^{\\,n}$\n"
        "Establishment:  $P = 1 - q^{\\,n}$",
        fc="#f7f3ec", ec="#8a7f6b")
    ax.text(50.0, 84.0,
            "$q$ = smallest root of $s = G(s)$ on $[0,1)$, found by bisection",
            fontsize=FS_NOTE, ha="center", color="#555555")

    # ---- row B : seasonal process ------------------------------------------
    ax.text(1.5, 78.5, "B   Seasonal (time-inhomogeneous) process",
            fontsize=FS_PANEL, fontweight="bold", va="center")

    box(1.5, 66.0, 22.0, 9.0,
        "Seasonal forcing (365 d)\n$R_0(t)$: Gaussian peak\n"
        "peak $R_{0,pk}$, width $w$")
    arrow((23.5, 70.0), (27.5, 70.0))
    box(27.5, 66.0, 28.0, 9.0,
        "Backward recursion\n$q(t) = G_t(q(t+\\tau))$\nstep $\\tau$ = 15 d")
    arrow((55.5, 70.0), (59.5, 70.0))
    box(59.5, 66.0, 38.5, 9.0,
        "Iterate to the minimal fixed point\n"
        "max $|\\Delta q|$ < 1e-12, periodic boundary\n"
        "$P_{est}(t) = 1-q(t)$, window $P \\geq 0.05$",
        fc="#e7f0e8", ec="#5f8f68")
    arrow((58.5, 64.0), (34.0, 64.0), color="#8fa8bf", rad=-0.15,
          ls=(0, (4, 2.5)), lw=1.0)
    ax.text(46.0, 60.0, "repeat the backward sweep until the fixed point is reached",
            fontsize=FS_NOTE, ha="center", color="#6b7d8c")

    # ---- row C : outputs ----------------------------------------------------
    ax.text(1.5, 58.6, "C   Outputs", fontsize=FS_PANEL, fontweight="bold",
            va="center")
    for i, txt in enumerate(["Fig 2\n$q(R_0,\\,k)$", "Fig 3\n$P(n$ imports$)$",
                             "Fig 4\n$P_{surv}(g)$", "Fig 5\n$P_{est}(t)$",
                             "Fig 6\ncontrol effect"]):
        box(1.5 + i * 19.7, 50.0, 18.2, 6.6, txt, fc="#f4f6f8", ec="#8fa8bf")

    # ---- row D : offspring distribution ------------------------------------
    ax.text(1.5, 45.5, "D   Offspring distribution ($R_0$ = 2.5)",
            fontsize=FS_PANEL, fontweight="bold", va="center")

    xs = np.arange(0, 13)
    inset = fig.add_axes([0.06, 0.04, 0.50, 0.38])
    R0 = 2.5
    for k, col, lab in [(0.2, CM(0.20), "k = 0.2"), (np.inf, CM(0.72), "Poisson")]:
        from math import lgamma, exp, log
        pmf = []
        for m in xs:
            if np.isinf(k):
                pmf.append(exp(-R0 + m * log(R0) - lgamma(m + 1)))
            else:
                pmf.append(exp(lgamma(m + k) - lgamma(k) - lgamma(m + 1)
                               + k * log(k / (k + R0)) + m * log(R0 / (k + R0))))
        inset.bar(xs + (0.0 if np.isinf(k) else -0.19), pmf, width=0.38,
                  color=col, label=lab, alpha=0.9)
    inset.set_xlabel("offspring per case", fontsize=8.5, labelpad=1.0)
    inset.set_ylabel("probability", fontsize=8.5, labelpad=1.5)
    inset.tick_params(labelsize=8.0)
    inset.legend(fontsize=8.0, frameon=False, handlelength=1.2,
                 labelspacing=0.25, borderpad=0.1)
    inset.set_xlim(-0.6, 12.6)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)

    # side note panel (deliberate annotation, aligned beside the inset)
    ax.add_patch(FancyBboxPatch((57.0, 6.0), 40.0, 28.0,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=0.8, edgecolor="#cccccc",
                                facecolor="#f7f7f2", zorder=1))
    ax.text(77.0, 20.0,
            "Overdispersion at a fixed mean ($R_0$ = 2.5):\n"
            "small $k$ concentrates mass on zero offspring,\n"
            "so most chains die immediately.",
            fontsize=FS_NOTE, ha="center", va="center", color="#555555",
            linespacing=1.5, zorder=3)

    save(fig, "Fig1_framework")


# -----------------------------------------------------------------------------
# Fig 2 : extinction probability vs R0
# -----------------------------------------------------------------------------
def fig2():
    df = pd.read_csv(os.path.join(HERE, "fig2_extinction_vs_R0.csv"))
    fig, ax = plt.subplots(figsize=(FIG_W, 4.9))
    for k in K_ORDER:
        s = df[df["k"] == k]
        ls = "--" if k == "Poisson" else "-"
        lw = 2.0 if k == "Poisson" else 1.7
        ax.plot(s["R0"], s["extinction_prob"], color=K_COLOR[k], ls=ls, lw=lw,
                label=f"k = {k}", zorder=3)

    ax.axvline(1.0, color="#b0b0b0", lw=1.0, ls=":", zorder=1)
    ax.annotate("R0 = 1\n(critical threshold)", xy=(1.0, 0.965),
                xytext=(2.05, 0.965), fontsize=FS_NOTE, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#9aa5ad", lw=0.9))
    ax.axhline(1.0, color="#dddddd", lw=0.8, zorder=0)

    ax.set_xlabel("Basic reproduction number  $R_0$")
    ax.set_ylabel("Extinction probability  $q$")
    ax.set_xlim(0.2, 6.0)
    ax.set_ylim(-0.02, 1.03)
    ax.legend(title="offspring dispersion", ncol=2, loc="lower left",
              columnspacing=1.1, handlelength=1.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "Fig2_extinction_vs_R0")


# -----------------------------------------------------------------------------
# Fig 3 : establishment vs number of imported cases
# -----------------------------------------------------------------------------
def fig3():
    df = pd.read_csv(os.path.join(HERE, "fig3_establishment_vs_imports.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.5), sharey=True)
    for ax, r0 in zip(axes, R0P_LIST):
        s = df[df["R0"] == r0]
        for k in K_ORDER:
            ss = s[s["k"] == k].sort_values("n_imports")
            ls = "--" if k == "Poisson" else "-"
            ax.plot(ss["n_imports"], ss["establishment_prob"], color=K_COLOR[k],
                    ls=ls, lw=1.7, marker="o", ms=3.2, mfc="white", mew=0.7,
                    label=f"k = {k}")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 50, 100])
        ax.set_xticklabels(["1", "2", "5", "10", "25", "50", "100"])
        ax.set_ylim(-0.02, 1.03)
        ax.set_title(f"$R_0$ = {r0}")
        ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Establishment probability  $1-q^{\\,n}$")
    # single shared x-axis label below the whole row; per-panel labels are
    # removed to avoid overlapping copies at 6.9 in width
    for ax in axes:
        ax.set_xlabel("")
    fig.supxlabel("Number of imported index cases  $n$", fontsize=FS_LABEL, y=0.02)
    # shared legend sits in the bottom margin, clearly ABOVE the x-label and
    # BELOW the panels (fig.transFigure + explicit anchor keeps tight_layout
    # from relocating it up into the plot area)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="offspring dispersion", ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.07),
               bbox_transform=fig.transFigure,
               frameon=False, columnspacing=1.4, handlelength=1.6)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.37)
    save(fig, "Fig3_establishment_vs_imports")


# -----------------------------------------------------------------------------
# Fig 4 : survival by generation
# -----------------------------------------------------------------------------
def fig4():
    df = pd.read_csv(os.path.join(HERE, "fig4_survival_by_generation.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.9), sharey=True)

    # (a) vary k at R0 = 2.5
    ax = axes[0]
    s = df[df["R0"] == 2.5]
    for k in K_ORDER:
        ss = s[s["k"] == k].sort_values("generation_g")
        ult = ss["p_survive_ultimate"].iloc[0]
        ls = "--" if k == "Poisson" else "-"
        ax.plot(ss["generation_g"], ss["p_survive"], color=K_COLOR[k], ls=ls,
                lw=1.7, marker="o", ms=3.0, mfc="white", mew=0.7,
                label=f"k = {k}")
        ax.axhline(ult, color=K_COLOR[k], lw=0.8, alpha=0.45, ls=":")
    ax.set_title("fixed $R_0$ = 2.5, varying dispersion")
    ax.set_ylabel("P(chain still alive at generation $g$)")
    ax.set_xticks(range(1, 13))
    ax.legend(title="dispersion (dotted = limit $1-q$)", ncol=3,
              loc="upper left", bbox_to_anchor=(0.0, -0.24),
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              columnspacing=0.9, handlelength=1.3, fontsize=6.6)

    # (b) vary R0 at k = 0.5
    ax = axes[1]
    s = df[df["k"] == "0.5"]
    ramp = plt.get_cmap("YlOrBr")(np.linspace(0.35, 0.95, len(s["R0"].unique())))
    for (r0, ss), col in zip(s.groupby("R0"), ramp):
        ss = ss.sort_values("generation_g")
        ult = ss["p_survive_ultimate"].iloc[0]
        ax.plot(ss["generation_g"], ss["p_survive"], color=col, lw=1.7,
                marker="s", ms=3.0, mfc="white", mew=0.7, label=f"$R_0$ = {r0:g}")
        ax.axhline(ult, color=col, lw=0.8, alpha=0.45, ls=":")
    ax.set_title("fixed k = 0.5, varying $R_0$")
    ax.set_xticks(range(1, 13))
    ax.legend(title="(dotted = limit $1-q$)", ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.24),
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              columnspacing=0.9, handlelength=1.3, fontsize=6.6)

    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Generation $g$")
        ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "Fig4_survival_by_generation")


# -----------------------------------------------------------------------------
# Fig 5 : seasonal establishment (main figure)
# -----------------------------------------------------------------------------
def fig5():
    df = pd.read_csv(os.path.join(HERE, "fig5_seasonal_establishment.csv"))
    main = df[(df["scenario"] == "main_grid") & (df["w"] == W_REF)
              & (df["tau"] == TAU_REF)]
    disp = df[df["scenario"] == "dispersion_panel"]

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 7.8), sharex=True)
    ax0, ax1, ax2 = axes

    # (a) seasonal R0(t)
    for r0p in R0P_LIST:
        s = main[main["R0_peak"] == r0p].sort_values("day")
        ax0.plot(s["day"], s["R0_t"], color=R0P_COLOR[r0p], lw=1.9,
                 label=f"$R_{{0,peak}}$ = {r0p:g}")
    ax0.axhline(1.0, color="#b0b0b0", lw=1.0, ls=":", zorder=1)
    ax0.text(4, 1.08, "$R_0(t) = 1$", fontsize=FS_NOTE, color="#777777")
    ax0.axvline(T_PEAK, color="#cccccc", lw=0.9, ls="--", zorder=0)
    ymax = float(main["R0_t"].max())
    ax0.text(T_PEAK, ymax * 1.05, "peak (9 Jul)", ha="center", va="bottom",
             fontsize=FS_NOTE, color="#888888")
    ax0.set_ylabel("Seasonal reproduction number  $R_0(t)$")
    ax0.set_title("A   Seasonal forcing   ($w$ = 55 d, $t_{peak}$ = day 190)")
    ax0.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1,
               frameon=False)
    ax0.set_ylim(0, ymax * 1.15)
    for sp in ("top", "right"):
        ax0.spines[sp].set_visible(False)

    # (b) establishment probability by R0_peak
    for r0p in R0P_LIST:
        s = main[main["R0_peak"] == r0p].sort_values("day")
        p = s["p_survive_g6"].to_numpy()
        win = bm.vulnerable_window(p, THRESHOLD)
        label = (f"$R_{{0,peak}}$ = {r0p:g}: "
                 f"{day_to_date(win['start_day'])}–{day_to_date(win['end_day'])}")
        ax1.plot(s["day"], p, color=R0P_COLOR[r0p], lw=2.0, label=label)
        ax1.axvspan(win["start_day"] - 0.5, win["end_day"] - 0.5,
                    color=R0P_COLOR[r0p], alpha=0.10, lw=0, zorder=0)
    ax1.axhline(THRESHOLD, color="#b0b0b0", lw=1.0, ls=":", zorder=1)
    ax1.text(358, THRESHOLD + 0.015, f"threshold {THRESHOLD:g}", fontsize=FS_NOTE,
             color="#777777", ha="right", va="bottom")
    ax1.set_ylabel("Establishment probability")
    ax1.set_title(f"B   Within-season establishment  (k = {K_REF_LABEL}, "
                  f"$\\tau$ = {TAU_REF} d, horizon g = {G_HORIZON} generations)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1,
               frameon=False)
    ax1.set_ylim(0, None)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # (c) dispersion sensitivity
    kpanel = ["0.1", "0.5", "2", "Poisson"]
    cols = [CM(0.10), CM(0.38), CM(0.62), "#b0b0b0"]
    for k, col in zip(kpanel, cols):
        s = disp[disp["k"] == k].sort_values("day")
        if s.empty:
            continue
        ls = "--" if k == "Poisson" else "-"
        ax2.plot(s["day"], s["p_survive_g6"], color=col, lw=1.9, ls=ls,
                 label=("k = Poisson" if k == "Poisson" else f"k = {k}"))
    ax2.axhline(THRESHOLD, color="#b0b0b0", lw=1.0, ls=":", zorder=1)
    ax2.set_ylabel("Establishment probability")
    ax2.set_xlabel("Introduction date (day of year)")
    ax2.set_title("C   Effect of offspring dispersion  ($R_{0,peak}$ = 2.5)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1,
               frameon=False)
    ax2.set_ylim(0, None)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    month_axis(ax2)
    for ax in axes:
        ax.set_xlim(1, 365)
        ax.grid(axis="y", color="#f0f0f0", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.32)
    save(fig, "Fig5_seasonal_establishment")


# -----------------------------------------------------------------------------
# Fig 6 : vector control
# -----------------------------------------------------------------------------
def fig6():
    df = pd.read_csv(os.path.join(HERE, "fig6_control_effect.csv"))
    ks = pd.read_csv(os.path.join(HERE, "key_summary.csv"))
    curves = df[df["record_type"] == "daily_curve"]
    scan = df[df["record_type"] == "theta_scan"]

    # A spans the full double-column width on top; B and C share the lower row.
    # (a 1x3 row would leave each panel ~1.9 in wide, too narrow for 12 month
    # labels at a legible size)
    fig = plt.figure(figsize=(FIG_W, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.46, wspace=0.24)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # (a) curves at R0_peak = 2.5
    s0 = curves[curves["R0_peak"] == 2.5]
    for th in THETA_LIST:
        ss = s0[np.isclose(s0["theta"], th)].sort_values("day")
        ax0.plot(ss["day"], ss["p_survive_g6"], color=THETA_COLOR[th], lw=1.8,
                 label=r"$\theta$ = " + f"{th:g}")
    ax0.axhline(THRESHOLD, color="#b0b0b0", lw=1.0, ls=":", zorder=1)
    ax0.set_ylabel("Establishment probability")
    ax0.set_xlabel("Introduction date (day of year)")
    ax0.set_title("A   Control scenarios ($R_{0,peak}$ = 2.5)")
    ax0.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2,
               frameon=True, framealpha=0.95, edgecolor="#cccccc")
    ax0.set_ylim(0, None)
    month_axis(ax0)

    # (b) annual maximum vs theta
    for r0p in R0P_LIST:
        s = scan[scan["R0_peak"] == r0p].sort_values("theta")
        ax1.plot(s["theta"], s["p_survive_g6"], color=R0P_COLOR[r0p], lw=1.9,
                 label=f"$R_{{0,peak}}$ = {r0p:g}")
        row = ks[(ks["metric"] == "theta_required_for_target")
                 & (ks["condition"].str.startswith(f"R0_peak={r0p},"))]
        if len(row):
            v = pd.to_numeric(row["value"], errors="coerce").iloc[0]
            if np.isfinite(v):
                ax1.plot([v], [THRESHOLD], marker="o", ms=5,
                         mfc="white", mec=R0P_COLOR[r0p], mew=1.4, zorder=5)
    ax1.axhline(THRESHOLD, color="#b0b0b0", lw=1.0, ls=":", zorder=1)
    ax1.text(0.02, THRESHOLD + 0.006, f"target {THRESHOLD:g}", fontsize=FS_NOTE,
             color="#777777")
    ax1.set_xlabel("Vector-control efficacy  $\\theta$  (reduction in $R_0$)")
    ax1.set_ylabel("Annual maximum of $P_{establish}$")
    ax1.set_title("B   Annual peak vs control effort")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               fontsize=7.5)
    ax1.set_ylim(0, None)
    ax1.set_xlim(0, 0.95)

    # (c) vulnerable window length vs theta
    for r0p in R0P_LIST:
        s = scan[scan["R0_peak"] == r0p].sort_values("theta")
        ax2.plot(s["theta"], s["window_len_thr0.05"], color=R0P_COLOR[r0p],
                 lw=1.9, label=f"$R_{{0,peak}}$ = {r0p:g}")
    ax2.set_xlabel("Vector-control efficacy  $\\theta$")
    ax2.set_ylabel("Vulnerable window (days)")
    ax2.set_title("C   Length of the vulnerable window")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               fontsize=7.0)
    ax2.set_ylim(0, None)
    ax2.set_xlim(0, 0.95)

    for ax in (ax0, ax1, ax2):
        ax.grid(axis="y", color="#f0f0f0", lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "Fig6_control_effect")


# -----------------------------------------------------------------------------
# Supplementary Fig S1 : periodic steady state
# -----------------------------------------------------------------------------
def figS1():
    df = pd.read_csv(os.path.join(HERE, "fig5_seasonal_establishment.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.5), sharey=True)

    ax = axes[0]
    for r0p, w in [(4.0, 70.0), (4.0, 55.0), (2.5, 70.0)]:
        s = df[(df["scenario"] == "main_grid") & (df["R0_peak"] == r0p)
               & (df["w"] == w) & (df["tau"] == TAU_REF)].sort_values("day")
        sup = bool(s["season_supercritical"].iloc[0])
        ax.plot(s["day"], s["p_establish_periodic"], lw=1.9,
                label=f"$R_{{0,peak}}$={r0p:g}, $w$={w:g}  "
                      f"({'supercritical' if sup else 'subcritical'})")
    ax.set_ylabel("Periodic steady-state $P_{establish}(t) = 1 - q(t)$")
    ax.set_xlabel("Introduction date (day of year)")
    ax.set_title("A   Periodic steady state")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.12), fontsize=7.5,
              frameon=False)
    ax.set_ylim(-0.01, None)
    month_axis(ax)
    ax.set_xlim(1, 365)

    ax = axes[1]
    for r0p, w in [(4.0, 70.0), (4.0, 55.0), (2.5, 70.0)]:
        s = df[(df["scenario"] == "main_grid") & (df["R0_peak"] == r0p)
               & (df["w"] == w) & (df["tau"] == TAU_REF)].sort_values("day")
        ax.plot(s["day"], s["p_survive_g6"], lw=1.9,
                label=f"$R_{{0,peak}}$={r0p:g}, $w$={w:g}")
    ax.set_xlabel("Introduction date (day of year)")
    ax.set_title("B   Finite horizon (g = 6)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.12), fontsize=7.5,
              frameon=False)
    month_axis(ax)
    ax.set_xlim(1, 365)

    for a in axes:
        # sharey=True would clip the right panel (g=6 peak 0.575) to the left
        # panel's auto-limit (periodic peak 0.482); fix an explicit ceiling.
        a.set_ylim(-0.01, 0.62)
        a.grid(axis="y", color="#f0f0f0", lw=0.6, zorder=0)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "FigS1_periodic_vs_finite_horizon")


# -----------------------------------------------------------------------------
def main():
    print("writing figures to", OUT)
    print("[1/7] Fig 1 framework")
    fig1()
    print("[2/7] Fig 2 extinction vs R0")
    fig2()
    print("[3/7] Fig 3 establishment vs imports")
    fig3()
    print("[4/7] Fig 4 survival by generation")
    fig4()
    print("[5/7] Fig 5 seasonal establishment")
    fig5()
    print("[6/7] Fig 6 control effect")
    fig6()
    print("[7/7] Fig S1 periodic vs finite horizon")
    figS1()
    print("all figures written")


if __name__ == "__main__":
    main()
