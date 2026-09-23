#!/usr/bin/env python3
# Regenerate Graphical Abstract for #12 (MBS) with clean layout.
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.linewidth": 1.2,
    "text.color": "#222222",
    "axes.edgecolor": "#444444",
})

# ---------- data for right panel (real seasonal curve, k=0.5, theta=0) ----------
with open("/tmp/ga_curve.json") as fh:
    d = json.load(fh)
day = np.array(d["day"], float)
P = np.array(d["P"], float)
peak_i = int(np.argmax(P))
peak_day = int(day[peak_i]); peak_P = P[peak_i]
mask = P >= 0.05
win_lo = int(day[mask][0]); win_hi = int(day[mask][-1])

fig = plt.figure(figsize=(13.6, 7.4))
fig.subplots_adjust(left=0.04, right=0.96, top=0.81, bottom=0.06, wspace=0.10)

# ===== TITLE =====
fig.text(0.5, 0.945,
         "Why most imported Aedes-borne arbovirus cases fail to establish local transmission:\nthe role of offspring overdispersion",
         ha="center", va="top", fontsize=17, fontweight="bold", color="#1a1a1a", linespacing=1.25)
fig.text(0.5, 0.892, "Graphical Abstract", ha="center", va="top",
         fontsize=11.5, style="italic", color="#555555")

# ===== LEFT PANEL: overdispersed offspring collapse establishment =====
axL = fig.add_axes([0.045, 0.10, 0.44, 0.66])
axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")

axL.text(5, 9.55, "Overdispersed offspring collapse establishment",
         ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1a1a1a")

# imported index case (root)
root_x, root_y = 1.0, 5.0
axL.scatter([root_x], [root_y], s=900, color="#b30000", zorder=5, edgecolor="white", linewidth=1.5)
axL.text(root_x, root_y, "Imported\nindex case", ha="center", va="center",
         color="white", fontsize=9.5, fontweight="bold", zorder=6)

# offspring branch targets (highly dispersed: 1 huge + several tiny)
targets = [(4.0, 8.2, 1.0, "#b30000"), (4.0, 6.4, 0.12, "#bbbbbb"),
           (4.0, 4.4, 0.10, "#bbbbbb"), (4.0, 2.6, 0.09, "#bbbbbb"),
           (4.0, 1.2, 0.07, "#bbbbbb")]
for tx, ty, w, col in targets:
    axL.plot([root_x, tx], [root_y, ty], color="#888888", lw=1.6*w+0.6, zorder=2)
    if col == "#b30000":
        axL.scatter([tx], [ty], s=820, color="#b30000", zorder=5, edgecolor="white", linewidth=1.5)
        axL.text(tx, ty, "rare\nchain", ha="center", va="center", color="white",
                 fontsize=9, fontweight="bold", zorder=6)
    else:
        axL.scatter([tx], [ty], s=230, color="#cfcfcf", zorder=4, edgecolor="#999999", linewidth=0.8)

# second generation off the rare (red) chain -> establishment
g2 = [(7.4, 8.6, "#b30000"), (7.4, 7.6, "#b30000"), (7.4, 6.4, "#d98a8a")]
axL.plot([4.0, 7.4], [8.2, 8.6], color="#b30000", lw=2.0, zorder=3)
axL.plot([4.0, 7.4], [8.2, 7.6], color="#b30000", lw=2.0, zorder=3)
axL.plot([4.0, 7.4], [8.2, 6.4], color="#b30000", lw=2.0, zorder=3)
for gx, gy, col in g2:
    axL.scatter([gx], [gy], s=720 if col=="#b30000" else 260, color=col, zorder=5,
                edgecolor="white", linewidth=1.2)
axL.text(7.4, 9.15, "establishment\n(few chains)", ha="center", va="bottom",
         color="#b30000", fontsize=10.5, fontweight="bold", zorder=6)

# extinction label near grey dead-ends
axL.text(4.0, 0.55, "extinction (most chains)", ha="center", va="bottom",
         color="#777777", fontsize=10.5, fontweight="bold", zorder=6)

# negative-binomial k annotation
axL.text(5, 3.05, "Negative-binomial offspring, small k:\nmost imported cases seed 0 onward infections",
         ha="center", va="center", fontsize=10, color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fbf3d9", ec="#caa84a", lw=1.2))

# ===== RIGHT PANEL: seasonal receptive window & control =====
axR = fig.add_axes([0.55, 0.12, 0.40, 0.62])
axR.set_xlim(0, 365); axR.set_ylim(0, 0.62)
axR.set_xlabel("Day of year", fontsize=12)
axR.set_ylabel("Establishment probability", fontsize=12)
axR.tick_params(labelsize=10)
axR.set_title("Seasonal receptive window & vector control", fontsize=13.5,
              fontweight="bold", color="#1a1a1a", pad=10)

# shaded receptive window
axR.axvspan(win_lo, win_hi, color="#ffe28a", alpha=0.55, zorder=1)
axR.text((win_lo+win_hi)/2, 0.02,
         f"receptive window  (P \u2265 0.05)\ndays {win_lo}\u2013{win_hi}",
         ha="center", va="bottom", fontsize=9.5, color="#9a7a12", fontweight="bold", zorder=6)

# baseline (no control) curve
xr = np.linspace(0, 365, 400)
Pr = np.interp(xr, day, P)
axR.plot(xr, Pr, color="#2b2b2b", lw=2.4, zorder=3, label="no control")

# peak star — meaning made explicit: optimal import date = peak establishment probability
axR.scatter([peak_day], [peak_P], s=540, marker="*", color="#1a8a3c", zorder=6,
            edgecolor="white", linewidth=1.2)
axR.annotate(f"optimal import date\n(peak P, day {peak_day})",
             xy=(peak_day, peak_P),
             xytext=(peak_day-150, peak_P+0.06), fontsize=10, color="#1a8a3c", fontweight="bold",
             ha="center", va="bottom", zorder=7,
             arrowprops=dict(arrowstyle="-", color="#1a8a3c", lw=1.0))

# control curve (theta reduces R0 -> lower, narrower)
Pc = np.interp(xr, day, P) * 0.45
axR.plot(xr, Pc, color="#b30000", lw=2.2, ls="--", zorder=4, label="vector control (\u03b8 reduces R\u2080)")
# red downward arrow where the two curves are clearly separated (rising flank, ~day 110)
ax_arrow = 110
axR.annotate("", xy=(ax_arrow, np.interp(ax_arrow, day, P)*0.45),
             xytext=(ax_arrow, np.interp(ax_arrow, day, P)),
             arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=2.4), zorder=7)
axR.text(ax_arrow+8, (np.interp(ax_arrow, day, P)+np.interp(ax_arrow, day, P)*0.45)/2,
         "\u03b8 \u2193 R\u2080", color="#b30000", fontsize=9.5, fontweight="bold",
         ha="left", va="center", zorder=7)

axR.legend(loc="upper right", fontsize=9, framealpha=0.9)

# ===== shared footer note =====
fig.text(0.5, 0.018,
         "Offspring overdispersion pushes establishment probability below the Poisson benchmark; the effect compounds with seasonal receptive-window timing.",
         ha="center", va="bottom", fontsize=10.5, color="#444444", style="italic")

out_png = "."
out_pdf = "."
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
print("WROTE", out_png, out_pdf)
print("peak_day", peak_day, "win", win_lo, win_hi)
