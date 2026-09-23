"""
run_analysis.py
===============
Runs the full parameter grid for the imported-case establishment analysis and
writes every numerical result to CSV.

Outputs (all in this directory)
-------------------------------
  fig2_extinction_vs_R0.csv
  fig3_establishment_vs_imports.csv
  fig4_survival_by_generation.csv
  fig5_seasonal_establishment.csv
  fig6_control_effect.csv
  key_summary.csv

Run:  ../.venv/bin/python run_analysis.py

Conventions
-----------
  * Day-of-year is 1-based (1 = 1 January).
  * k = inf is the Poisson limit; it is stored as the string "inf" in the `k`
    column and labelled "Poisson" in `k_label`.
  * "Seasonal establishment" reported here is the FINITE-HORIZON quantity
        P_est(t) = P_survive(t, g) = P(chain still alive g generations after an
        introduction on day t),
    with G_HORIZON = 6 generations (~90 days at tau = 15 d).  The periodic
    steady-state quantity 1 - q(t) is computed as well and stored in the column
    `p_establish_periodic`; see README.md for why the two differ.

English only.  No external data, no network.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import branching_model as bm

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
DAYS = np.arange(1, 366)
PERIOD = 365

R0_GRID = np.linspace(0.2, 6.0, 400)                     # fine grid for Fig 2
K_LIST = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
K_LABEL = {np.inf: "Poisson"}

N_IMPORTS = [1, 2, 5, 10, 25, 50, 100]
R0_IMPORT_PANELS = [1.5, 2.5, 4.0]                       # Fig 3 panels
FIG3_REFERENCE_R0 = 2.5

FIG4_R0 = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
FIG4_GMAX = 12

T_PEAK = 190.0
W_REF = 55.0
TAU_REF = 15
R0_PEAK_LIST = [1.5, 2.5, 4.0]
W_LIST = [40.0, 55.0, 70.0]
TAU_LIST = [10, 15, 20]
K_REF = 0.5                       # reference dispersion for the seasonal figures
K_SEASONAL_PANEL = [0.1, 0.5, 2.0, np.inf]

G_HORIZON = 6                     # headline seasonal horizon (generations)
G_SET = [2, 4, 6, 8, 12]          # horizons written to the seasonal CSV

THETA_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
THETA_SCAN = np.linspace(0.0, 0.95, 96)
WINDOW_THRESHOLDS = [0.05, 0.10]
TARGET_P = 0.05                   # vector-control target for the annual maximum


def k_label(k):
    return K_LABEL.get(k, f"{k:g}")


def seasonal_block(R0_peak, w, tau, k, theta=0.0):
    """Return (R0_t, dict of finite-horizon survival arrays, periodic P)."""
    R0_t = bm.seasonal_R0(DAYS, R0_peak, w, T_PEAK) * (1.0 - theta)
    prof = R0_t[None, :]
    surv = {g: bm.seasonal_survival(prof, k, tau, g)[0] for g in G_SET}
    q_per = bm.periodic_extinction(prof, k, tau)[0]
    return R0_t, surv, 1.0 - q_per


# -----------------------------------------------------------------------------
# Fig 2 : extinction probability vs R0
# -----------------------------------------------------------------------------
def make_fig2():
    rows = []
    for k in K_LIST:
        q = bm.extinction_prob(R0_GRID, k)
        for r0, qq in zip(R0_GRID, q):
            rows.append({"R0": float(r0), "k": k_label(k), "k_label": k_label(k),
                         "extinction_prob": float(qq),
                         "establishment_n1": float(1.0 - qq)})
    df = pd.DataFrame(rows)
    df.to_csv("fig2_extinction_vs_R0.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Fig 3 : establishment probability vs number of imported cases
# -----------------------------------------------------------------------------
def make_fig3():
    rows = []
    for r0 in R0_IMPORT_PANELS:
        for k in K_LIST:
            q = float(bm.extinction_prob(np.array([r0]), k)[0])
            for n in N_IMPORTS:
                rows.append({"R0": float(r0), "k": k_label(k), "k_label": k_label(k),
                             "n_imports": int(n),
                             "extinction_prob_q": q,
                             "establishment_prob": float(1.0 - q ** n)})
    df = pd.DataFrame(rows)
    df.to_csv("fig3_establishment_vs_imports.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Fig 4 : survival to generation g
# -----------------------------------------------------------------------------
def make_fig4():
    rows = []
    for r0 in FIG4_R0:
        for k in K_LIST:
            surv = bm.survival_by_generation(np.array([r0]), k, FIG4_GMAX)[:, 0]
            ult = float(1.0 - bm.extinction_prob(np.array([r0]), k)[0])
            for gi, g in enumerate(range(1, FIG4_GMAX + 1)):
                rows.append({"R0": float(r0), "k": k_label(k), "k_label": k_label(k),
                             "generation_g": int(g),
                             "p_survive": float(surv[gi]),
                             "p_survive_ultimate": ult,
                             "frac_of_ultimate": float(surv[gi] / ult) if ult > 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv("fig4_survival_by_generation.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Fig 5 : seasonal establishment
# -----------------------------------------------------------------------------
def make_fig5():
    rows = []
    scen = []

    def add(name, R0_peak, w, tau, k, R0_t, surv, p_per):
        logm = bm.criticality_logM(R0_t[None, :], tau)[0]
        sup = bool((logm > 0).any())
        rows.extend(
            {"scenario": name, "R0_peak": R0_peak, "w": w, "tau": tau,
             "k": k_label(k), "k_label": k_label(k), "theta": 0.0,
             "day": int(d), "R0_t": float(R0_t[i]),
             **{f"p_survive_g{g}": float(surv[g][i]) for g in G_SET},
             "p_establish_periodic": float(p_per[i]),
             "log_cycle_mean_product": float(logm.max()),
             "season_supercritical": sup}
            for i, d in enumerate(DAYS))

    # (a) main grid: R0_peak x w x tau at the reference dispersion
    for r0p in R0_PEAK_LIST:
        for w in W_LIST:
            for tau in TAU_LIST:
                R0_t, surv, p_per = seasonal_block(r0p, w, tau, K_REF)
                add("main_grid", r0p, w, tau, K_REF, R0_t, surv, p_per)
                scen.append(("main_grid", r0p, w, tau, K_REF))

    # (b) dispersion panel at the reference season
    for k in K_SEASONAL_PANEL:
        R0_t, surv, p_per = seasonal_block(2.5, W_REF, TAU_REF, k)
        add("dispersion_panel", 2.5, W_REF, TAU_REF, k, R0_t, surv, p_per)

    df = pd.DataFrame(rows)
    df.to_csv("fig5_seasonal_establishment.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Fig 6 : vector control
# -----------------------------------------------------------------------------
def make_fig6():
    rows = []

    def annual_max(r0p, w, tau, k, theta):
        R0_t = bm.seasonal_R0(DAYS, r0p, w, T_PEAK) * (1.0 - theta)
        s = bm.seasonal_survival(R0_t[None, :], k, tau, G_HORIZON)[0]
        return R0_t, s

    # (a) daily curves for the six control levels
    for r0p in R0_PEAK_LIST:
        for theta in THETA_LIST:
            R0_t, s = annual_max(r0p, W_REF, TAU_REF, K_REF, theta)
            for i, d in enumerate(DAYS):
                rows.append({"record_type": "daily_curve", "R0_peak": r0p,
                             "w": W_REF, "tau": TAU_REF, "k": k_label(K_REF),
                             "theta": float(theta), "day": int(d),
                             "R0_t": float(R0_t[i]),
                             "p_survive_g6": float(s[i])})

    # (b) theta scan: annual maximum and window length
    for r0p in R0_PEAK_LIST:
        for theta in THETA_SCAN:
            R0_t, s = annual_max(r0p, W_REF, TAU_REF, K_REF, theta)
            rec = {"record_type": "theta_scan", "R0_peak": r0p, "w": W_REF,
                   "tau": TAU_REF, "k": k_label(K_REF), "theta": float(theta),
                   "day": "", "R0_t": float(R0_t.max()),
                   "p_survive_g6": float(s.max())}
            for thr in WINDOW_THRESHOLDS:
                win = bm.vulnerable_window(s, thr)
                rec[f"window_start_thr{thr:g}"] = win["start_day"]
                rec[f"window_end_thr{thr:g}"] = win["end_day"]
                rec[f"window_len_thr{thr:g}"] = win["length"]
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv("fig6_control_effect.csv", index=False)
    return df


def theta_star(r0p, target=TARGET_P):
    """Control effort theta needed to push the annual maximum below `target`."""
    def f(theta):
        R0_t = bm.seasonal_R0(DAYS, r0p, W_REF, T_PEAK) * (1.0 - theta)
        s = bm.seasonal_survival(R0_t[None, :], K_REF, TAU_REF, G_HORIZON)[0]
        return float(s.max()) - target

    lo, hi = 0.0, 0.95
    flo, fhi = f(lo), f(hi)
    if flo <= 0.0:
        return 0.0, f(0.0) + target, "already below target at theta = 0"
    if fhi > 0.0:
        return np.nan, f(hi) + target, "not reachable below theta = 0.95"
    th = brentq(f, lo, hi, xtol=1e-8, rtol=1e-12)
    return float(th), float(f(th) + target), "brentq root"


# -----------------------------------------------------------------------------
# key summary
# -----------------------------------------------------------------------------
def make_key_summary(fig5, fig6):
    out = []

    def add(metric, value, unit, condition, note=""):
        out.append({"metric": metric, "value": value, "unit": unit,
                    "condition": condition, "note": note})

    # --- homogeneous extinction / establishment ------------------------------
    for k in [0.1, 0.5, 1.0, 2.0, np.inf]:
        for r0 in [1.5, 2.5, 4.0]:
            q = float(bm.extinction_prob(np.array([r0]), k)[0])
            add("extinction_probability_q", round(q, 6), "probability",
                f"R0={r0}, k={k_label(k)}, n=1",
                "probability one imported case fails to ignite transmission")
            add("establishment_probability_n1", round(1.0 - q, 6), "probability",
                f"R0={r0}, k={k_label(k)}, n=1", "P_establish = 1 - q")

    # --- effect of the number of imported cases ------------------------------
    for k in [0.1, 0.5, np.inf]:
        q = float(bm.extinction_prob(np.array([FIG3_REFERENCE_R0]), k)[0])
        for n in [1, 10, 100]:
            add("establishment_probability_n", round(1.0 - q ** n, 6), "probability",
                f"R0={FIG3_REFERENCE_R0}, k={k_label(k)}, n={n}",
                "P_establish = 1 - q**n")

    # --- overdispersion effect -----------------------------------------------
    q_pois = float(bm.extinction_prob(np.array([2.5]), np.inf)[0])
    q_k01 = float(bm.extinction_prob(np.array([2.5]), 0.1)[0])
    add("overdispersion_risk_ratio", round((1 - q_k01) / (1 - q_pois), 6), "ratio",
        "R0=2.5, k=0.1 vs Poisson",
        "establishment probability relative to the Poisson benchmark")
    add("overdispersion_absolute_gap", round((1 - q_pois) - (1 - q_k01), 6),
        "probability", "R0=2.5, k=0.1 vs Poisson",
        "Poisson minus k=0.1 establishment probability")

    # --- how many imported cases before establishment becomes likely ---------
    for k in [0.1, 0.5, 1.0, np.inf]:
        for r0 in [1.5, 2.5, 4.0]:
            q = float(bm.extinction_prob(np.array([r0]), k)[0])
            add("expected_imports_per_establishment",
                round(float(1.0 / (1.0 - q)), 4), "cases",
                f"R0={r0}, k={k_label(k)}",
                "1 / (1 - q): mean number of imported cases per successful "
                "establishment if imports are independent")
            if 0.0 < q < 1.0:
                for tgt in (0.5, 0.95):
                    n_need = int(np.ceil(np.log(1.0 - tgt) / np.log(q)))
                    add(f"n_imports_for_{tgt:g}_establishment", n_need, "cases",
                        f"R0={r0}, k={k_label(k)}",
                        f"smallest n with 1 - q**n >= {tgt:g}")

    # --- how fast P_survive(g) approaches its limit --------------------------
    # P_survive(g) DECREASES towards 1 - q (extinction is absorbing), so the
    # convergence metric is the first g within 5% of the limiting value.
    for k in [0.1, 0.5, np.inf]:
        surv = bm.survival_by_generation(np.array([2.5]), k, 60)[:, 0]
        ult = float(1.0 - bm.extinction_prob(np.array([2.5]), k)[0])
        close = np.where(surv <= 1.05 * ult)[0]
        g5 = int(close[0] + 1) if close.size else -1
        add("generations_to_within_5pct_of_ultimate", g5, "generations",
            f"R0=2.5, k={k_label(k)}",
            "first g with P_survive(g) <= 1.05 * P_survive(infinity)")
        add("p_survive_g1_over_ultimate", round(float(surv[0] / ult), 4), "ratio",
            f"R0=2.5, k={k_label(k)}",
            "P_survive(g=1) relative to the limiting value")

    # --- seasonal: main figures ----------------------------------------------
    for r0p in R0_PEAK_LIST:
        sel = fig5[(fig5["scenario"] == "main_grid") & (fig5["R0_peak"] == r0p)
                   & (fig5["w"] == W_REF) & (fig5["tau"] == TAU_REF)]
        col = "p_survive_g6"
        p = sel[col].to_numpy()
        imax = int(np.argmax(p))
        add("seasonal_max_establishment", round(float(p.max()), 6), "probability",
            f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, g={G_HORIZON}",
            "annual maximum of the within-season establishment probability")
        add("seasonal_optimal_introduction_day", int(sel["day"].to_numpy()[imax]),
            "day-of-year",
            f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, g={G_HORIZON}",
            "day at which the establishment probability peaks")
        for thr in WINDOW_THRESHOLDS:
            w_ = bm.vulnerable_window(p, thr)
            add(f"vulnerable_window_length_thr{thr:g}", w_["length"], "days",
                f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, g={G_HORIZON}",
                f"days with P >= {thr:g}; from day {w_['start_day']} to day {w_['end_day']}")
            add(f"vulnerable_window_start_thr{thr:g}", w_["start_day"], "day-of-year",
                f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, g={G_HORIZON}",
                "")
            add(f"vulnerable_window_end_thr{thr:g}", w_["end_day"], "day-of-year",
                f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, g={G_HORIZON}",
                "")
        # winter trough
        add("seasonal_min_establishment", round(float(p.min()), 8), "probability",
            f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, g={G_HORIZON}",
            "annual minimum of the establishment probability")

    # --- seasonal: dispersion sensitivity ------------------------------------
    for k in K_SEASONAL_PANEL:
        sel = fig5[(fig5["scenario"] == "dispersion_panel")
                   & (fig5["k"] == k_label(k))]
        p = sel["p_survive_g6"].to_numpy()
        add("seasonal_max_establishment_by_k", round(float(p.max()), 6), "probability",
            f"R0_peak=2.5, w={W_REF:g}, tau={TAU_REF}, k={k_label(k)}, g={G_HORIZON}",
            "annual maximum, dispersion sensitivity")

    # --- seasonal: width / generation-time sensitivity -----------------------
    for w in W_LIST:
        for tau in TAU_LIST:
            sel = fig5[(fig5["scenario"] == "main_grid") & (fig5["R0_peak"] == 2.5)
                       & (fig5["w"] == w) & (fig5["tau"] == tau)]
            p = sel["p_survive_g6"].to_numpy()
            add("seasonal_max_establishment_sensitivity", round(float(p.max()), 6),
                "probability",
                f"R0_peak=2.5, w={w:g}, tau={tau}, k={k_label(K_REF)}, g={G_HORIZON}",
                "season width and generation-time sensitivity")

    # --- periodic steady state ------------------------------------------------
    for r0p in R0_PEAK_LIST:
        for w in W_LIST:
            sel = fig5[(fig5["scenario"] == "main_grid") & (fig5["R0_peak"] == r0p)
                       & (fig5["w"] == w) & (fig5["tau"] == TAU_REF)]
            pper = sel["p_establish_periodic"].to_numpy()
            sup = bool(sel["season_supercritical"].iloc[0])
            add("periodic_steadystate_max_establishment", round(float(pper.max()), 8),
                "probability",
                f"R0_peak={r0p}, w={w:g}, tau={TAU_REF}, k={k_label(K_REF)}",
                "1 - q(t) in the periodic steady state; 0 means the seasonal "
                "cycle is subcritical and persistence is impossible")
            add("seasonal_cycle_supercritical", int(sup), "0/1",
                f"R0_peak={r0p}, w={w:g}, tau={TAU_REF}, k={k_label(K_REF)}",
                "1 iff the product of generation means around the seasonal "
                "cycle exceeds 1")

    # analytic threshold: R0_peak needed for a supercritical seasonal cycle
    for w in W_LIST:
        base = bm.seasonal_R0(DAYS, 1.0, w, T_PEAK)[None, :]
        logm = bm.criticality_logM(base, TAU_REF)[0]      # = -C/L * L  at R0_peak = 1
        L = 365 // logm.size
        threshold = float(np.exp(-logm.min() / L))
        add("critical_R0_peak_threshold", round(threshold, 4), "R0",
            f"w={w:g}, tau={TAU_REF}, t_peak={T_PEAK:g}",
            "minimum R0_peak for which the seasonal cycle is supercritical "
            "(periodic persistence possible)")

    # --- vector control -------------------------------------------------------
    scan = fig6[fig6["record_type"] == "theta_scan"]
    for r0p in R0_PEAK_LIST:
        sub = scan[scan["R0_peak"] == r0p]
        base = float(sub[sub["theta"] == sub["theta"].min()]["p_survive_g6"].iloc[0])
        for theta in [0.1, 0.3, 0.5]:
            v = float(sub[np.isclose(sub["theta"], theta)]["p_survive_g6"].iloc[0])
            add("annual_max_establishment_under_control", round(v, 6), "probability",
                f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, "
                f"theta={theta:g}, g={G_HORIZON}",
                "annual maximum of P_est(t) when R0 is scaled by (1 - theta)")
            add("control_relative_reduction", round(1.0 - v / base, 6), "fraction",
                f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, "
                f"theta={theta:g}, g={G_HORIZON}",
                "relative drop of the annual maximum vs theta = 0")
            wl = float(sub[np.isclose(sub["theta"], theta)]["window_len_thr0.05"].iloc[0])
            add("vulnerable_window_length_under_control", int(wl), "days",
                f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, "
                f"theta={theta:g}, g={G_HORIZON}, threshold=0.05",
                "")
        th, val, note = theta_star(r0p)
        add("theta_required_for_target", ("" if np.isnan(th) else round(th, 6)),
            "fraction of R0",
            f"R0_peak={r0p}, w={W_REF:g}, tau={TAU_REF}, k={k_label(K_REF)}, "
            f"g={G_HORIZON}, target={TARGET_P:g}",
            f"theta s.t. annual max P_est = {TARGET_P:g}; {note}")

    df = pd.DataFrame(out)
    df.to_csv("key_summary.csv", index=False)
    return df


# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[1/6] Fig 2 : extinction probability vs R0 ...")
    f2 = make_fig2()
    print(f"      {len(f2)} rows")

    print("[2/6] Fig 3 : establishment vs number of imported cases ...")
    f3 = make_fig3()
    print(f"      {len(f3)} rows")

    print("[3/6] Fig 4 : survival by generation ...")
    f4 = make_fig4()
    print(f"      {len(f4)} rows")

    print("[4/6] Fig 5 : seasonal establishment ...")
    f5 = make_fig5()
    print(f"      {len(f5)} rows")

    print("[5/6] Fig 6 : vector control ...")
    f6 = make_fig6()
    print(f"      {len(f6)} rows")

    print("[6/6] key summary ...")
    ks = make_key_summary(f5, f6)
    print(f"      {len(ks)} rows")

    print(f"\ndone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
