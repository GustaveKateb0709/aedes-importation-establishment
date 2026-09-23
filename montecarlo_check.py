"""
montecarlo_check.py
===================
Independent validation of the analytical / numerical branching-process results
by direct stochastic simulation of the Galton-Watson process.

Blocks
------
  1. analytic_k1             numeric root of s = G(s) vs the closed form 1/R0 (k = 1)
  2. homogeneous             Monte Carlo extinction vs G^g(0) and vs q
  3. homogeneous_n_imports   Monte Carlo P(establish) for n index cases vs 1 - q0**n
  4. seasonal                Monte Carlo survival under a seasonal R0(t) vs the
                             seasonal PGF composition
  5. mode_crosscheck         per-individual draws vs the exact aggregated draw

Run:  ../.venv/bin/python montecarlo_check.py
Out:  montecarlo_validation.csv

Why the "aggregated" draw is exact, not an approximation
--------------------------------------------------------
The negative binomial family is infinitely divisible: the sum of Z independent
NB(mean m, dispersion k) variates is exactly NB(mean Z*m, dispersion Z*k).
(Proof: NB(m, k) is Poisson(lambda) with lambda ~ Gamma(shape k, scale m/k);
the sum of Z iid Gamma(k, m/k) variates is Gamma(Z*k, m/k).)  The simulator
therefore draws one variate per individual while the population is small, and
switches to the exact aggregated draw once the population is large -- otherwise
exploding populations would make the run intractable.  Block 5 verifies
empirically that the two routes agree.

English only.  No external data, no network.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import branching_model as bm

SEED = 20260906
N_REP = 100_000
CHUNK = 10_000
G_HORIZON = 25          # bias-free comparison horizon (generations)
G_MAX = 60              # horizon used for the "ultimate" extinction estimate
POP_CAP = 10_000        # a replicate above this is treated as non-extinct
AGG_THRESHOLD = 32      # switch from per-individual to aggregated draws above this
CROSSCHECK_POP_CAP = 400   # keeps the forced per-individual run tractable

OUT_CSV = "montecarlo_validation.csv"

COLUMNS = ["scenario_type", "R0", "R0_peak", "k", "n_imports", "intro_day",
           "generation_g", "n_rep", "quantity", "analytic_value", "mc_value",
           "abs_diff", "mc_se", "z_score", "method"]


# ----------------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------------
def _draw_offspring(rng, z_counts, r0, k, force_mode=None):
    """Total offspring produced by each entry of `z_counts`.

    z_counts : (N,) current population sizes (int64)
    r0       : float, mean offspring number per individual this generation
    Returns  : (N,) next-generation sizes
    """
    n = z_counts.size
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out

    use_small = z_counts <= AGG_THRESHOLD
    use_large = ~use_small
    if force_mode == "per_individual":
        use_small = np.ones(n, dtype=bool)
        use_large = np.zeros(n, dtype=bool)
    elif force_mode == "aggregated":
        use_small = np.zeros(n, dtype=bool)
        use_large = np.ones(n, dtype=bool)

    # --- literal Galton-Watson: one variate per individual ---
    if use_small.any():
        idx = np.where(use_small & (z_counts > 0))[0]
        counts = z_counts[idx]
        total = int(counts.sum())
        if total > 0:
            if np.isinf(k):
                draws = rng.poisson(r0, size=total)
            else:
                p = k / (k + r0)
                draws = rng.negative_binomial(k, p, size=total)
            starts = np.cumsum(counts) - counts
            out[idx] = np.add.reduceat(draws, starts).astype(np.int64)

    # --- exact aggregated draw, NB(mean Z*r0, dispersion Z*k) ---
    if use_large.any():
        idx = np.where(use_large)[0]
        zl = z_counts[idx].astype(float)
        if np.isinf(k):
            draws = rng.poisson(r0 * zl)
        else:
            p = k / (k + r0)
            draws = rng.negative_binomial(k * zl, p)
        out[idx] = draws.astype(np.int64)

    return out


def simulate(R0, k, n_rep=N_REP, g_max=G_MAX, pop_cap=POP_CAP,
             n_imports=1, rng=None, force_mode=None):
    """Direct Galton-Watson simulation.

    R0 : float, or a 1-D array of length g_max giving the per-generation mean
         (used for the seasonal check, where generation j reproduces with
         mean R0[j]).

    Returns
    -------
    n_extinct_by_g : int64 array, length g_max.  n_extinct_by_g[g-1] is the
        number of replicates in which ALL n_imports chains are extinct at
        generation g (i.e. Z_g = 0 for every chain).
    n_rep : int
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    R0_arr = np.broadcast_to(np.asarray(R0, dtype=float), (g_max,))

    n_extinct_by_g = np.zeros(g_max, dtype=np.int64)
    done = 0
    while done < n_rep:
        m = min(CHUNK, n_rep - done)
        z = np.ones((m, n_imports), dtype=np.int64)
        escaped = np.zeros(m, dtype=bool)
        for g in range(1, g_max + 1):
            active = np.where((z.sum(axis=1) > 0) & ~escaped)[0]
            if active.size == 0:
                n_extinct_by_g[g - 1:] += int((z.sum(axis=1) == 0).sum())
                break
            r0g = float(R0_arr[g - 1])
            for c in range(n_imports):
                col = z[active, c]
                live = col > 0
                if not live.any():
                    continue
                sub = active[live]
                z[sub, c] = _draw_offspring(rng, z[sub, c], r0g, k,
                                            force_mode=force_mode)
            tot = z.sum(axis=1)
            escaped |= tot > pop_cap
            z[escaped] = pop_cap + 1
            n_extinct_by_g[g - 1] += int((z.sum(axis=1) == 0).sum())
        done += m
    return n_extinct_by_g, n_rep


def _se(p, n):
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def _row(**kw):
    r = {c: "" for c in COLUMNS}
    r.update(kw)
    return r


# ----------------------------------------------------------------------------
# validation blocks
# ----------------------------------------------------------------------------
def block_analytic_k1(rows):
    """Numeric root of s = G(s) against the closed form q = 1/R0 for k = 1."""
    R0 = np.linspace(1.001, 6.0, 400)
    q_num = bm.extinction_prob(R0, 1.0)
    q_ana = bm.analytic_extinction_k1(R0)
    diff = np.abs(q_num - q_ana)
    for r, qn, qa, d in zip(R0, q_num, q_ana, diff):
        rows.append(_row(scenario_type="analytic_k1", R0=float(r), k=1.0,
                         n_imports=1, quantity="extinction_probability",
                         analytic_value=float(qa), mc_value=float(qn),
                         abs_diff=float(d), method="bisection_vs_closed_form"))
    print(f"  grid of {R0.size} points, R0 in [{R0.min()}, {R0.max()}]")
    return float(diff.max())


def block_homogeneous(rows, rng):
    R0_list = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 4.0, 6.0]
    k_list = [0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
    max_dev = 0.0
    for R0 in R0_list:
        for k in k_list:
            child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            n_ext, n_rep = simulate(R0, k, rng=child)

            surv = bm.survival_by_generation(np.array([R0]), k, G_HORIZON)[:, 0]
            ana_gen = 1.0 - float(surv[G_HORIZON - 1])       # G^25(0)
            mc_gen = float(n_ext[G_HORIZON - 1]) / n_rep
            se_gen = _se(mc_gen, n_rep)
            d_gen = abs(ana_gen - mc_gen)

            ana_ult = float(bm.extinction_prob(np.array([R0]), k)[0])
            mc_ult = float(n_ext[G_MAX - 1]) / n_rep
            se_ult = _se(mc_ult, n_rep)
            d_ult = abs(ana_ult - mc_ult)
            max_dev = max(max_dev, d_gen, d_ult)

            rows.append(_row(scenario_type="homogeneous", R0=R0, k=k, n_imports=1,
                             generation_g=G_HORIZON, n_rep=n_rep,
                             quantity="extinct_by_generation",
                             analytic_value=ana_gen, mc_value=mc_gen,
                             abs_diff=d_gen, mc_se=se_gen,
                             z_score=d_gen / se_gen if se_gen > 0 else "",
                             method="galton_watson"))
            rows.append(_row(scenario_type="homogeneous", R0=R0, k=k, n_imports=1,
                             generation_g=G_MAX, n_rep=n_rep,
                             quantity="ultimate_extinction_probability",
                             analytic_value=ana_ult, mc_value=mc_ult,
                             abs_diff=d_ult, mc_se=se_ult,
                             z_score=d_ult / se_ult if se_ult > 0 else "",
                             method="galton_watson"))
            print(f"  R0={R0:<4} k={k!s:>5} | gen25 ana={ana_gen:.5f} mc={mc_gen:.5f} "
                  f"d={d_gen:.5f} | ultimate ana={ana_ult:.5f} mc={mc_ult:.5f} d={d_ult:.5f}")
    return max_dev


def block_n_imports(rows, rng):
    R0, k = 2.5, 0.5
    surv = bm.survival_by_generation(np.array([R0]), k, G_HORIZON)[:, 0]
    q0 = 1.0 - float(surv[G_HORIZON - 1])          # P(one chain extinct by gen 25)
    max_dev = 0.0
    for n in (2, 5, 10):
        child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        n_ext, n_rep = simulate(R0, k, n_imports=n, rng=child)
        ana = 1.0 - q0 ** n
        # n_ext counts replicates in which ALL n chains are extinct, i.e. q0**n
        mc = 1.0 - float(n_ext[G_HORIZON - 1]) / n_rep
        se = _se(mc, n_rep)
        d = abs(ana - mc)
        max_dev = max(max_dev, d)
        rows.append(_row(scenario_type="homogeneous_n_imports", R0=R0, k=k,
                         n_imports=n, generation_g=G_HORIZON, n_rep=n_rep,
                         quantity="establishment_probability",
                         analytic_value=ana, mc_value=mc, abs_diff=d, mc_se=se,
                         z_score=d / se if se > 0 else "", method="galton_watson"))
        print(f"  n={n:<3} ana={ana:.5f} mc={mc:.5f} d={d:.5f}")
    return max_dev


def block_seasonal(rows, rng):
    days = np.arange(1, 366)
    tau, w, t_peak, g = 15, 55.0, 190.0, 6
    combos = [(2.5, 0.5, [15, 100, 150, 178, 190, 250, 320]),
              (4.0, 0.5, [150, 190]),
              (2.5, 0.1, [150, 190])]
    max_dev = 0.0
    for R0_peak, k, intro_days in combos:
        R0_t = bm.seasonal_R0(days, R0_peak, w, t_peak)
        ana_all = bm.seasonal_survival(R0_t[None, :], k, tau, g)[0]
        for t in intro_days:
            r0_seq = np.array([R0_t[(int(t) - 1 + j * tau) % 365] for j in range(g)])
            child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            n_ext, n_rep = simulate(r0_seq, k, g_max=g, rng=child)
            mc = 1.0 - float(n_ext[g - 1]) / n_rep
            se = _se(1.0 - mc, n_rep)
            ana = float(ana_all[int(t) - 1])
            d = abs(ana - mc)
            max_dev = max(max_dev, d)
            rows.append(_row(scenario_type="seasonal", R0_peak=R0_peak, k=k,
                             n_imports=1, intro_day=int(t), generation_g=g,
                             n_rep=n_rep, quantity="survival_probability_at_g",
                             analytic_value=ana, mc_value=mc, abs_diff=d, mc_se=se,
                             z_score=d / se if se > 0 else "",
                             method="galton_watson_seasonal"))
            print(f"  R0_peak={R0_peak} k={k} day={t:<4} ana={ana:.5f} mc={mc:.5f} d={d:.5f}")
    return max_dev


def block_mode_crosscheck(rows, rng):
    """Per-individual draws vs the exact aggregated NB draw on the same parameter set."""
    R0, k, g = 2.5, 0.5, 25
    surv = bm.survival_by_generation(np.array([R0]), k, g)[:, 0]
    ana = 1.0 - float(surv[g - 1])
    out = {}
    for mode in ("per_individual", "aggregated"):
        child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        n_ext, n_rep = simulate(R0, k, g_max=g, pop_cap=CROSSCHECK_POP_CAP,
                                rng=child, force_mode=mode)
        mc = float(n_ext[g - 1]) / n_rep
        se = _se(mc, n_rep)
        out[mode] = mc
        rows.append(_row(scenario_type="mode_crosscheck", R0=R0, k=k, n_imports=1,
                         generation_g=g, n_rep=n_rep,
                         quantity="extinct_by_generation",
                         analytic_value=ana, mc_value=mc, abs_diff=abs(ana - mc),
                         mc_se=se, z_score=abs(ana - mc) / se if se > 0 else "",
                         method=mode))
        print(f"  mode={mode:<15} mc={mc:.5f}  (analytic {ana:.5f})")
    d = abs(out["per_individual"] - out["aggregated"])
    rows.append(_row(scenario_type="mode_crosscheck", R0=R0, k=k, n_imports=1,
                     generation_g=g, n_rep=N_REP,
                     quantity="per_individual_minus_aggregated",
                     mc_value=d, abs_diff=d, method="comparison"))
    print(f"  |per_individual - aggregated| = {d:.6f}")
    return d


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []

    print("[1/5] analytic closed form, k = 1 ...")
    dev_k1 = block_analytic_k1(rows)

    print("[2/5] Monte Carlo vs analytic, homogeneous ...")
    dev_hom = block_homogeneous(rows, rng)

    print("[3/5] Monte Carlo vs analytic, n imported cases ...")
    dev_n = block_n_imports(rows, rng)

    print("[4/5] Monte Carlo vs numeric, seasonal ...")
    dev_sea = block_seasonal(rows, rng)

    print("[5/5] draw-mode cross-check ...")
    dev_mode = block_mode_crosscheck(rows, rng)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(OUT_CSV, index=False)

    mc_mask = df["scenario_type"].isin(["homogeneous", "homogeneous_n_imports", "seasonal"])
    max_se = float(pd.to_numeric(df.loc[mc_mask, "mc_se"], errors="coerce").max())
    dev_mc = max(dev_hom, dev_n, dev_sea)

    print("\n" + "=" * 74)
    print(f"MAX ABS DEVIATION (k=1 numeric vs analytic 1/R0): {dev_k1:.4e}")
    print(f"MAX ABS DEVIATION (Monte Carlo vs analytic/numeric): {dev_mc:.4e}")
    print(f"    homogeneous block : {dev_hom:.4e}")
    print(f"    n-imports block   : {dev_n:.4e}")
    print(f"    seasonal block    : {dev_sea:.4e}")
    print(f"    draw-mode cross-check (per-individual vs aggregated): {dev_mode:.4e}")
    print(f"largest Monte Carlo standard error    : {max_se:.4e}")
    print(f"95% Monte Carlo tolerance (1.96 * se) : {1.96 * max_se:.4e}")
    print(f"rows written: {len(df)}  ->  {OUT_CSV}")
    print(f"runtime: {time.time() - t0:.1f} s")
    print("=" * 74)

    if dev_k1 > 0.01 or dev_mc > 0.01:
        raise SystemExit("VALIDATION FAILED: deviation above 0.01 -- "
                         "investigate the root cause before using any result.")


if __name__ == "__main__":
    main()
