# Establishment probability of imported Aedes-borne arbovirus cases

Analysis code and numerical results for the manuscript

> *Why most imported Aedes-borne arbovirus cases fail to establish local
> transmission: the role of offspring overdispersion
> distributions*

Pure-theory / simulation study. No external data, no network access, no
proprietary dependencies.

---

## 1. How to run

```bash
# one-off: create the environment (Python 3.13, numpy/scipy/matplotlib/pandas)
python3 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib pandas

# then, from this directory, in this order:
.venv/bin/python montecarlo_check.py    # ~13 s   validation, writes montecarlo_validation.csv
.venv/bin/python run_analysis.py        # ~1 s    full grid, writes the fig*_*.csv + key_summary.csv
.venv/bin/python make_figures.py        # ~10 s   writes PNG (300 dpi) + PDF into ./figures/ (created automatically)
```

Everything is deterministic: all random draws use `numpy.random.default_rng(20260906)`
with per-scenario child seeds, so re-running the pipeline reproduces every number
and every figure byte-for-byte (up to PNG metadata timestamps).

Runtime on an 8 GB laptop: well under one minute in total.

## 2. Files

| file | what it is |
|---|---|
| `branching_model.py` | model core: PGF, extinction probability (bisection), survival to generation *g*, seasonal `R0(t)`, seasonal finite-horizon survival, periodic steady state, vulnerable-window helper |
| `run_analysis.py` | runs the parameter grid and writes all result CSVs |
| `make_figures.py` | draws Figs 1-6 plus supplementary Fig S1 |
| `montecarlo_check.py` | independent Galton-Watson simulation used to validate the analytic/numeric results |
| `fig2_extinction_vs_R0.csv` | q vs R0 on a 400-point grid, one row per (R0, k) |
| `fig3_establishment_vs_imports.csv` | 1 - q^n for n = 1..100, R0 in {1.5, 2.5, 4.0}, one row per (R0, k, n) |
| `fig4_survival_by_generation.csv` | P_survive(g), g = 1..12, plus the ultimate value 1 - q |
| `fig5_seasonal_establishment.csv` | daily seasonal results: R0(t), P_survive(t, g) for g in {2,4,6,8,12}, periodic steady state, criticality diagnostic |
| `fig6_control_effect.csv` | control scenarios: daily curves for theta in {0,...,0.5} and a fine theta scan (annual maximum + window length) |
| `key_summary.csv` | every number quoted in the text: metric, value, unit, parameter condition, note |
| `montecarlo_validation.csv` | analytic vs Monte Carlo, row by row, with standard errors and z-scores |

## 3. Model

**Offspring law.** Negative binomial with mean `R0` and dispersion `k`
(variance `R0 + R0^2/k`), PGF `G(s) = [1 + (R0/k)(1-s)]^(-k)`. `k = inf` is the
Poisson limit `exp(R0(s-1))`.

**Extinction.** `q` is the smallest root of `s = G(s)` on `[0, 1)`, found by
bisection (the bracket stops at `1 - 1e-12` because `s = 1` is always a root).
For `R0 <= 1`, `q = 1`.

**n independent imported cases.** `P_establish(n) = 1 - q^n`.

**Observable outbreak.** `P_survive(g) = 1 - G^g(0)`, where `G^g` is the g-fold
composition of `G`; converges to `1 - q` from above.

**Seasonality.** `R0(t) = R0_peak * exp(-d(t, t_peak)^2 / (2 w^2))` with `d` the
cyclic distance on a 365-day circle (single-peaked, periodic, peak `R0_peak` at
`t_peak = day 190`, width `w`).

**Seasonal finite-horizon establishment (headline seasonal metric).**
`P_est(t) = P_survive(t, g)`, the probability that a chain started by one
imported case on day `t` is still alive `g` generations later, where generation
`j` reproduces with mean `R0(t + j*tau)`:

```
P(Z_g = 0 | t) = G_t( G_{t+tau}( ... G_{t+(g-1)tau}(0) ... ) )
```

computed by a backward recursion with terminal value 0. The headline horizon is
`g = 6` generations (~90 days at `tau = 15 d`); `g = 2, 4, 8, 12` are also in the
CSV.

**Periodic steady state.** The infinite-horizon extinction probability of the
seasonal process satisfies `q(t) = G_t(q(t + tau))` with periodic boundary
conditions. The *minimal* solution is used. The map `t -> t + tau (mod 365)`
splits the year into `gcd(tau, 365)` disjoint cycles; on each cycle the process
is subcritical - and therefore `q = 1` identically - iff the product of the
generation means around the cycle is `<= 1` (Athreya & Ney). We test this in log
space and short-circuit subcritical cycles; supercritical cycles are solved by
bisection on the cycle return map. This is reported in
`p_establish_periodic` and in Fig S1.

**Vector control.** `R0(t) -> (1 - theta) R0(t)`, `theta` in `{0, 0.1, ..., 0.5}`;
plus a root-find for the `theta` that pushes the annual maximum of `P_est(t)`
below 0.05.

## 4. Validation (self-checks that must pass)

1. **Closed form, k = 1.** For geometric offspring `q = 1/R0` exactly.
   Max |numeric - analytic| over a 400-point grid: **5.6e-14**.
2. **Monte Carlo.** Direct Galton-Watson simulation, 100,000 replicates per
   parameter set. Max |MC - analytic| across all blocks: **4.3e-3**, against a
   largest Monte Carlo standard error of 1.6e-3 (95% tolerance 3.1e-3); the
   single largest deviation is 2.7 standard errors, consistent with noise over
   ~100 simultaneous comparisons. The seasonal backward recursion is validated
   the same way (max deviation 3.2e-3).
3. **Draw-mode cross-check.** Per-individual sampling vs the exact aggregated
   negative-binomial draw (exact by infinite divisibility) agree to 4.5e-3,
   i.e. within Monte Carlo error.

`montecarlo_check.py` raises a hard failure if any deviation exceeds 0.01.

## 5. Parameter grid

| parameter | values |
|---|---|
| `R0` (Fig 2) | 400 points on [0.2, 6.0] |
| `k` | 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, and the Poisson limit |
| `n` imported cases | 1, 2, 5, 10, 25, 50, 100 |
| `R0_peak` (seasonal) | 1.5, 2.5, 4.0 |
| `t_peak` | day 190 |
| `w` | 40, 55, 70 days |
| `tau` (generation time) | 10, 15, 20 days |
| `theta` (control) | 0, 0.1, 0.2, 0.3, 0.4, 0.5 + a 96-point scan to 0.95 |
| reference dispersion for seasonal figures | k = 0.5 |

## 6. Note on the two seasonal quantities

With a realistic Gaussian season and `w = 55 d`, the seasonal cycle is
**subcritical** for every `R0_peak` in {1.5, 2.5, 4.0}: the product of the
generation means around the year is far below 1, so the periodic steady state
gives `q(t) = 1` and `P_establish(t) = 0` exactly. The critical peak value is
`R0_peak > 6.27` for `w = 55 d` (and `> 32.1` for `w = 40 d`, `> 3.11` for
`w = 70 d`).

That is a real property of the model, not a numerical artefact, and it is the
reason the paper reports the **finite-horizon** quantity `P_survive(t, g)` as
the headline seasonal metric: it answers the public-health question ("how likely
is a chain introduced on day t to still be circulating ~90 days later") and it
is non-degenerate. Fig S1 shows both quantities side by side.
