"""
branching_model.py
==================
Core model for the manuscript:

  "Why most imported Aedes-borne arbovirus cases fail to establish local
   transmission: the role of offspring overdispersion
   distributions"

Everything here is deterministic, vectorised numpy.  No external data, no
network access, no I/O.

Contents
--------
  pgf                      offspring probability generating function (NB / Poisson)
  pgf_deriv                dG/ds
  extinction_prob          q  = smallest root of s = G(s)        (bisection)
  analytic_extinction_k1   1/R0  -- closed form for the geometric (k = 1) case
  establishment_prob       P = 1 - q**n   for n independent imported cases
  survival_by_generation   P_survive(g) = 1 - G^g(0)
  seasonal_R0              R0(t) Gaussian seasonal profile on a 365-day cycle
  seasonal_survival        P_survive(t, g) under a seasonal (time-inhomogeneous) R0(t)
  periodic_extinction      q(t) periodic steady state of the seasonal process
  vulnerable_window        contiguous run of days with P >= threshold

Conventions
-----------
  * Day index in arrays is 0-based; day-of-year printed to the user is 1-based
    (index t  <->  day t+1).
  * `k = np.inf` selects the Poisson limit.
  * Arrays of scenarios are stored row-wise: shape (m, 365).

Branching-process model and analysis pipeline. All identifiers and comments in English.
"""

from __future__ import annotations

import numpy as np

PERIOD = 365                 # length of the seasonal cycle, days
POISSON_K = np.inf           # sentinel for the Poisson (k -> infinity) limit


# ----------------------------------------------------------------------------
# 1. Offspring probability generating function
# ----------------------------------------------------------------------------
def pgf(s, R0, k):
    """Offspring PGF.

    Negative binomial with mean R0 and dispersion k:
        G(s) = [1 + (R0 / k) (1 - s)]^(-k)
    Poisson limit (k -> inf):
        G(s) = exp(R0 (s - 1))

    Parameters
    ----------
    s, R0 : array_like, broadcastable
    k     : float (np.inf for Poisson)

    Returns
    -------
    ndarray, G(s) in [0, 1]
    """
    s = np.asarray(s, dtype=float)
    R0 = np.asarray(R0, dtype=float)
    if np.isinf(k):
        return np.exp(R0 * (s - 1.0))
    kk = float(k)
    a = (R0 / kk) * (1.0 - s)
    return np.exp(-kk * np.log1p(a))


def pgf_deriv(s, R0, k):
    """dG/ds at s.  For the NB family G'(s) = R0 [1 + (R0/k)(1-s)]^(-k-1)."""
    s = np.asarray(s, dtype=float)
    R0 = np.asarray(R0, dtype=float)
    if np.isinf(k):
        return R0 * np.exp(R0 * (s - 1.0))
    kk = float(k)
    a = (R0 / kk) * (1.0 - s)
    return R0 * np.exp(-(kk + 1.0) * np.log1p(a))


# ----------------------------------------------------------------------------
# 2. Extinction probability
# ----------------------------------------------------------------------------
def extinction_prob(R0, k, n_iter=120, hi=1.0 - 1e-12):
    """Extinction probability q = smallest root of s = G(s) on [0, 1].

    For R0 <= 1 the extinction probability is 1 (certain extinction).
    For R0 >  1 the root is found by BISECTION on [0, hi]; note that
    f(s) = G(s) - s always vanishes at s = 1, so the bracket must stop
    short of 1 (hence `hi = 1 - 1e-12`).

    Parameters
    ----------
    R0 : array_like   basic reproduction number(s)
    k  : float        dispersion parameter (np.inf -> Poisson)

    Returns
    -------
    ndarray with the same shape as R0
    """
    R0 = np.asarray(R0, dtype=float)
    scalar_input = (R0.ndim == 0)
    R = np.atleast_1d(R0)

    lo = np.zeros_like(R)
    up = np.full_like(R, float(hi))
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + up)
        f = pgf(mid, R, k) - mid
        neg = f < 0.0
        up = np.where(neg, mid, up)
        lo = np.where(neg, lo, mid)
    q = 0.5 * (lo + up)
    q = np.where(R <= 1.0, 1.0, q)
    return float(q[0]) if scalar_input else q


def analytic_extinction_k1(R0):
    """Closed-form extinction probability for k = 1 (geometric offspring): q = 1/R0."""
    R0 = np.asarray(R0, dtype=float)
    return np.where(R0 <= 1.0, 1.0, 1.0 / np.maximum(R0, 1e-300))


def establishment_prob(R0, k, n=1):
    """P(establishment) = 1 - q**n for n independent imported index cases."""
    q = extinction_prob(R0, k)
    return 1.0 - np.power(q, float(n))


# ----------------------------------------------------------------------------
# 3. Survival to generation g  ("observable outbreak")
# ----------------------------------------------------------------------------
def survival_by_generation(R0, k, g_max):
    """P_survive(g) = 1 - G^g(0), the probability the chain is still alive at
    generation g, for g = 1 .. g_max.

    G^g denotes the g-fold composition of G with itself.  Computed by the
    backward recursion  x <- G(x)  starting from x = 0.

    Returns
    -------
    ndarray, shape (g_max,) + R0.shape
    """
    R0 = np.asarray(R0, dtype=float)
    x = np.zeros_like(R0)
    out = np.empty((int(g_max),) + R0.shape, dtype=float)
    for g in range(1, int(g_max) + 1):
        x = pgf(x, R0, k)
        out[g - 1] = 1.0 - x
    return out


# ----------------------------------------------------------------------------
# 4. Seasonality
# ----------------------------------------------------------------------------
def seasonal_R0(days, R0_peak, w, t_peak, period=PERIOD):
    """Gaussian seasonal profile of the basic reproduction number.

        R0(t) = R0_peak * exp( -0.5 * ( d(t, t_peak) / w )**2 )

    where d(.,.) is the *cyclic* distance on the 365-day circle, so the
    profile is periodic and single-peaked with maximum R0_peak at t_peak.
    `w` controls the width of the favourable season.

    Parameters
    ----------
    days : array_like, 1-based day-of-year (1 .. 365)
    """
    days = np.asarray(days, dtype=float)
    d = np.mod(days - float(t_peak) + period / 2.0, period) - period / 2.0
    return float(R0_peak) * np.exp(-0.5 * (d / float(w)) ** 2)


def seasonal_survival(R0_t, k, tau, g):
    """Seasonal finite-horizon survival probability.

    P_survive(t, g) = 1 - P(Z_g = 0 | one case introduced on day t), where the
    chain advances one generation every `tau` days and the offspring law at
    generation j uses R0(t + j*tau).

    P(Z_g = 0) = G_{t}( G_{t+tau}( ... G_{t+(g-1)tau}(0) ... ) )
    obtained by the backward recursion x <- G_{t + j*tau}(x).

    Parameters
    ----------
    R0_t : ndarray (m, period)   R0 profile, one row per scenario
    k    : float
    tau  : int    generation time in days
    g    : int    number of generations

    Returns
    -------
    ndarray (m, period): survival probability for every introduction day
    """
    R0_t = np.asarray(R0_t, dtype=float)
    m, P = R0_t.shape
    x = np.zeros((m, P), dtype=float)
    for j in range(int(g) - 1, -1, -1):
        shift = (np.arange(P) + int(tau) * j) % P
        x = pgf(x, R0_t[:, shift], k)
    return 1.0 - x


# ----------------------------------------------------------------------------
# 5. Periodic steady state of the seasonal branching process
# ----------------------------------------------------------------------------
def _orbit_list(nxt):
    """Decompose the map t -> nxt[t] into disjoint cycles.

    Returns a list of 1-D integer arrays; within each array orb,
    orb[j+1] = nxt[orb[j]] (cyclically).
    """
    P = nxt.size
    oid = np.full(P, -1, dtype=int)
    orbits = []
    for i in range(P):
        if oid[i] >= 0:
            continue
        cur = []
        j = i
        while oid[j] < 0:
            oid[j] = len(orbits)
            cur.append(j)
            j = nxt[j]
        orbits.append(np.asarray(cur, dtype=int))
    return orbits


def _cycle_fixed_point(R, k, n_bisect=100, n_refine=400, tol=1e-13):
    """Minimal fixed point of the cyclic system q_j = G_j(q_{j+1 mod L}).

    R has shape (n, L): R[:, j] is the mean offspring number at position j of
    the cycle.  The cycle is assumed supercritical (product of means > 1);
    otherwise the minimal fixed point is q = 1 and this function is not used.

    Strategy
    --------
    The return map H(y) = G_0 o G_1 o ... o G_{L-1}(y) is increasing and convex
    with H(1) = 1, so the relevant root of H(y) - y = 0 is bracketed by
    [0, 1-eps] and found by bisection (robust, no slow near-critical
    convergence).  The remaining positions follow from q_{L-1} = G_{L-1}(q_0),
    q_{L-2} = G_{L-2}(q_{L-1}), ...  A few Jacobi sweeps polish the result.

    Returns
    -------
    q : ndarray (n, L)
    """
    n, L = R.shape

    def H(y):
        z = np.array(y, dtype=float, copy=True)
        for j in range(L - 1, -1, -1):
            z = pgf(z, R[:, j], k)
        return z

    lo = np.zeros(n)
    hi = np.ones(n)
    # bracket: find a y close enough to 1 that H(y) - y < 0 for every row
    hi_ok = np.zeros(n, dtype=bool)
    for eps in (1e-6, 1e-9, 1e-12, 1e-15):
        cand = 1.0 - eps
        good = (~hi_ok) & ((H(np.full(n, cand)) - cand) < 0.0)
        hi = np.where(good, cand, hi)
        hi_ok = hi_ok | good
        if hi_ok.all():
            break
    # rows that could not be bracketed are numerically indistinguishable from q = 1
    q = np.ones((n, L))
    if not hi_ok.any():
        return q

    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        f = H(mid) - mid
        neg = f < 0.0
        hi = np.where(neg, mid, hi)
        lo = np.where(neg, lo, mid)
    y0 = 0.5 * (lo + hi)
    y0 = np.where(hi_ok, y0, 1.0)

    qq = np.empty((n, L))
    qq[:, 0] = y0
    if L > 1:
        qq[:, L - 1] = pgf(y0, R[:, L - 1], k)
        for p in range(L - 2, 0, -1):
            qq[:, p] = pgf(qq[:, p + 1], R[:, p], k)

    for _ in range(int(n_refine)):
        nxt = pgf(np.roll(qq, -1, axis=1), R, k)
        delta = np.abs(nxt - qq).max()
        qq = nxt
        if delta < tol:
            break
    return qq


def periodic_extinction(R0_t, k, tau, period=PERIOD, return_diag=False):
    """Periodic steady-state extinction probability q(t) of the seasonal process.

    The extinction probability of a time-inhomogeneous Galton-Watson process
    started by one case on day t satisfies the backward recursion

        q(t) = G_t( q(t + tau) ),

    with G_t built from R0(t), and periodic boundary conditions on the
    365-day circle.  The *minimal* non-negative solution is the extinction
    probability; q == 1 (componentwise) is always a solution, so the iteration
    must be started from below (equivalently the minimal fixed point is used).

    NOTE ON SUBCRITICAL ENVIRONMENTS
    --------------------------------
    The map t -> t + tau (mod 365) splits the year into gcd(tau, 365) disjoint
    cycles.  On each cycle the process is subcritical -- and therefore
    q(t) = 1 for every t on that cycle -- iff the product of the generation
    means around the cycle is <= 1 (Athreya & Ney).  We evaluate this
    criterion in log space analytically and short-circuit those cycles, which
    avoids the very slow (algebraic) convergence of the fixed-point iteration
    in the critical / subcritical regime.

    Parameters
    ----------
    R0_t : ndarray (m, period)
    k    : float
    tau  : int

    Returns
    -------
    q : ndarray (m, period)   extinction probabilities
    (diagnostics dict if return_diag=True)
    """
    R0_t = np.asarray(R0_t, dtype=float)
    m, P = R0_t.shape
    nxt = (np.arange(P) + int(tau)) % P
    orbits = _orbit_list(nxt)

    q = np.ones((m, P))
    resid = np.zeros(m)
    n_super = 0

    logR_safe = np.where(R0_t > 0.0, np.log(np.where(R0_t > 0.0, R0_t, 1.0)), 0.0)
    has_zero = (R0_t <= 0.0).any(axis=1)

    for orb in orbits:
        L = orb.size
        # rotate the cycle so that position 0 is the most favourable day
        start = int(np.argmax(R0_t[0][orb]))
        d = orb[(np.arange(L) + start) % L]

        R = R0_t[:, d]                                   # (m, L)
        logM = logR_safe[:, d].sum(axis=1)               # log of cycle mean product
        sub = has_zero | (logM <= 0.0)
        sup = ~sub

        q[:, d] = 1.0
        if not sup.any():
            continue
        n_super += int(sup.sum())
        rows = np.where(sup)[0]
        qq = _cycle_fixed_point(R[rows, :], k)
        q[rows[:, None], d[None, :]] = qq

    if return_diag:
        # residual of the backward recursion on the whole profile
        qs = q[:, nxt]
        res = np.abs(pgf(qs, R0_t, k) - q).max(axis=1)
        # per-cycle log of the product of generation means (criticality criterion)
        per_orbit = np.stack([logR_safe[:, o].sum(axis=1) for o in orbits], axis=1)
        diag = {
            "max_residual": float(res.max()) if m else 0.0,
            "n_orbits": len(orbits),
            "orbit_length": int(orbits[0].size) if orbits else 0,
            "n_supercritical_rows": n_super,
            "log_cycle_mean_product": per_orbit,     # (m, n_orbits)
        }
        return q, diag
    return q


def criticality_logM(R0_t, tau):
    """Per-orbit log of the product of generation means around the seasonal cycle.

    The seasonal branching process is supercritical (non-zero establishment
    probability in the periodic steady state) iff this quantity is > 0.

    Returns ndarray (m, n_orbits).
    """
    R0_t = np.asarray(R0_t, dtype=float)
    m, P = R0_t.shape
    nxt = (np.arange(P) + int(tau)) % P
    orbits = _orbit_list(nxt)
    safe = np.where(R0_t > 0.0, np.log(np.where(R0_t > 0.0, R0_t, 1.0)), -np.inf)
    return np.stack([safe[:, o].sum(axis=1) for o in orbits], axis=1)


# ----------------------------------------------------------------------------
# 6. Derived summaries
# ----------------------------------------------------------------------------
def vulnerable_window(p, threshold=0.05):
    """Longest contiguous (cyclic) run of days with p >= threshold.

    Parameters
    ----------
    p : 1-D array of length 365, index t <-> day-of-year t+1

    Returns
    -------
    dict with start_day, end_day, length (1-based day-of-year; length 0 if none,
    length 365 and start_day = 1 if the whole year qualifies).
    """
    p = np.asarray(p, dtype=float)
    P = p.size
    ok = p >= threshold
    if ok.all():
        return {"start_day": 1, "end_day": P, "length": int(P)}
    if not ok.any():
        return {"start_day": -1, "end_day": -1, "length": 0}

    # rotate so that a failing day sits at index 0, then take the single run
    first_bad = int(np.argmin(ok))
    rot = np.roll(ok, -first_bad)
    idx = np.where(rot)[0]
    # contiguous run starts after the leading block of False
    run_start = int(idx[0])
    run_end = int(idx[-1])                       # inclusive
    length = run_end - run_start + 1
    start_idx = (run_start + first_bad) % P
    end_idx = (run_end + first_bad) % P
    return {"start_day": int(start_idx + 1),
            "end_day": int(end_idx + 1),
            "length": int(length)}


def window_from_profile(p, threshold=0.05):
    """Alias kept for readability in the analysis scripts."""
    return vulnerable_window(p, threshold)
