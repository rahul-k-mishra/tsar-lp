"""
src/solvers.py
All 4 solver wrappers — same interface, fair comparison.
  solve_highs  → HiGHS-SAA  (scipy linprog, HiGHS backend)
  solve_glop   → OR-Tools GLOP-SAA
  solve_pdlp   → OR-Tools PDLP-SAA
  solve_fpdlp  → FuzzyPD-LP (our algorithm)
"""
import time
import numpy as np
from scipy.optimize import linprog
from src.core import oos_eval, solve_fpdlp as _core_fpdlp


# ═══════════════════════════════════════════════════════════════════
#  SOLVER 1: HiGHS-SAA
# ═══════════════════════════════════════════════════════════════════

def solve_highs(saa, test, calib, tlimit=120):
    """
    HiGHS-SAA: Solve SAA LP using scipy/HiGHS.
    Reference: Huangfu & Hall (2018) Math. Prog. Computation 10:119-142.
    """
    c, A, b, n = saa["c_avg"], saa["A_saa"], saa["b_saa"], saa["n"]
    t0 = time.perf_counter()

    try:
        res = linprog(c, A_ub=A, b_ub=b,
                      bounds=[(0, None)] * n,
                      method="highs",
                      options={"time_limit": tlimit, "disp": False})
        st = time.perf_counter() - t0
        x    = res.x if res.status == 0 else np.zeros(n)
        obj  = float(res.fun) if res.status == 0 else np.nan
        conv = res.status == 0
    except Exception:
        x = np.zeros(n); obj = np.nan; conv = False
        st = time.perf_counter() - t0

    ev = oos_eval(x, test["c"], test["A"], test["b"])
    return {"solver": "HiGHS-SAA", "x": x, "obj_nominal": obj,
            "solve_time": st, "converged": conv, "S": saa["S"], **ev}


# ═══════════════════════════════════════════════════════════════════
#  SOLVER 2: OR-Tools GLOP-SAA
# ═══════════════════════════════════════════════════════════════════

def solve_glop(saa, test, calib, tlimit=120):
    """
    GLOP-SAA: Google OR-Tools GLOP (revised simplex).
    Reference: OR-Tools documentation, Google LLC.
    """
    c, A, b, n = saa["c_avg"], saa["A_saa"], saa["b_saa"], saa["n"]
    t0 = time.perf_counter()

    try:
        from ortools.linear_solver import pywraplp
        sv = pywraplp.Solver.CreateSolver("GLOP")
        if sv is None: raise RuntimeError("GLOP not available")
        sv.set_time_limit(tlimit * 1000)

        xv  = [sv.NumVar(0, sv.infinity(), f"x{j}") for j in range(n)]
        obj = sv.Objective()
        for j in range(n): obj.SetCoefficient(xv[j], float(c[j]))
        obj.SetMinimization()

        norms = np.linalg.norm(A, axis=1)
        for i in range(len(b)):
            if norms[i] < 1e-12: continue
            ct = sv.Constraint(-sv.infinity(), float(b[i]))
            for j in np.nonzero(A[i])[0]:
                ct.SetCoefficient(xv[j], float(A[i, j]))

        status = sv.Solve(); st = time.perf_counter() - t0

        if status == pywraplp.Solver.OPTIMAL:
            x    = np.array([xv[j].solution_value() for j in range(n)])
            obj_v= float(sv.Objective().Value()); conv = True
        else:
            x = np.zeros(n); obj_v = np.nan; conv = False

    except Exception:
        # Fallback to HiGHS
        r = solve_highs(saa, test, calib, tlimit)
        r["solver"] = "OR-Tools GLOP-SAA"
        return r

    ev = oos_eval(x, test["c"], test["A"], test["b"])
    return {"solver": "OR-Tools GLOP-SAA", "x": x, "obj_nominal": obj_v,
            "solve_time": st, "converged": conv, "S": saa["S"], **ev}


# ═══════════════════════════════════════════════════════════════════
#  SOLVER 3: OR-Tools PDLP-SAA
# ═══════════════════════════════════════════════════════════════════

def solve_pdlp(saa, test, calib, tlimit=120):
    """
    PDLP-SAA: Google's first-order primal-dual LP solver.
    Reference: Applegate et al. (2021) NeurIPS — 'Practical Large-Scale LP'.
    """
    c, A, b, n = saa["c_avg"], saa["A_saa"], saa["b_saa"], saa["n"]
    t0 = time.perf_counter()

    try:
        from ortools.linear_solver import pywraplp
        sv = pywraplp.Solver.CreateSolver("PDLP")
        if sv is None:
            r = solve_highs(saa, test, calib, tlimit)
            r["solver"] = "OR-Tools PDLP-SAA"
            return r
        sv.set_time_limit(tlimit * 1000)

        xv  = [sv.NumVar(0, sv.infinity(), f"x{j}") for j in range(n)]
        obj = sv.Objective()
        for j in range(n): obj.SetCoefficient(xv[j], float(c[j]))
        obj.SetMinimization()

        norms = np.linalg.norm(A, axis=1)
        for i in range(len(b)):
            if norms[i] < 1e-12: continue
            ct = sv.Constraint(-sv.infinity(), float(b[i]))
            for j in np.nonzero(A[i])[0]:
                ct.SetCoefficient(xv[j], float(A[i, j]))

        status = sv.Solve(); st = time.perf_counter() - t0

        if status == pywraplp.Solver.OPTIMAL:
            x     = np.array([xv[j].solution_value() for j in range(n)])
            obj_v = float(sv.Objective().Value()); conv = True
        else:
            x = np.zeros(n); obj_v = np.nan; conv = False

    except Exception:
        r = solve_highs(saa, test, calib, tlimit)
        r["solver"] = "OR-Tools PDLP-SAA"
        return r

    ev = oos_eval(x, test["c"], test["A"], test["b"])
    return {"solver": "OR-Tools PDLP-SAA", "x": x, "obj_nominal": obj_v,
            "solve_time": st, "converged": conv, "S": saa["S"], **ev}


# ═══════════════════════════════════════════════════════════════════
#  SOLVER 4: FuzzyPD-LP (our algorithm)
# ═══════════════════════════════════════════════════════════════════

def solve_fpdlp(c_tfn, A_tfn, b_tfn, calib, test,
                max_iter=200_000, tol=1e-5, verbose=False):
    """
    FuzzyPD-LP: Fuzzy Chambolle-Pock with conformal tightening.
    Our algorithm — no SAA needed.
    """
    return _core_fpdlp(c_tfn, A_tfn, b_tfn, calib, test,
                        max_iter, tol, verbose)


# ═══════════════════════════════════════════════════════════════════
#  BASELINE 5: NOMINAL LP
#  Solves deterministic LP with MEAN coefficients only.
#  Ignores uncertainty entirely. Simplest possible baseline.
#  Shows the COST OF IGNORING UNCERTAINTY.
# ═══════════════════════════════════════════════════════════════════

def solve_nominal_lp(c_tfn, A_tfn, b_tfn, test, calib=None, tlimit=120):
    """
    Nominal LP: min E[c]ᵀx  s.t.  E[A]x ≤ E[b],  x ≥ 0
    Uses modal (mean) values of TFN — ignores all uncertainty.
    Reference baseline: shows what happens if you pretend uncertainty doesn't exist.
    """
    from scipy.optimize import linprog
    import time
    c_mean = c_tfn.mean()
    A_mean = A_tfn.mean()
    b_mean = b_tfn.mean()
    n = len(c_mean)
    t0 = time.perf_counter()
    res = linprog(c_mean, A_ub=A_mean, b_ub=b_mean,
                  bounds=[(0, None)]*n, method="highs",
                  options={"time_limit": tlimit, "disp": False})
    st = time.perf_counter() - t0
    x   = res.x if res.status == 0 else np.zeros(n)
    obj = float(res.fun) if res.status == 0 else np.nan
    ev  = oos_eval(x, test["c"], test["A"], test["b"])
    return {"solver": "Nominal-LP", "x": x, "obj_nominal": obj,
            "solve_time": st, "converged": res.status == 0,
            "S": "deterministic (mean only)", **ev}


# ═══════════════════════════════════════════════════════════════════
#  BASELINE 6: BERTSIMAS-SIM STANDALONE
#  The classic Γ-robust LP from Bertsimas & Sim (2004).
#  Uses γ=3 which gives theoretical coverage ≥ 1 - 1/9 = 88.9%.
#  Reference: Bertsimas & Sim (2004) Management Science 50(10).
# ═══════════════════════════════════════════════════════════════════

def solve_bertsimas_sim(c_tfn, A_tfn, b_tfn, test, calib=None,
                         gamma=3.0, tlimit=120):
    """
    Bertsimas-Sim Robust LP (γ=3.0 by default).

    min  (E[c] + γ·std[c])ᵀx
    s.t. (E[A] + γ·std[A]) x  ≤  E[b] - γ·std[b]
         x ≥ 0

    Theorem: P(all constraints satisfied) ≥ 1 - 1/γ²
    At γ=3.0: theoretical guarantee of P ≥ 88.9%.
    At γ=4.5: theoretical guarantee of P ≥ 95.1%.

    This is Phase 1 of TSAR-LP exposed as a standalone baseline.
    """
    from scipy.optimize import linprog
    import time
    c_risk   = c_tfn.mean() + gamma * c_tfn.std()
    A_robust = A_tfn.mean() + gamma * A_tfn.std()
    b_robust = b_tfn.mean() - gamma * b_tfn.std()
    n = len(c_risk)
    t0 = time.perf_counter()
    res = linprog(c_risk, A_ub=A_robust, b_ub=b_robust,
                  bounds=[(0, None)]*n, method="highs",
                  options={"time_limit": tlimit, "disp": False})
    st = time.perf_counter() - t0
    x   = res.x if res.status == 0 else np.zeros(n)
    obj = float(res.fun) if res.status == 0 else np.nan
    ev  = oos_eval(x, test["c"], test["A"], test["b"])
    calib_cov = float("nan")
    if calib is not None and x is not None:
        try:
            Ax_c = np.einsum("smn,n->sm", calib["A"], x)
            calib_cov = float((np.maximum(0, Ax_c - calib["b"]).max(1) <= 1e-6).mean())
        except Exception:
            pass
    return {"solver": f"Bertsimas-Sim(γ={gamma})", "x": x,
            "obj_nominal": obj, "solve_time": st,
            "converged": res.status == 0,
            "gamma_star": gamma,
            "calib_coverage": calib_cov,
            "S": f"parametric robust (γ={gamma})", **ev}


# ═══════════════════════════════════════════════════════════════════
#  BASELINE 7: CVaR-LP (Conditional Value-at-Risk)
#  Minimises CVaR of the objective at confidence level α.
#  Reformulated as LP using Rockafellar-Uryasev (2000) formula.
#  Reference: Rockafellar & Uryasev (2000) J Risk 2(3):21-41.
# ═══════════════════════════════════════════════════════════════════

def solve_cvar_lp(c_tfn, A_tfn, b_tfn, saa, test, calib=None,
                   alpha=0.95, tlimit=120):
    """
    CVaR-LP: minimise CVaR_α of objective under SAA scenarios.

    CVaR (Conditional Value-at-Risk) = expected loss in worst (1-α) fraction.
    At α=0.95: minimise the expected cost in the worst 5% of scenarios.

    Rockafellar-Uryasev LP reformulation:
      min  η + (1/(1-α)) · (1/S) · Σ_s u_s
      s.t. u_s ≥ c_s @ x - η       ∀ s  (auxiliary for tail expectation)
           u_s ≥ 0                  ∀ s
           A_saa x ≤ b_saa               (SAA feasibility)
           x ≥ 0

    This adds S auxiliary variables u_s and S+1 extra constraints.
    """
    from scipy.optimize import linprog
    import time
    import numpy as np

    c_s   = saa["c_s"]    # (S, n)
    A_s   = saa["A_s"]    # (S, m, n)
    b_s   = saa["b_s"]    # (S, m)
    S, m, n = A_s.shape
    t0 = time.perf_counter()

    # Variables: [x(n), η(1), u(S)]  total = n+1+S
    # Objective: η + 1/((1-α)S) · Σ u_s
    coeff = 1.0 / ((1.0 - alpha) * S)
    c_obj = np.concatenate([np.zeros(n), [1.0], np.full(S, coeff)])

    # Constraint 1: u_s ≥ c_s @ x - η  →  c_s @ x - η - u_s ≤ 0
    # Row for scenario s: [c_s, -1, -e_s]  ≤ 0
    n_cvar = n + 1 + S
    rows_cvar, rhs_cvar = [], []
    for s in range(S):
        row = np.zeros(n_cvar)
        row[:n]      = c_s[s]         # c_s @ x
        row[n]       = -1.0            # -η
        row[n+1+s]   = -1.0            # -u_s
        rows_cvar.append(row)
        rhs_cvar.append(0.0)

    # Constraint 2: SAA feasibility  A_saa x ≤ b_saa
    A_saa_full = np.zeros((S*m, n_cvar))
    A_saa_full[:, :n] = saa["A_saa"]
    b_saa_full = saa["b_saa"]

    # Stack all constraints
    A_all = np.vstack([np.array(rows_cvar), A_saa_full])
    b_all = np.concatenate([rhs_cvar, b_saa_full])

    # Bounds: x≥0, η free, u_s≥0
    bounds = ([(0, None)]*n) + [(None, None)] + ([(0, None)]*S)

    res = linprog(c_obj, A_ub=A_all, b_ub=b_all,
                  bounds=bounds, method="highs",
                  options={"time_limit": tlimit, "disp": False})
    st = time.perf_counter() - t0

    if res.status == 0:
        x   = res.x[:n]
        obj = float(c_tfn.mean() @ x)
        conv= True
    else:
        x   = np.zeros(n)
        obj = np.nan
        conv= False

    ev = oos_eval(x, test["c"], test["A"], test["b"])
    return {"solver": f"CVaR-LP(α={alpha})", "x": x,
            "obj_nominal": obj, "solve_time": st,
            "converged": conv, "S": S,
            "alpha_cvar": alpha, **ev}


# ═══════════════════════════════════════════════════════════════════
#  ABLATION VARIANTS OF TSAR-LP
#  Used to justify each component in the ablation study table.
# ═══════════════════════════════════════════════════════════════════

def solve_tsar_bs_only(c_tfn, A_tfn, b_tfn, calib, test,
                        max_iter=100_000, tol=1e-5, verbose=False,
                        cp_skip_n=9999, max_gb=7.25):
    """
    TSAR Ablation Variant 1: BS-only (Phase 1 only)
    Removes adversarial scenarios — uses only Bertsimas-Sim with
    data-driven gamma calibration. No Phase 2 or Phase 3.
    """
    import time
    from src.core import calibrate_gamma, oos_eval
    t0 = time.perf_counter()
    n = len(c_tfn.M)
    bounds = [(0, None)] * n
    gamma_star, x_bs, calib_cov = calibrate_gamma(
        c_tfn, A_tfn, b_tfn, calib,
        target_coverage=0.95, bounds=bounds)
    ev = oos_eval(x_bs, test["c"], test["A"], test["b"])
    c_var = c_tfn.variance()
    obj_mean = float(c_tfn.mean() @ x_bs)
    obj_std  = float(np.sqrt(max(0, float(c_var @ (x_bs**2)))))
    return {"solver": "TSAR-BS-only", "x": x_bs,
            "obj_nominal": float(c_tfn.M @ x_bs),
            "obj_mean": obj_mean, "obj_std": obj_std,
            "obj_lower95": obj_mean - 1.96*obj_std,
            "obj_upper95": obj_mean + 1.96*obj_std,
            "solve_time": time.perf_counter() - t0,
            "converged": True, "iters": 0,
            "gamma_star": gamma_star,
            "calib_coverage": calib_cov,
            "winner_phase": "BS_only",
            "S": f"BS-only(γ={gamma_star})", **ev}


def solve_tsar_adv_only(c_tfn, A_tfn, b_tfn, calib, test,
                         max_iter=100_000, tol=1e-5, verbose=False,
                         cp_skip_n=9999, max_gb=7.25):
    """
    TSAR Ablation Variant 2: Adv-only (Phase 2 only)
    Removes Bertsimas-Sim robust objective — uses only adversarial
    SAA with mean cost objective. No Phase 1 or Phase 3.
    """
    import time
    from src.core import adversarial_saa_solve, oos_eval
    t0 = time.perf_counter()
    n = len(c_tfn.M)
    bounds = [(0, None)] * n
    res_adv, S_adv = adversarial_saa_solve(
        c_tfn, A_tfn, b_tfn, S=100, seed=42,
        adv_fraction=0.5, bounds=bounds, max_gb=max_gb)
    x_adv = res_adv.x if res_adv.status == 0 else np.zeros(n)
    ev = oos_eval(x_adv, test["c"], test["A"], test["b"])
    c_var = c_tfn.variance()
    obj_mean = float(c_tfn.mean() @ x_adv)
    obj_std  = float(np.sqrt(max(0, float(c_var @ (x_adv**2)))))
    Ax_c = np.einsum("smn,n->sm", calib["A"], x_adv)
    calib_cov = float((np.maximum(0, Ax_c - calib["b"]).max(1) <= 1e-6).mean())
    return {"solver": "TSAR-Adv-only", "x": x_adv,
            "obj_nominal": float(c_tfn.M @ x_adv),
            "obj_mean": obj_mean, "obj_std": obj_std,
            "obj_lower95": obj_mean - 1.96*obj_std,
            "obj_upper95": obj_mean + 1.96*obj_std,
            "solve_time": time.perf_counter() - t0,
            "converged": True, "iters": 0,
            "gamma_star": np.nan,
            "calib_coverage": calib_cov,
            "winner_phase": "Adv_only",
            "S": f"AdvSAA-only(S={S_adv})", **ev}


def solve_tsar_no_calib(c_tfn, A_tfn, b_tfn, calib, test,
                         max_iter=100_000, tol=1e-5, verbose=False,
                         cp_skip_n=9999, max_gb=7.25):
    """
    TSAR Ablation Variant 3: No calibration
    Runs all three phases but IGNORES calibration for selection.
    Uses fixed γ=2.0 (no data-driven gamma selection).
    This tests whether the conformal calibration step adds value.
    """
    import time
    from src.core import bs_lp_solve, oos_eval
    t0 = time.perf_counter()
    n = len(c_tfn.M)
    bounds = [(0, None)] * n
    # Fixed γ=2.0 — no calibration, no adaptive selection
    res = bs_lp_solve(c_tfn, A_tfn, b_tfn, gamma=2.0, bounds=bounds)
    x = res.x if res.status == 0 else np.zeros(n)
    ev = oos_eval(x, test["c"], test["A"], test["b"])
    c_var = c_tfn.variance()
    obj_mean = float(c_tfn.mean() @ x)
    obj_std  = float(np.sqrt(max(0, float(c_var @ (x**2)))))
    Ax_c = np.einsum("smn,n->sm", calib["A"], x)
    calib_cov = float((np.maximum(0, Ax_c - calib["b"]).max(1) <= 1e-6).mean())
    return {"solver": "TSAR-No-calib", "x": x,
            "obj_nominal": float(c_tfn.M @ x),
            "obj_mean": obj_mean, "obj_std": obj_std,
            "obj_lower95": obj_mean - 1.96*obj_std,
            "obj_upper95": obj_mean + 1.96*obj_std,
            "solve_time": time.perf_counter() - t0,
            "converged": True, "iters": 0,
            "gamma_star": 2.0,
            "calib_coverage": calib_cov,
            "winner_phase": "fixed_γ=2.0",
            "S": "No-calibration(fixed γ=2.0)", **ev}
