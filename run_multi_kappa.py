"""
run_multi_kappa.py  (v5 — FINAL: crash-proof via subprocess isolation)
=======================================================================

WHY EVERY PREVIOUS VERSION CRASHED (real root cause, found in v4 log):

  Block 1 (cap41) printed solver times summing to ~57 s, yet block 2
  started 19.2 MINUTES later. The hidden 18 minutes is TFNArray.sample()
  in src/core.py — a pure-Python loop calling rng.triangular() once per
  matrix element:

      sample_test: 500 x (866x816) = 353,000,000 single Python rng calls
      build_saa  : 100 x (866x816) =  70,000,000 more

  per OR-Library block. During those ~19 minutes the process holds the
  2.83 GB test tensor plus ~2 GB of scipy/HiGHS copies on a 12 GB laptop.
  Sustained memory pressure + heap fragmentation -> Windows silently
  kills the process (native OOM / HiGHS crash, no Python traceback).
  That is why the crash point moved every run.

THE TWO STRUCTURAL FIXES IN v5 (no edits to your src/ files needed):

  FIX 1 — Vectorized TFN sampling (monkey-patch, applied at runtime):
      TFNArray.sample_batch = rng.triangular(L, M, U, size=(S,)+shape)
      353,000,000 Python calls -> 1 vectorized call.
      ~18 min/block -> ~3 seconds/block.
      NOTE: this changes the random stream, so kappa=0.20 feasibility
      values match the main experiment statistically (within a few %)
      rather than bit-exactly. The sanity check tolerance covers it.

  FIX 2 — Subprocess isolation:
      The orchestrator (this script, no args) runs each (instance, kappa)
      block in a FRESH child Python process:
        - fresh heap every block: zero fragmentation build-up
        - if HiGHS native code crashes, ONLY the child dies; the
          orchestrator logs it and continues to the next block
        - the child appends one CSV row AFTER EACH SOLVER, so a crash
          can never lose completed work
      The orchestrator itself never calls a solver, so it cannot crash.

ALSO IN v5:
  - Fixed CSV schema (csv.DictWriter) so appended rows can never
    misalign with the existing header; bulky 'x' solution vectors
    are dropped.
  - stdout forced to UTF-8 (no more cp1252 print crashes).
  - OR-Library processed FIRST; zero-feasibility instances skipped.
  - Full resume: already-saved (instance, kappa, solver) rows skipped.

HOW TO RUN (from D:\\Research\\fuzzypd_lp\\):
      python run_multi_kappa.py
  (PYTHONIOENCODING no longer required, but harmless if set.)

EXPECTED TIME (now that sampling is vectorized):
  OR-Library : 20 inst x 4 kappas x ~90-150 s  ~= 2.5-3 h (HiGHS dominates)
  MIPLIB     :  8 inst x 4 kappas x ~20 s      ~= 11 min
  NETLIB     : 10 inst x 4 kappas x ~10 s      ~=  7 min  (mostly TSAR fills)
  TOTAL      ~= 3 hours

OUTPUT: results/multi_kappa_results.csv  (same file, resume-compatible)
"""

import os, sys, gc, time, pickle, warnings, traceback, threading, subprocess
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Force UTF-8 stdout so unicode prints can never crash a child on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import config as CFG

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------

KAPPAS            = [0.05, 0.10, 0.20, 0.30]
S_TRAIN           = CFG.S_TRAIN          # 100
T_TEST            = CFG.T_TEST           # 500
TLIMIT            = CFG.SOLVER_TIME_LIM  # 120 s (HiGHS internal limit)
MAX_GB            = CFG.SAA_MAX_GB
MAX_VARS          = CFG.MAX_VARS
SOLVER_WALL_LIMIT = 180                  # per-solver wall clock inside child
BLOCK_TIMEOUT     = 900                  # orchestrator kills child after this

RESULTS_FILE = CFG.RESULTS_DIR / "multi_kappa_results.csv"
EXISTING_CSV = CFG.RESULTS_DIR / "all_results_live.csv"
CACHE_DIR    = CFG.DATA_DIR / "_mkcache"   # pickled instances for workers

SEED_TFN  = 42
SEED_TEST = 999
SEED_SAA  = 42

LBL_TSAR   = "TSAR-LP"
LBL_BS     = "Bertsimas-Sim(\u03b3=3.0)"
LBL_HIGHS  = "HiGHS-SAA"
LBL_NOM    = "Nominal-LP"
LBL_NOCAL  = "TSAR-No-calib"
ALL_LABELS = [LBL_TSAR, LBL_BS, LBL_HIGHS, LBL_NOM, LBL_NOCAL]

# Instances where EVERY solver scored feas=0.000 in the main experiment.
# They contribute nothing to the kappa-sensitivity figure and are the
# fastest to trigger native crashes, so they are excluded outright.
SKIP_ZERO_INSTANCES = {
    "2894", "gesa2", "klein1", "sctest",                    # NETLIB
    "2171", "lseu", "p0548",                                # MIPLIB
    "abilene",                                              # SNDlib
    "dataco_CASH", "dataco_DEBIT",
    "dataco_PAYMENT", "dataco_TRANSFER",                    # DataCo
}

# Clean schema used only if the results CSV does not exist yet.
NEW_FILE_COLUMNS = [
    "instance", "source", "n_vars", "n_cons", "kappa", "solver",
    "feasibility_rate", "infeasibility_rate",
    "mean_obj", "std_obj", "mean_violation", "max_violation", "n_test",
    "obj_nominal", "obj_mean", "obj_std", "obj_lower95", "obj_upper95",
    "gamma_star", "calib_coverage", "winner_phase", "S",
    "solve_time", "converged", "iters",
]


# ----------------------------------------------------------------------
# FIX 1: vectorized TFN sampling (monkey-patch; src/ files untouched)
# ----------------------------------------------------------------------

def apply_fast_sampling_patch():
    """
    Replace TFNArray.sample_batch's per-element Python loop with one
    vectorized numpy call. inject_tfn guarantees L < M < U elementwise
    (spread = |arr|*delta + 1e-8 > 0), which is exactly what
    numpy's triangular sampler requires.
    """
    from src.core import TFNArray

    def _fast_sample_batch(self, S, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.triangular(self.L, self.M, self.U,
                              size=(S,) + self.M.shape)

    TFNArray.sample_batch = _fast_sample_batch


# ----------------------------------------------------------------------
# Shared: CSV helpers (fixed schema -> rows can never misalign)
# ----------------------------------------------------------------------

def csv_fieldnames():
    """Existing header if the file exists, else the clean new schema."""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                first = f.readline()
            names = next(_csv.reader([first]), [])
            if names:
                return names
        except Exception:
            pass
    return list(NEW_FILE_COLUMNS)


def append_row(row, fieldnames):
    """Append one row; flush + fsync. Retries if the CSV is locked
    (e.g. open in Excel) so a completed solver result is never lost."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(6):
        try:
            is_new = not RESULTS_FILE.exists()
            with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore", restval="")
                if is_new:
                    w.writeheader()
                w.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            return
        except PermissionError as e:
            last_err = e
            print("      !  results CSV is locked (close it in Excel) - "
                  f"retry {attempt + 1}/6 in 5s", flush=True)
            time.sleep(5)
    raise last_err


def load_done_set():
    """(instance, kappa rounded to 3dp, solver) triples already saved."""
    if not RESULTS_FILE.exists():
        return set()
    try:
        df = pd.read_csv(RESULTS_FILE)
        return set(zip(df["instance"], df["kappa"].round(3), df["solver"]))
    except Exception:
        return set()


# ----------------------------------------------------------------------
# Wall-clock guard for a single solver call (inside the child)
# ----------------------------------------------------------------------

def run_with_wall_limit(fn, args, kwargs, limit_secs, label):
    result = [None]
    error  = [None]

    def _target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception:
            error[0] = traceback.format_exc()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=limit_secs)

    if t.is_alive():
        print(f"      !  {label}: wall limit {limit_secs}s exceeded",
              flush=True)
        return None
    if error[0]:
        print(f"      !  {label}: {error[0].splitlines()[-1][:90]}",
              flush=True)
        return None
    return result[0]


# ======================================================================
# WORKER MODE  —  runs ONE (instance, kappa) block in a fresh process
# ======================================================================

def worker_main(inst_name, kappa):
    apply_fast_sampling_patch()

    from src.core    import inject_tfn, build_saa, sample_test
    from src.solvers import (solve_highs, solve_nominal_lp,
                              solve_bertsimas_sim,
                              solve_tsar_bs_only, solve_tsar_no_calib)

    pkl_path = CACHE_DIR / f"{inst_name}.pkl"
    if not pkl_path.exists():
        print(f"  WORKER ERROR: cache missing for {inst_name}", flush=True)
        sys.exit(2)
    with open(pkl_path, "rb") as f:
        inst = pickle.load(f)

    name, source = inst["name"], inst["source"]
    nv, nc = inst["n_vars"], inst["n_ineq"]

    fieldnames = csv_fieldnames()
    done_set   = load_done_set()
    pending    = [lbl for lbl in ALL_LABELS
                  if (name, round(kappa, 3), lbl) not in done_set]
    if not pending:
        sys.exit(0)

    # --- TFN at this kappa (lo/hi ratio 0.25 preserved from main run) ---
    lo  = round(kappa * 0.25, 6)
    hi  = kappa
    rng = np.random.default_rng(SEED_TFN)
    c_tfn = inject_tfn(inst["c"],    lo, hi, rng)
    A_tfn = inject_tfn(inst["A_ub"], lo, hi, rng)
    b_tfn = inject_tfn(inst["b_ub"], lo, hi, rng)

    # --- scenarios (fast now thanks to FIX 1) ---------------------------
    all_t = sample_test(c_tfn, A_tfn, b_tfn,
                        T=T_TEST, seed=SEED_TEST, max_gb=MAX_GB)
    nc2   = max(10, len(all_t["c"]) // 5)
    calib = {k: v[:nc2] for k, v in all_t.items()}
    test  = {k: v[nc2:] for k, v in all_t.items()}

    meta = {"instance": name, "source": source,
            "n_vars": nv, "n_cons": nc, "kappa": kappa}

    def run_one(label, fn, *args, **kwargs):
        if label not in pending:
            return
        r = run_with_wall_limit(fn, args, kwargs,
                                SOLVER_WALL_LIMIT, label)
        if r is None:
            r = {"feasibility_rate": 0.0, "infeasibility_rate": 1.0,
                 "solve_time": float(SOLVER_WALL_LIMIT),
                 "converged": False}
        r.pop("x", None)            # never write solution vectors
        r["solver"] = label         # normalise name -> resume-safe
        fr = r.get("feasibility_rate", float("nan"))
        st = r.get("solve_time", 0.0)
        gs = r.get("gamma_star", "")
        gtxt = f"  gamma*={gs}" if gs not in ("", None) and gs == gs else ""
        print(f"      {label:<32}  feas={fr:.3f}  t={st:.1f}s{gtxt}",
              flush=True)
        append_row({**meta, **r}, fieldnames)   # saved IMMEDIATELY

    # Solver order: cheap + important first, HiGHS (heavy) last.
    run_one(LBL_TSAR,  solve_tsar_bs_only, c_tfn, A_tfn, b_tfn,
            calib, test, max_gb=MAX_GB)
    run_one(LBL_BS,    solve_bertsimas_sim, c_tfn, A_tfn, b_tfn,
            test, calib, gamma=3.0)
    run_one(LBL_NOM,   solve_nominal_lp, c_tfn, A_tfn, b_tfn, test)
    run_one(LBL_NOCAL, solve_tsar_no_calib, c_tfn, A_tfn, b_tfn,
            calib, test, max_gb=MAX_GB)

    if LBL_HIGHS in pending:
        saa = build_saa(c_tfn, A_tfn, b_tfn,
                        S=S_TRAIN, seed=SEED_SAA, max_gb=MAX_GB)
        run_one(LBL_HIGHS, solve_highs, saa, test, calib, TLIMIT)
        del saa

    sys.exit(0)


# ======================================================================
# ORCHESTRATOR MODE  —  loads data once, spawns one child per block
# ======================================================================

def load_filter_order_instances():
    from src.datasets import (download_netlib_miplib, download_orlib,
                              download_sndlib, load_dataco)
    all_inst = []

    print("\n[NETLIB + MIPLIB]")
    nl, ml = download_netlib_miplib(
        CFG.DATA_DIR / "netlib", CFG.NETLIB_NAMES, verbose=True)
    all_inst.extend(nl); all_inst.extend(ml)
    print(f"  NETLIB: {len(nl)}  MIPLIB: {len(ml)}")

    print("\n[OR-Library]")
    ol = download_orlib(CFG.DATA_DIR / "orlib", CFG.ORLIB_NAMES, verbose=True)
    all_inst.extend(ol)
    print(f"  OR-Library: {len(ol)}")

    print("\n[SNDlib]")
    sl = download_sndlib(CFG.DATA_DIR / "sndlib",
                         CFG.SNDLIB_SPECS, max_vars=MAX_VARS, verbose=True)
    all_inst.extend(sl)
    print(f"  SNDlib: {len(sl)}")

    print("\n[DataCo]")
    dc = load_dataco(CFG.DATA_DIR / "dataco",
                     CFG.DATACO_ALL_COLS, CFG.DATACO_NUMERIC_COLS,
                     CFG.DATACO_STATS)
    all_inst.extend(dc)
    print(f"  DataCo: {len(dc)}")

    all_inst = [i for i in all_inst
                if 4 <= i["n_vars"] <= MAX_VARS and i["n_ineq"] >= 2]

    if EXISTING_CSV.exists():
        names = set()
        with open(EXISTING_CSV, encoding="utf-8") as f:
            for row in _csv.reader(f):
                if row and row[0] != "instance":
                    names.add(row[0])
        all_inst = [i for i in all_inst if i["name"] in names]
        print(f"\n  After filter to existing : {len(all_inst)}")

    before = len(all_inst)
    all_inst = [i for i in all_inst if i["name"] not in SKIP_ZERO_INSTANCES]
    print(f"  After skip zero-feas     : {len(all_inst)} "
          f"(removed {before - len(all_inst)} crash-prone)")

    order = {"OR-Library": 0, "MIPLIB": 1, "NETLIB": 2,
             "SNDlib": 3, "DataCo": 4}
    all_inst.sort(key=lambda i: (order.get(i["source"], 9), i["name"]))
    return all_inst


def pickle_instances(instances):
    """
    Write each instance's LP data to a pickle the workers load in ~0.2 s,
    then drop the heavy arrays from the orchestrator's memory.
    Returns a lightweight metadata list.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_list = []
    for inst in instances:
        p = CACHE_DIR / f"{inst['name']}.pkl"
        if not p.exists():
            with open(p, "wb") as f:
                pickle.dump({
                    "name":   inst["name"],
                    "source": inst["source"],
                    "n_vars": inst["n_vars"],
                    "n_ineq": inst["n_ineq"],
                    "c":      np.asarray(inst["c"],    dtype=float),
                    "A_ub":   np.asarray(inst["A_ub"], dtype=float),
                    "b_ub":   np.asarray(inst["b_ub"], dtype=float),
                }, f, protocol=4)
        meta_list.append({"name": inst["name"], "source": inst["source"],
                          "n_vars": inst["n_vars"],
                          "n_ineq": inst["n_ineq"]})
    instances.clear()
    gc.collect()
    return meta_list


def orchestrator_main():
    print("=" * 66)
    print("  TSAR-LP  -  Multi-Kappa Sensitivity Experiment  (v5 FINAL)")
    print(f"  kappas       : {KAPPAS}   (0.20 = main experiment)")
    print("  Architecture : subprocess-per-block (native crashes isolated)")
    print("  Sampling     : vectorized triangular (was the 18-min/block bug)")
    print(f"  Wall limits  : {SOLVER_WALL_LIMIT}s/solver, "
          f"{BLOCK_TIMEOUT}s/block")
    print("=" * 66)

    print("\nSTEP 1 - Loading instances (cached files, no re-download)")
    instances = load_filter_order_instances()
    if not instances:
        print("ERROR: no instances loaded."); sys.exit(1)

    print("\nSTEP 2 - Caching instances for workers")
    meta = pickle_instances(instances)
    by_src = {}
    for m in meta:
        by_src[m["source"]] = by_src.get(m["source"], 0) + 1
    for src, n in by_src.items():
        print(f"    {src:12s}: {n} instances")

    done_set = load_done_set()
    blocks = [(m, k) for m in meta for k in KAPPAS]
    total  = len(blocks)
    full_done = sum(
        1 for m, k in blocks
        if all((m["name"], round(k, 3), lbl) in done_set
               for lbl in ALL_LABELS)
    )
    print(f"\n  Blocks: {total}  ({full_done} complete, "
          f"{total - full_done} remaining)")
    print(f"  Saved rows: {len(done_set)}")

    print("\nSTEP 3 - Running (one fresh child process per block)")
    print("-" * 66)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t_global = time.time()
    crashed_blocks = []

    for idx, (m, kappa) in enumerate(blocks, 1):
        name = m["name"]
        elapsed = (time.time() - t_global) / 60

        if all((name, round(kappa, 3), lbl) in done_set
               for lbl in ALL_LABELS):
            print(f"  [{idx:3d}/{total}] SKIP  {m['source']}/{name}"
                  f"  kappa={kappa:.2f}", flush=True)
            continue

        print(f"\n  [{idx:3d}/{total}]  {m['source']}/{name}"
              f"  n={m['n_vars']}  m={m['n_ineq']}"
              f"  kappa={kappa:.2f}  ({elapsed:.1f} min)", flush=True)

        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--worker", name, f"{kappa}"],
                cwd=str(ROOT), env=env, timeout=BLOCK_TIMEOUT)
            if proc.returncode != 0:
                print(f"      !! child exited code {proc.returncode} "
                      f"(crash isolated; completed solvers were already "
                      f"saved). Continuing.", flush=True)
                crashed_blocks.append((name, kappa, proc.returncode))
        except subprocess.TimeoutExpired:
            print(f"      !! child exceeded {BLOCK_TIMEOUT}s - killed. "
                  f"Continuing.", flush=True)
            crashed_blocks.append((name, kappa, "timeout"))

        # refresh done_set (child wrote rows directly to the CSV)
        done_set = load_done_set()

    # ------------------------------------------------------------------
    total_elapsed = (time.time() - t_global) / 60
    print(f"\n{'=' * 66}")
    print(f"  DONE in {total_elapsed:.1f} min  ->  {RESULTS_FILE}")

    if crashed_blocks:
        print(f"\n  Blocks with isolated child crashes "
              f"({len(crashed_blocks)}):")
        for nme, k, rc in crashed_blocks:
            print(f"    {nme}  kappa={k}  ({rc})")
        print("  Re-run this script to retry any missing solver rows.")

    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
        df_c = df[df["solver"].isin(ALL_LABELS)]
        print(f"\n  Clean rows: {len(df_c)}")
        if not df_c.empty:
            print("\n  Mean feasibility (solver x kappa):")
            pivot = (df_c.groupby(["solver", "kappa"])["feasibility_rate"]
                     .mean().unstack().reindex(ALL_LABELS))
            print(pivot.round(3).to_string())

        if EXISTING_CSV.exists():
            mdf = pd.read_csv(EXISTING_CSV)
            mm = mdf["solver"].isin(["FuzzyPD-LP", "TSAR-BS-only",
                                     "TSAR-LP"])
            main_f = mdf[mm]["feasibility_rate"].mean() if mm.any() \
                     else float("nan")
            mk = df_c[(df_c["solver"] == LBL_TSAR) &
                      (df_c["kappa"].round(2) == 0.20)
                      ]["feasibility_rate"].mean()
            ok = "OK" if abs(main_f - mk) < 0.08 else "CHECK"
            print(f"\n  Sanity kappa=0.20 vs main: main={main_f:.3f}  "
                  f"multi-kappa={mk:.3f}  [{ok}]")
            print("  (Vectorized sampling uses a different random stream,")
            print("   so values match statistically, not bit-exactly.)")
    print("=" * 66)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], float(sys.argv[3]))
    else:
        orchestrator_main()
