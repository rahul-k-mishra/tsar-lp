"""
run.py — FuzzyPD-LP Benchmark Runner (32GB Full-Throttle)
=========================================================
Usage:
  python run.py                          # full run, all datasets
  python run.py --fast                   # quick test (~5 min)
  python run.py --datasets netlib,orlib  # specific tiers only
  python run.py --workers 4              # 4 parallel CPU cores
  python run.py --dataco path/to/file.csv

All results → D:\\Research\\fuzzypd_lp\\results\\
Progressive save: results are saved after EACH instance (crash-safe)
"""
import os, sys, gc, time, json, argparse, warnings, traceback
import csv as _csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import config as CFG
from src.core    import inject_tfn, build_saa, sample_test, solve_fpdlp
from src.solvers import solve_highs, solve_glop, solve_pdlp
from src.reporter import (compile_df, aggregate, print_table,
                            print_tier_table, print_significance,
                            generate_latex, save_results, generate_plots)
from src.datasets import (download_netlib_miplib, download_orlib,
                            download_sndlib, load_dataco)


# ── CLI ───────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="FuzzyPD-LP Benchmark")
    ap.add_argument("--fast",     action="store_true")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--workers",  type=int, default=CFG.N_WORKERS)
    ap.add_argument("--S",        type=int, default=None)
    ap.add_argument("--T",        type=int, default=None)
    ap.add_argument("--max_vars", type=int, default=None)
    ap.add_argument("--dataco",   type=str, default=None)
    ap.add_argument("--verbose",  action="store_true")
    return ap.parse_args()


# ── Progressive CSV saver (crash-safe) ────────────────────────────
class ProgressiveSaver:
    def __init__(self, path):
        self.path   = Path(path)
        self.first  = not self.path.exists()
        self._fp    = open(self.path, "a", newline="", encoding="utf-8")
        self._fields= None
        self.count  = 0

    def write(self, record):
        safe = {k: (v if not hasattr(v, "tolist") else v.tolist())
                for k, v in record.items() if k != "x"}
        if self._fields is None:
            self._fields = list(safe.keys())
            self._writer = _csv.DictWriter(
                self._fp, fieldnames=self._fields, extrasaction="ignore")
            if self.first: self._writer.writeheader()
        self._writer.writerow(safe)
        self._fp.flush()
        self.count += 1

    def close(self): self._fp.close()


# ── Run one instance ──────────────────────────────────────────────
def run_instance(inst, params):
    # ALL solver imports inside function — required for thread safety
    from src.solvers import (solve_highs, solve_glop, solve_pdlp,
                              solve_nominal_lp, solve_bertsimas_sim, solve_cvar_lp,
                              solve_tsar_bs_only, solve_tsar_adv_only, solve_tsar_no_calib)
    from src.core    import solve_fpdlp
    import gc

    S_TRAIN    = params["S_TRAIN"]
    T_TEST     = params["T_TEST"]
    TLIMIT     = params["TLIMIT"]
    FPDLP_IT   = params["FPDLP_IT"]
    FPDLP_TOL  = params["FPDLP_TOL"]
    VERBOSE    = params["VERBOSE"]
    MAX_GB     = params["MAX_GB"]
    CP_SKIP_N  = params["CP_SKIP_N"]

    name, source = inst["name"], inst["source"]
    nv, nc       = inst["n_vars"], inst["n_ineq"]
    meta         = {"instance": name, "source": source,
                    "n_vars": nv, "n_cons": nc}

    try:
        rng   = np.random.default_rng(42)
        c_tfn = inject_tfn(inst["c"],    0.05, 0.20, rng)
        A_tfn = inject_tfn(inst["A_ub"], 0.05, 0.20, rng)
        b_tfn = inject_tfn(inst["b_ub"], 0.05, 0.20, rng)

        # Build SAA (auto-reduces S if needed for memory)
        saa   = build_saa(c_tfn, A_tfn, b_tfn,
                           S=S_TRAIN, seed=42, max_gb=MAX_GB)

        # Sample test + calibration (auto-reduces T if needed for memory)
        all_t = sample_test(c_tfn, A_tfn, b_tfn,
                             T=T_TEST, seed=999, max_gb=MAX_GB)
        nc2   = max(10, len(all_t["c"]) // 5)
        calib = {k: v[:nc2] for k, v in all_t.items()}
        test  = {k: v[nc2:] for k, v in all_t.items()}

        results = []

        def run_solver(label, fn, *args, **kw):
            r = fn(*args, **kw)
            fr = r.get("feasibility_rate", float("nan"))
            st = r.get("solve_time", 0)
            # gamma_star shown for TSAR-LP and Bertsimas-Sim variants
            # calib_coverage only exists for TSAR-LP (not standalone BS)
            if "gamma_star" in r:
                cc = r.get("calib_coverage", float("nan"))
                cc_str = f"{cc:.2f}" if cc == cc else "n/a"  # nan check
                g = f"  γ*={r['gamma_star']}  calib_cov={cc_str}"
            else:
                g = ""
            ph = f"  [{r.get('winner_phase','?')}]" if "winner_phase" in r else ""
            print(f"    {label:<26} feas={fr:.3f}  t={st:.3f}s{g}{ph}")
            results.append({**meta, **r})

        run_solver("Nominal-LP",
                   solve_nominal_lp, c_tfn, A_tfn, b_tfn, test)
        run_solver("HiGHS-SAA",
                   solve_highs, saa, test, calib, TLIMIT)
        run_solver("OR-Tools GLOP-SAA",
                   solve_glop,  saa, test, calib, TLIMIT)
        run_solver("OR-Tools PDLP-SAA",
                   solve_pdlp,  saa, test, calib, TLIMIT)
        run_solver("Bertsimas-Sim(gamma=3)",
                   solve_bertsimas_sim, c_tfn, A_tfn, b_tfn, test,
                   gamma=3.0)
        run_solver("CVaR-LP(a=0.95)",
                   solve_cvar_lp, c_tfn, A_tfn, b_tfn, saa, test,
                   alpha=0.95)
        # Free SAA before TSAR-LP -- halves peak memory per worker
        del saa; gc.collect()
        run_solver("TSAR-LP (ours)",
                   solve_fpdlp, c_tfn, A_tfn, b_tfn,
                   calib, test, FPDLP_IT, FPDLP_TOL, VERBOSE,
                   cp_skip_n=CP_SKIP_N)
        # ── Ablation variants ──────────────────────────────────────
        run_solver("TSAR-BS-only (ablation)",
                   solve_tsar_bs_only, c_tfn, A_tfn, b_tfn,
                   calib, test)
        run_solver("TSAR-Adv-only (ablation)",
                   solve_tsar_adv_only, c_tfn, A_tfn, b_tfn,
                   calib, test)
        run_solver("TSAR-No-calib (ablation)",
                   solve_tsar_no_calib, c_tfn, A_tfn, b_tfn,
                   calib, test)

        del all_t, calib, test, c_tfn, A_tfn, b_tfn
        gc.collect()
        return results, None

    except Exception as e:
        return [], traceback.format_exc()


# ── Main ──────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    FAST     = args.fast
    DATASETS = [d.strip() for d in args.datasets.split(",")]
    if "all" in DATASETS:
        DATASETS = ["netlib", "miplib", "orlib", "sndlib", "dataco"]

    S_TRAIN   = args.S or (20  if FAST else CFG.S_TRAIN)
    T_TEST    = args.T or (100 if FAST else CFG.T_TEST)
    MAX_VARS  = args.max_vars or (500 if FAST else CFG.MAX_VARS)
    N_WORKERS = args.workers
    VERBOSE   = args.verbose
    MAX_INST  = 5 if FAST else 999

    params = {
        "S_TRAIN":   S_TRAIN,
        "T_TEST":    T_TEST,
        "TLIMIT":    CFG.SOLVER_TIME_LIM,
        "FPDLP_IT":  50_000 if FAST else CFG.FPDLP_MAX_ITER,
        "FPDLP_TOL": CFG.FPDLP_TOL,
        "VERBOSE":   VERBOSE,
        "MAX_GB":    CFG.SAA_MAX_GB,
        "CP_SKIP_N": CFG.CP_SKIP_N,
    }

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   FuzzyPD-LP vs HiGHS-SAA vs GLOP-SAA vs PDLP-SAA           ║
║   Full Real-Data Benchmark — 5 Tiers                         ║
╚══════════════════════════════════════════════════════════════╝
  Datasets   : {DATASETS}
  S_TRAIN    : {S_TRAIN}   T_TEST : {T_TEST}
  MAX_VARS   : {MAX_VARS}  Workers: {N_WORKERS}
  RAM budget : {CFG.SAA_MAX_GB:.2f} GB/worker  ({CFG.RAM_TOTAL_GB}GB total)
  CP_SKIP_N  : {CFG.CP_SKIP_N} (CP skipped for n>{CFG.CP_SKIP_N})
  Results    : {CFG.RESULTS_DIR}
""")

    # ── Step 1: Load datasets ─────────────────────────────────────
    print("─"*60 + "\nSTEP 1 — Downloading datasets\n" + "─"*60)

    all_instances = []

    if "netlib" in DATASETS or "miplib" in DATASETS:
        print("\n[NETLIB + MIPLIB]")
        nl, ml = download_netlib_miplib(
            CFG.DATA_DIR / "netlib", CFG.NETLIB_NAMES, verbose=True)
        if "netlib" in DATASETS:
            all_instances.extend(nl[:MAX_INST])
            print(f"  ✓ NETLIB: {len(nl)} instances")
        if "miplib" in DATASETS:
            all_instances.extend(ml[:MAX_INST])
            print(f"  ✓ MIPLIB: {len(ml)} LP relaxations")

    if "orlib" in DATASETS:
        print("\n[OR-Library]")
        ol = download_orlib(CFG.DATA_DIR / "orlib",
                             CFG.ORLIB_NAMES, verbose=True)
        all_instances.extend(ol[:MAX_INST])
        print(f"  ✓ OR-Library: {len(ol)} CFL instances")

    if "sndlib" in DATASETS:
        print("\n[SNDlib]")
        sl = download_sndlib(CFG.DATA_DIR / "sndlib",
                              CFG.SNDLIB_SPECS,
                              max_vars=MAX_VARS, verbose=True)
        all_instances.extend(sl[:MAX_INST])
        print(f"  ✓ SNDlib: {len(sl)} MCF instances")

    if "dataco" in DATASETS:
        print("\n[DataCo]")
        if args.dataco:
            src = Path(args.dataco)
            dst = CFG.DATA_DIR / "dataco" / src.name
            if not dst.exists():
                import shutil; shutil.copy(src, dst)
        dc = load_dataco(CFG.DATA_DIR / "dataco",
                          CFG.DATACO_ALL_COLS, CFG.DATACO_NUMERIC_COLS,
                          CFG.DATACO_STATS, max_insts=MAX_INST)
        all_instances.extend(dc)
        print(f"  ✓ DataCo: {len(dc)} supply chain instances")

    all_instances = [i for i in all_instances
                     if 4 <= i["n_vars"] <= MAX_VARS and i["n_ineq"] >= 2]
    # Resume: skip done
    import csv as _csv
    _lcsv=CFG.RESULTS_DIR/"all_results_live.csv"
    _done=set()
    if _lcsv.exists():
        [_done.add(r[0]) for r in _csv.reader(open(_lcsv,encoding="utf-8")) if r and r[0]!="instance"]
    if _done:
        _b=len(all_instances);all_instances=[i for i in all_instances if i["name"] not in _done];print(f"  Resuming: skipping {_b-len(all_instances)} done, {len(all_instances)} remaining")
    print(f"\n  Total instances: {len(all_instances)}")

    if not all_instances:
        print("  ERROR: No instances loaded."); sys.exit(1)

    # ── Step 2: Run solvers ───────────────────────────────────────
    print("\n" + "─"*60 + "\nSTEP 2 — Running all 4 solvers\n" + "─"*60)

    saver   = ProgressiveSaver(CFG.RESULTS_DIR / "all_results_live.csv")
    records = []
    n_total = len(all_instances)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_instance, inst, params): (idx, inst)
            for idx, inst in enumerate(all_instances)
        }

        for future in as_completed(futures):
            idx, inst = futures[future]
            elapsed   = time.time() - t_start
            name, source = inst["name"], inst["source"]
            nv, nc = inst["n_vars"], inst["n_ineq"]

            print(f"\n[{idx+1}/{n_total}] {source}/{name}"
                  f"  n={nv}  m={nc}"
                  f"  ({elapsed/60:.1f}min)")

            try:
                recs, err = future.result()
                if err:
                    print(f"  ERROR:\n{err}")
                    continue
                for r in recs:
                    records.append(r)
                    saver.write(r)
            except Exception as e:
                print(f"  EXCEPTION: {e}")

    saver.close()
    print(f"\n  Total records: {len(records)}")

    if not records:
        print("  ERROR: No results."); sys.exit(1)

    # ── Step 3: Report ────────────────────────────────────────────
    print("\n" + "─"*60 + "\nSTEP 3 — Generating report\n" + "─"*60)

    df  = compile_df(records)
    agg = aggregate(df)

    print_table(agg)
    print_tier_table(df)
    print_significance(df)

    save_results(df, agg, records, CFG.RESULTS_DIR)
    generate_latex(agg, CFG.RESULTS_DIR / "table_comparison.tex")
    generate_plots(df, agg, CFG.RESULTS_DIR)

    # ── Final summary ─────────────────────────────────────────────
    fpdlp     = agg[agg["solver"] == "FuzzyPD-LP"]
    fpdlp_feas= fpdlp["mean_feasibility"].values[0] if len(fpdlp) else float("nan")
    best_row  = agg.loc[agg["mean_feasibility"].idxmax()]

    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  EXPERIMENT COMPLETE                                         ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    for _, r in agg.iterrows():
        flag = " ← BEST" if r["solver"] == best_row["solver"] else ""
        print(f"║  {r['solver']:<28} feas={r['mean_feasibility']:.4f}{flag:<12}║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  FuzzyPD-LP feasibility : {fpdlp_feas:.4f}                          ║")
    print(f"║  Results → {str(CFG.RESULTS_DIR)[:48]:<48}║")
    print(f"╚══════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
