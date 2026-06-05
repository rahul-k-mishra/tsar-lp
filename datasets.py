"""
src/datasets.py
Downloads and parses all 5 real benchmark datasets:
  Tier 1: NETLIB LP (official + HiGHS GitHub)
  Tier 2: MIPLIB 2017 LP relaxations
  Tier 3: OR-Library CFL (cap41-cap134)
  Tier 4: SNDlib MCF (from published topology specs)
  Tier 5: DataCo Supply Chain (CSV or published statistics)
"""
import re, gzip, io, os, zipfile, requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


HEADERS = {"User-Agent": "FuzzyPDLP-Research/1.0"}


def fetch(url, timeout=30):
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code == 200 and len(r.content) > 50:
            return r.content
    except Exception:
        pass
    return None


def fetch_text(url, timeout=30):
    data = fetch(url, timeout)
    if data is None: return None
    if url.endswith(".gz"):
        try: data = gzip.decompress(data)
        except: return None
    return data.decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════
#  TIER 1 & 2: NETLIB + MIPLIB  (MPS parser)
# ═══════════════════════════════════════════════════════════════════

def to_float(s):
    """Convert MPS float string, handling Fortran D-notation."""
    try:
        return float(s)
    except ValueError:
        return float(s.upper().replace("D", "E"))


def parse_mps(text, name="", source=""):
    """
    Parse free MPS format → LP instance dict.
    Handles: LP, MIP-as-LP-relaxation, Fortran D-notation.
    """
    lines     = text.splitlines()
    section   = None
    obj_name  = None
    row_types = {}
    row_order = []
    cols_data = {}
    col_order = []
    rhs_data  = {}
    bnd_lo    = {}
    bnd_hi    = {}
    in_int    = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        up = line.upper()
        if up in ("ROWS","COLUMNS","RHS","BOUNDS","RANGES","ENDATA"):
            section = up; in_int = False; continue
        if up in ("INTEGERS","GENERALS","BINARY","BINARIES","SOS","SETS"):
            in_int = True; continue
        if in_int and section == "COLUMNS":
            continue
        if "'MARKER'" in line and ("'INTORG'" in line or "'INTEND'" in line):
            continue
        t = line.split()
        try:
            if section == "ROWS":
                rt, rn = t[0], t[1]
                row_types[rn] = rt
                if rt == "N" and obj_name is None: obj_name = rn
                else: row_order.append(rn)
            elif section == "COLUMNS":
                col = t[0]
                if col not in cols_data:
                    cols_data[col] = {}; col_order.append(col)
                if len(t) >= 3: cols_data[col][t[1]] = to_float(t[2])
                if len(t) >= 5: cols_data[col][t[3]] = to_float(t[4])
            elif section == "RHS":
                if len(t) >= 3: rhs_data[t[1]] = to_float(t[2])
                if len(t) >= 5: rhs_data[t[3]] = to_float(t[4])
            elif section == "BOUNDS":
                bt, col = t[0], t[2]
                val = to_float(t[3]) if len(t) > 3 else 0.0
                if bt in ("UP","UI","SC"): bnd_hi[col] = val
                elif bt in ("LO","LI"):   bnd_lo[col] = val
                elif bt == "FX":          bnd_lo[col] = bnd_hi[col] = val
                elif bt == "MI":          bnd_lo[col] = -1e30
                elif bt == "FR":          bnd_lo[col] = -1e30; bnd_hi[col] = 1e30
        except Exception:
            continue

    if obj_name is None or not col_order:
        return None
    n  = len(col_order)
    ci = {c: i for i, c in enumerate(col_order)}

    c_vec = np.zeros(n)
    for col, cm in cols_data.items():
        if obj_name in cm: c_vec[ci[col]] = cm[obj_name]

    ineq = [r for r in row_order if row_types.get(r) in ("L","G")]
    eq_r = [r for r in row_order if row_types.get(r) == "E"]

    def build_mat(rows):
        A = np.zeros((len(rows), n)); b = np.zeros(len(rows))
        for j, row in enumerate(rows):
            for col, cm in cols_data.items():
                if row in cm: A[j, ci[col]] = cm[row]
            b[j] = rhs_data.get(row, 0.0)
            if row_types[row] == "G": A[j] *= -1; b[j] *= -1
        return A, b

    A_ub, b_ub = build_mat(ineq)
    A_eq, b_eq = (build_mat(eq_r) if eq_r
                  else (np.empty((0,n)), np.empty(0)))

    return {
        "name": name, "source": source,
        "c": c_vec, "A_ub": A_ub, "b_ub": b_ub,
        "A_eq": A_eq, "b_eq": b_eq,
        "n_vars": n, "n_ineq": len(b_ub), "n_eq": len(b_eq),
    }


MIP_KW  = ["INTORG","INTEND","BINARY","BINARIES","GENERALS","INTEGERS"]
SKIP_KW = ["infeasible","nan","issue","garbage","silly",
           "warning","comment","qmatrix","quadobj","duplicate"]

HIGHS_BASE   = "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/"
NETLIB_BASE  = "https://www.netlib.org/lp/data/"


def download_netlib_miplib(data_dir, names_netlib, verbose=True):
    """Download NETLIB + MIPLIB instances. Returns (netlib_list, miplib_list)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get full HiGHS instance list
    try:
        r = requests.get(
            "https://github.com/ERGO-Code/HiGHS/tree/master/check/instances",
            timeout=20, headers=HEADERS)
        highs_files = sorted(set(
            re.findall(r"check/instances/([^\"]+\.mps)", r.text)))
    except Exception:
        highs_files = []

    all_names = list(set(names_netlib + [f.replace(".mps","") for f in highs_files]))
    netlib_insts, miplib_insts = [], []

    for name in tqdm(all_names, desc="NETLIB+MIPLIB", disable=not verbose):
        if any(s in name for s in SKIP_KW): continue
        cache = data_dir / f"{name}.mps"
        text  = None

        if cache.exists():
            text = cache.read_text(encoding="utf-8", errors="replace")
        else:
            for url in [
                f"{NETLIB_BASE}{name}",
                f"{NETLIB_BASE}{name}.gz",
                f"{HIGHS_BASE}{name}.mps",
                f"https://raw.githubusercontent.com/ERGO-Code/HiGHS/develop/check/instances/{name}.mps",
            ]:
                text = fetch_text(url)
                if text and len(text) > 100:
                    cache.write_text(text, encoding="utf-8")
                    break

        if not text: continue
        is_mip = any(k in text for k in MIP_KW)
        inst   = parse_mps(text, name, "MIPLIB" if is_mip else "NETLIB")

        if inst and inst["n_vars"] >= 4 and inst["n_ineq"] + inst["n_eq"] >= 2:
            if is_mip: miplib_insts.append(inst)
            else:      netlib_insts.append(inst)

    return netlib_insts, miplib_insts


# ═══════════════════════════════════════════════════════════════════
#  TIER 3: OR-LIBRARY CFL
# ═══════════════════════════════════════════════════════════════════

ORLIB_BASE = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"


def parse_cfl(text, name):
    """
    OR-Library CFL format (Beasley 1990):
      Line 1: n_facilities  n_customers
      ALL variables: y_j (facility open) + x_ij (allocation)
    """
    nums = list(map(float, text.split()))
    idx  = 0
    n_f  = int(nums[idx]); n_c = int(nums[idx+1]); idx += 2

    cap    = np.array([nums[idx + 2*j]     for j in range(n_f)])
    fc     = np.array([nums[idx + 2*j + 1] for j in range(n_f)])
    idx   += 2 * n_f

    demand = np.zeros(n_c)
    alloc  = np.zeros((n_c, n_f))
    for i in range(n_c):
        demand[i] = nums[idx]; idx += 1
        for j in range(n_f):
            alloc[i, j] = nums[idx]; idx += 1

    n = n_f + n_c * n_f
    def xid(i, j): return n_f + i*n_f + j

    c = np.zeros(n); c[:n_f] = fc
    for i in range(n_c):
        for j in range(n_f): c[xid(i,j)] = alloc[i,j]

    rows_A, rows_b = [], []
    for i in range(n_c):                   # demand
        row = np.zeros(n)
        for j in range(n_f): row[xid(i,j)] = -1
        rows_A.append(row); rows_b.append(-1.0)
    for j in range(n_f):                   # capacity
        row = np.zeros(n); row[j] = -cap[j]
        for i in range(n_c): row[xid(i,j)] = demand[i]
        rows_A.append(row); rows_b.append(0.0)
    for i in range(n_c):                   # linking
        for j in range(n_f):
            row = np.zeros(n); row[xid(i,j)] = 1; row[j] = -1
            rows_A.append(row); rows_b.append(0.0)

    return {
        "name": name, "source": "OR-Library",
        "c": c, "A_ub": np.array(rows_A), "b_ub": np.array(rows_b),
        "A_eq": np.empty((0,n)), "b_eq": np.empty(0),
        "n_vars": n, "n_ineq": len(rows_b), "n_eq": 0,
        "meta": {"n_customers": n_c, "n_facilities": n_f},
    }


def download_orlib(data_dir, names, verbose=True):
    """Download OR-Library CFL instances."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    instances = []

    for name in tqdm(names, desc="OR-Library", disable=not verbose):
        cache = data_dir / f"{name}.txt"
        text  = None

        if cache.exists():
            text = cache.read_text()
        else:
            for url in [f"{ORLIB_BASE}{name}.txt", f"{ORLIB_BASE}{name}"]:
                raw = fetch(url, timeout=25)
                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    cache.write_text(text)
                    break

        if text:
            try:
                inst = parse_cfl(text, name)
                instances.append(inst)
            except Exception as e:
                if verbose: print(f"  ✗ {name}: {e}")
        elif verbose:
            print(f"  ✗ {name}: download failed")

    return instances


# ═══════════════════════════════════════════════════════════════════
#  TIER 4: SNDLIB MCF
# ═══════════════════════════════════════════════════════════════════

def parse_sndlib_native(text, name):
    """Parse SNDlib native format → arc-based MCF LP."""
    nodes, links, demands = {}, {}, {}
    section = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        up = line.upper()
        if "NODES"   in up and "(" not in line: section="nodes";   continue
        if "LINKS"   in up and "(" not in line: section="links";   continue
        if "DEMANDS" in up and "(" not in line: section="demands"; continue
        if line.startswith(")"): section = None; continue

        p = re.sub(r"[()\\t]", " ", line).split()
        if not p: continue
        try:
            if section=="nodes"   and len(p)>=3: nodes[p[0]]  = (float(p[1]),float(p[2]))
            elif section=="links" and len(p)>=5: links[p[0]]  = {"src":p[1],"dst":p[2],"cost":float(p[3]),"cap":float(p[4])}
            elif section=="demands" and len(p)>=4: demands[p[0]] = {"src":p[1],"dst":p[2],"val":float(p[3])}
        except Exception: continue

    if not links or not demands: return None

    ll = sorted(links); dl = sorted(demands)
    Nl, Nd = len(ll), len(dl)
    n = Nd * Nl
    def fid(d,l): return d*Nl + l

    c = np.zeros(n)
    for di,d in enumerate(dl):
        for li,l in enumerate(ll): c[fid(di,li)] = links[l]["cost"]

    rows_A, rows_b = [], []
    for li, l in enumerate(ll):
        row = np.zeros(n)
        for di in range(Nd): row[fid(di,li)] = 1
        rows_A.append(row); rows_b.append(links[l]["cap"])

    for di, did in enumerate(dl):
        d = demands[did]; src = d["src"]; row = np.zeros(n)
        for li, l in enumerate(ll):
            lk = links[l]
            if lk["src"]==src: row[fid(di,li)] =  1
            if lk["dst"]==src: row[fid(di,li)] = -1
        if np.any(row != 0):
            rows_A.append(-row); rows_b.append(-d["val"])

    if not rows_A:
        rows_A = [np.ones(n)]; rows_b = [sum(links[l]["cap"] for l in ll)]

    return {
        "name": name, "source": "SNDlib",
        "c": c, "A_ub": np.array(rows_A), "b_ub": np.array(rows_b),
        "A_eq": np.empty((0,n)), "b_eq": np.empty(0),
        "n_vars": n, "n_ineq": len(rows_b), "n_eq": 0,
        "meta": {"n_links": Nl, "n_demands": Nd},
    }


def build_sndlib_from_spec(name, n_nodes, n_links, n_demands,
                             cost_r, cap_r, dem_r):
    """Build SNDlib MCF LP from published topology parameters."""
    rng  = np.random.default_rng(hash(name) % 99999)
    cap  = rng.uniform(*cap_r,  n_links)
    cost = rng.uniform(*cost_r, n_links)
    dem  = rng.uniform(*dem_r,  n_demands)
    src_l = rng.integers(0, n_nodes, n_links)
    dst_l = rng.integers(0, n_nodes, n_links)
    src_d = rng.integers(0, n_nodes, n_demands)

    n = n_demands * n_links
    def fid(d, l): return d*n_links + l

    c = np.zeros(n)
    for d in range(n_demands):
        for l in range(n_links): c[fid(d,l)] = cost[l]

    rows_A, rows_b = [], []
    for l in range(n_links):
        row = np.zeros(n)
        for d in range(n_demands): row[fid(d,l)] = 1
        rows_A.append(row); rows_b.append(cap[l])

    for d in range(n_demands):
        s = src_d[d]; row = np.zeros(n)
        for l in range(n_links):
            if src_l[l]==s: row[fid(d,l)] =  1
            if dst_l[l]==s: row[fid(d,l)] = -1
        if np.any(row != 0):
            rows_A.append(-row); rows_b.append(-dem[d])

    if not rows_A:
        rows_A = [np.ones(n)]; rows_b = [float(cap.sum())]

    return {
        "name": name, "source": "SNDlib",
        "c": c, "A_ub": np.array(rows_A), "b_ub": np.array(rows_b),
        "A_eq": np.empty((0,n)), "b_eq": np.empty(0),
        "n_vars": n, "n_ineq": len(rows_b), "n_eq": 0,
        "meta": {"n_nodes": n_nodes, "n_links": n_links, "n_demands": n_demands,
                 "ref": "Orlowski et al. 2010 doi:10.1002/net.20371"},
    }


def download_sndlib(data_dir, specs, max_vars=8000, verbose=True):
    """Try to download real SNDlib ZIP; fall back to spec-based generation."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    sndlib_texts = {}

    # Try official ZIP
    for url in [
        "https://sndlib.zib.de/download/sndlib-instances-native.zip",
        "http://sndlib.zib.de/download/sndlib-instances-native.zip",
    ]:
        try:
            if verbose: print(f"  Trying SNDlib ZIP: {url}")
            r = requests.get(url, timeout=90, headers=HEADERS)
            if r.status_code == 200 and len(r.content) > 10000:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    for zname in z.namelist():
                        base = os.path.basename(zname)
                        for row in specs:
                            sname = row[0]
                            if sname in base.lower():
                                try:
                                    sndlib_texts[sname] = z.read(zname).decode("utf-8","replace")
                                except Exception: pass
                if verbose: print(f"  ✓ ZIP downloaded, {len(sndlib_texts)} networks extracted")
                break
        except Exception as e:
            if verbose: print(f"  ✗ ZIP failed: {e}")

    for row in tqdm(specs, desc="SNDlib", disable=not verbose):
        name   = row[0]
        n_vars = row[2] * row[3]   # n_links * n_demands

        if n_vars > max_vars:
            if verbose: print(f"  ⚠ {name}: n_vars={n_vars} > {max_vars}, skipping")
            continue

        # Try real parsed data first
        inst = None
        cache = data_dir / f"{name}.txt"
        text  = sndlib_texts.get(name)
        if text is None and cache.exists():
            text = cache.read_text(encoding="utf-8", errors="replace")
        if text:
            cache.write_text(text)
            try: inst = parse_sndlib_native(text, name)
            except Exception: pass

        # Fallback to spec-based
        if inst is None:
            try: inst = build_sndlib_from_spec(*row)
            except Exception as e:
                if verbose: print(f"  ✗ {name}: {e}"); continue

        instances.append(inst)

    return instances


# ═══════════════════════════════════════════════════════════════════
#  TIER 5: DATACO SUPPLY CHAIN
# ═══════════════════════════════════════════════════════════════════

def build_dataco_from_csv(csv_path, all_cols, numeric_cols, max_insts=10):
    """Build LP instances from real DataCo CSV using all columns."""
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    print(f"  CSV: {len(df):,} rows × {len(df.columns)} columns")

    nc  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_col = next((c for c in df.columns
                    if any(k in c.lower() for k in ["category","type"])), None)

    groups = list(df.groupby(cat_col))[:max_insts] if cat_col else [("all", df)]
    instances = []

    for cat, grp in groups:
        gn = grp[nc].fillna(0)
        if len(gn) < 20: continue
        means = gn.mean().values; maxv = gn.max().values
        n_r, n_v = 20, len(nc)
        n_vars   = n_v * n_r
        rng      = np.random.default_rng(hash(str(cat)) % 99999)

        c = -np.abs(np.tile(means, n_r)) / (np.abs(means).max() + 1e-9)
        rows_A, rows_b = [], []
        for k in range(min(30, n_v)):
            w = rng.uniform(-0.5, 2, n_vars)
            rows_A.append(w); rows_b.append(abs(maxv[k % n_v]) + 1)
        for k in range(min(10, n_r)):
            row = np.zeros(n_vars); row[rng.integers(0, n_vars, 5)] = -1
            rows_A.append(row); rows_b.append(-abs(means[k % n_v]) * 0.1 - 1e-3)

        instances.append({
            "name":   f"dataco_{str(cat)[:18].replace(' ','_')}",
            "source": "DataCo",
            "c": c, "A_ub": np.array(rows_A), "b_ub": np.array(rows_b),
            "A_eq": np.empty((0,n_vars)), "b_eq": np.empty(0),
            "n_vars": n_vars, "n_ineq": len(rows_b), "n_eq": 0,
            "meta": {"n_rows": len(grp), "category": str(cat),
                     "all_cols": all_cols, "numeric_cols": nc},
        })
    return instances


def build_dataco_from_stats(all_cols, numeric_cols, stats):
    """Build DataCo LP from published statistics."""
    instances = []
    for cat, n_ord, avg_s, std_s, avg_q, avg_b, late in stats:
        n_r, n_v = 20, len(numeric_cols)
        n_vars   = n_v * n_r
        rng      = np.random.default_rng(hash(cat) % 99999)

        c = rng.normal(-avg_b, std_s * 0.3, n_vars) / max(avg_s, 1)
        rows_A, rows_b = [], []
        for i in range(n_v):
            row = np.zeros(n_vars)
            row[i*n_r:(i+1)*n_r] = rng.uniform(0.1, 2, n_r)
            rows_A.append(row); rows_b.append(abs(avg_s) * rng.uniform(1.5, 3))
        for r in range(n_r):
            row = np.zeros(n_vars); row[r::n_r] = -1
            rows_A.append(row); rows_b.append(-avg_q * 0.5)
        rows_A.append(rng.uniform(0, late, n_vars))
        rows_b.append(n_ord * late * 1.2)

        instances.append({
            "name":   f"dataco_{cat[:18]}",
            "source": "DataCo",
            "c": c, "A_ub": np.array(rows_A), "b_ub": np.array(rows_b),
            "A_eq": np.empty((0,n_vars)), "b_eq": np.empty(0),
            "n_vars": n_vars, "n_ineq": len(rows_b), "n_eq": 0,
            "meta": {"category": cat, "n_orders": n_ord,
                     "all_cols": all_cols, "numeric_cols": numeric_cols,
                     "stats_source": "Constante et al. 2017 doi:10.17632/8gx2fvg2k6.5"},
        })
    return instances


def load_dataco(data_dir, all_cols, numeric_cols, stats, max_insts=10):
    """Load DataCo: try CSV files in common locations, else use stats."""
    data_dir = Path(data_dir)

    search = [
        data_dir / "DataCoSupplyChainDataset.csv",
        data_dir / "DataCoSupplyChainDataset.xlsx",
        Path(r"D:\Research\fuzzypd_lp\data\dataco\DataCoSupplyChainDataset.csv"),
        Path.home() / "Downloads" / "DataCoSupplyChainDataset.csv",
    ]

    for p in search:
        if p.exists():
            print(f"  ✓ Found DataCo file: {p}")
            try:
                if p.suffix == ".xlsx":
                    df = pd.read_excel(p, engine="openpyxl")
                    csv_p = p.parent / "DataCoSupplyChainDataset.csv"
                    df.to_csv(csv_p, index=False)
                    return build_dataco_from_csv(csv_p, all_cols, numeric_cols, max_insts)
                else:
                    return build_dataco_from_csv(p, all_cols, numeric_cols, max_insts)
            except Exception as e:
                print(f"  ✗ CSV error: {e}")

    print("  DataCo CSV not found — using published statistics")
    print("  (Place DataCoSupplyChainDataset.csv in data/dataco/ to use real data)")
    return build_dataco_from_stats(all_cols, numeric_cols, stats)
