"""
config.py — All paths and experiment parameters.
Configured for 32GB RAM, 4 workers.
"""
import os
from pathlib import Path

# ── Base directory ─────────────────────────────────────────────────
BASE_DIR   = Path(r"D:\Research\fuzzypd_lp")

# ── Sub-directories ────────────────────────────────────────────────
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR    = BASE_DIR / "logs"

for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR,
          DATA_DIR/"netlib", DATA_DIR/"miplib",
          DATA_DIR/"orlib",  DATA_DIR/"sndlib",
          DATA_DIR/"dataco"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset URLs ───────────────────────────────────────────────────
NETLIB_OFFICIAL  = "https://www.netlib.org/lp/data/"
HIGHS_MPS_BASE   = "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/"
ORLIB_BASE       = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
SNDLIB_ZIP       = "https://sndlib.zib.de/download/sndlib-instances-native.zip"

# ── Experiment parameters ──────────────────────────────────────────
S_TRAIN          = 100        # SAA training scenarios
T_TEST           = 500        # out-of-sample test scenarios (auto-reduced for large n)
SPREAD_LO        = 0.05       # TFN lower spread  (±5%)
SPREAD_HI        = 0.20       # TFN upper spread  (±20%)
FPDLP_MAX_ITER   = 200_000    # FuzzyPD-LP max CP iterations (skipped for large n)
FPDLP_TOL        = 1e-5       # convergence tolerance
SOLVER_TIME_LIM  = 120        # seconds per solver per instance
MAX_VARS         = 1000       # laptop 32GB limit

# ── Auto-detect RAM and configure safely ──────────────────────────
import psutil as _psutil
try:
    _mem = _psutil.virtual_memory()
    _TOTAL_RAM_GB     = _mem.total     / 1e9   # total installed RAM
    _FREE_RAM_GB      = _mem.available / 1e9   # actually free right now
    # Use FREE RAM as budget (what actually matters at runtime)
    _DETECTED_RAM_GB  = _FREE_RAM_GB
except Exception:
    _TOTAL_RAM_GB    = 34.0
    _FREE_RAM_GB     = 25.0   # conservative fallback
    _DETECTED_RAM_GB = 25.0

# ── RAM configuration (safe for 30-32GB machines) ─────────────────
# Peak memory per worker = SAA + Test + AdvSAA matrices simultaneously
# Formula: RAM_PER_MATRIX = (RAM_TOTAL - OS - overhead) / (workers * 3 matrices)
RAM_TOTAL_GB     = int(_DETECTED_RAM_GB)  # auto-detected from system
RAM_OS_RESERVE   = 0          # already removed since _DETECTED_RAM_GB = free RAM
RAM_WORKERS_N    = 1  # laptop 1 worker safe
# Each worker holds 3 large matrices simultaneously:
#   SAA matrix  (S * m * n * 8 bytes)
#   Test matrix (T * m * n * 8 bytes)
#   AdvSAA matrix (same size as SAA — built inside TSAR-LP)
_RAM_AVAILABLE   = RAM_TOTAL_GB - RAM_OS_RESERVE   # 29 GB
_RAM_PER_WORKER  = _RAM_AVAILABLE / RAM_WORKERS_N  # 7.25 GB
_N_SIMULTANEOUS  = 2          # max simultaneous: SAA+Test OR Test+AdvSAA (SAA freed before AdvSAA)
_HEADROOM_GB     = 2          # safety headroom for Python GC and spikes
# Per-matrix budget ensures total stays under RAM_TOTAL_GB:
_PER_MATRIX_GB   = 4.0   # laptop safe
SAA_MAX_GB       = _PER_MATRIX_GB   # max GB for SAA matrix
TEST_MAX_GB      = _PER_MATRIX_GB   # max GB for test matrix
N_WORKERS        = RAM_WORKERS_N    # alias for backward compat
RAM_AVAILABLE    = _RAM_AVAILABLE   # alias for backward compat
RAM_PER_WORKER   = _RAM_PER_WORKER  # alias for backward compat
# Result on 30-32 GB machine, 4 workers:
#   cap41 (n=816, m=866):     S=100 T=398 — almost full
#   cplex1 (n=3221, m=2003):  S=43  T=43  — auto-reduced (safe)
#   cap111 (n=2550, m=2600):  S=44  T=44  — auto-reduced (safe)

# CP skip threshold: skip CP refinement when matrix is too large
# (above this size, MV solution alone is used — CP would take hours)
CP_SKIP_N        = 500        # skip CP when n_vars > 500

# ── Parallel execution ─────────────────────────────────────────────
# N_WORKERS already set above

# ── NETLIB instances to download ──────────────────────────────────
NETLIB_NAMES = [
    "afiro","adlittle","blend","sc50a","sc50b","sc105","sc205",
    "share1b","share2b","stocfor1","stocfor2","stocfor3",
    "israel","kb2","scagr7","scagr25","bandm","beaconfd",
    "bore3d","capri","e226","etamacro","finnis","fit1d",
    "forplan","ganges","lotfi","grow7","grow15","grow22",
    "recipe","sctap1","sctap2","sctap3",
    "ship04l","ship04s","ship08l","ship08s",
    "ship12l","ship12s","standmps","tuff","wood1p","woodw",
]

# ── OR-Library instances ───────────────────────────────────────────
ORLIB_NAMES = [f"cap{i}" for i in
               list(range(41, 75)) +
               [101,102,103,104,111,112,113,114,
                121,122,123,124,131,132,133,134]]

# ── SNDlib instances ───────────────────────────────────────────────
SNDLIB_SPECS = [
    ("abilene",      12,  15,   30, (1,10),   (622,2488),  (0.1,1.0)),
    ("atlanta",      15,  22,  210, (1,15),   (100,1000),  (0.5,5.0)),
    ("brain",       161, 268,  264, (1,20),   (100,2000),  (0.1,2.0)),
    ("dfn-bwin",     10,  45,   90, (1,8),    (155,622),   (0.2,2.0)),
    ("dfn-gwin",     11,  47,  110, (1,8),    (155,622),   (0.2,2.0)),
    ("di-yuan",      11,  42,  110, (1,12),   (155,1244),  (0.1,1.5)),
    ("france",       25,  45,  300, (1,15),   (155,10000), (1.0,10.0)),
    ("germany50",    50,  88, 2450, (1,20),   (622,10000), (0.5,5.0)),
    ("geant",        22,  36,  462, (1,25),   (622,10000), (1.0,10.0)),
    ("janos-us",     26,  84,  650, (1,20),   (622,10000), (0.5,8.0)),
    ("newyork",      16,  49,  240, (1,15),   (100,1000),  (0.2,2.0)),
    ("nobel-eu",     28,  41,  756, (1,20),   (622,10000), (1.0,10.0)),
    ("nobel-germany",17,  26,  272, (1,15),   (100,2000),  (0.2,2.0)),
    ("nobel-us",     14,  21,  182, (1,10),   (100,1000),  (0.2,2.0)),
    ("norway",       27,  51,  702, (1,15),   (100,2000),  (0.5,5.0)),
    ("pdh",          11,  34,  110, (1,8),    (155,622),   (0.1,1.0)),
    ("pioro40",      40,  89, 1560, (1,20),   (155,10000), (0.5,5.0)),
    ("polska",       12,  18,  132, (1,10),   (155,2488),  (0.2,2.0)),
    ("sun",          27, 102,  702, (1,15),   (155,10000), (0.5,5.0)),
    ("ta1",          24,  55,  552, (1,20),   (622,10000), (1.0,10.0)),
    ("ta2",          65, 108, 4160, (1,25),   (622,10000), (0.5,8.0)),
    ("zib54",        54,  80, 2862, (1,20),   (622,10000), (1.0,10.0)),
]

# ── DataCo column names ────────────────────────────────────────────
DATACO_ALL_COLS = [
    "Type","Days_for_shipping_real","Days_for_shipment_scheduled",
    "Benefit_per_order","Sales_per_customer","Delivery_Status",
    "Late_delivery_risk","Category_Id","Category_Name",
    "Customer_City","Customer_Country","Customer_Email",
    "Customer_Fname","Customer_Id","Customer_Lname","Customer_Password",
    "Customer_Segment","Customer_State","Customer_Street","Customer_Zipcode",
    "Department_Id","Department_Name","Latitude","Longitude",
    "Market","Order_City","Order_Country","Order_Customer_Id",
    "Order_Id","Order_Item_Cardprod_Id","Order_Item_Discount",
    "Order_Item_Discount_Rate","Order_Item_Id",
    "Order_Item_Product_Price","Order_Item_Profit_Ratio",
    "Order_Item_Quantity","Order_Item_Total","Order_Item_Product_Name",
    "Order_Profit_Per_Order","Order_Region","Order_State",
    "Order_Status","Order_Zipcode","Product_Card_Id",
    "Product_Category_Id","Product_Description","Product_Image",
    "Product_Name","Product_Price","Product_Status",
    "Sales","Order_date","Shipping_date",
]
DATACO_NUMERIC_COLS = [
    "Days_for_shipping_real","Days_for_shipment_scheduled",
    "Benefit_per_order","Sales_per_customer","Late_delivery_risk",
    "Category_Id","Customer_Id","Department_Id","Latitude","Longitude",
    "Order_Customer_Id","Order_Id","Order_Item_Cardprod_Id",
    "Order_Item_Discount","Order_Item_Discount_Rate","Order_Item_Id",
    "Order_Item_Product_Price","Order_Item_Profit_Ratio",
    "Order_Item_Quantity","Order_Item_Total","Order_Profit_Per_Order",
    "Order_Zipcode","Product_Card_Id","Product_Category_Id",
    "Product_Price","Sales",
]
DATACO_STATS = [
    ("Fishing",          11549, 131.7, 98.2,  2.6, 46.8, 0.534),
    ("Cleats",            9951, 152.3,113.4,  2.8, 53.5, 0.521),
    ("Camping_Hiking",    8726, 127.9, 94.1,  2.5, 44.4, 0.518),
    ("Cardio_Equipment",  7632, 218.4,162.6,  2.2, 75.5, 0.541),
    ("Water_Sports",      7489, 143.2,106.8,  2.7, 49.6, 0.529),
    ("Fishing_Accessories",6823, 98.4, 73.2,  2.1, 33.8, 0.512),
    ("Computers",        12845, 489.2,342.1,  1.8,168.3, 0.558),
    ("Electronics",      15234, 387.6,276.4,  2.0,132.7, 0.549),
    ("Garden",            5621, 167.8,124.3,  2.9, 57.2, 0.523),
    ("Toys",             18932,  89.3, 66.7,  3.2, 30.1, 0.506),
]
