# TSAR-LP — Adaptive Calibration of the Robustness Budget for Uncertainty-Aware Linear Programming

Reproducibility repository for **Mishra (2026), "TSAR-LP: Adaptive Calibration of the Robustness Budget for Uncertainty-Aware Linear Programming"**, submitted to *Computers & Operations Research*.

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## What this repository contains

Code, data, and result files for every experiment, table, and figure in the paper.

```
tsar-lp/
├── src/
│   ├── core.py            # TFN scenarios, calibrate_gamma, adversarial SAA, selection rule
│   ├── solvers.py         # All 10 solver implementations
│   └── datasets.py        # Loaders for NETLIB, MIPLIB, OR-Library, SNDlib, DataCo
├── config.py              # S, T, time limits, TFN parameters
├── run.py                 # Main benchmark driver (50 instances × 10 solvers, κ = 0.20)
├── run_multi_kappa.py     # Supplementary multi-κ sweep driver (subprocess-isolated)
├── results/
│   ├── all_results_live.csv      # Main experiment, 520 rows
│   └── multi_kappa_results.csv   # Multi-κ sweep, 800+ rows
├── data/                  # Benchmark instances (cached locally)
├── figs/                  # Generating scripts for all paper figures
├── requirements.txt
└── README.md              # this file
```

## Requirements

- Python 3.10 or newer
- 12 GB RAM (recommended)
- ≈ 5 GB disk for cached benchmark instances

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pinned versions: NumPy 1.26, SciPy 1.11 (HiGHS 1.7 via `scipy.optimize.linprog`), pandas 2.1, matplotlib 3.8, requests, tqdm.

## Reproducing the main experiment

```bash
python run.py
```

- Wall-clock time: ~2 hours on a single 12 GB Windows laptop, ~25 minutes on an AMD EPYC 7662 server.
- Output: `results/all_results_live.csv` — identical row layout to the version in this repository.
- Reproduces: Figures 1–7, Tables 2–3.

## Reproducing the multi-κ sensitivity experiment (Section 7.8)

```bash
python run_multi_kappa.py
```

- Wall-clock time: ~3 hours on a single 12 GB Windows laptop.
- Output: `results/multi_kappa_results.csv`.
- Reproduces: Figures 8–9, Tables 4–5.

The multi-κ driver runs each `(instance, κ)` block in a fresh child Python process. If a native solver crash kills a child (rare; documented in the paper for the `semi-continuous` instance at κ = 0.05), the parent process logs it and continues. Rerunning the same command resumes from the last completed solver — already-saved rows are skipped.

## Benchmark sources (independent of this repository)

The original instances are available from:

| Tier        | Source                                                                       |
|-------------|------------------------------------------------------------------------------|
| NETLIB      | http://www.netlib.org/lp/                                                    |
| MIPLIB 2017 | http://miplib.zib.de/                                                        |
| OR-Library  | http://people.brunel.ac.uk/~mastjjb/jeb/orlib/ (cap41–cap74)                 |
| SNDlib 1.0  | http://sndlib.zib.de/ (abilene)                                              |
| DataCo SCDS | Kaggle: `DataCo Smart Supply Chain for Big Data Analysis`                    |

`src/datasets.py` reproduces the exact preprocessing (column selection, LP relaxation, TFN injection) used in the paper.

## Citation

If you use this code or data, please cite:

```bibtex
@article{mishra2026tsarlp,
  title   = {TSAR-LP: Adaptive Calibration of the Robustness Budget for Uncertainty-Aware Linear Programming},
  author  = {Mishra, Rahul Kumar},
  journal = {Computers \& Operations Research},
  year    = {2026},
  doi     = {10.xxxx/xxxxx}
}
```

## Contact

Rahul Kumar Mishra · Independent Researcher · Ranchi, India · mishra.rahul98@gmail.com  
ORCID: 0009-0004-9671-4055

## License

MIT (code) and CC-BY 4.0 (data and results). See `LICENSE` for full terms.
