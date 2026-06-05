# Benchmark data

This folder holds locally cached benchmark instances. Subfolders that are
empty in this repository will be populated automatically on first run
of `run.py` or `run_multi_kappa.py` via the loaders in `src/datasets.py`.

## What's included
- `netlib/`  — NETLIB MPS files (cached if available)
- `orlib/`   — Beasley OR-Library cap series (cached if available)

## What's auto-downloaded on first run
- `miplib/`  — MIPLIB 2017 LP relaxations, fetched from http://miplib.zib.de/
- `sndlib/`  — SNDlib network design instances, fetched from http://sndlib.zib.de/

## What you need to provide yourself
- `dataco/DataCoSupplyChainDataset.csv` — the DataCo Smart Supply Chain
  dataset. License terms forbid redistribution, so download it from
  https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
  and place the CSV at `data/dataco/DataCoSupplyChainDataset.csv` before
  running any experiment that uses DataCo subsets.

All four other sources (NETLIB, MIPLIB, OR-Library, SNDlib) are
freely redistributable and their files have stable URLs.
