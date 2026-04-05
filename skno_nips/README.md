# SKNO NeurIPS Prototype (MPS-ready)

This package provides a compact but complete experiment runner for a **Sparse Koopman Neural Operator (SKNO)** with automatic data handling for three datasets/use cases:

1. **Navier–Stokes** via PDEBench / DaRUS (HDF5)
2. **Wildfire** via NASA FIRMS active-fire CSV API
3. **Atmospheric precipitation** via NOAA/NCEI GPCP Daily NetCDF files

The code is designed to:
- auto-detect **Apple MPS**, CUDA, or CPU
- auto-download datasets where possible
- preprocess into fixed rolling windows
- train a minimal SKNO model
- generate NeurIPS-style figures automatically

## Quick start

```bash
cd skno_neurips
python run_experiment.py --dataset atmospheric --root ./data --epochs 5
python run_experiment.py --dataset wildfire --root ./data --epochs 5 --firms-map-key YOUR_KEY
python run_experiment.py --dataset navier_stokes --root ./data --epochs 5
```

## Notes

### 1) Wildfire / NASA FIRMS
NASA FIRMS requires a free MAP_KEY. You can either:
- pass `--firms-map-key ...`, or
- set `NASA_FIRMS_MAP_KEY` in your environment.

### 2) Navier–Stokes / PDEBench
The downloader queries the DaRUS / Dataverse API for the PDEBench dataset and tries to fetch the smallest matching file whose name contains `Navier` and ends in `.hdf5`. If you want a more specific filename match, use:

```bash
python run_experiment.py --dataset navier_stokes --navier-match "Navier" \
  --root ./data --epochs 5
```

### 3) Atmospheric / NOAA GPCP Daily
By default this fetches a compact date slice of daily precipitation files and subsets them to a North America box.

## Recommended commands

### Atmospheric
```bash
python run_experiment.py --dataset atmospheric --root ./data --epochs 10 \
  --start-date 2023-01-01 --end-date 2023-03-31
```

### Wildfire
```bash
python run_experiment.py --dataset wildfire --root ./data --epochs 10 \
  --start-date 2023-08-01 --end-date 2023-09-15 \
  --bbox -125 32 -113 43 \
  --firms-map-key YOUR_KEY
```

### Navier–Stokes
```bash
python run_experiment.py --dataset navier_stokes --root ./data --epochs 10 \
  --navier-match "Navier"
```

## Outputs

Each run creates an output directory with:
- `metrics.json`
- `checkpoint.pt`
- `loss_curves.png`
- `koopman_matrix.png`
- `qualitative_prediction.png`
- `rollout.png`
- `neurips_panel.png`

## Package layout

- `skno/model.py` – SKNO architecture
- `skno/datasets.py` – loaders + downloaders
- `skno/runner.py` – training / evaluation / orchestration
- `skno/figures.py` – NeurIPS-style figure generation
- `run_experiment.py` – single CLI entrypoint
