# Plots

This directory contains the intermediate figures and data generated during the
Monte Carlo optimisation of the differentiable KWN precipitation-hardening
model.

Each optimisation run is stored separately to allow inspection of the
optimisation process and comparison between different random initialisations.

## Folder structure

```
plots/
├── seed1/
├── seed2/
├── seed3/
└── ...
```

Each `seed*` directory corresponds to one independent Monte Carlo optimisation
run using a different random initialisation of the trainable KWN parameters.

## Contents of each seed folder

Typical files include:

- **Combined optimisation plots**
  - Yield-strength prediction
  - Loss evolution
  - Parameter convergence

- **Physics validation plots**
  - Total Number Density (TND)
  - Mean Particle Radius (MPR)
  - Total Volume Fraction (TVF)
  - Yield Strength (YS)

- **Excel outputs**
  - Loss history
  - Parameter trajectories
  - Predicted yield-strength curve

## Purpose

These files are intermediate optimisation outputs used to monitor the
calibration process. They allow inspection of:

- convergence of the optimisation algorithm,
- evolution of the trainable parameters,
- agreement between model predictions and experimental data,
- evolution of the internal KWN state variables (TND, MPR and TVF).

The final calibrated predictions from all Monte Carlo runs are collected
separately in the `optimized_results/` directory for uncertainty analysis and
publication-quality figures.
