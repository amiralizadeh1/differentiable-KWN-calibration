# Optimisation Benchmark

This directory contains the complete benchmarking framework used to evaluate the proposed gradient-based calibration strategy against traditional gradient-free optimisation methods for the differentiable KWN precipitation hardening model.

The scripts reproduce the optimisation experiments and post-processing presented in the accompanying publication, including convergence analysis, parameter uncertainty, prediction accuracy, and validation on independent datasets.

---

## Directory Structure

### `Comparison.py`

Main benchmarking script.

This script performs the optimisation benchmark by calibrating the differentiable KWN model using multiple optimisation algorithms under identical conditions. It records optimisation history, convergence behaviour, execution time, calibrated parameters, yield-strength predictions, and intermediate microstructural quantities.

Outputs include:

- calibrated model parameters
- optimisation history
- yield-strength predictions
- convergence curves
- benchmarking statistics

---

### `Physics-based KWN.py`

Reference implementation of the original physics-based KWN precipitation hardening model.

Unlike the differentiable implementation, this version is intended primarily for verification and comparison of the underlying precipitation model.

---

### `Data-driven Phys-based KWN.py`

Differentiable implementation of the KWN model used for gradient-based optimisation.

This implementation introduces differentiable approximations to enable automatic differentiation while preserving the underlying physical model.

This is the core model used throughout the benchmarking experiments.

---

### `validation.py`

Validation script for evaluating the calibrated model on independent ageing datasets.

The script compares predicted yield-strength curves against experimental measurements from multiple literature sources, allowing assessment of the model's generalisation capability beyond the calibration dataset.

---

### `plot_utils.py`

Utility functions used throughout the optimisation process.

Responsibilities include:

- plotting optimisation progress
- exporting predicted yield-strength curves
- saving optimisation history
- computing goodness-of-fit metrics
- generating publication-quality figures

---

### `parameters error bar.py`

Post-processing script for analysing optimisation repeatability.

The script loads calibrated parameters from multiple optimisation runs and computes:

- mean parameter values
- standard deviations
- parameter uncertainty
- parallel-coordinate visualisations
- publication-ready summary tables

This script was used to quantify optimisation robustness.

---

### `YS error bars.py`

Computes uncertainty in predicted yield-strength curves across repeated optimisation runs.

Outputs include:

- mean prediction
- standard deviation
- confidence bands
- yield-strength uncertainty plots

---

### `YS_data.py`

Collection of experimental ageing datasets used for model calibration and validation.

The file contains literature data from several aluminium alloys and ageing conditions used throughout the benchmarking and validation studies.

---

### `get_mean.ipynb`

Jupyter notebook used for exploratory statistical analysis of repeated optimisation experiments.

Primarily used during manuscript preparation to compute summary statistics and verify numerical results.

---

## Output Directories

### `plots/`

Contains automatically generated figures including:

- optimisation convergence
- yield-strength predictions
- parameter evolution
- uncertainty plots
- validation figures

---

### `results/`

Stores numerical outputs generated during optimisation, including calibrated parameters, prediction data, and benchmarking statistics.

---

## Benchmark Objectives

The benchmarking framework compares gradient-based and gradient-free optimisation methods with respect to:

- optimisation runtime
- number of objective function evaluations
- convergence behaviour
- optimisation stability
- parameter repeatability
- prediction accuracy
- physical plausibility of calibrated parameters

The implementation reproduces the benchmarking results reported in:

> **Alizadeh, A., Souissi, M., Zhou, M., & Assadi, H.**
> *Gradient-Based Calibration of a Precipitation Hardening Model for 6xxx Series Aluminium Alloys.*
> *Metals*, 2025, 15(9), 1035.
