# Differentiable KWN Calibration

Gradient-based calibration of a differentiable Kampmann–Wagner Numerical (KWN) model for precipitation hardening in 6xxx-series aluminium alloys.

This repository contains research code developed during my PhD on physics-informed data-driven modelling of aluminium processing. The main contribution is a differentiable implementation of a simplified KWN precipitation-hardening model, enabling automatic calibration of model parameters using gradient-based optimisation.

## Overview

Precipitation hardening is a key strengthening mechanism in 6xxx aluminium alloys. Physics-based models such as KWN can predict microstructural evolution and yield strength, but they contain free parameters that are difficult to calibrate manually.

This project reformulates parts of the KWN framework to make the model compatible with automatic differentiation and gradient-based optimisation. The calibrated model predicts yield strength over ageing time and is validated against experimental and microstructural data.

## Key Features

- Differentiable simplified KWN model
- Yield-strength prediction for 6xxx aluminium alloys
- Gradient-based calibration using Adam
- Comparison with gradient-free optimisation methods
- Physics-informed regularisation of trainable parameters
- Validation using microstructural quantities such as mean particle radius and total number density
- Reproducible scripts for thesis/paper results

## Scientific Context

This code supports the work:

> A. Alizadeh, M. Souissi, M. Zhou, and H. Assadi,  
> "Gradient-Based Calibration of a Precipitation Hardening Model for 6xxx Series Aluminium Alloys,"  
> Metals, 15(9), 1035, 2025.  
> DOI: 10.3390/met15091035
