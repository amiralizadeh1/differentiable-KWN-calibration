optimized_results/

Purpose
-------
This folder stores the final outputs generated after Monte Carlo optimisation of
the differentiable KWN model.

Unlike the plots folder, which contains optimisation progress during individual
training runs, this directory contains the final calibrated predictions and
summary results used for uncertainty analysis and publication figures.

Typical contents
----------------
• Final predictions from every Monte Carlo optimisation run.
• Combined prediction datasets (e.g. all_mc_final_predictions.xlsx).
• Statistical summaries of the calibrated model.
• Figures comparing the mean Monte Carlo prediction with experimental data.
• Confidence bands (mean ± standard deviation) showing prediction uncertainty.

Role in the workflow
--------------------
The optimisation scripts first calibrate the KWN model independently for
multiple Monte Carlo runs using different random parameter initialisations.
After optimisation, the final predictions from all runs are collected and saved
in this folder.

These stored results are then used by post-processing scripts (such as
plot_mc_predictions.py) to:
    • calculate the mean prediction,
    • estimate prediction uncertainty,
    • compare predictions with experimental measurements,
    • generate publication-quality figures.

Purpose of the analysis
-----------------------
This folder supports the uncertainty analysis presented in the thesis. It
demonstrates the repeatability and robustness of the calibrated KWN framework
by analysing the variation in model predictions obtained from independent
optimisation runs.
