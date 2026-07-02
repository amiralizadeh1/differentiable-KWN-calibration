# Differentiable KWN Calibration

Gradient-based calibration of a differentiable Kampmann–Wagner Numerical (KWN) model for precipitation hardening in 6xxx-series aluminium alloys.

This repository contains research code developed during my PhD on physics-informed data-driven modelling of aluminium processing. The main contribution is a differentiable KWN calibration framework that combines precipitation-hardening physics, CALPHAD thermodynamics, automatic differentiation, Adam optimisation, regularisation, and microstructural validation.

## Overview

Precipitation hardening is a key strengthening mechanism in 6xxx aluminium alloys. Physics-based KWN models can predict precipitate evolution and yield strength, but they contain free parameters that are difficult to calibrate manually.

This project reformulates a simplified KWN model to make it compatible with TensorFlow automatic differentiation. The model is calibrated against ageing data, then evaluated using both macroscopic yield-strength behaviour and internal microstructural variables such as total number density (TND), mean particle radius (MPR), and total volume fraction (TVF).

## Key Features

- Differentiable KWN precipitation-hardening model
- Gradient-based calibration using Adam
- Physics-informed regularisation of trainable parameters
- Parameter rebounding regularisation to reduce non-unique solutions
- Monte Carlo optimisation runs with different random initialisations
- Validation against yield-strength, hardness, TND, and MPR data
- Visualisation of optimisation convergence and internal physics
- Unit test showing recovery of known synthetic parameters
- Post-processing scripts for uncertainty and robustness analysis

## Scientific Context

This repository supports the work:

> A. Alizadeh, M. Souissi, M. Zhou, and H. Assadi,  
> **Gradient-Based Calibration of a Precipitation Hardening Model for 6xxx Series Aluminium Alloys**,  
> *Metals*, 15(9), 1035, 2025.  
> https://doi.org/10.3390/met15091035

This implementation further expands the methodology presented in the publication with additional validation utilities, Monte Carlo analyses, visualisation tools, unit tests, processed experimental datasets, and reproducible scripts used throughout the PhD thesis.

## Relation to Previous Work

Recent work by Machine Learning-Assisted Process Optimization of Al-Mg-Si Alloys and A differentiable precipitation model for Bayesian calibration has demonstrated the growing importance of combining machine learning with precipitation-hardening models. In particular, the npj Computational Materials paper is an important contribution toward differentiable physics-based modelling, showing that automatic differentiation can be successfully integrated with precipitation simulations for  parameter estimation.

This repository addresses a complementary problem: efficient gradient-based calibration of recursive precipitation-hardening models. As discussed in my PhD thesis, previous optimisation strategies in KWN-type frameworks have relied predominantly on gradient-free algorithms, which require a large number of expensive model evaluations and become increasingly inefficient for recursive, coupled physical models.

By reformulating the simplified KWN model into a differentiable computational graph, smoothing discontinuous conditional logic, and introducing physics-informed regularisation (including parameter rebounding and unity regularisation), gradients can be propagated through the entire precipitation model and exploited by modern first-order optimisers such as Adam.

Metric	Adam	Powell	Nelder–Mead
Training time (min)	17.9 ± 2.1	57 ± 12	45 ± 16
Function evaluations	139 ± 17	886 ± 190	620 ± 170
Invalid evaluations / trial	0	40 ± 38	3.2 ± 1.2
Computational complexity	O(N)	O(N²)	O(N)

The benchmark demonstrates that exploiting analytical gradients substantially reduces optimisation time and required model evaluations while simultaneously eliminating invalid parameter evaluations during calibration. Combined with the proposed physics-informed regularisation strategy, the optimisation converges to physically meaningful parameter sets and produces stable predictions of yield strength and latent microstructural variables (total number density, mean particle radius, and total volume fraction).

## Data Sources

The Sekhar ageing dataset used for primary calibration is from:

> Sekhar et al.,  
> **Evolution of hardness and yield strength in Al-Mg-Si alloys**,  
> *IOP Conference Series: Materials Science and Engineering*, 338, 012011, 2018.  
> https://doi.org/10.1088/1757-899X/338/1/012011

The Myhr Alloy 4 validation data were digitised from Figure 3 of:

> Myhr, O.R. and Grong, Ø. (2000).  
> **Modelling of the Age Hardening Behaviour of Al–Mg–Si Alloys**,  
> *Acta Materialia*, 48(7), 1605–1615.  
> https://doi.org/10.1016/S1359-6454(99)00430-8

The Myhr paper was used only as a source of validation data. The implementation and calibration framework in this repository are independent.

## Repository Structure

```text
.
├── revision kwn.py              # Main differentiable KWN calibration script using Sekhar data
├── validation.py                # Secondary validation using Myhr Alloy 4 data
├── unit.py                      # Algorithmic unit test for parameter recovery
├── Physics-based KWN.py         # Forward physics-based KWN simulation with fixed parameters
├── plot_utils.py                # Shared plotting and result-export functions
├── validation_data.py           # Processing and inspection of validation datasets
├── plot_averaged.py             # Plots processed averaged experimental data
├── params_mc_deviation.py       # Monte Carlo parameter-deviation analysis
├── plot_mc_predictions.py       # Monte Carlo prediction uncertainty plots
│
├── plots/
│   ├── seed1/
│   ├── seed2/
│   └── seed3/
│
├── optimized_results/
│   ├── all_mc_final_predictions.xlsx
│   └── mc_predictions_plot.png
│
├── README.md
├── LICENSE
└── requirements.txt
```

## Main Scripts

### `revision kwn.py`

Primary calibration script. It trains the differentiable KWN model on the Sekhar Al-Mg-Si ageing dataset at 150 °C. It uses Adam optimisation to update five trainable parameters and includes regularisation terms to keep parameter relationships physically meaningful.

### `validation.py`

Secondary validation script. It applies the framework to the Myhr Alloy 4 case at 185 °C. It compares predicted yield strength with validation data and also sends TND, MPR, TVF, and yield strength to the visualisation utilities for physical validation.

### `unit.py`

Algorithmic verification script. It generates a synthetic target from a known parameter set and tests whether the optimiser can recover the original parameters. In the thesis, the unit test recovered all five latent parameters with less than 0.32% error.

### `plot_utils.py`

Shared plotting utility. It generates convergence plots, loss curves, parameter trajectories, yield-strength plots, and physical validation plots for TND, MPR, TVF, and yield strength.

### `params_mc_deviation.py`

Post-processing script for Monte Carlo parameter stability. It compares parameter spread with and without parameter rebounding regularisation, showing why regularisation is needed to reduce non-unique solutions.

### `plot_mc_predictions.py`

Post-processing script for prediction uncertainty. It reads final predictions from multiple Monte Carlo runs, calculates mean ± standard deviation, and compares predictions with experimental hardness, TND, and MPR data.

## Output Folders

### `plots/`

Contains intermediate outputs from individual optimisation runs. Each seed folder corresponds to one Monte Carlo run with a different random initialisation.

Typical contents include:

- loss curves,
- parameter trajectories,
- yield-strength prediction plots,
- TND, MPR, TVF physical validation plots,
- Excel files containing optimisation histories.

### `optimized_results/`

Contains final outputs after optimisation. These results are used for uncertainty analysis and thesis/paper figures.

Typical contents include:

- final predictions from all Monte Carlo runs,
- combined prediction spreadsheets,
- mean ± standard deviation prediction plots,
- comparison against experimental validation data.

## Installation

```bash
git clone https://github.com/amiralizadeh1/differentiable-KWN-calibration.git
cd differentiable-KWN-calibration

python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Requirements

The code was developed using:

- Python
- NumPy
- SciPy
- TensorFlow
- Matplotlib
- Pandas
- Kawin
- OpenPyXL

Exact package versions should be listed in `requirements.txt`.

## Typical Workflow

1. Run the main calibration:

```bash
python "revision kwn.py"
```

2. Run secondary validation:

```bash
python validation.py
```

3. Run unit-test verification:

```bash
python unit.py
```

4. Analyse parameter non-uniqueness:

```bash
python params_mc_deviation.py
```

5. Plot Monte Carlo prediction uncertainty:

```bash
python plot_mc_predictions.py
```

## Research Contribution

The main contribution of this repository is not simply fitting an ageing curve. It demonstrates that a physics-based precipitation model can be reformulated as a differentiable computational graph, allowing gradient-based optimisation to calibrate physically meaningful model parameters.

The framework also highlights an important inverse-problem issue: multiple parameter sets can produce similar yield-strength predictions. Parameter rebounding regularisation is therefore introduced to reduce non-unique solutions and improve physical interpretability.

## Citation

If you use this code, please cite:

```bibtex
@article{alizadeh2025gradient,
  title={Gradient-Based Calibration of a Precipitation Hardening Model for 6xxx Series Aluminium Alloys},
  author={Alizadeh, Amir and Souissi, Maaouia and Zhou, Mian and Assadi, Hamid},
  journal={Metals},
  volume={15},
  number={9},
  pages={1035},
  year={2025},
  doi={10.3390/met15091035}
}
```

## License

This project is released under the **Apache License 2.0**.

Apache 2.0 is a permissive open-source license that allows academic and commercial reuse while also including an explicit patent grant. This makes it suitable for research software that may later be extended, reused, or incorporated into larger scientific and industrial software projects.

See `LICENSE` for details.

## Disclaimer

This repository contains research software developed for PhD work. Results depend on model assumptions, alloy composition, thermodynamic data, ageing conditions, digitised experimental data, and optimisation settings. The code is provided for reproducibility, academic research, and further development.

## Extended Dataset

In addition to the datasets used directly in the calibration and validation scripts, I have also compiled a broader literature-based dataset for aluminium yield strength as a function of Mg content, Si content, temperature, and ageing condition.

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/amiralizadeh9720/aluminum-yield-strength-mg-si-temp

Readers are encouraged to use this dataset to test the model on a wider range of alloy compositions and processing conditions beyond the specific Sekhar and Myhr cases included in this repository. This can support further work on model generalisation, transferability, and uncertainty analysis across 6xxx-series aluminium alloys.
