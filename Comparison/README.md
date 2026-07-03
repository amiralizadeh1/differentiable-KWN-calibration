# Physics-Informed Data-Driven Modelling of Aluminium Processing

This project focuses on modeling the precipitation hardening process in aluminum alloys, specifically within the Al-Mg-Si system. Precipitation hardening is a crucial mechanism in metallurgy that enhances the yield strength of materials by forming fine precipitates within the metal matrix.

This project employs a simplified version of the KWN model, where certain physical aspects are deliberately omitted. To account for these simplifications, a set of free parameters is strategically incorporated into the physics-based equations. An optimizer is then used to adjust these parameters, refining them through iterations until they align with the experimental data.

This work utilizes physics-informed neural networks (PINN), where limited training data compensates for gaps in understanding the underlying dynamics.

This project is particularly useful in the field of materials science and engineering, where understanding and predicting the mechanical properties of alloys are critical. The integration of machine learning techniques with traditional thermodynamic and kinetic modeling can provide more accurate predictions and help in designing new alloys with desired properties.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Contact](#contact)

## Installation

To get a local copy up and running follow these simple steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amiralizadeh1/KWN.git

## Usage

1. In scenarios where the physical dynamics are complex, and even a network of intricate physics-based equations leaves certain aspects of the physics unknown.

2. In cases where the available training data is insufficient, incorporating physical constraints helps compensate for the data shortage.

3. For integrating the traditional methods of modeling the precipitation hardening process with more modern machine learning techniques.

## Features

1. **Thermodynamic Modeling**: The project leverages thermodynamic models based on CALPHAD (CALculation of PHAse Diagrams) methods to calculate the driving forces for phase transformations and precipitate formation in aluminum alloys. These calculations are handled using the MulticomponentThermodynamics class from the kawin.Thermodynamics library, which considers multiple phases, such as FCC_A1 and MG5SI6_B_DP.

2. **Numerical Simulation:** The project simulates the nucleation, growth, and coarsening of precipitates over time using various physical constants and material-specific parameters. TensorFlow is employed to manage variables, perform gradient computation, and optimize the model, highlighting the project's integration of machine learning techniques with traditional materials science methodologies.

3. **Optimization:** The project involves optimizing several parameters (e.g., param1, param2, param3, etc.) to minimize the difference between simulated and experimentally derived yield strength values (Yield_t). Custom learning rates are assigned to each parameter, allowing for effective fine-tuning of the model.

4. **Visualization:** The project includes tools for visualizing the evolution of key quantities, such as number density, mean particle radius, total volume fraction, and yield strength over time. These visualizations are crucial for evaluating the accuracy and behavior of the simulation.

5. **Automation:** The project automates the processes of gradient computation and parameter updating, streamlining the execution of iterative simulations and optimizations.

## Contributing

Amir Alizadeh

## Licence
This project is licensed under the MIT License 

## Authors

Amir Alizadeh

## Contact

alizadehamir21@gmail.com

