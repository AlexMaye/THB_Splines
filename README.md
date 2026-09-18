# THBSplines

## Truncated Hierarchical B-Splines

This repository contains a Python implementation of truncated hierarchical B-Splines, based on the article [Multi-level Bézier extraction for hierarchical local refinement of Isogeometric Analysis](https://doi.org/10.1016/j.cma.2017.08.017).

The code structure is based on the article [Algorithms for the implementation of adaptive isogeometric methods using hierarchical B-splines](https://doi.org/10.1016/j.apnum.2017.08.006).

This code does not manage the construction of mass/stiffness matrices. Instead, it is intended to be used with [FEniCSx](https://fenicsproject.org/), version 0.10 at the time (Spring 2026) of writing this README. Tutorials for this library are available [online](https://jsdokken.com/dolfinx-tutorial/). 

## Installation
This code was tested with [Conda](https://anaconda.org/) version 26.1.1. 
The relevant packages can be installed with the file `environment.yml` using the command 
```
conda create --file environment.yml
```
See the [Conda official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information.

After this, run
```
pip install -e .
```
to install the repository as a library.

## Running the code
Several examples can be found in the `examples/` folder, and can be run directly from the terminal. There are L2 approximation examples in 2D, approximation a Poisson problem on a deformed mesh and linear elasticity approximations with adaptive refinement using dual weighted residuals when the underlying analytical solution is unknown.
Read for example _Adaptive Finite Element Methods for Differential Equations_ by _Wolfgang Bangerth_ and 
_Rolf Rannacher_ for a theoretical overview of the method.\
The script produces convergence plots and displays the convergence rate on a log-log plot when relevant.

All examples can also be run step-by-step as jupyter notebooks, located in the `notebooks` repository.