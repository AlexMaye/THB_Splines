# THBSplines

## Truncated Hierarchical B-Splines in Python

This repository contains a Python-implementation of truncated hierarchical B-Splines, based on the article [Multi-level Bézier extraction for hierarchical local refinement of Isogeometric Analysis](https://doi.org/10.1016/j.cma.2017.08.017).

The code structure is based on the article [Algorithms for the implementation of adaptive isogeometric methods using hierarchical B-splines](https://doi.org/10.1016/j.apnum.2017.08.006).

This code does not manage the construction of mass/stiffness matrices. Instead, it is intended to be used with [Fenicsx](https://fenicsproject.org/), version 0.10 at the time (Spring 2026) of writing this README. Tutorials for this library are available [online](https://jsdokken.com/dolfinx-tutorial/). 


This work was initially based on the [THBSplines repository writen by qTipTip](https://github.com/qTipTip/THBSplines), but only retains a small portion of the original code.