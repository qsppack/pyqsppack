Solver Module
=============

The solver module provides the main functionality for solving Quantum Signal Processing (QSP) problems.

.. currentmodule:: qsppack.solver

.. autofunction:: solve

The solve function is the main entry point for QSP optimization. It takes a target polynomial and returns the optimized phase factors.

Example usage:

.. code-block:: python

    import numpy as np
    from qsppack.solver import solve

    # Define a target polynomial (e.g., P(x) = x)
    target_poly = np.array([0, 1])

    # Solve for phase factors
    result = solve(target_poly, method='lbfgs') 