Objective Module
================

The objective module provides functions for computing objective functions and their gradients in QSP optimization.

.. currentmodule:: qsppack.objective

.. autofunction:: obj_sym
.. autofunction:: grad_sym
.. autofunction:: grad_sym_real

These functions are used internally by the optimization methods to evaluate the quality of phase factor solutions and compute gradients for optimization.

Example usage:

.. code-block:: python

    import numpy as np
    from qsppack.objective import obj_sym, grad_sym

    # Define phase factors and target polynomial
    phase_factors = np.array([0.1, 0.2, 0.3])
    target_poly = np.array([0, 1])

    # Compute objective value
    obj_value = obj_sym(phase_factors, target_poly)

    # Compute gradient
    gradient = grad_sym(phase_factors, target_poly) 