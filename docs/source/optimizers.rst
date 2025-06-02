Optimizers Module
=================

The optimizers module provides various optimization methods for QSP phase factor optimization.

.. currentmodule:: qsppack.optimizers

.. autofunction:: lbfgs
.. autofunction:: coordinate_minimization
.. autofunction:: newton

These optimization methods can be used directly or through the main :func:`qsppack.solve` function.

Example usage:

.. code-block:: python

    import numpy as np
    from qsppack.optimizers import lbfgs, coordinate_minimization, newton

    # Define target polynomial
    target_poly = np.array([0, 1])

    # Use L-BFGS optimizer
    result_lbfgs = lbfgs(target_poly)

    # Use coordinate minimization
    result_coord = coordinate_minimization(target_poly)

    # Use Newton's method
    result_newton = newton(target_poly) 