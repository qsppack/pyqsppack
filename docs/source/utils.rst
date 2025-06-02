Utilities Module
================

The utils module provides various utility functions for QSP operations.

.. currentmodule:: qsppack.utils

.. autofunction:: get_unitary
.. autofunction:: reduced_to_full
.. autofunction:: chebyshev_to_func
.. autofunction:: cvx_poly_coef

These utility functions provide essential operations for QSP calculations, including unitary matrix generation, phase factor conversion, and polynomial coefficient manipulation.

Example usage:

.. code-block:: python

    import numpy as np
    from qsppack.utils import get_unitary, reduced_to_full, chebyshev_to_func

    # Generate unitary matrix for given phase factors
    phase_factors = np.array([0.1, 0.2, 0.3])
    unitary = get_unitary(phase_factors)

    # Convert reduced phase factors to full set
    full_phases = reduced_to_full(phase_factors)

    # Convert Chebyshev coefficients to function values
    cheb_coefs = np.array([1, 0, 1])
    x = np.linspace(-1, 1, 100)
    func_values = chebyshev_to_func(cheb_coefs, x) 