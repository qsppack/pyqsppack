Nonlinear Fourier Analysis Module
=================================

The nonlinear Fourier analysis (NLFA) module provides functions for working with nonlinear Fourier transforms and related operations.

.. currentmodule:: qsppack.nlfa

.. autofunction:: b_from_cheb
.. autofunction:: weiss
.. autofunction:: inverse_nonlinear_FFT
.. autofunction:: forward_nlft

These functions provide essential operations for nonlinear Fourier analysis, including:
- Converting Chebyshev coefficients to complex polynomial coefficients
- Computing the Weiss algorithm for polynomial coefficients
- Performing inverse nonlinear FFT operations
- Computing forward nonlinear Fourier transforms

Example usage:

.. code-block:: python

    import numpy as np
    from qsppack.nlfa import b_from_cheb, weiss, inverse_nonlinear_FFT, forward_nlft

    # Convert Chebyshev coefficients to complex polynomial coefficients
    cheb_coefs = np.array([2, -1, 6, -7, 1])
    b_coefs = b_from_cheb(cheb_coefs, parity=0)

    # Compute Weiss algorithm
    a_coefs = weiss(b_coefs, N=8)

    # Perform inverse nonlinear FFT
    gammas, xi_n, eta_n = inverse_nonlinear_FFT(a_coefs, b_coefs)

    # Compute forward nonlinear Fourier transform
    result = forward_nlft(gammas) 