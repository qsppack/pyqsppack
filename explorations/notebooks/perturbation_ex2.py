import numpy as np
from numpy.polynomial import chebyshev as cheb
from qsppack import nlfa
from scipy.fftpack import dct
from scipy.signal import convolve
from matplotlib import pyplot as plt
from numpy.polynomial import Chebyshev



# Find the Chebyshev approximation
degree = 51
scaling_factors = np.linspace(10/11, 10, 100)
# scaling_factors = np.array([10/11, 1.5])
linf_ac = np.zeros(len(scaling_factors))
linf_an = np.zeros(len(scaling_factors))
x_values = np.linspace(-1, 1, 2000)

for i, scaling_factor in enumerate(scaling_factors):

    # Extract the coefficients, ensuring odd parity
    coeffs = np.zeros(degree + 1)
    coeffs[1::2] = np.random.rand(int((degree + 1) / 2))
    coeffs = coeffs / np.max(np.abs(cheb.chebval(x_values, coeffs))) * scaling_factor

    bcoeffs = nlfa.b_from_cheb(coeffs[1::2], 1)
    acoeffs = nlfa.weiss(bcoeffs, 2**9)
    acoeffs = acoeffs.astype(np.float64)
    bcoeffs = bcoeffs.astype(np.float64)

    gammas, _, _ = nlfa.inverse_nonlinear_FFT(acoeffs, bcoeffs)
    new_a, new_b = nlfa.forward_nonlinear_FFT(gammas)

    new_coeffs = np.zeros(len(coeffs))
    new_coeffs[1::2] = new_b[int(len(new_b)/2-1)::-1] + new_b[int(len(new_b)/2)::]
    old_y_poly = cheb.chebval(x_values, coeffs)
    new_y_poly = cheb.chebval(x_values, new_coeffs)
    clipped_y_poly = np.clip(old_y_poly, -1, 1)

    linf_ac[i] = np.linalg.norm(old_y_poly - clipped_y_poly, ord=np.inf)
    linf_an[i] = np.linalg.norm(new_y_poly - old_y_poly, ord=np.inf)
    print(f"Scaling factor: {scaling_factor}, linf_ac: {linf_ac[i]}, linf_an: {linf_an[i]}")
    # plt.plot(x_values, old_y_poly, 'r-', label=f'Original Function')
    # plt.show()


# --- Plot the results ---
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(1/scaling_factors, linf_ac, 'b-.', label=r'$\ell_\infty$ of approx - clipped')
plt.plot(1/scaling_factors, linf_an, 'y-.', label=r'$\ell_\infty$ of post-NLFT - approx')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("1/Scaling Factor")
plt.legend()
plt.show()