import numpy as np
from numpy.polynomial import chebyshev as cheb
from qsppack import nlfa
from scipy.fftpack import dct
from scipy.signal import convolve
from matplotlib import pyplot as plt
from numpy.polynomial import Chebyshev



# Define the piecewise odd function
def piecewise_odd_function(x, delta, x0):
    if x < 0:
        return -piecewise_odd_function(-x, delta, x0)
    elif x < delta:
        return x/delta
    elif x < x0-delta:
        return 1
    elif x < x0:
        return 1-(x-(x0-delta))/delta
    else:
        return 0


# Generate x values over the interval [-1, 1]
x_values = np.linspace(-1, 1, 1000)

# Evaluate the function at these x values
y_values = np.array([piecewise_odd_function(x, 0.05, 0.9) for x in x_values])

plt.plot(x_values, y_values, 'b-', label=f'Original Function')
plt.xlabel('x (Standard Interval [-1, 1])')
plt.ylabel('y')
plt.legend()
plt.show()

# Find the Chebyshev approximation
degrees = np.arange(3, 100, 2)
linf_ta = np.zeros(len(degrees))
linf_tn = np.zeros(len(degrees))
linf_ac = np.zeros(len(degrees))
linf_an = np.zeros(len(degrees))

for i, n_degree in enumerate(degrees):
    cheb_approx = Chebyshev.fit(x_values, y_values, deg=n_degree)

    # Extract the coefficients, ensuring odd parity
    coeffs = np.zeros(n_degree + 1)
    coeffs[1::2] = cheb_approx.coef[1::2]  # Only keep odd coefficients

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

    linf_ta[i] = np.linalg.norm(old_y_poly - y_values, ord=2)
    linf_tn[i] = np.linalg.norm(new_y_poly - y_values, ord=2)
    linf_ac[i] = np.linalg.norm(old_y_poly - clipped_y_poly, ord=2)
    linf_an[i] = np.linalg.norm(new_y_poly - old_y_poly, ord=2)

# coefs_constraint = convolve(acoeffs, np.flip(np.conj(acoeffs)), mode='full') + convolve(bcoeffs, np.flip(np.conj(bcoeffs)), mode='full')
# coefs_constraint_new = convolve(new_a, np.flip(np.conj(new_a)), mode='full') + convolve(new_b, np.flip(np.conj(new_b)), mode='full')


# Plot the coefficients
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# xvals = np.linspace(-1, 1, len(coefs_constraint))
# axs[0].plot(xvals, np.real(coefs_constraint))
# axs[0].set_yscale("log")
# axs[0].set_title(r"Original Coefficients $aa^* + bb^*$")
# axs[1].plot(xvals, np.real(coefs_constraint_new))
# axs[1].set_yscale("log")
# axs[1].set_title(r"New Coefficients $aa^* + bb^*$")
# plt.tight_layout()
# plt.show()


# --- Plot the results ---
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(degrees, linf_ta, 'r-.', label=r'$\ell_2$ of true - approx')
plt.plot(degrees, linf_tn, 'g-.', label=r'$\ell_2$ of true - post-NLFT')
plt.plot(degrees, linf_ac, 'b-.', label=r'$\ell_2$ of approx - clipped')
plt.plot(degrees, linf_an, 'y-.', label=r'$\ell_2$ of post-NLFT - approx')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()



# # Plot the post-NLFT approximation constrained
# ax.plot(x_values, y_values, 'b-', label=f'Original Function')
# ax.plot(x_values, old_y_poly, 'r--', label=f'Odd Polynomial Approximation (n={n_degree})')
# ax.plot(x_values, new_y_poly, 'g--', label=f'Odd Polynomial Approximation Post-NLFT (n={n_degree})')
# ax.set_xlabel('x (Standard Interval [-1, 1])')
# ax.set_ylabel('y')
# ax.legend()
# plt.show()

# plt.plot(x_values, np.abs(old_y_poly - new_y_poly), 'b-', label=f'First Approximation - Post-NLFT Approximation')
# plt.plot(x_values, np.abs(old_y_poly - np.clip(old_y_poly, -1, 1)), 'r-', label=f'First Approximation - Clipped')
# plt.yscale("log")
# plt.legend()
# plt.show()