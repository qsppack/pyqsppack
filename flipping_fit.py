# import necessary dependencies
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn.functional import conv1d, pad
from torch.fft import fft
from torchaudio.transforms import FFTConvolve
import time
from qsppack import nlfa
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.signal import fftconvolve


def high_precision_convolution(a, b):
    """
    High precision convolution using mpmath.
    
    Parameters
    ----------
    a, b : arrays of mpmath complex numbers
    
    Returns
    -------
    array of mpmath complex numbers
    """
    from mpmath import mpc
    
    # Convert to lists for easier manipulation
    a_list = list(a)
    b_list = list(b)
    
    # Result length
    result_len = len(a_list) + len(b_list) - 1
    result = [mpc(0) for _ in range(result_len)]
    
    # Direct convolution
    for i in range(len(a_list)):
        for j in range(len(b_list)):
            result[i + j] += a_list[i] * b_list[j]
    
    return np.array(result)


def high_precision_inverse_nonlinear_FFT(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the inverse nonlinear FFT of a and b.

    Parameters
    ----------
    a : np.ndarray
        Input array a of length n.
    b : np.ndarray
        Input array b of length n.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing gammas, xi_n, and eta_n.
    
    Examples
    --------
    >>> inverse_nonlinear_FFT(np.array([0.1, -0.5, -0.6]), np.array([0.2, -0.5, 0.3]))[0]
    array([2., 1., 3.])
    """
    from mpmath import mp, mpf, mpc, sqrt, conj

    # Set the desired precision (e.g., 50 decimal places)
    mp.dps = 100

    n = len(a)

    # Step 1: base case
    if n == 1:
        # Convert numpy complex to mpmath complex properly
        b_complex = mpc(float(b[0].real), float(b[0].imag))
        a_complex = mpc(float(a[0].real), float(a[0].imag))
        gammas = np.array([b_complex/a_complex])
        eta_1 = np.array([1 / sqrt(1 + abs(gammas[0])**2)])
        xi_1 = gammas * eta_1
        return gammas, xi_1, eta_1
    
    # Step 2: first recursive call
    m = int(mp.ceil(n/2))
    gammas = np.zeros(n, dtype=object)
    gammas[:m], xi_m, eta_m = high_precision_inverse_nonlinear_FFT(a[:m], b[:m])

    # Step 3: compute coefficients of am and bm
    # Convert numpy arrays to mpmath arrays for high precision operations
    eta_m_sharp = np.append(mpc(0), eta_m[::-1])
    xi_m_sharp = np.append(mpc(0), xi_m[::-1])
    
    # Convert a and b to mpmath arrays for high precision convolution
    a_mp = np.array([mpc(float(x.real), float(x.imag)) for x in a])
    b_mp = np.array([mpc(float(x.real), float(x.imag)) for x in b])
    
    # Use high precision convolution
    am = (high_precision_convolution(eta_m_sharp, a_mp) + high_precision_convolution(xi_m_sharp, b_mp))[m:]
    bm = (high_precision_convolution(eta_m, b_mp) - high_precision_convolution(xi_m, a_mp))[m:]

    # Step 4: second recursive call
    gammas[m:], xi_mn, eta_mn = high_precision_inverse_nonlinear_FFT(am[:n-m], bm[:n-m])

    # Step 5: final calculation and output
    xi_n = high_precision_convolution(eta_m_sharp, xi_mn) + np.append(high_precision_convolution(xi_m, eta_mn), mpc(0))
    eta_n = np.append(high_precision_convolution(eta_m, eta_mn), mpc(0)) - high_precision_convolution(xi_m_sharp, xi_mn)
    return gammas, xi_n, eta_n


def f(x):
    if x < -1/2:
        return 0
    elif x >= -1/2 and x <= 1/2:
        return 2*x
    else:
        return 0
def f_vec(x):
    return np.array([f(xval) for xval in x])

x = np.linspace(-1, 1, 500)
y = f_vec(x)
degree = 61
coeffs = cheb.chebinterpolate(f_vec, degree)
coeffs[::2] = np.zeros(len(coeffs)//2) # enforce odd constraint

factor = 0.95
coeffs = factor * coeffs

# format and show
plt.grid()
plt.plot(x, y, label="exact")
plt.plot(x, cheb.chebval(x, coeffs), label=f"d={degree}")
plt.xlim(-1, 1)
plt.xlabel(r"$x$")
plt.ylabel(r"XRect$(x)$")
plt.legend()
plt.show()


bcoeffs = nlfa.b_from_cheb(coeffs[1::2], 1)

def objective_torch(x, P):
    """
    Computes the loss for the optimization problem.

    This function calculates the loss as the squared norm of the difference
    between the target tensor P and the convolution of x with its flipped version.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor for which the loss is computed.
    P : torch.Tensor
        The target tensor to compare against.

    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    x.requires_grad = True

    # Compute loss using squared distance function
    loss = torch.norm(P - FFTConvolve("full").forward(x, torch.flip(x, dims=[0])))**2
    return 1000*loss


# Set the size of the polynomial
N = len(bcoeffs)

# Generate a random polynomial of size N on the specified device
poly = torch.tensor(bcoeffs, dtype=torch.float32, device=device)

# # Define the granularity for padding
# granularity = 2 ** 25

# # Pad the polynomial to match the granularity
# P = pad(poly, (0, granularity - poly.shape[0]))

# # Compute the FFT of the padded polynomial
# ft = fft(P)

# # Normalize the polynomial using the maximum norm of its FFT
# P_norms = ft.abs()
# poly /= torch.max(P_norms)

# Compute the negative convolution of the polynomial with its flipped version
conv_p_negative = FFTConvolve("full").forward(poly, torch.flip(poly, dims=[0])) * -1

# Adjust the last element to ensure the norm condition
conv_p_negative[poly.shape[0] - 1] = 1 - torch.norm(poly) ** 2


def closure():
    """
    Closure function for the optimizer.

    This function zeroes the gradients, computes the loss using the objective_torch
    function, and performs backpropagation to compute the gradients.

    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    optimizer.zero_grad()
    loss = objective_torch(initial, conv_p_negative)
    loss.backward()
    return loss

# Set up optimizer
torch.manual_seed(55)
initial = torch.randn(poly.shape[0], device=device, requires_grad=True)
# initial = torch.ones(poly.shape[0], device=device, requires_grad=True)
# initial = torch.zeros(poly.shape[0], device=device, requires_grad=True)
initial = (initial / torch.norm(initial)).clone().detach().requires_grad_(True)
optimizer = torch.optim.LBFGS([initial], max_iter=1000, line_search_fn="strong_wolfe", tolerance_grad=1e-10)

# Perform the optimization step using the closure function and record the time
t0 = time.time()
optimizer.step(closure)
t1 = time.time()
total = t1 - t0

print(f'N: {N}')
print(f'Time: {total}')
print(f'Final: {closure().item()}')
print(f"# Iterations: {optimizer.state[optimizer._params[0]]['n_iter']}")
print("-----------------------------------------------------")


acoeffs = optimizer._params[0].detach().cpu().numpy()
if acoeffs[0] < 0:
    acoeffs = -acoeffs
# print(f"acoeffs: {acoeffs}")
print(f"bcoeffs: {bcoeffs[:5]}")
print("Running high precision inverse nonlinear FFT...")
gammas, _, _ = high_precision_inverse_nonlinear_FFT(acoeffs, bcoeffs)
# Convert mpmath complex numbers to numpy complex numbers
gammas = np.array([complex(float(g.real), float(g.imag)) for g in gammas])
# print(f"gammas: {gammas}")
new_a, new_b = nlfa.forward_nonlinear_FFT(gammas)
# print(f"new_a: {new_a}")
print(f"new_b: {new_b[:5]}")


# Verify constraint becomes satisfied
coefs_constraint = fftconvolve(acoeffs, np.flip(np.conj(acoeffs)), mode='full') + fftconvolve(bcoeffs, np.flip(np.conj(bcoeffs)), mode='full')
coefs_constraint_new = fftconvolve(new_a, np.flip(np.conj(new_a)), mode='full') + fftconvolve(new_b, np.flip(np.conj(new_b)), mode='full')

plt.plot(np.real(coefs_constraint))
plt.yscale("log")
plt.show()
plt.plot(np.real(coefs_constraint_new))
plt.yscale("log")
plt.show()


new_coeffs = np.zeros(len(coeffs))
new_a_coeffs = np.zeros(len(coeffs))
new_coeffs[1::2] = np.real(new_b[int(len(new_b)/2-1)::-1] + new_b[int(len(new_b)/2)::])
new_a_coeffs[1::2] = np.real(new_a[int(len(new_a)/2-1)::-1] + new_a[int(len(new_a)/2)::])

plt.grid()
plt.plot(x, y, label="exact")
plt.plot(x, cheb.chebval(x, coeffs), label=f"d={degree} original P")
# plt.plot(x, cheb.chebval(x, acoeffs), label=f"d={degree} original Q")
plt.plot(x, cheb.chebval(x, new_coeffs), label=f"d={degree} perturbed P")
# plt.plot(x, cheb.chebval(x, new_a_coeffs), label=f"d={degree} perturbed Q")
plt.xlim(-1, 1)
plt.xlabel(r"$x$")
plt.ylabel(r"XRect$(x)$")
plt.legend()
plt.title("Finding Complementary with Wiebe/Motlagh Optimization")
plt.show()