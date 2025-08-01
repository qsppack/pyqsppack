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


# correct nonlinear FFT
def inverse_nonlinear_FFT(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    n = len(a)

    # Step 1: base case
    if n == 1:
        gammas = np.array([b[0]/a[0]])
        eta_1 = np.array([1 / np.sqrt(1 + np.abs(gammas[0])**2)])
        xi_1 = gammas * eta_1
        return gammas, xi_1, eta_1
    
    # Step 2: first recursive call
    m = int(np.ceil(n/2))
    gammas = np.zeros(n, dtype=np.complex128)
    gammas[:m], xi_m, eta_m = inverse_nonlinear_FFT(a[:m], b[:m])

    # Step 3: compute coefficients of am and bm
    eta_m_sharp = np.append(0, eta_m[::-1])
    xi_m_sharp = np.append(0, xi_m[::-1])
    am = (fftconvolve(eta_m_sharp, a) + fftconvolve(xi_m_sharp, b))[m:]
    bm = (fftconvolve(eta_m, b) - fftconvolve(xi_m, a))[m:]

    # Step 4: second recursive call
    gammas[m:], xi_mn, eta_mn = inverse_nonlinear_FFT(am[:n-m], bm[:n-m])

    # Step 5: final calculation and output
    xi_n = fftconvolve(eta_m_sharp, xi_mn) + np.append(fftconvolve(xi_m, eta_mn), 0)
    eta_n = np.append(fftconvolve(eta_m, eta_mn), 0) - fftconvolve(xi_m_sharp, xi_mn)
    return gammas, xi_n, eta_n


def forward_nonlinear_FFT(gammas: np.ndarray, m=0, debug=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward nonlinear FFT, producing the polynomials a^* and b
    from the rotation parameters gammas.

    Parameters
    ----------
    gammas : np.ndarray
        Array of complex rotation parameters gamma_k of length n.
    m : int, optional
        starting index of gammas to use in the recursion.

    Returns
    -------
    tuple of np.ndarray
        A tuple (a_star, b) of length n+1 and n respectively, where a_star
        is the conjugate polynomial coefficients of a^*(z), and b is b(z).
    """
    n = len(gammas)

    # base case
    if n <= 2:
        prefactor = 1 / np.sqrt(1 + np.abs(gammas[0])**2)
        if n == 2:
            prefactor /= np.sqrt(1 + np.abs(gammas[1])**2)
        if debug:
            prefactor = 1
        b = prefactor * np.append(np.zeros(m, dtype=np.complex128), gammas)
        a_star = prefactor * np.array([1, -np.conj(gammas[0])*gammas[1]]) if n == 2 else np.array([prefactor])
        return a_star, b
    
    # recursive step
    m_new = int(np.ceil(n/2))
    if debug:
        print("m={}, n={}".format(m, n))
    a_star_left, b_left = forward_nonlinear_FFT(gammas[:m_new], debug=debug)
    a_star_right, b_right = forward_nonlinear_FFT(gammas[m_new:], m_new, debug=debug)
    
    # compute the convolution of the left and right parts
    if debug:
        print("a_star_left: ", a_star_left)
        print("b_left: ", b_left)
        print("a_star_right: ", a_star_right)
        print("b_right: ", b_right)
        print("b1: ", fftconvolve(np.conj(a_star_left[::-1]), b_right))
        print("b2: ", fftconvolve(a_star_right, b_left))
    b1 = fftconvolve(np.conj(a_star_left[::-1]), b_right)
    b1 = b1[len(a_star_left)-1:]
    b2 = fftconvolve(a_star_right, b_left)
    b2 = np.append(b2, np.zeros(len(b1)-len(b2), dtype=np.complex128))
    a_star1 = -fftconvolve(np.conj(b_left[::-1]), b_right)
    a_star1 = a_star1[len(b_left)-1:]
    a_star2 = fftconvolve(a_star_left, a_star_right)
    a_star2 = np.append(a_star2, np.zeros(len(a_star1)-len(a_star2), dtype=np.complex128))
    b = b1 + b2
    a_star = a_star1 + a_star2

    return a_star, np.append(np.zeros(m, dtype=np.complex128), b)



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

factor = 1
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


# Set the size of the polynomial
N = len(bcoeffs)

# Generate a random polynomial of size N on the specified device
poly = torch.tensor(bcoeffs, dtype=torch.float32, device=device)

# Define the granularity for padding
granularity = 2 ** 25

# Pad the polynomial to match the granularity
P = pad(poly, (0, granularity - poly.shape[0]))

# Compute the FFT of the padded polynomial
ft = fft(P)

# Normalize the polynomial using the maximum norm of its FFT
P_norms = ft.abs()
poly /= torch.max(P_norms)

# Compute the negative convolution of the polynomial with its flipped version
conv_p_negative = FFTConvolve("full").forward(poly, torch.flip(poly, dims=[0])) * -1

# Adjust the last element to ensure the norm condition
conv_p_negative[poly.shape[0] - 1] = 1 - torch.norm(poly) ** 2

# Set up optimizer
initial = torch.randn(poly.shape[0], device=device, requires_grad=True)
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


acoeffs = initial.detach().numpy()
# print(f"acoeffs: {acoeffs}")
print(f"bcoeffs: {bcoeffs[:5]}")
gammas, _, _ = inverse_nonlinear_FFT(acoeffs, bcoeffs)
# print(f"gammas: {gammas}")
new_a, new_b = forward_nonlinear_FFT(gammas)
# print(f"new_a: {new_a}")
print(f"new_b: {new_b[:5]}")


coefs_constraint = fftconvolve(acoeffs, np.flip(np.conj(acoeffs)), mode='full') + fftconvolve(bcoeffs, np.flip(np.conj(bcoeffs)), mode='full')
coefs_constraint_new = fftconvolve(new_a, np.flip(np.conj(new_a)), mode='full') + fftconvolve(new_b, np.flip(np.conj(new_b)), mode='full')

plt.plot(np.real(coefs_constraint))
plt.yscale("log")
plt.show()
plt.plot(np.real(coefs_constraint_new))
plt.yscale("log")
plt.show()


new_coeffs = np.zeros(len(coeffs))
new_coeffs[1::2] = new_b[int(len(new_b)/2-1)::-1] + new_b[int(len(new_b)/2)::]

plt.grid()
plt.plot(x, y, label="exact")
plt.plot(x, cheb.chebval(x, coeffs), label=f"d={degree} original")
plt.plot(x, cheb.chebval(x, new_coeffs), label=f"d={degree} perturbed")
plt.xlim(-1, 1)
plt.xlabel(r"$x$")
plt.ylabel(r"XRect$(x)$")
plt.legend()
plt.title("Finding Complementary with Wiebe/Motlagh Optimization")
plt.show()