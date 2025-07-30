import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

def pauli_matrices():
    σ_x = np.array([[0, 1], [1, 0]], dtype='complex')
    σ_y = np.array([[0, -1], [1, 0]], dtype='complex') * 1j
    σ_z = np.array([[1, 0], [0, -1]], dtype='complex')

    return σ_x ,σ_y ,σ_z

#%%
def winding_number_angle(Hk, Nk=2000):
    """
    Compute the 1D winding number of a chiral (σx,σy) model H(k),
    by discretizing k in [-π,π).

    Parameters
    ----------
    Hk : function
        Callable Hk(k) returning a 2×2 complex ndarray at momentum k.
    Nk : int
        Number of k‑points (higher = more accurate).

    Returns
    -------
    ν : int
        The winding number.
    """

    σx, σy, σz = pauli_matrices()

    # sample k
    ks = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    # preallocate angle array
    phis = np.empty(Nk, dtype=float)

    for i, k in enumerate(ks):
        H = Hk(k)
        # project onto σx, σy
        hx = 0.5 * np.trace(H @ σz).real
        hy = 0.5 * np.trace(H @ σy).real
        phis[i] = np.arctan2(hy, hx)

    # unwrap the phase to remove 2π jumps
    phis_un = np.unwrap(phis)
    # finite differences
    dphi = np.diff(phis_un)
    # total change
    Δφ = np.sum(dphi)
    # winding = Δφ/(2π), rounded to nearest integer
    return int(np.rint(Δφ / (2*np.pi)))

#%% Momentum-space Hamiltonians
def momentum_H_unit_cell_boson_ladder(δ, g_Φ, g_c, g_q, k):
    σx, σy, σz = pauli_matrices()
    Hk = (g_c-g_q) * np.cos(k) * np.eye(2) + (δ/2 + (g_c+g_q)*np.cos(k)) * σz +  g_Φ * σx + 2 * np.sqrt(g_c*g_q) * np.sin(k) * σy

    return Hk

#%% Real space Hamiltonians


def real_H_unit_cell_boson_ladder(omega_c, omega_q, g_Φ, g_c, g_q, N=100):
    # Total Hilbert space dimension: 2 modes per cell
    dim = 2 * N

    # Initialize the Hamiltonian matrix (complex valued)
    H_real = np.zeros((dim, dim), dtype=np.complex128)

    # Loop over unit cells to build H_real
    for n in range(N):
        # Basis ordering: index 2*n for mode a, index 2*n+1 for mode b at cell n.
        idx_a = 2 * n  # mode a in cell n
        idx_b = 2 * n + 1  # mode b in cell n

        # Onsite terms
        H_real[idx_a, idx_a] += omega_c
        H_real[idx_b, idx_b] += omega_q

        # Intra-cell (onsite) coupling between a and b (and its Hermitian conjugate)
        H_real[idx_a, idx_b] += g_Φ
        H_real[idx_b, idx_a] += g_Φ

        # Inter-cell couplings: couple cell n with cell n+1
        n_next = (n + 1) % N
        idx_a_next = 2 * n_next  # a in cell n+1
        idx_b_next = 2 * n_next + 1  # b in cell n+1

        if n == N - 1:
            continue
        # Coupling between adjacent a modes (resonators): g_c (real, + sign)
        H_real[idx_a, idx_a_next] += g_c
        H_real[idx_a_next, idx_a] += g_c

        # Coupling between adjacent b modes (qubits): g_q (real, - sign)
        H_real[idx_b, idx_b_next] += -g_q
        H_real[idx_b_next, idx_b] += -g_q

        # Cross-coupling: from b in cell n to a in cell n+1: -np.sqrt(g_c * g_q)
        H_real[idx_b, idx_a_next] += -np.sqrt(g_c * g_q)
        H_real[idx_a_next, idx_b] += -np.sqrt(g_c * g_q)

        # Cross-coupling: from a in cell n to b in cell n+1: +np.sqrt(g_c * g_q)
        H_real[idx_a, idx_b_next] += np.sqrt(g_c * g_q)
        H_real[idx_b_next, idx_a] += np.sqrt(g_c * g_q)

    return H_real




def build_H_single_excitation(omega_c, omega_q, g_Φ, g_c, g_q, N, kappa=0.0, gamma=0.0, open_bc=True,
                              subtract_trace_average=True, dense=True):
    """
    Returns H (csr_matrix, shape (2L,2L)) for the single-excitation subspace.
    Optional local losses: kappa on cavities a_n, gamma on qubits b_n
    (included as -i*kappa/2 and -i*gamma/2 on the diagonal).
    """
    dim = 2*N
    H = lil_matrix((dim, dim), dtype=np.complex128)

    def ia(n): return 2*n
    def ib(n): return 2*n+1

    # On-site terms (allow non-Hermitian losses)
    for n in range(N):
        H[ia(n), ia(n)] += omega_c - 1j*kappa/2
        H[ib(n), ib(n)] += omega_q - 1j*gamma/2
        # On-site a<->b coupling g_Φ
        H[ia(n), ib(n)] += g_Φ
        H[ib(n), ia(n)] += g_Φ

    # Intercell terms
    for n in range(N-1 if open_bc else N):
        m = (n+1) % N
        # a_n <-> a_{n+1} with +g_c
        H[ia(n), ia(m)] += g_c
        H[ia(m), ia(n)] += g_c
        # b_n <-> b_{n+1} with -g_q
        H[ib(n), ib(m)] += -g_q
        H[ib(m), ib(n)] += -g_q
        # cross terms:
        # b_n <-> a_{n+1} with -sqrt(g_q*g_c)
        s = -np.sqrt(g_q*g_c)
        H[ib(n), ia(m)] += s
        H[ia(m), ib(n)] += s
        # a_n <-> b_{n+1} with +sqrt(g_q*g_c)
        s = +np.sqrt(g_q*g_c)
        H[ia(n), ib(m)] += s
        H[ib(m), ia(n)] += s

    H = H.tocsr()

    # Optional: subtract a multiple of identity to reduce spectral radius.
    # (This removes only a global phase; observables are unchanged.)
    if subtract_trace_average:
        tr_avg = (omega_q + omega_c)/2.0
        H = H - tr_avg * csr_matrix(np.eye(dim))

    if dense:
        H = np.array(H.todense())

    return H


def time_evolve(H, psi0, t_array):
    """
    psi(t) = exp(-i H t) psi0 for all t in t_array (in the same units as H).
    Uses SciPy's expm_multiply (Krylov). Returns array shape (T, dim).
    """
    # expm_multiply expects exp(t*A) with A a matrix. Here A = -i H.
    A = (-1j) * H
    t0, t1 = float(t_array[0]), float(t_array[-1])
    # SciPy can produce the whole trajectory in one shot:
    Ps = expm_multiply(A, psi0, start=t0, stop=t1, num=len(t_array), endpoint=True)
    # expm_multiply returns an array of shape (num, dim)
    return np.asarray(Ps)

