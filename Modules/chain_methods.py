import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


def build_H_single_excitation(omega_c, omega_q, g_Φ, g_c, g_q, N, kappa=0.0, gamma=0.0, open_bc=True, dense=True):
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

    if dense:
        H = np.array(H.todense())

    return H

#%% Time evolution without driving

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

def basis_state(L, which="a", n=0):
    dim = 2*L
    psi = np.zeros(dim, dtype=np.complex128)
    idx = 2*n if which == "a" else 2*n+1
    psi[idx] = 1.0
    return psi

def left_edge_state(δ, g, N ):
    λ = δ / (4*g)
    r = (λ) **2
    N0 =  2 * np.sqrt( (1-r**N) / (1-r)  ) * np.sqrt(2)**-1
    a_R = 1/N0   * λ ** np.arange(N)
    return a_R

def observables_from_trajectory(Ps, L, n_edge=3):
    """
    Ps: array (T, 2L) of states over time.
    Returns:
      p_a, p_b: arrays (T, L)
      P_edge: array (T,)
    """
    T, dim = Ps.shape
    assert dim == 2*L
    p_a = np.abs(Ps[:, 0::2])**2
    p_b = np.abs(Ps[:, 1::2])**2
    P_edge = (p_a[:, :n_edge].sum(axis=1) + p_b[:, :n_edge].sum(axis=1))
    return p_a, p_b, P_edge

#%% Time evolution with driving

def augment_with_vacuum(H0):
    """H0: csr (2L x 2L). Return H_aug (D x D) with vacuum at index 0."""
    Z  = sp.csr_matrix((1, 1), dtype=complex)
    r1 = sp.csr_matrix((1, H0.shape[1]), dtype=complex)
    c1 = sp.csr_matrix((H0.shape[0], 1), dtype=complex)
    H_aug = sp.bmat([[Z, r1],
                     [c1, H0]], format='csr')
    return H_aug

def idx_vac(): return 0
def idx_a(n):  return 1 + 2*n
def idx_b(n):  return 1 + 2*n + 1

def drive_operator_b(N, j, D=None):
    """
    Drive operator on qubit j in the augmented single-excitation basis:
        D_j = |b_j><vac| + |vac><b_j|
    Basis order: [vac, a0, b0, a1, b1, ..., a_{N-1}, b_{N-1}]
    """
    if not (0 <= j < N):
        raise ValueError(f"site j={j} out of range for N={N}")
    if D is None:
        D = 1 + 2*N
    if D != 1 + 2*N:
        raise ValueError(f"D={D} inconsistent with N={N} (expected {1+2*N})")

    rows = [idx_vac(), idx_b(j)]
    cols = [idx_b(j) , idx_vac()]
    data = [1.0, 1.0]
    return sp.csr_matrix((data, (rows, cols)), shape=(D, D), dtype=complex)

class Drive:
    def __init__(self, site, Omega, omega, phi=0.0, t_on=0.0, t_off=np.inf):
        self.site  = int(site)
        self.Omega = float(Omega)
        self.omega = float(omega)
        self.phi   = float(phi)
        self.t_on  = float(t_on)
        self.t_off = float(t_off)
    def envelope(self, t):
        return 1.0 if (t >= self.t_on) and (t <= self.t_off) else 0.0
    def coeff(self, t):
        return self.Omega * np.cos(self.omega * t + self.phi) * self.envelope(t)

def evolve_with_drives(H0, N, drives, t_array, psi0_aug, rtol=1e-7, atol=1e-9, method='DOP853'):
    H_aug = augment_with_vacuum(H0)              # H0 is (2N x 2N)
    Ddim  = H_aug.shape[0]
    # sanity check
    if Ddim != 1 + 2*N:
        raise ValueError(f"H_aug has dim {Ddim}, expected {1+2*N} for N={N}")
    D_ops = [drive_operator_b(N, d.site, Ddim) for d in drives]

    def rhs(t, psi):
        accum = H_aug @ psi
        for d, Dj in zip(drives, D_ops):
            c = d.coeff(t)
            if c != 0.0:
                accum = accum + (c * (Dj @ psi))
        return -1j * accum

    sol = solve_ivp(rhs, (t_array[0], t_array[-1]), psi0_aug,
                    t_eval=t_array, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    # states is (T, D)
    return sol.y.T

def observables_augmented(states, L):
    """
    states: (T, 1+2L) on basis [vac, a0,b0,a1,b1,...].
    Returns:
        p_vac (T,), p_a (T,L), p_b (T,L), P_edge (T,)
    """
    T, D = states.shape
    assert D == 1 + 2*L
    p = np.abs(states)**2
    p_vac = p[:, 0]
    p_a = np.zeros((T, L))
    p_b = np.zeros((T, L))
    for n in range(L):
        p_a[:, n] = p[:, 1 + 2*n]
        p_b[:, n] = p[:, 1 + 2*n + 1]
    P_edge = (p_a[:, :3].sum(axis=1) + p_b[:, :3].sum(axis=1))
    return p_vac, p_a, p_b, P_edge

def plot_offsets(t, p_a, p_b):
    L = p_a.shape[1]
    fig, ax = plt.subplots()
    for i in range(L):
        ax.plot(t, p_a[:, i] + i, lw=1.2)
        ax.plot(t, p_b[:, i] + i, ':', lw=1.1)
        ax.hlines(i, t[0], t[-1], color='k', lw=0.5)
    ax.set_xlabel('time'); ax.set_ylabel('cell index (offset)')
    ax.set_title('RWA with qubit drive: a (solid), b (dotted)')
    fig.tight_layout()
    return fig, ax

#%% Spectroscopy with driving on first qubit
def drive_op_qubit(N, j, D=None):
    if not (0 <= j < N): raise ValueError("j out of range")
    if D is None: D = 1 + 2*N
    rows = [0, idx_b(j)]
    cols = [idx_b(j), 0]
    data = [1.0, 1.0]
    return sp.csr_matrix((data, (rows, cols)), shape=(D, D), dtype=complex)

def evolve_drive_time_sweep(H0, N, j_drive, Omega, omega_list, phi=0.0,
                            gamma=1e-3, T_trans=500.0, T_avg=200.0,
                            points_per_unit=50):
    """
    Returns: omegas, <total photon number> time-averaged over last T_avg
    """
    H_aug = augment_with_vacuum(H0) - 1j*sp.diags([0.0] + [gamma]*(2*N))
    Ddim = H_aug.shape[0]
    Dj = drive_op_qubit(N, j_drive, Ddim)

    psi0 = np.zeros(Ddim, dtype=complex); psi0[0] = 1.0

    def rhs(t, psi, omega):
        return -1j*(H_aug @ psi + Omega*np.cos(omega*t + phi)*(Dj @ psi))

    def run_one(omega):
        t0, t1 = 0.0, T_trans + T_avg
        nsteps = int((t1 - t0) * points_per_unit) + 1
        t_eval = np.linspace(t0, t1, nsteps)
        sol = solve_ivp(lambda t, y: rhs(t, y, omega),
                        (t0, t1), psi0, t_eval=t_eval,
                        method='DOP853', rtol=1e-7, atol=1e-9)
        p = np.abs(sol.y.T)**2  # (T, 1+2N)

        # photon number on the first resonator
        nph_t = p[:, idx_b(j_drive)]

        # time-average over last T_avg
        mask = t_eval >= (t1 - T_avg)
        return nph_t[mask].mean()

    Nph_avg = np.array([run_one(om) for om in omega_list])
    return np.array(omega_list), Nph_avg


