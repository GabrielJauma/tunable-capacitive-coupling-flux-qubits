import SQcircuit as sq
import numpy as np
import scipy as sp
import qutip as qt

# %% Constants
GHz  = 1e9
nH = 1e-9

# %% Premade circuits
def KIT_qubit(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):

    loop = sq.Loop(φ_ext)

    C_01 = sq.Capacitor(C,       'fF')
    C_02 = sq.Capacitor(C,       'fF')
    C_12 = sq.Capacitor(CJ+Csh,  'fF')
    L_03 = sq.Inductor(Lr,       'nH')
    L_31 = sq.Inductor(Lq/2 - Δ, 'nH',  loops=[loop])
    L_23 = sq.Inductor(Lq/2 + Δ, 'nH',  loops=[loop])
    JJ_12= sq.Junction(EJ,       'GHz', loops=[loop])

    elements = {
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31],
        (1, 2): [C_12, JJ_12],
        (2, 3): [L_23],
    }

    return sq.Circuit(elements)


def KIT_resonator(C = 15, Lq = 25, Lr = 10, Δ = 0.1):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    resonator_elements = {
        (0, 1): [sq.Capacitor(C / 2, 'fF'), sq.Inductor(l / Lq, 'nH')],
    }
    return sq.Circuit(resonator_elements)


def KIT_fluxonium(C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    loop_fluxonium = sq.Loop(φ_ext)
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C / 2 + Csh + CJ, 'fF'),
                 sq.Inductor(l / (Lq + 4 * Lr), 'nH', loops=[loop_fluxonium]),
                 sq.Junction(EJ, 'GHz', loops=[loop_fluxonium])],
    }
    return sq.Circuit(fluxonium_elements)


def KIT_qubit_no_JJ(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1):

    C_01 = sq.Capacitor(C,       'fF')
    C_02 = sq.Capacitor(C,       'fF')
    C_12 = sq.Capacitor(CJ+Csh,  'fF')
    L_03 = sq.Inductor(Lr,       'nH')
    L_31 = sq.Inductor(Lq/2 - Δ, 'nH')
    L_23 = sq.Inductor(Lq/2 + Δ, 'nH')

    elements = {
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31],
        (1, 2): [C_12],
        (2, 3): [L_23],
    }

    return sq.Circuit(elements)


def KIT_fluxonium_no_JJ(C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1 ):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C / 2 + Csh + CJ, 'fF'),
                 sq.Inductor(l / (Lq + 4 * Lr), 'nH')],
    }
    return sq.Circuit(fluxonium_elements)


# %% Premade hamiltonians of circuits
def hamiltonian_frc(fluxonium, resonator, Δ, Lq = 25, Lr = 10):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    H_f = fluxonium.hamiltonian()
    H_r = resonator.hamiltonian()

    I_f = qt.identity(H_f.shape[0])
    I_r = qt.identity(H_r.shape[0])

    Φ_f = fluxonium._memory_ops["phi"][0]
    Φ_r = resonator._memory_ops["phi"][0]

    H = qt.tensor(I_r, H_f) + qt.tensor(H_r, I_f) + qt.tensor(Φ_r, Φ_f) * 2 * Δ / l / 1e-9

    return H


# %%  Generic effective Hamiltonians
def H_eff_p1(circ_0, circ, out='GHz', real=True):
    ψ_0 = np.array([ψ_i.__array__()[:, 0] for ψ_i in circ_0._evecs]).T
    if real:
        ψ_0 = real_eigenvectors(ψ_0)

    H = circ.hamiltonian().__array__()
    H_eff = ψ_0.conj().T @ H @ ψ_0

    if out == 'GHz':
        H_eff /= GHz * 2 * np.pi

    if real:
        if np.allclose(np.imag(H_eff),0):
            H_eff = np.real(H_eff)

    return H_eff

def H_eff_p1_hamil(H_0, H, n_eig, out='GHz', real=True):
    ψ_0 = diag(H_0, n_eig, real=real)[1]
    H_eff = ψ_0.conj().T @ H.__array__() @ ψ_0

    if out == 'GHz':
        H_eff /= GHz * 2 * np.pi

    if real:
        if np.allclose(np.imag(H_eff),0):
            H_eff = np.real(H_eff)

    return H_eff


# %% Generic mathematical functions
def diag(H, n_eig=4, out=None, real='False'):
    H = qt.Qobj(H)
    efreqs, evecs = sp.sparse.linalg.eigs(H.data, n_eig, which='SR')

    efreqs_sorted = np.sort(efreqs.real)

    sort_arg = np.argsort(efreqs)
    if isinstance(sort_arg, int):
        sort_arg = [sort_arg]
    evecs_sorted = evecs[:, sort_arg]

    if real:
        evecs_sorted = real_eigenvectors(evecs_sorted)
    if out=='GHz':
        efreqs_sorted /= 2 * np.pi * GHz

    return efreqs_sorted, evecs_sorted


def eigs_sorted(w, v):
    """Sorts the eigenvalues in ascending order and the corresponding eigenvectors.

    Input:
    w=array containing the eigenvalues in random order.
    v=array representing the eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i].

    Output:
    w=array containing the eigenvalues in ascending order.
    v=array representing the eigenvectors."""

    ndx = np.argsort(w)  # gives the correct order for the numbers in v, from smallest to biggest.
    return w[ndx], v[:, ndx]


def real_eigenvectors(U):
    '''Fixes the phase of a vector.

    Input:
    U=vector

    Output:
    U= vector without phase.'''

    l = U.shape[0]
    try:
        avgz = np.sum(U[:l // 2, :] * np.abs(U[:l // 2, :]) ** 2, 0)
    except:
        avgz = np.sum(U[:l // 2] * np.abs(U[:l // 2]) ** 2, 0)
    avgz = avgz / np.abs(avgz)
    # U(i,j) = U(i,j) / z(j) for all i,j
    U = U * avgz.conj()
    return U

# %% Generic transforming, labeling and sorting functions
def get_node_variables(circuit, basis, isolated=False):
    n_modes = circuit.n
    if isolated:
        Φ_normal = [circuit._flux_op_isolated  (i) for i in range(n_modes)]
        Q_normal = [circuit._charge_op_isolated(i) for i in range(n_modes)]
    else:
        Φ_normal = [circuit.flux_op(i, basis=basis) for i in range(n_modes)]
        Q_normal = [circuit.charge_op(i, basis=basis) for i in range(n_modes)]

    Φ_nodes = []
    Q_nodes = []

    for i in range(n_modes):
        Φ_sum = 0
        Q_sum = 0
        for j in range(n_modes):
            Φ_sum += Φ_normal[j] * circuit.S[i, j]
            Q_sum += Q_normal[j] * circuit.R[i, j]

        Φ_nodes.append(Φ_sum)
        Q_nodes.append(Q_sum)

    return Φ_nodes, Q_nodes


def print_charge_transformation(circuit):
    '''
    Prints the transformed charge variables as a function of node variables.
    '''
    for i in range(circuit.R.shape[1]):
        normalized = circuit.R[:, i] / np.abs(circuit.R[:, i]).max()
        formatted_vector = [f"{num:5.3f}" for num in normalized]
        vector_str = ' '.join(formatted_vector)
        print(f'q_{i + 1} = [{vector_str}]')


def print_flux_transformation(circuit):
    '''
    Prints the transformed flux variables as a function of node variables.
    '''
    for i in range(circuit.S.shape[1]):
        normalized = circuit.S[:, i] / np.abs(circuit.S[:, i]).max()
        formatted_vector = [f"{num:5.3f}" for num in normalized]
        vector_str = ' '.join(formatted_vector)
        print(f'Φ_{i + 1} = [{vector_str}]')