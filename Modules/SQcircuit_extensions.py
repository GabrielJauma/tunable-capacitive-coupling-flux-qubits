import SQcircuit as sq
import Modules.figures as figs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import qutip as qt

plt.rcParams['backend'] = 'QtAgg'

#%% Constants
GHz  = 1e9
nH = 1e-9
Phi0 = 2.067833831e-15  # Flux quantum (in Wb)
hbar = 1.0545718e-34

#%% Premade circuits
def KIT_qubit(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5, trunc_res=15, trunc_flux=25):

    # Initialize loop(s)
    loop = sq.Loop(φ_ext)

    # Circuit components
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

    qubit = sq.Circuit(elements)
    try:
        qubit.set_trunc_nums([trunc_res,trunc_flux])
    except:
        qubit.set_trunc_nums([1, trunc_res, trunc_flux])

    # Create and return the circuit
    return qubit

def KIT_qubit_no_JJ(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, φ_ext=0.5):

    # Initialize loop(s)
    # loop = sq.Loop(φ_ext)

    # Circuit components
    C_01 = sq.Capacitor(C,       'fF')
    C_02 = sq.Capacitor(C,       'fF')
    C_12 = sq.Capacitor(CJ+Csh,  'fF')
    L_03 = sq.Inductor(Lr,       'nH')
    L_31 = sq.Inductor(Lq/2 - Δ, 'nH')#,  loops=[loop])
    L_23 = sq.Inductor(Lq/2 + Δ, 'nH')#,  loops=[loop])

    elements = {
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31],
        (1, 2): [C_12],
        (2, 3): [L_23],
    }

    # Create and return the circuit
    return sq.Circuit(elements)


def KITqubit_asym( Cc, α, C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):

    # Initialize loop(s)
    loop = sq.Loop(φ_ext)

    # Circuit components
    C_01 = sq.Capacitor(C + α*Cc, 'fF')
    C_02 = sq.Capacitor(C + Cc,   'fF')
    C_12 = sq.Capacitor(CJ+Csh,   'fF')
    L_03 = sq.Inductor(Lr,        'nH')
    L_31 = sq.Inductor(Lq/2 - Δ,  'nH',  loops=[loop])
    L_23 = sq.Inductor(Lq/2 + Δ,  'nH',  loops=[loop])
    JJ_12= sq.Junction(EJ,        'GHz', loops=[loop])

    elements = {
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31],
        (1, 2): [C_12, JJ_12],
        (2, 3): [L_23],
    }

    # Create and return the circuit
    return sq.Circuit(elements)


def KIT_resonator(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5, trunc_res=15, trunc_flux=25):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    resonator_elements = {
        (0, 1): [sq.Capacitor(C / 2, 'fF'), sq.Inductor(l / Lq, 'nH')],
    }
    resonator = sq.Circuit(resonator_elements)
    resonator.set_trunc_nums([trunc_res])
    return resonator


def KIT_fluxonium(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5, trunc_res=15, trunc_flux=25):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    loop_fluxonium = sq.Loop(φ_ext)
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C / 2 + Csh + CJ, 'fF'),
                 sq.Inductor(l / (Lq + 4 * Lr), 'nH', loops=[loop_fluxonium]),
                 sq.Junction(EJ, 'GHz', loops=[loop_fluxonium])],
    }
    fluxonium = sq.Circuit(fluxonium_elements)
    fluxonium.set_trunc_nums([trunc_flux])
    return fluxonium

def KIT_fluxonium_no_JJ(C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1 ):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C / 2 + Csh + CJ, 'fF'),
                 sq.Inductor(l / (Lq + 4 * Lr), 'nH')],
    }
    return sq.Circuit(fluxonium_elements)

def KIT_qubit_vs_param(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5, trunc_res=15, trunc_flux=25, model='composition'):

    parameters_list = expand_list_with_array([C, CJ, Csh, Lq, Lr, Δ, EJ, φ_ext, trunc_res, trunc_flux])
    H_qubit_list = []

    for parameters in parameters_list:
        if model == 'full_circuit':
            H_qubit_list.append(KIT_qubit(*parameters).hamiltonian())
        if model == 'composition':
            fluxonium = KIT_fluxonium(*parameters)
            resonator = KIT_resonator(*parameters)
            H_qubit_list.append(hamiltonian_frc(fluxonium, resonator, Δ = parameters[5], Lq = parameters[3], Lr = parameters[4]))

    return H_qubit_list

#%% Premade hamiltonians of circuits
def hamiltonian_frc(fluxonium, resonator, Δ, Lq = 25, Lr = 10, factor=1):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    H_f = fluxonium.hamiltonian()
    H_r = resonator.hamiltonian()

    I_f = qt.identity(H_f.shape[0])
    I_r = qt.identity(H_r.shape[0])

    Φ_f = fluxonium.flux_op(0)
    Φ_r = resonator.flux_op(0)

    H = qt.tensor(I_r, H_f) + qt.tensor(H_r, I_f) + factor * qt.tensor(Φ_r, Φ_f) * 2 * Δ / l / 1e-9

    return H


def hamiltonian_qubit_C_qubit(nmax_r, nmax_f, Cc, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, trunc_res=15, trunc_flux=25):
    resonator = KIT_resonator(C = C+Cc,                     Lq = Lq, Lr = Lr, Δ = Δ, trunc_res =trunc_res)
    fluxonium = KIT_fluxonium(C = C+Cc, CJ = CJ, Csh = Csh, Lq = Lq, Lr = Lr, Δ = Δ, trunc_flux=trunc_flux)

    H_left  = hamiltonian_frc(fluxonium, resonator, Δ)
    H_right = hamiltonian_frc(fluxonium, resonator, Δ)

    I_r = qt.identity(nmax_r)
    I_f = qt.identity(nmax_f)
    I_H_left = qt.identity(H_left.dims[0])
    I_H_right = qt.identity(H_right.dims[0])

    q_r = qt.tensor(resonator.charge_op(0), I_f)
    q_f = qt.tensor(I_r, fluxonium.charge_op(0))

    if Cc > 0:
        H = qt.tensor(H_left, I_H_right) + qt.tensor(I_H_left, H_right) + 1 / Cc * (qt.tensor(q_r, q_r) +
                                                                                    qt.tensor(q_f, q_f) +
                                                                                    qt.tensor(q_r, q_f) +
                                                                                    qt.tensor(q_f, q_r))
    else:
        H = qt.tensor(H_left, I_H_right) + qt.tensor(I_H_left, H_right)

    return H


# %% KIT's qubit internal coupling perturbation theory with fluxonium + resonator decomposition
def H_eff_p1_fluxonium_resonator(fluxonium_0, fluxonium, resonator_0, resonator, N_f, N_r, Δ, Lq = 25, Lr = 10):
    '''
    DANGER, N_f and N_r shold be the enery levels of the non interacting resonator and fluxonium.
    Now its fine because they are the same as the interacting ones, but careful.
    '''
    l   = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    ψ_0_f = np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium_0._evecs]).T
    ψ_0_r = np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator_0._evecs]).T
    Φ_f =  fluxonium.flux_op(0).__array__()
    Φ_r =  resonator.flux_op(0).__array__()

    n_eig = ψ_0_f.shape[1]

    H_eff_p1 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.

    for i in range(n_eig):
        for j in range(n_eig):
            H_eff_1_f_i_j = np.abs(ψ_0_f[:, N_f[i]].conj().T @ Φ_f @ ψ_0_f[:, N_f[j]])
            H_eff_1_r_i_j = np.abs(ψ_0_r[:, N_r[i]].conj().T @ Φ_r @ ψ_0_r[:, N_r[j]])
            H_eff_p1[i,j] = H_eff_1_f_i_j*H_eff_1_r_i_j

    return H_eff_p1 * 2 * Δ / l / 1e-9  / (2 * np.pi * GHz)


def H_eff_p2_fluxonium_resonator(fluxonium_0, fluxonium, resonator_0, resonator, N_f, N_r, Δ, Lq = 25, Lr = 10):
    '''
    DANGER, N_f and N_r shold be the enery levels of the non interacting resonator and fluxonium.
    Now its fine because they are the same as the interacting ones, but careful.
    '''
    l   = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    ψ_0_f = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium_0._evecs]).T)
    ψ_f   = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium  ._evecs]).T)

    ψ_0_r = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator_0._evecs]).T)
    ψ_r   = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator  ._evecs]).T)

    Φ_f =  fluxonium.flux_op(0).__array__() / (2 * np.pi * GHz)
    Φ_r =  resonator.flux_op(0).__array__() / (2 * np.pi * GHz)

    n_eig = ψ_0_f.shape[1]
    H_eff_p2 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.

    for i in range(n_eig):
        E_0_i = fluxonium_0.efreqs[N_f[i]] + resonator_0.efreqs[N_r[i]]
        for j in range(n_eig):
            E_0_j = fluxonium_0.efreqs[N_f[j]] + resonator_0.efreqs[N_r[j]]
            for k in range(n_eig):
                E_k = fluxonium.efreqs[N_f[k]] + resonator.efreqs[N_r[k]]
                H_eff_2_f_ijk = ψ_0_f[:, N_f[i]].conj().T @ Φ_f @ ψ_f  [:, N_f[k]] *  \
                                  ψ_f[:, N_f[k]].conj().T @ Φ_f @ ψ_0_f[:, N_f[j]]

                H_eff_2_r_ijk = ψ_0_r[:, N_r[i]].conj().T @ Φ_r @ ψ_r  [:, N_r[k]] *  \
                                  ψ_r[:, N_r[k]].conj().T @ Φ_r @ ψ_0_r[:, N_r[j]]

                coef = 1 / (E_0_i-E_k) + 1 / (E_0_j-E_k)
                H_eff_p2[i,j] += coef * np.abs(H_eff_2_f_ijk * H_eff_2_r_ijk) #/  GHz #/ 2 / np.pi

    return Δ**2/2  * H_eff_p2 * 2 * Δ / l / 1e-9



# %%  Generic effective Hamiltonians

def H_eff_p1(H_0, H, n_eig, out='GHz', real=True, remove_ground = False, solver='scipy'):

    ψ_0 = diag(H_0, n_eig, real=real, solver=solver)[1]

    H_eff = ψ_0.conj().T @ H.__array__() @ ψ_0

    if out == 'GHz':
        H_eff /= GHz * 2 * np.pi

    if remove_ground:
        H_eff -=  H_eff[0,0]*np.eye(len(H_eff))

    if real:
        if np.allclose(np.imag(H_eff),0):
            H_eff = np.real(H_eff)

    return H_eff


def H_eff_p2(H_0, H, n_eig, out='GHz', real=True, remove_ground=False, solver='scipy'):
    E_0, ψ_0 = diag(H_0, n_eig, real=real, solver=solver, out=out)
    E,   ψ   = diag(H  , n_eig, real=real, solver=solver, out=out)
    H_0 = H_0.__array__()
    H   = H  .__array__()
    V = H - H_0

    if out == 'GHz':
        H_0 /= GHz * 2 * np.pi
        H   /= GHz * 2 * np.pi
        V   /= GHz * 2 * np.pi

    H_eff_1 = ψ_0.conj().T @ H @ ψ_0

    H_eff_2 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.

    for i in range(n_eig):
        for j in range(n_eig):
            H_eff_2[i, j] = 1 / 2 * sum(
                          (1 / (E_0[i] - E[k]) + 1 / (E_0[j] - E[k])) *
                           (ψ_0[:, i].T.conj() @ V @ ψ  [:, k]) *
                           (ψ  [:, k].T.conj() @ V @ ψ_0[:, j])
                           for k in range(n_eig))

    # H_eff = H_eff_1 + H_eff_2
    H_eff = H_eff_2


    # if out == 'GHz':
    #     H_eff /= GHz * 2 * np.pi

    if remove_ground:
        H_eff -= H_eff[0, 0] * np.eye(len(H_eff))

    if real:
        if np.allclose(np.imag(H_eff), 0):
            H_eff = np.real(H_eff)

    return H_eff


def H_eff_SWT(H_0, H, n_eig, out='GHz', real=True, remove_ground=False, solver='scipy',return_transformation=False):

    ψ_0  = diag(H_0, n_eig, real=real, solver=solver) [1]
    E, ψ = diag(H  , n_eig, real=real, solver=solver)

    Q = ψ_0.T.conj() @ ψ
    U, s, Vh = np.linalg.svd(Q)
    A = U @ Vh

    H_eff = A @ np.diag(E) @ A.T.conj()

    if out == 'GHz':
        H_eff /= GHz * 2 * np.pi

    if remove_ground:
        H_eff -= H_eff[0, 0] * np.eye(len(H_eff))

    if real:
        if np.allclose(np.imag(H_eff), 0):
            H_eff = np.real(H_eff)

    if return_transformation:
        return H_eff, A
    else:
        return H_eff



#%% Sorting and labeling functions for the fluxonum + resonator model
def get_energy_indices(qubit, fluxonium, resonator, n_eig=3):

    try:
        E_qubit = qubit.efreqs - qubit.efreqs[0]
        E_fluxonium = fluxonium.efreqs - fluxonium.efreqs[0]
        E_resonator = resonator.efreqs - resonator.efreqs[0]
    except:
        qubit    .diag(n_eig+2)
        fluxonium.diag(n_eig)
        resonator.diag(n_eig)
        E_qubit      = qubit.efreqs    - qubit.efreqs[0]
        E_fluxonium = fluxonium.efreqs - fluxonium.efreqs[0]
        E_resonator = resonator.efreqs - resonator.efreqs[0]

    n_eig = len(E_qubit)

    N_fluxonium = np.zeros(n_eig, dtype='int')
    N_resonator = np.zeros(n_eig, dtype='int')

    E_matrix = E_fluxonium[:, np.newaxis] + E_resonator

    tol = E_qubit[1]-E_qubit[0]
    for k in range(n_eig):
        ΔE_matrix = np.abs(E_matrix - E_qubit[k])
        if ΔE_matrix.min() < tol:
            N_fluxonium[k], N_resonator[k] = np.unravel_index(ΔE_matrix.argmin(), ΔE_matrix.shape)
        else:
            N_fluxonium[k], N_resonator[k] = [-123, -123]
    return N_fluxonium, N_resonator


#%% Operators
def internal_coupling_fluxonium_resonator(fluxonium, resonator, Δ, Lq = 25, Lr = 10):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    Φ_r = resonator.flux_op(0)
    Φ_f = fluxonium.flux_op(0)

    return qt.tensor(Φ_r, Φ_f) * 2 * Δ / l / 1e-9

def resonator_N_operator(resonator, Z_r, clean=True):
    Φ_nodes, Q_nodes = get_node_variables(resonator)
    Φ_r = Φ_nodes[0] + Φ_nodes[1]
    Q_r = Q_nodes[0] + Q_nodes[1]
    N_r = 1 / 2 / Z_r * (Φ_r ** 2 + Z_r ** 2 * Q_r ** 2)
    if clean:
        try:
            return rank_by_multiples(np.diag(N_r[:len(N_r.__array__()) // 2]))
        except:
            return np.diag(N_r[:len(N_r.__array__()) // 2])
    else:
        return N_r


#%% Generic mathematical functions
def diag(H, n_eig=4, out=None, real=False, solver='scipy'):
    H = qt.Qobj(H)

    if solver == 'scipy':
        efreqs, evecs = sp.sparse.linalg.eigs(H.data, n_eig, which='SR')
        # efreqs, evecs = sp.sparse.linalg.eigsh(H.data, n_eig, which='SR')
    elif solver == 'numpy':
        efreqs, evecs = np.linalg.eigh(H.__array__())
        efreqs = efreqs[:n_eig]
        evecs  = evecs [:,:n_eig]
    elif solver == 'Qutip':
        efreqs, evecs = H.eigenstates(eigvals=n_eig, sparse=True)
        evecs = np.array([ψ.__array__() for ψ in evecs])[:, :, 0].T


    efreqs_sorted = np.sort(efreqs.real)
    # efreqs_sorted = efreqs_sorted - efreqs_sorted[0]

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

#%% Plotting functions
def plot_H_eff_vs_param(H_eff_vs_params, H_eff_0, param_values, param_name, N_f, N_r, threshold=1e-3, n_eig_plot = False, scale='linear'):

    Δ_01 = H_eff_0[1,1]- H_eff_0[0,0]
    H_eff_vs_params /= Δ_01
    H_eff_0 /= Δ_01

    if n_eig_plot == False:
        n_eig_plot = len(H_eff_0)

    colors = figs.generate_colors_from_colormap(20, 'tab20')
    label_color_dict = {}
    titles = ['Couplings', 'Renormalizations']
    y_labels = [r'$g/\Delta_{01}$', r'$E/\Delta_{01}$']
    fig, [ax1, ax2] = plt.subplots(ncols=2, dpi=150, figsize=[8, 4.5])
    k_ij = 0; k_ii = 0
    for i in range(n_eig_plot):
        for j in range(i, n_eig_plot):
            if i != j and np.any(np.abs(H_eff_vs_params[:, i, j]) > threshold):
                label = get_state_label(N_f, N_r, i, j)
                color, label_color_dict, _ = get_or_assign_color(label, colors, label_color_dict)
                if scale == 'log':
                    ax1.plot(param_values[k_ij:], np.abs(H_eff_vs_params[k_ij:, i, j]), markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                else:
                    ax1.plot(param_values[k_ij:], H_eff_vs_params[k_ij:, i, j], markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                k_ij += 1

            elif i == j and np.any(np.abs(H_eff_vs_params[:, i, j] - H_eff_0[i, j]) > threshold):
                label =  get_state_label(N_f, N_r, i, j)
                color, label_color_dict, _ = get_or_assign_color(label, colors, label_color_dict)
                if scale == 'log':
                    ax2.plot(param_values[k_ii:], np.abs(H_eff_vs_params[k_ii:, i, j] - H_eff_0[i, j]), markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                else:
                    ax2.plot(param_values[k_ii:], H_eff_vs_params[k_ii:, i, j] - H_eff_0[i, j], markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                k_ii += 1
    if scale == 'log':
        ax2.plot(param_values, np.abs(1-np.abs(H_eff_vs_params[:, 1, 1] - H_eff_vs_params[:, 0, 0])), ':k',label=r'$1-\Delta_{01}('+param_name+')/\Delta_{01}$')
    else:
        ax2.plot(param_values, 1-np.abs(H_eff_vs_params[:, 1, 1] - H_eff_vs_params[:, 0, 0]), ':k',label=r'$1-\Delta_{01}('+param_name+')/\Delta_{01}$')

    for i, ax in enumerate([ax1, ax2]):
        ax.set_xlabel('$'+param_name+'$')
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_title(titles[i])
        ax.set_ylabel(y_labels[i])
        ax.legend()

    fig.tight_layout()
    fig.show()
    return fig, ax1, ax2

#%% Generic labeling and sorting functions
def print_charge_transformation(circuit):
    for i in range(circuit.R.shape[1]):
        normalized = circuit.R[:, i] / np.abs(circuit.R[:, i]).max()
        formatted_vector = [f"{num:5.2f}" for num in normalized]
        vector_str = ' '.join(formatted_vector)
        print(f'q_{i + 1} = [{vector_str}]')


def print_flux_transformation(circuit):
    for i in range(circuit.S.shape[1]):
        normalized = circuit.S[:, i] / np.abs(circuit.S[:, i]).max()
        formatted_vector = [f"{num:5.2f}" for num in normalized]
        vector_str = ' '.join(formatted_vector)
        print(f'Φ_{i + 1} = [{vector_str}]')


def get_state_label(N_f, N_r, i, j, return_numeric_indices=False):
    if i == j:
        label = f'({N_f[i]}f,{N_r[i]}r)'
    else:
        label = f'({N_f[i]}f,{N_r[i]}r) - ({N_f[j]}f,{N_r[j]}r)'

    if return_numeric_indices:
        return label, N_f[i], N_f[j], N_r[i], N_r[j]
    else:
        return label


def find_indices(data):
    result = []
    current_string = data[0]
    start_index = 0

    for i, s in enumerate(data[1:], 1):  # start from index 1
        if s != current_string:
            result.append((current_string, [start_index, i - 1]))
            current_string = s
            start_index = i

    result.append((current_string, [start_index, len(data) - 1]))  # for the last string

    return result


def get_or_assign_color(s, colors, color_dict):
    # Check if string is already associated with a color
    if s in color_dict:
        newly_assigled=False
        return color_dict[s], color_dict, newly_assigled

    # If not, associate with the next available color
    for color in colors:
        if color not in color_dict.values():  # Check if this color is not already used
            newly_assigled = True
            color_dict[s] = color
            return color, color_dict, newly_assigled
    # If all colors are already used, you might want to handle this case.
    # For now, it returns None and the unchanged dictionary.
    print('I am out of colorssssssss bro')


def rank_by_multiples(arr, tol = 0.1):
    # Receives an array of repeting real values and outputs an array of integers that identify the smallest element and
    # The rest of the elements as multiples of the delta between the smallest and next smallest element.
    # e.g. in=[2.41, 1.23, 2.42, 3.63] out=[1, 0, 1, 2]
    # aka puting labels to usorted energy levels.

    # Identify the smallest value
    min_value = np.min(arr)

    # Subtract the smallest value from the entire array
    diff_array = arr - min_value

    # Identify the next smallest positive value in the subtracted array
    unit = np.min([val for val in diff_array if val > tol])

    return np.array( np.round(diff_array / unit), dtype='int')


def decomposition_in_pauli_2x2(A):
    '''Performs Pauli decomposition of a 2x2 matrix.

    Input:
    A= matrix to decompose.

    Output:
    P= 4 coefficients such that A = P[0]*I + P[1]*σx + P[2]*σy + P[3]*σz'''

    # Pauli matrices.
    I = np.eye(2)
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [I, σx, σy, σz]  # array containing the matrices.

    P = np.zeros(4)  # array to store our results.
    # Loop to obtain each coefficient.
    for i in range(4):
        P[i] = 0.5 * np.trace(s[i].T.conjugate() @ A)

    return P


def decomposition_in_pauli_4x4(A, rd, Print=True):
    '''Performs Pauli decomposition of a 2x2 matrix.

    Input:
    A= matrix to decompose.
    rd= number of decimals to use when rounding a number.

    Output:
    P= coefficients such that A = ΣP[i,j]σ_iσ_j where i,j=0, 1, 2, 3. '''

    i = np.eye(2)  # σ_0
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [i, σx, σy, σz]  # array containing the matrices.
    labels = ['I', 'σx', 'σy', 'σz']  # useful to print the result.

    P = np.zeros((4, 4), dtype=complex)  # array to store our results.
    # Loop to obtain each coefficient.
    for i in range(4):
        for j in range(4):
            label = labels[i] + ' \U00002A02' + labels[j]
            S = np.kron(s[i], s[j])  # S_ij=σ_i /otimes σ_j.
            P[i, j] = np.round(0.25 * (np.dot(S.T.conjugate(), A)).trace(), rd)  # P[i,j]=(1/4)tr(S_ij^t*A)
            if P[i, j] != 0.0 and Print == True:
                print(" %s\t*\t %s " % (P[i, j], label))

    return P

#%% Generic functions

def expand_list_with_array(input_list):
    # Find the array and its position in the list
    array_element = None
    array_index = None
    for i, element in enumerate(input_list):
        if isinstance(element, np.ndarray):
            array_element = element
            array_index = i
            break

    # If no array is found, return the original list wrapped in another list
    if array_element is None:
        return [input_list]

    # Create the list of lists
    expanded_list = []
    for value in array_element:
        new_list = input_list.copy()
        new_list[array_index] = value
        expanded_list.append(new_list)

    return expanded_list


#%% Truncation convergence
def truncation_convergence(circuit, n_eig, trunc_nums=False, threshold=1e-2, refine=True, plot=True):
    '''
    This function tests the convergence of set_trunc_nums.
    It increases the truncation numbers until the convergence condition is met.
    The convergence condition is that the max difference between the spectrum calculated using some truncation numbers,
    versus those truncation numbers + 1, must be below a certain "threshold" (given by user or set as 0.1%).

    If refine is True then it reduces the truncation number of each mode iteratively to see if the convergence condition
    is still met.

    TO DO: This function is not very optimal to obtain the spectrum vs phi, it diagonalizes too many times for already converged results.
    '''

    if not trunc_nums:
        trunc_nums = [2] * circuit.n

    ΔE = 1
    E = []
    trunc_nums_list = []
    while ΔE > threshold:
        trunc_nums = [n + 1 for n in trunc_nums]
        circuit.set_trunc_nums(trunc_nums)
        E.append(circuit.diag(n_eig)[0])
        trunc_nums_list.append(trunc_nums.copy())
        if len(E) == 1:
            continue

        ΔE = np.abs((E[-1] - E[-2]) / E[-2]).max()

    E_conv = E[-1].copy()

    if refine:
        modes_to_refine = np.array([True] * circuit.n)
        while np.any(modes_to_refine) == True:
            for mode in range(circuit.n):
                if modes_to_refine[mode] == False:
                    continue
                trunc_nums[mode] -= 1
                circuit.set_trunc_nums(trunc_nums)
                E.append(circuit.diag(n_eig)[0])
                trunc_nums_list.append(trunc_nums.copy())
                ΔE = np.abs((E[-1] - E_conv) / E_conv).max()
                if ΔE < threshold:
                    if trunc_nums[mode] == 1:
                        modes_to_refine[mode] = False
                    continue
                else:
                    modes_to_refine[mode] = False
                    trunc_nums[mode] += 1
                    del E[-1], trunc_nums_list[-1]
                    ΔE = np.abs((E[-1] - E_conv) / E_conv).max()

    if plot:
        print(trunc_nums, E[-1], ΔE)
        fig, ax = plt.subplots()
        ax.plot(np.abs(E_conv - np.array(E[1:])) / E_conv)
        ax.set_yscale('log')
        ax.set_ylabel(r'$(E-E_{conv}) / E_{conv}$')
        labels_trunc_nums = [str(l) for l in trunc_nums_list]
        ax.set_xlabel('trunc_nums')
        ax.set_xticks(np.arange(len(E))[::2], labels_trunc_nums[::2], rotation=-30)
        fig.show()

    circuit.set_trunc_nums(trunc_nums)
    return circuit


#%% Elephants' graveyard

# def hamiltonian_frc_qubit(qubit, fluxonium, resonator, Δ, Lq = 25, Lr = 10, factor=1):
#     l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#
#     H_f = fluxonium.hamiltonian()
#     H_r = resonator.hamiltonian()
#
#     I_f = qt.identity(H_f.shape[0])
#     I_r = qt.identity(H_r.shape[0])
#
#     Φ, Q = get_node_variables(qubit,'FC', isolated=True)
#
#     Φ_f = Φ[1]-Φ[0]
#     Φ_r = Φ[0]+Φ[1]
#
#     H = qt.tensor(I_r, H_f) + qt.tensor(H_r, I_f) + factor * qt.tensor(Φ_r, Φ_f) * 2 * Δ / l / 1e-9
#     return H

# def KIT_qubit_triangle(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):
#
#     R1 = Lq/2-Δ
#     R2 = Lq/2+Δ
#     R3 = Lr
#
#     Rp = R1*R2 + R1*R3 + R2*R3
#     Ra = Rp/R1
#     Rb = Rp/R2
#     Rc = Rp/R3
#
#     # Initialize loop(s)
#     loop = sq.Loop(φ_ext)
#     loop_fictitious = sq.Loop(φ_ext)
#
#     # Circuit components
#     C_01 = sq.Capacitor(C,       'fF')
#     C_02 = sq.Capacitor(C,       'fF')
#     C_12 = sq.Capacitor(CJ+Csh,  'fF')
#     L_01 = sq.Inductor(Rb, 'nH',  loops=[loop_fictitious])
#     L_02 = sq.Inductor(Ra, 'nH',  loops=[loop_fictitious])
#     L_12 = sq.Inductor(Rc, 'nH',  loops=[loop_fictitious, loop])
#     JJ_12= sq.Junction(EJ,'GHz',  loops=[loop])
#
#     elements = {
#         (0, 1): [C_01, L_01],
#         (0, 2): [C_02, L_02],
#         (1, 2): [C_12, JJ_12, L_12],
#     }
#
#     # Create and return the circuit
#     return sq.Circuit(elements)


# def H_eff_SWT_circ(circuit_0, circuit, return_transformation = False):
#     ψb0 = real_eigenvectors(np.array([ψb0_i.__array__()[:,0] for ψb0_i in circuit_0._evecs]).T)
#     ψb  = real_eigenvectors(np.array([ψb0_i.__array__()[:,0] for ψb0_i in circuit  ._evecs]).T)
#     E = circuit.efreqs
#
#     Q = ψb0.T.conj() @ ψb
#     U, s, Vh = np.linalg.svd(Q)
#     A = U @ Vh
#     H_eff = A @ np.diag(E) @ A.T.conj()
#
#     if return_transformation:
#         return H_eff, A
#     else:
#         return H_eff

# def H_eff_p1_circ(circ_0, circ, out='GHz', real=True, remove_ground = False):
#     ψ_0 = np.array([ψ_i.__array__()[:, 0] for ψ_i in circ_0._evecs]).T
#
#     if real:
#         ψ_0 = real_eigenvectors(ψ_0)
#
#     H = circ.hamiltonian().__array__()
#     H_eff = ψ_0.conj().T @ H @ ψ_0
#
#     if out == 'GHz':
#         H_eff /= GHz * 2 * np.pi
#
#     if remove_ground:
#         H_eff -=  H_eff[0,0]*np.eye(len(H_eff))
#
#     if real:
#         if np.allclose(np.imag(H_eff),0):
#             H_eff = np.real(H_eff)
#
#     return H_eff

# def H_eff_p2_circ(circ_0, circ):
#     ψ_0 = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in circ_0._evecs]).T)
#     ψ   = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in circ.  _evecs]).T)
#     E_0 = circ_0._efreqs
#     E   = circ  ._efreqs
#     H_0 = circ_0.hamiltonian().__array__()
#     H   = circ  .hamiltonian().__array__()
#     V   = H-H_0
#
#     # H_eff_1 = ψ_0.conj().T @ H @ ψ_0
#
#     n_eig = ψ_0.shape[1]
#     H_eff_2 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.
#
#     for i in range(n_eig):
#         for j in range(n_eig):
#             H_eff_2[i, j] = 1 / 2 * sum(
#                           (1 / (E_0[i] - E[k]) + 1 / (E_0[j] - E[k])) *
#                            (ψ_0[:, i].T.conj() @ V @ ψ[:, k]) * \
#                            (ψ[:, k].T.conj() @ V @ ψ_0[:, j])
#                            for k in range(n_eig))
#     # return H_eff_1
#     return H_eff_2 / GHz / 2 / np.pi


# def H_eff_SWT_eigs(ψb0, ψb, E):
#     Q = ψb0.T.conj() @ ψb
#     U, s, Vh = np.linalg.svd(Q)
#     A = U @ Vh
#     H_eff = A @ np.diag(E) @ A.T.conj()
#     return H_eff

# def H_eff_p1_frc(n_eig, H_frc_0, H_frc, Δ, Lq = 25, Lr = 10):
#     l   = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#
#     ψ_0 = diag(H_frc_0, n_eig)[1]
#
#     H_eff_p1 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.
#
#     for i in range(n_eig):
#         for j in range(n_eig):
#             H_eff_1_f_i_j = np.abs(ψ_0_f[:, N_f[i]].conj().T @ fluxonium.flux_op(0).__array__() @ ψ_0_f[:, N_f[j]])
#             H_eff_1_r_i_j = np.abs(ψ_0_r[:, N_r[i]].conj().T @ resonator.flux_op(0).__array__() @ ψ_0_r[:, N_r[j]])
#             H_eff_p1[i,j] = H_eff_1_f_i_j*H_eff_1_r_i_j
#
#     # return Δ  * (H_eff_p1 / (Δ * L_c) ) / GHz # / 2 / np.pi  Why not this 2pi!!!
#     return H_eff_p1 * 2 * Δ / l / 1e-9  / (2 * np.pi * GHz)

# def H_eff_p1_fluxonium_resonator_ij(fluxonium_0, fluxonium, resonator_0, resonator, i_f, j_f, i_r, j_r, Δ, Lq = 25, Lr = 10):
#     l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#     L_c = l / Δ * nH
#
#     ψ_0_f = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium_0._evecs]).T)
#     ψ_0_r = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator_0._evecs]).T)
#
#     Φ_f = ψ_0_f[:, i_f].conj().T @ fluxonium.flux_op(0).__array__() @ ψ_0_f[:, j_f]
#     Φ_r = ψ_0_r[:, i_r].conj().T @ resonator.flux_op(0).__array__() @ ψ_0_r[:, j_r]
#
#     return Φ_f * Φ_r / L_c / GHz #/ 2 /np.pi

#%% Functions that are actually in sqcircuits file circuit.py

# def flux_op(self, mode: int, basis: str = 'FC') -> Qobj:
#     """Return flux operator for specific mode in the Fock/Charge basis or
#     the eigenbasis.
#
#     Parameters
#     ----------
#         mode:
#             Integer that specifies the mode number.
#         basis:
#             String that specifies the basis. It can be either ``"FC"``
#             for original Fock/Charge basis or ``"eig"`` for eigenbasis.
#     """
#
#     error1 = "Please specify the truncation number for each mode."
#     assert len(self.m) != 0, error1
#
#     # charge operator in Fock/Charge basis
#     Φ_FC = self._memory_ops["phi"][mode]
#
#     if basis == "FC":
#
#         return Φ_FC
#
#     elif basis == "eig":
#         ψ = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in self._evecs]).T)
#
#         Φ_eig = ψ.conj().T @ Φ_FC.__array__() @ ψ
#
#         return qt.Qobj(Φ_eig)
