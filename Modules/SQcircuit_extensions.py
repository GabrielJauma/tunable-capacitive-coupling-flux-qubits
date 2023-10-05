import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['backend'] = 'QtAgg'

# %%
def KITqubit(C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):

    # Initialize loop(s)
    loop = sq.Loop(φ_ext)

    # Circuit components
    C_01 = sq.Capacitor(C ,      'fF')
    C_02 = sq.Capacitor(C ,      'fF')
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


# %%
def H_eff_p1(circ_0, circ):
    ψ = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in circ_0._evecs]).T)
    H_eff = ψ.conj().T @ circ.hamiltonian().__array__() @ ψ
    return H_eff

# Second order perturbation theory
def H_eff_p2(circ_0, circ):
    ψ_0 = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in circ_0._evecs]).T)
    ψ = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in circ._evecs]).T)
    E_0 = circ_0._efreqs
    E = circ._efreqs
    H_0 = circ_0.hamiltonian().__array__()
    H = circ.hamiltonian().__array__()

    n_eig = ψ_0.shape[1]
    H_eff = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.

    # This si wrong, I cant do the H-H_0 shit here since they are in diferent basis...
    # Loop to obtain each element of the matrix: <O>.
    for i in range(n_eig):
        for j in range(n_eig):
            H_eff[i, j] = ψ_0[:, i].T.conj() @ H @ ψ_0[:, j] + 1 / 2 * \
                          sum(
                              (1 / (E_0[i] - E[k]) + 1 / (E_0[j] - E[k])) * (ψ_0[:, i].T.conj() @ (H - H_0) @ ψ[:, k]) * \
                              (ψ[:, k].T.conj() @ (H - H_0) @ ψ_0[:, j])
                              for k in range(n_eig)
                          )
    return H_eff


# def H_eff_p2(ψb, Eb, ψs, Es, H0, H):
#     l = ψb.shape[1]
#     H_eff = np.zeros((l, l), dtype=complex)  # matrix to store our results.
#
#     # Loop to obtain each element of the matrix: <O>.
#     for i in range(l):
#         for j in range(l):
#             H_eff[i, j] = ψb[:, i].T.conj() @ H @ ψb[:, j] + 1 / 2 * \
#                           sum(
#                               (1 / (Eb[i] - Es[k]) + 1 / (Eb[j] - Es[k])) * (ψb[:, i].T.conj() @ (H - H0) @ ψs[:, k]) * \
#                               (ψs[:, k].T.conj() @ (H - H0) @ ψb[:, j])
#                               for k in range(ψs.shape[1])
#                           )
#     return H_eff

def H_eff_SWT_circuit(circuit_0, circuit):
    ψb0 = real_eigenvectors(np.array([ψb0_i.__array__()[:,0] for ψb0_i in circuit_0._evecs]).T)
    ψb = real_eigenvectors(np.array([ψb0_i.__array__()[:,0] for ψb0_i in circuit._evecs]).T)
    E = circuit._efreqs

    Q = ψb0.T.conj() @ ψb
    U, s, Vh = np.linalg.svd(Q)
    A = U @ Vh
    H_eff = A @ np.diag(E) @ A.T.conj()
    return H_eff

def H_eff_SWT_eigs(ψb0, ψb, E):
    Q = ψb0.T.conj() @ ψb
    U, s, Vh = np.linalg.svd(Q)
    A = U @ Vh
    H_eff = A @ np.diag(E) @ A.T.conj()
    return H_eff


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



#%%

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

#%%
def real_eigenvectors(U):
    '''Fixes the phase of a vector.

    Input:
    U=vector

    Output:
    U= vector without phase.'''

    l = U.shape[0]
    avgz = np.sum(U[:l // 2, :] * np.abs(U[:l // 2, :]) ** 2, 0)
    avgz = avgz / np.abs(avgz)
    # U(i,j) = U(i,j) / z(j) for all i,j
    U = U * avgz.conj()
    return U


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


# def print_transformation(circuit):
#     for i in range(circuit.S.shape[1]):
#         print(f'Mode {i+1} = {(circuit.S[:,i] / np.abs(circuit.S[:,i]).max()).round(1) }')
#

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