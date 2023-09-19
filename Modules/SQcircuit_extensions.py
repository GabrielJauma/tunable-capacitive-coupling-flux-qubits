import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['backend'] = 'QtAgg'


# %%

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


def truncation_convergence(circuit, n_eig, trunc_nums=False, threshold=1e-3, refine=True, plot=False):
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
def print_transformation(circuit):
    for i in range(circuit.S.shape[1]):
        normalized = circuit.S[:, i] / np.abs(circuit.S[:, i]).max()
        formatted_vector = [f"{num:5.1f}" for num in normalized]  # Adjust the width (5 in this case) as per your needs
        vector_str = ' '.join(formatted_vector)
        print(f'Mode {i + 1} = [{vector_str}]')
