import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'QtAgg'

#%%

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
        print(trunc_nums, E[-1],  ΔE )
        fig, ax = plt.subplots()
        ax.plot(np.abs(E_conv-np.array(E[1:]))/E_conv)
        ax.set_yscale('log')
        ax.set_ylabel(r'$(E-E_{conv}) / E_{conv}$')
        labels_trunc_nums = [str(l) for l in trunc_nums_list]
        ax.set_xlabel('trunc_nums')
        ax.set_xticks(np.arange(len(E))[::2],labels_trunc_nums[::2], rotation=-30)
        fig.show()

    circuit.set_trunc_nums(trunc_nums)
    return circuit

    # def flux_op(self, mode: int, basis: str = 'FC') -> Qobj:
    #     """Return charge operator for specific mode in the Fock/Charge basis or
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
    #     phi_FC = self._memory_ops["phi"][mode-1]
    #
    #     if basis == "FC":
    #
    #         return phi_FC
    #
    #     elif basis == "eig":
    #
    #         # number of eigenvalues
    #         n_eig = len(self.efreqs)
    #
    #         phi_eig = np.zeros((n_eig, n_eig), dtype=complex)
    #
    #         for i in range(n_eig):
    #             φ_i = sq.Qobj(real_eigenvectors(self._evecs[i].__array__()))
    #             for j in range(n_eig):
    #                 φ_j = sq.Qobj(real_eigenvectors(self._evecs[j].__array__()))
    #                 phi_eig[i, j] = (φ_i.dag() * phi_FC * φ_j).data[0, 0]
    #
    #         return qt.Qobj(phi_eig)
    #
    # def charge_op(self, mode: int, basis: str = 'FC') -> Qobj:
    #     """Return charge operator for specific mode in the Fock/Charge basis or
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
    #     Q_FC = self._memory_ops["Q"][mode-1]
    #
    #     if basis == "FC":
    #
    #         return Q_FC
    #
    #     elif basis == "eig":
    #
    #         # number of eigenvalues
    #         n_eig = len(self.efreqs)
    #
    #         Q_eig = np.zeros((n_eig, n_eig), dtype=complex)
    #
    #         for i in range(n_eig):
    #             φ_i = sq.Qobj(real_eigenvectors(self._evecs[i].__array__()))
    #             for j in range(n_eig):
    #                 φ_j = sq.Qobj(real_eigenvectors(self._evecs[j].__array__()))
    #                 Q_eig[i, j] = (φ_i.dag() * Q_FC * φ_j).data[0, 0]
    #
    #         return qt.Qobj(Q_eig)