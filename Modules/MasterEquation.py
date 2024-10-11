import numpy as np
from scipy.linalg import expm
import Modules.SQcircuit_extensions as sq_ext


class LindbladSolver:
    def __init__(self, H_eff, D, rho_0, t_points):
        self.H_eff = H_eff
        self.D = D  # List of dissipators [gamma_i, A_i, B_i]
        self.rho_0 = rho_0
        self.t_points = t_points
        self.d = H_eff.shape[0]  # Dimension of the Hilbert space
        self.L = self.construct_L_superoperator()
        self.rho_t = None  # To be computed in evolve()
        self.observables = {}
        self.expectation_values = {}

    def construct_L_superoperator(self):
        I_d = np.eye(self.d, dtype=complex)

        # Hamiltonian part
        L_H = -1j * (np.kron(self.H_eff, I_d) - np.kron(I_d, self.H_eff.T))

        # Dissipator part
        L_D = np.zeros((self.d ** 2, self.d ** 2), dtype=complex)
        for dissipator in self.D:
            gamma_i, A_i, B_i = dissipator

            # Compute terms
            term1 = gamma_i * np.kron(A_i, B_i.T)
            term2 = -0.5 * gamma_i * np.kron(B_i @ A_i, I_d)
            term3 = -0.5 * gamma_i * np.kron(I_d, (B_i @ A_i).T)

            # Add terms to L_D
            L_D += term1 + term2 + term3

        # Total superoperator
        L_tot = L_H + L_D
        return L_tot

    def evolve(self):
        num_time_points = len(self.t_points)
        self.rho_t = np.zeros((num_time_points, self.d, self.d), dtype=complex)
        rho_0_vec = self.rho_0.flatten()
        for idx, t in enumerate(self.t_points):
            exp_Lt = expm(self.L * t)
            rho_t_vec = exp_Lt @ rho_0_vec
            self.rho_t[idx] = rho_t_vec.reshape((self.d, self.d))

    def set_observables(self, observables_dict):
        """
        observables_dict: A dictionary where keys are observable names (strings)
                          and values are the corresponding operators (numpy arrays).
        """
        self.observables = observables_dict

    def apply_unitary_to_all(self, U):
        """
        Applies a unitary transformation U to all evolved density matrices.
        """
        if self.rho_t is None:
            raise ValueError("Density matrix evolution not computed. Call evolve() first.")
        self.rho_t_transformed = np.zeros_like(self.rho_t)
        for idx in range(len(self.rho_t)):
            self.rho_t_transformed[idx] = U.conj().T @ self.rho_t[idx] @ U

    def compute_expectation_values(self, use_transformed=False):
        if use_transformed and not hasattr(self, 'rho_t_transformed'):
            raise ValueError("Transformed density matrices not available. Call apply_unitary_to_all() first.")
        rho_array = self.rho_t_transformed if use_transformed else self.rho_t
        self.expectation_values = {}
        for name, operator in self.observables.items():
            exp_vals = np.real([np.trace(operator @ rho) for rho in rho_array])
            self.expectation_values[name] = exp_vals


#%%

import numpy as np
from scipy.linalg import expm


def simulate_protocol1(t_points, t0, J, gamma1, gamma2, gamma_deph1, gamma_deph2, delta1, delta2):
    # Pauli matrices
    sigma_x, sigma_y, sigma_z = sq_ext.pauli_matrices()
    sigma_plus, sigma_minus = sq_ext.annihilate(2), sq_ext.create(2)
    I = np.eye(2, dtype=complex)

    sigma1_p = np.kron(sigma_plus, I)
    sigma1_m = np.kron(sigma_minus, I)
    sigma1_z = np.kron(sigma_z, I)

    # Operators for qubit 2
    sigma2_p = np.kron(I, sigma_plus)
    sigma2_m = np.kron(I, sigma_minus)
    sigma2_z = np.kron(I, sigma_z)

    # Hadamard gate
    Hadamard = np.array([[1, 1],
                         [1, -1]], dtype=complex) / np.sqrt(2)

    # Operators acting on qubits
    Hadamard1 = np.kron(Hadamard, I)

    # Effective Hamiltonian
    # H_eff = J * (sigma1_p @ sigma2_m + sigma1_m @ sigma2_p) + (delta / 2) * (sigma1_z + sigma2_z)
    H_eff = J * (sigma1_p @ sigma2_m + sigma1_m @ sigma2_p) + (delta1 / 2) * sigma1_z  + (delta2 / 2) * sigma2_z

    D = []

    # Detuning for qubit 1
    if gamma1 > 0:
        A1 = np.sqrt(gamma1) * sigma1_m
        B1 = sigma1_p / np.sqrt(gamma1)
        D1 = [gamma1, A1, B1]
        D.append(D1)

    # Detuning for qubit 1
    if gamma2 > 0:
        A2 = np.sqrt(gamma2) * sigma2_m
        B2 = sigma2_p / np.sqrt(gamma2)
        D2 = [gamma2, A2, B2]
        D.append(D2)

    # Dephasing for qubit 1
    if gamma_deph1 > 0:
        A_deph1 = sigma1_z
        B_deph1 = sigma1_z
        D_deph1 = [gamma_deph1, A_deph1, B_deph1]
        D.append(D_deph1)

    # Dephasing for qubit 2
    if gamma_deph2 > 0:
        A_deph2 = sigma2_z
        B_deph2 = sigma2_z
        D_deph2 = [gamma_deph2, A_deph2, B_deph2]
        D.append(D_deph2)

    # Initial state for Protocol 1: Hadamard applied to qubit 1 of |00⟩
    # Computational basis state |00⟩⟨00|
    P00 = np.kron(np.outer([1, 0], [1, 0]), np.outer([1, 0], [1, 0]))

    # Protocol 1 initial state
    rho_0_protocol1 = Hadamard1 @ P00 @ Hadamard1.conj().T

    # Create an instance of LindbladSolver
    solver = LindbladSolver(H_eff=H_eff, D=D, rho_0=rho_0_protocol1, t_points=t_points+t0)

    # Evolve the system
    solver.evolve()

    # Apply Hadamard to qubit 1 after evolution
    solver.apply_unitary_to_all(Hadamard1)

    # Observable: σ₁⁺σ₁⁻ (population of qubit 1)
    sigma1_p_proj = sigma1_p @ sigma1_m

    observables = {
        'pop1': sigma1_p_proj
    }

    solver.set_observables(observables)
    solver.compute_expectation_values(use_transformed=True)

    # Return the expectation values
    return solver.expectation_values['pop1']



