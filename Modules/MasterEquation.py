import numpy as np
from scipy.linalg import expm


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


