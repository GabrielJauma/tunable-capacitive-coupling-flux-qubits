import numpy as np


def build_projectors(H0, n_low):
    """
    Diagonalize H0, return:
      E: eigenvalues (ascending)
      U: eigenvectors (columns)
      P0: projector onto the first n_low eigenstates (the 'low' subspace)
      Q0: projector onto the complement
    """
    E, U = np.linalg.eigh(H0)
    dim = H0.shape[0]
    # Low subspace
    U_low = U[:, :n_low]  # shape (dim, n_low)
    P0 = U_low @ U_low.conj().T
    Q0 = np.eye(dim, dtype=complex) - P0
    return E, U, P0, Q0


def matrix_element(V, psi_bra, psi_ket):
    """
    Returns the complex number  <psi_bra| V |psi_ket>.
    Each psi_* is assumed to be a (dim,) numpy array (column vector).
    """
    return np.vdot(psi_bra, V @ psi_ket)


def build_eff_ham(H0, V, n_low, n_order):
    """
    Main driver.
    Steps:
      1) Diagonalize H0 => E, U
      2) Build the 'low' subspace indices (0..n_low-1) vs. 'high' (n_low..dim-1)
      3) For each i,j in low subspace, sum up:
            - 1st order: <i|V|j>
            - 2nd order: sum_{k in high} ...
            - ...
            - nth order: sum_{(k1,...,k_{m-1}) in high} ...
         Then assemble into an (n_low x n_low) matrix, which is H_eff^(m).
      4) Combine everything to get total H_eff up to n-th order:
         H_eff = P0 H0 P0 + sum_{m=1..n_order} epsilon^m * H_eff^(m)
         (We won't keep the factor epsilon^m in the code; you can factor that out yourself.)
    Returns:
      H_eff_list: a list of [H_eff^(0), H_eff^(1), ..., H_eff^(n_order)],
                  each is (n_low x n_low) as a numpy array.
                  H_eff^(0) = P0 H0 P0.
    """
    E, U = np.linalg.eigh(H0)
    dim = H0.shape[0]
    idx_low = range(n_low)
    idx_high = range(n_low, dim)

    # Convert V into the same eigenbasis of H0 for convenience.
    # We'll compute <i|V|j> by:
    #    <i|V|j> = (U[:,i])^\dagger V U[:,j]
    # We can store them in a matrix M_{i,j} = <i|V|j>, i,j in [0..dim-1].
    # Then partial sums are easier.
    M = np.zeros((dim, dim), dtype=complex)
    # Build each (i,j) once.
    for i in range(dim):
        psi_i = U[:, i]  # (dim,) vector
        for j in range(dim):
            psi_j = U[:, j]
            M[i, j] = matrix_element(V, psi_i, psi_j)

    # Prepare H_eff^(m) as (n_low x n_low) for m=0..n_order
    H_eff_list = []

    # (0)th order = just the block of H0 in low subspace => P0 H0 P0
    # but in the H0 eigenbasis, <i|H0|j> = E_i delta_{i,j}.
    # => if i = j in low subspace, it is E_i, else 0.
    # So the (0)th order in the low subspace is just diag(E[i]) for i in idx_low.
    H_eff0 = np.zeros((n_low, n_low), dtype=complex)
    for iL, i_global in enumerate(idx_low):
        H_eff0[iL, iL] = E[i_global]
    H_eff_list.append(H_eff0)

    # Now build each higher order up to n_order
    # We'll define a function that enumerates all 'chains' in high subspace of length (m-1).
    # We'll then multiply the matrix elements <i|V|k1> <k1|V|k2> ... <k_{m-1}|V|j>
    # and the denominators in the "symmetric" form.

    def second_order_term(i, j, E, M):
        """
        Standard symmetrical 2nd order:
          0.5 * sum_{k in high} [1/(E_i - E_k) + 1/(E_j - E_k)] * M[i,k]*M[k,j]
        """
        val = 0.0j
        Ei, Ej = E[i], E[j]
        for k in idx_high:
            Ek = E[k]
            denom = 0.5 * ((1. / (Ei - Ek)) + (1. / (Ej - Ek)))
            val += denom * M[i, k] * M[k, j]
        return val

    def mth_order_term(i, j, m, E, M):
        """
        Generic symmetrical mth order.

        For m=2 => same as second_order_term.
        For m>=3 => sum_{k1,..,k_{m-1} in high}, factor:
             COEFF * [1/( (Ei-E_{k1})...(Ei-E_{k_{m-1}} )) + 1/( (Ej-E_{k1})...(Ej-E_{k_{m-1}} )) ]
             * M[i,k1]*M[k1,k2]...M[k_{m-2},k_{m-1}]*M[k_{m-1},j].
        Where COEFF depends on the standard "symmetric" expansions. Typically = 1/(m!) for the fully symmetrical form,
        but let's do a direct approach with the standard known results:
          - 2nd order => 1/2 in front
          - 3rd order => 1/6
          - 4th order => 1/24
          etc.  => 1/(m!)
        That is indeed the factor that emerges from the "fully-symmetric" partition (similar to eqn expansions with permutations).

        We'll implement it with recursion or nested loops. For clarity, let's do a simple nested approach.
        For large m, you'd want recursion. We'll do a small recursion approach.
        """
        if m == 1:
            # first order => just M[i,j]
            return M[i, j]
        if m == 2:
            # we already have second_order_term
            return second_order_term(i, j, E, M)

        # For m >= 3 => do nested sums over k1..k_{m-1} in idx_high
        # Factor 1/(m!) times the sum of
        #   [1 / prod_{r=1..m-1} (Ei-E_{k_r}) + same with Ej]
        #   * product of M's
        from math import factorial
        prefactor = 1. / (factorial(m))  # e.g. 3rd order => 1/6, 4th => 1/24, etc.

        val = 0.0j

        # we'll do a simple recursive approach to build all (k1..k_{m-1})
        def recurse_product(i_state, depth, path_indices):
            # i_state is the "entry" index for the matrix element
            # depth: how many more V multiplications we want
            # path_indices: the list of high states visited so far

            if depth == 0:
                # we are done => multiply by < final| V | j >? Actually not: we do it outside
                pass
            else:
                for k_next in idx_high:
                    yield from recurse_product(k_next, depth - 1, path_indices + [k_next])

        # We'll just gather all sequences of length (m-1) from idx_high:
        # Then compute the denominators, matrix element product, etc.
        # However, we need them to start from i and end at j, i.e. i->k1->k2->..->k_{m-1}-> j
        # Let's do an explicit approach:

        # Build all sequences of length (m-1) in high subspace:
        from itertools import product
        for chain in product(idx_high, repeat=(m - 1)):
            # chain = (k1, k2, ..., k_{m-1})
            # compute denominators:
            Ei_prod = 1.0
            Ej_prod = 1.0
            Ei_curr = E[i]
            Ej_curr = E[j]
            for kr in chain:
                Ei_prod *= (Ei_curr - E[kr])
                Ej_prod *= (Ej_curr - E[kr])
            # sum of 1/Ei_prod + 1/Ej_prod
            denom_factor = (1. / Ei_prod + 1. / Ej_prod)

            # matrix elements M[i,k1]*M[k1,k2]*...*M[k_{m-1}, j]
            mat_prod = M[i, chain[0]]
            for r in range(len(chain) - 1):
                mat_prod *= M[chain[r], chain[r + 1]]
            mat_prod *= M[chain[-1], j]

            val += denom_factor * mat_prod

        return prefactor * val

    # Now build H_eff^(m) for m=1..n_order
    # We'll store them in a list, each is (n_low x n_low).
    # Then the total H_eff = H_eff^(0) + sum_{m=1..n_order} H_eff^(m).

    for m in range(1, n_order + 1):
        H_m = np.zeros((n_low, n_low), dtype=complex)
        for idx_i, i_global in enumerate(idx_low):
            for idx_j, j_global in enumerate(idx_low):
                H_m[idx_i, idx_j] = mth_order_term(i_global, j_global, m, E, M)
        H_eff_list.append(H_m)

    return H_eff_list


def perturbation_H_eff(H0, V, n_low, n_order):
    """
    Full helper:
     1) Get [H_eff^(0), H_eff^(1), ..., H_eff^(n_order)] via build_eff_ham
     2) Sum them => total H_eff^(<= n_order)
    Returns:
      total_H_eff: (n_low x n_low) matrix
      H_eff_terms: list of each term
    """
    H_eff_terms = build_eff_ham(H0, V, n_low, n_order)
    # H_eff_terms[0] = H_eff^(0) = P0 H0 P0
    # H_eff_terms[m] = H_eff^(m) for m=1..n_order
    total_H_eff = np.sum(H_eff_terms, axis=0)
    return total_H_eff, H_eff_terms
