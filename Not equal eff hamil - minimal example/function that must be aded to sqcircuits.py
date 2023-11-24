# This function must be aded to the circuit class of circuit.py


def flux_op(self, mode: int, basis: str = 'FC') -> Qobj:
    """Return flux operator for specific mode in the Fock/Charge basis or
    the eigenbasis.

    Parameters
    ----------
        mode:
            Integer that specifies the mode number.
        basis:
            String that specifies the basis. It can be either ``"FC"``
            for original Fock/Charge basis or ``"eig"`` for eigenbasis.
    """

    error1 = "Please specify the truncation number for each mode."
    assert len(self.m) != 0, error1

    # charge operator in Fock/Charge basis
    Φ_FC = self._memory_ops["phi"][mode]

    if basis == "FC":

        return Φ_FC

    elif basis == "eig":
        ψ = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in self._evecs]).T)

        Φ_eig = ψ.conj().T @ Φ_FC.__array__() @ ψ

        return qt.Qobj(Φ_eig)