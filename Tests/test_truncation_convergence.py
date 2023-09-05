import Modules.SQcircuit_extensions as sq_ext
import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt
import importlib

importlib.reload(sq_ext)
plt.rcParams['backend'] = 'Qt5Agg'

# %% Define circuit for testing
# Circuit parameters
Csh = 15
C   = 15
CJ  = 3
Cg  = 20
Lq = 25
Lr = 10
Δ  = 0.1
EJ  = 10.0

# Initialize loop(s)
loop = sq.Loop(0.0)  # "Value" corresponds to phiExt / phi0 threading the loop (can change later)

# Circuit elements
C_01 = sq.Capacitor(C, 'fF')
C_02 = sq.Capacitor(C, 'fF')
C_12 = sq.Capacitor(CJ+Csh, 'fF')
C_04 = sq.Capacitor(Cg, 'fF')

L_03 = sq.Inductor(Lr, 'nH')
L_31 = sq.Inductor(Lq/2 - Δ, 'nH', loops=[loop])
L_23 = sq.Inductor(Lq/2 + Δ, 'nH', loops=[loop])

JJ = sq.Junction(EJ, 'GHz', loops=[loop])

# Create the circuit
elements = {
    # (0,4): [C_04],
    (0, 3): [L_03],
    (0, 1): [C_01],
    (0, 2): [C_02],
    (3, 1): [L_31],
    (1, 2): [C_12, JJ],
    (2, 3): [L_23],
}

circuit = sq.Circuit(elements)

# %% Test truncation_convergence
n_eig = 5
loop.set_flux(0)
trunc_nums = [5] * circuit.n
circuit = sq_ext.truncation_convergence(circuit, n_eig, refine=True, plot=True)


#%%

# spectrum of the circuit
phi = np.linspace(0,1,100)
n_eig=8
spec = np.zeros((n_eig, len(phi)))
E_qubit = np.zeros((n_eig, len(phi)))

eig_colors = plt.get_cmap('viridis_r')(np.linspace(0, 255, n_eig).astype('int'))
for i in range(len(phi)):
    # set the external flux for the loop
    loop.set_flux(phi[i])
    loop.set_flux(phi[i])
    # diagonalize the circuit
    spec[:, i] = circuit.diag(n_eig)[0]

fig, ax = plt.subplots(dpi=150)
for i in range(n_eig):
    ax.plot(phi, spec[i,:]- spec[0,:], color=eig_colors[i], linewidth=2)

ax.set_xlabel(r"$\Phi_{ext}/\Phi_0$", fontsize=13)
ax.set_ylabel(r"$f_i-f_0$[GHz]", fontsize=13)
fig.show()