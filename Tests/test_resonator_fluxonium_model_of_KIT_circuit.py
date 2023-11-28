import Modules.SQcircuit_extensions as sq_ext
import SQcircuit as sq
import qutip as qt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import importlib
importlib.reload(sq_ext)
importlib.reload(sq)

#%% Fundamental constants
h    = 6.626e-34
GHz  = 1e9
e0   = 1.602e-19
Φ_0 = h/(2*e0)

#%%  Circuit parameters
Csh = 15
C   = 15
Cg  = 10
CJ  = 3
Lq  = 25
Lr  = 10
Δ   = 0.1
EJ  = 10.0
l = Lq*(Lq+4*Lr) - 4*Δ**2

#%% Circuit elements
φ_ext = 0.5
loop_fluxonium = sq.Loop(φ_ext)
loop_ficticious = sq.Loop(0.0)

C_r_eq = sq.Capacitor(C/2, 'fF')
L_r_eq = sq.Inductor ( 1/(  1/(l/Lq) -  1/(l/Δ)  ), 'nH')

C_f_eq = sq.Capacitor(C/2 + Csh + CJ, 'fF')
L_f_eq = sq.Inductor ( 1/(  1/(l/(Lq+4*Lr)) -  1/(l/Δ)  ), 'nH',  loops=[loop_fluxonium, loop_ficticious])
JJ_f_eq = sq.Junction (EJ, 'GHz', loops=[loop_fluxonium])

L_c_eq = sq.Inductor (l/Δ, 'nH')

#%%
equiv_elements = {
    (0, 1): [C_r_eq, L_r_eq],
    (1, 2): [L_c_eq],
    (0, 2): [C_f_eq, L_f_eq, JJ_f_eq],
}
circ_equiv = sq.Circuit(equiv_elements )
circ_equiv.description()

# %%
circ_equiv.S

# %%
circ_equiv.set_trunc_nums([15, 15])
#
# # spectrum of the circ_equiv
# phi = np.linspace(0,1,100)
# # phi = np.linspace(0.36,0.38,500)
# n_eig=8
# spec = np.zeros((n_eig, len(phi)))
#
# eig_colors = plt.get_cmap('viridis_r')(np.linspace(0, 255, n_eig).astype('int'))
# for i in range(len(phi)):
#     # set the external flux for the loop
#     loop_fluxonium.set_flux(phi[i])
#     # diagonalize the circ_equiv
#     spec[:, i] = circ_equiv.diag(n_eig)[0]
#
# fig, ax = plt.subplots(dpi=150)
# for i in range(n_eig):
#     ax.plot(phi, spec[i,:]- spec[0,:], color=eig_colors[i], linewidth=1)
#
# ax.set_xlabel(r"$\Phi_{ext}/\Phi_0$", fontsize=13)
# ax.set_ylabel(r"$f_i-f_0$[GHz]", fontsize=13)
# # ax.set_ylim(7,7.5)
# fig.show()

#%%
_ = circ_equiv.diag(2)

#%%
sq_ext.decomposition_in_pauli_2x2(circ_equiv.flux_op(0, basis='eig').__array__())[1]


