import Modules.SQcircuit_extensions as sq_ext
import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(sq_ext)


#%% define the circuit ’s elements
loop1 = sq.Loop()
# define the circuit ’s elements
C_r = sq.Capacitor(20.3, "fF")
C_q = sq.Capacitor(5.3, "fF")
L_r = sq.Inductor (15.6, "nH")
L_q = sq.Inductor(386, "nH", loops=[loop1])
L_s = sq.Inductor(4.5, "nH", loops=[loop1])
JJ  = sq.Junction(6.2, "GHz", loops=[loop1])

# # define the circuit
elements = {(0, 1): [C_r],
            (1, 2): [L_r],
            (0, 2): [L_s],
            (2, 3): [L_q],
            (0, 3): [JJ, C_q]}

#%% Kits circuit
# Circuit parameters
Csh = 15
C   = 15
CJ  = 3
Cg  = 20

Lq  = 25
Lr  = 10
Δ   = 0.1

EJ  = 10.0

# Initialize loop(s)
loop = sq.Loop(0.0)  # "Value" corresponds to phiExt / phi0 threading the loop (can change later)

# Circuit elements
C_01 = sq.Capacitor(C*12313, 'fF')
C_02 = sq.Capacitor(C, 'fF')
C_12 = sq.Capacitor(CJ+Csh, 'fF')
C_04 = sq.Capacitor(Cg, 'fF')

L_03 = sq.Inductor(Lr, 'nH')
L_31 = sq.Inductor(Lq/2 - Δ, 'nH', loops=[loop])
L_23 = sq.Inductor(Lq/2 + Δ, 'nH', loops=[loop])

JJ = sq.Junction(EJ, 'GHz', loops=[loop])

# Create the circuit
elements = {
    # (0, 4): [C_04],
    (0, 3): [L_03],
    (0, 1): [C_01],
    (0, 2): [C_02],
    (3, 1): [L_31],
    (1, 2): [C_12, JJ],
    (2, 3): [L_23],
}

#%%
cr = sq.Circuit(elements)
cr.description()

#%%
cr.C

# %%
loop1.set_flux(0.5)
cr.set_trunc_nums([1,9,23])
_, _ = cr.diag(n_eig=5)

#%% create a range for each mode
phi1 = 0
phi2 = np.linspace(-0.01,0.01,100)
phi3 = np.linspace(-0.5,0.5,100)

# creat the grid list
grid = [phi1, phi2, phi3]

#%%
# the ground state
state0 = cr.eig_phase_coord(0, grid = grid)

# the first excited state
state1 = cr.eig_phase_coord(1, grid = grid)

# the second excited state
state2 = cr.eig_phase_coord(2, grid = grid)

# the third excited state
state3 = cr.eig_phase_coord(3, grid = grid)

#%%
fig, axs = plt.subplots(1, 4,figsize=(16,4), sharey='row')
axs[0].pcolor(phi3, phi2, np.abs(state0.T)**2,cmap="Purples",shading='auto',label='state0')
axs[1].pcolor(phi3, phi2, np.abs(state1.T)**2,cmap="Greens",shading='auto',label='state1')
axs[2].pcolor(phi3, phi2, np.abs(state2.T)**2,cmap="Oranges",shading='auto',label='state2')
axs[3].pcolor(phi3, phi2, np.abs(state3.T)**2,cmap="Reds",shading='auto',label='state3')
for i in range(4):
    axs[i].set_xlabel(r"$\varphi_2$",fontsize=13)
    axs[i].legend(handletextpad=-0.1, handlelength=0.0)
axs[0].set_ylabel(r"$\varphi_3$",fontsize=13)
plt.subplots_adjust(wspace=0)

fig.show()