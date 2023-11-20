import Modules.SQcircuit_extensions as sq_ext
import Modules.figures as figs
import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt
import importlib
import qutip as qt
import scipy as sp

plt.rcParams['text.usetex'] = False
importlib.reload(sq_ext)
importlib.reload(sq)
importlib.reload(figs)
np.set_printoptions(linewidth=200, formatter={'float': '{:.3f}'.format})

#%% Premade circuits
def premade_coupled_resonators(Δ):
    C   = sq.Capacitor(1, 'F')
    L   = sq.Inductor(1, 'H')
    L_Δ = sq.Inductor(Δ, 'H')
    elements = {(0, 1): [L, C],
                (0, 2): [L, C],
                (1, 2): [L_Δ ], }
    return sq.Circuit(elements)

def premade_coupled_resonators(Δ):
    if Δ == 0:
        C   = sq.Capacitor(1, 'F')
        L   = sq.Inductor(1, 'H')
        elements = {(0, 1): [L, C],
                    (0, 2): [L, C], }
        return sq.Circuit(elements)

    else:
        C = sq.Capacitor(1, 'F')
        L = sq.Inductor(1, 'H')
        L_Δ = sq.Inductor(Δ, 'H')
        elements = {(0, 1): [L, C],
                    (0, 2): [L, C],
                    (1, 2): [L_Δ], }
        return sq.Circuit(elements)

def premade_single_resonator(Δ):
    if Δ == 0:
        C   = sq.Capacitor(1, 'F')
        L   = sq.Inductor(1, 'H')
        elements = {(0, 1): [L, C],}
        return sq.Circuit(elements)

    else:
        C   = sq.Capacitor(1, 'F')
        L   = sq.Inductor(1/(1+1/Δ), 'H')
        elements = {(0, 1): [L, C],}
        return sq.Circuit(elements)


#%% Set trunc nums and diag
n_eig = 4
trunc_num = 20
Δ=1

coupled_res     = premade_coupled_resonators    (Δ=Δ)
res             = premade_single_resonator      (Δ)

uncoupled_res   = premade_coupled_resonators    (Δ=0)
res_0           = premade_single_resonator      (Δ=0)

coupled_res     .set_trunc_nums([trunc_num,trunc_num])
uncoupled_res   .set_trunc_nums([trunc_num,trunc_num])
res             .set_trunc_nums([trunc_num])
res_0           .set_trunc_nums([trunc_num])

_ = coupled_res     .diag(n_eig)
_ = uncoupled_res   .diag(n_eig)
_ = res             .diag(n_eig)
_ = res_0           .diag(n_eig)

#%%
Q = res.charge_op(0)
Φ = res.flux_op(0)

Q_0 = res_0.charge_op(0)
Φ_0 = res_0.flux_op(0)

#%%
H_res_0 = 1/2 * Q_0**2 + 1/2 * Φ_0**2
H_res   = 1/2 * Q**2 + 1/2 * (1+1/Δ) * Φ**2

# H_res_0 = res_0.hamiltonian()
# H_res   = res.hamiltonian()

I = qt.identity(H_res.shape[0])

H_unc_res = qt.tensor(H_res_0,I) + qt.tensor(I,H_res_0)

H_coup_res = qt.tensor(H_res,I) + qt.tensor(I,H_res) + (1/Δ) *qt.tensor(Φ,Φ)

#%%
E_hand = sq_ext.diag(H_coup_res , n_eig, out=None)[0]

print( (E_hand-E_hand[0]))
print( coupled_res._efreqs )

print( (E_hand-E_hand[0])[1])
print( coupled_res._efreqs[1] )

#%%
E_hand = sq_ext.diag(H_unc_res , n_eig, out=None)[0]

print( (E_hand-E_hand[0]))
print( uncoupled_res._efreqs )

print( (E_hand-E_hand[0])[1])
print( uncoupled_res._efreqs[1] )

#%%
E_hand = sq_ext.diag(H_res , n_eig, out=None)[0]
print( (E_hand-E_hand[0])[1])
print( res._efreqs[1] )

#%%
E_hand = sq_ext.diag(H_res_0 , n_eig, out=None)[0]
print( E_hand-E_hand[0])
print( res_0._efreqs )

#%%
def KIT_resonator(C=15, Lq=25, Lr=10, Δ=0.1):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    resonator_elements = {
        (0, 1): [sq.Capacitor(C / 2, 'fF'), sq.Inductor(l / Lq, 'nH')],
    }
    return sq.Circuit(resonator_elements)


def KIT_fluxonium_no_JJ(C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C / 2 + Csh + CJ, 'fF'),
                 sq.Inductor(l / (Lq + 4 * Lr), 'nH')],
    }
    return sq.Circuit(fluxonium_elements)

def KIT_qubit_no_JJ(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, φ_ext=0.5):

    # Initialize loop(s)
    # loop = sq.Loop(φ_ext)

    # Circuit components
    C_01 = sq.Capacitor(C,       'fF')
    C_02 = sq.Capacitor(C,       'fF')
    C_12 = sq.Capacitor(CJ+Csh,  'fF')
    L_03 = sq.Inductor(Lr,       'nH')
    L_31 = sq.Inductor(Lq/2 - Δ, 'nH')#,  loops=[loop])
    L_23 = sq.Inductor(Lq/2 + Δ, 'nH')#,  loops=[loop])

    elements = {
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31],
        (1, 2): [C_12],
        (2, 3): [L_23],
    }

    # Create and return the circuit
    return sq.Circuit(elements)

#%% Set trunc nums and diag
n_eig = 4
trunc_num = 20
Δ=1
#
res     = KIT_resonator          (Δ=Δ)
flu     = KIT_fluxonium_no_JJ    (Δ=Δ)
qubit   = KIT_qubit_no_JJ        (Δ=Δ)
res_0   = KIT_resonator          (Δ=0)
flu_0   = KIT_fluxonium_no_JJ    (Δ=0)
qubit_0 = KIT_qubit_no_JJ        (Δ=0)
#
res     .set_trunc_nums([trunc_num])
flu     .set_trunc_nums([trunc_num])
qubit   .set_trunc_nums([1, trunc_num,trunc_num])
res_0   .set_trunc_nums([trunc_num])
flu_0   .set_trunc_nums([trunc_num])
qubit_0 .set_trunc_nums([1, trunc_num,trunc_num])
#
_ = res     .diag(n_eig)
_ = flu     .diag(n_eig)
_ = qubit   .diag(n_eig)
_ = res_0   .diag(n_eig)
_ = flu_0   .diag(n_eig)
_ = qubit_0 .diag(n_eig)

#%%
Q_res = res.charge_op(0)
Φ_res = res.flux_op(0)

Q_res_0 = res_0.charge_op(0)
Φ_res_0 = res_0.flux_op(0)

Q_flu = flu.charge_op(0)
Φ_flu = flu.flux_op(0)

Q_flu_0 = flu_0.charge_op(0)
Φ_flu_0 = flu_0.flux_op(0)

#%%
H_res_0 = 1/2 * Q_res_0**2 + 1/2 * Φ_res_0**2
# H_res   = 1/2 * Q**2 + 1/2 * (1+1/Δ) * Φ**2

# I = qt.identity(H_res.shape[0])

# H_unc_res = qt.tensor(H_res_0,I) + qt.tensor(I,H_res_0)
#
# H_coup_res = qt.tensor(H_res,I) + qt.tensor(I,H_res) + (1/Δ) *qt.tensor(Φ,Φ)

#%%
print( sq_ext.diag(H_res_0 , n_eig, out=None)[0] )
print( res_0._efreqs )




