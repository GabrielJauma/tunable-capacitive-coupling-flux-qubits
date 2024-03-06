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
#%%
Csh = 15
C   = 15
CJ  = 3
Lq  = 25
Lr  = 10

Cf = C/2 + Csh + CJ
C_f = Cf * 1e-15
C_r = C/2 * 1e-15
#%% Premade circuits
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
n_eig = 3
trunc_num = 40
Δ=10

coupled_res     = premade_coupled_resonators    (Δ=Δ)
res             = premade_single_resonator      (Δ=Δ)

uncoupled_res   = premade_coupled_resonators    (Δ=0)
res_0           = premade_single_resonator      (Δ=0)

coupled_res     .set_trunc_nums([trunc_num, trunc_num])
uncoupled_res   .set_trunc_nums([trunc_num, trunc_num])
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

H_coup_res = qt.tensor(H_res,I) + qt.tensor(I,H_res) + (1/Δ) * qt.tensor(Φ,Φ)

#%%
Φ_unc, Q = sq_ext.get_node_variables(uncoupled_res, 'FC', True)

#%%
ψ_full = np.array([ψ_i.__array__()[:, 0] for ψ_i in res_0._evecs]).T
H_full = res_0.hamiltonian().__array__()

print(np.abs(ψ_full.conj().T @ H_full @ ψ_full))

#%%
ψ_full = np.array([ψ_i.__array__()[:, 0] for ψ_i in uncoupled_res._evecs]).T
H_full = uncoupled_res.hamiltonian().__array__()

print(np.abs(ψ_full.conj().T @ H_full @ ψ_full))

#%%
# ψ_frc = sq_ext.diag(H_unc_res,n_eig)[1]
ψ_frc = np.linalg.eigh(H_unc_res)[1]
print(np.abs(ψ_frc.conj().T @ H_unc_res.__array__() @ ψ_frc))

#%%
H_eff_full = sq_ext.H_eff_p1(uncoupled_res,uncoupled_res, out=None)
# H_eff_full -= H_eff_full[0,0]*np.eye(n_eig)

H_eff_frc = sq_ext.H_eff_p1_hamil(H_unc_res, H_unc_res, n_eig, out=None)
# H_eff_frc -= H_eff_frc[0,0]*np.eye(n_eig)

print(np.abs(H_eff_full))
print(np.abs(H_eff_frc))
print(np.round(np.abs(H_eff_full-H_eff_frc),4))

#%%
H_eff_full = sq_ext.H_eff_p1(coupled_res,coupled_res, out=None)
# H_eff_full -= H_eff_full[0,0]*np.eye(n_eig)

H_eff_frc = sq_ext.H_eff_p1_hamil(H_coup_res, H_coup_res, n_eig, out=None)
# H_eff_frc -= H_eff_frc[0,0]*np.eye(n_eig)

print(np.abs(H_eff_full))
print(np.abs(H_eff_frc))
print(np.round(np.abs(H_eff_full-H_eff_frc),4))
#%%
np.abs(uncoupled_res.hamiltonian_op('eig').__array__())


#%%
H_eff_full = sq_ext.H_eff_p1(res,res, out=None)
# H_eff_full -= H_eff_full[0,0]*np.eye(n_eig)

H_eff_frc = sq_ext.H_eff_p1_hamil(H_res, H_res, n_eig, out=None)
# H_eff_frc -= H_eff_frc[0,0]*np.eye(n_eig)

# print(np.abs(H_eff_full))
# print(np.abs(H_eff_frc))
print(np.round(np.abs(H_eff_full-H_eff_frc),4))

#%%
E, ψ = sq_ext.diag(H_unc_res , n_eig, out='GHz')
ψ_sq = np.array([ψ_i.__array__()[:, 0] for ψ_i in uncoupled_res._evecs])

print( E-E[0])
print( uncoupled_res.efreqs )
for i in range(n_eig):
    print(f'Energy error in state{i} = {np.abs((E-E[0])[i]- (uncoupled_res.efreqs[i]-uncoupled_res.efreqs[0]))}')
    print(f'Wavefunction error in state{i} = {1 - np.abs(ψ[i,:].conj().T @ uncoupled_res._evecs[i].__array__())[0]}')

#%%
E = sq_ext.diag(H_coup_res , n_eig, out=None)[0]

print( (E-E[0]))
print( coupled_res._efreqs )

print( (E-E[0])[1])
print( coupled_res._efreqs[1] )

#%%
E = sq_ext.diag(H_unc_res , n_eig, out=None)[0]

print( (E-E[0]))
print( uncoupled_res._efreqs )

print( (E-E[0])[1])
print( uncoupled_res._efreqs[1] )

#%%
E = sq_ext.diag(H_res , n_eig, out=None)[0]
print( (E-E[0])[1])
print( res._efreqs[1] )

#%%
E = sq_ext.diag(H_res_0 , n_eig, out=None)[0]
print( E )
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

def KIT_fluxonium(C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    loop_fluxonium = sq.Loop(φ_ext)
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C / 2 + Csh + CJ, 'fF'),
                 sq.Inductor(l / (Lq + 4 * Lr), 'nH', loops=[loop_fluxonium]),
                 sq.Junction(EJ, 'GHz', loops=[loop_fluxonium])],
    }
    return sq.Circuit(fluxonium_elements)

def KIT_qubit(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):

    # Initialize loop(s)
    loop = sq.Loop(φ_ext)

    # Circuit components
    C_01 = sq.Capacitor(C,       'fF')
    C_02 = sq.Capacitor(C,       'fF')
    C_12 = sq.Capacitor(CJ+Csh,  'fF')
    L_03 = sq.Inductor(Lr,       'nH')
    L_31 = sq.Inductor(Lq/2 - Δ, 'nH',  loops=[loop])
    L_23 = sq.Inductor(Lq/2 + Δ, 'nH',  loops=[loop])
    JJ_12= sq.Junction(EJ,       'GHz', loops=[loop])

    elements = {
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31],
        (1, 2): [C_12, JJ_12],
        (2, 3): [L_23],
    }

    # Create and return the circuit
    return sq.Circuit(elements)

#%% Set trunc nums and diag
n_eig = 4
trunc_num = 20
Δ=1

res     = KIT_resonator          (Δ=Δ)
# flu     = sq_fluxonium   (Δ=Δ)
# qubit   = sq_qubit      (Δ=Δ)
flu     = KIT_fluxonium_no_JJ    (Δ=Δ)
qubit   = KIT_qubit_no_JJ        (Δ=Δ)

res_0   = KIT_resonator          (Δ=0)
# flu_0   = sq_fluxonium    (Δ=0)
# qubit_0 = sq_qubit       (Δ=0)
flu_0   = KIT_fluxonium_no_JJ    (Δ=0)
qubit_0 = KIT_qubit_no_JJ        (Δ=0)
#
res     .set_trunc_nums([trunc_num])
flu     .set_trunc_nums([trunc_num])
qubit   .set_trunc_nums([1, trunc_num,trunc_num])
# qubit   .set_trunc_nums( [trunc_num,trunc_num])
res_0   .set_trunc_nums([trunc_num])
flu_0   .set_trunc_nums([trunc_num])
qubit_0 .set_trunc_nums([1, trunc_num,trunc_num])
# qubit_0 .set_trunc_nums([trunc_num,trunc_num])

_ = res     .diag(n_eig)
_ = flu     .diag(n_eig)
_ = qubit   .diag(n_eig)
_ = res_0   .diag(n_eig)
_ = flu_0   .diag(n_eig)
_ = qubit_0 .diag(n_eig)

#%%
l = Lq*(Lq+4*Lr) - 4*Δ**2
L_r = l/Lq * 1e-9
L_f = l/(Lq+4*Lr) * 1e-9
L_c = l/Δ * 1e-9

L_r_0 = (Lq+4*Lr) * 1e-9
L_f_0 = Lq * 1e-9

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
H_res_0 = 1/2 * Q_res_0**2 / C_r  + 1/2 * Φ_res_0**2 / L_r_0
# H_res = 1/2 * Q_res**2 / C_r  + 1/2 * Φ_res**2 / L_r
H_res = res.hamiltonian()

H_flu_0 = 1/2 * Q_flu_0**2 / C_f + 1/2 * Φ_flu_0**2 / L_f_0
# H_flu = 1/2 * Q_flu**2 / C_f + 1/2 * Φ_flu**2 / L_f
H_flu = flu.hamiltonian()

I_res = qt.identity(H_res.shape[0])
I_flu = qt.identity(H_flu.shape[0])

H_qubit_0 = qt.tensor(H_res_0,I_res) + qt.tensor(I_res,H_flu_0)
H_qubit = qt.tensor(H_res,I_flu) + qt.tensor(I_res,H_flu) + qt.tensor(Φ_res,Φ_flu) * 2 * Δ / l / 1e-9

#%%
Φ, Q = sq_ext.get_node_variables(qubit , 'FC', isolated='True')

#%%
E, ψ = sq_ext.diag(H_qubit , n_eig, out='GHz')
ψ_sq = np.array([ψ_i.__array__()[:, 0] for ψ_i in qubit._evecs])

print( E-E[0])
print( qubit.efreqs )
for i in range(n_eig):
    print(f'Energy error in state{i} = {np.abs((E-E[0])[i]- (qubit.efreqs[i]-qubit.efreqs[0]))}')
    print(f'Wavefunction error in state{i} = {1 - np.abs(ψ[:,i].conj().T @ qubit._evecs[i].__array__())[0]}')


#%%
E, ψ = sq_ext.diag(H_qubit , n_eig, out='GHz')
ψ_sq = np.array([ψ_i.__array__()[:, 0] for ψ_i in qubit._evecs])

print( E-E[0])
print( qubit.efreqs )
for i in range(n_eig):
    print(f'Energy error in state{i} = {np.abs((E-E[0])[i]- (qubit.efreqs[i]-qubit.efreqs[0]))}')
    print(f'Wavefunction error in state{i} = {1 - np.abs(ψ[:,i].conj().T @ qubit._evecs[i].__array__())[0]}')


#%%
E = sq_ext.diag(H_qubit_0 , n_eig, out='GHz')[0]

print( E-E [0])
print( qubit_0.efreqs )

print( (E-E[0])[1])
print( qubit_0.efreqs[1] )

#%%
E = sq_ext.diag(H_res , n_eig, out='GHz')[0]
print(E-E[0] )
print( res.efreqs )

print( (E-E[0])[1])
print( res.efreqs[1] )

#%%
E = sq_ext.diag(H_flu_0, n_eig, out='GHz')[0]
print(E-E[0])
print( flu_0.efreqs )

#%%
E = sq_ext.diag(H_res_0 , n_eig, out='GHz')[0] 
print(E-E[0] )
print( res_0.efreqs )




