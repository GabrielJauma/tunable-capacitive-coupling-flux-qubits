import numpy as np
# import scipy.sparse as sp
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import time
from palettable.cartocolors.sequential import BluGrn_7, OrYel_7 #, Magenta_7
from palettable.cartocolors.qualitative import Bold_4
# import Modules.colorlines as cl

import Modules.circuit_QED as cq

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#%% Uncoupled 3FQ
def FQ_single(Φ, α, β, nmax, V=0, μ=0):
    
    #Iniciate circuit.
    FQ = cq.Circuit(3, nmax=nmax)

    #Add capacitances and junctions.
    FQ.add_capacitance(1.0, 0, 1)
    FQ.add_capacitance(1.0, 0, 2)
    FQ.add_capacitance(α + β, 1, 2)
    
    FQ.add_junction(1, 1, +1, 0, -1)
    FQ.add_junction(1, 2, -1, 0, +1)
    FQ.add_junction(α, 2, -1, 1, +1, -Φ)
    
    if V != 0:
        FQ.add_potential(1, μ, V)
        
    if V == 0 and μ != 0:
        FQ.C[2,2] += μ
    
    return FQ.set_ground([0])

#%% Real flux basis
def real_flux_basis(ψb0, ϕ, α, β, nmax, EJ, EC):
    # The derivative of the energy w.r.t. the flux is 
    # flux operator because it act so on the +/- superposition states of currents
    e=1e-5
    Hf_plus = FQ_single( ϕ+e*2*np.pi, α, β, nmax).Hamiltonian(EC=EC,EJ=EJ)
    Hf_minus= FQ_single( ϕ-e*2*np.pi, α, β, nmax).Hamiltonian(EC=EC,EJ=EJ)
    F = (Hf_plus - Hf_minus)/(2*e*2*np.pi)
    
    #Substraction of the phase so <0|F|1> is real.
    Phase=ψb0[:,0].T.conj() @ F @ ψb0[:,1]/np.abs(ψb0[:,0].T.conj() @ F @ ψb0[:,1])
    ψb0_new=ψb0
    ψb0_new[:,1]=ψb0[:,1]/Phase
    
    return ψb0_new
    

#%% Effective Hamiltonians

# First order perturbation theory
def H_eff_p1(ψb, H):
    H_eff = ψb.T.conj() @ H @ ψb
    return H_eff

# Second order perturbation theory
def H_eff_p2(ψb, Eb, ψs, Es, H0, H):
    
    l=ψb.shape[1]
    H_eff = np.zeros((l,l),dtype=complex) #matrix to store our results.
    
    #Loop to obtain each element of the matrix: <O>.
    for i in range(l):
        for j in range(l):
            H_eff[i,j] = ψb[:,i].T.conj() @ H @ ψb[:,j] + 1/2 * \
            sum(
                (1/(Eb[i]-Es[k])+1/(Eb[j]-Es[k]))*(ψb[:,i].T.conj() @ (H-H0) @ ψs[:,k]) * \
                (ψs[:,k].T.conj() @ (H-H0) @ ψb[:,j])
                for k in range(ψs.shape[1])
                ) 
    return H_eff

# Schrieffer-Wolff Transformation
def H_eff_SWT( ψb0, ψb, E):
    
    Q = ψb0.T.conj()@ψb
    U, s, Vh = np.linalg.svd(Q)
    A = U@Vh
    H_eff = A@np.diag(E)@A.T.conj()   
    return H_eff

#%% Perturbation vs Φ
def E_vs_Φ(Φ, α, r, β, nmax, n0, V, μ, effective):
    
    EJ = 1 
    EC = r**-1
    
    if effective == False:
        E = np.zeros( (len(Φ), n0) )
            
        for i in range(len(Φ)):
            FQ = FQ_single(Φ[i], α, β, nmax, V, μ)
            E[i,:], _ = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)
        return E
    
    elif effective == 'First_order_perturbation':
        E_eff = np.zeros( (len(Φ), n0) )
        J_eff = np.zeros( (len(Φ), 4) )
        
        _, ψb0  = FQ_single(np.pi, α, β, nmax, V, μ).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = real_flux_basis(ψb0, np.pi, α, β, nmax, EJ, EC)
    
        for i in range(len(Φ)):
            FQ       = FQ_single(Φ[i], α, β, nmax, V, μ)
            H_eff    = H_eff_p1(ψb0, FQ.Hamiltonian(EC=EC,EJ=EJ))
            
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff, 10, Print=False)
        return E_eff, J_eff
    
    elif effective == 'Second_order_perturbation':
        E_eff = np.zeros( (len(Φ), n0) )
        J_eff = np.zeros( (len(Φ), 4) )
        
        E, ψ  = FQ_single(np.pi, α, β, nmax, V, μ).diagonalize(EC=EC, EJ=EJ, neig=n0+4)
        ψ     = real_flux_basis(ψ, np.pi, α, β, nmax, EJ, EC)
        
        ψb = ψ[:,:2]
        ψs = ψ[:,2:]
        Eb = E[:2]
        Es = E[2:]
        H0 = FQ_single(np.pi, α, β, nmax, V, μ).Hamiltonian(EJ=EJ,EC=EC)
    
        for i in range(len(Φ)):
            H       = FQ_single(Φ[i], α, β, nmax, V, μ).Hamiltonian(EJ=EJ,EC=EC)
            H_eff   = H_eff_p2(ψb, Eb, ψs, Es, H0, H)
            
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff, 10, Print=False)
        return E_eff, J_eff
    
    elif effective == 'SWT':
        E = np.zeros( (len(Φ), n0) )
        E_eff = np.zeros( (len(Φ), n0) )
        J_eff = np.zeros( (len(Φ), 4) )
        
        _, ψb0  = FQ_single(np.pi, α, β, nmax, V, μ).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = real_flux_basis(ψb0, np.pi, α, β, nmax, EJ, EC)
        
        for i in range(len(Φ)):
            FQ = FQ_single(Φ[i], α, β, nmax, V, μ)
            E[i,:], ψb = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)
            
            H_eff = H_eff_SWT( ψb0, ψb, E[i,:])
            
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff, 10, Print=False)
        return E_eff, J_eff

#%% Perturbation vs V   
def E_vs_V(Φ, α, r, β, nmax, n0, V, μ, effective):
    
    EJ = 1 
    EC = r**-1
    
    if effective == False:
        E = np.zeros( (len(V), n0) )
            
        for i in range(len(V)):
            FQ = FQ_single(Φ, α, β, nmax, V[i], μ)
            E[i,:], _ = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)
        return E
    
    elif effective == 'First_order_perturbation':
        E_eff = np.zeros( (len(V), n0) )
        J_eff = np.zeros( (len(V), 4) )
        
        _, ψb0  = FQ_single(np.pi, α, β, nmax, 0, 0).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = real_flux_basis(ψb0, np.pi, α, β, nmax, EJ, EC)
    
        for i in range(len(V)):
            FQ       = FQ_single(Φ, α, β, nmax, V[i], μ)
            H_eff    = H_eff_p1(ψb0, FQ.Hamiltonian(EC=EC,EJ=EJ))
            
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff, 10, Print=False)
        return E_eff, J_eff
    
    elif effective == 'Second_order_perturbation':
        E_eff = np.zeros( (len(V), n0) )
        J_eff = np.zeros( (len(V), 4) )
        
        E, ψ  = FQ_single(np.pi, α, β, nmax, V, μ).diagonalize(EC=EC, EJ=EJ, neig=n0+4)
        ψ     = real_flux_basis(ψ, np.pi, α, β, nmax, EJ, EC)
        
        ψb = ψ[:,:2]
        ψs = ψ[:,2:]
        Eb = E[:2]
        Es = E[2:]
        H0 = FQ_single(np.pi, α, β, nmax, V, μ).Hamiltonian(EJ=EJ,EC=EC)
    
        for i in range(len(V)):
            H       = FQ_single(Φ, α, β, nmax, V[i], μ).Hamiltonian(EJ=EJ,EC=EC)
            H_eff   = H_eff_p2(ψb, Eb, ψs, Es, H0, H)
            
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff, 10, Print=False)
        return E_eff, J_eff
    
    elif effective == 'SWT':
        E = np.zeros( (len(V), n0) )
        E_eff = np.zeros( (len(V), n0) )
        J_eff = np.zeros( (len(V), 4) )
        
        _, ψb0  = FQ_single(np.pi, α, β, nmax, 0, 0).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = real_flux_basis(ψb0, np.pi, α, β, nmax, EJ, EC)
        
        for i in range(len(V)):
            FQ = FQ_single(Φ, α, β, nmax, V[i], μ)
            E[i,:], ψb = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)
            
            H_eff = H_eff_SWT( ψb0, ψb, E[i,:])
            
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff, 10, Print=False)
        return E_eff, J_eff

#%% Perturbation in Φ
n0 = 2 
nmax = 4

Φ = np.linspace(0.5,0.6, 51)*2*np.pi  #Must have an odd number to include the middle (qubit point)
α = 0.7 #np.linspace(0.5,1,31)
r = 50 #np.linspace(20, 100, 5) 
β = 0 #np.linspace(0,4,11)
V = 0
μ = 0

E_pert_1, J_pert_1 = E_vs_Φ(Φ, α, r, β, nmax, n0, V, μ, effective = 'First_order_perturbation')
E_pert_2, J_pert_2 = E_vs_Φ(Φ, α, r, β, nmax, n0, V, μ, effective = 'Second_order_perturbation')
E_SWT,    J_SWT    = E_vs_Φ(Φ, α, r, β, nmax, n0, V, μ, effective = 'SWT')
E_exact            = E_vs_Φ(Φ, α, r, β, nmax, n0, V, μ, effective = False)

#%% Spectrum vs Φ
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $E$ ')

ax.plot(Φ/(2*np.pi), E_exact[:,0], color = BluGrn_7 .mpl_colors[-3])
ax.plot(Φ/(2*np.pi), E_exact[:,1], color = OrYel_7  .mpl_colors[-3])

ax.plot(Φ/(2*np.pi), E_pert_1[:,0], color = BluGrn_7 .mpl_colors[-3], marker='.', linestyle='None')
ax.plot(Φ/(2*np.pi), E_pert_1[:,1], color = OrYel_7  .mpl_colors[-3], marker='.', linestyle='None')

ax.plot(Φ/(2*np.pi), E_pert_2[:,0], color = BluGrn_7 .mpl_colors[-3], marker='s', markersize=3, linestyle='None')
ax.plot(Φ/(2*np.pi), E_pert_2[:,1], color = OrYel_7  .mpl_colors[-3], marker='s', markersize=3, linestyle='None')

ax.plot(Φ/(2*np.pi), E_SWT[:,0], color = BluGrn_7 .mpl_colors[-3], marker='x', linestyle='None')
ax.plot(Φ/(2*np.pi), E_SWT[:,1], color = OrYel_7  .mpl_colors[-3], marker='x', linestyle='None')

ax.set_xlim([Φ[0]/(2*np.pi),Φ[-1]/(2*np.pi)])

ax.legend(['$E_0$','$E_1$','$E_0^{P1}$','$E_1^{P1}$','$E_0^{SWT}$','$E_1^{SWT}$'])
# fig.suptitle('Spectrum for $\\alpha =  $' + np.str(α) + ', $E_J/E_C =  $' + np.str(r) + ', $\\beta = $' + np.str(β)   )
    
# plt.savefig("Figures\E vs Phi.pdf")

#%% H_eff vs φ
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $J$ ')

ax.plot(Φ/(2*np.pi), J_pert_1[:,1], color = Bold_4.mpl_colors[1], marker='.')
ax.plot(Φ/(2*np.pi), J_pert_2[:,1], color = Bold_4.mpl_colors[3], marker='s', markersize=3)
ax.plot(Φ/(2*np.pi), J_SWT[:,1], color = Bold_4.mpl_colors[2], marker='x')

ax.set_xlim([Φ[0]/(2*np.pi),Φ[-1]/(2*np.pi)])

ax.legend(['$J_X^{P1}$','$J_Y^{P1}$','$J_Z^{P1}$,$J_X^{SWT}$','$J_Y^{SWT}$','$J_Z^{SWT}$'])

#%% Perturbation in V
n0 = 2 
nmax = 4

Φ = np.pi  
α = 0.7 
r = 50 
β = 0
V = np.linspace(0.0001,0.5,50)
μ = 1

E_pert_1, J_pert_1 = E_vs_V(Φ, α, r, β, nmax, n0, V, μ, effective = 'First_order_perturbation')
E_SWT,    J_SWT    = E_vs_V(Φ, α, r, β, nmax, n0, V, μ, effective = 'SWT')
E_exact            = E_vs_V(Φ, α, r, β, nmax, n0, V, μ, effective = False)

#%% Spectrum vs V
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $V$ ')
ax.set_ylabel(' $E$ ')

ax.plot(V, E_exact[:,0], color = BluGrn_7 .mpl_colors[-3])
ax.plot(V, E_exact[:,1], color = OrYel_7  .mpl_colors[-3])

ax.plot(V, E_pert_1[:,0], color = BluGrn_7 .mpl_colors[-3], marker='.', linestyle='None')
ax.plot(V, E_pert_1[:,1], color = OrYel_7  .mpl_colors[-3], marker='.', linestyle='None')

ax.plot(V, E_SWT[:,0], color = BluGrn_7 .mpl_colors[-3], marker='x', linestyle='None')
ax.plot(V, E_SWT[:,1], color = OrYel_7  .mpl_colors[-3], marker='x', linestyle='None')

ax.legend(['$E_0$','$E_1$','$E_0^{P1}$','$E_1^{P1}$'])
# fig.suptitle('Spectrum for $\\alpha =  $' + np.str(α) + ', $E_J/E_C =  $' + np.str(r) + ', $\\beta = $' + np.str(β)   )
    
# plt.savefig("Figures\E vs Phi.pdf")

#%% H_eff vs V
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $V$ ')
ax.set_ylabel(' $J$ ')

ax.plot(V, J_pert_1[:,2], color = Bold_4.mpl_colors[1], marker='.')
ax.plot(V, J_SWT[:,2], color = Bold_4.mpl_colors[2], marker='x')

ax.legend(['$J_X$','$J_Y$','$J_Z$'])