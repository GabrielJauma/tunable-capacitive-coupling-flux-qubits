import numpy as np
import matplotlib.pyplot as plt

from palettable.cartocolors.sequential import BluGrn_7, OrYel_7 #, Magenta_7
from palettable.cartocolors.qualitative import Bold_4

import Modules.circuit_QED as cq

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#%% Single flux qubit H_eff vs perturbation
def E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective):
    
    Φ = np.array(Φ)
    V = np.array(V)
    
    EJ = 1 
    EC = r**-1
    if np.size(Φ) > np.size(V):
        E     = np.zeros( (np.size(Φ), n0) )
        E_eff = np.zeros( (np.size(Φ), n0) )
        J_eff = np.zeros( (np.size(Φ),  4) )
    else:
        E     = np.zeros( (np.size(V), n0) )
        E_eff = np.zeros( (np.size(V), n0) )
        J_eff = np.zeros( (np.size(V),  4) )
        
    
    if effective == False:
        for i in range( max( np.size(Φ),np.size(V)  ) ):
            if np.size(Φ) > np.size(V):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V[i], γ)
            E[i,:], _ = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)
        return E
    
    
    elif effective == 'First_order_perturbation':
        _, ψb0  = cq.FQ_single(np.pi, α, β, nmax, 0, γ).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = cq.real_flux_basis_single(ψb0, np.pi, α, β, nmax, EJ, EC, γ)
        for i in range( max( np.size(Φ),np.size(V)  ) ):
            if np.size(Φ) > np.size(V):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V[i], γ)
                
            H_eff    = cq.H_eff_p1(ψb0, FQ.Hamiltonian(EC=EC,EJ=EJ))
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff )
        return E_eff, J_eff
    
    
    elif effective == 'Second_order_perturbation':
        E, ψ  = cq.FQ_single(np.pi, α, β, nmax, 0, γ).diagonalize(EC=EC, EJ=EJ, neig=n0+10)
        ψ     = cq.real_flux_basis_single(ψ, np.pi, α, β, nmax, EJ, EC, γ)
        
        ψb = ψ[:,:2]
        ψs = ψ[:,2:]
        Eb = E[:2]
        Es = E[2:]
        H0 = cq.FQ_single(np.pi, α, β, nmax, 0, γ).Hamiltonian(EJ=EJ,EC=EC)
    
        for i in range( max( np.size(Φ),np.size(V)  ) ):
            if np.size(Φ) > np.size(V):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V[i], γ)
                
            H_eff   = cq.H_eff_p2(ψb, Eb, ψs, Es, H0, FQ.Hamiltonian(EC=EC,EJ=EJ))
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff )
        return E_eff, J_eff
    
    
    elif effective == 'SWT':
        _, ψb0  = cq.FQ_single(np.pi, α, β, nmax, 0, γ).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = cq.real_flux_basis_single(ψb0, np.pi, α, β, nmax, EJ, EC, γ)
        
        for i in range( max( np.size(Φ),np.size(V)  ) ):
            if np.size(Φ) > np.size(V):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V[i], γ)
                
            E[i,:], ψb = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)
            H_eff = cq.H_eff_SWT( ψb0, ψb, E[i,:])
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff)
        return E_eff, J_eff

#%% Perturbation in Φ
n0 = 2 
nmax = 8

f = np.linspace(0.5,0.55, 50)
Φ = f*2*np.pi  #Must have an odd number to include the middle (qubit point)
α = 0.7 #np.linspace(0.5,1,31)
r = 50 #np.linspace(20, 100, 5) 
β = 0 #np.linspace(0,4,11)
V = 0
γ = 0

E_pert_1, J_pert_1 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'First_order_perturbation')
E_pert_2, J_pert_2 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'Second_order_perturbation')
E_SWT,    J_SWT    = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'SWT')
E_exact            = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = False)

#%% Spectrum vs Φ
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $E$ ')

ax.plot(f, E_exact[:,0], color = BluGrn_7 .mpl_colors[-3])
ax.plot(f, E_exact[:,1], color = OrYel_7  .mpl_colors[-3])

ax.plot(f, E_pert_1[:,0], color = BluGrn_7 .mpl_colors[-3], marker='.', linestyle='None')
ax.plot(f, E_pert_1[:,1], color = OrYel_7  .mpl_colors[-3], marker='.', linestyle='None')

ax.plot(f, E_pert_2[:,0], color = BluGrn_7 .mpl_colors[-3], marker='s', markersize=3, linestyle='None')
ax.plot(f, E_pert_2[:,1], color = OrYel_7  .mpl_colors[-3], marker='s', markersize=3, linestyle='None')

ax.plot(f, E_SWT[:,0], color = BluGrn_7 .mpl_colors[-3], marker='x', linestyle='None')
ax.plot(f, E_SWT[:,1], color = OrYel_7  .mpl_colors[-3], marker='x', linestyle='None')

ax.set_xlim([Φ[0]/(2*np.pi),Φ[-1]/(2*np.pi)])

ax.legend(['$E_0$','$E_1$','$E_0^{P1}$','$E_1^{P1}$','$E_0^{P2}$','$E_1^{P2}$','$E_0^{SWT}$','$E_1^{SWT}$'])

#%% Error spectrum vs Φ
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $E$ ')

ax.plot(f, np.abs(E_pert_1[:,0]-E_exact[:,0]), color = BluGrn_7 .mpl_colors[-3], linestyle='--')
ax.plot(f, np.abs(E_pert_1[:,1]-E_exact[:,1]), color = OrYel_7  .mpl_colors[-3], linestyle='--')

ax.plot(f, np.abs(E_pert_2[:,0]-E_exact[:,0]), color = BluGrn_7 .mpl_colors[-3], linestyle=':')
ax.plot(f, np.abs(E_pert_2[:,1]-E_exact[:,1]), color = OrYel_7  .mpl_colors[-3], linestyle=':')

ax.set_xlim([Φ[1]/(2*np.pi),Φ[-1]/(2*np.pi)])
ax.set_ylim([1e-12,1e-1])
ax.set_yscale('log')

ax.legend(['$E_0^{P1}$','$E_1^{P1}$','$E_0^{P2}$','$E_1^{P2}$'])
# fig.suptitle('Spectrum for $\\alpha =  $' + np.str(α) + ', $E_J/E_C =  $' + np.str(r) + ', $\\beta = $' + np.str(β)   )
    
# plt.savefig("Figures\E vs Phi.pdf")

#%% H_eff vs Φ
f0 = np.arccos(1/(2*α))
m  = r*(2*α+1)/4
w  = np.sqrt( ((4*α**2-1)/α)/m )
#df_op_h = -1j*np.pi*α*(np.exp(4*1j*f0)-1)*np.exp(-2*1j*f0-1/(m*w))
df_op_h = np.pi*np.sqrt(4*α**2-1)*np.exp(-1/(m*w))/α

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $J_I$ ')
ax.plot(f, J_pert_1[:,0], color = Bold_4.mpl_colors[1])
ax.plot(f, J_pert_2[:,0], color = Bold_4.mpl_colors[3])
ax.plot(f, J_SWT[:,0], color = Bold_4.mpl_colors[2])
ax.legend(['$P1$','$P2$','$SWT$'])

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $J_X$ ')
ax.plot(f, J_pert_1[:,1], color = Bold_4.mpl_colors[1])
ax.plot(f, J_pert_2[:,1], color = Bold_4.mpl_colors[3])
ax.plot(f, J_SWT[:,1], color = Bold_4.mpl_colors[2])
ax.plot(f, (f-0.5)*df_op_h, color = Bold_4.mpl_colors[0])
ax.legend(['$P1$','$P2$','$SWT$','$P1_h$'])

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $J_Y$ ')
ax.plot(f, J_pert_1[:,2], color = Bold_4.mpl_colors[1])
ax.plot(f, J_pert_2[:,2], color = Bold_4.mpl_colors[3])
ax.plot(f, J_SWT[:,2], color = Bold_4.mpl_colors[2])
ax.legend(['$P1$','$P2$','$SWT$'])

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $J_Z$ ')
ax.plot(f, J_pert_1[:,3], color = Bold_4.mpl_colors[1])
ax.plot(f, J_pert_2[:,3], color = Bold_4.mpl_colors[3])
ax.plot(f, J_SWT[:,3], color = Bold_4.mpl_colors[2])
ax.legend(['$P1$','$P2$','$SWT$'])

#%% Perturbation in V
n0 = 2 
nmax = 10

V = np.linspace(0.0001,0.01,4)

Φ = np.pi  
α = 0.85
r = 70 
β = 0
γ = 0.01

d = 2*α + 1 + γ*(α+1)

f0 = np.arccos(1/(2*α))
m  = r*d/(2*γ+4)
w  = np.sqrt( ((4*α**2-1)/α)/m )

Jy_harm = -8 * (α*r*γ/d) * m*w*f0 * np.exp(-m*w*f0**2)

E_pert_1, J_pert_1 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'First_order_perturbation')
E_pert_2, J_pert_2 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'Second_order_perturbation')
E_SWT,    J_SWT    = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'SWT')
E_exact            = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = False)

#%% Spectrum vs V
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $V$ ')
ax.set_ylabel(' $E$ ')

ax.plot(V, E_exact[:,0], color = BluGrn_7 .mpl_colors[-3])
ax.plot(V, E_exact[:,1], color = OrYel_7  .mpl_colors[-3])

ax.plot(V, E_pert_1[:,0], color = BluGrn_7 .mpl_colors[-3], marker='.', linestyle='None')
ax.plot(V, E_pert_1[:,1], color = OrYel_7  .mpl_colors[-3], marker='.', linestyle='None')

ax.plot(V, E_pert_2[:,0], color = BluGrn_7 .mpl_colors[-3], marker='s', markersize=3, linestyle='None')
ax.plot(V, E_pert_2[:,1], color = OrYel_7  .mpl_colors[-3], marker='s', markersize=3, linestyle='None')

ax.plot(V, E_SWT[:,0], color = BluGrn_7 .mpl_colors[-3], marker='x', linestyle='None')
ax.plot(V, E_SWT[:,1], color = OrYel_7  .mpl_colors[-3], marker='x', linestyle='None')

ax.legend(['$E_0$','$E_1$','$E_0^{P1}$','$E_1^{P1}$','$E_0^{P2}$','$E_1^{P2}$','$E_0^{SWT}$','$E_1^{SWT}$'])

#%% Error spectrum vs V
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $V$ ')
ax.set_ylabel(' $E$ ')

ax.plot(V, np.abs(E_pert_1[:,0]-E_exact[:,0]), color = BluGrn_7 .mpl_colors[-3], linestyle='--')
ax.plot(V, np.abs(E_pert_1[:,1]-E_exact[:,1]), color = OrYel_7  .mpl_colors[-3], linestyle='--')

ax.plot(V, np.abs(E_pert_2[:,0]-E_exact[:,0]), color = BluGrn_7 .mpl_colors[-3], linestyle=':')
ax.plot(V, np.abs(E_pert_2[:,1]-E_exact[:,1]), color = OrYel_7  .mpl_colors[-3], linestyle=':')

ax.set_xlim(V[1],V[-1])

ax.set_ylim([1e-12,1e0])
ax.set_yscale('log')

ax.legend(['$E_0^{P1}$','$E_1^{P1}$','$E_0^{P2}$','$E_1^{P2}$'])

#%% H_eff vs V
fig, ax = plt.subplots(ncols=4, nrows=1,figsize=(6*4, 4.5))

for i in range(4):

    ax[i].set_xlabel(' $V$ ')
    ax[i].set_ylabel(' $J_I$ ')
    ax[i].plot(V, J_pert_1[:,i], color = Bold_4.mpl_colors[1])
    ax[i].plot(V, J_pert_2[:,i], color = Bold_4.mpl_colors[3])
    ax[i].plot(V, J_SWT[:,i],    color = Bold_4.mpl_colors[2])
    if i == 2:
        ax[i].plot(V, Jy_harm * V *(2/r), color='k')
    ax[i].legend(['$P1$','$P2$','$SWT$'])

# # fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
# ax.set_xlabel(' $V$ ')
# ax.set_ylabel(' $J_X$ ')
# ax.plot(V, J_pert_1[:,1], color = Bold_4.mpl_colors[1])
# ax.plot(V, J_pert_2[:,1], color = Bold_4.mpl_colors[3])
# ax.plot(V, J_SWT[:,1],    color = Bold_4.mpl_colors[2])
# ax.legend(['$P1$','$P2$','$SWT$'])

# # fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
# ax.set_xlabel(' $V$ ')
# ax.set_ylabel(' $J_Y$ ')
# ax.plot(V, J_pert_1[:,2], color = Bold_4.mpl_colors[1] )
# ax.plot(V, J_pert_2[:,2], color = Bold_4.mpl_colors[3] )
# ax.plot(V, J_SWT[:,2],    color = Bold_4.mpl_colors[2] )
# ax.legend(['$P1$','$P2$','$SWT$'])

# # fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
# ax.set_xlabel(' $V$ ')
# ax.set_ylabel(' $J_Z$ ')
# ax.plot(V, J_pert_1[:,3], color = Bold_4.mpl_colors[1])
# ax.plot(V, J_pert_2[:,3], color = Bold_4.mpl_colors[3])
# ax.plot(V, J_SWT[:,3],    color = Bold_4.mpl_colors[2])
# ax.legend(['$P1$','$P2$','$SWT$'])