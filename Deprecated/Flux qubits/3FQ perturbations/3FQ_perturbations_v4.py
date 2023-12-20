import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from palettable.cartocolors.sequential import BluGrn_7, OrYel_7 #, Magenta_7
from palettable.cartocolors.qualitative import Bold_4

import Modules.circuit_QED as cq
import Modules.figures as figs

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}", r"\usepackage{amsmath}"]
plt.rcParams.update({'font.size': 18})


#%% Single flux qubit H_eff vs perturbation
def E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective):
    
    Φ = np.array(Φ)
    γ = np.array(γ)
    
    EJ = 1 
    EC = r**-1
    if np.size(Φ) > np.size(γ):
        E     = np.zeros( (np.size(Φ), n0) )
        E_eff = np.zeros( (np.size(Φ), n0) )
        J_eff = np.zeros( (np.size(Φ),  4) )
    else:
        E     = np.zeros( (np.size(γ), n0) )
        E_eff = np.zeros( (np.size(γ), n0) )
        J_eff = np.zeros( (np.size(γ),  4) )
        
    
    if effective == False:
        for i in range( max( np.size(Φ),np.size(γ)  ) ):
            if np.size(Φ) > np.size(γ):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V, γ[i])
            E[i,:], _ = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)
        return E
    
    
    elif effective == 'First_order_perturbation':

        for i in range( max( np.size(Φ),np.size(γ)  ) ):
            _, ψb0  = cq.FQ_single(np.pi, α, β, nmax, 0, γ[i]).diagonalize(EC=EC, EJ=EJ, neig=n0)
            ψb0     = cq.real_flux_basis_single(ψb0, np.pi, α, β, nmax, EJ, EC, γ[i])
            if np.size(Φ) > np.size(γ):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V, γ[i])
                
            H_eff    = cq.H_eff_p1(ψb0, FQ.Hamiltonian(EC=EC,EJ=EJ))
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff )
        return E_eff, J_eff
    
    
    elif effective == 'Second_order_perturbation':

    
        for i in range( max( np.size(Φ),np.size(γ)  ) ):
            E, ψ  = cq.FQ_single(np.pi, α, β, nmax, 0, γ[i]).diagonalize(EC=EC, EJ=EJ, neig=n0+10)
            ψ     = cq.real_flux_basis_single(ψ, np.pi, α, β, nmax, EJ, EC, γ[i])
        
            ψb = ψ[:,:2]
            ψs = ψ[:,2:]
            Eb = E[:2]
            Es = E[2:]
            H0 = cq.FQ_single(np.pi, α, β, nmax, 0, γ[i]).Hamiltonian(EJ=EJ,EC=EC)
            if np.size(Φ) > np.size(γ):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V, γ[i])
                
            H_eff   = cq.H_eff_p2(ψb, Eb, ψs, Es, H0, FQ.Hamiltonian(EC=EC,EJ=EJ))
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff )
        return E_eff, J_eff
    
    
    elif effective == 'SWT':

        for i in range( max( np.size(Φ),np.size(γ)  ) ):
            _, ψb0  = cq.FQ_single(np.pi, α, β, nmax, 0, γ[i]).diagonalize(EC=EC, EJ=EJ, neig=n0)
            ψb0     = cq.real_flux_basis_single(ψb0, np.pi, α, β, nmax, EJ, EC, γ[i])
            if np.size(Φ) > np.size(γ):
                FQ = cq.FQ_single(Φ[i], α, β, nmax, V, γ)
            else:
                FQ = cq.FQ_single(Φ, α, β, nmax, V, γ[i])
            E[i,:], ψb = FQ.diagonalize(EC=EC, EJ=EJ, neig=n0)

            H_eff = cq.H_eff_SWT( ψb0, ψb, E[i,:])
            E_eff[i,:], _  = np.linalg.eigh( H_eff ) 
            J_eff[i,:]     = cq.decomposition_in_pauli_2x2( H_eff)
        return E_eff, J_eff
	
    elif effective == 'Charge_operator_First_order_perturbation':
        _, ψb0  = cq.FQ_single(np.pi, α, β, nmax, 0, γ).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = cq.real_flux_basis_single(ψb0, np.pi, α, β, nmax, EJ, EC, γ)

        H = -sp.diags( np.arange(-nmax,nmax+1) )
        H =  sp.kron(H, sp.eye(2*nmax+1))
		
        H_eff    = cq.H_eff_p1(ψb0, H)
        E_eff[0,:], _  = np.linalg.eigh( H_eff ) 
        J_eff[0,:]     = cq.decomposition_in_pauli_2x2( H_eff )
        return E_eff[0,:], J_eff[0,:]
	
    
    elif effective == 'Charge_operator_Second_order_perturbation':
        E, ψ  = cq.FQ_single(np.pi, α, β, nmax, 0, γ).diagonalize(EC=EC, EJ=EJ, neig=n0+10)
        ψ     = cq.real_flux_basis_single(ψ, np.pi, α, β, nmax, EJ, EC, γ)
        
        ψb = ψ[:,:2]
        ψs = ψ[:,2:]
        Eb = E[:2]
        Es = E[2:]
        H0 = cq.FQ_single(np.pi, α, β, nmax, 0, γ).Hamiltonian(EJ=EJ,EC=EC)
    
        H = -sp.diags( np.arange(-nmax,nmax+1) )
        H =  sp.kron(H, sp.eye(2*nmax+1))
	    
        H_eff   = cq.H_eff_p2(ψb, Eb, ψs, Es, H0, H)
        E_eff[0,:], _  = np.linalg.eigh( H_eff ) 
        J_eff[0,:]     = cq.decomposition_in_pauli_2x2( H_eff )
        return E_eff[0,:], J_eff[0,:]
    
    
    elif effective == 'Charge_operator_SWT':
        _, ψb0  = cq.FQ_single(np.pi, α, β, nmax, 0, γ).diagonalize(EC=EC, EJ=EJ, neig=n0)
        ψb0     = cq.real_flux_basis_single(ψb0, np.pi, α, β, nmax, EJ, EC, γ)
       
        H = -sp.diags( np.arange(-nmax,nmax+1) )
        H =  sp.kron(H, sp.eye(2*nmax+1))
  
        E[0,:], ψb = cq.diagonalize(H, neig=n0)
        H_eff = cq.H_eff_SWT( ψb0, ψb, E[0,:])
        E_eff[0,:], _  = np.linalg.eigh( H_eff ) 
        J_eff[0,:]     = cq.decomposition_in_pauli_2x2( H_eff)
        return E_eff[0,:], J_eff[0,:]

#%% Perturbation in Φ
n0 = 2 
nmax = 10

df =  np.linspace(0,0.05, 50)
f = 0.5 + df
Φ = f*2*np.pi
α = 0.7
r = 50 
β = 0 
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

ax.plot(df, E_exact[:,0], color = BluGrn_7 .mpl_colors[-3])
ax.plot(df, E_exact[:,1], color = OrYel_7  .mpl_colors[-3])

ax.plot(df, E_pert_1[:,0], color = BluGrn_7 .mpl_colors[-3], marker='.', linestyle='None')
ax.plot(df, E_pert_1[:,1], color = OrYel_7  .mpl_colors[-3], marker='.', linestyle='None')

ax.plot(df, E_pert_2[:,0], color = BluGrn_7 .mpl_colors[-3], marker='s', markersize=3, linestyle='None')
ax.plot(df, E_pert_2[:,1], color = OrYel_7  .mpl_colors[-3], marker='s', markersize=3, linestyle='None')

ax.plot(df, E_SWT[:,0], color = BluGrn_7 .mpl_colors[-3], marker='x', linestyle='None')
ax.plot(df, E_SWT[:,1], color = OrYel_7  .mpl_colors[-3], marker='x', linestyle='None')

ax.set_xlim([Φ[0]/(2*np.pi),Φ[-1]/(2*np.pi)])

ax.legend(['$E_0$','$E_1$','$E_0^{P1}$','$E_1^{P1}$','$E_0^{P2}$','$E_1^{P2}$','$E_0^{SWT}$','$E_1^{SWT}$'])

#%% Error spectrum vs Φ
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $\\Phi$ ')
ax.set_ylabel(' $E$ ')

ax.plot(df, np.abs(E_pert_1[:,0]-E_exact[:,0]), color = BluGrn_7 .mpl_colors[-3], linestyle='--')
ax.plot(df, np.abs(E_pert_1[:,1]-E_exact[:,1]), color = OrYel_7  .mpl_colors[-3], linestyle='--')

ax.plot(df, np.abs(E_pert_2[:,0]-E_exact[:,0]), color = BluGrn_7 .mpl_colors[-3], linestyle=':')
ax.plot(df, np.abs(E_pert_2[:,1]-E_exact[:,1]), color = OrYel_7  .mpl_colors[-3], linestyle=':')

ax.set_xlim([Φ[1]/(2*np.pi),Φ[-1]/(2*np.pi)])
ax.set_ylim([1e-12,1e-1])
ax.set_yscale('log')

ax.legend(['$E_0^{P1}$','$E_1^{P1}$','$E_0^{P2}$','$E_1^{P2}$'])
# fig.suptitle('Spectrum for $\\alpha =  $' + np.str(α) + ', $E_J/E_C =  $' + np.str(r) + ', $\\beta = $' + np.str(β)   )
    
# plt.savefig("Figures\E vs Phi.pdf")

#%% H_eff vs Φ
plt.rcParams.update({'font.size': 22})

f0 = np.arccos(1/(2*α))
m  = r*(2*α+1)/4
w  = np.sqrt( ((4*α**2-1)/α)/m )

df_op_h = np.pi*np.sqrt(4*α**2-1)*np.exp(-1/(m*w))/α

fig, ax = plt.subplots(ncols=3, nrows=1,figsize=(5*3, 4.5))

labels = [r'$J_I$',r'$J_x$',r'$J_y$',r'$J_z$']

ax[0].plot(df, df*df_op_h, color='k', linestyle = '--')

for i in range(1,4):
    
    ax[i-1].set_title(labels[i])
    ax[i-1].set_xlabel(r' $\delta_f$ ')
    ax[i-1].plot(df, J_pert_1[:,i], color = Bold_4.mpl_colors[1], marker='^', markevery=6,     linewidth=2)
    ax[i-1].plot(df, J_pert_2[:,i], color = Bold_4.mpl_colors[2], marker='s', markevery=(2,6), linewidth=2)
    ax[i-1].plot(df, J_SWT[:,i],    color = Bold_4.mpl_colors[3], marker='o', markevery=(4,6), linewidth=2)
    ax[i-1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[i-1].set_xlim([0,0.05])
    ax[i-1].set_xticks([0,0.025,0.05])
 
# ax[0].legend([])
ax[0].legend([r'$\epsilon$','$P1$','$P2$','$SWT$'],handlelength=1)	
ax[1].set_ylim([-1e-13,1e-13])

fig.tight_layout(pad=0.25)

figs.export('Perturbation flux.pdf')

#%% Perturbation in γ
n0 = 2 
nmax = 7

γ = np.linspace(0,10,10)

Φ = np.pi  
α = 0.7
r = 50 
β = 0
V = 1

d = 2*α + 1 + γ*(α+1)

f0 = np.arccos(1/(2*α))
m  = r*d/(2*γ+4)
w  = np.sqrt( ((4*α**2-1)/α)/m )

Jy_harm = -4 * (α*γ/(d*r)) * m*w*f0 * np.exp(-m*w*f0**2)

JI_harm = (α+1)*γ**2/(d*r)

E_pert_1, J_pert_1 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'First_order_perturbation')
E_pert_2, J_pert_2 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'Second_order_perturbation')
E_SWT,    J_SWT    = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'SWT')
E_exact            = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = False)

J_pert_1[:,0] -= J_pert_1[0,0] #Remove the constant term from sigma_I
J_pert_2[:,0] -= J_pert_2[0,0] #Remove the constant term from sigma_I
J_SWT   [:,0] -= J_SWT   [0,0] #Remove the constant term from sigma_I
#%% Spectrum vs V
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(' $V$ ')
ax.set_ylabel(' $E$ ')

ax.plot(γ, E_exact[:,0], color = BluGrn_7 .mpl_colors[-3])
ax.plot(γ, E_exact[:,1], color = OrYel_7  .mpl_colors[-3])

ax.plot(γ, E_pert_1[:,0], color = BluGrn_7 .mpl_colors[-3], marker='.', linestyle='None')
ax.plot(γ, E_pert_1[:,1], color = OrYel_7  .mpl_colors[-3], marker='.', linestyle='None')

ax.plot(γ, E_pert_2[:,0], color = BluGrn_7 .mpl_colors[-3], marker='s', markersize=3, linestyle='None')
ax.plot(γ, E_pert_2[:,1], color = OrYel_7  .mpl_colors[-3], marker='s', markersize=3, linestyle='None')

ax.plot(γ, E_SWT[:,0], color = BluGrn_7 .mpl_colors[-3], marker='x', linestyle='None')
ax.plot(γ, E_SWT[:,1], color = OrYel_7  .mpl_colors[-3], marker='x', linestyle='None')

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
plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(ncols=3, nrows=1,figsize=(5*3, 4.5))
labels = [r'$J_I$',r'$J_x$',r'$J_y$',r'$J_z$']

ax[1].plot(γ, Jy_harm , color='k', linestyle='--', label=r'$\eta$')
# ax[1].legend()
for i in range(1,4):
    ax[i-1].set_xlabel(r' $\gamma$ ')
    ax[i-1].set_title(labels[i])
    ax[i-1].plot(γ, J_pert_1[:,i], color = Bold_4.mpl_colors[1], marker='^', markevery=6,     linewidth=2)
    ax[i-1].plot(γ, J_pert_2[:,i], color = Bold_4.mpl_colors[3], marker='s', markevery=(2,6), linewidth=2)
    ax[i-1].plot(γ, J_SWT[:,i],    color = Bold_4.mpl_colors[2], marker='o', markevery=(4,6), linewidth=2) 
    ax[i-1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
# ax[0].set_ylabel(r'$\gamma=$'+str(γ), fontsize=26)   
# ax[0].legend(['$P1$','$P2$','$SWT$'])	

ax[0].set_ylim([-1e-12,1e-12])

fig.tight_layout(pad=0.7)
figs.export('Perturbation voltage gamma ='+ str(γ) +'.pdf')

#%% Charge operator
n0 = 2 
nmax = 10

Φ = np.pi  
α = 0.75
r = 50
β = 0
γ = 0
V = [0]

d = 2*α + 1 + γ*(α+1)
f0 = np.arccos(1/(2*α))
m  = r*d/(2*γ+4)
w  = np.sqrt( ((4*α**2-1)/α)/m )

Jy_harm = m*w*f0 * np.exp(-m*w*f0**2)

E_pert_1, J_pert_1 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'Charge_operator_First_order_perturbation')
E_pert_2, J_pert_2 = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'Charge_operator_Second_order_perturbation')
E_SWT,    J_SWT    = E_vs_pert(Φ, α, r, β, nmax, n0, V, γ, effective = 'Charge_operator_SWT')

print(Jy_harm,'\n')
print(J_pert_1,'\n')
print(J_pert_2,'\n')
print(J_SWT,'\n')