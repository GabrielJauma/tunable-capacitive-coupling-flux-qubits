import Modules.SQcircuit_extensions as sq_ext
import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(sq_ext)

#%% define the circuit ’s elements
loop1 = sq.Loop()
C_r = sq.Capacitor(0.00020, "fF")
L_r = sq.Inductor (15.6, "nH")
L_2r = sq.Inductor (15.6 + 15.6, "nH")

# define the circuit
elements = {
            (0, 1): [L_r],
            (0, 2): [C_r],
            (1, 2): [L_r],
            # (0, 1): [L_2r, C_r],
            }

cr = sq.Circuit(elements)
cr.description()

#%%
cr.C

#%%
cr.set_trunc_nums([1,40])
# number of eigenvalues we aim for
n_eig=10

# array that contains the spectrum
phi = np.linspace(-0.1,0.6,3)

# array that contains the spectrum
spec = np.zeros((n_eig, len(phi)))

fig, ax = plt.subplots()
for i in range(len(phi)):
    # set the value of the flux external flux
    loop1.set_flux(phi[i])

    # diagonlize the circuit
    spec[:, i], _ = cr.diag(n_eig)

for i in range(1,n_eig):
    print(spec[i,:]- spec[0,:])
    ax.plot(phi, spec[i,:]- spec[0,:])

ax.set_xlabel("$\Phi_{ext}/\Phi_0$", fontsize=13)
ax.set_ylabel(r"$f_i-f_0$[GHz]", fontsize=13)
fig.show()