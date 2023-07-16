import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt

# Fundamental constants
h    = 6.626e-34
GHz = 1e9
e0   = 1.602e-19
phi0 = h/(2*e0)
phi0_red = phi0/2/np.pi

# Circuit parameters in paper's convention
EJ = 50.0 * GHz * h
EC = 1.0 * GHz * h
αQ = 0.63
βQ = 0.15
κQ = 0.00
σQ = 0.00

# Pre-convert to explicit circuit elements for checking values
L_val   = βQ*(phi0_red**2/EJ)*(1/(1+κQ) + 1/(1-κQ) + 1/αQ)
C1_val  = (e0**2/(2*EC))*(1+κQ)
C2_val  = (e0**2/(2*EC))*(1-κQ)
C3_val  = (e0**2/(2*EC)) * αQ
JJ1_val = EJ/h*(1+κQ)
JJ2_val = EJ/h*(1-κQ)
JJ3_val = EJ / h * αQ

# Initialize loop(s)
loop1 = sq.Loop(0.0)  # "Value" corresponds to phiExt / phi0 threading the loop (can change later)

# Create circuit
C1 = sq.Capacitor(C1_val, 'F')
C2 = sq.Capacitor(C2_val, 'F')
C3 = sq.Capacitor(C1_val, 'F')
JJ1 = sq.Junction(JJ1_val, 'Hz', loops=[loop1])
JJ2 = sq.Junction(JJ2_val, 'Hz', loops=[loop1])
JJ3 = sq.Junction(JJ1_val, 'Hz', loops=[loop1])

elements = {
    (0, 1): [JJ1, C1], # Only include C_j if not included in JJ_j object
    (1, 2): [JJ2, C2],
    (2, 0): [JJ3, C3],
}

cr1 = sq.Circuit(elements)

#%%
cr1.description()

#%%
cr1.set_trunc_nums([10, 10])

#%%
n_eig = 7
n_ext = 300
phi_ext = np.linspace(0.4, 0.6, n_ext)
# To generate the spectrum of the circuit, firstly, we need to change and sweep the external flux of loop1 by the set_flux() method. Then, we need to find the eigenfrequencies of the circuit that correspond to that external flux via diag() method. The following lines of code find the spec a 2D NumPy array so that each column of it contains the eigenfrequencies with respect to its external flux.

# Calculate eigenvalue spectrum
spec = np.zeros((n_eig, len(phi_ext)))

for i, phi in enumerate(phi_ext):
    loop1.set_flux(phi)
    spec[:, i], _ = cr1.diag(n_eig=n_eig)
# Display eigenvalue spectrum
plt.figure()
for i in range(n_eig):
    plt.plot(phi_ext, (spec[i, :] - spec[0, :]))

plt.xlabel(r"$\Phi_{ext}/\Phi_0$")
plt.ylabel(r" $\omega_n / 2\pi$  (GHz)")
plt.show()



