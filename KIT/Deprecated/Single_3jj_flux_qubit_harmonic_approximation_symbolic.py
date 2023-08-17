import sympy as s 
s.init_printing()

# %%
phi = s.symbols('phi', real=True)

L = s.Function('L')(phi)
R = s.Function('R')(phi)

alpha, r = s.symbols('alpha, r', real=True)

m, omega, phi_0 = s.symbols('m, omega, φ0', real=True)

# %% Constants

# φ0 = s.acos(1/(2*α))
# m  = (2*α+1)/(4*r)
# ω = s.sqrt( 4*r*(2*α+1)/α)

# %% Harmonic wave functions
L = (m*ω/s.pi)**(1/4) * s.exp(-m*ω*(φ-φ0)**2/2)
G = (m*ω/s.pi)**(1/4) * s.exp(-m*ω*(φ+φ0)**2/2)

# %% Current operator

Q00 = s.integrate(L*G, φ)