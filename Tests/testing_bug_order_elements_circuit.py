import Modules.SQcircuit_extensions as sq_ext
import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(sq_ext)

#%% Define the circuit ’s elements
elements_I_first = {
    (0, 1): [sq.Inductor (10,  'nH'), sq.Capacitor(10,  'fF')],
}

elements_C_first = {
    (1, 0): [sq.Capacitor(10,  'fF')],
    (0, 1): [sq.Inductor (10,  'nH')],
}

#%% Create the circuits
circ_I_first  = sq.Circuit(elements_I_first)
circ_I_first.description()

#%%
circ_C_first  = sq.Circuit(elements_C_first)
circ_C_first.description()

# %% Things tested so far
# It is independent of the units, it doesnt even care if I dont give units.