import SQcircuit as sq
import Modules.figures as figs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import qutip as qt
from itertools import product

plt.rcParams['backend'] = 'QtAgg'

#%% Constants
GHz = 1e9
nH  = 1e-9
fF  = 1e-15
h   = 6.626e-34
e0  = 1.602e-19
Φ_0 = h / (2 * e0)

#%% Conversion functions
def L_to_EL(L, L_units='nH', E_units='GHz'):

    if L_units=='nH':
        L = L * nH
    E_L = (Φ_0/(2*np.pi))**2 / L
    if E_units == 'GHz':
        return E_L / (GHz*h)
    else:
        return E_L

def C_to_EC(C, C_units='fF', E_units='GHz'):
    if C_units == 'fF':
        C = C * fF
    E_C = e0**2 / (2 * C)
    if E_units == 'GHz':
        return E_C / (GHz*h)
    else:
        return E_C

def EL_to_L(E_L, L_units='nH', E_units='GHz'):
    if E_units == 'GHz':
        E_L = E_L*GHz*h
    L = (Φ_0/(2*np.pi))**2 / E_L
    if L_units == 'nH':
        return L/nH
    else:
        return L

def EC_to_C(E_C, C_units='fF', E_units='GHz'):
    if E_units == 'GHz':
        E_C = E_C*GHz*h
    C = e0**2 / (2 * E_C)
    if C_units == 'fF':
        return C/fF
    else:
        return C

def ωR_to_LR_CR(ωR, LR=None, CR=None):
    if CR is None:
        CR = 1 / (LR * nH) / (ωR * 2 * np.pi * GHz) ** 2 / fF
    elif LR is None:
        LR = 1 / (CR * fF) / (ωR * 2 * np.pi * GHz) ** 2 / nH
    return LR, CR

def LF_LR_eff_to_Lq_Lr(LF, LR, Δ):
    Lq = 1/2*( LF + np.sqrt(LF * (16*Δ**2+LF*LR) / LR  ) )
    Lr = (LR- LF) * ( np.sqrt(LF*LR) + np.sqrt(16*Δ**2+LF*LR) ) / (8*np.sqrt(LF*LR))
    return Lq, Lr

def Lq_Lr_to_LF_LR_eff(Lq, Lr, Δ):
    l2 = Lq*(Lq+4*Lr) - 4*Δ**2
    LF = l2 / (Lq+4*Lr)
    LR = l2 / Lq
    return LF, LR

def CF_CR_eff_to_C_CJ_Csh(CF, CR):
    # This conversion does not fully define CJ and Csh, only their sum, but it is irreleventa so I split it equally
    C = CR * 2
    CJ_plus_Csh = CF - C/2
    CJ  = CJ_plus_Csh / 2
    Csh = CJ_plus_Csh / 2
    return C, CJ, Csh

def C_CJ_Csh_to_CF_CR_eff(C, CJ, Csh):
    CF = C/2 + CJ + Csh
    CR = C/2
    return CF, CR

#%% Experimental parameters
def get_experimental_parameters(qubit_name,return_effective=True):
    if qubit_name == 'qubit_1' or qubit_name == 'resonator_1':
        # qR7
        LF  = 22.06
        CF  = 32.15
        EJ  = 6.19
        ω_r = 6.46
        Δ   = 0.38 * 2
        LR  = 20.03 * 4

        CR = 1 / (LR * nH) / (ω_r * 2 * np.pi * GHz) ** 2 / fF
        Lq, Lr     =  LF_LR_eff_to_Lq_Lr   (LF=LF, LR=LR, Δ=Δ)
        C, CJ, Csh =  CF_CR_eff_to_C_CJ_Csh(CF=CF, CR=CR)

    elif qubit_name == 'qubit_2' or qubit_name == 'resonator_2':
        # bG1
        LF  = 20.2
        CF  = 22.7
        EJ  = 9.6
        ω_r = 6.274
        Δ   = 0.14 * 2
        LR  = 25.26 * 4

        CR = 1 / (LR* nH) / (ω_r* 2 * np.pi * GHz) ** 2 / fF
        Lq, Lr     = LF_LR_eff_to_Lq_Lr   (LF=LF, LR=LR, Δ=Δ)
        C, CJ, Csh = CF_CR_eff_to_C_CJ_Csh(CF=CF, CR=CR)

    elif qubit_name == 'qubit_3' or qubit_name == 'resonator_3':
        # qS16
        LF  = 31.6
        CF  = 25.2
        EJ  = 5.6
        ω_r = 5.22
        Δ   = 0.64 * 2
        LR  = 20.66 * 4

        CR = 1 / (LR * nH) / (ω_r * 2 * np.pi * GHz) ** 2 / fF
        Lq, Lr     = LF_LR_eff_to_Lq_Lr   (LF=LF, LR=LR, Δ=Δ)
        C, CJ, Csh = CF_CR_eff_to_C_CJ_Csh(CF=CF, CR=CR)

    if qubit_name == 'qubit_1_single_1' or qubit_name == 'resonator_1_single_1':
        # qR7
        LF  = 27.64
        CF  = 24.47
        EJ  = 5.06
        ω_r = 6.624
        Δ   = 0.47 * 2
        LR  = 25.09 * 4

        CR = 1 / (LR * nH) / (ω_r * 2 * np.pi * GHz) ** 2 / fF
        Lq, Lr     =  LF_LR_eff_to_Lq_Lr   (LF=LF, LR=LR, Δ=Δ)
        C, CJ, Csh =  CF_CR_eff_to_C_CJ_Csh(CF=CF, CR=CR)

    if return_effective :
        return CR, CF, LF, LR, EJ, Δ, ω_r
    else:
        return C, CJ, Csh, Lq, Lr, Δ, EJ



#%% Basic circuits made with SQcircuits
def sq_fluxonium(C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, EJ=10.0, φ_ext=0.5, nmax_r=15, nmax_f=25, C_F_eff=False, L_F_eff=False, E_L=False, E_C=False):

    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    if L_F_eff == False:
        L_F_eff = l / (Lq + 4 * Lr)

    if C_F_eff == False:
        C_F_eff = C / 2 + Csh + CJ

    if E_L:
        L_F_eff = EL_to_L(E_L)
    if E_C:
        C_F_eff = EC_to_C(E_C)

    loop_fluxonium = sq.Loop(φ_ext)
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C_F_eff, 'fF'),
                 sq.Inductor(L_F_eff , 'nH', loops=[loop_fluxonium]),
                 sq.Junction(EJ,      'GHz', loops=[loop_fluxonium])],
    }

    fluxonium = sq.Circuit(fluxonium_elements)
    fluxonium.set_trunc_nums([nmax_f])
    return fluxonium

def sq_resonator(C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, EJ=10.0, φ_ext=0.5, nmax_r=15, nmax_f=25, C_R_eff=False, L_R_eff=False, E_L=False, E_C=False):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    if C_R_eff == False:
        C_R_eff = C / 2

    if L_R_eff == False:
        L_R_eff = l / Lq

    if E_L:
        L_R_eff = EL_to_L(E_L)
    if E_C:
        C_R_eff = EC_to_C(E_C)

    resonator_elements = {
        (0, 1): [sq.Capacitor(C_R_eff, 'fF'), sq.Inductor(L_R_eff, 'nH')],
    }

    resonator = sq.Circuit(resonator_elements)
    resonator.set_trunc_nums([nmax_r])
    return resonator

def sq_qubit(C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, EJ=10.0, φ_ext=0.5, nmax_r=15, nmax_f=25):

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

    qubit = sq.Circuit(elements)
    try:
        qubit.set_trunc_nums([nmax_r, nmax_f])
    except:
        qubit.set_trunc_nums([1, nmax_r, nmax_f])

    # Create and return the circuit
    return qubit

#%% Composite circuits made with SQcircuits
def sq_qubit_C_qubit(CC, C, CJ, Csh, Lq, Lr, Δ, EJ, C_prime, CJ_prime, Csh_prime, Lq_prime, Lr_prime, Δ_prime, EJ_prime,
                     φ_ext=0.5, φ_ext_prime=0.5, nmax_r=5, nmax_f=10, only_inner = True, compensate_extra_cap = False, only_renormalization=False):
    loop = sq.Loop(φ_ext)
    loop_prime = sq.Loop(φ_ext_prime)

    # Circuit components
    # Unit Cell, nodes [0, 1, 2, 3]
    C_01 = sq.Capacitor(C, 'fF')
    C_02 = sq.Capacitor(C, 'fF')
    C_12 = sq.Capacitor(CJ + Csh, 'fF')
    L_03 = sq.Inductor(Lr, 'nH')
    L_31  = sq.Inductor(Lq / 2 - Δ, 'nH', loops=[loop])
    L_23  = sq.Inductor(Lq / 2 + Δ, 'nH', loops=[loop])
    JJ_12 = sq.Junction(EJ, 'GHz', loops=[loop])

    # Unit Cell prime, nodes [0, 4, 5, 6]
    C_04 = sq.Capacitor(C_prime, 'fF')
    C_05 = sq.Capacitor(C_prime, 'fF')
    C_45 = sq.Capacitor(CJ_prime + Csh_prime, 'fF')
    L_06 = sq.Inductor(Lr_prime, 'nH')
    L_64  = sq.Inductor(Lq_prime / 2 - Δ_prime, 'nH', loops=[loop_prime])
    L_56  = sq.Inductor(Lq_prime / 2 + Δ_prime, 'nH', loops=[loop_prime])
    JJ_45 = sq.Junction(EJ_prime, 'GHz', loops=[loop_prime])

    # Capacitive coupling
    C_C = sq.Capacitor(CC, 'fF')

    elements_qubit_C_qubit = {
        # qubit 1, nodes [0, 1, 2, 3]
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31],
        (1, 2): [C_12, JJ_12],
        (2, 3): [L_23],
        # qubit 2, nodes [0, 4, 5, 6]
        (0, 6): [L_06],
        (0, 4): [C_04],
        (0, 5): [C_05],
        (6, 4): [L_64],
        (4, 5): [C_45, JJ_45],
        (5, 6): [L_56],
    }

    if CC != 0:
        if only_inner :
            if only_renormalization:
                elements_qubit_C_qubit[(0, 2)] = [C_02, C_C]
                elements_qubit_C_qubit[(0, 4)] = [C_04, C_C]
            else:
                elements_qubit_C_qubit[(2, 4)] = [C_C]
            if compensate_extra_cap:
                elements_qubit_C_qubit[(0, 1)] = [C_01, C_C]
                elements_qubit_C_qubit[(0, 5)] = [C_05, C_C]
        else:
            if only_renormalization:
                elements_qubit_C_qubit[(0, 2)] = [C_02, C_C]
                elements_qubit_C_qubit[(0, 4)] = [C_04, C_C]
                elements_qubit_C_qubit[(0, 1)] = [C_01, C_C]
                elements_qubit_C_qubit[(0, 5)] = [C_05, C_C]
            else:
                elements_qubit_C_qubit[(2, 4)] = [C_C]
                elements_qubit_C_qubit[(1, 5)] = [C_C]

    qubit_C_qubit = sq.Circuit(elements_qubit_C_qubit)

    # Check where are the resonator modes and assign then their corresponding trunc num
    # This shit does not really work...
    CF, CR = C_CJ_Csh_to_CF_CR_eff(C=C, CJ=CJ, Csh=Csh)
    LF, LR = Lq_Lr_to_LF_LR_eff(Lq=Lq, Lr=Lr, Δ=Δ)
    CF_prime, CR_prime = C_CJ_Csh_to_CF_CR_eff(C=C_prime, CJ=CJ_prime, Csh=Csh_prime)
    LF_prime, LR_prime = Lq_Lr_to_LF_LR_eff(Lq=Lq_prime, Lr=Lr_prime, Δ=Δ_prime)

    C0_mat = np.array([[CF, 0, 0, 0],
                       [0, CR, 0, 0],
                       [0, 0, CF_prime, 0],
                       [0, 0, 0, CR_prime]])

    if only_inner == True:
        if not compensate_extra_cap:
            CC_mat = np.array([[CC / 4, -CC / 4, CC / 4, CC / 4],
                               [-CC / 4, CC / 4, -CC / 4, -CC / 4],
                               [CC / 4, -CC / 4, CC / 4, CC / 4],
                               [CC / 4, -CC / 4, CC / 4, CC / 4]])
        else:
            CC_mat = np.array([[CC / 2, 0, CC / 4, CC / 4],
                               [0, CC / 2, -CC / 4, -CC / 4],
                               [CC / 4, -CC / 4, CC / 2, 0],
                               [CC / 4, -CC / 4, 0, CC / 2]])

    else:
        CC_mat = np.array([[CC / 2, 0, CC / 2, 0],
                           [0, CC / 2, 0, -CC / 2],
                           [CC / 2, 0, CC / 2, 0],
                           [0, -CC / 2, 0, CC / 2]])

    C_mat = C0_mat + CC_mat

    C_inv = np.linalg.inv(C_mat)
    CR_tilde = C_inv[1, 1] ** -1
    CR_prime_tilde = C_inv[3, 3] ** -1

    ω = 1 / np.sqrt(CR_tilde * fF * LR * nH)
    ω_prime = 1 / np.sqrt(CR_prime_tilde * fF * LR_prime * nH)

    trunc_nums = [nmax_f, nmax_f, nmax_f, nmax_f]

    res_index_list = []
    for ω_r in [ω, ω_prime]:
        res_index = np.argsort( np.abs(ω_r - qubit_C_qubit.omega)/ω_r )[0]
        if res_index not in res_index_list:
            res_index_list.append(res_index)
            trunc_nums[res_index] = nmax_r
        else:
            res_index = np.argsort(np.abs(ω_r - qubit_C_qubit.omega)/ω_r)[1]
            res_index_list.append(res_index)
            trunc_nums[res_index] = nmax_r

    qubit_C_qubit.set_trunc_nums(trunc_nums)

    return qubit_C_qubit


def sq_qubit_C_qubit_C_qubit(Cc, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, EJ=10.0, φ_ext=0.5, nmax_r=5, nmax_f=10):
    # Initialize loop(s)
    loop1 = sq.Loop(φ_ext)  # "Value" corresponds to phiExt / phi0 threading the loop (can change later)
    loop2 = sq.Loop(φ_ext)
    loop3 = sq.Loop(φ_ext)

    C_01 = sq.Capacitor(C, 'fF')
    C_01_rc = sq.Capacitor(C + Cc, 'fF')  # With extra Cc for redout or coupling
    C_02 = sq.Capacitor(C, 'fF')
    C_02_rc = sq.Capacitor(C + Cc, 'fF')  # With extra Cc for redout or coupling
    C_12 = sq.Capacitor(CJ + Csh, 'fF')
    L_03 = sq.Inductor(Lr, 'nH')
    C_C = sq.Capacitor(Cc, 'fF')

    # Circuit elements
    # Elements for the three qubits decoupled
    L_31, L_23, JJ = [[] for _ in range(3)]
    for loop in [loop1, loop2, loop3]:
        L_31.append(sq.Inductor(Lq / 2 - Δ, 'nH', loops=[loop]))
        L_23.append(sq.Inductor(Lq / 2 + Δ, 'nH', loops=[loop]))
        JJ.append(sq.Junction(EJ, 'GHz', loops=[loop]))

    # Create the circuit
    elements = {
        # qubit 1, nodes [0, 1, 2, 3]
        (0, 3): [L_03],
        (0, 1): [C_01],
        (0, 2): [C_02],
        (3, 1): [L_31[0]],
        (1, 2): [C_12, JJ[0]],
        (2, 3): [L_23[0]],
        # qubit 2, nodes [0, 4, 5, 6]
        (0, 6): [L_03],
        (0, 4): [C_01],
        (0, 5): [C_02],
        (6, 4): [L_31[1]],
        (4, 5): [C_12, JJ[1]],
        (5, 6): [L_23[1]],
        # qubit 3, nodes [0,7, 8, 9]
        (0, 9): [L_03],
        (0, 7): [C_01],
        (0, 8): [C_02],
        (9, 7): [L_31[2]],
        (7, 8): [C_12, JJ[2]],
        (8, 9): [L_23[2]],
        # capacitive coupling
        (2, 4): [C_C],
        (5, 7): [C_C],
        (8, 1): [C_C],
    }

    qubit_C_qubit_C_qubit = sq.Circuit(elements)
    qubit_C_qubit_C_qubit.set_trunc_nums([nmax_r, nmax_r, nmax_r, nmax_f, nmax_f, nmax_f])

    return qubit_C_qubit_C_qubit


def sq_qubit_C_qubit_C_qubit_not_periodic(Cc, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, EJ=10.0, φ_ext=0.5, nmax_r=5,
                                          nmax_f=10):
    # Initialize loop(s)
    loop1 = sq.Loop(φ_ext)  # "Value" corresponds to phiExt / phi0 threading the loop (can change later)
    loop2 = sq.Loop(φ_ext)
    loop3 = sq.Loop(φ_ext)

    C_01 = sq.Capacitor(C, 'fF')
    C_01_rc = sq.Capacitor(C + Cc, 'fF')  # With extra Cc for redout or coupling
    C_02 = sq.Capacitor(C, 'fF')
    C_02_rc = sq.Capacitor(C + Cc, 'fF')  # With extra Cc for redout or coupling
    C_12 = sq.Capacitor(CJ + Csh, 'fF')
    L_03 = sq.Inductor(Lr, 'nH')
    C_C = sq.Capacitor(Cc, 'fF')

    # Circuit elements
    # Elements for the three qubits decoupled
    L_31, L_23, JJ = [[] for _ in range(3)]
    for loop in [loop1, loop2, loop3]:
        L_31.append(sq.Inductor(Lq / 2 - Δ, 'nH', loops=[loop]))
        L_23.append(sq.Inductor(Lq / 2 + Δ, 'nH', loops=[loop]))
        JJ.append(sq.Junction(EJ, 'GHz', loops=[loop]))

    # Create the circuit
    elements = {
        # qubit 1, nodes [0, 1, 2, 3]
        (0, 3): [L_03],
        (0, 1): [C_01_rc],
        (0, 2): [C_02],
        (3, 1): [L_31[0]],
        (1, 2): [C_12, JJ[0]],
        (2, 3): [L_23[0]],
        # qubit 2, nodes [0, 4, 5, 6]
        (0, 6): [L_03],
        (0, 4): [C_01],
        (0, 5): [C_02],
        (6, 4): [L_31[1]],
        (4, 5): [C_12, JJ[1]],
        (5, 6): [L_23[1]],
        # qubit 3, nodes [0,7, 8, 9]
        (0, 9): [L_03],
        (0, 7): [C_01],
        (0, 8): [C_02_rc],
        (9, 7): [L_31[2]],
        (7, 8): [C_12, JJ[2]],
        (8, 9): [L_23[2]],
        # capacitive coupling
        (2, 4): [C_C],
        (5, 7): [C_C],
    }

    qubit_C_qubit_C_qubit = sq.Circuit(elements)
    qubit_C_qubit_C_qubit.set_trunc_nums([nmax_r, nmax_r, nmax_r, nmax_f, nmax_f, nmax_f])

    return qubit_C_qubit_C_qubit

#%% Miscellaneous circuits made with SQcircuits
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

def KITqubit_asym( Cc, α, C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):

    # Initialize loop(s)
    loop = sq.Loop(φ_ext)

    # Circuit components
    C_01 = sq.Capacitor(C + α*Cc, 'fF')
    C_02 = sq.Capacitor(C + Cc,   'fF')
    C_12 = sq.Capacitor(CJ+Csh,   'fF')
    L_03 = sq.Inductor(Lr,        'nH')
    L_31 = sq.Inductor(Lq/2 - Δ,  'nH',  loops=[loop])
    L_23 = sq.Inductor(Lq/2 + Δ,  'nH',  loops=[loop])
    JJ_12= sq.Junction(EJ,        'GHz', loops=[loop])

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

def KIT_fluxonium_no_JJ(C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1, nmax_r=15, nmax_f=25):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
    fluxonium_elements = {
        (0, 1): [sq.Capacitor(C / 2 + Csh + CJ, 'fF'),
                 sq.Inductor(l / (Lq + 4 * Lr), 'nH')],
    }
    fluxonium = sq.Circuit(fluxonium_elements)
    fluxonium.set_trunc_nums([nmax_f])
    return fluxonium

#%% Capacitance matrices
def C_mat_qubit_C_qubit(CC, CR, CF, CR_prime, CF_prime, only_inner, compensate_extra_cap, only_renormalization=False):
    # The basis here is changed with rspecto to the notes: first the fluxonium and then the resonator
    C0_mat = np.array([[CF, 0, 0, 0],
                       [0, CR, 0, 0],
                       [0, 0, CF_prime, 0],
                       [0, 0, 0, CR_prime]])

    if only_inner == True:
        if not compensate_extra_cap:
            CC_mat = np.array([[CC / 4, -CC / 4, CC / 4, CC / 4],
                               [-CC / 4, CC / 4, -CC / 4, -CC / 4],
                               [CC / 4, -CC / 4, CC / 4, CC / 4],
                               [CC / 4, -CC / 4, CC / 4, CC / 4]])
            if only_renormalization:
                CC_mat = np.array([[CC / 4, -CC / 4, 0, 0],
                                   [-CC / 4, CC / 4, 0, 0],
                                   [0, 0, CC / 4, CC / 4],
                                   [0, 0, CC / 4, CC / 4]])
        else:
            CC_mat = np.array([[CC / 2, 0, CC / 4, CC / 4],
                               [0, CC / 2, -CC / 4, -CC / 4],
                               [CC / 4, -CC / 4, CC / 2, 0],
                               [CC / 4, -CC / 4, 0, CC / 2]])
            if only_renormalization:
                CC_mat = np.array([[CC / 2, 0, 0, 0],
                                   [0, CC / 2, 0, 0],
                                   [0, 0, CC / 2, 0],
                                   [0, 0, 0, CC / 2]])

    else:
        CC_mat = np.array([[CC / 2, 0, CC / 2, 0],
                           [0, CC / 2, 0, -CC / 2],
                           [CC / 2, 0, CC / 2, 0],
                           [0, -CC / 2, 0, CC / 2]])
        if only_renormalization:
            CC_mat = np.array([[CC / 2, 0, 0, 0],
                               [0, CC / 2, 0, 0],
                               [0, 0, CC / 2, 0],
                               [0, 0, 0, CC / 2]])

    C_mat = C0_mat + CC_mat

    return C_mat

def C_mat_qubit_C_qubit_C_qubit(CC, CF1, CR1, CF2, CR2, CF3, CR3):

    C0_mat = np.diag([CF1, CR1, CF2, CR2, CF3, CR3])

    CC_mat = CC * np.array([[     1/4,  -1/4,    1/4,    1/4,      0,      0],
                            [    -1/4,   1/4,   -1/4,   -1/4,      0,      0],
                            [     1/4,  -1/4,    1/2,      0,    1/4,    1/4],
                            [     1/4,  -1/4,      0,    1/2,   -1/4,   -1/4],
                            [       0,     0,    1/4,   -1/4,    1/4,    1/4],
                            [       0,     0,    1/4,   -1/4,    1/4,    1/4]])

    return C0_mat + CC_mat

def C_mat_fluxonium_C_fluxonium_C_fluxonium(CC, CF1, CF2, CF3):

    C0_mat = np.diag([CF1, CF2, CF3])

    CC_mat = CC * np.array([[     1/4,     1/4,        0,  ],
                            [     1/4,     1/2,      1/4,  ],
                            [       0,     1/4,      1/4,  ]])

    return C0_mat + CC_mat

#%% Hamiltonians made by composing small circuits made with sqcircuits
def hamiltonian_qubit(fluxonium = None, resonator=None, Δ=0.1, C=15, CJ=3, Csh=15, Lq=25, Lr=10, EJ=10, φ_ext=0.5, nmax_r=15, nmax_f=25, C_int=None, return_Ψ_nonint=False, n_eig_Ψ_nonint=4):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    if fluxonium is None:
        fluxonium = sq_fluxonium(C=C, CJ=CJ, Csh=Csh, Lq=Lq, Lr=Lr, Δ=Δ, EJ=EJ, φ_ext=φ_ext, nmax_r=nmax_r, nmax_f=nmax_f)
    if resonator is None:
        resonator = sq_resonator(C=C, CJ=CJ, Csh=Csh, Lq=Lq, Lr=Lr, Δ=Δ, EJ=EJ, φ_ext=φ_ext, nmax_r=nmax_r, nmax_f=nmax_f)

    H_f = fluxonium.hamiltonian()
    H_r = resonator.hamiltonian()

    if return_Ψ_nonint:
        E_f, Ψ_f = diag(H_f, n_eig_Ψ_nonint, solver='numpy', remove_ground=True)
        E_r, Ψ_r = diag(H_r, n_eig_Ψ_nonint, solver='numpy', remove_ground=True)
        E_0, Nf_Nr = generate_and_prioritize_energies([E_f, E_r], n_eig_Ψ_nonint)
        Ψ_0 = [qt.tensor(qt.Qobj(Ψ_f[:,Nf_Nr_i[0]]), qt.Qobj(Ψ_r[:,Nf_Nr_i[1]]) ) for Nf_Nr_i in Nf_Nr]

    I_f = qt.identity(H_f.shape[0])
    I_r = qt.identity(H_r.shape[0])

    Φ_f = fluxonium.flux_op(0)
    Φ_r = resonator.flux_op(0)
    g_Φ = 2 * Δ / (l * nH)

    if C_int is None:
        H = qt.tensor(H_f, I_r) + qt.tensor(I_f, H_r) + g_Φ * qt.tensor(Φ_f, Φ_r)
    else:
        Q_f = fluxonium.charge_op(0)
        Q_r = resonator.charge_op(0)
        H = qt.tensor(H_f, I_r) + qt.tensor(I_f, H_r) + g_Φ * qt.tensor(Φ_f, Φ_r) +  C_int**-1 * fF**-1 * qt.tensor(Q_f,Q_r)

    if return_Ψ_nonint:
        return H, Ψ_0, E_0
    else:
        return H

def hamiltonian_qubit_C_qubit(CC, CR, CF, LF, LR, EJ, Δ, φ_ext, CR_prime, CF_prime, LF_prime, LR_prime, EJ_prime, Δ_prime, φ_ext_prime,
                              nmax_r=5, nmax_f=15, return_Ψ_nonint=False, n_eig_Ψ_nonint=4, only_inner = True, compensate_extra_cap=False, only_renormalization=False):

    C_mat = C_mat_qubit_C_qubit(CC, CR, CF, CR_prime, CF_prime, only_inner, compensate_extra_cap, only_renormalization)
    C_inv = np.linalg.inv(C_mat)

    CF_tilde = C_inv[0, 0] ** -1
    CR_tilde = C_inv[1, 1] ** -1
    CF_prime_tilde = C_inv[2, 2] ** -1
    CR_prime_tilde = C_inv[3, 3] ** -1
    fluxonium       = sq_fluxonium(C_F_eff=CF_tilde,       L_F_eff=LF,       Δ=Δ,       EJ=EJ,       nmax_f=nmax_f, φ_ext=φ_ext)
    resonator       = sq_resonator(C_R_eff=CR_tilde,       L_R_eff=LR,       Δ=Δ,       EJ=EJ,       nmax_r=nmax_r)
    fluxonium_prime = sq_fluxonium(C_F_eff=CF_prime_tilde, L_F_eff=LF_prime, Δ=Δ_prime, EJ=EJ_prime, nmax_f=nmax_f, φ_ext=φ_ext_prime)
    resonator_prime = sq_resonator(C_R_eff=CR_prime_tilde, L_R_eff=LR_prime, Δ=Δ_prime, EJ=EJ_prime, nmax_r=nmax_r)


    Lq,       Lr        = LF_LR_eff_to_Lq_Lr(LF=LF, LR=LR, Δ=Δ)
    Lq_prime, Lr_prime  = LF_LR_eff_to_Lq_Lr(LF=LF_prime, LR=LR_prime, Δ=Δ_prime)

    if return_Ψ_nonint:
        H_uc, Ψ_q_0, E_q_0 = hamiltonian_qubit(fluxonium, resonator, Lq=Lq, Lr=Lr, Δ=Δ, return_Ψ_nonint=return_Ψ_nonint)
        H_uc_prime, Ψ_q_0_prime, E_q_0_prime = hamiltonian_qubit(fluxonium_prime, resonator_prime, Lq=Lq_prime, Lr=Lr_prime, Δ=Δ_prime, return_Ψ_nonint=return_Ψ_nonint)
        Nq_Nq = generate_and_prioritize_energies([E_q_0, E_q_0_prime], n_eig_Ψ_nonint)[1]
        Ψ_0 = [qt.tensor([Ψ_q_0[Nq_Nq_i[0]], Ψ_q_0_prime[Nq_Nq_i[1]] ]) for Nq_Nq_i in Nq_Nq]
    else:
        H_uc = hamiltonian_qubit(fluxonium, resonator, Lq=Lq, Lr=Lr, Δ=Δ)
        H_uc_prime = hamiltonian_qubit(fluxonium_prime, resonator_prime, Lq=Lq_prime, Lr=Lr_prime, Δ=Δ_prime)

    I_uc = qt.identity(H_uc.dims[0])
    I_F  = qt.identity(nmax_f)
    I_R  = qt.identity(nmax_r)
    H_0 = qt.tensor(H_uc, I_uc) + qt.tensor(I_uc, H_uc_prime)

    if CC == 0:
        if return_Ψ_nonint:
            return H_0, Ψ_0
        else:
            return H_0

    Q_F       = fluxonium.charge_op(0)
    Q_R       = resonator.charge_op(0)
    Q_F_prime = fluxonium_prime.charge_op(0)
    Q_R_prime = resonator_prime.charge_op(0)
    Q_vec = [Q_F, Q_R, Q_F_prime, Q_R_prime]
    H_coupling = 0
    for i in range(4):
        for j in range(4):
            op_list = [I_F, I_R, I_F, I_R]
            if i == j: # we ommit the diagonal terms since we have already included the reonarmalizations (LR and LF tilde) in H_0.
                continue
            else:
                op_list[i] = Q_vec[i]
                op_list[j] = Q_vec[j]
                H_coupling += 1/2 * C_inv[i,j] * fF**-1 * qt.tensor(op_list)
    H = H_0 + H_coupling

    if return_Ψ_nonint:
        return H, Ψ_0
    else:
        return H

def hamiltonian_fluxonium_C_fluxonium (nmax_f, Cc, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, EJ=10.0, φ_ext=0.5, return_Ψ_nonint=False, n_eig_Ψ_nonint=4):
    fF = 1e-15
    C_C = Cc
    C_F = C / 2 + Csh + CJ

    C_mat = np.array([ [ C_F + C_C / 2,   -C_C / 2],
                       [ -C_C / 2,  C_F + C_C / 2]])

    C_inv = np.linalg.inv(C_mat)
    C_F_tilde = C_inv[1, 1] ** -1
    if Cc == 0:
        pass
    else:
        C_FF = C_inv[0, 1] ** -1

    fluxonium = sq_fluxonium(Lq=Lq, Lr=Lr, Δ=Δ, EJ=EJ, φ_ext=φ_ext, nmax_f=nmax_f, C_F_eff=C_F_tilde)

    if return_Ψ_nonint:
        H_qubit = fluxonium.hamiltonian()
        fluxonium.diag(2)
        Ψ_f = diag(H_qubit,2, solver='numpy')[1]
        Ψ_0 = [qt.tensor(qt.Qobj(Ψ_f[:,0]),qt.Qobj(Ψ_f[:,0])),qt.tensor(qt.Qobj(Ψ_f[:,0]),qt.Qobj(Ψ_f[:,1])),qt.tensor(qt.Qobj(Ψ_f[:,1]),qt.Qobj(Ψ_f[:,0])),qt.tensor(qt.Qobj(Ψ_f[:,1]),qt.Qobj(Ψ_f[:,1]))]
    else:
        H_qubit = fluxonium.hamiltonian()

    I_qubit = qt.identity(H_qubit.dims[0])

    q_f = fluxonium.charge_op(0)

    H_0 = qt.tensor(H_qubit, I_qubit) + qt.tensor(I_qubit, H_qubit)
    if Cc == 0:
        if return_Ψ_nonint:
            return H_0, Ψ_0
        else:
            return H_0

    H_coupling =  1 / (C_FF * fF) * qt.tensor(q_f, q_f)

    H = H_0 + H_coupling

    if return_Ψ_nonint:
        return H, Ψ_0
    else:
        return H

def hamiltonian_qubit_C_qubit_circuits(C_inv, circuits, Δs, nmax_r=5, nmax_f=10, return_H_0=False):

    fluxonium_1, resonator_1, fluxonium_2, resonator_2 = circuits
    Δ_1, Δ_2 = Δs

    H_qubit_1 = hamiltonian_qubit(fluxonium_1, resonator_1, Δ_1 )
    H_qubit_2 = hamiltonian_qubit(fluxonium_2, resonator_2, Δ_2 )

    I_R = qt.identity(nmax_r)
    I_F = qt.identity(nmax_f)
    I_qubit = qt.identity(H_qubit_1.dims[0])

    H_0 = (  qt.tensor(H_qubit_1, I_qubit)
           + qt.tensor(I_qubit, H_qubit_2) )


    Q_F1 = fluxonium_1.charge_op(0)
    Q_R1 = resonator_1.charge_op(0)
    Q_F2 = fluxonium_2.charge_op(0)
    Q_R2 = resonator_2.charge_op(0)


    Q_vec = [Q_F1, Q_R1, Q_F2, Q_R2]
    H_coupling = 0
    for i in range(4):
        for j in range(4):
            op_list = [I_F, I_R, I_F, I_R]
            if i == j: # we ommit the diagonal terms since we have already included the reonarmalizations (LR and LF tilde) in H_0.
                continue
            else:
                op_list[i] = Q_vec[i]
                op_list[j] = Q_vec[j]
                H_coupling += 1/2 * C_inv[i,j] * fF**-1 * qt.tensor(op_list)
    H = H_0 + H_coupling

    if return_H_0:
        return H_0, H
    else:
        return H


def hamiltonian_fluxonium_C_fluxonium_C_fluxonium(C_inv, circuits, nmax_f=10, return_H_0=False):

    fluxonium_1, fluxonium_2, fluxonium_3 = circuits

    H_1 = fluxonium_1.hamiltonian()
    H_2 = fluxonium_2.hamiltonian()
    H_3 = fluxonium_3.hamiltonian()

    I = qt.identity(nmax_f)

    H_0 = (  qt.tensor(H_1, I, I)
           + qt.tensor(I, H_2, I)
           + qt.tensor(I, I, H_3) )


    Q_F1 = fluxonium_1.charge_op(0)
    Q_F2 = fluxonium_2.charge_op(0)
    Q_F3 = fluxonium_3.charge_op(0)


    Q_vec = [Q_F1, Q_F2, Q_F3]
    H_coupling = 0
    for i in range(len(Q_vec)):
        for j in range(len(Q_vec)):
            op_list = [I, I, I]
            if i == j: # we ommit the diagonal terms since we have already included the reonarmalizations (LR and LF tilde) in H_0.
                continue
            else:
                op_list[i] = Q_vec[i]
                op_list[j] = Q_vec[j]
                H_coupling += 1/2 * C_inv[i,j] * fF**-1 * qt.tensor(op_list)
    H = H_0 + H_coupling

    if return_H_0:
        return H_0, H
    else:
        return H

def hamiltonian_qubit_C_qubit_C_qubit(C_inv, circuits, Δs, nmax_r=5, nmax_f=10, return_H_0=False):

    fluxonium_1, resonator_1, fluxonium_2, resonator_2, fluxonium_3, resonator_3 = circuits
    Δ_1, Δ_2, Δ_3 = Δs

    H_qubit_1 = hamiltonian_qubit(fluxonium_1, resonator_1, Δ_1 )
    H_qubit_2 = hamiltonian_qubit(fluxonium_2, resonator_2, Δ_2 )
    H_qubit_3 = hamiltonian_qubit(fluxonium_3, resonator_3, Δ_3 )

    I_R = qt.identity(nmax_r)
    I_F = qt.identity(nmax_f)
    I_qubit = qt.identity(H_qubit_1.dims[0])

    H_0 = (  qt.tensor(H_qubit_1, I_qubit, I_qubit)
           + qt.tensor(I_qubit, H_qubit_2, I_qubit)
           + qt.tensor(I_qubit, I_qubit, H_qubit_3) )


    Q_F1 = fluxonium_1.charge_op(0)
    Q_R1 = resonator_1.charge_op(0)
    Q_F2 = fluxonium_2.charge_op(0)
    Q_R2 = resonator_2.charge_op(0)
    Q_F3 = fluxonium_3.charge_op(0)
    Q_R3 = resonator_3.charge_op(0)


    Q_vec = [Q_F1, Q_R1, Q_F2, Q_R2, Q_F3, Q_R3]
    H_coupling = 0
    for i in range(6):
        for j in range(6):
            op_list = [I_F, I_R, I_F, I_R, I_F, I_R]
            if i == j: # we ommit the diagonal terms since we have already included the reonarmalizations (LR and LF tilde) in H_0.
                continue
            else:
                op_list[i] = Q_vec[i]
                op_list[j] = Q_vec[j]
                H_coupling += 1/2 * C_inv[i,j] * fF**-1 * qt.tensor(op_list)
    H = H_0 + H_coupling

    if return_H_0:
        return H_0, H
    else:
        return H

#%% Low-energy hamiltonians
def hamiltonian_fluxonium_low_ene(ω_q, μ, φ_ext):
    σ_x, σ_y, σ_z = pauli_matrices()

    H = ω_q/2 * σ_z + μ * (φ_ext-0.5) * σ_x

    return H

def hamiltonian_qubit_low_ene(ω_q, μ, ω_r, g_Φ, φ_ext, g_q=0):
    σ_x, σ_y, σ_z = pauli_matrices()
    a_dag   = create(3)
    a       = annihilate(3)

    H_f = hamiltonian_fluxonium_low_ene(ω_q, μ, φ_ext)
    H_r = ω_r * a_dag @ a

    I_r = qt.identity(H_r.shape[0])
    I_f = qt.identity(H_f.shape[0])

    H = np.kron(H_f, I_r) + np.kron(I_f, H_r) + g_Φ * np.kron(σ_x, a_dag + a) + g_q * np.kron(σ_y, 1j*(a_dag - a))

    return H

def hamiltonian_qubit_low_ene_mod_phase(ω_q, μ, ω_r, g, φ_ext, θ=0):
    σ_x, σ_y, σ_z = pauli_matrices()
    a_dag   = create(3)
    a       = annihilate(3)

    H_f = hamiltonian_fluxonium_low_ene(ω_q, μ, φ_ext)
    H_r = ω_r * a_dag @ a

    I_r = qt.identity(H_r.shape[0])
    I_f = qt.identity(H_f.shape[0])

    H = np.kron(H_f, I_r) + np.kron(I_f, H_r) + g * (np.cos(θ) * np.kron(σ_x, a_dag + a) + np.sin(θ) * np.kron(σ_y, 1j*(a_dag - a)))

    return H

def hamiltonian_fluxonium_C_fluxonium_low_ene(H_f1, H_f2, g_q):
    σ_x, σ_y, σ_z = pauli_matrices()

    # H_f1 = hamiltonian_fluxonium_low_ene(ω_q_1, μ_1, φ_ext_1)
    # H_f2 = hamiltonian_fluxonium_low_ene(ω_q_2, μ_2, φ_ext_2)

    I = qt.identity(H_f1.shape[0])

    H = np.kron(H_f1, I) + np.kron(I, H_f2) + g_q * np.kron(σ_y, σ_y)

    return H

def hamiltonian_fluxonium_C_fluxonium_fluxonium_low_ene(H_f1, H_f2, H_f3, g_q):
    σ_x, σ_y, σ_z = pauli_matrices()

    # H_f1 = hamiltonian_fluxonium_low_ene(ω_q_1, μ_1, φ_ext_1)
    # H_f2 = hamiltonian_fluxonium_low_ene(ω_q_2, μ_2, φ_ext_2)

    I = qt.identity(H_f1.shape[0])

    H = np.kron(H_f1, I) + np.kron(I, H_f2) + g_q * np.kron(σ_y, σ_y)

    return H

#%% Circuits vs parameters
def KIT_qubit_vs_param(C = 15, CJ = 3, Csh= 15, Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5, nmax_r=15, nmax_f=25, model='composition'):

    parameters_list = expand_list_with_array([C, CJ, Csh, Lq, Lr, Δ, EJ, φ_ext, nmax_r, nmax_f])
    H_qubit_list = []

    for parameters in parameters_list:
        if model == 'full_circuit':
            H_qubit_list.append(sq_qubit(*parameters).hamiltonian())
        if model == 'composition':
            fluxonium = sq_fluxonium(*parameters)
            resonator = sq_resonator(*parameters)
            H_qubit_list.append(hamiltonian_qubit(fluxonium, resonator, Δ = parameters[5], Lq = parameters[3], Lr = parameters[4]))

    return H_qubit_list

#%% Transformations to obtain effective Hamiltonians

def H_eff_p1(H_0, H, n_eig, out='GHz', real=True, remove_ground = False):

    ψ_0 = diag(H_0, n_eig, real=real, solver='numpy')[1]
    H_eff  = ψ_0.conj().T @ H.__array__() @ ψ_0

    if out == 'GHz':
        H_eff /= GHz * 2 * np.pi

    if remove_ground:
        H_eff -=  H_eff[0,0]*np.eye(len(H_eff))

    if real:
        if np.allclose(np.imag(H_eff),0):
            H_eff = np.real(H_eff)

    return H_eff

def H_eff_p1_large(ψ_0, H, out='GHz', real=True, remove_ground = False):
    n_eig = len(ψ_0)
    H_eff = np.zeros([n_eig, n_eig], dtype='complex')
    for i in range(n_eig):
        for j in range(n_eig):
            H_eff[i, j] = (ψ_0[i].dag() * H * ψ_0[j]).data[0, 0]

    if out == 'GHz':
        H_eff /= GHz * 2 * np.pi

    if remove_ground:
        H_eff -=  H_eff[0,0]*np.eye(len(H_eff))

    if real:
        if np.allclose(np.imag(H_eff),0):
            H_eff = np.real(H_eff)

    return H_eff

def H_eff_p2(H_0, H, n_eig, out='GHz', real=False, remove_ground=False, solver='scipy', ψ_0=None):
    if ψ_0 is None:
        E_0, ψ_0 = diag(H_0, n_eig, real=real, solver='numpy', out=out)
        E, ψ = diag(H, n_eig, real=False, solver='scipy', out=out)
        try:
            H_0 = H_0.__array__()
        except:
            pass
        try:
            H = H.__array__()
        except:
            pass
        V = H - H_0

        if out == 'GHz':
            H_0 /= GHz * 2 * np.pi
            H /= GHz * 2 * np.pi
            V /= GHz * 2 * np.pi

        # H_eff_1 = ψ_0.conj().T @ H @ ψ_0

        H_eff_2 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.

        for i in range(n_eig):
            for j in range(n_eig):
                H_eff_2[i, j] = 1 / 2 * sum(
                    (1 / (E_0[i] - E[k]) + 1 / (E_0[j] - E[k])) *
                    (ψ_0[:,i].conj().T @ V @ ψ[:,k])*
                    (ψ[:,k].conj().T @ V @ ψ_0[:,j])
                    for k in range(n_eig))


    else:
        E_0 = diag(H, n_eig=len(ψ_0), real=False, solver='Qutip', out=out, qObj=True)[0]
        E, ψ = diag(H, n_eig=len(ψ_0), real=False, solver='Qutip', out=out, qObj=True)
        subspace_indices = find_closest_indices(E_0, E, tolerance=0.1)
        if len(subspace_indices) != len(ψ_0):
            raise ValueError('Something is wrong with the tolerance')
        ψ = ψ[subspace_indices]
        n_eig = len(ψ_0)

        V = H - H_0

        if out == 'GHz':
            H_0 /= GHz * 2 * np.pi
            H /= GHz * 2 * np.pi
            V /= GHz * 2 * np.pi

        # H_eff_1 = ψ_0.conj().T @ H @ ψ_0

        H_eff_2 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.

        for i in range(n_eig):
            for j in range(n_eig):
                H_eff_2[i, j] = 1 / 2 * sum(
                    (1 / (E_0[i] - E[k]) + 1 / (E_0[j] - E[k])) *
                    (ψ_0[i].dag() * V * ψ[k]).data[0, 0] *
                    (ψ[k].dag() * V * ψ_0[j]).data[0, 0]
                    for k in range(n_eig))



    # H_eff = H_eff_1 + H_eff_2
    H_eff = H_eff_2


    if remove_ground:
        H_eff -= H_eff[0, 0] * np.eye(len(H_eff))

    if real:
        if np.allclose(np.imag(H_eff), 0):
            H_eff = np.real(H_eff)

    return H_eff

def H_eff_p2_large(ψ_0, ψ, E_0, E, V, remove_ground=False):
    n_eig = len(ψ_0)
    H_eff = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.

    for i in range(n_eig):
        for j in range(n_eig):
            H_eff[i, j] = 1 / 2 * sum(
                (1 / (E_0[i] - E[k]) + 1 / (E_0[j] - E[k])) *
                (ψ_0[i].dag() * V * ψ[k]).data[0, 0] *
                (ψ[k].dag() * V * ψ_0[j]).data[0, 0]
                for k in range(n_eig))

    if remove_ground:
        H_eff -= H_eff[0, 0] * np.eye(len(H_eff))

    return H_eff

def H_eff_SWT(H_0, H, n_eig, out='GHz', real=False, remove_ground=False, return_transformation=False,ψ_0=None ):

    ψ_0 = diag(H_0, n_eig, real=real, solver='numpy')[1]
    E, ψ = diag(H, n_eig, real=False, solver='scipy', out=out)

    Q = np.zeros((n_eig, n_eig), dtype=complex)
    for i in range(n_eig):
        for j in range(n_eig):
            Q[i, j] = ψ_0[:,i].conj().T @ ψ[:,j]


    U, s, Vh = np.linalg.svd(Q)
    A = U @ Vh

    H_eff = A @ np.diag(E) @ A.T.conj()


    if remove_ground:
        H_eff -= H_eff[0, 0] * np.eye(len(H_eff))

    if real:
        if np.allclose(np.imag(H_eff), 0):
            H_eff = np.real(H_eff)

    if return_transformation:
        return H_eff, A
    else:
        return H_eff


def H_eff_SWT_large(ψ_0, ψ, E, remove_ground=False, return_Q=False):
    n_eig = len(ψ_0)
    Q = np.zeros((n_eig, n_eig), dtype=complex)
    for i in range(n_eig):
        for j in range(n_eig):
            Q[i, j] = (ψ_0[i].dag() * ψ[j]).data[0,0]

    U, s, Vh = np.linalg.svd(Q)
    A = U @ Vh

    H_eff = A @ np.diag(E) @ A.T.conj()

    if remove_ground:
        H_eff -= H_eff[0, 0] * np.eye(len(H_eff))

    if return_Q:
        return H_eff, Q
    else:
        return H_eff


#%% Optimization functions
def find_resonance(H_target, input_circuit):
    # Step 1: Calculate the target gap ω_target
    ω_target = diag(H_target, n_eig=2, remove_ground=True)[0][1]

    # Step 2: Define the objective function to minimize the difference between ω_target and ω_input
    def objective(φ_ext):
        # Set the external flux of the input circuit
        loop = input_circuit.loops[0]
        loop.set_flux(φ_ext)

        # Diagonalize the input circuit Hamiltonian
        E_input = input_circuit.diag(n_eig=2)[0]
        ω_input = E_input[1] - E_input[0]

        # Return the absolute difference between ω_target and ω_input
        return np.abs(ω_target - ω_input)

    # Step 3: Use an optimization method to find the optimal φ_ext
    result = sp.optimize.minimize_scalar(objective, bounds=(0.5, 1), method='bounded')

    # Return the optimal φ_ext and the corresponding gap
    optimal_φ_ext = result.x

    return optimal_φ_ext
#%% Sorting and labeling functions
# def sq_get_energy_indices(qubit, fluxonium, resonator, n_eig=3):
#     try:
#         E_qubit = qubit.efreqs - qubit.efreqs[0]
#         E_fluxonium = fluxonium.efreqs - fluxonium.efreqs[0]
#         E_resonator = resonator.efreqs - resonator.efreqs[0]
#     except:
#         qubit    .diag(n_eig+2)
#         fluxonium.diag(n_eig)
#         resonator.diag(n_eig)
#         E_qubit      = qubit.efreqs    - qubit.efreqs[0]
#         E_fluxonium = fluxonium.efreqs - fluxonium.efreqs[0]
#         E_resonator = resonator.efreqs - resonator.efreqs[0]
#
#     n_eig = len(E_qubit)
#
#     N_fluxonium = np.zeros(n_eig, dtype='int')
#     N_resonator = np.zeros(n_eig, dtype='int')
#
#     E_matrix = E_fluxonium[:, np.newaxis] + E_resonator
#
#     tol = E_qubit[1]-E_qubit[0]
#     for k in range(n_eig):
#         ΔE_matrix = np.abs(E_matrix - E_qubit[k])
#         if ΔE_matrix.min() < tol:
#             N_fluxonium[k], N_resonator[k] = np.unravel_index(ΔE_matrix.argmin(), ΔE_matrix.shape)
#         else:
#             N_fluxonium[k], N_resonator[k] = [-123, -123]
#     return N_fluxonium, N_resonator

def sq_get_energy_indices_hamiltonian(H_qubit, H_fluxonium, H_resonator, n_eig=2):
    E_qubit     = diag(H_qubit    , n_eig=n_eig+4, remove_ground=True)[0]
    E_fluxonium = diag(H_fluxonium, n_eig=n_eig, remove_ground=True)[0]
    E_resonator = diag(H_resonator, n_eig=n_eig, remove_ground=True)[0]

    n_eig = len(E_qubit)

    N_fluxonium = np.zeros(n_eig, dtype='int')
    N_resonator = np.zeros(n_eig, dtype='int')

    E_matrix = E_fluxonium[:, np.newaxis] + E_resonator

    tol = E_qubit[1]-E_qubit[0]
    for k in range(n_eig):
        ΔE_matrix = np.abs(E_matrix - E_qubit[k])
        if ΔE_matrix.min() < tol:
            N_fluxonium[k], N_resonator[k] = np.unravel_index(ΔE_matrix.argmin(), ΔE_matrix.shape)
        else:
            N_fluxonium[k], N_resonator[k] = [-123, -123]
    return N_fluxonium, N_resonator

def sq_get_energy_indices(E_composite, E_1, E_2):
    n_eig = len(E_composite)

    N_fluxonium = np.zeros(n_eig, dtype='int')
    N_resonator = np.zeros(n_eig, dtype='int')

    E_matrix = E_1[:, np.newaxis] + E_2

    tol = E_composite[1]-E_composite[0]
    for k in range(n_eig):
        ΔE_matrix = np.abs(E_matrix - E_composite[k])
        if ΔE_matrix.min() < tol:
            N_fluxonium[k], N_resonator[k] = np.unravel_index(ΔE_matrix.argmin(), ΔE_matrix.shape)
        else:
            N_fluxonium[k], N_resonator[k] = [-123, -123]
    return N_fluxonium, N_resonator

def get_unique_energy_indices_generalized(E_combined, E_elements):
    tolerance_E_degenerate = 4  # in number of decimals
    E_combined = np.round(E_combined, tolerance_E_degenerate)

    indices_combinations = list(product(*[range(len(e)) for e in E_elements]))

    total_energies = [sum(E_elements[i][indices[i]] for i in range(len(E_elements))) for indices in
                      indices_combinations]

    # Dictionary to store already found index combinations for each energy level
    used_combinations = {E: [] for E in set(E_combined)}

    matched_indices = []
    for E in E_combined:
        differences = np.abs(np.array(total_energies) - E)
        sorted_diff_indices = np.argsort(differences)

        # Find the closest match that has not been used yet
        for idx in sorted_diff_indices:
            if indices_combinations[idx] not in used_combinations[E]:
                matched_indices.append(indices_combinations[idx])
                used_combinations[E].append(indices_combinations[idx])
                break
        else:
            # If all combinations have been used or no match found within tolerance
            matched_indices.append(tuple([-123] * len(E_elements)))

    return reorganize_matched_indices(E_combined, np.array(matched_indices))
def reorganize_matched_indices(E_combined, matched_indices):
    # Identify degenerate energy levels and their indices
    unique, counts = np.unique(E_combined, return_counts=True)
    degenerate_levels = unique[counts > 1]

    for level in degenerate_levels:
        degenerate_indices = np.where(E_combined == level)[0]
        # Extract the matched indices for these degenerate levels
        degenerate_matched_indices = matched_indices[degenerate_indices]

        # Define a custom sort function that prioritizes the first non-zero index
        def sort_key(x):
            for i, xi in enumerate(x):
                if xi > 0:
                    return (i, xi)
            return (len(x), 0)  # Handle the case where all elements are zero

        # Sort based on the custom key
        sorted_indices = sorted(range(len(degenerate_matched_indices)),
                                key=lambda i: sort_key(degenerate_matched_indices[i]))

        # Reassign sorted matched indices back to the original array
        matched_indices[degenerate_indices] = degenerate_matched_indices[sorted_indices]

    return matched_indices

def generate_and_prioritize_energies(E_elements, n_eig):
    # Generate all possible combinations of indices for the given energy levels
    all_indices = list(product(*[range(len(e)) for e in E_elements]))

    # Calculate combined energies for each combination and keep track of indices
    combined_energies_and_indices = [(sum(E_elements[i][indices[i]] for i in range(len(E_elements))), indices) for
                                     indices in all_indices]

    # First, sort by the prioritization of advancement in states
    prioritized_energies_and_indices = sorted(combined_energies_and_indices, key=lambda x: (
    x[0], -sum(i * j for i, j in zip(x[1], range(len(x[1]), 0, -1)))))

    # Extract up to n_eig, ensuring the prioritization
    if len(prioritized_energies_and_indices) > n_eig:
        prioritized_energies_and_indices = prioritized_energies_and_indices[:n_eig]

    E_combined, correctly_reorganized_indices = zip(*prioritized_energies_and_indices)

    return np.array(E_combined), [list(index) for index in correctly_reorganized_indices]

def get_subspace(H, E, threshold):
    # Check diagonal elements of the hamiltonian
    diagonal_elements = np.diag(H)

    # Create a boolean array where True indicates the diagonal element is within the threshold of any vector element
    mask = np.any(np.abs(diagonal_elements[:, None] - E) <= threshold, axis=1)

    # Filter the rows and columns based on the mask
    H_subspace = H[mask][:, mask]

    return H_subspace

def get_ψ_basis(basis_states, H_comp, E_A, E_B, nmax_A, nmax_B, n_eig_comp=4):
    # states = [(0,0),(0,1)]
    # H_comp must be of dimension H_A *otimes* H_B
    E_comp, ψ_comp = diag(H_comp, n_eig_comp, solver='numpy')

    Nf, Nr = sq_get_energy_indices(E_comp, E_A, E_B)
    index_0 = np.intersect1d(np.where(Nf == basis_states[0][0]), np.where(Nr == basis_states[0][0])).tolist()[0]
    index_1 = np.intersect1d(np.where(Nf == basis_states[1][0]), np.where(Nr == basis_states[1][1])).tolist()[0]

    ψ_0 = qt.Qobj(ψ_comp[:, index_0])
    ψ_1 = qt.Qobj(ψ_comp[:, index_1])

    ψ_0.dims = [[nmax_A, nmax_B], [1, 1]]
    ψ_1.dims = [[nmax_A, nmax_B], [1, 1]]

    return ψ_0, ψ_1, index_0, index_1



def compute_combined_eigenstates_2_body(energies, eigenstates):

    eigvals_A, eigvals_B = energies
    eigvecs_A, eigvecs_B = eigenstates

    # List to store the combined eigenstates and eigenvalues
    combined_eigenstates = []
    combined_eigenvalues = []
    for i, eigvec_A in enumerate(eigvecs_A):
        for j, eigvec_B in enumerate(eigvecs_B):
            # Tensor product of eigenstates
            combined_state = qt.tensor(eigvec_A, eigvec_B)
            combined_eigenstates.append(combined_state)
            # Sum of eigenvalues
            combined_energy = eigvals_A[i] + eigvals_B[j]
            combined_eigenvalues.append(combined_energy)

    # Create a list of tuples (energy, state)
    combined = list(zip(combined_eigenvalues, combined_eigenstates))

    # Sort by energy
    combined.sort(key=lambda x: x[0])

    # Unzip into sorted eigenvalues and eigenstates
    sorted_eigenvalues, sorted_eigenstates = zip(*combined)

    # Convert the sorted eigenstates back to Qobj list if needed
    sorted_eigenstates = [qt.Qobj(state) for state in sorted_eigenstates]

    return sorted_eigenvalues, sorted_eigenstates


def compute_combined_eigenstates_3_body(energies, eigenstates, nmax_f, nmax_r):
    eigvals_A, eigvals_B, eigvals_C = energies
    eigvecs_A, eigvecs_B, eigvecs_C = eigenstates

    # List to store the combined eigenstates and eigenvalues
    combined_eigenstates = []
    combined_eigenvalues = []
    for i, eigval_A in enumerate(eigvals_A):
        for j, eigval_B in enumerate(eigvals_B):
            for k, eigval_C in enumerate(eigvals_C):
                # Tensor product of eigenstates
                ψ_A = qt.Qobj(eigvecs_A[:,i])
                ψ_B = qt.Qobj(eigvecs_B[:,j])
                ψ_C = qt.Qobj(eigvecs_C[:,k])

                ψ_A.dims = [[nmax_f, nmax_r], [1, 1]]
                ψ_B.dims = [[nmax_f, nmax_r], [1, 1]]
                ψ_C.dims = [[nmax_f, nmax_r], [1, 1]]

                combined_state = qt.tensor(ψ_A,ψ_B,ψ_C)
                combined_eigenstates.append(combined_state)
                # Sum of eigenvalues
                combined_energy = eigval_A + eigval_B + eigval_C
                combined_eigenvalues.append(combined_energy)

    # Create a list of tuples (energy, state)
    combined = list(zip(combined_eigenvalues, combined_eigenstates))

    # Sort by energy
    combined.sort(key=lambda x: x[0])

    # Unzip into sorted eigenvalues and eigenstates
    sorted_eigenvalues, sorted_eigenstates = zip(*combined)

    # Convert the sorted eigenstates back to Qobj list if needed
    sorted_eigenstates = [qt.Qobj(state) for state in sorted_eigenstates]

    return sorted_eigenvalues, sorted_eigenstates


from itertools import product


def combine_eigenvalues(energy_matrix):
    # Get the shape of the input matrix
    n_systems, n_eig = energy_matrix.shape

    # Generate all possible combinations of indices
    index_combinations = list(product(range(n_eig), repeat=n_systems))

    # Compute the sum of eigenvalues for each combination
    energy_combinations = []
    for indices in index_combinations:
        energy_sum = sum(energy_matrix[i, idx] for i, idx in enumerate(indices))
        energy_combinations.append(energy_sum)

    # Combine energies and indices and sort them
    combined = sorted(zip(energy_combinations, index_combinations))

    # Separate the sorted energies and indices
    sorted_energies = [x[0] for x in combined]
    sorted_indices = [list(x[1]) for x in combined]

    return np.array(sorted_energies), np.array(sorted_indices)


#%% Operators
def internal_coupling_fluxonium_resonator(fluxonium, resonator, Δ, Lq = 25, Lr = 10):
    l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2

    Φ_r = resonator.flux_op(0)
    Φ_f = fluxonium.flux_op(0)

    return qt.tensor(Φ_f, Φ_r) * 2 * Δ / l / 1e-9

def resonator_N_operator(resonator, Z_r, clean=True):
    Φ_nodes, Q_nodes = get_node_variables(resonator)
    Φ_r = Φ_nodes[0] + Φ_nodes[1]
    Q_r = Q_nodes[0] + Q_nodes[1]
    N_r = 1 / 2 / Z_r * (Φ_r ** 2 + Z_r ** 2 * Q_r ** 2)
    if clean:
        try:
            return rank_by_multiples(np.diag(N_r[:len(N_r.__array__()) // 2]))
        except:
            return np.diag(N_r[:len(N_r.__array__()) // 2])
    else:
        return N_r

# Resonator operators
def create(n):
    return np.diag(np.sqrt(np.arange(1, n)), -1)

def annihilate(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1)

def pauli_matrices():
    σ_x = np.array([[0, 1], [1, 0]])
    σ_y = np.array([[0, -1], [1, 0]]) * 1j
    σ_z = np.array([[1, 0], [0, -1]])

    return σ_x ,σ_y ,σ_z

#%% Generic mathematical functions
def diag(H, n_eig=4, out='GHz', real=False, solver='scipy', remove_ground=False, qObj=False):
    H = qt.Qobj(H)

    if solver == 'scipy':
        # efreqs, evecs = sp.sparse.linalg.eigs(H.data, n_eig, which='SR')
        efreqs, evecs = sp.sparse.linalg.eigsh(H.data, n_eig, which='SR')
    elif solver == 'numpy':
        efreqs, evecs = np.linalg.eigh(H.__array__())
        efreqs = efreqs[:n_eig]
        evecs  = evecs [:,:n_eig]
    elif solver == 'Qutip':
        efreqs, evecs = H.eigenstates(eigvals=n_eig, sparse=True)
        if not qObj:
            evecs = np.array([ψ.__array__() for ψ in evecs])[:, :, 0].T
        efreqs_sorted = efreqs
        evecs_sorted = evecs

    if not qObj:
        efreqs_sorted = np.sort(efreqs.real)
        # efreqs_sorted = efreqs_sorted - efreqs_sorted[0]

        sort_arg = np.argsort(efreqs)
        if isinstance(sort_arg, int):
            sort_arg = [sort_arg]
        evecs_sorted = evecs[:, sort_arg]

    if real:
        evecs_sorted = real_eigenvectors(evecs_sorted)
    if out=='GHz':
        efreqs_sorted /= 2 * np.pi * GHz
    elif out =='Hz':
        efreqs_sorted /= 2 * np.pi
    if remove_ground:
        efreqs_sorted -= efreqs_sorted[0]

    return efreqs_sorted, evecs_sorted

def eigs_sorted(w, v):
    """Sorts the eigenvalues in ascending order and the corresponding eigenvectors.

    Input:
    w=array containing the eigenvalues in random order.
    v=array representing the eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i].

    Output:
    w=array containing the eigenvalues in ascending order.
    v=array representing the eigenvectors."""

    ndx = np.argsort(w)  # gives the correct order for the numbers in v, from smallest to biggest.
    return w[ndx], v[:, ndx]

def real_eigenvectors(U):
    '''Fixes the phase of a vector.

    Input:
    U=vector

    Output:
    U= vector without phase.'''

    l = U.shape[0]
    try:
        avgz = np.sum(U[:l // 2, :] * np.abs(U[:l // 2, :]) ** 2, 0)
    except:
        avgz = np.sum(U[:l // 2] * np.abs(U[:l // 2]) ** 2, 0)
    avgz = avgz / np.abs(avgz)
    # U(i,j) = U(i,j) / z(j) for all i,j
    U = U * avgz.conj()
    return U

def get_node_variables(circuit, basis, isolated=False):
    n_modes = circuit.n
    if isolated:
        Φ_normal = [circuit._flux_op_isolated  (i) for i in range(n_modes)]
        Q_normal = [circuit._charge_op_isolated(i) for i in range(n_modes)]
    else:
        Φ_normal = [circuit.flux_op(i, basis=basis) for i in range(n_modes)]
        Q_normal = [circuit.charge_op(i, basis=basis) for i in range(n_modes)]

    Φ_nodes = []
    Q_nodes = []

    for i in range(n_modes):
        Φ_sum = 0
        Q_sum = 0
        for j in range(n_modes):
            Φ_sum += Φ_normal[j] * circuit.S[i, j]
            Q_sum += Q_normal[j] * circuit.R[i, j]

        Φ_nodes.append(Φ_sum)
        Q_nodes.append(Q_sum)

    return Φ_nodes, Q_nodes

#%% Plotting functions
def plot_H_eff_vs_param(H_eff_vs_params, H_eff_0, param_values, param_name, N_f, N_r, threshold=1e-3, n_eig_plot = False, scale='linear'):

    Δ_01 = H_eff_0[1,1]- H_eff_0[0,0]
    H_eff_vs_params /= Δ_01
    H_eff_0 /= Δ_01

    if n_eig_plot == False:
        n_eig_plot = len(H_eff_0)

    colors = figs.generate_colors_from_colormap(20, 'tab20')
    label_color_dict = {}
    titles = ['Couplings', 'Renormalizations']
    y_labels = [r'$g/\Delta_{01}$', r'$E/\Delta_{01}$']
    fig, [ax1, ax2] = plt.subplots(ncols=2, dpi=150, figsize=[8, 4.5])
    k_ij = 0; k_ii = 0
    for i in range(n_eig_plot):
        for j in range(i, n_eig_plot):
            if i != j and np.any(np.abs(H_eff_vs_params[:, i, j]) > threshold):
                label = get_state_label(N_f, N_r, i, j)
                color, label_color_dict, _ = get_or_assign_color(label, colors, label_color_dict)
                if scale == 'log':
                    ax1.plot(param_values[k_ij:], np.abs(H_eff_vs_params[k_ij:, i, j]), markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                else:
                    ax1.plot(param_values[k_ij:], H_eff_vs_params[k_ij:, i, j], markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                k_ij += 1

            elif i == j and np.any(np.abs(H_eff_vs_params[:, i, j] - H_eff_0[i, j]) > threshold):
                label =  get_state_label(N_f, N_r, i, j)
                color, label_color_dict, _ = get_or_assign_color(label, colors, label_color_dict)
                if scale == 'log':
                    ax2.plot(param_values[k_ii:], np.abs(H_eff_vs_params[k_ii:, i, j] - H_eff_0[i, j]), markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                else:
                    ax2.plot(param_values[k_ii:], H_eff_vs_params[k_ii:, i, j] - H_eff_0[i, j], markersize=4, label=label, color=color, marker='o', markerfacecolor='w')
                k_ii += 1
    if scale == 'log':
        ax2.plot(param_values, np.abs(1-np.abs(H_eff_vs_params[:, 1, 1] - H_eff_vs_params[:, 0, 0])), ':k',label=r'$1-\Delta_{01}('+param_name+')/\Delta_{01}$')
    else:
        ax2.plot(param_values, 1-np.abs(H_eff_vs_params[:, 1, 1] - H_eff_vs_params[:, 0, 0]), ':k',label=r'$1-\Delta_{01}('+param_name+')/\Delta_{01}$')

    for i, ax in enumerate([ax1, ax2]):
        ax.set_xlabel('$'+param_name+'$')
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_title(titles[i])
        ax.set_ylabel(y_labels[i])
        ax.legend()

    fig.tight_layout()
    fig.show()
    return fig, ax1, ax2

#%% Generic labeling and sorting functions
def print_charge_transformation(circuit):
    for i in range(circuit.R.shape[1]):
        normalized = circuit.R[:, i] / np.abs(circuit.R[:, i]).max()
        formatted_vector = [f"{num:5.2f}" for num in normalized]
        vector_str = ' '.join(formatted_vector)
        print(f'q_{i + 1} = [{vector_str}]')

def print_flux_transformation(circuit):
    for i in range(circuit.S.shape[1]):
        normalized = circuit.S[:, i] / np.abs(circuit.S[:, i]).max()
        formatted_vector = [f"{num:5.2f}" for num in normalized]
        vector_str = ' '.join(formatted_vector)
        print(f'Φ_{i + 1} = [{vector_str}]')

def get_state_label(N_f, N_r, i, j, return_numeric_indices=False):
    if i == j:
        label = f'({N_f[i]}f,{N_r[i]}r)'
    else:
        label = f'({N_f[i]}f,{N_r[i]}r) - ({N_f[j]}f,{N_r[j]}r)'

    if return_numeric_indices:
        return label, N_f[i], N_f[j], N_r[i], N_r[j]
    else:
        return label

def find_indices(data):
    result = []
    current_string = data[0]
    start_index = 0

    for i, s in enumerate(data[1:], 1):  # start from index 1
        if s != current_string:
            result.append((current_string, [start_index, i - 1]))
            current_string = s
            start_index = i

    result.append((current_string, [start_index, len(data) - 1]))  # for the last string

    return result

def get_or_assign_color(s, colors, color_dict):
    # Check if string is already associated with a color
    if s in color_dict:
        newly_assigled=False
        return color_dict[s], color_dict, newly_assigled

    # If not, associate with the next available color
    for color in colors:
        if color not in color_dict.values():  # Check if this color is not already used
            newly_assigled = True
            color_dict[s] = color
            return color, color_dict, newly_assigled
    # If all colors are already used, you might want to handle this case.
    # For now, it returns None and the unchanged dictionary.
    print('I am out of colorssssssss bro')
def rank_by_multiples(arr, tol = 0.1):
    # Receives an array of repeting real values and outputs an array of integers that identify the smallest element and
    # The rest of the elements as multiples of the delta between the smallest and next smallest element.
    # e.g. in=[2.41, 1.23, 2.42, 3.63] out=[1, 0, 1, 2]
    # aka puting labels to usorted energy levels.

    # Identify the smallest value
    min_value = np.min(arr)

    # Subtract the smallest value from the entire array
    diff_array = arr - min_value

    # Identify the next smallest positive value in the subtracted array
    unit = np.min([val for val in diff_array if val > tol])

    return np.array( np.round(diff_array / unit), dtype='int')

def decomposition_in_pauli_2x2(A, print=False):
    '''Performs Pauli decomposition of a 2x2 matrix.

    Input:
    A= matrix to decompose.

    Output:
    P= 4 coefficients such that A = P[0]*I + P[1]*σx + P[2]*σy + P[3]*σz'''

    # Pauli matrices.
    I = np.eye(2)
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [I, σx, σy, σz]  # array containing the matrices.

    P = np.zeros(4)  # array to store our results.
    # Loop to obtain each coefficient.
    for i in range(4):
        P[i] = 0.5 * np.trace(s[i].T.conjugate() @ A)

    return P

def decomposition_in_pauli_4x4(A,  print_pretty=True):
    '''Performs Pauli decomposition of a 4x4 matrix.

    Input:
    A= matrix to decompose.
    rd= number of decimals to use when rounding a number.

    Output:
    P= coefficients such that A = ΣP[i,j]σ_iσ_j where i,j=0, 1, 2, 3. '''

    i = np.eye(2)  # σ_0
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [i, σx, σy, σz]  # array containing the matrices.
    labels = ['I', 'σx', 'σy', 'σz']  # useful to print the result.

    P = np.zeros((4, 4), dtype=float)  # array to store our results.
    # Loop to obtain each coefficient.
    for i in range(4):
        for j in range(4):
            label = labels[i] + ' \U00002A02' + labels[j]
            S = np.kron(s[i], s[j])  # S_ij=σ_i /otimes σ_j.
            P[i, j] = 0.25 * (np.dot(S.T.conjugate(), A)).trace() # P[i,j]=(1/4)tr(S_ij^t*A)
            if np.abs(P[i, j])>1e-13 and print_pretty == True:
                print(" %s\t*\t %s " % (P[i, j], label))

    return P

def decomposition_in_pauli_4x4_qubit_resonator(A,  print_pretty=True):
    '''Performs Pauli decomposition of a 4x4 matrix.

    Input:
    A= matrix to decompose.
    rd= number of decimals to use when rounding a number.

    Output:
    P= coefficients such that A = ΣP[i,j]σ_iσ_j where i,j=0, 1, 2, 3. '''

    i = np.eye(2)  # σ_0
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [i, σx, σy, σz]  # array containing the matrices.

    def create(n):
        return np.diag(np.sqrt(np.arange(1, n)), -1)

    def annihilate(n):
        return np.diag(np.sqrt(np.arange(1, n)), 1)
    n = 2

    a = create(n)
    a_dagger = annihilate(n)
    r = [i, a_dagger+a, 1j*(a_dagger-a), a*a_dagger ]

    labels = ['I', 'σx', 'σy', 'σz']  # useful to print the result.
    labels_r = ['I', 'a_d+a', 'i(a_d-a)', 'a_d a']  # useful to print the result.

    P = np.zeros((4, 4), dtype=complex)  # array to store our results.
    # Loop to obtain each coefficient.
    for i in range(4):
        for j in range(4):
            label = labels[i] + ' \U00002A02' + labels_r[j]
            S = np.kron(s[i], r[j])  # S_ij=σ_i /otimes σ_j.
            P[i, j] = 0.25 * (np.dot(S.T.conjugate(), A)).trace() # P[i,j]=(1/4)tr(S_ij^t*A)
            if P[i, j] != 0.0 and print_pretty == True:
                print(" %s\t*\t %s " % (P[i, j], label))

    return P

import numpy as np

def decomposition_in_pauli_8x8(A, rd, print=True):
    '''Performs Pauli decomposition of an 8x8 matrix.

    Input:
    A= matrix to decompose.
    rd= number of decimals to use when rounding a number.

    Output:
    P= coefficients such that A = ΣP[i,j,k]σ_iσ_jσ_k where i,j,k=0, 1, 2, 3. '''

    i = np.eye(2)  # σ_0
    σx = np.array([[0, 1], [1, 0]])
    σy = np.array([[0, -1j], [1j, 0]])
    σz = np.array([[1, 0], [0, -1]])
    s = [i, σx, σy, σz]  # array containing the matrices.
    labels = ['I', 'σx', 'σy', 'σz']  # useful to print the result.

    P = np.zeros((4, 4, 4), dtype=complex)  # array to store our results.
    # Loop to obtain each coefficient.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                label = labels[i] + ' \U00002A02 ' + labels[j] + ' \U00002A02 ' + labels[k]
                S = np.kron(np.kron(s[i], s[j]), s[k])  # S_ijk=σ_i ⊗ σ_j ⊗ σ_k.
                P[i, j, k] = np.round(0.125 * (np.dot(S.T.conjugate(), A)).trace(), rd)  # P[i,j,k]=(1/8)tr(S_ijk^t*A)
                if P[i, j, k] != 0.0 and print:
                    print(" %s\t*\t %s " % (P[i, j, k], label))

    return P


#%% Generic functions
def expand_list_with_array(input_list):
    # Find the array and its position in the list
    array_element = None
    array_index = None
    for i, element in enumerate(input_list):
        if isinstance(element, np.ndarray):
            array_element = element
            array_index = i
            break

    # If no array is found, return the original list wrapped in another list
    if array_element is None:
        return [input_list]

    # Create the list of lists
    expanded_list = []
    for value in array_element:
        new_list = input_list.copy()
        new_list[array_index] = value
        expanded_list.append(new_list)

    return expanded_list


def find_close_indices(E_0_ψ_0, E_0):
    result_indices = []
    for value in E_0_ψ_0:

        # Check if any element in E_0 is within the tolerance range
        result_indices.append(np.where(np.abs(E_0-value)<=1e-8)[0][0])

    return result_indices

#%% Truncation convergence
def truncation_convergence(circuit, n_eig, trunc_nums=False, threshold=1e-2, refine=True, plot=True):
    '''
    This function tests the convergence of set_trunc_nums.
    It increases the truncation numbers until the convergence condition is met.
    The convergence condition is that the max difference between the spectrum calculated using some truncation numbers,
    versus those truncation numbers + 1, must be below a certain "threshold" (given by user or set as 0.1%).

    If refine is True then it reduces the truncation number of each mode iteratively to see if the convergence condition
    is still met.

    TO DO: This function is not very optimal to obtain the spectrum vs phi, it diagonalizes too many times for already converged results.
    '''

    if not trunc_nums:
        trunc_nums = [2] * circuit.n

    ΔE = 1
    E = []
    trunc_nums_list = []
    while ΔE > threshold:
        trunc_nums = [n + 1 for n in trunc_nums]
        circuit.set_trunc_nums(trunc_nums)
        E.append(circuit.diag(n_eig)[0])
        trunc_nums_list.append(trunc_nums.copy())
        if len(E) == 1:
            continue

        ΔE = np.abs((E[-1] - E[-2]) / E[-2]).max()

    E_conv = E[-1].copy()

    if refine:
        modes_to_refine = np.array([True] * circuit.n)
        while np.any(modes_to_refine) == True:
            for mode in range(circuit.n):
                if modes_to_refine[mode] == False:
                    continue
                trunc_nums[mode] -= 1
                circuit.set_trunc_nums(trunc_nums)
                E.append(circuit.diag(n_eig)[0])
                trunc_nums_list.append(trunc_nums.copy())
                ΔE = np.abs((E[-1] - E_conv) / E_conv).max()
                if ΔE < threshold:
                    if trunc_nums[mode] == 1:
                        modes_to_refine[mode] = False
                    continue
                else:
                    modes_to_refine[mode] = False
                    trunc_nums[mode] += 1
                    del E[-1], trunc_nums_list[-1]
                    ΔE = np.abs((E[-1] - E_conv) / E_conv).max()

    if plot:
        print(trunc_nums, E[-1], ΔE)
        fig, ax = plt.subplots()
        ax.plot(np.abs(E_conv - np.array(E[1:])) / E_conv)
        ax.set_yscale('log')
        ax.set_ylabel(r'$(E-E_{conv}) / E_{conv}$')
        labels_trunc_nums = [str(l) for l in trunc_nums_list]
        ax.set_xlabel('trunc_nums')
        ax.set_xticks(np.arange(len(E))[::2], labels_trunc_nums[::2], rotation=-30)
        fig.show()

    circuit.set_trunc_nums(trunc_nums)
    return circuit

#%% Elephants' graveyard

#
# def hamiltonian_qubit_C_qubit_C_qubit(nmax_r, nmax_f, Cc, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, periodic=True, only_outer=False, return_Ψ_nonint=False, n_eig_Ψ_nonint=4):
#     fF = 1e-15
#     C_R = C / 2
#     C_C = Cc
#     C_F = C / 2 + Csh + CJ
#     if periodic == True:
#         C_mat = np.array([[C_R + C_C / 2, 0, -C_C / 2, 0, -C_C / 2, 0],
#                           [0, C_F + C_C / 2, 0, +C_C / 2, 0, +C_C / 2],
#                           [-C_C / 2, 0, C_R + C_C / 2, 0, -C_C / 2, 0],
#                           [0, +C_C / 2, 0, C_F + C_C / 2, 0, +C_C / 2],
#                           [-C_C / 2, 0, -C_C / 2, 0, C_R + C_C / 2, 0],
#                           [0, +C_C / 2, 0, +C_C / 2, 0, C_F + C_C / 2]])
#     else:
#         C_mat = np.array([[C_R + C_C / 2, 0             , -C_C / 2      , +C_C / 2      , 0             ,        0      ],
#                           [0            , C_F + C_C / 2 , -C_C / 2      , +C_C / 2      , 0             , 0             ],
#                           [-C_C / 2     , +C_C / 2      , C_R + C_C / 2 , 0             , -C_C / 2      , +C_C / 2      ],
#                           [-C_C / 2     , +C_C / 2      , 0             , C_F + C_C / 2 , -C_C / 2      , +C_C / 2      ],
#                           [0            , 0             , -C_C / 2      , +C_C / 2      , C_R + C_C / 2 , 0             ],
#                           [0            , 0             , -C_C / 2      , +C_C / 2      , 0             , C_F + C_C / 2 ]])
#
#     C_inv = np.linalg.inv(C_mat)
#
#     resonator_1 = sq_resonator(Lq=Lq, Lr=Lr, Δ=Δ, nmax_r=nmax_r, C_R_eff=C_inv[0, 0] ** -1)
#     fluxonium_1 = sq_fluxonium(Lq=Lq, Lr=Lr, Δ=Δ, nmax_f=nmax_f, C_F_eff=C_inv[1, 1] ** -1)
#     resonator_2 = sq_resonator(Lq=Lq, Lr=Lr, Δ=Δ, nmax_r=nmax_r, C_R_eff=C_inv[2, 2] ** -1)
#     fluxonium_2 = sq_fluxonium(Lq=Lq, Lr=Lr, Δ=Δ, nmax_f=nmax_f, C_F_eff=C_inv[3, 3] ** -1)
#     resonator_3 = sq_resonator(Lq=Lq, Lr=Lr, Δ=Δ, nmax_r=nmax_r, C_R_eff=C_inv[4, 4] ** -1)
#     fluxonium_3 = sq_fluxonium(Lq=Lq, Lr=Lr, Δ=Δ, nmax_f=nmax_f, C_F_eff=C_inv[5, 5] ** -1)
#
#     if return_Ψ_nonint:
#         H_qubit_1, Ψ_q_0_1, E_q_0_1 = hamiltonian_qubit(fluxonium_1, resonator_1, Δ, return_Ψ_nonint=return_Ψ_nonint)
#         H_qubit_2, Ψ_q_0_2, E_q_0_2 = hamiltonian_qubit(fluxonium_2, resonator_2, Δ, return_Ψ_nonint=return_Ψ_nonint)
#         H_qubit_3, Ψ_q_0_3, E_q_0_3 = hamiltonian_qubit(fluxonium_3, resonator_3, Δ, return_Ψ_nonint=return_Ψ_nonint)
#         # Nq_Nq_Nq = generate_and_prioritize_energies([E_q_0_1, E_q_0_2, E_q_0_3], n_eig_Ψ_nonint)[1]
#         Nq_Nq_Nq = np.array([[0, 0, 0],
#                              [1, 0, 0],
#                              [0, 1, 0],
#                              [1, 1, 0],
#                              [0, 0, 1],
#                              [0, 1, 1],
#                              [1, 0, 1],
#                              [1, 1, 1]])
#         Ψ_0 = [qt.tensor([Ψ_q_0_1[Nq_Nq_Nq_i[0]], Ψ_q_0_2[Nq_Nq_Nq_i[1]],Ψ_q_0_3[Nq_Nq_Nq_i[2]]  ]) for Nq_Nq_Nq_i in Nq_Nq_Nq]
#     else:
#         H_qubit_1 = hamiltonian_qubit(fluxonium_1, resonator_1, Δ )
#         H_qubit_2 = hamiltonian_qubit(fluxonium_2, resonator_2, Δ )
#         H_qubit_3 = hamiltonian_qubit(fluxonium_3, resonator_3, Δ )
#
#     I_r = qt.identity(nmax_r)
#     I_f = qt.identity(nmax_f)
#     I_qubit = qt.identity(H_qubit_1.dims[0])
#
#     q_r_1 = qt.tensor(I_f, resonator_1.charge_op(0))
#     q_f_1 = qt.tensor(fluxonium_1.charge_op(0), I_r)
#     q_r_2 = qt.tensor(I_f, resonator_2.charge_op(0))
#     q_f_2 = qt.tensor(fluxonium_2.charge_op(0), I_r)
#     q_r_3 = qt.tensor(I_f, resonator_3.charge_op(0))
#     q_f_3 = qt.tensor(fluxonium_3.charge_op(0), I_r)
#
#     if only_outer:
#         H_0 = (  qt.tensor(H_qubit_1, I_qubit, I_qubit)
#                + qt.tensor(I_qubit, I_qubit, H_qubit_3) )
#     else:
#         H_0 = (  qt.tensor(H_qubit_1, I_qubit, I_qubit)
#                + qt.tensor(I_qubit, H_qubit_2, I_qubit)
#                + qt.tensor(I_qubit, I_qubit, H_qubit_3) )
#
#     if Cc == 0:
#         if return_Ψ_nonint:
#             return H_0, Ψ_0
#         else:
#             return H_0
#
#     # I should do this propperly by multiplying matrices...
#     # Q_vec = [q_r, q_f, q_f, q_r, q_f, q_r]
#     # H_coupling = 0
#     # for i in range(6):
#     #     for j in range(6):
#     #         H_coupling +=
#
#     if periodic == True:
#         C_RR = C_inv[0, 2] ** -1 * fF
#         C_FF = C_inv[1, 3] ** -1 * fF
#         #maybe i can kill fluxonium fluxonium...
#         H_coupling = qt.tensor(q_r_1, q_r_2, I_qubit) / C_RR + qt.tensor(q_f_1, q_f_2, I_qubit) / C_FF + \
#                      qt.tensor(I_qubit, q_r_2, q_r_3) / C_RR + qt.tensor(I_qubit, q_f_2, q_f_3) / C_FF + \
#                      qt.tensor(q_r_1, I_qubit, q_r_3) / C_RR + qt.tensor(q_f_1, I_qubit, q_f_3) / C_FF
#     else:
#         C_RR = C_inv[0, 2] ** -1 * fF
#         C_FF = C_inv[1, 3] ** -1 * fF
#         C_RF = C_inv[0, 3] ** -1 * fF
#         H_coupling = (qt.tensor(q_r_1, q_r_2, I_qubit) / (C_inv[0, 2] ** -1 * fF) + qt.tensor(q_f_1, q_f_2, I_qubit) / (C_inv[1, 3] ** -1 * fF )+
#                       qt.tensor(q_r_1, q_f_2, I_qubit) / (C_inv[0, 3] ** -1 * fF) + qt.tensor(q_f_1, q_r_2, I_qubit) / (C_inv[0, 3] ** -1 * fF )+
#                       qt.tensor(I_qubit, q_r_2, q_r_3) / (C_inv[2, 4] ** -1 * fF) + qt.tensor(I_qubit, q_f_2, q_f_3) / (C_inv[3, 4] ** -1 * fF )+
#                       qt.tensor(I_qubit, q_r_2, q_f_3) / (C_inv[2, 5] ** -1 * fF) + qt.tensor(I_qubit, q_f_2, q_r_3) / (C_inv[2, 5] ** -1 * fF))
#
#     H = H_0 + H_coupling
#
#     if return_Ψ_nonint:
#         return H, Ψ_0
#     else:
#         return H

# def H_eff_SWT(H_0, H, n_eig, out='None', real=True, remove_ground=False, solver='scipy', return_transformation=False,
#               sparse=False):
#     if sparse == False:
#         ψ_0 = diag(H_0, n_eig, real=real, solver=solver)[1]
#         E, ψ = diag(H, n_eig, real=real, solver=solver)
#         Q = ψ_0.T.conj() @ ψ
#     else:
#         ψ_0 = H_0.eigenstates(sparse=True, eigvals=n_eig, phase_fix=0)[1]
#         E, ψ = H.eigenstates(sparse=True, eigvals=n_eig, phase_fix=0)
#         Q = np.zeros((n_eig, n_eig), dtype=complex)
#         for i in range(n_eig):
#             for j in range(n_eig):
#                 Q[i, j] = (ψ_0[i].dag() * ψ[j]).data[0, 0]
#
#     U, s, Vh = np.linalg.svd(Q)
#     A = U @ Vh
#
#     H_eff = A @ np.diag(E) @ A.T.conj()
#
#     if out == 'GHz':
#         H_eff /= GHz * 2 * np.pi
#
#     if remove_ground:
#         H_eff -= H_eff[0, 0] * np.eye(len(H_eff))
#
#     if real:
#         if np.allclose(np.imag(H_eff), 0):
#             H_eff = np.real(H_eff)
#
#     if return_transformation:
#         return H_eff, A
#     else:
#         return H_eff

# def hamiltonian_frc_qubit(qubit, fluxonium, resonator, Δ, Lq = 25, Lr = 10, factor=1):
#     l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#
#     H_f = fluxonium.hamiltonian()
#     H_r = resonator.hamiltonian()
#
#     I_f = qt.identity(H_f.shape[0])
#     I_r = qt.identity(H_r.shape[0])
#
#     Φ, Q = get_node_variables(qubit,'FC', isolated=True)
#
#     Φ_f = Φ[1]-Φ[0]
#     Φ_r = Φ[0]+Φ[1]
#
#     H = qt.tensor(H_f, I_r) + qt.tensor(I_f, H_r) + factor * qt.tensor(Φ_f, Φ_r) * 2 * Δ / l / 1e-9
#     return H

# def KIT_qubit_triangle(C = 15, CJ = 3, Csh= 15 , Lq = 25, Lr = 10, Δ = 0.1, EJ = 10.0, φ_ext=0.5):
#
#     R1 = Lq/2-Δ
#     R2 = Lq/2+Δ
#     R3 = Lr
#
#     Rp = R1*R2 + R1*R3 + R2*R3
#     Ra = Rp/R1
#     Rb = Rp/R2
#     Rc = Rp/R3
#
#     # Initialize loop(s)
#     loop = sq.Loop(φ_ext)
#     loop_fictitious = sq.Loop(φ_ext)
#
#     # Circuit components
#     C_01 = sq.Capacitor(C,       'fF')
#     C_02 = sq.Capacitor(C,       'fF')
#     C_12 = sq.Capacitor(CJ+Csh,  'fF')
#     L_01 = sq.Inductor(Rb, 'nH',  loops=[loop_fictitious])
#     L_02 = sq.Inductor(Ra, 'nH',  loops=[loop_fictitious])
#     L_12 = sq.Inductor(Rc, 'nH',  loops=[loop_fictitious, loop])
#     JJ_12= sq.Junction(EJ,'GHz',  loops=[loop])
#
#     elements = {
#         (0, 1): [C_01, L_01],
#         (0, 2): [C_02, L_02],
#         (1, 2): [C_12, JJ_12, L_12],
#     }
#
#     # Create and return the circuit
#     return sq.Circuit(elements)


#% Spin-boson hamiltonians
# def spin_boson_qubit(nmax_r, nmax_f, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, EJ=10, N_R=1, interaction_prefactor=1):
#     nH = 1e-9
#     resonator = sq_resonator(C=C, CJ=CJ, Csh=Csh, Lq=Lq, Lr=Lr, Δ=Δ, EJ=EJ, trunc_res=nmax_r)
#     resonator_0 = sq_resonator(C=C, CJ=CJ, Csh=Csh, Lq=Lq, Lr=Lr, Δ=0, EJ=EJ, trunc_res=nmax_r)
#     fluxonium = sq_fluxonium(C=C, CJ=CJ, Csh=Csh, Lq=Lq, Lr=Lr, Δ=Δ, EJ=EJ, trunc_flux=nmax_f)
#     fluxonium_0 = sq_fluxonium(C=C, CJ=CJ, Csh=Csh, Lq=Lq, Lr=Lr, Δ=0, EJ=EJ, trunc_flux=nmax_f)
#     fluxonium.diag(2)
#     resonator_0.diag(2)
#     resonator.diag(2)
#     fluxonium_0.diag(2)
#
#     ω_f = fluxonium.efreqs[1] - fluxonium.efreqs[0]  # Fluxonium frequency
#     ω_r = resonator.efreqs[1] - resonator.efreqs[0]  # Resonator frequency
#
#     # Φ_f = fluxonium.flux_op(0)[0, 1]
#     # Φ_r = resonator.flux_op(0)[0, 1]
#     Φ_f = fluxonium.flux_op(0, basis='eig')[0, 1]
#     Φ_r = resonator.flux_op(0, basis='eig')[0, 1]
#     # Φ_f = H_eff_SWT(fluxonium_0.hamiltonian(), fluxonium.flux_op(0), 2, out=None)[0, 1]
#     # Φ_r = H_eff_SWT(resonator_0.hamiltonian(), resonator.flux_op(0), 2, out=None)[0, 1]
#
#     # Pauli matrices for the fluxonium
#     sigma_x = np.array([[0, 1], [1, 0]])
#     sigma_z = np.array([[1, 0], [0, -1]])
#
#     # Creation and annihilation operators for the resonator
#     a = create(N_R)
#     a_dagger = annihilate(N_R)
#
#     # Identity matrices
#     I_resonator = np.eye(N_R)
#     I_fluxonium = np.eye(2)
#
#     # Isolated Hamiltonians
#     H_fluxonium = ω_f / 2 * sigma_z
#     H_resonator = ω_r * np.dot(a_dagger, a)
#
#     # Construct the system Hamiltonian with tensor products
#     H_0_qubit = np.kron(H_resonator, I_fluxonium) + np.kron(I_resonator, H_fluxonium)
#
#     l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#     g_Φ = 2 * Δ / (l * nH) * Φ_f * Φ_r / 2 / np.pi / GHz
#     H_interaction = g_Φ * np.kron(a + a_dagger, sigma_x)
#
#     return H_0_qubit + H_interaction * interaction_prefactor
#
#
# def spin_boson_qubit_C_qubit(nmax_r, nmax_f, Cc, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1, N_R=1):
#     fF = 1e-15
#     l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#     C_R = C / 2
#     C_C = Cc
#     C_F = C / 2 + Csh + CJ
#
#     C_mat = np.array([[C_R + C_C / 2, 0, -C_C / 2, 0],
#                       [0, C_F + C_C / 2, 0, -C_C / 2],
#                       [-C_C / 2, 0, C_R + C_C / 2, 0],
#                       [0, -C_C / 2, 0, C_F + C_C / 2]])
#
#     C_inv = np.linalg.inv(C_mat)
#     C_R_tilde = C_inv[0, 0] ** -1
#     C_F_tilde = C_inv[1, 1] ** -1
#     C_RR = C_inv[0, 2] ** -1
#     C_FF = C_inv[1, 3] ** -1
#
#     resonator = sq_resonator(C_R_eff=C_R_tilde, Lq=Lq, Lr=Lr, Δ=Δ, trunc_res=nmax_r)
#     fluxonium = sq_fluxonium(C_F_eff=C_F_tilde, Lq=Lq, Lr=Lr, Δ=Δ, trunc_flux=nmax_f)
#     fluxonium.diag(2)
#     resonator.diag(2)
#
#     ω_f = fluxonium.efreqs[1] - fluxonium.efreqs[0]  # Fluxonium frequency
#     ω_r = resonator.efreqs[1] - resonator.efreqs[0]  # Resonator frequency
#
#     Φ_f = fluxonium.flux_op(0)[0, 1]
#     Φ_r = resonator.flux_op(0)[0, 1]
#     Q_F = fluxonium.charge_op(0)[0, 1]
#     Q_R = resonator.charge_op(0)[0, 1]
#
#     # Pauli matrices for the fluxonium
#     sigma_x = np.array([[0, 1], [1, 0]])
#     sigma_y = np.array([[0, -1], [1, 0]]) * 1j
#     sigma_z = np.array([[1, 0], [0, -1]])
#
#     sigma_p = np.array([[0, 1], [0, 0]])
#     sigma_m = np.array([[0, 0], [1, 0]])
#
#     # Creation and annihilation operators for the resonator
#     a = create(N_R)
#     a_dagger = annihilate(N_R)
#
#     # Identity matrices
#     I_resonator = np.eye(N_R)
#     I_fluxonium = np.eye(2)
#
#     # Isolated Hamiltonians
#     H_fluxonium = ω_f / 2 * sigma_z
#     H_resonator = ω_r * np.dot(a_dagger, a)
#
#     # Construct the system Hamiltonian with tensor products
#     H_0_qubit = np.kron(H_resonator, I_fluxonium) + np.kron(I_resonator, H_fluxonium)
#
#     g_Φ = 2 * Δ / (l * nH) * Φ_f * Φ_r / 2 / np.pi / GHz
#     H_interaction = g_Φ * np.kron(a + a_dagger, sigma_x)
#
#     # Circuit Hamiltonian
#     H_circuit = H_0_qubit + H_interaction
#     I_circuit = np.eye(len(H_circuit))
#     H_0 = np.kron(H_circuit, I_circuit) + np.kron(I_circuit, H_circuit)
#
#     # Pali and fock operators with the circuit's hamiltonian shape
#     a_sys = np.kron(a, I_fluxonium)
#     a_dagger_sys = np.kron(a_dagger, I_fluxonium)
#     sigma_y_sys = np.kron(I_resonator, sigma_y)
#
#     H_coupling = Q_R ** 2 / (C_RR * fF) * np.kron(a_dagger_sys - a_sys, a_dagger_sys - a_sys) + Q_F ** 2 / (
#             C_FF * fF) * np.kron(sigma_y_sys, sigma_y_sys)
#     H_coupling /= 2 * np.pi * GHz
#
#     return H_0 + H_coupling

#
# # %% KIT's qubit internal coupling perturbation theory with fluxonium + resonator decomposition
# def H_eff_p1_fluxonium_resonator(fluxonium_0, fluxonium, resonator_0, resonator, N_f, N_r, Δ, Lq = 25, Lr = 10):
#     '''
#     DANGER, N_f and N_r shold be the enery levels of the non interacting resonator and fluxonium.
#     Now its fine because they are the same as the interacting ones, but careful.
#     '''
#     l   = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#
#     ψ_0_f = np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium_0._evecs]).T
#     ψ_0_r = np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator_0._evecs]).T
#     Φ_f =  fluxonium.flux_op(0).__array__()
#     Φ_r =  resonator.flux_op(0).__array__()
#
#     n_eig = ψ_0_f.shape[1]
#
#     H_eff_p1 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.
#
#     for i in range(n_eig):
#         for j in range(n_eig):
#             H_eff_1_f_i_j = np.abs(ψ_0_f[:, N_f[i]].conj().T @ Φ_f @ ψ_0_f[:, N_f[j]])
#             H_eff_1_r_i_j = np.abs(ψ_0_r[:, N_r[i]].conj().T @ Φ_r @ ψ_0_r[:, N_r[j]])
#             H_eff_p1[i,j] = H_eff_1_f_i_j*H_eff_1_r_i_j
#
#     return H_eff_p1 * 2 * Δ / l / 1e-9  / (2 * np.pi * GHz)
#
#
# def H_eff_p2_fluxonium_resonator(fluxonium_0, fluxonium, resonator_0, resonator, N_f, N_r, Δ, Lq = 25, Lr = 10):
#     '''
#     DANGER, N_f and N_r shold be the enery levels of the non interacting resonator and fluxonium.
#     Now its fine because they are the same as the interacting ones, but careful.
#     '''
#     l   = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#
#     ψ_0_f = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium_0._evecs]).T)
#     ψ_f   = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium  ._evecs]).T)
#
#     ψ_0_r = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator_0._evecs]).T)
#     ψ_r   = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator  ._evecs]).T)
#
#     Φ_f =  fluxonium.flux_op(0).__array__() / (2 * np.pi * GHz)
#     Φ_r =  resonator.flux_op(0).__array__() / (2 * np.pi * GHz)
#
#     n_eig = ψ_0_f.shape[1]
#     H_eff_p2 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.
#
#     for i in range(n_eig):
#         E_0_i = fluxonium_0.efreqs[N_f[i]] + resonator_0.efreqs[N_r[i]]
#         for j in range(n_eig):
#             E_0_j = fluxonium_0.efreqs[N_f[j]] + resonator_0.efreqs[N_r[j]]
#             for k in range(n_eig):
#                 E_k = fluxonium.efreqs[N_f[k]] + resonator.efreqs[N_r[k]]
#                 H_eff_2_f_ijk = ψ_0_f[:, N_f[i]].conj().T @ Φ_f @ ψ_f  [:, N_f[k]] *  \
#                                   ψ_f[:, N_f[k]].conj().T @ Φ_f @ ψ_0_f[:, N_f[j]]
#
#                 H_eff_2_r_ijk = ψ_0_r[:, N_r[i]].conj().T @ Φ_r @ ψ_r  [:, N_r[k]] *  \
#                                   ψ_r[:, N_r[k]].conj().T @ Φ_r @ ψ_0_r[:, N_r[j]]
#
#                 coef = 1 / (E_0_i-E_k) + 1 / (E_0_j-E_k)
#                 H_eff_p2[i,j] += coef * np.abs(H_eff_2_f_ijk * H_eff_2_r_ijk) #/  GHz #/ 2 / np.pi
#
#     return Δ**2/2  * H_eff_p2 * 2 * Δ / l / 1e-9
#


#%
# def H_eff_SWT_circ(circuit_0, circuit, return_transformation = False):
#     ψb0 = real_eigenvectors(np.array([ψb0_i.__array__()[:,0] for ψb0_i in circuit_0._evecs]).T)
#     ψb  = real_eigenvectors(np.array([ψb0_i.__array__()[:,0] for ψb0_i in circuit  ._evecs]).T)
#     E = circuit.efreqs
#
#     Q = ψb0.T.conj() @ ψb
#     U, s, Vh = np.linalg.svd(Q)
#     A = U @ Vh
#     H_eff = A @ np.diag(E) @ A.T.conj()
#
#     if return_transformation:
#         return H_eff, A
#     else:
#         return H_eff

# def H_eff_p1_circ(circ_0, circ, out='GHz', real=True, remove_ground = False):
#     ψ_0 = np.array([ψ_i.__array__()[:, 0] for ψ_i in circ_0._evecs]).T
#
#     if real:
#         ψ_0 = real_eigenvectors(ψ_0)
#
#     H = circ.hamiltonian().__array__()
#     H_eff = ψ_0.conj().T @ H @ ψ_0
#
#     if out == 'GHz':
#         H_eff /= GHz * 2 * np.pi
#
#     if remove_ground:
#         H_eff -=  H_eff[0,0]*np.eye(len(H_eff))
#
#     if real:
#         if np.allclose(np.imag(H_eff),0):
#             H_eff = np.real(H_eff)
#
#     return H_eff

# def H_eff_p2_circ(circ_0, circ):
#     ψ_0 = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in circ_0._evecs]).T)
#     ψ   = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in circ.  _evecs]).T)
#     E_0 = circ_0._efreqs
#     E   = circ  ._efreqs
#     H_0 = circ_0.hamiltonian().__array__()
#     H   = circ  .hamiltonian().__array__()
#     V   = H-H_0
#
#     # H_eff_1 = ψ_0.conj().T @ H @ ψ_0
#
#     n_eig = ψ_0.shape[1]
#     H_eff_2 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.
#
#     for i in range(n_eig):
#         for j in range(n_eig):
#             H_eff_2[i, j] = 1 / 2 * sum(
#                           (1 / (E_0[i] - E[k]) + 1 / (E_0[j] - E[k])) *
#                            (ψ_0[:, i].T.conj() @ V @ ψ[:, k]) * \
#                            (ψ[:, k].T.conj() @ V @ ψ_0[:, j])
#                            for k in range(n_eig))
#     # return H_eff_1
#     return H_eff_2 / GHz / 2 / np.pi


# def H_eff_SWT_eigs(ψb0, ψb, E):
#     Q = ψb0.T.conj() @ ψb
#     U, s, Vh = np.linalg.svd(Q)
#     A = U @ Vh
#     H_eff = A @ np.diag(E) @ A.T.conj()
#     return H_eff

# def H_eff_p1_frc(n_eig, H_frc_0, H_frc, Δ, Lq = 25, Lr = 10):
#     l   = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#
#     ψ_0 = diag(H_frc_0, n_eig)[1]
#
#     H_eff_p1 = np.zeros((n_eig, n_eig), dtype=complex)  # matrix to store our results.
#
#     for i in range(n_eig):
#         for j in range(n_eig):
#             H_eff_1_f_i_j = np.abs(ψ_0_f[:, N_f[i]].conj().T @ fluxonium.flux_op(0).__array__() @ ψ_0_f[:, N_f[j]])
#             H_eff_1_r_i_j = np.abs(ψ_0_r[:, N_r[i]].conj().T @ resonator.flux_op(0).__array__() @ ψ_0_r[:, N_r[j]])
#             H_eff_p1[i,j] = H_eff_1_f_i_j*H_eff_1_r_i_j
#
#     # return Δ  * (H_eff_p1 / (Δ * L_c) ) / GHz # / 2 / np.pi  Why not this 2pi!!!
#     return H_eff_p1 * 2 * Δ / l / 1e-9  / (2 * np.pi * GHz)

# def H_eff_p1_fluxonium_resonator_ij(fluxonium_0, fluxonium, resonator_0, resonator, i_f, j_f, i_r, j_r, Δ, Lq = 25, Lr = 10):
#     l = Lq * (Lq + 4 * Lr) - 4 * Δ ** 2
#     L_c = l / Δ * nH
#
#     ψ_0_f = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in fluxonium_0._evecs]).T)
#     ψ_0_r = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in resonator_0._evecs]).T)
#
#     Φ_f = ψ_0_f[:, i_f].conj().T @ fluxonium.flux_op(0).__array__() @ ψ_0_f[:, j_f]
#     Φ_r = ψ_0_r[:, i_r].conj().T @ resonator.flux_op(0).__array__() @ ψ_0_r[:, j_r]
#
#     return Φ_f * Φ_r / L_c / GHz #/ 2 /np.pi


#
# def hamiltonian_qubit_C_qubit(nmax_r, nmax_f, Cc, C=15, CJ=3, Csh=15, Lq=25, Lr=10, Δ=0.1,EJ=10, return_Ψ_nonint=False, n_eig_Ψ_nonint=4, periodic = True):
#     fF = 1e-15
#     C_R = C / 2
#     C_C = Cc
#     C_F = C / 2 + Csh + CJ
#
#     # if inverse == 'Numeric':
#     if periodic == True:
#         C_mat = np.array([[C_R + C_C / 2, 0, -C_C / 2, 0],
#                           [0, C_F + C_C / 2, 0, +C_C / 2],
#                           [-C_C / 2, 0, C_R + C_C / 2, 0],
#                           [0, +C_C / 2, 0, C_F + C_C / 2]])
#     else:
#         C_mat = np.array([[C_R + C_C / 2   , 0  , -C_C / 2, +C_C / 2],
#                           [0, C_F + C_C / 2, -C_C / 2 , +C_C / 2],
#                           [-C_C / 2 , +C_C / 2 , C_R + C_C / 2, 0],
#                           [-C_C / 2 , +C_C / 2, 0, C_F + C_C / 2]])
#
#         # C_mat = np.array([[C_R + C_C / 2   , 0  , -C_C / 2, +C_C / 2],
#         #                   [0, C_F + C_C / 2, -C_C / 2 , +C_C / 2],
#         #                   [-C_C / 2 , +C_C / 2 , C_R + C_C / 2, 0],
#         #                   [-C_C / 2 , +C_C / 2, 0, C_F + C_C / 2]])
#
#     C_inv = np.linalg.inv(C_mat)
#     C_R_tilde = C_inv[0, 0] ** -1
#     C_F_tilde = C_inv[1, 1] ** -1
#     if Cc == 0:
#         pass
#     else:
#         C_RR = C_inv[0, 2] ** -1
#         C_FF = C_inv[1, 3] ** -1
#     #
#     # elif inverse == 'Analytic':
#     #     C_R_tilde = (C_C / 2 + C_R) / (C_R * (C_C + C_R))
#     #     C_F_tilde = (C_C / 2 + C_F) / (C_F * (C_C + C_F))
#     #     C_RR = C_C / 2 / (C_R * (C_C + C_R))
#     #     C_FF = C_C / 2 / (C_F * (C_C + C_F))
#     #
#     # elif inverse == 'Approx':
#     #     C_R_tilde = C_R
#     #     C_F_tilde = C_F
#     #     C_RR = 2 * C_R ** 2 / C_C
#     #     C_FF = 2 * C_F ** 2 / C_C
#
#     fluxonium = sq_fluxonium(Lq=Lq, Lr=Lr, Δ=Δ, EJ=EJ, nmax_f=nmax_f, C_F_eff=C_F_tilde)
#     resonator = sq_resonator(Lq=Lq, Lr=Lr, Δ=Δ, EJ=EJ, nmax_r=nmax_r, C_R_eff=C_R_tilde)
#
#
#     if return_Ψ_nonint:
#         H_qubit, Ψ_q_0, E_q_0 = hamiltonian_qubit( nmax_f=nmax_f,nmax_r=nmax_r,Lq=Lq, Lr=Lr, Δ=Δ,EJ=EJ,  return_Ψ_nonint=return_Ψ_nonint)
#         # H_qubit, Ψ_q_0, E_q_0 = hamiltonian_qubit(fluxonium, resonator, Δ, return_Ψ_nonint=return_Ψ_nonint)
#         Nq_Nq = generate_and_prioritize_energies([E_q_0, E_q_0], n_eig_Ψ_nonint)[1]
#         Ψ_0 = [qt.tensor([Ψ_q_0[Nq_Nq_i[0]], Ψ_q_0[Nq_Nq_i[1]] ]) for Nq_Nq_i in Nq_Nq]
#     else:
#         H_qubit = hamiltonian_qubit(fluxonium, resonator, Δ)
#
#     I_r = qt.identity(nmax_r)
#     I_f = qt.identity(nmax_f)
#     I_qubit = qt.identity(H_qubit.dims[0])
#
#     q_r = qt.tensor(I_f, resonator.charge_op(0))
#     q_f = qt.tensor(fluxonium.charge_op(0), I_r)
#
#     H_0 = qt.tensor(H_qubit, I_qubit) + qt.tensor(I_qubit, H_qubit)
#     if Cc == 0:
#         if return_Ψ_nonint:
#             return H_0, Ψ_0
#         else:
#             return H_0
#
#     if periodic == True:
#         H_coupling = 1 / (C_RR * fF) * qt.tensor(q_r, q_r) + 1 / (C_FF * fF) * qt.tensor(q_f, q_f)
#     else:
#         C_RF = C_inv[0, 3] ** -1
#         H_coupling = 1 / (C_RR * fF) * qt.tensor(q_r, q_r) + 1 / (C_FF * fF) * qt.tensor(q_f, q_f) + \
#                      + 1 / (C_RF * fF) * qt.tensor(q_r, q_f) + + 1 / (C_RF * fF) * qt.tensor(q_f, q_r)
#
#     H = H_0 + H_coupling
#
#     if return_Ψ_nonint:
#         return H, Ψ_0
#     else:
#         return H

#%% Functions that are actually in sqcircuits file circuit.py

# def flux_op(self, mode: int, basis: str = 'FC') -> Qobj:
#     """Return flux operator for specific mode in the Fock/Charge basis or
#     the eigenbasis.
#
#     Parameters
#     ----------
#         mode:
#             Integer that specifies the mode number.
#         basis:
#             String that specifies the basis. It can be either ``"FC"``
#             for original Fock/Charge basis or ``"eig"`` for eigenbasis.
#     """
#
#     error1 = "Please specify the truncation number for each mode."
#     assert len(self.m) != 0, error1
#
#     # charge operator in Fock/Charge basis
#     Φ_FC = self._memory_ops["phi"][mode]
#
#     if basis == "FC":
#
#         return Φ_FC
#
#     elif basis == "eig":
#         ψ = real_eigenvectors(np.array([ψ_i.__array__()[:, 0] for ψ_i in self._evecs]).T)
#
#         Φ_eig = ψ.conj().T @ Φ_FC.__array__() @ ψ
#
#         return qt.Qobj(Φ_eig)


    # def diag(self, n_eig: int, solver='scipy', real=False) -> Tuple[ndarray, List[Qobj]]:
    #     """
    #     Diagonalize the Hamiltonian of the circuit and return the
    #     eigenfrequencies and eigenvectors of the circuit up to specified
    #     number of eigenvalues.
    #
    #     Parameters
    #     ----------
    #         n_eig:
    #             Number of eigenvalues to output. The lower ``n_eig``, the
    #             faster ``SQcircuit`` finds the eigenvalues.
    #     Returns
    #     ----------
    #         efreq:
    #             ndarray of eigenfrequencies in frequency unit of SQcircuit
    #             (gigahertz by default)
    #         evecs:
    #             List of eigenvectors in qutip.Qobj format.
    #     """
    #     error1 = "Please specify the truncation number for each mode."
    #     assert len(self.m) != 0, error1
    #     error2 = "n_eig (number of eigenvalues) should be an integer."
    #     assert isinstance(n_eig, int), error2
    #
    #     H = self.hamiltonian()
    #
    #     # get the data out of qutip variable and use sparse scipy eigen
    #     # solver which is faster.
    #     if solver == 'scipy':
    #         efreqs, evecs = scipy.sparse.linalg.eigs(H.data, n_eig, which='SR')
    #     elif solver == 'numpy':
    #         efreqs, evecs = np.linalg.eigh(H.__array__())
    #         efreqs = efreqs[:n_eig]
    #         evecs = evecs[:, :n_eig]
    #
    #     if real==True:
    #         evecs = real_eigenvectors(evecs)
    #
    #     # the output of eigen solver is not sorted
    #     efreqs_sorted = np.sort(efreqs.real)
    #
    #     sort_arg = np.argsort(efreqs)
    #     if isinstance(sort_arg, int):
    #         sort_arg = [sort_arg]
    #
    #     evecs_sorted = [
    #         qt.Qobj(evecs[:, ind], dims=[self.ms, len(self.ms) * [1]])
    #         for ind in sort_arg
    #     ]
    #
    #
    #     # store the eigenvalues and eigenvectors of the circuit Hamiltonian
    #     self._efreqs = efreqs_sorted
    #     self._evecs = evecs_sorted
    #
    #     return efreqs_sorted / (2*np.pi*unt.get_unit_freq()), evecs_sorted
