import os
import pickle
import Modules.SQcircuit_extensions as sq_ext
import numpy as np

data_dir = r'/data'
opt_dir = r'/opt_results/'

#%% Constants
GHz = 1e9
nH = 1e-9
fF = 1e-15
h = 6.626e-34
e0 = 1.602e-19
Φ_0 = h / (2 * e0)

#%% Experimental spectra
def get_experimental_spectrum(experiment_name):
    if experiment_name == 'qubit_1_single_1':
        with open(os.getcwd() + data_dir + r'/current_extracted__tt_q1_single_circuit_1_up.pkl', 'rb') as f:
            current_extracted__tt_q1_single_circuit_1_up = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/mw_freq_extracted__tt_q1_single_circuit_1_up.pkl', 'rb') as f:
            mw_freq_extracted__tt_q1_single_circuit_1_up = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/current_extracted__tt_q1_single_circuit_1_down.pkl', 'rb') as f:
            current_extracted__tt_q1_single_circuit_1_down = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/mw_freq_extracted__tt_q1_single_circuit_1_down.pkl', 'rb') as f:
            mw_freq_extracted__tt_q1_single_circuit_1_down = pickle.load(f)

        current_extracted__tt_q1_single_circuit_1 = np.concatenate(
            [current_extracted__tt_q1_single_circuit_1_down, current_extracted__tt_q1_single_circuit_1_up])
        mw_freq_extracted__tt_q1_single_circuit_1 = np.concatenate(
            [mw_freq_extracted__tt_q1_single_circuit_1_down, mw_freq_extracted__tt_q1_single_circuit_1_up])

        I0 = (0.008275 + 0.007575) / 5
        Iss = -0.00035
        I_exp = current_extracted__tt_q1_single_circuit_1

        phase__tt_q1_single_circuit_1 = (I_exp - Iss + I0 / 2) / I0
        φ_ext_exp = phase__tt_q1_single_circuit_1
        ω_exp = mw_freq_extracted__tt_q1_single_circuit_1

        return φ_ext_exp, ω_exp, I_exp, I0, Iss

    elif experiment_name == 'resonator_1_single_1':
        with open(os.getcwd() + data_dir + r'/current_q1_single_circuit_1_crossings.pkl', 'rb') as f:
            current_q1_single_circuit_1_crossings = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/freq_q1_single_circuit_1_crossings.pkl', 'rb') as f:
            freq_q1_single_circuit_1_crossings = pickle.load(f)

        I0 = (0.008275 + 0.007575) / 5
        Iss = -0.00035
        I_exp = current_q1_single_circuit_1_crossings
        φ_ext_exp = (I_exp - Iss + I0 / 2) / I0

        ω_exp = freq_q1_single_circuit_1_crossings

        return φ_ext_exp, ω_exp, I_exp, I0, Iss

    elif experiment_name == 'resonator_and_qubit_1_single_1':
        with open(os.getcwd() + data_dir + r'/current_q1_single_circuit_1_crossings.pkl', 'rb') as f:
            current_q1_single_circuit_1_crossings = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/freq_q1_single_circuit_1_crossings.pkl', 'rb') as f:
            freq_q1_single_circuit_1_crossings = pickle.load(f)

        with open(os.getcwd() + data_dir + r'/current_extracted__tt_q1_single_circuit_1_up.pkl', 'rb') as f:
            current_extracted__tt_q1_single_circuit_1_up = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/mw_freq_extracted__tt_q1_single_circuit_1_up.pkl', 'rb') as f:
            mw_freq_extracted__tt_q1_single_circuit_1_up = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/current_extracted__tt_q1_single_circuit_1_down.pkl', 'rb') as f:
            current_extracted__tt_q1_single_circuit_1_down = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/mw_freq_extracted__tt_q1_single_circuit_1_down.pkl', 'rb') as f:
            mw_freq_extracted__tt_q1_single_circuit_1_down = pickle.load(f)

        I_exp_F = np.concatenate(
            [current_extracted__tt_q1_single_circuit_1_down, current_extracted__tt_q1_single_circuit_1_up])
        ω_exp_F = np.concatenate(
            [mw_freq_extracted__tt_q1_single_circuit_1_down, mw_freq_extracted__tt_q1_single_circuit_1_up])
        I0_F = (0.008275 + 0.007575) / 5
        Iss_F = -0.00035
        φ_ext_exp_F = (I_exp_F - Iss_F + I0_F / 2) / I0_F

        I_exp_R = current_q1_single_circuit_1_crossings
        ω_exp_R = freq_q1_single_circuit_1_crossings
        I0_R = (0.008275 + 0.007575) / 5
        Iss_R = -0.00035
        φ_ext_exp_R = (I_exp_R - Iss_R + I0_R / 2) / I0_R

        return φ_ext_exp_F, ω_exp_F, I_exp_F, I0_F, Iss_F, φ_ext_exp_R, ω_exp_R, I_exp_R, I0_R, Iss_R

    elif experiment_name == 'resonator_and_qubit_1_single_2':
        with open(os.getcwd() + data_dir + r'/current_q1_single_circuit_2_crossings.pkl', 'rb') as f:
            current_q1_single_circuit_2_crossings = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/freq_q1_single_circuit_2_crossings.pkl', 'rb') as f:
            freq_q1_single_circuit_2_crossings = pickle.load(f)

        with open(os.getcwd() + data_dir + r'/current_extracted__tt_q1_single_circuit_2.pkl', 'rb') as f:
            current_extracted__tt_q1_single_circuit_2 = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/mw_freq_extracted__tt_q1_single_circuit_2.pkl', 'rb') as f:
            mw_freq_extracted__tt_q1_single_circuit_2 = pickle.load(f)

        I_exp_F = current_extracted__tt_q1_single_circuit_2
        ω_exp_F = mw_freq_extracted__tt_q1_single_circuit_2
        I0_F = (0.00825 + 0.00762) / 5
        Iss_F = 0.0013
        φ_ext_exp_F = (I_exp_F - Iss_F + I0_F / 2) / I0_F

        I_exp_R = current_q1_single_circuit_2_crossings
        ω_exp_R = freq_q1_single_circuit_2_crossings
        I0_R = (0.00825 + 0.00762) / 5
        Iss_R = 0.001285
        φ_ext_exp_R = (I_exp_R - Iss_R + I0_R / 2) / I0_R

        return φ_ext_exp_F, ω_exp_F, I_exp_F, I0_F, Iss_F, φ_ext_exp_R, ω_exp_R, I_exp_R, I0_R, Iss_R

    elif experiment_name == 'qubit_1':
        with open(os.getcwd() + data_dir + r'/x__q1_tt_low_q1.pkl', 'rb') as f:
            x__q1_tt_low = pickle.load(f)[0]
        with open(os.getcwd() + data_dir + r'/y__q1_tt_low_q1.pkl', 'rb') as f:
            y__q1_tt_low = pickle.load(f)[0]

        with open(os.getcwd() + data_dir + r'/x__q1_tt_up_q1.pkl', 'rb') as f:
            x__q1_tt_up = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/y__q1_tt_up_q1.pkl', 'rb') as f:
            y__q1_tt_up = pickle.load(f)

        Anzahl_Perioden = 2
        I_ss_low__fs = -0.00375
        I_ss_up__fs = 0.00267
        I0_q1 = (I_ss_up__fs - I_ss_low__fs) / Anzahl_Perioden
        I_ss__q1_tt_low = 1e-5

        x__q1_tt = np.concatenate([x__q1_tt_low, x__q1_tt_up])
        y__q1_tt = np.concatenate([y__q1_tt_low, y__q1_tt_up])
        phi_q1_tt = (x__q1_tt - I_ss__q1_tt_low + I0_q1 / 2) / I0_q1

        I0 = I0_q1
        Iss = I_ss__q1_tt_low
        φ_ext_exp = phi_q1_tt
        ω_exp = y__q1_tt
        I_exp = x__q1_tt

        return φ_ext_exp, ω_exp, I_exp, I0, Iss

    elif experiment_name == 'resonator_1':

        with open(os.getcwd() + data_dir + r'/current__fres_q1_coil1.pkl', 'rb') as f:
            current_q1_coil1 = pickle.load(f)
        with open(os.getcwd() + data_dir + r'/fres__fres_q1_coil1.pkl', 'rb') as f:
            fres_q1_coil1 = pickle.load(f)

        current_q1_coil1 = np.concatenate([current_q1_coil1[0], current_q1_coil1[1], current_q1_coil1[2]])
        fres_q1_coil1 = np.concatenate([fres_q1_coil1[0], fres_q1_coil1[1], fres_q1_coil1[2]])

        I_ss__q1_fs = 1.12 * 1e-5
        I_ss_low__fs = -0.00375
        I_ss_up__fs = 0.00267
        Anzahl_Perioden = 2
        I0_q1 = (I_ss_up__fs - I_ss_low__fs) / Anzahl_Perioden
        phi_q1_coil1 = (current_q1_coil1 - I_ss__q1_fs + I0_q1 / 2) / I0_q1

        I0 = I0_q1
        Iss = I_ss__q1_fs
        I_exp = current_q1_coil1
        φ_ext_exp = phi_q1_coil1
        ω_exp = fres_q1_coil1

        return φ_ext_exp, ω_exp, I_exp, I0, Iss



#%% Theoretical spectra
def get_theoretical_spectrum(experiment_name):
    if experiment_name == 'qubit_1_single_1' or experiment_name == 'qubit_1':
        def qubit_spectrum(parameters, data_set, out='error'):
            CF, LF, EJ, I0, I_origin = parameters
            I_exp, ω_exp = data_set

            φ_ext_values = (I_exp - I_origin) / I0

            fluxonium = sq_ext.sq_fluxonium(C_F_eff=CF, L_F_eff=LF, EJ=EJ)
            loop = fluxonium.loops[0]
            ω_vs_φ_ext = np.zeros(len(φ_ext_values))

            for i, φ_ext in enumerate(φ_ext_values):
                loop.set_flux(φ_ext)
                fluxonium.diag(2)
                ω_vs_φ_ext[i] = fluxonium.efreqs[1] - fluxonium.efreqs[0]

            if out == 'error':
                error = np.sum((ω_vs_φ_ext - ω_exp * 1e-9) ** 2)
                print(error)
                return error

            elif out == 'spectrum':
                return φ_ext_values, ω_vs_φ_ext * 1e9
        return qubit_spectrum

    if experiment_name == 'resonator_1_single_1':
        def r_q_av_cross_single_spectrum(parameters, data_set, out='error'):

            CR, LR, Δ, I0, I_origin = parameters
            I_exp, ω_exp, crossing_index_1, crossing_index_2, CF, LF, EJ, nmax_r, nmax_f = data_set

            φ_ext_values = (I_exp - I_origin) / I0

            Lq, Lr = sq_ext.LF_LR_eff_to_Lq_Lr(LF=LF, LR=LR, Δ=Δ)
            resonator = sq_ext.sq_resonator(C_R_eff=CR, L_R_eff=LR, nmax_r=nmax_r)
            qubit = sq_ext.sq_fluxonium(C_F_eff=CF, L_F_eff=LF, EJ=EJ, nmax_f=nmax_f)
            loop = qubit.loops[0]

            ω_vs_φ_ext = np.zeros([len(φ_ext_values), 2])
            for i, φ_ext in enumerate(φ_ext_values):
                loop.set_flux(φ_ext)
                H = sq_ext.hamiltonian_qubit(fluxonium=qubit, resonator=resonator, Lq=Lq, Lr=Lr, Δ=Δ)
                ω_vs_φ_ext[i] = sq_ext.diag(H, 3, remove_ground=True)[0][1:]
            ω_vs_φ_ext = np.concatenate(
                [ω_vs_φ_ext[0:crossing_index_1, 0], ω_vs_φ_ext[crossing_index_1:-crossing_index_2, 1],
                 ω_vs_φ_ext[-crossing_index_2:, 0]])

            if out == 'error':
                error = np.sum((ω_vs_φ_ext - ω_exp * 1e-9) ** 2)
                print(error)
                return error
            elif out == 'spectrum':
                return φ_ext_values, ω_vs_φ_ext * 1e9
        return r_q_av_cross_single_spectrum

    if experiment_name == 'resonator_and_qubit_1_single_1':
        def unit_cell_single_spectrum(parameters, data_set, out='error'):

            CF, LF, EJ, I0_F, I_origin_F, CR, LR, Δ, I0_R, I_origin_R = parameters
            I_exp_F, ω_exp_F, I_exp_R, ω_exp_R, crossing_index_1_F, crossing_index_1_R, crossing_index_2_R, nmax_r, nmax_f = data_set

            Lq, Lr = sq_ext.LF_LR_eff_to_Lq_Lr(LF=LF, LR=LR, Δ=Δ)
            resonator = sq_ext.sq_resonator(C_R_eff=CR, L_R_eff=LR, nmax_r=nmax_r)
            qubit = sq_ext.sq_fluxonium(C_F_eff=CF, L_F_eff=LF, EJ=EJ, nmax_f=nmax_f)
            loop = qubit.loops[0]

            φ_ext_R = (I_exp_R - I_origin_R) / I0_R
            ωR_vs_φ_ext = np.zeros([len(φ_ext_R), 2])
            for i, φ_ext in enumerate(φ_ext_R):
                loop.set_flux(φ_ext)
                H = sq_ext.hamiltonian_qubit(fluxonium=qubit, resonator=resonator, Lq=Lq, Lr=Lr, Δ=Δ)
                ωR_vs_φ_ext[i] = sq_ext.diag(H, 3, remove_ground=True)[0][1:]
            ωR_vs_φ_ext = np.concatenate(
                [ωR_vs_φ_ext[0:crossing_index_1_R, 0], ωR_vs_φ_ext[crossing_index_1_R:-crossing_index_2_R, 1],
                 ωR_vs_φ_ext[-crossing_index_2_R:, 0]])

            φ_ext_F = (I_exp_F - I_origin_F) / I0_F
            ωF_vs_φ_ext = np.zeros([len(φ_ext_F), 2])
            for i, φ_ext in enumerate(φ_ext_F):
                loop.set_flux(φ_ext)
                H = sq_ext.hamiltonian_qubit(fluxonium=qubit, resonator=resonator, Lq=Lq, Lr=Lr, Δ=Δ)
                ωF_vs_φ_ext[i] = sq_ext.diag(H, 3, remove_ground=True)[0][1:]
            ωF_vs_φ_ext = np.concatenate([ωF_vs_φ_ext[:crossing_index_1_F, 0], ωF_vs_φ_ext[crossing_index_1_F:, 1]])

            if out == 'error':
                error = np.sum((ωR_vs_φ_ext - ω_exp_R * 1e-9) ** 2) + np.sum((ωF_vs_φ_ext - ω_exp_F * 1e-9) ** 2)
                print(error)
                return error
            elif out == 'spectrum':
                return φ_ext_F, ωF_vs_φ_ext * 1e9, φ_ext_R, ωR_vs_φ_ext * 1e9
        return unit_cell_single_spectrum

    if experiment_name == 'resonator_and_qubit_1_single_2':
        def unit_cell_single_spectrum(parameters, data_set, out='error'):

            CF, LF, EJ, I0_F, I_origin_F, CR, LR, Δ, I0_R, I_origin_R = parameters
            I_exp_F, ω_exp_F, I_exp_R, ω_exp_R, crossing_index_1_F, crossing_index_1_R, crossing_index_2_R, nmax_r, nmax_f = data_set

            Lq, Lr = sq_ext.LF_LR_eff_to_Lq_Lr(LF=LF, LR=LR, Δ=Δ)
            resonator = sq_ext.sq_resonator(C_R_eff=CR, L_R_eff=LR, nmax_r=nmax_r)
            qubit = sq_ext.sq_fluxonium(C_F_eff=CF, L_F_eff=LF, EJ=EJ, nmax_f=nmax_f)
            loop = qubit.loops[0]

            φ_ext_R = (I_exp_R - I_origin_R) / I0_R
            ωR_vs_φ_ext = np.zeros([len(φ_ext_R), 2])
            for i, φ_ext in enumerate(φ_ext_R):
                loop.set_flux(φ_ext)
                H = sq_ext.hamiltonian_qubit(fluxonium=qubit, resonator=resonator, Lq=Lq, Lr=Lr, Δ=Δ)
                ωR_vs_φ_ext[i] = sq_ext.diag(H, 3, remove_ground=True)[0][1:]
            ωR_vs_φ_ext = np.concatenate(
                [ωR_vs_φ_ext[0:crossing_index_1_R, 0], ωR_vs_φ_ext[crossing_index_1_R:-crossing_index_2_R, 1],
                 ωR_vs_φ_ext[-crossing_index_2_R:, 0]])

            φ_ext_F = (I_exp_F - I_origin_F) / I0_F
            ωF_vs_φ_ext = np.zeros(len(φ_ext_F))
            for i, φ_ext in enumerate(φ_ext_F):
                loop.set_flux(φ_ext)
                H = sq_ext.hamiltonian_qubit(fluxonium=qubit, resonator=resonator, Lq=Lq, Lr=Lr, Δ=Δ)
                ωF_vs_φ_ext[i] = sq_ext.diag(H, 2, remove_ground=True)[0][1]

            if out == 'error':
                error = np.sum((ωR_vs_φ_ext - ω_exp_R * 1e-9) ** 2) + np.sum((ωF_vs_φ_ext - ω_exp_F * 1e-9) ** 2)
                print(error)
                return error
            elif out == 'spectrum':
                return φ_ext_F, ωF_vs_φ_ext * 1e9, φ_ext_R, ωR_vs_φ_ext * 1e9
        return unit_cell_single_spectrum

    elif experiment_name == 'resonator_1':
        def r_q_av_cross_spectrum(parameters, data_set, out='error'):

            C_int, CR, LR, Δ, I0, I_origin = parameters
            I_exp, ω_exp, crossing_index_1, crossing_index_2, CF, LF, EJ, nmax_r, nmax_f = data_set

            Lq, Lr = sq_ext.LF_LR_eff_to_Lq_Lr(LF=LF, LR=LR, Δ=Δ)

            resonator = sq_ext.sq_resonator(C_R_eff=CR, L_R_eff=LR, nmax_r=nmax_r)
            qubit = sq_ext.sq_fluxonium(C_F_eff=CF, L_F_eff=LF, EJ=EJ, nmax_f=nmax_f)
            loop = qubit.loops[0]

            φ_ext_values = (I_exp - I_origin) / I0
            ω_vs_φ_ext = np.zeros([len(φ_ext_values), 2])
            for i, φ_ext in enumerate(φ_ext_values):
                loop.set_flux(φ_ext)
                H = sq_ext.hamiltonian_qubit(fluxonium=qubit, resonator=resonator, Lq=Lq, Lr=Lr, Δ=Δ, C_int=C_int)
                ω_vs_φ_ext[i] = sq_ext.diag(H, 3, remove_ground=True)[0][1:]
            ω_vs_φ_ext = np.concatenate([ω_vs_φ_ext[0:crossing_index_1, 0], ω_vs_φ_ext[crossing_index_1:-crossing_index_2, 1], ω_vs_φ_ext[-crossing_index_2:, 0]])

            if out == 'error':
                error = np.sum((ω_vs_φ_ext - ω_exp*1e-9) ** 2)
                print(error)
                return error
            elif out == 'spectrum':
                return φ_ext_values, ω_vs_φ_ext * GHz
        # hope not
        return r_q_av_cross_spectrum

                #%% Optimization results
def load_optimization_results(experiment_name):
    experiment_file = os.getcwd() + opt_dir + experiment_name + '.npz'
    data = np.load(experiment_file)

    parameters_opt   = data['parameters_opt']
    parameters_guess = data['parameters_guess']
    bounds           = (data['bounds'])

    return parameters_opt, parameters_guess, bounds

