import time
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib as mpl

from copy import deepcopy
import pickle

import cantera as ct


def get_cantera_Y_str(dict_Yk, idx_Z, idx_C):
    """
    Create the mass fraction composition in the cantera format
    (Non-zero mass fraction only)
    """
    str_out = ''
    for key in dict_Yk.keys():
        if dict_Yk[key][idx_Z, idx_C]:
            str_out += key.decode()
            str_out += ':'
            str_out += "%f" % dict_Yk[key][idx_Z, idx_C]
            str_out += ', '
    return str_out[0:-2]


def get_cantera_mixture_nodes(dict_ZCT, dict_Yk, idx_Z, idx_C, ctml_file, pressure=ct.one_atm):
    """
    Create a Cantera mixture at from composition and temperature
    corresponding to (Z, C) nodes in the HDF5 FPVA table
    """
    T_table = dict_ZCT['T'][idx_Z, idx_C]
    Y_str = get_cantera_Y_str(dict_Yk, idx_Z, idx_C)
    gas_mix = ct.Solution(ctml_file)
    gas_mix.TPY = T_table, pressure, Y_str
    return gas_mix


def compute_ignition_time_hist_Tb(dict_ZCT, dict_Yk, idx_Z, idx_C, ctml_file, pressure=ct.one_atm):
    """
    Compute the ignition time for a given Z, C and T from table
    """

    print(idx_Z, idx_C)

    T_ref = dict_ZCT['T'][idx_Z, idx_C]

    gas = get_cantera_mixture_nodes(dict_ZCT, dict_Yk, idx_Z, idx_C, ctml_file, pressure=pressure)

    r = ct.IdealGasReactor(contents=gas, name='Batch Reactor')
    reactor_network = ct.ReactorNet([r])
    reactor_network.set_max_time_step(1e-3)

    # This is a starting estimate. If you do not get an ignition within this time, increase it
    # estimatedIgnitionDelayTime = 10e-3
    # estimatedIgnitionDelayTime = 5
    estimatedIgnitionDelayTime = 1.
    t = 0

    counter = 0
    dict_data = {'t': [], 'T': [], 'OH': []}

    while (t < estimatedIgnitionDelayTime):
        t = reactor_network.step()
        if not counter % 20:
            dict_data['t'].append(t)
            dict_data['T'].append(reactor_network.get_state()[2])
            dict_data['OH'].append(reactor_network.get_state()[8])

        if counter > 10000:
            print("\tmaximum number of iterations reached (%s) for idx : (%s,%s)" % (counter, idx_Z, idx_C))
            return np.nan, np.nan, np.nan

        counter += 1
    #
    df_T = pd.DataFrame.from_dict(dict_data)
    T_burnt = df_T['T'].max()

    try:
        # dT_dt = np.gradient(df_T['T'], df_T['t'])
        # idx = np.argmax(dT_dt)
        idx = np.argmax(df_T['OH'])
        ignition_time = df_T['t'][idx]
    except IndexError:
        # plt.plot(df_T['t'], df_T['T'])
        # plt.savefig("Figures/indexErrorMaxOH_idx_z_%s_idx_c_%s.png" % (idx_Z, idx_Z))
        # plt.close()
        ignition_time = np.nan

    if T_burnt <= 1.05 * T_ref:
        print("\tT_burnt < T_ref")
        ignition_time = np.nan

    if ignition_time > 0.9 * estimatedIgnitionDelayTime:
        # ignition_time = estimatedIgnitionDelayTime
        # plt.figure()
        # plt.plot(df_T['t'], df_T['T'])
        # plt.savefig('Figures/tau_ign_hugeTime_%s_%s.png' % (idx_Z, idx_C))
        # plt.close()
        ignition_time = np.nan

    alpha_tmp = ((T_burnt + 1e-6) - T_ref) / T_burnt

    return float(ignition_time), float(T_burnt), float(alpha_tmp)


if __name__ == "__main__":
    # define cantera mechanism and transport
    ctml_file = './Data/Table_TauIgn/0.025_OHstar.xml'

    # Load Yk, Z, C, and T
    with open('Data/Dict_Yk.pickle', 'rb') as handle:
        dict_Yk = pickle.load(handle)

    with open('Data/Dict_ZCT.pickle', 'rb') as handle:
        dict_ZCT = pickle.load(handle)

    # get length
    len_z, len_c = dict_Yk[b'CH4'].shape
    n_ct_simu = len_z * len_c

    # Compute ignition time
    tau_ign = np.zeros_like(dict_ZCT['T'])
    T_burnt = np.zeros_like(dict_ZCT['T'])
    alpha = np.zeros_like(dict_ZCT['T'])

    list_args = [(dict_ZCT, dict_Yk, idx_zz, idx_cc, ctml_file) for idx_zz in range(len_z) for idx_cc in range(len_c)]
    list_idx = [(idx_zz, idx_cc) for idx_zz in range(len_z) for idx_cc in range(len_c)]

    start_time = time.time()

    para = 1
    if para:
        print("Parallel computation")
        # Use ALL CPUs
        with Pool() as pool:
            map_result = pool.starmap(compute_ignition_time_hist_Tb, list_args)

        # Unpack results
        for idx, (res, ind) in enumerate(zip(map_result, list_idx)):
            print(tau_ign[ind])
            tau_ign[ind], T_burnt[ind], alpha[ind] = res

    else:
        print("Serial computation")
        for n_ite, (args, ind) in enumerate(zip(list_args, list_idx)):
            idx_z, idx_c = ind
            res = compute_ignition_time_hist_Tb(*args)
            tau_ign[ind], T_burnt[ind], alpha[ind] = res

    end_time = time.time()
    print("Execution time = %e" % (end_time - start_time))

    dict_res = {}
    dict_res['tau_ign'] = tau_ign
    dict_res['T_burnt'] = T_burnt
    dict_res['delta_T'] = T_burnt - dict_ZCT['T']
    dict_res['HeatReleaseParameter'] = (T_burnt - dict_ZCT['T']) / T_burnt

    with open('Data/Dict_res.pickle', 'wb') as handle:
        pickle.dump(dict_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

