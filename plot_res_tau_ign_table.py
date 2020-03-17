"""
Plot results of the tau_ignition_table_nodes.py script
--> read pickle of dictionnaries
"""

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

dir_path = os.path.dirname(os.path.realpath(__file__))
print("running from %s" % dir_path)

# try:
# plt.style.use('../Cantera/style/my538_cnf.mplstyle')
# plt.style.use('my538_PCI.mplstyle')
#plt.style.use('/home/douasbin/Python/matplotlib_styles/my538_2020_03_09_cnf.mplstyle')
plt.style.use('my538_2020_03_09_cnf.mplstyle')
# except(OSError):
#     print("Style matplotlib not found")

# global vars
PLOT = 0
SMALL = 1e-4
WIREFRAMES = False

# list_ext = ['png', 'pdf', 'eps']
list_ext = ['png', 'pdf']
# list_ext = ['pdf']


def get_ticks(arr, n_label=11):
    return np.linspace(np.nanmin(arr), np.nanmax(arr), n_label, endpoint=True)


def interpolate_result_on_mixing_line(data, Z, C, C_coflow):
    """
    Interpolate a 2d field data(Z, C) on the mixing line.
    For each Z value, we find the 2 closest C points and linearly interpolate the quantity
    :param data: 2D filed. f = f(Z,C)
    :param Z: Mixture fraction 1D array
    :param C: Progress variable 2D array
    :param C_coflow: value of c in the coflow (oxidizer stream)
    :return: xi, 1d_data(xi)
    """
    xi = np.zeros_like(Z)
    data_of_xi = np.zeros_like(Z)

    xi = Z + (1. - Z) * C_coflow
    print(np.shape(xi))

    for idx_z, _ in enumerate(Z):
        c_val = (1. - Z[idx_z]) * C_coflow
        c_arr = dict_ZCT['C']
        data_idx_z = data[idx_z, :]
        data_of_xi[idx_z] = np.interp(c_val, c_arr, data_idx_z)
        # if data_of_xi[idx_z] == np.nan:
        #     data_of_xi = -1

    return xi, data_of_xi


def mask_below_mixing_line(data, Z, C, C_coflow):
    """
    Interpolate a 2d field data(Z, C) on the mixing line.
    For each Z value, we find the 2 closest C points and linearly interpolate the quantity
    :param data: 2D filed. f = f(Z,C)
    :param Z: Mixture fraction 1D array
    :param C: Progress variable 2D array
    :param C_coflow: value of c in the coflow (oxidizer stream)
    :return: xi, 1d_data(xi)
    """
    for idx_z, _ in enumerate(Z):
        for idx_c, C_tab in enumerate(C):
            c_val_mix = (1. - Z[idx_z]) * C_coflow
            if C_tab < c_val_mix:
                data[idx_z, idx_c] = np.nan
    return data


def mask_below_mixing_line_margin_C(data, Z, C, C_coflow, margin_C=0.05):
    """
    Interpolate a 2d field data(Z, C) on the mixing line.
    For each Z value, we find the 2 closest C points and linearly interpolate the quantity
    :param data: 2D filed. f = f(Z,C)
    :param Z: Mixture fraction 1D array
    :param C: Progress variable 2D array
    :param C_coflow: value of c in the coflow (oxidizer stream)
    :param margin_C: margin around mixing line in C direction (for grads)
    :return: xi, 1d_data(xi)
    """
    for idx_z, _ in enumerate(Z):
        for idx_c, C_tab in enumerate(C):
            c_val_mix = (1. - Z[idx_z]) * C_coflow
            if C_tab < c_val_mix - margin_C:
                data[idx_z, idx_c] = np.nan
    return data


def mask_field(data):
    """
    Mask stupid data that makes tau_ig results not relevant (or continuous)
    """
    tmp = data
    tmp[alpha < thresh_alpha] = np.nan
    tmp[tau_ign < thresh_tau_ign_min] = np.nan
    tmp[tau_ign > thresh_tau_ign_max] = np.nan
    return tmp


def mask_field_below_small(data, threshold=SMALL):
    """
    Masks the data that is below the SMALL threshold
    """
    data[np.abs(data) < threshold] = np.nan
    return data


if __name__ == "__main__":
    # --------------------------------------------------------
    # thresholds
    # Values used for table PCI
    # thresh_alpha = 0.05
    # thresh_tau_ign_min = 1e-5
    # thresh_tau_ign_max = 1e-2

    # Curent values
    thresh_alpha = 0.01
    thresh_tau_ign_min = 1e-5
    thresh_tau_ign_max = 1e-2
    n_labels_log = int(np.log10(thresh_tau_ign_max) - np.log10(
        thresh_tau_ign_min)) + 1

    # --------------------------------------------------------
    # Load Data
    with open('Data/Dict_Yk.pickle', 'rb') as handle:
        dict_Yk = pickle.load(handle)

    with open('Data/Dict_ZCT.pickle', 'rb') as handle:
        dict_ZCT = pickle.load(handle)

    with open('Data/Dict_res.pickle', 'rb') as handle:
        dict_res = pickle.load(handle)

    x = dict_ZCT['Z']
    y = dict_ZCT['C']
    prog = dict_Yk[b'CO'] + dict_Yk[b'CO2'] + dict_Yk[b'H2O'] + dict_Yk[b'H2']

    # Add small to compute T0 -T_Burnt without numerical error
    T0 = dict_ZCT['T'] + SMALL

    X, Y = np.meshgrid(x, y)

    tau_ign = dict_res['tau_ign']
    T_burnt = dict_res['T_burnt']
    diff_T = T_burnt - T0
    alpha = (T_burnt - T0) / T_burnt

    tau_ign = mask_field(tau_ign)

    # Crop data below mixing line (keep a margin in C for grad computation)
    #  tau_ign = mask_below_mixing_line_margin_C(tau_ign,
    #                                            dict_ZCT['Z'],
    #                                            dict_ZCT['C'],
    #                                            C_coflow=0.121478,
    #                                            margin_C=0.01)

    # --------------------------------------------------------

    # --------------------------------------------------------
    # Compute Gradients

    # first gradient in Y, then gradient in X
    dtau_dC, dtau_dZ = np.gradient(tau_ign.T,
                                   dict_ZCT['C'],
                                   dict_ZCT['Z'],
                                   edge_order=2)

    check_grad_comp = 0
    if check_grad_comp:
        plt.close("all")

        # Analytical field
        T0 = np.transpose(
            np.sin(2. * 3. * np.pi * np.ones_like(T0) * x) / (6 * np.pi))
        T0 += np.cos(2. * np.pi * Y.T) / (2. * np.pi)

        # Do gradients
        dT0_dC, dT0_dZ = np.gradient(T0.T,
                                     dict_ZCT['C'],
                                     dict_ZCT['Z'],
                                     edge_order=2)

        # Plot res
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"T0")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, T0.T, alpha=0.75, color='C1', label='T0')
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dT0_dZ")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dT0_dZ, alpha=0.75)
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dT0_dC")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dT0_dC, alpha=0.75)
        plt.legend()
        plt.tight_layout()

        plt.show()

  # # ---- Plot prog ----#
  # plt.rcParams['image.cmap'] = "hot"
  # fig, ax = plt.subplots()
  # cf = ax.contourf(X, Y, prog.T, 100)
  # for cnt in cf.collections:
  #     cnt.set_edgecolor("face")
  # ax.set_xlabel("Mixture fraction $Z$ [-]")
  # ax.set_ylabel("Normalized progress variable $C$ [-]")
  # ax.tick_params(axis='x', pad=3)
  # lim = np.linspace(np.amin(prog),
  #                   np.amax(prog), 2)
  # print(lim)
  # n_labels = 10
  # cb1 = plt.colorbar(cf, ticks=get_ticks(lim, n_label=n_labels))
  # cb1.set_label(r'$\Lambda$ [-]')
  # plt.tight_layout()
  # for ext in list_ext:
  #     plt.savefig('Figures/CNF_table_prog_var.%s' % ext, transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=300)
  # plt.rcParams['image.cmap'] = "hot"

  # fig, ax = plt.subplots()
  # cf = ax.contourf(X, Y, prog.T, 100)
  # for cnt in cf.collections:
  #     cnt.set_edgecolor("face")
  # ax.set_xlabel("Mixture fraction $Z$ [-]")
  # ax.set_ylabel("Normalized progress variable $C$ [-]")
  # ax.tick_params(axis='x', pad=3)
  # lim = np.linspace(np.amin(prog),
  #                   np.amax(prog), 2)
  # print(lim)
  # n_labels = 10
  # cb1 = plt.colorbar(cf, ticks=get_ticks(lim, n_label=n_labels))
  # cb1.set_label(r'$\Lambda$ [-]')
  # plt.tight_layout()
  # for ext in list_ext:
  #     plt.savefig('Figures/CNF_table_prog_var.%s' % ext, transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=300)

  # plt.rcParams['image.cmap'] = "hot_r"
  # # plt.rcParams['image.cmap'] = "Blues_r"
  # fig, ax = plt.subplots()
  # cf = ax.contourf(X, Y, np.log10(tau_ign.T), 100, antialiased=True)
  # for cnt in cf.collections:
  #     cnt.set_edgecolor("face")
  # cb1 = plt.colorbar(cf, ticks=get_ticks(np.log10(tau_ign), n_label=11))
  # levels = [0.05, None]
  # CT = ax.contour(X, Y, alpha.T,
  #                 levels,
  #                 colors='k')
  # ax.set_xlim(0, 0.3)
  # ax.set_xlabel("Mixture fraction $Z$ [-]")
  # ax.set_ylabel("Normalized progress variable $C$ [-]")
  # ax.tick_params(axis='x', pad=3)
  # cb1.set_label(r'$\log_{10}(\tau_{\rm ig})$ [s]')
  # plt.tight_layout()
  # for ext in list_ext:
  #     plt.savefig('Figures/CNF_tau_ig.%s' % ext, transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=300)

    # Plot non-normalized prog
    fig, ax = plt.subplots()
    cf = ax.contourf(prog, Y, dict_ZCT['T'].T, 100)
    for cnt in cf.collections:
        cnt.set_edgecolor("face")
    ax.set_xlabel("Mixture fraction $Z$ [-]")
    ax.set_ylabel("Normalized progress variable $C$ [-]")
    ax.tick_params(axis='x', pad=3)
    lim = np.linspace(np.amin(prog),
                      np.amax(prog), 2)
    print(lim)
    n_labels = 10
    cb1 = plt.colorbar(cf, ticks=get_ticks(lim, n_label=n_labels))
    cb1.set_label(r'$\Lambda$ [-]')
    #  plt.tight_layout()
    #  for ext in list_ext:
    #      plt.savefig('Figures/CNF_unormed_prog_var_T.%s' % ext, transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.show()

    plt.rcParams['image.cmap'] = "hot_r"
    # plt.rcParams['image.cmap'] = "Blues_r"
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, np.log10(tau_ign.T), 100, antialiased=True)
    for cnt in cf.collections:
        cnt.set_edgecolor("face")
    cb1 = plt.colorbar(cf, ticks=get_ticks(np.log10(tau_ign), n_label=11))
    levels = [0.05, None]
    CT = ax.contour(X, Y, alpha.T,
                    levels,
                    colors='k')
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("Mixture fraction $Z$ [-]")
    ax.set_ylabel("Normalized progress variable $C$ [-]")
    ax.tick_params(axis='x', pad=3)
    cb1.set_label(r'$\log_{10}(\tau_{\rm ig})$ [s]')
    plt.tight_layout()
    for ext in list_ext:
        plt.savefig('Figures/CNF_tau_ig.%s' % ext, transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=300)
    # plt.savefig("Figures/CNF_tau_ig.ext")

    # plt.show()

    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, T_burnt.T, 201)
    # plt.title(r"T_Burnt")
    # cb1 = plt.colorbar(cf, ticks=get_ticks(T_burnt, n_label=11))
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    # cb1.set_label(r'[K]')
    #
    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, X, 201)
    # plt.title(r"X -> Z")
    # cb1 = plt.colorbar(cf, ticks=get_ticks(X, n_label=11))
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    #
    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, Y, 201)
    # plt.title(r"Y -> C")
    # cb1 = plt.colorbar(cf, ticks=get_ticks(Y, n_label=11))
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    #
    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, tau_ign_xy, 201)
    # plt.title(r"tau_ig")
    # cb1 = plt.colorbar(cf, ticks=get_ticks(tau_ign_xy, n_label=11))
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    # cb1.set_label(r'tau_ig')
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_title(r"dTau_dZ")
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    # ax.plot_wireframe(X, Y, dtau_dZ)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_title(r"dTau_dC")
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    # ax.plot_wireframe(X, Y, dtau_dC)
    # #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_title(r"tau_ig")
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    # ax.scatter(X, Y, np.log10(tau_ign_xy), '.', alpha=0.5)

    if WIREFRAMES:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"tau_ign")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, tau_ign.T, alpha=0.75, color='C1', label='tau_ign')
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dtau_ign_dZ")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dtau_dZ, alpha=0.75)
        plt.legend()
        plt.tight_layout()

        # plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dtau_ign_dC")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dtau_dC, alpha=0.75)
        plt.legend()
        plt.tight_layout()

    #
    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, dtau_dZ, 201)
    # plt.title(r"dtau_dZ")
    # cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dZ, n_label=11))
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    # cb1.set_label(r'dtau_dZ')
    #
    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, Y, dtau_dC, 201)
    # plt.title(r"dtau_dC")
    # cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dC, n_label=11))
    # ax.set_xlabel("Mixture fraction $Z$ [-]")
    # ax.set_ylabel("C/Cmax")
    # cb1.set_label(r'dtau_dC')

    # plt.show()

    # -------------- Calc sensitivity coeff T -----------------
    dT_dC, dT_dZ = np.gradient(T0.T,
                               dict_ZCT['C'],
                               dict_ZCT['Z'],
                               edge_order=2)

    masked_tau = mask_field(tau_ign)

    # dTau_dT
    masked_inv_dT_dZ = 1. / mask_field_below_small(dT_dZ)
    masked_inv_dT_dC = 1. / mask_field_below_small(dT_dC)

    dtau_dT = dtau_dZ * masked_inv_dT_dZ + dtau_dC * masked_inv_dT_dC
    # ---------------------------------------------------------

    # Plot res
    if WIREFRAMES:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"T0")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, T0.T, alpha=0.75, color='C1', label='T0')
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dT_dZ")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dT_dZ, alpha=0.75)
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dT_dC")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dT_dC, alpha=0.75)
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dZ_dT")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, masked_inv_dT_dZ, alpha=0.75)
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dC_dT")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, masked_inv_dT_dC, alpha=0.75)
        plt.legend()
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dtau_dT")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dtau_dT, alpha=0.75)
        plt.legend()
        plt.tight_layout()

    # -------------- Calc sensitivity coeff OH -----------------
    Y_OH = mask_field(dict_Yk[b'OH'])
    dY_OH_dC, dY_OH_dZ = np.gradient(Y_OH.T,
                                     dict_ZCT['C'],
                                     dict_ZCT['Z'],
                                     edge_order=2)

    inv_dY_OH_dZ = 1. / mask_field_below_small(dY_OH_dZ)
    inv_dY_OH_dC = 1. / mask_field_below_small(dY_OH_dC)

    masked_inv_dY_OH_dZ = inv_dY_OH_dZ
    masked_inv_dY_OH_dC = inv_dY_OH_dC

    dtau_dY_OH = dtau_dZ * masked_inv_dY_OH_dZ + dtau_dC * masked_inv_dY_OH_dC
    # ----------------------------------------------------------
    if WIREFRAMES:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"OH")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dict_Yk[b'OH'].T, alpha=0.75)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dY_OH_dZ")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dY_OH_dZ, alpha=0.75)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dY_OH_dC")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dY_OH_dC, alpha=0.75)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"inv dY_OH_dZ")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, masked_inv_dY_OH_dZ, alpha=0.75)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"inv dY_OH_dC")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, masked_inv_dY_OH_dC, alpha=0.75)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r"dtau_dY_OH")
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C/Cmax")
        ax.plot_wireframe(X, Y, dtau_dY_OH, alpha=0.75)
        plt.tight_layout()

    # --------------------------------------------------------

    # --------------------------------------------------------
    # Output results
    dict_gradients = {}
    dict_gradients[b"tau_ig"] = np.array(np.nan_to_num(tau_ign))
    dict_gradients[b"dT_dZ"] = np.array(np.nan_to_num(dT_dZ.T))
    dict_gradients[b"dT_dC"] = np.array(np.nan_to_num(dT_dC.T))
    dict_gradients[b"dtau_dZ"] = np.array(np.nan_to_num(dtau_dZ.T))
    dict_gradients[b"dtau_dC"] = np.array(np.nan_to_num(dtau_dC.T))
    dict_gradients[b"dtau_dT"] = np.array(np.nan_to_num(dtau_dT.T))
    dict_gradients[b"dtau_dY_OH"] = np.array(np.nan_to_num(dtau_dY_OH.T))

    debug_output_field = 0
    if debug_output_field:
        plt.close("all")

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dtau_dZ, 201)
        plt.title(r"dtau_dZ field")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dZ, n_label=7))
        ax.set_xlabel("Z")
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        ax.tick_params(axis='x', pad=3)
        plt.savefig("output_filt_dtau_dZ.png")

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, np.nan_to_num(dict_gradients[b'dtau_dZ']), 201)
        plt.title(r"dtau_dZ out")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dZ, n_label=7))
        ax.set_xlabel("Z")
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        ax.tick_params(axis='x', pad=3)
        plt.savefig("output_filt_to_zero_dtau_dZ.png")

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dtau_dC, 201)
        plt.title(r"dtau_dZ field")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dC, n_label=7))
        ax.set_xlabel("Z")
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        ax.tick_params(axis='x', pad=3)
        plt.savefig("output_filt_dtau_dC.png")

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dict_gradients[b'dtau_dC'], 201)
        plt.title(r"dtau_dZ out")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dC, n_label=7))
        ax.set_xlabel("Z")
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        ax.tick_params(axis='x', pad=3)
        plt.savefig("output_filt_to_zero_dtau_dC.png")

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dict_gradients[b'dtau_dT'], 201)
        plt.title(r"dtau_dT out")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dict_gradients[b'dtau_dT'],
                                               n_label=7))
        ax.set_xlabel("Z")
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        ax.tick_params(axis='x', pad=3)

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dict_gradients[b'dtau_dY_OH'], 201)
        plt.title(r"dtau_dY_OH out")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dict_gradients[b'dtau_dY_OH'],
                                               n_label=7))
        ax.set_xlabel("Z")
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        ax.tick_params(axis='x', pad=3)

        plt.show()

    with open('Data/Dict_gradients.pickle', 'wb') as handle:
        print("Writting results.")
        pickle.dump(dict_gradients, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done.")

    # --------------------------------------------------------
    # --------------------------------------------------------
    # Plot

    ## plot flamelet
    # plt.figure()
    # plt.title('progress variables vs mixture fraction')
    # plt.plot(x, prog[:, ::100])
    #
    # plt.figure()
    # plt.title('tau_ign vs mixture fraction')
    # plt.plot(x, tau_ign[:, 0], '--')
    # # plt.plot(x, tau_ign[:, ::100])
    #
    # plt.figure()
    # plt.title('dtau_dZ vs mixture fraction')
    # plt.plot(x, dtau_dZ[:, ::100])
    #
    # plt.figure()
    # plt.title('dtau_dT vs mixture fraction')
    # plt.plot(x, dtau_dT[:, ::100])
    #
    # plt.figure()
    # plt.title('dtau_dC vs mixture fraction')
    # plt.plot(x, dtau_dC[:, ::100])
    # plt.show()

    if 0:
        fig, ax = plt.subplots()
        plt.title('1')
        cf = ax.contourf(X, Y, np.log10(masked_tau.T), 201)
        plt.plot([0, 1], [0.121478, 0.], '--', color='k')
        plt.plot([0.03, 0.03], [0, 1.], ':', color='C1')
        lim = np.linspace(np.log10(thresh_tau_ign_min),
                          np.log10(thresh_tau_ign_max), 2)
        print(lim)
        cb1 = plt.colorbar(cf, ticks=get_ticks(lim, n_label=n_labels_log))

        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        ax.tick_params(axis='x', pad=3)
        plt.xlim(0, 0.15)
        cb1.set_label(r'$\log_{10}(\tau_{\rm ig})$ [s]')
        plt.savefig('Figures/tau_ign_zoom.png')

    if 0:
        fig, ax = plt.subplots()
        plt.title('12')
        cf = ax.contourf(X, Y, np.log10(masked_tau.T), 201)
        # plt.title(r"$\tau_{\rm ig}$")
        plt.plot([0, 1], [0.121478, 0.], '--', color='k')
        plt.plot([0.03, 0.03], [0, 1.], ':', color='C1')

        # lim = np.log10([thresh_tau_ign_min, thresh_tau_ign_max])
        lim = np.linspace(np.log10(thresh_tau_ign_min),
                          np.log10(thresh_tau_ign_max), 2)
        print(lim)
        cb1 = plt.colorbar(cf, ticks=get_ticks(lim, n_label=n_labels_log))

        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        ax.tick_params(axis='x', pad=3)
        # plt.xlim(0, 0.15)
        cb1.set_label(r'$\log_{10}(\tau_{\rm ig})$ [s]')
        plt.savefig('Figures/tau_ign.png')

    # Plot T_Burnt
    if 0:
        # fig, ax = plt.subplots(figsize=(two_col_ctr / 3, two_col_ctr / 3.6))
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, T_burnt.T, 1001, vmin=300, vmax=2300)
        cb1 = plt.colorbar(cf, ticks=get_ticks(T_burnt, n_label=5))
        ax.tick_params(axis='x', pad=3)
        ax.set_xlabel("$Z$ [-]")
        # ax.set_ylabel(r"$C/C_{\rm max}$ [-]")
        # ax.set_yticklabels([])
        ax.tick_params(axis='x', pad=3)
        cb1.set_label(r'$T_{\rm eq}$ [K]')
        plt.plot([0, 1], [0.121478, 0.], '--', lw=1.0, color='k')
        plt.grid(False)
        ax.text(0.03, 0.03, r'(b)', color='k', fontsize=7)
        for ext in ['png']:
            plt.savefig("Figures/T_burnt.%s" % ext, transparent=False,
                        bbox_inches='tight',
                        pad_inches=0.01, dpi=1000)

    # Plot T0
    if 0:
        # fig, ax = plt.subplots(figsize=(two_col_ctr / 3, two_col_ctr / 3.6))
        fig, ax = plt.subplots()
        # cf = ax.contourf(X, Y, T_zero.T, 1001)
        cf = ax.contourf(X, Y, dict_ZCT['T'].T, 1001, vmin=300, vmax=2300)
        # plt.title(r"$T_0$")
        # cb1 = plt.colorbar(cf, ticks=get_ticks(dict_ZCT['T']))
        cb1 = plt.colorbar(cf, ticks=get_ticks(dict_ZCT['T'], n_label=5))
        cf = ax.contourf(X, Y, dict_ZCT['T'].T, 1001)
        # cb1 = plt.colorbar(cf, ticks=get_ticks(T_burnt))
        ax.tick_params(axis='x', pad=3)
        # plt.plot([0, 1], [0.121478, 0.], '--', lw=1.0, color='k')
        ax.set_xlabel("$Z$ [-]")
        ax.set_ylabel(r"$C/C_{\rm max}$ [-]")
        cb1.set_label(r'$T_{\rm ad}$ [K]')
        # ax.text(0.87, 0.89, r'a)', color='w')
        ax.text(0.03, 0.03, r'(a)', color='k', fontsize=7)
        plt.grid(False)
        for ext in ['png']:
            plt.savefig("Figures/T0.%s" % ext, transparent=False,
                        bbox_inches='tight',
                        pad_inches=0.01, dpi=1000)

    # Plot alpha
    if 1:
        plt.rcParams["image.cmap"] = "hot"
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, alpha.T, 100)
        levels = [0.05, None]
        CT = ax.contour(X, Y, alpha.T,
                        levels,
                        colors='white')
        # plt.clabel(CT)
        ax.annotate(r"$\alpha  = \frac{T_{f} - T_{0}}{T_{f}} = 0.05$", xy=(0.3, 0.56), xytext=(0.5, 0.8), color='white', fontsize=8,
                     arrowprops=dict(arrowstyle="->", linewidth=0.5))
        for cnt in cf.collections:
            cnt.set_edgecolor("face")

        # cf.set_rasterization_zorder(-10)

        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        ax.tick_params(axis='x', pad=3)
        cb1 = plt.colorbar(cf, ticks=get_ticks([0, 0.5]))
        cb1.set_label(r'$\alpha$ [-]')
        plt.tight_layout()
        for ext in list_ext:
            plt.savefig('Figures/CNF_heat_release_parameter.%s' % ext, transparent=True, bbox_inches='tight', pad_inches=0.01, dpi=300)

        plt.show()

    # Plot diffT
    if 0:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, diff_T.T, 1001)
        levels = [None, 0.05]
        CT = plt.contour(X, Y, alpha.T,
                         levels,
                         colors='white',
                         extend='both')
        # plt.clabel(CT)
        plt.title(r"$T_{\rm burnt} - T_0$")
        cb1 = plt.colorbar(cf, ticks=get_ticks(np.abs(diff_T)))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.tick_params(axis='x', pad=3)
        ax.set_ylabel("Normalized progress variable $C$ [-]")
        cb1.set_label('[K]')
        # plt.savefig('Figures/diff_T.png')
        for ext in list_ext:
            plt.savefig("Figures/diff_T.%s" % ext, transparent=False,
                        bbox_inches='tight',
                        pad_inches=0.01)

    # Plot gradZ
    if 0:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dtau_dZ.T, 1001)
        plt.title(r"$\partial \tau_{\rm ig} / \partial Z$")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dZ))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        cb1.set_label('[s]')
        plt.savefig('Figures/dtau_dZ.png')

    # Plot gradZ
    if 0:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dtau_dC.T, 1001)
        plt.title(r"$\partial \tau_{\rm ig} / \partial C$")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dC))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        cb1.set_label('[s]')
        plt.savefig('Figures/dtau_dC.png')

    # Plot dT_dZ
    if 0:
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dT_dZ.T, 201)
        plt.title(r"$\partial T / \partial Z$")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dT_dZ))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        cb1.set_label('[K]')
        plt.savefig('Figures/dT_dZ.png')

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dT_dC.T, 201)
        plt.title(r"$\partial T / \partial C$")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dT_dC))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        cb1.set_label('[K]')
        plt.savefig('Figures/dT_dC.png')

        fig, ax = plt.subplots()
        masked_inv_dT_dZ[0, :] = np.nan
        masked_inv_dT_dZ[99, :] = np.nan
        masked_inv_dT_dZ[:, 0] = np.nan
        masked_inv_dT_dZ[:, 99] = np.nan
        cf = ax.contourf(X, Y, masked_inv_dT_dZ.T, 201)
        plt.title(r"$(\partial T / \partial Z)^{-1}$ (masked inv)")
        # cb1 = plt.colorbar(cf, ticks=get_ticks(masked_inv_dT_dZ))
        cb1 = plt.colorbar(cf)
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        cb1.set_label('[K$^{-1}$]')
        plt.savefig('Figures/inv_dT_dZ.png')

        fig, ax = plt.subplots()
        # masked_inv_dT_dC = 1. / (dict_ZCT['T'] + 1e-7)

        masked_inv_dT_dC[0, :] = np.nan
        masked_inv_dT_dC[99, :] = np.nan
        masked_inv_dT_dC[:, 0] = np.nan
        masked_inv_dT_dC[:, 99] = np.nan

        # cf = ax.contourf(X, Y, dict_ZCT['T'].T, 1001)
        cf = ax.contourf(X, Y, masked_inv_dT_dC.T, 201)
        plt.title(r"$(\partial T / \partial C)^{-1}$ (masked inv)")
        # cb1 = plt.colorbar(cf, ticks=get_ticks(masked_inv_dT_dC))
        cb1 = plt.colorbar(cf)
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        cb1.set_label('[K$^{-1}$]')
        plt.savefig('Figures/inv_dT_dC.png')

    if 0:
        fig, ax = plt.subplots()
        dtau_dT = np.ma.masked_where(dtau_dT < -20., dtau_dT)
        cf = ax.contourf(X, Y, dtau_dT.T, 251)
        plt.title(r"$(\partial \tau_{\rm ig} / \partial T)$")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dT))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        cb1.set_label('[s.K$^{-1}$]')
        plt.savefig('Figures/dtau_dT.png')

        # Plot dY_OH_dZ
        if 0:
            fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dict_Yk[b'OH'].T + 1e-9, 201)
        plt.title(r"$Y_{\rm OH}$ [-]")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dict_Yk[b'OH']))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        # cb1.set_label('[K]')
        plt.savefig('Figures/Y_OH.png')

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dY_OH_dZ.T + 1e-9, 201)
        plt.title(r"$\partial Y_{\rm OH} / \partial Z$ [-]")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dY_OH_dZ))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        # cb1.set_label('[K]')
        plt.savefig('Figures/dY_OH_dZ.png')

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dY_OH_dC.T + 1e-9, 201)
        plt.title(r"$\partial Y_{\rm OH} / \partial C$ [-]")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dY_OH_dC))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        # cb1.set_label('[K]')
        plt.savefig('Figures/dY_OH_dC.png')

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, masked_inv_dY_OH_dZ.T + 1e-9, 201)
        plt.title(
            r"$\left(\partial Y_{\rm OH} / \partial Z \right)^{-1}$ [-]")
        cb1 = plt.colorbar(cf, ticks=get_ticks(masked_inv_dY_OH_dZ))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        # cb1.set_label('[K]')
        plt.savefig('Figures/inv_masked_dY_OH_dZ.png')

        fig, ax = plt.subplots()
        # cf = ax.contourf(X, Y, T_burnt.T, 201)
        # plt.title(r"T_Burnt")
        # cb1 = plt.colorbar(cf, ticks=get_ticks(T_burnt, n_label=11))
        # ax.set_xlabel("Mixture fraction $Z$ [-]")
        # ax.set_ylabel("C/Cmax")
        # cb1.set_label(r'[K]')
        cf = ax.contourf(X, Y, masked_inv_dY_OH_dC.T + 1e-9, 201)
        plt.title(
            r"$\left(\partial Y_{\rm OH} / \partial C \right)^{-1}$ [-]")
        cb1 = plt.colorbar(cf, ticks=get_ticks(masked_inv_dY_OH_dC))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        # cb1.set_label('[K]')
        plt.savefig('Figures/inv_masked_dY_OH_dC.png')

        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, dtau_dY_OH.T + 1e-9, 201)
        plt.title(r"$\partial \tau_{\rm ig}  / \partial Y_{\rm OH}$ [-]")
        cb1 = plt.colorbar(cf, ticks=get_ticks(dtau_dY_OH))
        ax.set_xlabel("Mixture fraction $Z$ [-]")
        ax.set_ylabel("C")
        # cb1.set_label('[K]')
        plt.savefig('Figures/dtau_dY_oh.png')

    plot_line = 0
    if plot_line:
        # Get results on mixing line
        _, tau_ign_xi = interpolate_result_on_mixing_line(tau_ign,
                                                          dict_ZCT['Z'],
                                                          dict_ZCT['C'],
                                                          C_coflow=0.121478)
        _, dtau_dZ_xi = interpolate_result_on_mixing_line(dtau_dZ,
                                                          dict_ZCT['Z'],
                                                          dict_ZCT['C'],
                                                          C_coflow=0.121478)
        _, dtau_dC_xi = interpolate_result_on_mixing_line(dtau_dC,
                                                          dict_ZCT['Z'],
                                                          dict_ZCT['C'],
                                                          C_coflow=0.121478)
        _, dtau_dT_xi = interpolate_result_on_mixing_line(dtau_dT,
                                                          dict_ZCT['Z'],
                                                          dict_ZCT['C'],
                                                          C_coflow=0.121478)
        plt.figure()
        plt.plot(dict_ZCT['Z'], dtau_dZ_xi, '-', lw=0.25)
        plt.plot([0.03, 0.03], [np.nanmin(dtau_dZ_xi), np.nanmax(dtau_dZ_xi)],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.xlabel(r"Mixture fraction $Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial Z$ [s]")
        plt.savefig('Figures/mixing_line_dTau_dZ.png')

        plt.figure()
        plt.plot(dict_ZCT['Z'], dtau_dC_xi, '-', lw=0.25)
        plt.plot([0.03, 0.03], [np.nanmin(dtau_dC_xi), np.nanmax(dtau_dC_xi)],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.xlabel(r"Mixture fraction $Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial C$ [s]")
        plt.savefig('Figures/mixing_line_dTau_dC.png')

        plt.figure()
        plt.plot(dict_ZCT['Z'], dtau_dT_xi, '-', lw=0.25)
        plt.plot([0.03, 0.03], [np.nanmin(dtau_dT_xi), np.nanmax(dtau_dT_xi)],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.xlabel(r"Mixture fraction $Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial T$ [s/K]")
        plt.savefig('Figures/mixing_line_dTau_dT.png')

        plt.figure()
        plt.semilogy(dict_ZCT['Z'], 1e3 * tau_ign_xi, '-', lw=0.25)

    if 0:
        plt.plot([0.03, 0.03],
                 [1e3 * np.nanmin(tau_ign_xi), 1e3 * np.nanmax(tau_ign_xi)],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\tau_{\rm ig}$ [ms]")
        plt.savefig('Figures/mixing_line_Tau.png')

        plt.figure()
        plt.plot(dict_ZCT['Z'][x < 0.04], dtau_dZ_xi[x < 0.04], '-', lw=0.25)
        plt.plot([0.03, 0.03],
                 [np.nanmin(dtau_dZ_xi[x < 0.04]),
                  np.nanmax(dtau_dZ_xi[x < 0.04])],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial Z$ [s]")
        plt.savefig('Figures/mixing_line_dTau_dZlean_.png')

        plt.figure()
        plt.plot(dict_ZCT['Z'][x < 0.04], dtau_dC_xi[x < 0.04], '-', lw=0.25)
        plt.plot([0.03, 0.03],
                 [np.nanmin(dtau_dC_xi[x < 0.04]),
                  np.nanmax(dtau_dC_xi[x < 0.04])],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial C$ [s]")
        plt.savefig('Figures/mixing_line_dTau_dClean_.png')

        plt.figure()
        plt.plot(dict_ZCT['Z'][x < 0.04], dtau_dT_xi[x < 0.04], '-', lw=0.25)
        plt.plot([0.03, 0.03],
                 [np.nanmin(dtau_dT_xi[x < 0.04]),
                  np.nanmax(dtau_dT_xi[x < 0.04])],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial T$ [s/K]")
        plt.savefig('Figures/mixing_line_dTau_dTlean_.png')

        plt.figure()
        plt.semilogy(dict_ZCT['Z'][x < 0.04], 1e3 * tau_ign_xi[x < 0.04], '-',
                     lw=0.25)
        plt.plot([0.03, 0.03], [1e3 * np.nanmin(tau_ign_xi[x < 0.04]),
                                1e3 * np.nanmax(tau_ign_xi[x < 0.04])], '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\tau_{\rm ig}$ [ms]")
        plt.savefig('Figures/mixing_line_Taulean_.png')

        plt.plot([0.03, 0.03],
                 [1e3 * np.nanmin(tau_ign_xi), 1e3 * np.nanmax(tau_ign_xi)],
                 '--',
                 lw=.5, color='C0',
                 label=r'$Z_{\rm st}$')
        plt.plot([0.0048, 0.0048],
                 [1e3 * np.nanmin(tau_ign_xi), 1e3 * np.nanmax(tau_ign_xi)],
                 '--',
                 lw=.5, color='C1',
                 label=r'$Z_{\rm mr}$')
        plt.xlabel(r"Mixture fraction $Z$ [-]")
        plt.ylabel(r"Autoignition delay time $\tau_{\rm ig}$ [ms]")
        plt.legend()
        plt.ylim(1e-1, 100)
        plt.xlim(0, 0.12)
        for ext in list_ext:
            plt.savefig("Figures/mixing_line_Tau.%s" % ext, transparent=True,
                        bbox_inches='tight',
                        pad_inches=0.01)

    # lean
    if 0:
        plt.figure()
        plt.plot(dict_ZCT['Z'][x < 0.04], dtau_dZ_xi[x < 0.04], '-', lw=0.25)
        plt.plot([0.03, 0.03],
                 [np.nanmin(dtau_dZ_xi[x < 0.04]),
                  np.nanmax(dtau_dZ_xi[x < 0.04])],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial Z$ [s]")
        plt.savefig('Figures/mixing_line_dTau_dZlean_.png')

        plt.figure()
        plt.plot(dict_ZCT['Z'][x < 0.04], dtau_dC_xi[x < 0.04], '-', lw=0.25)
        plt.plot([0.03, 0.03],
                 [np.nanmin(dtau_dC_xi[x < 0.04]),
                  np.nanmax(dtau_dC_xi[x < 0.04])],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial C$ [s]")
        plt.savefig('Figures/mixing_line_dTau_dClean_.png')

        plt.figure()
        plt.plot(dict_ZCT['Z'][x < 0.04], dtau_dT_xi[x < 0.04], '-', lw=0.25)
        plt.plot([0.03, 0.03],
                 [np.nanmin(dtau_dT_xi[x < 0.04]),
                  np.nanmax(dtau_dT_xi[x < 0.04])],
                 '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\partial\tau_{\rm ig} / \partial T$ [s/K]")
        plt.savefig('Figures/mixing_line_dTau_dTlean_.png')

        plt.figure()
        plt.semilogy(dict_ZCT['Z'][x < 0.04], 1e3 * tau_ign_xi[x < 0.04], '-',
                     lw=0.25)
        plt.plot([0.03, 0.03], [1e3 * np.nanmin(tau_ign_xi[x < 0.04]),
                                1e3 * np.nanmax(tau_ign_xi[x < 0.04])], '--')
        plt.xlabel(r"$Z$ [-]")
        plt.ylabel(r"$\tau_{\rm ig}$ [ms]")
        plt.savefig('Figures/mixing_line_Taulean_.png')

    plt.show()
