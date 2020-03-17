import numpy as np
import h5py

import matplotlib.pyplot as plt

import pickle


def read_table_nodes(read_file_name):
    table = h5py.File(read_file_name, 'r')

    # Get path and length of dims
    path_Z = table['Coordinates']['Coor_0'].name
    len_Z = table['Coordinates']['Coor_0'].shape
    len_Z = int(len_Z[0])

    print(path_Z, len_Z)

    path_Z_var = table['Coordinates']['Coor_1'].name
    len_Z_var = table['Coordinates']['Coor_1'].shape
    len_Z_var = int(len_Z_var[0])

    print(path_Z_var, len_Z_var)

    path_C = table['Coordinates']['Coor_2'].name
    len_C = table['Coordinates']['Coor_2'].shape
    len_C = int(len_C[0])

    print(path_C, len_C)

    print(table['Header'].keys())
    print(table['Header']['Doubles'].keys())
    print(table['Header']['Number of dimensions'][0])
    n_dim = table['Header']['Number of dimensions'][0]
    print(table['Header']['Number of variables'][0])
    n_vars = table['Header']['Number of variables'][0]
    print(table['Header']['Strings'].keys())
    print(table['Header']['Strings']['String_0'][0:10])
    print(table['Header']['Strings']['String_1'][0:10])
    print(table['Header']['Variable Names'][0:100])

    name_var = table['Header']['Variable Names'][0:n_vars]

    print(name_var)

    len_data = table['Data'].shape
    len_data = int(len_data[0])
    print(len_data)

    data = table['Data'][0:len_data]

    data_r = data.reshape(n_vars, len_data // n_vars)

    idx_var = 1

    T_dataset = np.zeros((len_Z, len_Z_var, len_C))

    offset = idx_var * (len_Z + len_Z_var + len_C) - 1
    count = 0

    for idx_z in range(len_Z):
        for idx_z_var in range(len_Z_var):
            for idx_c in range(len_C):
                T_dataset[idx_z, idx_z_var, idx_c] = data_r[idx_var, count]
                count += 1

    x = table[path_Z][0:len_Z]
    y = table[path_C][0:len_C]

    X, Y = np.meshgrid(x, y)
    Z = T_dataset[:, 0, :]
    print(X.shape)
    Z.shape

    # Plot temperature
    contours = plt.contourf(X, Y, Z.T, 200, color='black');
    plt.xlabel("Z")
    plt.ylabel("C")
    plt.title(r'Temperature [K]')
    plt.xlabel("Z")
    plt.ylabel("C")
    plt.colorbar();
    plt.savefig('Figures/read_temperature_from_FPVA.png')
    plt.show()

    dict_data = {}
    name_var = table['Header']['Variable Names'][0:n_vars]

    for idx_var, data_name in enumerate(name_var):

        dataset = np.zeros((len_Z, len_Z_var, len_C))

        offset = idx_var * (len_Z + len_Z_var + len_C) - 1
        count = 0

        for idx_z in range(len_Z):
            for idx_z_var in range(len_Z_var):
                for idx_c in range(len_C):
                    dataset[idx_z, idx_z_var, idx_c] = data_r[idx_var, count]
                    count += 1
        dict_data[data_name] = dataset

    print(dict_data[b'T0'].shape)

    dict_Z_var0 = {}

    for key in dict_data.keys():
        dict_Z_var0[key] = dict_data[key][:, 0, :]

    # In[39]:

    dict_Z_var0[b'T0'].shape

    # In[40]:

    dict_Yk = {}

    for key in dict_Z_var0.keys():
        use_data = True

        for filt in ['SRC_', '0', 'GAMMA', 'LOC', 'PROG', 'HeatRelease', 'ROM',
                     'HOT', 'COLD', 'ZBilger', 'AMU']:
            if filt in key.decode():
                use_data = False

        if use_data:
            print(key.decode())
            dict_Yk[key] = dict_Z_var0[key]

    len(dict_Yk)

    # get zct dict
    dict_ZCT = {}
    dict_ZCT['T'] = dict_Z_var0[b'T0']
    dict_ZCT['Z'] = table[path_Z][0:len_Z]
    dict_ZCT['C'] = table[path_C][0:len_C]

    # output pickle of dictionnary

    with open('Data/Dict_Yk.pickle', 'wb') as handle:
        pickle.dump(dict_Yk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Data/Dict_ZCT.pickle', 'wb') as handle:
        pickle.dump(dict_ZCT, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_ZC_Zvar0(table, dict_data_Zvar0, var_name):
        x = table[path_Z][0:len_Z]
        y = table[path_C][0:len_C]

        X, Y = np.meshgrid(x, y)
        Z = dict_data_Zvar0[var_name]
        print(X.shape)
        Z.shape

        contours = plt.contourf(X, Y, Z.T, 500, color='black', vmin=np.amin(Z),
                                vmax=np.amax(Z));
        plt.xlabel("Z")
        plt.ylabel("C")
        plt.title(r'%s' % var_name.decode())
        plt.colorbar();
        plt.savefig('Figures/read_%s_from_FPVA.png' % var_name.decode())

    plot_ZC_Zvar0(table, dict_Z_var0, b'T0')
    plt.show()


if __name__ == "__main__":
    read_table_nodes("Data/CH4_O2_p1_tf290_to1490_GRI_PEC_AddSrc_Ohstar_500x2x500_renamedSpec.h5")
