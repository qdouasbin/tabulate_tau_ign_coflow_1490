import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle


def add_derivative_to_table(read_file_name):

    table = h5py.File(read_file_name, 'r+')

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

    n_dim = table['Header']['Number of dimensions'][0]
    n_vars = table['Header']['Number of variables'][0]
    name_var = table['Header']['Variable Names'][0:n_vars]
    print(n_dim)
    print(n_vars)
    print(name_var)

    print(type(table['Header']['Variable Names'][0]))
    print(table['Header']['Variable Names'])

    len_data = table['Data'].shape
    len_data = int(len_data[0])
    print(len_data)

    with open('Data/Dict_Yk.pickle', 'rb') as handle:
        dict_Yk = pickle.load(handle)

    with open('Data/Dict_ZCT.pickle', 'rb') as handle:
        dict_ZCT = pickle.load(handle)

    with open('Data/Dict_res.pickle', 'rb') as handle:
        dict_res = pickle.load(handle)

    with open('Data/Dict_gradients.pickle', 'rb') as handle:
        dict_gradients = pickle.load(handle)

    n_add_var = len(dict_gradients.keys())
    print(n_add_var, dict_gradients.keys())

    shape_1, shape_2 = dict_gradients[b'dtau_dZ'].shape
    dict_gradients[b'dtau_dZ'].shape

    ## Find out the dimension of the new data table
    len_data / (len_Z * len_Z_var * len_C)

    ## First change the number of variables
    del table['Header']['Number of variables']

    dset = table.create_dataset(b"Header/Number of variables",
                                data=[n_vars + n_add_var])

    # Add names of variables
    list_new_names = []

    for name in table['Header']['Variable Names'][0:n_vars]:
        list_new_names.append(name)

    for name in dict_gradients.keys():
        list_new_names.append(name.upper())

    print(list_new_names)

    print(type(list_new_names[0]))


    # Replace the data by the new one
    def printname(name):
        print(name)


    table.visit(printname)
    print()

    try:
        del table['Header']['Variable Names']
    except KeyError:
        print("\tDataset is already deleted")

    try:
        del table['MyRefs']
    except KeyError:
        print("\tDataset is already deleted")

    # CharclesX requires the following :
    # STRSIZE H5T_VARIABLE     --> need to use ASCII for variable length
    # STRPAD H5T_STR_SPACEPAD  --> need to do something here!
    # CSET H5T_CSET_ASCII      --> OK
    # CTYPE H5T_C_S1           --> OK

    # Define Variable-length ASCII type
    dt = h5py.special_dtype(vlen=bytes)

    dset = table.create_dataset("Header/Variable Names", (len(list_new_names),),
                                dtype=dt)

    print(table['Header/Strings/String_0'][1])
    print(table['Header/Doubles/Double_1'][0])

    arr_var_names = np.array(list_new_names, dtype=dt)
    arr_var_names

    for idx, name in enumerate(arr_var_names):
        table['Header']['Variable Names'][idx] = name

    for idx, name in enumerate(arr_var_names):
        print(table['Header']['Variable Names'][idx])

    # ## Populate de "Data" dataset

    # Create new data
    n_new_vars = table['Header']['Number of variables'][0]
    new_data = np.zeros((n_new_vars * len_Z * len_Z_var * len_C))
    print(new_data.shape)

    # Populate the new data array
    if 1:
        # First do the good old data
        for idx, data in enumerate(table['Data']):
            if idx < len_data:
                new_data[idx] = data

        # Compute the offset and set counter to zero
        count = 0
        if 1:
            for idx_var, key in enumerate(dict_gradients.keys()):
                print("\tvar %s" % key)
                offset = len_data
                for idx_z in range(len_Z):
                    for idx_z_var in range(len_Z_var):
                        for idx_c in range(len_C):
                            new_data[offset + count] = dict_gradients[key][
                                idx_z, idx_c]
                            count += 1
    print(offset + count)
    print(count)

    print(len_data + 4 * 100 * 2 * 100)

    len_Z, len_Z_var, len_C, len_data


    # Replace the data by the new one
    def printname(name):
        print(name)


    table.visit(printname)

    del table[b'Data']
    dset = table.create_dataset(b"Data", data=new_data)


    def printname(name):
        print(name)


    print(table.visit(printname))

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

    n_dim = table['Header']['Number of dimensions'][0]
    n_vars = table['Header']['Number of variables'][0]
    name_var = table['Header']['Variable Names'][0:n_vars]
    print(n_dim)
    print(n_vars)
    print(name_var)

    table.close()

    print("Done")

if __name__ == "__main__":
    print(h5py.__version__)

    read_file_name = "Data/CH4_O2_p1_tf290_to1490_GRI_PEC_AddSrc_Ohstar_500x2x500_AddDerivativesNoCropBelowMixingLine.h5"

    add_derivative_to_table(read_file_name)
