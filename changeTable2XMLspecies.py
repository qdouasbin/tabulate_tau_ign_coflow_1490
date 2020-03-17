import h5py
import numpy as np

if __name__ == "__main__":
    read_file_name = "Data/CH4_O2_p1_tf290_to1490_GRI_PEC_AddSrc_Ohstar_500x2x500_renameSpec.h5"

    h5_pec = h5py.File(read_file_name, 'r+')

    for idx, name in enumerate(h5_pec['Header']['Variable Names']):
        print('Not changing this field: %s' % name)
        if name == b'CH2GSG-CH2':
            print('Changing this field: %s' % name)
            h5_pec['Header']['Variable Names'][idx] = b'CH2(S)'
        if name == b'OHD-OH':
            print('Changing this  field: %s --> OH*' % name)
            h5_pec['Header']['Variable Names'][idx] = b'OH*'
        if name == 'C':
            print('Changing this  field: %s --> Ca' % name)
            h5_pec['Header']['Variable Names'][idx] = 'Ca'

    print(h5_pec['Header']['Variable Names'])
    h5_pec.close()

