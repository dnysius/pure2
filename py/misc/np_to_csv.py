# -*- coding: utf-8 -*-
import numpy as np
from os.path import join


def to_csv(path):
    with open(join(path, "varr.pkl"), 'rb') as rd:
        arr = np.load(rd)
    t, one, sig = np.shape(arr)
    csv = np.empty(shape=(t, sig), dtype=float)
    csv[:, :] = arr[:, 0, :]
    with open(join(path, "voltages.csv"), 'w+') as cs:
        np.savetxt(cs, csv, delimiter=',')


if __name__ == '__main__':
#    pass
    directory_path = "C:\\Users\\dionysius\\Documents\\pure repo\\data\\1D SCANS"
    FOLDER_NAME = "1D-3FOC50cm-60um"
    to_csv(join(directory_path, FOLDER_NAME))
