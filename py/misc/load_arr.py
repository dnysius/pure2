# -*- coding: utf-8 -*-
import numpy as np
from os.path import join


def load_arr(FILENAME, output_folder):
    ftarr = join(output_folder, "tarr.pkl")
    fvarr = join(output_folder, FILENAME)
    with open(ftarr, 'rb') as rd:
        tarr = np.load(rd, allow_pickle=True)
        tarr = tarr[:, 0, :]
    with open(fvarr, 'rb') as rd:
        varr = np.load(rd, allow_pickle=True)
        if FILENAME == "varr.pkl":
            varr = varr[:, 0, :]
    return tarr, varr


def find_nearest(array, value):
    array = np.asarray(array, dtype=float)
    return np.abs(array - value).argmin()
