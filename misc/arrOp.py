# -*- coding: utf-8 -*-
"""
Array Operations
"""
import numpy as np
from path import Path
import os.path as op
import os
from numba import vectorize


@vectorize(['float64(float64, float64, float64)'], target='parallel')
def normalize(p, pmin, pmax):
    if p < 0:
        return p / abs(pmin)
    elif p > 0:
        return p / abs(pmax)
    else:
        return 0


def load_arr(folder, FILENAME="varr.npy"):
    ftarr = Path(folder)/"tarr.npy"
    fvarr = Path(folder)/FILENAME
    with open(ftarr, 'rb') as rd:
        tarr = np.load(rd, allow_pickle=False)
        tarr = tarr[:, :]
    with open(fvarr, 'rb') as rd:
        varr = np.load(rd, allow_pickle=False)
        if FILENAME == "varr.npy":
            varr = varr[:, :]
    return tarr, varr


def find_nearest(array, value):
    array = np.asarray(array, dtype=float)
    return np.abs(array - value).argmin()


def np_to_csv(path):
    p = Path(path)
    with open(p/"varr.npy", 'rb') as rd:
        varr = np.load(rd, allow_pickle=False)
    with open(p/"voltages.csv", 'w+') as wr:
        np.savetxt(wr, varr, delimiter=',')


def pkl_to_csv(path):
    p = Path(path)
    with open(p/"tarr.npy", 'rb') as rd:
        tarr = np.load(rd, allow_pickle=False)
    with open(p/"varr.npy", 'rb') as rd:
        varr = np.load(rd, allow_pickle=False)
    np.savetxt(p/"tarr.csv", tarr, delimiter=",")
    np.savetxt(p/"varr.csv", varr, delimiter=",")


def pkl_to_npy(path):
    data_path = Path("C:/Users/indra/Documents/GitHub")
    subfolders = [d for d in os.listdir(data_path)
                  if op.isdir(data_path/d) and d != "pure2"]
    for folder in subfolders:
        subp = data_path/folder
    with open(subp/"tarr.pkl", 'rb') as rd:
        tarr = np.load(rd, allow_pickle=True)
        if len(tarr.shape) == 3 and tarr.shape[1] == 1:
            tarr = tarr[:, 0, :]
        elif len(tarr.shape) == 3 and tarr.shape[2] == 1:
            tarr = tarr[:, :, 0]
        np.save(subp/"tarr.npy", tarr, allow_pickle=False)
    with open(subp/"varr.pkl", 'rb') as rd:
        varr =  np.load(rd, allow_pickle=True)
        if len(varr.shape) == 3 and varr.shape[1] == 1:
            varr = varr[:, 0, :]
        elif len(varr.shape) == 3 and varr.shape[2] == 1:
            varr = varr[:, :, 0]
        np.save(subp/"varr.npy", varr, allow_pickle=False)
