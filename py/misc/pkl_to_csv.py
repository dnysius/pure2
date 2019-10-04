# -*- coding: utf-8 -*-
import pickle
from os import getcwd
from os.path import join, dirname
import numpy as np

def pkl_to_csv(mypath):
    tarr = join(mypath, "tarr.pkl")
    varr = join(mypath, "varr.pkl")
    with open(tarr, 'rb') as rd:
        tarr = pickle.load(rd)
    with open(varr, 'rb') as rd:
        varr = pickle.load(rd)
    np.savetxt(join(mypath, "tarr.csv"), tarr[:, 0, :], delimiter=",")
    np.savetxt(join(mypath, "varr.csv"), varr[:, 0, :], delimiter=",")
    

if __name__ == '__main__':
    path = join(dirname(getcwd()), "data", "1D-3FOC3in")
    pkl_to_csv(path)