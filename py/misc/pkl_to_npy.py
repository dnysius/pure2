# -*- coding: utf-8 -*-
from os import getcwd, listdir
from os.path import join, dirname
import numpy as np
import pickle
FOLDER_NAME = "1D-15FOC5in"
SCAN_FOLDER = join(dirname(dirname(getcwd())), "data", "1D SCANS", FOLDER_NAME)


def load_save_arr(folder, filename):
    with open(join(folder, filename), 'rb') as rd:
        arr = pickle.load(rd)
    with open(join(SCAN_FOLDER, "{0}npy".format(filename[:-3])), 'wb') as wr:
        np.savez_compressed(wr, arr, allow_pickle=False)


def iterate_directory(folder=SCAN_FOLDER):
    for f in listdir(SCAN_FOLDER):
        if f[-3:]=="pkl":
            load_save_arr(SCAN_FOLDER, f)
            
if __name__=='__main__':
    iterate_directory(SCAN_FOLDER)