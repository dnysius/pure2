# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from os import getcwd
from os.path import join, dirname
from misc.load_arr import load_arr, find_nearest
global graph_title
FOLDER_NAME = "1D-FLAT5in"  # edit this
graph_title = "FLAT3in"  # and this
path_folder = join(dirname(getcwd()), "data", "1D SCANS", FOLDER_NAME)
Cw = 1498
Cm = 6320

tarr, varr = load_arr("varr.pkl", path_folder)
vmax = np.max(varr.flatten())
vmin = np.min(varr.flatten())
ZERO: int = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
lenT = len(T)  # length of time from ZERO to end of array
V = np.copy(varr[ZERO:, :])  # ZERO'd & sample width


def image(varr):
    fig, ax = plt.subplots(figsize=[14, 10])
    plt.imshow(varr, aspect='auto', cmap='gray')
    plt.colorbar()
    plt.show(fig)


def rf_time(t, v):
    fig, ax = plt.subplots(figsize=[18, 12])
    plt.plot(t, v)
    plt.ylabel("voltage (V)")
    plt.xlabel("time (s)")
    plt.title(graph_title)
    plt.ylim(vmin-1, vmax+1)
    plt.show(fig)


def rf_ind(v):
    fig, ax = plt.subplots(figsize=[18, 12])
    plt.plot(v, color='goldenrod', label='raw')
    plt.ylabel("voltage (V)")
    plt.xlabel("index")
    plt.title(graph_title)
    plt.ylim(vmin-1, vmax+1)
    plt.show(fig)


def rf_ind_hil(v):
    fig, ax = plt.subplots(figsize=[18, 12])
    plt.plot(np.imag(hilbert(v)), c='gray', label='Re(hilbert)', ls=':')
    plt.plot(v, color='goldenrod', label='raw')
    plt.ylabel("voltage (V)")
    plt.xlabel("index")
    plt.title(graph_title)
    plt.legend()
    plt.ylim(vmin-1, vmax+1)
    plt.show(fig)


if __name__ == '__main__':
    start = 0
    end = -1
#    start = 14500
#    end = 17500
    transducer_pos = 150
    image(varr)
#    rf_ind(varr[start:end, transducer_pos])
#    rf_ind_hil(varr[start:end, transducer_pos])
#    rf_time(tarr[start:end, transducer_pos], varr[start:end, transducer_pos])
