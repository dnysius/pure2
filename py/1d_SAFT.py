# -*- coding: utf-8 -*-
import numpy as np
import threading
from numpy import \
    power as np_power, \
    sqrt as np_sqrt, \
    sum as np_sum
from os import getcwd
from os.path import join, dirname
from scipy.signal import hilbert
import pickle
import matplotlib.pyplot as plt
global min_step, c_0, DEFAULT_ARR_FOLDER
global xarr, FD, SD, pbar, T, V, L, T_COMPARE, PRE_OUT, POST_OUT, xni

FOLDER_NAME = "1D-3FOC3in-PURE2"  # edit this
min_step = 4e-4  # and this
FOCAL_DEPTH = 0.0381*2  # and this
c_0 = 1498  # speed of sound in water

if FOLDER_NAME[:2] == "1D":
    par = "1D SCANS"
else:
    par = "2D SCANS"
DEFAULT_ARR_FOLDER = join(dirname(getcwd()), "data", par, FOLDER_NAME)


def load_arr(output_folder=DEFAULT_ARR_FOLDER):
    ftarr = join(output_folder, "tarr.pkl")
    fvarr = join(output_folder, "varr.pkl")
    with open(ftarr, 'rb') as rd:
        tarr = pickle.load(rd)
    with open(fvarr, 'rb') as rd:
        varr = pickle.load(rd)
    return tarr, varr


def find_nearest(array, value):
    array = np.asarray(array, dtype=float)
    return (np.abs(array - value)).argmin()


tarr, varr = load_arr()
tarr = tarr[:, 0, :]
varr = varr[:, 0, :]
ZERO = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
V = varr[ZERO:, :]  # 2D
FD = find_nearest(T, 2*FOCAL_DEPTH/c_0)  # focal depth
SD = len(T)
OFFSET = T[FD]
L = np.shape(V)[1]
PRE = np.flip(V[:FD, :], axis=0)
PRE_T = np.flip(T[:FD], axis=0)
PRE_OUT = np.empty(np.shape(PRE))
POST = V[FD:SD, :]
POST_T = T[FD:SD]
POST_OUT = np.empty(np.shape(POST))
xarr = np.linspace(-L/2, L/2, L)*min_step
xni = np.arange(0, L, 1)
tstep = np.mean(T[1:]-T[:-1])


def main(xi):  # xi is imaging index/position
    x = xarr[xi]  # x is imaging DISTANCE
    ti = 0  # imaging pixel for time/ z direction
    while ti < SD:
        z = T[ti]*c_0/2
        z2 = np_power(z, 2)  # distance, squared
        ind = (2/c_0)*np_sqrt(np_power(x-xarr[xni], 2)
                              + z2)
        zi = (ind/tstep).astype(int)
        if ti < FD:  # PRE
            PRE_OUT[ti, xi] = np_sum(V[zi[zi < FD], xni[zi < FD]])
        if ti >= FD:  # POST
            POST_OUT[ti-FD, xi] = np_sum(V[zi[zi < SD], xni[zi < SD]])
        ti += 1


if __name__ == '__main__':
    # Parallel processing
    jobs = []
    print("Append")
    for i in range(L):
        jobs.append(threading.Thread(target=main, args=(i,)))
    print("Starting")
    for job in jobs:
        job.start()
    print("Joining")
    for job in jobs:
        job.join()
    print("Stitching")
    PRE_OUT = np.flip(PRE_OUT, axis=0)
    b = np.abs(hilbert(POST_OUT[:, :], axis=0))
    b = 20*np.log10(b/np.max(b.flatten()))
    STITCHED = np.vstack((PRE_OUT, b))
    pickle.dump(STITCHED, open(join(DEFAULT_ARR_FOLDER,
                                    "SAFT-{}.pkl"
                                    .format(FOLDER_NAME)), "wb"))


fig = plt.figure(figsize=[10, 10])
plt.imshow(STITCHED[:, :], aspect='auto', cmap='hot')
plt.colorbar()
plt.show()
