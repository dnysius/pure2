# -*- coding: utf-8 -*-
import numpy as np
import threading
from numpy import \
    power as np_power, \
    sqrt as np_sqrt, \
    sum as np_sum
from os import getcwd
from os.path import join, dirname
import pickle
import matplotlib.pyplot as plt
global min_step, c_0, DEFAULT_ARR_FOLDER
global xarr, yarr, FD, SD, T, V, PRE_OUT, POST_OUT, zi
FOLDER_NAME = "2D-3FOC5in-pipe"
DEFAULT_ARR_FOLDER = join(dirname(dirname(getcwd())), FOLDER_NAME)
FOCAL_DEPTH = 0.0381*2  # 1.5 inch in metres
min_step = 4e-4
c_0 = 1498  # water


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
ZERO = find_nearest(tarr[:, 0, 0], 0)
T = tarr[ZERO:, 0, 0]  # 1D, time columns all the same
V = varr[ZERO:, :, :]  # 3D
tstep = np.mean(T[1:]-T[:-1])
FD = find_nearest(T, 2*FOCAL_DEPTH/c_0)  # focal depth
SD = len(T)-1
LY = np.shape(V)[1]
LX = np.shape(V)[2]
PRE = np.flip(V[:FD, :, :], axis=0)
PRE_T = np.flip(T[:FD], axis=0)
PRE_OUT = np.empty(np.shape(PRE))
POST = V[FD:SD, :, :]
POST_T = T[FD:SD]
POST_OUT = np.empty(np.shape(POST))
xarr = np.linspace(0, LX, LX)*min_step
xni = np.arange(0, LX, 1)
yarr = np.linspace(0, LY, LY)*min_step
yni = np.arange(0, LY, 1)
xx, yy = np.meshgrid(xni, yni)
tni = np.arange(0, SD, 1)
ind = np.arange(0, LX*LY, LX)
yind = np.empty((LX*LY,), dtype=int)
xind = np.tile(np.arange(0, LX, 1), LY)
for i in range(LY):
    yind[ind[i]:] = i


def main(yi, xi):
    x = xarr[xi]
    y = yarr[yi]
    ti = 0
    while ti < SD:
        z2 = np_power(T[ti]*c_0/2, 2)
        zi = ((2/c_0)*np_sqrt(np_power(x-xarr[xx.flatten('C')], 2)
                              + np_power(y-yarr[yy.flatten('C')], 2)
                              + z2)/tstep).astype(int)
        if ti < FD:
            PRE_OUT[ti, yi, xi] = np_sum(V[zi[zi < FD],
                                         yind[zi < FD], xind[zi < FD]])
        elif ti >= FD:
            POST_OUT[ti-FD, yi, xi] = np_sum(V[zi[zi < SD],
                                             yind[zi < SD], xind[zi < SD]])
        ti += 1


if __name__ == '__main__':
    jobs = []
    print("Append")
    for y in range(LY):
        for x in range(LX):
            jobs.append(threading.Thread(target=main, args=(y, x)))
    print("Starting")
    i = 0
    for job in jobs:
        if i % 10 == 0:
            print("Starting job ", i)
        job.start()
        i += 1
    i = 0
    print("Joining")
    for job in jobs:
        if i % 10 == 20:
            print("Joining job ", i)
        job.join()
    PRE_OUT = np.flip(PRE_OUT, axis=0)
    STITCHED = np.vstack((PRE_OUT, POST_OUT))
    pickle.dump(STITCHED, open(join(DEFAULT_ARR_FOLDER,
                               "SAFT-{}-3D.pkl".format(FOLDER_NAME)), "wb"))
    print("\nDone")
