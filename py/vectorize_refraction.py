# -*- coding: utf-8 -*-
'''
1. can we make a function that takes (ndarray, int32, int32)?
since we need solver(k, j, i)
2.
'''
from numba import vectorize
from numba import njit
import numpy as np
from os import getcwd
from os.path import join, dirname
from time import perf_counter_ns
import pickle
import matplotlib.pyplot as plt
global min_step, Cw, Cm, a, ARR_FOL, d1
global trans, lenT, T, V, dX, IMAGE, trans_index, foc
FOLDER_NAME = "1D-3FOC50cm-60um"
directory_path = "C:\\Users\\indra\\Documents\\GitHub"
min_step = 6e-4
#SAMPLE_START: int = 31500
#SAMPLE_END: int = 33000
#SAMPLE_END: int = SAMPLE_START + 600
#imgL: int = 50
#imgR: int = 140
SAMPLE_START: int = 31500
#SAMPLE_END: int = 33000
SAMPLE_END: int = SAMPLE_START + 2
imgL: int = 0
imgR: int = 200
Cw = 1498  # speed of sound in Water
Cm = 6320  # speed of sound in Metal
a = Cm/Cw  # ratio between two speeds of sound
foc = 0.0762  # metres
ARR_FOL = join(directory_path, FOLDER_NAME)


def load_arr(FILENAME, output_folder=ARR_FOL):
    ftarr = join(output_folder, "tarr.pkl")
    fvarr = join(output_folder, FILENAME)
    with open(ftarr, 'rb') as rd:
        tarr = pickle.load(rd)
        tarr = tarr[:, 0, :]
    with open(fvarr, 'rb') as rd:
        varr = pickle.load(rd)
        if FILENAME == "varr.pkl":
            varr = varr[:, 0, :]
    return tarr, varr


def find_nearest(array, value):
    array = np.asarray(array, dtype=float)
    return np.abs(array - value).argmin()


# import data
tarr, varr = load_arr("varr.pkl")
ZERO: int = find_nearest(tarr[:, 0], 0)
d2_start: int = SAMPLE_START - ZERO
d2_end: int = SAMPLE_END - ZERO
T = tarr[ZERO:, 0]  # 1D, time columns all the same
V = np.copy(varr[ZERO:, :])  # ZERO'd & sample width
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
d1 = T[d2_start]*Cw/2.  # distance to sample
d2 = T[d2_start:d2_end]*Cw/2. - d1  # sample column (y distance)
d1 -= foc

L = varr.shape[1]  # scanning width (positions)
#imgR = L
dY: int = d2_end-d2_start  # sample thickness
dX = imgR-imgL
lenT = len(T)  # length of time from ZERO to end of array
N = dY*dX
trans = np.linspace(-dX/2, dX/2, dX)*min_step
crit_angle = np.arcsin(1/a)
theta1s = np.linspace(0, crit_angle, int(1e5), dtype=np.float32)
stheta2s = a*np.sin(theta1s)
stheta2s[stheta2s >= 1] = 1
theta2s = np.arcsin(stheta2s)


#@vectorize(['float64(int64)'], target='parallel')
@njit()
def refr():  # lat_pix is imaging x coord
    i = c % dX  # x-coord of impix
    j = c // dX  # y-coord of impix
    dt = np.zeros(dX)  # delayed time: imaging pixel to transducer position
    aa = np.abs(trans - trans[i])
    res = 0
    for k in range(dX):
        if d2[j] != 0 and aa[k] != 0:
            stheta1 = np.abs(d1*np.tan(theta1s)
                             + d2[j]*np.tan(theta2s)
                             - aa[k]).argmin()
            if stheta1 <= 1/a:
                rad1 = np.arcsin(stheta1)  # theta_1
                rad2 = np.arcsin(stheta1*a)  # theta_2
                dt[k]: float = 2*(np.abs(d1/Cw/np.cos(rad1))
                                  + np.abs(d2[j]/Cm/np.cos(rad2))
                                  + foc/Cw)
        elif aa[k] == 0 and d2[j] != 0:
            dt[k] = 2*(d1/Cw + d2[j]/Cm + foc/Cw)
        elif aa[k] == 0 and d2[j] == 0:
            dt[k] = 2*(d1/Cw + foc/Cw)
        t = int(np.round(dt[k]/tstep))  # delayed t (indices)
        if t < lenT:
            res += V[t, k]
    return res


def plt_refr():
    start_time = perf_counter_ns()*1e-9
    impix = np.arange(N)
    p = refr(impix)
    POST = p.reshape((dY, dX))
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)
    plt.figure(figsize=[10, 10])
    plt.imshow(POST, aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} combined".format(FOLDER_NAME))
    plt.show()
    plt.figure(figsize=[10, 10])
    plt.imshow(V[d2_start:d2_end, imgL:imgR], aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} vectorized refraction ".format(FOLDER_NAME))
    plt.show()


if __name__ == '__main__':
    plt_refr()
