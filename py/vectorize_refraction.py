# -*- coding: utf-8 -*-
'''
1. can we make a function that takes (ndarray, int32, int32)?
since we need solver(k, j, i)
2.
'''
from numba import vectorize, int64, complex64, complex128, float64, float32
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
SAMPLE_END: int = 33000
#SAMPLE_END: int = SAMPLE_START + 600
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
d2 = T[d2_start:d2_end]*Cw/2. - d1  # sample grid (y distance)
d1 -= foc

L = varr.shape[1]  # number of transducer positions
#dY: int = d2_end-d2_start  # sample thickness
dX = imgR-imgL
dY = SAMPLE_END-SAMPLE_START
lenT = len(T)  # length of time from ZERO to end of array
#IMAGE = np.empty((dY, dX))  # empty final image array
trans_index = np.arange(0, dX, 1)  # transducer positions indices
N = dY*dX
trans = np.linspace(-dX/2, dX/2, dX)*min_step


@vectorize(['float64(int64)'], target='parallel')
def refr(c):  # lat_pix is imaging x coord
    i = c % dX  # x-coord of impix
    j = c // dX  # y-coord of impix
    dt = np.zeros(dX)  # delayed time: imaging pixel to transducer position
    aa = np.abs(trans - trans[i])
    res = 0
    for k in range(dX):
        if d2[j] != 0 and aa[k] != 0:
            P4 = aa[k]**2
            P3 = -2*d1*aa[k]
            P2 = (aa[k]**2-aa[k]**2*a**2+d1**2-a**2*d2[j]**2)
            P1 = 2*d1*aa[k]*(a**2-1)
            P0 = d1**2*(1-a**2)
            pol = np.array([P4, P3, P2, P1, P0])
            sol = np.roots(pol)  # theta 1
            root = 0
            for s in sol:
                q = complex(s)
                if abs(q.imag) < 1e-5 and s >= root:  # find max in sol
                    root = abs(q)
            y0 = np.sqrt(np.square(root) + 1)
            stheta1 = 1./y0
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


@vectorize(['float64(int64)'], target='parallel')
def saft(c):
    # c is the impix coordinate
    # use saft for outside sample range (when img pixel is not supposed to
    # be in the metal)
    i = c % dX  # x-coord of impix
    j = c // dX  # y-coord of impixz
#    dt = np.zeros(dX)  # delayed time: imaging pixel to transducer
    aa = np.abs(trans - trans[i])
    z: float = d2[j] + d1
    dt = (2/Cw)*np.sqrt(aa[:]**2 + z**2) + 2*foc/Cw
    res = 0
    for k in range(len(dt)):
        t = int(np.round(dt[k]/tstep))  # delayed t (indices)
        if t < lenT:
            res += V[t, k]
#    return np.sum(V[zi[zi < lenT], trans_index[zi < lenT]])
    return res


if __name__ == '__main__':
    start_time = perf_counter_ns()*1e-9
    impix = np.arange(N)
#    p = saft(impix)
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
    plt.title("{} combined".format(FOLDER_NAME))
    plt.show()

