# -*- coding: utf-8 -*-
'''
1. can we make a function that takes (ndarray, int32, int32)?
since we need solver(k, j, i)
2.
'''
from numba import vectorize
import numpy as np
from tqdm import tqdm
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
#imgL: int = 0
#imgR: int = 150
SAMPLE_START: int = 31500
SAMPLE_END: int = 33000
#SAMPLE_END: int = SAMPLE_START + 300
imgL: int = 0
imgR: int = 150
Cw = 1498  # speed of sound in Water
Cm = 6320  # speed of sound in Metal
#Cm = Cw
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
d2 = d2*a
d1 -= foc

dY: int = d2_end-d2_start  # sample thickness
L = varr.shape[1]  # scanning width (positions)
imgR = 200
dX = imgR-imgL
lenT = len(T)  # length of time from ZERO to end of array
N = dY*dX
trans = np.arange(L)*min_step
aa = trans - trans[0]
td_arr = np.empty((dY, L))


def determine_total_distance(j):
    # given impix coordinate (leftmost), generate array (corresponding
    # to transducers from one impix) with total travelling distance
    # j: impix d2 height in sample, x-coord is at index 0
    dt = np.zeros(L)
    for k in range(L):
        d2j = d2[j]
        aak = aa[k]
#        if j == 0:
#            t = (2/Cw)*(np.sqrt(d1**2 + aak**2) + foc)
#            dt[k] = int(np.round(t/tstep))
        if j == 0:
            j = 1
            d2j = d2[j]
        if j != 0 and k == 0:
            t = 2*(d1/Cw + d2j/Cm + foc/Cw)
            dt[k] = int(np.round(t/tstep))
        elif k != 0 and j != 0:
            P4 = aak**2
            P3 = -2*d1*aak
            P2 = (aak**2-aak**2*a**2+d1**2-a**2*d2j**2)
            P1 = 2*d1*aak*(a**2-1)
            P0 = d1**2*(1-a**2)
            saft_angle = np.abs(np.arctan(aak/d1))
            roots = np.roots([P4, P3, P2, P1, P0])  # theta 1
            roots = np.abs(roots[np.imag(roots) < 1e-7])
            if len(roots) != 0:
                y0 = np.sqrt(np.square(roots) + 1)
                stheta1 = 1./y0
                st1 = np.abs(stheta1[np.abs(stheta1) <= 1/a])
                if len(st1) > 1:
#                    stheta1 = st1[np.abs(saft_angle - st1).argmin()]
#                    stheta1 = np.abs(np.min(st1))
                    rad1 = np.arcsin(st1)  # theta_1
                    rad2 = np.arcsin(st1*a)  # theta_2
                    d = np.abs(d1/Cw/np.cos(rad1)) + np.abs(d2j/Cm/np.cos(rad2))
                    rad1 = st1[d.argmin()]
                    rad2 = np.arcsin(np.sin(rad1)*a)  # theta_2
                    dt[k] = int(np.round(2*(np.abs(d1/Cw/np.cos(rad1))
                                         + np.abs(d2j/Cm/np.cos(rad2))
                                         + foc/Cw)/tstep))
                elif len(st1) == 1:
                    stheta1 = st1[0]
                    rad1 = np.arcsin(stheta1)  # theta_1
                    rad2 = np.arcsin(stheta1*a)  # theta_2
                    dt[k] = int(np.round(2*(np.abs(d1/Cw/np.cos(rad1))
                                         + np.abs(d2j/Cm/np.cos(rad2))
                                         + foc/Cw)/tstep))
                elif len(st1) == 0:
                    roots = []
    return dt[:]


def create_td_arr():
    start_time = perf_counter_ns()*1e-9
    for j in tqdm(range(dY), desc='Creating td_arr row: '):
        td_arr[j, :] = determine_total_distance(j)
    duration = perf_counter_ns()*1e-9-start_time
#    np.save(join(ARR_FOL, 'td_arr.npy'), td_arr, allow_pickle=False)
    print(duration)
    return td_arr


def load_td_arr():
    start_time = perf_counter_ns()*1e-9
    td = np.load(join(ARR_FOL, 'td_arr.npy'))
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)
    return td


#td_arr = create_td_arr()
td_arr = load_td_arr()
var = np.std(np.arange(200))


@vectorize(['float64(int64)'], target='parallel')
def refr(c):
    i = int(c % dX)  # x-coord of impix
    j = int(c // dX)  # y-coord of impix
    res = 0
    for k in range(L):
        m = abs(i + imgL - k)
        t = int(td_arr[j, m])  # delayed t index
        w = np.exp((-1/2)*m**2/(var)**2)
        if t < d2_end:
            d = abs(float(V[t, k]*w))
            res += d
    return res


def plt_refr():
    start_time = perf_counter_ns()*1e-9
    impix = np.arange(N)
    p = refr(impix)
    POST = p.reshape((dY, dX), order='C')
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)
    plt.figure(figsize=[10, 10])
    plt.imshow(POST, aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} refraction".format(FOLDER_NAME))
#    plt.savefig(join(ARR_FOL, 'refraction.png'), dpi=600)
    plt.show()
    plt.figure(figsize=[10, 10])
    plt.imshow(V[d2_start:d2_end, imgL:imgR], aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} B-Scan ".format(FOLDER_NAME))
#    plt.savefig(join(ARR_FOL, 'b-scan.png'), dpi=600)
    plt.show()
    return POST


if __name__ == '__main__':
    POST = plt_refr()
#    pass
