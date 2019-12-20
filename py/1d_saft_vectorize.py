# -*- coding: utf-8 -*-
from numba import vectorize
import numpy as np
from os.path import join
from time import perf_counter_ns
from misc.load_arr import load_arr, find_nearest
from misc.normalize_image import normalize
import matplotlib.pyplot as plt
DATA_FOLDER = "1D-3FOC50cm-60um"
directory_path = "C:\\Users\\indra\\Documents\\GitHub"
ARR_FOL = join(directory_path, DATA_FOLDER)
tarr, varr = load_arr("varr.pkl", ARR_FOL)
ZERO: int = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
V = np.copy(varr[ZERO:, :])  # ZERO'd & sample width
L = V.shape[1]  # scanning width (positions)
min_step = 6e-4
imgL: int = 0
imgR: int = L
SAMPLE_START: int = 31500
SAMPLE_END: int = 32700
Cw = 1498  # speed of sound in Water
Cm = 6320  # speed of sound in Metal
foc = 0.0762  # metres
var = float(np.std(np.arange(80)))

dY: int = SAMPLE_END - SAMPLE_START  # sample thickness
dX: int = imgR-imgL
a = Cm/Cw  # ratio between two speeds of sound
d2_start: int = SAMPLE_START - ZERO
d2_end: int = SAMPLE_END - ZERO
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
dstep_w = Cw*tstep/2
dstep_m = dstep_w*a
d1 = T[d2_start]*Cw/2.  # distance to sample
d2 = T[d2_start:d2_end]*Cw/2. - d1  # sample column (y distance)
d1 -= foc
lenT = len(T)  # length of time from ZERO to end of array
N = dY*dX
trans = np.arange(L)*min_step


@vectorize(['float64(int64)'], target='parallel')
def saft(c):
    i = c % dX  # x-coord of impix
    i += imgL
    j = c // dX  # y-coord of impixz
    aa = np.abs(trans - trans[i])
    z: float = d2[j] + d1
    dt = (2/Cw)*np.sqrt(aa[:]**2 + z**2) + 2*foc/Cw
    res = 0
    for k in range(L):
        t = int(np.round(dt[k]/tstep))  # delayed t (indices)
        w = np.exp((-1/2)*(i-k)**2/(var**2))
        if t < d2_end:
#            d = (abs(V[t, k]) + abs(V[t-1, k]) +
#                 abs(V[t+1, k]) + abs(V[t-2, k]) +
#                 abs(V[t+2, k]) + abs(V[t+3, k]) +
#                 abs(V[t-3, k]))
#            d = abs(V[t, k])
            res += float(V[t, k])
    return res


def plt_saft():
    start_time = perf_counter_ns()*1e-9
    impix = np.arange(N)
    p = saft(impix)
    p = normalize(p, float(np.min(p)), float(np.max(p)))
    POST = p.reshape((dY, dX), order='C')
    duration = perf_counter_ns()*1e-9-start_time
    print("Summing and plotting took {} s".format(duration))
    plt.figure(figsize=[10, 10])
    plt.imshow(POST, aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} SAFT".format(DATA_FOLDER))
    plt.savefig(join(ARR_FOL, "saft.png"), dpi=400)
    plt.show()
    return POST


if __name__ == '__main__':
    POST_saft = plt_saft()
