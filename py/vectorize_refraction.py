# -*- coding: utf-8 -*-
from numba import vectorize
import numpy as np
from tqdm import tqdm
from os.path import join
from time import perf_counter_ns
from misc.load_arr import load_arr, find_nearest
from misc.normalize_image import normalize
import matplotlib.pyplot as plt
DATA_FOLDER: str = "1D-3FOC50cm-60um"
directory_path: str = "C:/Users/indra/Documents/GitHub"
ARR_FOL = join(directory_path, DATA_FOLDER)
tarr, varr = load_arr("varr.pkl", output_folder=ARR_FOL)

min_step = 6e-4
foc = 0.0762  # metres
Cw = 1498  # speed of sound in Water
Cm = 6320  # speed of sound in Metal
SAMPLE_START: int = 31500
SAMPLE_END: int = 32700
imgL: int = 40
imgR: int = 180

a = Cm/Cw  # ratio between two speeds of sound
ZERO: int = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
lenT = len(T)  # length of time from ZERO to end of array
V = np.copy(varr[ZERO:, :])  # ZERO'd & sample width
L = V.shape[1]  # scanning width (number of transducer positions)
dY = SAMPLE_END - SAMPLE_START  # sample thickness
dX = imgR-imgL
d2_start: int = SAMPLE_START - ZERO
d2_end: int = SAMPLE_END - ZERO
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
dstep: float = tstep*Cw/2
d1 = T[d2_start]*Cw/2.  # distance to sample
d2 = a*(T[d2_start:d2_end]*Cw/2. - d1)  # sample column (y distance)
d1 -= foc
N = dY*dX
trans = np.arange(L)*min_step
aa = trans - trans[0]
td_arr = np.empty((dY, L))


def determine_total_distance(j):
    dt = np.zeros(L)
    for k in range(L):
        d2j = d2[j]
        aak = aa[k]
        if j == 0:
            t = (2/Cw)*(np.sqrt(d1**2 + aak**2) + foc)
            dt[k] = int(np.round(t/tstep))
        elif j != 0 and k == 0:
            t = 2*(d1/Cw + d2j/Cm + foc/Cw)
            dt[k] = int(np.round(t/tstep))
        elif k != 0 and j != 0:
            P4 = aak**2
            P3 = -2*d1*aak
            P2 = (aak**2-aak**2*a**2+d1**2-a**2*d2j**2)
            P1 = 2*d1*aak*(a**2-1)
            P0 = d1**2*(1-a**2)
            roots = np.roots([P4, P3, P2, P1, P0])  # theta 1
            roots = np.abs(roots[np.imag(roots) < 1e-7])
            if len(roots) != 0:
                y0 = np.sqrt(np.square(roots) + 1)
                stheta1 = 1./y0
                st1 = np.abs(stheta1[np.abs(stheta1) <= 1/a])
                if len(st1) > 1:
                    rad1 = np.arcsin(st1)  # theta_1
                    rad2 = np.arcsin(st1*a)  # theta_2
                    d = d1/Cw/np.cos(rad1) + d2j/Cm/np.cos(rad2)
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
    return dt[:]


def create_td_arr():
    start_time = perf_counter_ns()*1e-9
    for j in tqdm(range(dY), desc='Creating td_arr row: '):
        td_arr[j, :] = determine_total_distance(j)
    duration = perf_counter_ns()*1e-9-start_time
#    np.save(join(ARR_FOL, 'td_arr.npy'), td_arr, allow_pickle=False)
    print("Creating td_arr took {} s".format(duration))
    return td_arr


def load_td_arr():
    start_time = perf_counter_ns()*1e-9
    td = np.load(join(ARR_FOL, 'td_arr.npy'))
    duration = perf_counter_ns()*1e-9-start_time
    print("Loading td_arr took {} s".format(duration))
    return td


#td_arr = create_td_arr()
td_arr = load_td_arr()
#var = np.std(np.arange(100))


@vectorize(['float64(int64)'], target='parallel')
def refr(c):
    i = int(c % dX)  # x-coord of impix
    j = int(c // dX)  # y-coord of impix
    res = 0
    for k in range(imgL, imgR):
        m = abs(i + imgL - k)
        t = int(td_arr[j, m])  # delayed t index
#        w = np.exp((-1/2)*m**2/(var)**2)
        if t < d2_end:
#            d = abs(float(V[t, k]*w))
            d = float(V[t, k])
            res += d
    return res


def plt_refr():
    start_time = perf_counter_ns()*1e-9
    impix = np.arange(N)
    p = refr(impix)
    p = normalize(p, float(np.min(p)), float(np.max(p)))
    POST = p.reshape((dY, dX), order='C')
    duration = perf_counter_ns()*1e-9-start_time
    print("Summation and plotting took {} s".format(duration))
    plt.figure(figsize=[10, 10])
    plt.imshow(POST, aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} refraction".format(DATA_FOLDER))
#    plt.savefig(join(ARR_FOL, 'refraction.png'), dpi=600)
    plt.show()
    plt.figure(figsize=[10, 10])
    vmin = np.min(V[d2_start:d2_end, imgL:imgR].flatten())
    vmax = np.max(V[d2_start:d2_end, imgL:imgR].flatten())
    view = normalize(V[d2_start:d2_end, imgL:imgR].flatten(),
                     float(vmin), float(vmax))
    view = view.reshape((dY, dX), order='C')
    plt.imshow(view, aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} B-Scan".format(DATA_FOLDER))
#    plt.savefig(join(ARR_FOL, 'b-scan.png'), dpi=600)
    plt.show()
    return POST


if __name__ == '__main__':
    POST_refr = plt_refr()
