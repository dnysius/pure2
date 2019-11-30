import numpy as np
import threading
from os import getcwd
from os.path import join, dirname
from scipy.signal import hilbert
from time import perf_counter_ns
import pickle
import matplotlib.pyplot as plt
global min_step, Cw, Cm, a, ARR_FOL, d1
global transducer_positions, lenT, T, V, L, IMAGE, trans_index, foc
FOLDER_NAME = "1D-3FOC50cm-60um"
directory_path = "C:\\Users\\indra\\Documents\\GitHub"
min_step = 6e-4
SAMPLE_START: int = 31500
SAMPLE_END: int = 33000
LEFT: int = 0
RIGHT: int = 175
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
V = np.copy(varr[ZERO:, LEFT:RIGHT])  # ZERO'd & sample width
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
d1 = T[d2_start]*Cw/2.  # distance to sample
d2 = T[d2_start:d2_end]*Cw/2. - d1  # sample grid (y distance)
d1 -= foc
L = varr.shape[1]  # number of transducer positions
lenT = len(T)  # length of time from ZERO to end of array
lend2: int = d2_end-d2_start  # sample thickness
IMAGE = np.empty((lend2, RIGHT-LEFT))  # empty final image array
transducer_positions = np.arange(L)*min_step
trans_index = np.arange(0, L, 1)  # transducer positions indices

# refraction
def refraction(j):  # lat_pix is imaging x coord
    k: int = 0
    dt = np.zeros(L)  # delayed time: imaging pixel to transducer
    aa = transducer_positions - transducer_positions[0]
    if j == 0:
        dt = (2/Cw)*np.sqrt(aa[:]**2 + d1**2) + 2*foc/Cw
    else:
        while k < L:
            if d2[j] != 0:
                P4 = aa[k]**2
                P3 = -2*d1*aa[k]
                P2 = (aa[k]**2-aa[k]**2*a**2+d1**2-a**2*d2[j]**2)
                P1 = 2*d1*aa[k]*(a**2-1)
                P0 = d1**2*(1-a**2)
                roots = np.roots([P4, P3, P2, P1, P0])  # theta 1
                roots = np.real(roots[np.isreal(roots)])
                if roots.size != 0:
                    y0 = np.sqrt(np.square(roots) + 1)
                    stheta1 = 1./y0
                    st1 = stheta1[np.abs(stheta1) <= 1/a]
                    if st1.size > 0:
                        stheta1 = np.max(np.abs(st1))
                        rad1 = np.arcsin(stheta1)  # theta_1
                        rad2 = np.arcsin(stheta1*a)  # theta_2
                        dt[k]: float = 2*(np.abs(d1/Cw/np.cos(rad1))
                                          + np.abs(d2[j]/Cm/np.cos(rad2))
                                          + foc/Cw)
                k += 1
    zi = np.round(dt/tstep).astype(int)  # delayed t (indices)
    return zi


def saft(j):
    dt = np.zeros(L)  # delayed time: imaging pixel to transducer
    aa = transducer_positions - transducer_positions[0]
    z: float = d2[j] + d1
    dt: float = (2/Cw)*np.sqrt(aa[:]**2 + z**2) + 2*foc/Cw
    zi = np.round(dt/tstep).astype(int)  # delayed t (indices)
    return zi

if __name__ == '__main__':
    start_time = perf_counter_ns()*1e-9
    dstep = tstep*Cw/2
    zi_saft = saft(1)
    zi_refr = refraction(1)
    zi = (zi_refr-zi_saft)[(zi_saft>0)*(zi_refr>0)]  # refr>saft
    print(zi_saft)
    duration = perf_counter_ns()*1e-9-start_time
