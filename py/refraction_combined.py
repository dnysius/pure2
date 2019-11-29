# -*- coding: utf-8 -*-
"""
Ensure that the parameters initialized after library imports are correct and
also look at the end of the program to check that the filenames for the
final ndarrays to be saved are correct.

Currently we use `threading` to speed up the program. We observed major
improvements in program runtime with the original SAFT algorithm (~10 min).
Due to the function solving in the refraction algorithm, we cannot vectorize
the algorithm entirely, and must resort to for loops. However, since the
processing of each imaging pixel is independent to another imaging pixel,
we should look into multiprocessing libraries in combination with a
sufficiently-powered computer. One example is `mpi4py`.
"""
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
L = varr.shape[1]  # number of transducer positions
lenT = len(T)  # length of time from ZERO to end of array
lend2: int = d2_end-d2_start  # sample thickness
IMAGE = np.empty((lend2, RIGHT-LEFT))  # empty final image array
transducer_positions = np.linspace(-L/2, L/2, L)*min_step
trans_index = np.arange(0, L, 1)  # transducer positions indices


def main(lat_pix: int) -> None:  # lat_pix is imaging x coord
    j: int = 0  # imaging pixel (time axis)
    dt = np.zeros(L)  # delayed time from imaging pixel to transducer position
    while j < lend2:
        aa = np.abs(transducer_positions - transducer_positions[lat_pix])
        k: int = 0  # angles b/n transducer IMAGEion and imaging pixel
        while k < L:
            P4 = aa[k]**2
            P3 = -2*d1*aa[k]
            P2 = (aa[k]**2-aa[k]**2*a**2+d1**2-a**2*d2[j]**2)
            P1 = 2*d1*aa[k]*(a**2-1)
            P0 = d1**2*(1-a**2)
            roots = np.roots([P4, P3, P2, P1, P0])  # theta 1
            roots = np.real(roots[np.isreal(roots)])
#            SAFT = False
            if roots.size != 0:
                y0 = np.sqrt(np.square(roots) + 1)
                stheta1 = 1./y0
                st1 = stheta1[np.abs(stheta1) <= 1/a]
                if st1.size > 0:
                    stheta1 = np.max(np.abs(st1))
                    rad1 = np.arcsin(stheta1)  # theta_1
                    rad2 = np.arcsin(stheta1*a)  # theta_2
                    dt[k]: float = 2*(np.abs(d1/Cw/np.cos(rad1))
                                      + np.abs(d2[j]/Cm/np.cos(rad2)))
#                else:
#                    SAFT = True
            # if no roots found, calculate delay like SAFT
#            if (roots.size == 0) or (SAFT is True):
#                z: float = d2[j] + d1 - foc
#                dt[k]: float = (2/Cw)*np.sqrt(aa[k]**2 + z**2) + 2*foc/Cw
            k += 1
        zi = np.round(dt/tstep).astype(int)  # delayed t (indices)
        IMAGE[j, lat_pix] = np.sum(varr[zi[zi < lenT],
                                   trans_index[zi < lenT]])
        j += 1
    return None


if __name__ == '__main__':
    # Parallel processing
    start_time = perf_counter_ns()*1e-9
    jobs = []
    print("Append")
    for i in range(LEFT, RIGHT, 1):
        jobs.append(threading.Thread(target=main, args=(i,)))
    print("Starting")
    for job in jobs:
        job.start()
    print("Joining")
    for job in jobs:
        job.join()
    print("Stitching")
    V[d2_start:d2_end, :] = IMAGE[:, :]
#    varr[SAMPLE_START:SAMPLE_END, LEFT:RIGHT] = IMAGE[:, :]
#    V_filtered = np.abs(hilbert(V[:, :], axis=0))
    V_path = open(join(ARR_FOL, "comb-thread-V-{}.pkl"
                       .format(FOLDER_NAME)), "wb")
    T_path = open(join(ARR_FOL, "comb-thread-T-{}.pkl"
                       .format(FOLDER_NAME)), "wb")
    V_npy_path = open(join(ARR_FOL, "comb-thread-fullview-{}.npy"
                      .format(FOLDER_NAME)), "wb")
    pickle.dump(V, V_path)
    pickle.dump(T, T_path)
    np.save(V_npy_path, varr, allow_pickle=False)
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)
    fig = plt.figure(figsize=[10, 10])
    plt.imshow(varr[SAMPLE_START:SAMPLE_END, LEFT:RIGHT], aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{}
    b-scan".format(FOLDER_NAME))
    plt.show()
    fig = plt.figure(figsize=[10, 10])
    plt.imshow(V[d2_start:d2_end, :], aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title("{} combined".format(FOLDER_NAME))
    plt.show()
