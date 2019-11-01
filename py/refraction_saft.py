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
from time import perf_counter_ns
from scipy.optimize import fsolve
import pickle
import matplotlib.pyplot as plt
global min_step, Cw, Cm, a, DEFAULT_ARR_FOLDER, d1
global lat_distance, FD, lenT, T, V, L, POST, lat_index, arr
arr = []
################################################################
FOLDER_NAME = "1D-3FOC5in"
directory_path = "C:\\Users\\dionysius\\Documents\\pure repo\\data\\1D SCANS"
min_step = 4e-4
FOCAL_DEPTH = 0.0381*2
SAMPLE_START = 15750
SAMPLE_END = 17750
SAMPLE_THICKNESS = .01
################################################################
Cw = 1498  # speed of sound in (w)ater
Cm = 6320  # speed of sound in (m)etal
a = Cm/Cw  # ratio between two speeds of sound
#DEFAULT_ARR_FOLDER = join(dirname(getcwd()), "data", "1D SCANS", FOLDER_NAME)
DEFAULT_ARR_FOLDER = join(directory_path, FOLDER_NAME)


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


def df(theta, d2):
    # derivative with tan(theta)
    first = d1/(np.square(np.cos(theta)))
    second = d2/(1-np.square(a*np.sin(theta)))
    return first + second


def newton(aa, d2):
    accuracy = 1e-9
    error = 1.
    x1 = np.zeros(len(aa))
    # newton's method
    while error > accuracy:
        x1, x0 = x1 - f(x1, aa, d2)/df(x1, d2), np.copy(x1)
        error = np.max(np.abs(x1-x0))
    return x1


# set up the data
tarr, varr = load_arr()
tarr = tarr[:, 0, :]
varr = varr[:, 0, :]
ZERO = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
varr = varr[ZERO:, 50:150]
tstep = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
# refraction algorithm
d1 = T[SAMPLE_START]*Cw/2  # distance
d2_start = SAMPLE_START  # index
d2_end = SAMPLE_END
d2 = T[d2_start:d2_end]*Cw/2 - d1
V = varr[:, :]  # 2D
L = np.shape(V)[1]  # number of transducer positions
lat_distance = np.linspace(-L/2, L/2, L)*min_step
lat_index = np.arange(0, L, 1)
lenT = len(T)
lend2 = d2_end-d2_start
POST = np.empty((lend2, L))  # final image array


def f(theta, aa, d2):
    first = d1*np.tan(theta)
    sec = (d2*a*np.sin(theta))/np.sqrt(1-np.square(a*np.sin(theta)))
    return first + sec - aa


def main(lat_pix):  # lat_pix is imaging x coord
    j = 0  # imaging pixel for time/ z direction
    zw = np.empty(L)
    while j < 1:
        aa = -1*lat_distance + lat_distance[lat_pix]
        k = 0
        while k in range(L):
            zw[k] = fsolve(f, lat_distance[-1], args=(aa[k], d2[j]))  # sin(theta_1)
            k += 1
        print(f(zw, aa, d2[j]))
        zm = zw*a  # find sin(theta_2)
        radM = np.arcsin(zm)  # theta_2
        radW = np.arcsin(zw)  # theta_1
        delay_t = np.abs((2/Cw)*(d1/np.cos(radW) + d2[j]/np.cos(radM)))
        zi = np.round(delay_t/tstep).astype(int)
        POST[j, lat_pix] = np_sum(V[zi[zi < lenT], lat_index[zi < lenT]])
        j += 1
        return zw


if __name__ == '__main__':
    # Parallel processing
    zi = main(0)
#    start_time = perf_counter_ns()*1e-9
#    jobs = []
#    print("Append")
#    for i in range(1):
#        jobs.append(threading.Thread(target=main, args=(i,)))
#    print("Starting")
#    for job in jobs:
#        job.start()
#    print("Joining")
#    for job in jobs:
#        job.join()
#    print("Stitching")
#    V[d2_start:d2_end, 50:150] = POST[:, 50:150]
#    b = np.abs(hilbert(V[:, :], axis=0))
#    pickle.dump(b, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-{}.pkl"
#                             .format(FOLDER_NAME)), "wb"))
#    pickle.dump(T, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-T-{}.pkl"), "wb"))
#    duration = perf_counter_ns()*1e-9-start_time
#    print(duration)
#
#fig = plt.figure(figsize=[10, 10])
#plt.imshow(POST[:, :], aspect='auto', cmap='gray')
#plt.colorbar()
#plt.show()
