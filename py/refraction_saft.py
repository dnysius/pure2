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
import scipy.optimize as so
import pickle
import matplotlib.pyplot as plt
global min_step, Cw, Cm, a, DEFAULT_ARR_FOLDER, d1
global lat_distance, FD, lenT, T, V, L, POST, lat_index
################################################################
FOLDER_NAME = "1D-3FOC50cm-60um"
directory_path = "C:\\Users\\dionysius\\Documents\\pure repo\\data\\1D SCANS"
min_step = 4e-4
FOCAL_DEPTH = 0.0381*2
SAMPLE_START = 31500
SAMPLE_THICKNESS = .014
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


def f(z, aa, d2):
    # z represents sin(theta_1)
    # aa lateral distances from imaging pixel to transducer
    # a*z is sin(theta_2)
    return ((d1*z)/np.sqrt(1-z**2)) + ((d2*a*z)/np.sqrt(1-(a*z)**2)) - aa


def df(z, aa, d2):
    # derivative of our function
    first = d1*(np.sqrt(1-z**2)+z**2*(1-z**2)**(-1/2))/np.sqrt(1-z**2)
    sec = d2*a*(np.sqrt(1-(a*z)**2)+a*z**2*(1-(a*z)**2)**(-1/2))/(1-(a*z)**2)*np.sqrt(1-z**2)
    return (first+sec-aa)


def newton(deltax, d2):
    accuracy = 1e-9
    error = 1.
    x1 = np.zeros(len(deltax))
    # newton's method
    while error > accuracy:
        x1, x0 = x1 - f(x1, deltax, d2)/df(x1, deltax, d2), x1
        error = np.max(np.abs(x1-x0))
    return x1


# set up the data
tarr, varr = load_arr()
tarr = tarr[:, 0, :]
varr = varr[:, 0, :]
ZERO = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
varr = varr[ZERO:, :]
tstep = np.mean(T[1:]-T[:-1])  # average timestep
# refraction algorithm
d1 = T[SAMPLE_START]*Cw/2  # distance
d2_start = SAMPLE_START  # index
d2_end = find_nearest(T, 2*(d1+SAMPLE_THICKNESS)/Cw)  # index
d2 = T[d2_start:d2_end]*Cw/2
V = varr[:, :]  # 2D
L = np.shape(V)[1]  # number of transducer positions
lat_distance = np.linspace(-L/2, L/2, L)*min_step
lat_index = np.arange(0, L, 1)
lenT = len(T)
lend2 = d2_end-d2_start
POST = np.empty((lend2, L))  # final image array


def main(lat_pix):  # lat_pix is imaging x coord
    j = 0  # imaging pixel for time/ z direction
    while j < lend2:
        aa = lat_distance[lat_pix] - lat_distance[:]
        zw = newton(aa, d2[j])  # sin(theta_1)
        zm = zw*a  # find sin(theta_2)
        radM = np.arcsin(zm)  # theta_2
        radW = np.arcsin(zw)  # theta_1
        delay_t = (2/Cw)*(d1/np.cos(radW) + d2[j]/np.cos(radM))
        zi = np.round(delay_t/tstep).astype(int)
        POST[j, lat_pix] = np_sum(V[zi[zi < lenT], lat_index[zi < lenT]])
        j += 1


if __name__ == '__main__':
    # Parallel processing
    start_time = perf_counter_ns()*1e-9
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
    b = np.abs(hilbert(POST[:, :], axis=0))
#    pickle.dump(b, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-{}.pkl"
#                             .format(FOLDER_NAME)), "wb"))
#    pickle.dump(T, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-T-{}.pkl"), "wb"))
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)

#fig = plt.figure(figsize=[10, 10])
#plt.imshow(POST[:, :], aspect='auto', cmap='gray')
#plt.colorbar()
#plt.show()


#tp = 250  # the number of transducer positions; lateral direction
#d2 = np.linspace(0., 1.3, 100)/100  # discretize the sample thickness at sample depth
#lat_distance = np.linspace(-5., 5., tp)/100  # 250 positions for -5cm to 5cm scan
