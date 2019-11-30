# -*- coding: utf-8 -*-
import numpy as np
import threading
from os import getcwd
from os.path import join, dirname
from scipy.signal import hilbert
from time import perf_counter_ns
import pickle
import matplotlib.pyplot as plt
global min_step, Cw, Cm, a, DEFAULT_ARR_FOLDER, d1
global lat_distance, FD, lenT, T, V, L, POST, lat_index, foc
################################################################
FOLDER_NAME = "1D-3FOC50cm-60um"
directory_path = "C:\\Users\\dionysius\\Documents\\pure repo\\data\\1D SCANS"
min_step = 6e-4
SAMPLE_START = 31500
SAMPLE_END = 33000
LEFT = 0
RIGHT = 175
################################################################
Cw = 1498  # speed of sound in (w)ater
Cm = 6320  # speed of sound in (m)etal
a = Cm/Cw  # ratio between two speeds of sound
foc = 0.0762  # metres
#DEFAULT_ARR_FOLDER = join(dirname(getcwd()), "data", "1D SCANS", FOLDER_NAME)
DEFAULT_ARR_FOLDER = join(directory_path, FOLDER_NAME)


def load_arr(FILENAME, output_folder=DEFAULT_ARR_FOLDER):
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
    return (np.abs(array - value)).argmin()


# set up the data
tarr, varr = load_arr("varr.pkl")
ZERO = find_nearest(tarr[:, 0], 0)
d2_start = SAMPLE_START - ZERO
d2_end = SAMPLE_END - ZERO
T = tarr[ZERO:, 0]  # 1D, time columns all the same
varr = varr[ZERO:, LEFT:RIGHT]  # ZERO'd & sample width
tstep = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
# refraction algorithm
d1 = T[d2_start]*Cw/2  # distance
d2 = T[d2_start:d2_end]*Cw/2 - d1
V = varr[:, :]  # 2D
L = np.shape(V)[1]  # number of transducer positions
lenT = len(T)
lend2 = d2_end-d2_start
POST = np.empty((lend2, L))  # final image array
lat_distance = np.linspace(-L/2, L/2, L)*min_step
lat_index = np.arange(0, L, 1)
crit_angle = np.arcsin(1/a)
thetas = np.linspace(0, crit_angle, int(1e4))


def approxf(x, aa, d2):
    # the nonlinear function, solving for theta_1
    z = a*np.sin(x)
    z[z >= 1] = int(1.)
    y = z + np.power(z, 3)/6+3*np.power(z, 5)/40
    first = d1*np.tan(x)
    sec = d2*(y+np.power(y, 3)/3+2*np.power(y, 5)/15)
    return first + sec - aa


def main(lat_pix):  # lat_pix is imaging x coord
    j = 0  # imaging pixel for time/ z direction
    zw = np.empty(L)  # left right, represents theta1 to transducer from impix
    while j < lend2:
        aa = np.abs(-1*lat_distance + lat_distance[lat_pix])
        k = 0
        while k < L:
            # sin(theta_1)
            zw[k] = thetas[np.abs(approxf(thetas, aa[k], d2[j])).argmin()]
            k += 1
        zw = np.sin(zw)
        zw[zw >= 1/a] = 1/a
        zm = zw*a  # find sin(theta_2)
        radM = np.arcsin(zm)  # theta_2
        radW = np.arcsin(zw)  # theta_1
        # different speed of sound for each term
        delay_t = np.abs(2*(d1/Cw/np.cos(radW) + d2[j]/Cm/np.cos(radM)))
        zi = np.abs(np.round(delay_t/tstep).astype(int))
        POST[j, lat_pix] = np.sum(V[zi[zi < lenT], lat_index[zi < lenT]])
        j += 1
    return POST


if __name__ == '__main__':
    # Parallel processing
#    pass
#    zi = main(0)
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
    V[d2_start:d2_end, LEFT:RIGHT] = POST[:, :]
#    b = np.abs(hilbert(V[:, :], axis=0))
    pickle.dump(V, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-{}.pkl"
                             .format(FOLDER_NAME)), "wb"))
    pickle.dump(T, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-T.pkl"), "wb"))
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)


fig = plt.figure(figsize=[10, 10])
plt.imshow(V[d2_start:d2_end, LEFT:RIGHT], aspect='auto', cmap='gray')
plt.colorbar()
plt.title("{} refraction".format(FOLDER_NAME))
plt.show()
