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
thetas = np.linspace(-np.pi/2, np.pi/2, int(2e5))


def approxf(x, aa, d2):
    z = a*(x-np.power(x, 3)/(3*2)+np.power(x, 5)/(5*4*3*2))
    z = a*np.sin(x)
    z[z >= 1] = int(1.)
    z[np.isnan(z)] = int(1.)
    y = z + np.power(z, 3)/6+3*np.power(z, 5)/40
#    first = d1*(x+np.power(x, 3)/3+2*np.power(x, 5)/15)
    first = d1*np.tan(x)
    sec = d2*(y+np.power(y, 3)/3+2*np.power(y, 5)/15)
    return first + sec - aa


def arcapprox(x):
    return x+x**3/6+3*x**5/40


def main(lat_pix):  # lat_pix is imaging x coord
    j = 0  # imaging pixel for time/ z direction
    zw = np.empty(L)  # left right, represents theta1 to transducer from impix
    while j < lend2:
        aa = -1*lat_distance + lat_distance[lat_pix]
        k = 0
        while k < L:
            # sin(theta_1)
            zw[k] = thetas[np.abs(approxf(thetas, aa[k], d2[j])).argmin()]
            if np.isnan(zw[k]):
                zw[k] = np.pi/2
            k += 1
        zm = np.sin(zw)*a  # find sin(theta_2)
        zm[zm > 1] = 1
        zm[np.isnan(zm)] = 1
        radM = arcapprox(zm)  # theta_2
        radW = zw  # theta_1
        delay_t = np.abs((2/Cw)*(d1/np.cos(radW) + d2[j]/np.cos(radM)))
        zi = np.round(delay_t/tstep).astype(int)
        POST[j, lat_pix] = np.sum(V[zi[zi < lenT], lat_index[zi < lenT]])
        j += 1
    return zi


if __name__ == '__main__':
    # Parallel processing
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
    V[d2_start:d2_end, 50:150] = POST[:, 50:150]
    b = np.abs(hilbert(V[:, :], axis=0))
    pickle.dump(b, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-{}.pkl"
                             .format(FOLDER_NAME)), "wb"))
    pickle.dump(T, open(join(DEFAULT_ARR_FOLDER, "refraction-SAFT-T-{}.pkl"), "wb"))
    duration = perf_counter_ns()*1e-9-start_time
    print(duration)


#fig = plt.figure(figsize=[10, 10])
#plt.imshow(POST[:, :], aspect='auto', cmap='gray')
#plt.colorbar()
#plt.show()
