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
import scipy.optimize as so
import pickle
import matplotlib.pyplot as plt
global min_step, Cw, Cm, a, DEFAULT_ARR_FOLDER, d1
global lat_distance, FD, SD, T, V, L, POST, lat_indices
FOLDER_NAME = "1D-3FOC5in-pipe-20um"  # edit this
min_step = 4e-4  # and this
FOCAL_DEPTH = 0.0381*2  # and this
Cw = 1498  # speed of sound in water
Cm = 6320  # speed of sound in metal
a = Cm/Cw  # ratio between two speeds of sound
d1 = .5  # in m, metal sample distance to the transducer
SAMPLE_DEPTH = d1
SAMPLE_THICKNESS = .013
if FOLDER_NAME[:2] == "1D":
    par = "1D SCANS"
else:
    par = "2D SCANS"
DEFAULT_ARR_FOLDER = join(dirname(getcwd()), "data", par, FOLDER_NAME)


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


def disMod(z, aa, d2):
    # z represents sin(theta_1)
    # aa lateral distances from imaging pixel to transducer
    # a*z is sin(theta_2)
    return ((d1*z)/np_sqrt(1-z**2)) + ((d2*a*z)/np_sqrt(1-(a*z)**2)) - aa


tarr, varr = load_arr()
tarr = tarr[:, 0, :]
varr = varr[:, 0, :]
ZERO = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
V = varr[ZERO:, :]  # 2D
FD = find_nearest(T, 2*FOCAL_DEPTH/Cw)  # focal depth
SD = len(T)
POST = np.empty(np.shape(V[FD:SD, :]))
T = T[FD:SD]
L = np.shape(V)[1]
lat_indices = np.arange(0, L, 1)
tstep = np.mean(T[1:]-T[:-1])  # average timestep
d2_start = find_nearest(T, 2*d1/Cw)
d2_end = find_nearest(T, 2*(d1+SAMPLE_THICKNESS)/Cw)
d2 = T[d2_start:(d2_end+1), :]*Cw/2
lat_distance = np.linspace(-L/2, L/2, L)*min_step
zw = np.zeros((L, len(d2), L))
for i in range(L):
    for j in range(len(d2)):
        aa = lat_distance[i] - lat_distance[:]
        z = so.fsolve(func=disMod, x0=0, args=(aa, d2[j]))
        zw[:, j, i] = z  # sin(theta_1)
zm = zw*a  # find sin(theta_2)
radM = np.arcsin(zm)  # theta_2
radW = np.arcsin(zw)  # theta_1
td_water = d1/np.cos(radW)  # real traveling distance inside water
d2arr = np.tile(d2.reshape((len(d2), 1)), (L, 1, L))
td_metal = np.zeros(zm.shape)
td_metal[:, :, :] = d2arr/np.cos(radM[:, :, :])  # traveling distance in metal
total_td = td_water + td_metal  # real traveling distance


def main(lat_pixel):  # lat_pixel is imaging x coord
    x = lat_distance[lat_pixel]  # x is imaging distance
    ti = FD  # imaging pixel for time/ z direction
    while ti < SD:
        if ti >= d2_start and ti <= d2_end:
            delay_t = (2/Cw)*np.flatten(total_td[:, ti-d2_start, lat_pixel])
        else:
            z = T[ti]*Cw/2
            z2 = np_power(z, 2)
            delay_t = (2/Cw)*np_sqrt(np_power(x-lat_distance[lat_indices], 2)
                                     + z2)
        zi = np.round(delay_t/tstep).astype(int)
        POST[ti-FD, lat_pixel] = np_sum(V[zi[zi < SD], lat_indices[zi < SD]])
        ti += 1


if __name__ == '__main__':
    # Parallel processing
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
#    b = 20*np.log10(b/np.max(b.flatten()))  # dont do this part
    pickle.dump(b, open(join(DEFAULT_ARR_FOLDER, "SAFT-{}.pkl"
                             .format(FOLDER_NAME)), "wb"))
    pickle.dump(T, open(join(DEFAULT_ARR_FOLDER, "SAFT-T-{}.pkl"), "wb"))

#fig = plt.figure(figsize=[10, 10])
#plt.imshow(STITCHED[:, :], aspect='auto', cmap='hot')
#plt.colorbar()
#plt.show()


#tp = 250  # the number of transducer positions; lateral direction
#d2 = np.linspace(0., 1.3, 100)/100  # discretize the sample thickness at sample depth
#lat_distance = np.linspace(-5., 5., tp)/100  # 250 positions for -5cm to 5cm scan
