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
global min_step, c_0, DEFAULT_ARR_FOLDER
global lat_distance, FD, SD, pbar, T, V, L, T_COMPARE, PRE_OUT, POST_OUT, lat_indices

FOLDER_NAME = "1D-3FOC5in-pipe-20um"  # edit this
min_step = 4e-4  # and this
FOCAL_DEPTH = 0.0381*2  # and this
c_0 = 1498  # speed of sound in water
SAMPLE_DEPTH = .5
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


##############################
def disMod(z, aa, d2):
    Cw = 1480  # speed of sound in water
    Cm = 5900  # speed of sound in metal
    a = Cm/Cw  # ratio between two speeds of sound
    d1 = 50  # in cm, metal sample distance to the transducer
    return ((d1*z)/np_sqrt(1-z**2)) + ((d2*a*z)/np_sqrt(1-(a*z)**2)) - aa


d2 = np.linspace(.0001, 1.3, 100)  # discretize the sample thickness at sample depth
tp = 250  # the number of transducer positions; lateral direction
zw = np.zeros((len(d2), tp))
for i in range(tp):
    for j in range(len(d2)):
        zg = np.zeros((tp,))
        lat_distance = np.linspace(-5., 5., tp)  # 250 positions for -5cm to 5cm scan
        z = so.fsolve(func=disMod, x0=zg[i], args=(lat_distance[i], d2[j]))  # solve function
        zw[j, i] = z
        # haven't found error analysis from fsolve

Cw = 1480  # speed of sound in water
Cm = 5900  # speed of sound in metal
a = Cm/Cw  # ratio between two speeds of sound
zm = zw*a
angleM = np.sin(np.degrees(zm))  # in degrees
angleW = np.sin(np.degrees(zw))

d1 = 50  # in cm, metal sample distance to the transducer

td_water = d1/np.cos(np.deg2rad(angleW))  # real traveling distance inside water
td_metal = np.zeros(zw.shape)
for k in range(len(d2)):
    temp = (d2[k]/100)/np.cos(np.deg2rad(angleM[k, :]))
    td_metal[k, :] = temp

total_td = td_water + td_metal  # real traveling distance
###############################
tarr, varr = load_arr()
tarr = tarr[:, 0, :]
varr = varr[:, 0, :]
ZERO = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
V = varr[ZERO:, :]  # 2D
FD = find_nearest(T, 2*FOCAL_DEPTH/c_0)  # focal depth
SD = len(T)
L = np.shape(V)[1]
PRE = np.flip(V[:FD, :], axis=0)
PRE_T = np.flip(T[:FD], axis=0)
PRE_OUT = np.empty(np.shape(PRE))
POST = V[FD:SD, :]
POST_T = T[FD:SD]
POST_OUT = np.empty(np.shape(POST))
lat_distance = np.linspace(-L/2, L/2, L)*min_step
lat_indices = np.arange(0, L, 1)
tstep = np.mean(T[1:]-T[:-1])  # average timestep
SAMPLE_DEPTH_INDEX = np.round(SAMPLE_DEPTH/tstep).astype(int)
SAMPLE_THICKNESS_INDEX = np.round(SAMPLE_THICKNESS/tstep).astype(int)
SAMPLE_RANGE = (SAMPLE_DEPTH_INDEX, SAMPLE_DEPTH_INDEX+SAMPLE_THICKNESS_INDEX)
SAMPLE = np.zeros((SAMPLE_THICKNESS_INDEX, L), dtype=float)
# if SAMPLE_RANGE[0] <= ti <=SAMPLE_RANGE[1]
# then we need to apply the compensation for refraction in metal

def main(lat_pixel):  # lat_pixel is imaging index/position
    x = lat_distance[lat_pixel]  # x is imaging DISTANCE
    ti = 0  # imaging pixel for time/ z direction
    while ti < SD:
        z = T[ti]*c_0/2
        z2 = np_power(z, 2)  # distance, squared
        ind = (2/c_0)*np_sqrt(np_power(x-lat_distance[lat_indices], 2)
                              + z2)
        zi = np.round(ind/tstep).astype(int)
        if ti < FD:  # PRE
            PRE_OUT[ti, lat_pixel] = np_sum(V[zi[zi < FD], lat_indices[zi < FD]])
        if ti >= FD:  # POST
            POST_OUT[ti-FD, lat_pixel] = np_sum(V[zi[zi < SD], lat_indices[zi < SD]])
        ti += 1


if __name__ == '__main__':
    pass
    # Parallel processing
#    jobs = []
#    print("Append")
#    for i in range(L):
#        jobs.append(threading.Thread(target=main, args=(i,)))
#    print("Starting")
#    for job in jobs:
#        job.start()
#    print("Joining")
#    for job in jobs:
#        job.join()
#    print("Stitching")
#    PRE_OUT = np.flip(PRE_OUT, axis=0)
#    b = np.abs(hilbert(POST_OUT[:, :], axis=0))
#    b = 20*np.log10(b/np.max(b.flatten()))
#    STITCHED = np.vstack((PRE_OUT, b))
#    pickle.dump(STITCHED, open(join(DEFAULT_ARR_FOLDER,
#                                    "SAFT-{}.pkl"
#                                    .format(FOLDER_NAME)), "wb"))


#fig = plt.figure(figsize=[10, 10])
#plt.imshow(STITCHED[:, :], aspect='auto', cmap='hot')
#plt.colorbar()
#plt.show()
