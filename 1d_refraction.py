# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
from pathlib import Path
from time import perf_counter_ns
from misc.arrOp import load_arr, find_nearest, normalize
from misc.load_conf import load_conf
from scipy.signal import hilbert
from numba import vectorize
from tqdm import tqdm
# Define paths
DATA_FOLDER = "3LENS7in-PURE2"  # folder containing scan data
directory_path: str = Path.cwd().parent
# Import data
ARR_FOL = directory_path/DATA_FOLDER
tarr, varr = load_arr(ARR_FOL)  # load time and voltage arrays from the folder
start_time = perf_counter_ns()*1e-9
conf = load_conf(ARR_FOL)  # load config files from conf.txt
duration = perf_counter_ns()*1e-9-start_time
print('Loading config took {} s'.format(duration))
title = conf['title']
min_step = conf['min_step']  # stepper motor step size
foc = conf['foc']  # focal depth of the transducer; 0 if Flat
Cw = conf['Cw']  # speed of sound in water
Cm = conf['Cm']  # speed of sound in metal
SAMPLE_START = conf['SAMPLE_START']  # the index/position of the sample surface
SAMPLE_END = conf['SAMPLE_END']  # the index/position of sample bottom surface
imgL = conf['imgL']  # left index of the image boundary
imgR = conf['imgR']  # right index of the image boundary
ymin = conf['ymin']  # bottom index of the line segment measuring thickness
ymax = conf['ymax']  # top index of the line segment measuring thickness
# Setting up arrays and calculating constants
a = Cm/Cw  # ratio between speeds of sound
ZERO: int = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
lenT = len(T)  # length of time from ZERO to end of array
V = varr[ZERO:, :]  # 2 dimensional array - Voltage data aka b-scan
L = V.shape[1]  # scanning width (number of transducer positions)
if (SAMPLE_START - ZERO) >= 0:
    d2_start: int = SAMPLE_START - ZERO
else:  # prevent d2_start < 0
    d2_start: int = ZERO
d2_end: int = SAMPLE_END - ZERO
dY = d2_end - d2_start  # sample thickness
dX = imgR-imgL  # image width
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
dstep_w = Cw*tstep/2  # distance corresponding average timestep (water)
dstep_m = dstep_w*a  # distance corresponding average timestep (metal)
d1 = T[d2_start]*Cw/2.  # distance to sample
d2 = a*(T[d2_start:d2_end]*Cw/2. - d1)  # sample column (y distance)
d1 -= foc
N = dY*dX  # total grid size
trans = np.arange(L)*min_step  # transducer positions
aa = trans - trans[0]  # lateral distance from leftmost position to other trans
td_arr = np.empty((dY, L))  # time delays (td) in a 2d array


def determine_total_distance(j):
    # Calculates the delay from the leftmost impix (in a given row) to all
    # transducer positions
    # j: the j-th row in the sample grid
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
    np.save(ARR_FOL/'td_arr.npy', td_arr, allow_pickle=False)
    print("Creating td_arr took {} s".format(duration))
    return td_arr


def load_td_arr():
    start_time = perf_counter_ns()*1e-9
    td = np.load(ARR_FOL/'td_arr.npy')
    duration = perf_counter_ns()*1e-9-start_time
    print("Loading td_arr took {} s".format(duration))
    return td


td_arr = create_td_arr()  # create delay
# td_arr = load_td_arr()  # load delay


@vectorize(['float64(int64)'], target='parallel')
def refr(c):
    i = int(c % dX)  # x-coord of impix
    j = int(c // dX)  # y-coord of impix
    res = 0
    for k in range(L):
        m = abs(i + imgL - k)
        t = int(td_arr[j, m])  # delayed t index
        if t < lenT:  # if the delayed time is in the voltage data (varr)
            d = float(V[t, k])
            res += d  # summation
    return res


def plt_refr():
    start_time = perf_counter_ns()*1e-9
    impix = np.arange(N)
    p = refr(impix)
    p = normalize(p, float(np.min(p)), float(np.max(p)))  # normalize to (-1,1)
    POST = p.reshape((dY, dX), order='C')
    POST = 20*np.log10(np.abs(hilbert(POST, axis=0)))  # filter
    fig, ax1 = plt.subplots(1, 1, figsize=(11, 10))
    im0 = plt.imshow(POST, aspect='auto', cmap='gray')
    ax2 = ax1.twinx()  # second scale on same axes
    ax1.set_xlabel("lateral distance (cm)")
    ax1.set_ylabel("index")
    ax1.set_xticklabels(np.round(ax1.get_xticks()*100*min_step, 4))
    ax1.set_yticklabels((ax1.get_yticks()).astype(int))
    im1 = ax2.imshow(POST, aspect='auto', cmap='gray',
                     interpolation='none', alpha=1)
    seg_x = (imgR+imgL)//2 - imgL  # line segment x position
    ax2.axvline(x=seg_x, ymin=(1 - ymin/dY),
                ymax=(1 - ymax/dY), c='red', lw=5,
                alpha=.7, label='{} m'.format(np.round(Cm*tstep*abs(ymin-ymax)/2, 7)))
    colorbar = fig.colorbar(im1, orientation='vertical', pad=0.1, fraction=.05, aspect=50)
    colorbar.set_label('relative signal strength')
    plt.title("{} refraction".format(title))
    y_formatter = FixedFormatter(np.round((ax2.get_yticks()+d2_start)*100*tstep*Cw/2, 2))
    ax2.yaxis.set_major_formatter(y_formatter)
    ax2.set_ylabel("axial distance (cm)")
    plt.legend(loc='upper right', framealpha=1)
    duration = perf_counter_ns()*1e-9-start_time
    start_time = perf_counter_ns()*1e-9
    print("Summation and plotting took {} s".format(duration))
    plt.savefig(ARR_FOL/'refraction.png', dpi=400)
    duration = perf_counter_ns()*1e-9-start_time
    print("Saving the picture took {} s".format(duration))
    plt.show()
    return POST


if __name__ == '__main__':
    POST_refr = plt_refr()
