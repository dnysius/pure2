# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FixedFormatter
from pathlib import Path
from time import perf_counter_ns
from misc.arrOp import load_arr, find_nearest, normalize
from misc.load_conf import load_conf
from scipy.signal import hilbert
from numba import vectorize
rc('font',**{'family':'serif','sans-serif':['Times New Roman'],'size':12})
# Define paths
DATA_FOLDER = "3LENS50cm-PURE-ALLHOLES"  # folder containing scan data
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

a = Cm/Cw  # ratio between speeds of sound
ZERO: int = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1 dimensional array - time axis off of oscilloscope
lenT = len(T)  # number of timesteps from ZERO to end of array
V = varr[ZERO:, :]  # 2 dimensional array - Voltage data aka b-scan
L = V.shape[1]  # number of scanning positions, or length of scan
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
d1 = T[d2_start]*Cw/2.  # distance (m) to sample
d2 = T[d2_start:d2_end]*Cw/2. - d1  # distances below sample surface, 1d array
d1 -= foc  # d1 corresponds to the distance from sample surface to focal depth
N = dY*dX  # total grid size
trans = np.arange(L)*min_step  # transducer positions


@vectorize(['float64(int64)'], target='parallel')
def saft(c):
    i = c % dX  # x-coord of impix
    j = c // dX  # y-coord of impix
    i += imgL
    aa = np.abs(trans - trans[i])  # lateral distance from impix to trans
    z: float = d2[j] + d1
    dt = (2/Cw)*np.sqrt(aa[:]**2 + z**2) + 2*foc/Cw  # delayed times
    res = 0
    for k in range(L):
        t = int(np.round(dt[k]/tstep))  # delayed time in indices
        if t < d2_end:
            d = V[t, k]
            res += float(d)  # summation
    return res


def plt_saft():
    start_time = perf_counter_ns()*1e-9
    impix = np.arange(N)
    p = saft(impix)
    p = normalize(p, float(np.min(p)), float(np.max(p)))  # normalize to (-1,1)
    POST = p.reshape((dY, dX), order='C')
    h = np.abs(hilbert(POST, axis=0))
    POST = 20*np.log10(h/np.max(h.flatten()))  # filter
    # PLOTTING
    fig, ax1 = plt.subplots(1, 1, figsize=(11, 10))
    im0 = plt.imshow(POST, aspect='auto', cmap='gray')
    ax2 = ax1.twinx()  # second scale on same axes
    im1 = ax2.imshow(POST, aspect='auto', cmap='gray',
                     interpolation='none', alpha=1)
    seg_x = (imgR+imgL)//2 - imgL  # line segment x position
    ax2.axvline(x=seg_x, ymin=(1 - ymin/dY),
                ymax=(1 - ymax/dY), c='red', lw=3,
                alpha=.7, label='{} m'.format(np.round(Cm*tstep*abs(ymin-ymax)/2, 7)))
    colorbar = fig.colorbar(im1, orientation='vertical', pad=0.12, fraction=.05, aspect=50)
    colorbar.set_label('dB')
    plt.title("{} SAFT".format(title))
    y_formatter = FixedFormatter(np.round((ax2.get_yticks()+d2_start)*100*tstep*Cw/2, 2))
    ax2.yaxis.set_major_formatter(y_formatter)
    ax1.set_xlabel("Lateral distance [cm]")
    ax1.set_ylabel("Timeseries index")
    ax2.set_ylabel("Axial distance [cm]")
    ax1.set_xticklabels(np.round(ax1.get_xticks()*100*min_step, 4))
    ax1.set_yticklabels((ax1.get_yticks()).astype(int))
    plt.legend(loc='upper right', framealpha=1)
    duration = perf_counter_ns()*1e-9-start_time
    print("Summing and plotting took {} s".format(duration))
    start_time = perf_counter_ns()*1e-9
    plt.savefig(ARR_FOL/"saft.png", dpi=400)  # save image
    duration = perf_counter_ns()*1e-9-start_time
    print("Saving the picture took {} s".format(duration))
    # plt.show()
    return POST


if __name__ == '__main__':
    POST_saft = plt_saft()
