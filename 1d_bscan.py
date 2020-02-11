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
rc('font',**{'family':'serif','sans-serif':['Times New Roman'],'size':12})
# Define paths
DATA_FOLDER = "FLAT50cm-PURE-ALLHOLES"  # folder containing scan data
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
dX = imgR-imgL
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
dstep_w = Cw*tstep/2  # distance corresponding average timestep (water)
dstep_m = dstep_w*a  # distance corresponding average timestep (metal)

if __name__ == '__main__':
    start_time = perf_counter_ns()*1e-9
    view = V[d2_start:d2_end, imgL:imgR].flatten()
    vmin = np.min(view)
    vmax = np.max(view)
    b = normalize(view, float(vmin), float(vmax))  # normalize to (-1,1)
    b = b.reshape((dY, dX))
    h = np.abs(hilbert(b, axis=0))
    b = 20*np.log10(h/np.max(h.flatten()))  # filter
    fig, ax1 = plt.subplots(1, 1, figsize=(11, 10))
    im0 = plt.imshow(b, aspect='auto', cmap='gray', interpolation='none', alpha=0)
    ax2 = ax1.twinx()  # second scale on same axes
    im1 = ax2.imshow(b, aspect='auto', cmap='gray',
                     interpolation='none', alpha=1)
    seg_x = (imgR+imgL)//2 - imgL  # line segment x position
    ax2.axvline(x=seg_x, ymin=(1 - ymin/dY),
                ymax=(1 - ymax/dY), c='red', lw=3,
                alpha=.7, label='{} m'.format(np.round(Cm*tstep*abs(ymin-ymax)/2, 7)))
    plt.title("{} B-Scan".format(title))
    colorbar = fig.colorbar(im1, orientation='vertical', pad=0.12, fraction=.05, aspect=50)
    colorbar.set_label('dB')
    y_formatter = FixedFormatter(np.round((ax2.get_yticks()+d2_start)*100*tstep*Cw/2, 2))
    ax2.yaxis.set_major_formatter(y_formatter)
    ax1.set_xlabel("Lateral distance [cm]")
    ax1.set_ylabel("Timeseries index")
    ax2.set_ylabel("Axial distance [cm]")
    ax1.set_xticklabels(np.round(ax1.get_xticks()*100*min_step, 4))
    ax1.set_yticklabels((ax1.get_yticks()).astype(int))
    plt.legend(loc='upper right', framealpha=1)
    duration = perf_counter_ns()*1e-9-start_time
    print("Plotting took {} s".format(duration))
    start_time = perf_counter_ns()*1e-9
    plt.savefig(ARR_FOL/'b-scan.png', dpi=200)
    duration = perf_counter_ns()*1e-9-start_time
    print("Saving the picture took {} s".format(duration))
    # plt.show()
