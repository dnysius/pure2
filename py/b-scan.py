# -*- coding: utf-8 -*-
import numpy as np
from os.path import join
from time import perf_counter_ns
from misc.load_arr import load_arr, find_nearest
from misc.normalize_image import normalize
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
from misc.load_conf import load_conf
DATA_FOLDER: str = "3FOC50cm-60um"
directory_path: str = "C:\\Users\\indra\\Documents\\GitHub"
ARR_FOL = join(directory_path, DATA_FOLDER)
tarr, varr = load_arr("varr.pkl", output_folder=ARR_FOL)
start_time = perf_counter_ns()*1e-9
conf = load_conf(ARR_FOL)
duration = perf_counter_ns()*1e-9-start_time
print('Loading config took {} s'.format(duration))
min_step = conf['min_step']
foc = conf['foc']
Cw = conf['Cw']
Cm = conf['Cm']
SAMPLE_START = conf['SAMPLE_START']
SAMPLE_END = conf['SAMPLE_END']
imgL = conf['imgL']
imgR = conf['imgR']

a = Cm/Cw  # ratio between two speeds of sound
ZERO: int = find_nearest(tarr[:, 0], 0)
T = tarr[ZERO:, 0]  # 1D, time columns all the same
lenT = len(T)  # length of time from ZERO to end of array
V = np.copy(varr[ZERO:, :])  # ZERO'd & sample width
L = V.shape[1]  # scanning width (number of transducer positions)
if (SAMPLE_START - ZERO) < 0:
    d2_start:int = ZERO
else:
    d2_start: int = SAMPLE_START - ZERO
d2_end: int = SAMPLE_END - ZERO
dY = d2_end - d2_start  # sample thickness
dX = imgR-imgL
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
dstep: float = tstep*Cw/2

if __name__ == '__main__':
    start_time = perf_counter_ns()*1e-9
    seg_x = (imgR+imgL)//2 - imgL
    ymin = 250
    ymax = 50
    view = V[d2_start:d2_end, imgL:imgR].flatten()
    vmin = np.min(view)
    vmax = np.max(view)
    b = normalize(view, float(vmin), float(vmax))
    b = b.reshape((dY, dX))
    fig, ax1 = plt.subplots(1, 1, figsize=(11, 10))
    im0 = plt.imshow(b, aspect='auto', cmap='gray', interpolation='none', alpha=0)
    ax2 = ax1.twinx()  # second scale on same axes
    im1 = ax2.imshow(b, aspect='auto', cmap='gray',
                     interpolation='none', alpha=1)
    ax2.axvline(x=seg_x, ymin=(1 - ymin/dY),
                ymax=(1 - ymax/dY), c='red', lw=5,
                alpha=.7, label='{} m'.format(Cm*tstep*abs(ymin-ymax)/2))
    plt.title("{} B-Scan".format(DATA_FOLDER))
    fig.colorbar(im1, orientation='vertical', pad=0.1, fraction=.05, aspect=50)
    y_formatter = FixedFormatter(np.round((ax2.get_yticks()+d2_start)*100*tstep*Cw/2, 2))
    ax2.yaxis.set_major_formatter(y_formatter)
    ax1.set_xlabel("lateral distance (cm)")
    ax1.set_ylabel("index")
    ax2.set_ylabel("axial distance (cm)")
    ax1.set_xticklabels(np.round(ax1.get_xticks()*100*min_step, 4))
    ax1.set_yticklabels((ax1.get_yticks()).astype(int))
    plt.legend(loc='upper right', framealpha=1)
    duration = perf_counter_ns()*1e-9-start_time
    print("Plotting took {} s".format(duration))
    start_time = perf_counter_ns()*1e-9
#    plt.savefig(join (ARR_FOL, 'b-scan.png'), dpi=400)
    duration = perf_counter_ns()*1e-9-start_time
    print("Saving the picture took {} s".format(duration))
    plt.show(fig)