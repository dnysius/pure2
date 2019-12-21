# -*- coding: utf-8 -*-
import numpy as np
from os.path import join
from time import perf_counter_ns
from misc.load_arr import load_arr, find_nearest
from misc.normalize_image import normalize
import matplotlib.pyplot as plt
from misc.load_conf import load_conf
DATA_FOLDER: str = "3FOC50cm-60um"
directory_path: str = "C:/Users/indra/Documents/GitHub"
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
dY = SAMPLE_END - SAMPLE_START  # sample thickness
dX = imgR-imgL
d2_start: int = SAMPLE_START - ZERO
d2_end: int = SAMPLE_END - ZERO
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
dstep: float = tstep*Cw/2

if __name__ == '__main__':
    start_time = perf_counter_ns()*1e-9
    view = V[d2_start:d2_end, imgL:imgR].flatten()
    vmin = np.min(view)
    vmax = np.max(view)
    b = normalize(view, float(vmin), float(vmax))
    b = b.reshape((dY, dX))
    plt.figure(figsize=[10, 10])
    plt.imshow(b, aspect='auto', cmap='gray')
    plt.title("{} B-Scan".format(DATA_FOLDER))
    plt.colorbar()
    duration = perf_counter_ns()*1e-9-start_time
    print("Plotting took {} s".format(duration))
#    plt.savefig(join(ARR_FOL, 'b-scan.png'), dpi=600)
    plt.show()
