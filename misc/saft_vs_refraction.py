import numpy as np
from pathlib import Path
from scipy.signal import hilbert
from time import perf_counter_ns
import pickle
import matplotlib.pyplot as plt
global min_step, Cw, Cm, a, ARR_FOL, d1
global transducer_positions, lenT, T, V, L, IMAGE, trans_index, foc
FOLDER_NAME = "1D-3FOC50cm-60um"
directory_path = Path.cwd().parent
min_step = 6e-4
SAMPLE_START: int = 31500
SAMPLE_END: int = 33000
LEFT: int = 0
RIGHT: int = 175
Cw = 1498  # speed of sound in Water
Cm = 6320  # speed of sound in Metal
a = Cm/Cw  # ratio between two speeds of sound
foc = 0.0762  # metres
ARR_FOL = directory_path/FOLDER_NAME
fix = 150


def load_arr(FILENAME, output_folder=ARR_FOL):
    ftarr = output_folder/"tarr.pkl"
    fvarr = output_folder/FILENAME
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
    return np.abs(array - value).argmin()


# import data
tarr, varr = load_arr("varr.pkl")
ZERO: int = find_nearest(tarr[:, 0], 0)
d2_start: int = SAMPLE_START - ZERO
d2_end: int = SAMPLE_END - ZERO
T = tarr[ZERO:, 0]  # 1D, time columns all the same
V = np.copy(varr[ZERO:, LEFT:RIGHT])  # ZERO'd & sample width
tstep: float = np.abs(np.mean(T[1:]-T[:-1]))  # average timestep
d1 = T[d2_start]*Cw/2.  # distance to sample
d2 = T[d2_start:d2_end]*Cw/2. - d1  # sample grid (y distance)
d1 -= foc
L = varr.shape[1]  # number of transducer positions
lenT = len(T)  # length of time from ZERO to end of array
lend2: int = d2_end-d2_start  # sample thickness
IMAGE = np.empty((lend2, RIGHT-LEFT))  # empty final image array
transducer_positions = np.arange(L)*min_step
trans_index = np.arange(0, L, 1)  # transducer positions indices

# refraction
def roots(j):  # lat_pix is imaging x coord
    k: int = 0
    dt = np.zeros(L)  # delayed time: imaging pixel to transducer
    aa = np.abs(transducer_positions - transducer_positions[fix])
    if j == 0:
        dt = (2/Cw)*np.sqrt(aa[:]**2 + d1**2) + 2*foc/Cw
    else:
        while k < L:
            if d2[j] != 0 and aa[k] != 0:
                P4 = aa[k]**2
                P3 = -2*d1*aa[k]
                P2 = (aa[k]**2-aa[k]**2*a**2+d1**2-a**2*d2[j]**2)
                P1 = 2*d1*aa[k]*(a**2-1)
                P0 = d1**2*(1-a**2)
                roots = np.roots([P4, P3, P2, P1, P0])  # theta 1
                roots = np.real(roots[np.isreal(roots)])
                if roots.size != 0:
                    y0 = np.sqrt(np.square(roots) + 1)
                    stheta1 = 1./y0
                    st1 = stheta1[np.abs(stheta1) <= 1/a]
                    if st1.size > 0:
                        print(np.degrees(np.arcsin(st1)))
                        stheta1 = np.min(np.abs(st1))
                        rad1 = np.arcsin(stheta1)  # theta_1
                        rad2 = np.arcsin(stheta1*a)  # theta_2
                        dt[k]: float = 2*(np.abs(d1/Cw/np.cos(rad1))
                                          + np.abs(d2[j]/Cm/np.cos(rad2))
                                          + foc/Cw)
            elif aa[k] == 0 and d2[j] != 0:
                dt[k] = 2*(d1/Cw + d2[j]/Cm + foc/Cw)
            elif aa[k] == 0 and d2[j] == 0:
                dt[k] = 2*(d1/Cw + foc/Cw)
            k += 1
    zi = np.round(dt/tstep).astype(int)  # delayed t (indices)
    return zi


def saft(j):
    dt = np.zeros(L)  # delayed time: imaging pixel to transducer
    aa = np.abs(transducer_positions - transducer_positions[fix])
    z: float = d2[j] + d1
    dt: float = (2/Cw)*np.sqrt(aa[:]**2 + z**2) + 2*foc/Cw
    zi = np.round(dt/tstep).astype(int)  # delayed t (indices)
    return zi


crit_angle = np.arcsin(1/a)
thetas = np.linspace(-1*crit_angle, crit_angle, int(2e5))


def approxf(x, aa, d2):
    # the nonlinear function, solving for theta_1
    z = a*np.sin(x)
    z[z >= 1] = int(1.)
    theta2 = np.arcsin(z)
#    y = z + np.power(z, 3)/6+3*np.power(z, 5)/40
    first = d1*np.tan(x)
#    sec = d2*(y+np.power(y, 3)/3+2*np.power(y, 5)/15)
    sec = d2*np.tan(theta2)
    return first + sec - aa


def refr(j):
    k = 0
    dt = np.zeros(L)  # delayed time: imaging pixel to transducer
    aa = transducer_positions - transducer_positions[fix]
    while k < L:
        rad1 = thetas[np.abs(approxf(thetas, abs(aa[k]), d2[j])).argmin()]
        if np.isnan(rad1):
            rad1 = crit_angle
        if np.sin(rad1)>1/a:
            rad1 = crit_angle
        rad2 = np.arcsin(a*np.sin(rad1))
        s = np.abs(d2[j]/Cm/np.cos(rad2))
        if np.isnan(s) or s < 0 or s is None:
            sec = 0
        else:
            sec = s
        dt[k]: float = 2*(np.abs(d1/Cw/np.cos(rad1)) + sec + foc/Cw)
        k += 1
    zi = np.round(dt/tstep).astype(int)  # delayed t (indices)
    return zi

if __name__ == '__main__':
    start_time = perf_counter_ns()*1e-9
    dstep = tstep*Cw/2
    layer = 50
    zi_saft = saft(layer)
    zi_root = roots(layer)
    zi_refr = refr(layer)
#    zi = (zi_refr-zi_saft)[(zi_saft>0)*(zi_refr>0)]  # refr>saft
    plt.figure(figsize=[10, 10])
    plt.plot(zi_saft, label='saft', ls=':')
    plt.plot(zi_refr, label='argmin', c='goldenrod',ls='--')
    plt.plot(zi_root, label='root', ls=':', c='blue', lw=3, alpha=.5)
    plt.title("2e5 points, imaging pix at 0")
    plt.xlabel("x-coord")
    plt.ylabel("time delay (index)")
    plt.legend()
    plt.show()
    duration = perf_counter_ns()*1e-9-start_time
