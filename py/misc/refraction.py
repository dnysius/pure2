import numpy as np
from numpy import \
    sqrt as np_sqrt
import scipy.optimize as so


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
angleM = np.degrees(np.arcsin(zm))  # in degrees
angleW = np.degrees(np.arcsin(zw))

d1 = 50  # in cm, metal sample distance to the transducer

td_water = d1/np.cos(np.deg2rad(angleW))  # real traveling distance inside water
td_metal = np.zeros(zw.shape)
for k in range(len(d2)):
    temp = (d2[k]/100)/np.cos(np.deg2rad(angleM[k, :]))
    td_metal[k, :] = temp

total_td = td_water + td_metal  # real