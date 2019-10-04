# -*- coding: utf-8 -*-
import pickle
import numpy as np
from os.path import join, dirname, isfile
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from os import getcwd, listdir
global tarr, SCAN_FOLDER, BSCAN_FOLDER, FOLDER_NAME, saft, min_step
min_step = 4e-4
FOLDER_NAME = "2D-3FOC5in"
SCAN_FOLDER = join(dirname(dirname(getcwd())), FOLDER_NAME)
BSCAN_FOLDER = join(dirname(getcwd()), "scans", "SAFT", "2D")

onlyfiles = [f for f in listdir(SCAN_FOLDER)
             if isfile(join(SCAN_FOLDER, f))]
for file in onlyfiles:
    with open(join(SCAN_FOLDER, file), "rb") as rd:
        if 'tarr' in file:
            tarr = pickle.load(rd)
        elif 'saft_view' in file:
            saft = np.load(rd)

with open(join(SCAN_FOLDER, "saft_view.npy"), "rb") as rd:
    saft = np.load(rd)
    saft = np.abs(hilbert(saft, axis=0))
    saft = 20*np.log10(saft/np.max(saft.flatten()))

#with open(join(SCAN_FOLDER, "saft_view.npy"), "wb") as wr:
#    np.save(wr, saft[14000:, :, :])

#
#def ibscan(figsize=[10, 10], y1=0, y2=-1):
#    bscan(figsize=figsize, view=view, y1=y1, y2=y2)
#    cmd = input('//\t')
#    if cmd == 'x':
#        print('Exit')
#        pass
#    elif cmd == 'c':
#        try:
#            a1 = input('y1 (default {}):\t'.format(y1))
#            a2 = input('y2 (default {}):\t'.format(y2))
#            if a1 == '':
#                a1 = y1
#            else:
#                a1 = int(a1)
#            if a2 == '':
#                a2 = y2
#            else:
#                a2 = int(a2)
#            ibscan(figsize=figsize, start=start,
#                   end=end, y1=a1, y2=a2)
#        except ValueError:
#            print("Invalid input")
#    elif cmd == 't':
#        try:
#            a1 = input('start (default {}):\t'.format(view[0]))
#            a2 = input('end (default {}):\t'.format(end))
#            if a1 == '':
#                a1 = start
#            else:
#                a1 = int(a1)
#            if a2 == '':
#                a2 = end
#            else:
#                a2 = int(a2)
#            ibscan(figsize=figsize, start=a1, end=a2, y1=y1, y2=y2)
#        except ValueError:
#            print('invalid input')
#    elif cmd == 'raw' or cmd == 'r':
#        ibscan(figsize=figsize, start=start,
#               end=end, y1=y1, y2=y2, sa=False)
#    elif cmd == 'sa' or cmd == 'saft':
#        ibscan(figsize=figsize, start=start,
#               end=end, y1=y1, y2=y2, sa=True)
#    elif cmd == 's':
#        ibscan(figsize=figsize, start=start, end=end, y1=y1, y2=y2)
#    else:
#        ibscan(figsize=figsize, start=start, end=end, y1=y1, y2=y2)
#


def view_yz(x=0, figsize=[10, 10], start=0, end=-1, y1=0, y2=-1):
    if figsize == [0, 0]:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    if y2 == -1:
        y2 = len(saft[start:end, :, 0]) - 1
    plt.imshow(saft[start:end, :, x], aspect='auto', cmap='gray', vmin=-30)
    plt.colorbar()
    plt.axhline(y=y1)
    plt.axhline(y=y2)
    dt = np.mean(tarr[y2+start, 0, :]) - np.mean(tarr[y1+start, 0, :])
    v_w = 1498
    v_m = 6420
    dw = v_w*dt/2
    dm = v_m*dt/2
    plt.axhline(y=0, label='water: {}'.format(dw), alpha=0)
    plt.axhline(y=0, label='aluminum: {}'.format(dm), alpha=0)
    plt.title("{0} {1}".format(FOLDER_NAME, "SAFT"))
    plt.legend()
    plt.xlabel("lateral")
    plt.ylabel("axial")
    plt.show(fig)


def view_xz(y=0, figsize=[10, 10], start=0, end=-1, y1=0, y2=-1):
    if figsize == [0, 0]:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    if y2 == -1:
        y2 = len(saft[start:end, 0, :]) - 1
    plt.imshow(saft[start:end, y, :], aspect='auto', cmap='gray', vmin=-30)
    plt.colorbar()
    plt.axhline(y=y1)
    plt.axhline(y=y2)
    dt = np.mean(tarr[y2+start, 0, :]) - np.mean(tarr[y1+start, 0, :])
    v_w = 1498
    v_m = 6420
    dw = v_w*dt/2
    dm = v_m*dt/2
    plt.axhline(y=0, label='water: {}'.format(dw), alpha=0)
    plt.axhline(y=0, label='aluminum: {}'.format(dm), alpha=0)
    plt.title("{0} {1}".format(FOLDER_NAME, "SAFT"))
    plt.legend()
    plt.xlabel("lateral")
    plt.ylabel("axial")
    plt.show(fig)


def view_xy(z=0, figsize=[10, 10], x1=0, x2=-1, y1=0, y2=-1):
    if figsize == [0, 0]:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    if y2 == -1:
        y2 = len(saft[0, :, 0]) - 1
    if x2 == -1:
        x2 = len(saft[0, 0, :]) - 1
    plt.imshow(saft[z, :, :], aspect='auto', cmap='gray', vmin=-30)
    plt.colorbar()
    plt.axhline(y=y1)
    plt.axhline(y=y2)
    plt.axvline(x=x1)
    plt.axvline(x=x2)
    dx = (x2-x1)*min_step
    dy = (y2-y1)*min_step
    plt.axhline(y=0, label='dy: {}'.format(dy), alpha=0)
    plt.axhline(y=0, label='dx: {}'.format(dx), alpha=0)
    plt.title("{0} {1}".format(FOLDER_NAME, "SAFT"))
    plt.legend()
    plt.xlabel("lateral")
    plt.ylabel("axial")
    plt.show(fig)


if __name__ == '__main__':
    pass
