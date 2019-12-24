# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, getcwd, makedirs
from os.path import isfile, isdir, join, dirname, exists
import re
import pickle
global tstep, BSCAN_FOLDER
BSCAN_FOLDER = join(dirname(getcwd()), "scans", "BSCAN")
tstep = 3.999999999997929e-08
_nsre = re.compile('([0-9]+)')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def indw2time(indw):
    v_m = 6420
    return v_m*indw*tstep/2


class Signal:
    def __init__(self, mypath, ftype='npy'):
        s = mypath.split(sep='\\')
        self.mypath = mypath
        if s[-1] == '':
            self.title = s[-2]
        else:
            self.title = s[-1]
        if not isdir(mypath):
            mkdir(mypath)
            self.BSCAN_FOLDER = BSCAN_FOLDER
            self.ftype = ftype
            self.fnames = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-3:] == self.ftype[-3:]]
            self.fnames.sort(key=natural_sort_key)
            self.angles = np.linspace(0, 30, len(self.fnames))
            self.angle_step = 30/len(self.fnames)
            self.signal_data = np.empty(shape=(len(self.fnames), len(self.__loadf(0)[:, 0]), 2))
            lenY = len(self.signal_data[0, :, 0])
            for i in range(len(self.fnames)):
                xy = self.__loadf(i)
                self.signal_data[i, :, :] = xy
        self.signal_data = self.signal_data[:, lenY//2:, :]
        self.signal_data[:, :, 1] = self.signal_data[:, :, 1]/np.amax(np.abs(self.signal_data[:, :, 1]).flatten())
        out = join(dirname(getcwd()), "obj", self.title+"_signal_data.pkl")
        with open(out, 'wb') as wr:
            pickle.dump(self.signal_data, wr, pickle.DEFAULT_PROTOCOL)

    def __loadf(self, i):
        if self.ftype == '.csv' or self.ftype == 'csv':
            xy = np.loadtxt(open(join(self.mypath, self.fnames[i]), "rb"), delimiter=',', skiprows=0)
        else:
            xy = np.load(open(join(self.mypath, self.fnames[i]), "rb"))
        if self.ftype == '.npz' or self.ftype == 'npz':
            xy = xy[xy.files[0]]
        return xy

    def plot_peak(self, i, start, width, x1=0, x2=-1):
        if width == -1:
            width = np.shape(self.signal_data)[1]-start
        if x2 == -1:
            x2 = width-1

        end = start + width
        y = np.abs(self.signal_data[i, start:end, 1])
        self.fig = plt.figure(figsize=[8, 6])
        plt.plot(y, c='goldenrod', ls='--', alpha=.7, label='signal')
        plt.axvline(x=x1, c='grey', label="x1={}".format(x1))
        plt.axvline(x=x2, c='grey', label='x2={}'.format(x2))
        plt.title('{0}deg, {1}m, ({2}, {3})'.format(self.angles[i], round(indw2time(x2-x1), 6), start, width))
        plt.xlabel('time (s)')
        plt.ylabel('voltage (V)')
        plt.legend()
        plt.show(self.fig)

    def analyze_peak(self, i, start, width, x1=0, x2=-1):
        self.plot_peak(i, start, width, x1, x2)
        cmd = input("//\t")
        if cmd == 's':
            self.fig.savefig(join(self.folder, "{0}.png".format(i)))
            print('done saving')
        elif cmd == 'z' or cmd =='':
            sind = input("\nstart (default {0}): \t".format(start))
            wind = input("\nwidth (default {0}): \t".format(width))
            try:
                if sind == '':
                    pass
                else:
                    start = int(sind)
                if wind == '':
                    pass
                else:
                    width = int(wind)

                self.analyze_peak(i, start, width, x1, x2)
            except:
                print("ERROR must enter an integer, or integer entered may be out of range")
        elif cmd == 'c' or cmd == 'cursor':
            nx1 = input("x1 (current {})\t".format(x1))
            nx2 = input("x2 (current {})\t".format(x2))
            if nx1 == '':
                pass
            else:
                x1 = int(nx1)
            if nx2 == '':
                pass
            else:
                x2 = int(nx2)
                self.analyze_peak(i, start, width, x1, x2)
        elif cmd=='esc' or cmd=='exit' or cmd=='x':
            print("Exit")
        else:
            print("invalid input")
            self.analyze_peak(i, start, width, x1, x2)

    def ang_bscan(self, domain=(2600, 5100), vmin=0, vmax=1, y1=0, y2=-1):
        START = domain[0]
        END = domain[1]
        if END == -1:
            END = len(self.signal_data[0, :, 1]) - 1
        else:
            pass
        if y2 == -1:
            y2 = END
        else:
            y2 += START
        if y1 == 0:
            y1 = START
        else:
            y1 += START
        y = np.abs(self.signal_data[:, START:END, 1]) / np.max(np.abs(self.signal_data[:, START:END, 1]).flatten())
        bscan = np.transpose(y)
        plt.ioff()
        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(self.title)
#       major_ticks = np.arange(tstep*START, tstep*END, tstep*((END-START)//10))
#       minor_ticks = np.arange(tstep*START, tstep*END, tstep*((END-START)//50))
        ax.imshow(bscan, cmap='gray', origin='upper', aspect='auto', alpha=.9, extent=[0, len(bscan[0, :]), END, START], vmin=vmin, vmax=vmax)
        ax.axhline(y=y1, c='goldenrod')
        ax.axhline(y=y2, c='goldenrod', label='thickness {}m'.format(round(indw2time(y2-y1), 6)))
        ax.set_xlabel('angle (degrees)')
        ax.set_ylabel('time (s)')
        ax.set_title('{0}, ({1}, {2})'.format(self.title, START, END))
        plt.legend()
#          ax.set_yticks(major_ticks)
#          ax.set_yticks(minor_ticks, minor=True)
#          ax.grid(True, axis='y', which="major", alpha=.5)
#          ax.grid(True, axis='y', which="minor", alpha=.2, linestyle="--")
#          ax.grid(True, axis='x', which="major", alpha=.2, linestyle="--")
        plt.savefig(join(self.BSCAN_FOLDER, self.title+'.png'), dpi=300)
        plt.show(fig)

    def Iang_bscan(self, domain=(2600, 5200), vmin=0, vmax=1, y1=0, y2=-1):
        START = domain[0]
        END = domain[1]
        if END == -1:
            END = np.shape(self.signal_data)[1] - 1
        self.ang_bscan((START, END), vmin, vmax, y1, y2)
        cmd = input('//\t')
        if cmd == 's':
            self.Iang_bscan((START, END), vmin, vmax, y1, y2)
        elif cmd == 'z':
            d0 = input('Start (default {}): \t'.format(START))
            d1 = input('End (default {}): \t'.format(END))
            if d0 == '':
                d0 = START
            if d1 == '':
                d1 = END
            try:
                START = int(d0)
                END = int(d1)
                self.Iang_bscan((START, END), vmin, vmax, y1, y2)
            except ValueError:
                print("z input error")
        elif cmd == 'v':
            v0 = input('vmin (default {}): \t'.format(vmin))
            v1 = input('vmax (default {}): \t'.format(vmax))
            if v0 == '':
                v0 = vmin
            if v1 == '':
                v1 = vmax
            try:
                vmin = int(v0)
                vmax = int(v1)
                self.Iang_bscan((START, END), vmin, vmax, y1, y2)
            except ValueError:
                print("v input error")
        elif cmd == 'c':
            ny1 = input('y1 (default {}): \t'.format(y1))
            ny2 = input('y2 (default {}): \t'.format(y2))
            if ny1 == '':
                ny1 = y1
            if ny2 == '':
                ny2 = y2
            try:
                y1 = int(ny1)
                y2 = int(ny2)
                self.Iang_bscan((START, END), vmin, vmax, y1, y2)
            except ValueError:
                print("c input error")
        elif cmd == 'x' or cmd == 'esc':
            pass
        else:
            self.Iang_bscan((START, END), vmin, vmax, y1, y2)


def graph_signals(trans, start, end):
    with open(join(r"C:\Users\dionysius\Desktop\PURE\pure\obj", "{}_signal_data.pkl".format(trans)), "rb") as rd:
        signal_data = np.load(rd)
        L = np.shape(signal_data)[0]
        ang = np.linspace(0, 30, L)
        spath = "C:\\Users\\dionysius\\Desktop\\PURE\\pure\\data\\30deg\\{}\\signals".format(trans)
        if not exists(spath):
            makedirs(spath)
    for i in range(L):
        plt.figure(figsize=[8, 6])
        plt.plot(np.arange(start, end, 1), signal_data[i, start:end, 1])
        plt.xlabel('time (unit of timestep)')
        plt.ylabel('voltage')
        plt.ylim(-1.1, 1.1)
        plt.title('{2} {0} degrees (index {1})'.format(ang[i], i, trans))
        plt.savefig(join(spath, "{}.png".format(i)), dpi=300)
        plt.show()


if __name__ == '__main__':
    trans = "FLAT5in"
    start = 4000
    end = 6000
    fpath = "C:\\Users\\dionysius\\Desktop\\PURE\\pure\\data\\30deg\\{}".format(trans)
    foc = Signal(fpath, ftype='npy')
    foc.analyze_peak(5,2200,5200)
