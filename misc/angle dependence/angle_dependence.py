# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:39:17 2019

@author: dionysius
angle | micrometer reading
0	24.2
1	22.52274209
2	20.84548419
3	19.16822628
4	17.49096838
5	15.81371047
6	14.13645257
7	12.45919466
8	10.78193676
9	9.10467885
10	7.42742095
11	5.75016304
12	4.07290514
13	2.39564723
14	0.71838933
15	-0.95886858
"""
# Import Signal class from data_analysis.py in the same directory
from data_analysis import Signal
from os import listdir, getcwd, makedirs
from os.path import join, isfile, dirname, exists
import pickle  # saving numpy binary data to files
import matplotlib.pyplot as plt  # plotting
import matplotlib.cm as cm  # colormap
import matplotlib
import numpy as np

global tot_folder, obj_folder, BSCAN_folder, timestep  # global variables
timestep = 3.999999999997929e-08  # oscilloscope time interval between data points
obj_folder = join(dirname(dirname(getcwd())), "data", "obj")  # save folder for arrays
tot_folder = join(dirname(dirname(getcwd())), "scans")  # save folder for angle dependence graph
BSCAN_folder = join(dirname(dirname(getcwd())), "scans\\BSCAN")  # save folder for BSCAN

if not exists(BSCAN_folder):
    makedirs(BSCAN_folder)
if not exists(obj_folder):
    makedirs(obj_folder)
if not exists(tot_folder):
    makedirs(tot_folder)


def init():
    # Initializes Signal objects using signal data (npy, npz, or csv)
    data_folder_path = join(dirname(getcwd()), "data", "ANGLE DEPENDENCE", "0deg")
    flat15_path = join(data_folder_path, 'FLAT_15cm')
    foc15_path = join(data_folder_path, '3FOC15cm')
    flat9_path = join(data_folder_path, 'FLAT9cm')
    foc9_path = join(data_folder_path, '3FOC9cm')
    fpath15 = join(data_folder_path, '1-5FOC15cm')
    fpath9 = join(data_folder_path, '1-5FOC9cm')

    flat = Signal(flat15_path, ftype='npz')
    foc = Signal(foc15_path, ftype='npz')
    foc2 = Signal(foc9_path, ftype='npy')
    flat2 = Signal(flat9_path, ftype='npy')
    foc15 = Signal(fpath15)
    foc9 = Signal(fpath9)
    obj_list = [flat, foc, foc2, flat2, foc15, foc9]
    for obj in obj_list:
        obj.write_all()
        save_obj(obj)


def BSCAN(signal_data, title='B-Scan', domain=(0, -1), DISPLAY=True, SAVE=False, vmin=0, vmax=1):
    # Performs B-scan for given set of measurements
    # signal_data is a list of Signal objects created in the Transducer class
    # domain is a tuple with start and end points of the signal
    # vmin/vmax is min/max of color range for imshow()
    START = domain[0] # offset the start of the signal; don't want to include transmitted part
    END = domain[1]
    arr = np.abs(signal_data[0].xy[START:END, 1])  # take abs of first signal
    max_val = 0
    for sig in signal_data:
        V = np.amax(sig.xy)
        if V > max_val:
            max_val = V
    arr /= max_val
    for i in range(1, len(signal_data)):  # start from second index
        next_arr = np.abs(signal_data[i].xy[START:END, 1])  # abs of the next signal
        next_arr /= max_val
        arr = np.vstack((arr, next_arr))  # add to previous array
        bscan = np.transpose(arr)  # take transpose, rename variable
        plt.ioff()
        fig = plt.figure(figsize=[14, 8])
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = np.arange(timestep*START, timestep*END, timestep*((END - START)//10))
        minor_ticks = np.arange(timestep*START, timestep*END, timestep*((END - START)//50))
        x_ticks = np.arange(0, 17, 1)
        ax.set_title(title)
        ax.imshow(bscan, cmap='gray', origin='upper', aspect='auto', alpha=.9, extent=[0, 16, timestep*END, timestep*START], vmin=vmin, vmax=vmax)
        ax.set_xlabel('angle (degrees)')
        ax.set_ylabel('time (s)')
        ax.set_xticks(x_ticks)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True, axis='y', which="major", alpha=.5)
        ax.grid(True, axis='y', which="minor", alpha=.2, linestyle="--")
        ax.grid(True, axis='x', which="major", alpha=.2, linestyle="--")
    if SAVE is True:
        plt.savefig(join(BSCAN_folder, title+'.png'))
    if DISPLAY is True:
        plt.show(fig)
    elif DISPLAY is False:
        plt.close(fig)
    plt.ion()


def save_obj(obj, output_folder=obj_folder):
    # Save numpy binary to .pkl file
    name = obj.name + ".pkl"
    output = join(output_folder, name)
    with open(output, 'wb') as wr:
        pickle.dump(obj, wr, pickle.DEFAULT_PROTOCOL)

    print("Done saving: {}".format(name))


def load_obj(obj_name, folder=obj_folder):
    # Load Transducer data saved in .pkl
    if obj_name[-4:] != '.pkl':
        obj_name = obj_name + '.pkl'

    output = join(folder, obj_name)
    with open(output, 'rb') as rd:
        transducer = pickle.load(rd)
    return transducer


def graph_totals(SAVE=False, DISPLAY=True):
     ## Plot graph of peak values vs angle
     obj_list = [load_obj(f) for f in listdir(obj_folder) if
                 isfile(join(obj_folder, f)) and "signal_data" not in f
                 and "1_5" not in f]
     font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
     matplotlib.rc('font', **font)
     fig = plt.figure(figsize=[15,14])
     plt.title("Echo Intensity vs Incident Angle")
     plt.xlabel('incident angle (degrees)')
     plt.ylabel('relative peak echo voltages')
     colors = cm.tab20(np.linspace(0, 1, len(obj_list[0].deg)))
     for i in range(len(obj_list)):
          obj = obj_list[i]
          x = obj.peak_totals
          rescaled = x / max(x) ##(x-min(x))/(max(x)-min(x))
          c = 2 ## this changes the colors used, [1,3]
          plt.scatter(obj.deg, rescaled, color = colors[c*i], alpha=.6, label = obj.name)
          plt.plot(obj.deg, rescaled, color=colors[c*i], ls=":", alpha=.6)

     plt.legend()
     if SAVE==True:
          plt.savefig(join(tot_folder, title+".png"), dpi=200)
     if DISPLAY==False:
          plt.close(fig)
     elif DISPLAY==True:
          plt.show(fig)


if __name__ == '__main__':
#    init()
    graph_totals()


#######################################################################################
#######################################################################################
# Appendix
#
#def save_bscans():
#     ## This function keeps all B-scan parameters for each dataset
#     BSCAN(load_obj("1_5FOC_9cm.pkl").signal_data, title="1.5 Focused 9 cm depth", domain=(28100,28800), vmax=.7, SAVE=True, DISPLAY=False)
#     BSCAN(load_obj("1_5FOC_15cm.pkl").signal_data, title="1.5 Focused 15 cm depth", domain=(30500,31200), vmax=.5, SAVE=True, DISPLAY=False)
#     BSCAN(load_obj("3FOC_9cm.pkl").signal_data, title="3 in Focused 9 cm depth", domain=(25700,26400), SAVE=True, DISPLAY=False)
#     BSCAN(load_obj("3FOC_15cm.pkl").signal_data, title="3 in Focused 15 cm depth", domain=(24600,25150), SAVE=True, DISPLAY=False)
#     BSCAN(load_obj("FLAT_9cm.pkl").signal_data, title="Flat 9 cm depth", domain=(20000, 30000), SAVE=True, DISPLAY=False)
#     BSCAN(load_obj("FLAT_9cm.pkl").signal_data, title="Flat 9 cm depth", domain=(25650, 26350), SAVE=True, DISPLAY=False)
#     BSCAN(load_obj("FLAT_15cm.pkl").signal_data, title="Flat 15 cm depth", domain=(24550, 25125), SAVE=True, DISPLAY=False)
#
#
