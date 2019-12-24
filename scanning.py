# -*- coding: utf-8 -*-
import numpy as np
from scope import Scope
from path import Path
from os import listdir, getcwd, remove
from os.path import join, isfile, dirname
import serial
from time import sleep
import serial.tools.list_ports
global TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT
global FILENAME, SCAN_FOLDER, min_step, arduino
DATA_FOLDER = "1D-3FOC3in-back-1in-bigangle"
min_step = 4e-4
FILENAME = "scope"
SCAN_FOLDER = Path(getcwd())
#arduino = serial.Serial('/dev/cu.usbmodem14201', 9600)
#arduino = serial.Serial('COM5', 9600)
arduino = None
ports = list(serial.tools.list_ports.comports())
for p in ports:
    if "Arduino" in p[1]:
        arduino = serial.Serial(p[0], 9600)
if arduino is None:
    print('No Arduino found')


def d2s(dist):
    # Converts distance in metres to number of steps
    return int(dist//min_step)


def step(command):
    # Sends bytes to arduino to signal step motor movement
    # 1: top motor forward -- black tape side Y axis
    # 2: top motor backward
    # 3: bottom motor backward
    # 4: bottom motor forward -- black tape side X axis
    sleep(1.5)
    try:
        arduino.write(str.encode("{}".format(command)))
    except TypeError:
        print("Command is not 1-4")
    except:
        print("Unexpected Error")


def clear_scan_folder():
    for f in listdir(SCAN_FOLDER):
        if isfile(join(SCAN_FOLDER, f)) and (FILENAME in f):
            remove(join(SCAN_FOLDER, f))


def load_arr(output_folder=SCAN_FOLDER):
    ftarr = join(output_folder, "tarr.npy")
    fvarr = join(output_folder, "varr.npy")
    with open(ftarr, 'rb') as rd:
        tarr = np.load(rd, allow_pickle=False)
    with open(fvarr, 'rb') as rd:
        varr = np.load(rd, allow_pickle=False)
    return tarr, varr


class Scan:
    def __init__(self, DIMENSIONS, START_POS, FOLDER=SCAN_FOLDER):
        self.SCAN_FOLDER = join(dirname(getcwd()), "data", FOLDER)
        self.TOP_LEFT = (0, 0)
        self.TOP_RIGHT = (0, -1)
        self.BOTTOM_LEFT = (-1, 0)
        self.BOTTOM_RIGHT = (-1, -1)
        self.left = -1
        self.right = 1
        self.up = 999
        self.down = -999
        self.STEP_DICT = {self.left: "-x", self.right: "x", self.up: "+y",
                          self.down: "-y"}
        POS_DICT = {"top left": self.TOP_LEFT, "top right": self.TOP_RIGHT,
                    "bottom left": self.BOTTOM_LEFT, "bottom right": self.BOTTOM_RIGHT}
        self.arr = np.array([])
        self.out_arr = np.array([], dtype=int)
        if DIMENSIONS[0] == 0 and DIMENSIONS[1] != 0:
            self.SAMPLE_DIMENSIONS = (1, d2s(DIMENSIONS[1])+1)
        elif DIMENSIONS[0] != 0 and DIMENSIONS[1] == 0:
            self.SAMPLE_DIMENSIONS = (d2s(DIMENSIONS[0])+1, 1)
        else:
            self.SAMPLE_DIMENSIONS = (d2s(DIMENSIONS[0])+1, d2s(DIMENSIONS[1])+1)
        self.START_POS = POS_DICT[START_POS]
        self.scope = Scope(SCAN_FOLDER, filename=FILENAME)
        clear_scan_folder()
        self.tstep = 0
        self.run()
        clear_scan_folder()

    def STEP(self, DIRECTION='+x'):
        try:
            if DIRECTION in ['x', 'X', '+x', '+X']:
                step(4)
            elif DIRECTION in ['-x', '-X']:
                step(3)
            elif DIRECTION in ['y', 'Y', '+y', '+Y']:
                step(1)
            elif DIRECTION in ['-y', '-Y']:
                step(2)
        except ValueError:
            print("DIRECTION is not type str")

    def STEP_PARAM(self):
        SAMPLE_DIMENSIONS = self.SAMPLE_DIMENSIONS
        if self.START_POS == self.TOP_LEFT or self.START_POS == self.TOP_RIGHT:
            self.VERTICAL_STEP = -999
        elif self.START_POS == self.BOTTOM_LEFT or self.START_POS == self.BOTTOM_RIGHT:
            self.VERTICAL_STEP = 999
        else:
            print("Unexpected Error")
        arr = np.zeros(SAMPLE_DIMENSIONS)
        if self.START_POS == self.TOP_RIGHT:
            for y in range(SAMPLE_DIMENSIONS[0]):
                if (y % 2) == 0:
                    arr[y, 1:] = -1
                    arr[y, 0] = self.VERTICAL_STEP
                else:
                    arr[y, 0:-1] = 1
                    arr[y, -1] = self.VERTICAL_STEP
                if SAMPLE_DIMENSIONS[0] % 2 != 0:
                    ENDPOS = (-1, 0)
                else:
                    ENDPOS = (-1, -1)
            arr[ENDPOS] = 0
        elif self.START_POS == self.TOP_LEFT:
            for y in range(SAMPLE_DIMENSIONS[0]):
                if (y % 2) == 0:
                    arr[y, 0:-1] = 1
                    arr[y, -1] = self.VERTICAL_STEP
                else:
                    arr[y, 1:] = -1
                    arr[y, 0] = self.VERTICAL_STEP
                if SAMPLE_DIMENSIONS[0] % 2 == 0:
                    ENDPOS = (-1, 0)
                else:
                    ENDPOS = (-1, -1)
            arr[ENDPOS] = 0
        elif self.START_POS == self.BOTTOM_LEFT:
            for y in range(SAMPLE_DIMENSIONS[0]):
                k = SAMPLE_DIMENSIONS[0]-1-y
                if (y % 2) == 0:
                    arr[k, :] = 1
                    arr[k, -1] = self.VERTICAL_STEP
                else:
                    arr[k, 1:] = -1
                    arr[k, 0] = self.VERTICAL_STEP
                if SAMPLE_DIMENSIONS[0] % 2 != 0:
                    ENDPOS = (0, -1)
                else:
                    ENDPOS = (0, 0)
            arr[ENDPOS] = 0
        elif self.START_POS == self.BOTTOM_RIGHT:
            for y in range(SAMPLE_DIMENSIONS[0]):
                k = SAMPLE_DIMENSIONS[0]-1-y
                if (y % 2) == 0:
                    arr[k, 1:] = -1
                    arr[k, 0] = self.VERTICAL_STEP
                else:
                    arr[k, 0:-1] = 1
                    arr[k, -1] = self.VERTICAL_STEP
                if SAMPLE_DIMENSIONS[0] % 2 == 0:
                    ENDPOS = (0, -1)
                else:
                    ENDPOS = (0, 0)
            arr[ENDPOS] = 0
        try:
            self.ENDPOS = ENDPOS
        except NameError:
            print("ENDPOS not defined")
        self.arr = np.copy(arr)

    def run(self):
        self.STEP_PARAM()
        out_arr = np.empty(np.shape(self.arr))
        pos = self.START_POS
        i = 0
        while self.arr[pos] != 0:
            V = self.arr[pos]
            self.scope.grab(i)
            out_arr[pos] = i
            self.STEP(self.STEP_DICT[V])
            if V == self.left:
                pos = (pos[0], pos[1] - 1)
            elif V == self.right:
                pos = (pos[0], pos[1] + 1)
            elif V == self.up:
                pos = (pos[0] - 1, pos[1])
            elif V == self.down:
                pos = (pos[0] + 1, pos[1])
            i += 1
            if self.arr[pos] == 0:
                self.scope.grab(i)
                out_arr[pos] = i
        self.out_arr = out_arr
        self.tarr, self.varr = self.sig2arr(self.out_arr)
        self.save_arr()
        self.tstep = np.abs(np.mean(self.tarr[1:, 0, 0] - self.tarr[:-1, 0, 0]))
        return self.tarr, self.varr

    def sig2arr(self, out_arr):
        with open(join(SCAN_FOLDER, "{}_0.npy".format(FILENAME)), "rb") as f:
            SIGNAL_LENGTH = len(np.load(f)[:, 0])
        START = 0
        END = SIGNAL_LENGTH
        tarr = np.empty((END - START, self.SAMPLE_DIMENSIONS[0],
                         self.SAMPLE_DIMENSIONS[1]), dtype=float)
        varr = np.empty((END - START, self.SAMPLE_DIMENSIONS[0],
                         self.SAMPLE_DIMENSIONS[1]), dtype=float)
        for y in range(self.SAMPLE_DIMENSIONS[0]):
            for x in range(self.SAMPLE_DIMENSIONS[1]):
                file = "{0}_{1}.npy".format(FILENAME, int(out_arr[y, x]))
                with open(join(SCAN_FOLDER, file), "rb") as npobj:
                    arr = np.load(npobj)
                    tarr[:, y, x] = arr[START:END, 0]
                    varr[:, y, x] = arr[START:END, 1]
        return tarr, varr

    def save_arr(self, output_folder=SCAN_FOLDER):
        output_tarr = join(output_folder, "tarr.npy")
        output_varr = join(output_folder, "varr.npy")
        with open(output_tarr, 'wb') as wr:
            np.save(self.tarr, wr, allow_pickle=False)
        with open(output_varr, 'wb') as wr:
            np.save(self.varr, wr, allow_pickle=False)
        print("Done saving arrays")


if __name__ == '__main__':
    foc = Scan(DIMENSIONS=(0, 0.085), START_POS="bottom left")

if arduino is not None:
    arduino.close()