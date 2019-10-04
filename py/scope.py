# -*- coding: utf-8 -*-
# Edit in the VISA address of the oscilloscope in VISA_ADDRESS
# Edit the VISA library path (if not using Windows)
# Grab oscilloscope data using s.grab()
import sys
import visa  # PyVisa info @ http://PyVisa.readthedocs.io/en/stable/
import numpy as np
from os import makedirs, getcwd
from os.path import join, exists, dirname
global VISA_ADDRESS, VISA_PATH, FILENAME
VISA_ADDRESS = 'USB0::0x0957::0x1799::MY52102738::INSTR'  # edit this
VISA_PATH = 'C:\\Windows\\System32\\visa32.dll'  # and this
FILENAME = "scope"  # and this


class Scope:
    def __init__(self, directory, filename=FILENAME):
        self.USER_REQUESTED_POINTS = 8000000
        self.SCOPE_VISA_ADDRESS = VISA_ADDRESS
        self.GLOBAL_TOUT = 10000  # IO time out in milliseconds
        self.BASE_FILE_NAME = filename + "_"
        self.BASE_DIRECTORY = directory
        if not exists(self.BASE_DIRECTORY):
            makedirs(self.BASE_DIRECTORY)
        try:
            self.rm = visa.ResourceManager(VISA_PATH)
        except:
            self.rm = visa.ResourceManager()
        try:
            self.KsInfiniiVisionX = self.rm.open_resource(self.SCOPE_VISA_ADDRESS)
        except Exception:
            print("Unable to connect to oscilloscope at " + str(self.SCOPE_VISA_ADDRESS) + ". Aborting script.\n")
            sys.exit()

        self.KsInfiniiVisionX.timeout = self.GLOBAL_TOUT
        self.KsInfiniiVisionX.clear()
        self.IDN = str(self.KsInfiniiVisionX.query("*IDN?"))
        self.IDN = self.IDN.split(',') # IDN parts are separated by commas, so parse on the commas
        self.MODEL = self.IDN[1]
        if list(self.MODEL[1]) == "9": # This is the test for the PXIe scope, M942xA)
            self.NUMBER_ANALOG_CHS = 2
        else:
            self.NUMBER_ANALOG_CHS = int(self.MODEL[len(self.MODEL)-2])
        if self.NUMBER_ANALOG_CHS == 2:
            self.CHS_LIST = [0,0] # Create empty array to store channel states
        else:
            self.CHS_LIST = [0,0,0,0]
        self.NUMBER_CHANNELS_ON = 0
        self.ANALOGVERTPRES = np.zeros([12])
        self.CH_UNITS = ["BLANK", "BLANK", "BLANK", "BLANK"]
        self.KsInfiniiVisionX.write(":WAVeform:POINts:MODE MAX")
        ch = 1
        for each_value in self.CHS_LIST:
            On_Off = int(self.KsInfiniiVisionX.query(":CHANnel" + str(ch) + ":DISPlay?"))
            if On_Off == 1:
                Channel_Acquired = int(self.KsInfiniiVisionX.query(":WAVeform:SOURce CHANnel" + str(ch) + ";POINts?"))
            else:
                Channel_Acquired = 0
            if Channel_Acquired == 0 or On_Off == 0:
                self.KsInfiniiVisionX.write(":CHANnel" + str(ch) + ":DISPlay OFF")
                self.CHS_LIST[ch-1] = 0
            else:
                self.CHS_LIST[ch-1] = 1
                self.NUMBER_CHANNELS_ON += 1
                Pre = self.KsInfiniiVisionX.query(":WAVeform:PREamble?").split(',')
                self.ANALOGVERTPRES[ch-1]  = float(Pre[7]) # Y INCrement, Voltage difference between data points; Could also be found with :WAVeform:YINCrement? after setting :WAVeform:SOURce
                self.ANALOGVERTPRES[ch+3]  = float(Pre[8]) # Y ORIGin, Voltage at center screen; Could also be found with :WAVeform:YORigin? after setting :WAVeform:SOURce
                self.ANALOGVERTPRES[ch+7]  = float(Pre[9]) # Y REFerence, Specifies the data point where y-origin occurs, always zero; Could also be found with :WAVeform:YREFerence? after setting :WAVeform:SOURce
                self.CH_UNITS[ch-1] = str(self.KsInfiniiVisionX.query(":CHANnel" + str(ch) + ":UNITs?").strip('\n')) # This isn't really needed but is included for completeness
            ch += 1
        del ch, each_value, On_Off, Channel_Acquired

        if self.NUMBER_CHANNELS_ON == 0:
            self.KsInfiniiVisionX.clear()
            self.KsInfiniiVisionX.close()
            sys.exit("No data has been acquired. Properly closing scope and aborting script.")

        ch = 1
        for each_value in self.CHS_LIST:
            if each_value == 1:
                self.FIRST_CHANNEL_ON = ch
                break
            ch += 1
        del ch, each_value
        ch = 1
        for each_value in self.CHS_LIST:
            if each_value == 1:
                self.LAST_CHANNEL_ON = ch
            ch += 1
        del ch, each_value
        self.CHS_ON = []
        ch = 1
        for each_value in self.CHS_LIST:
            if each_value == 1:
                self.CHS_ON.append(int(ch))
            ch += 1
        del ch, each_value
        self.KsInfiniiVisionX.write(":WAVeform:FORMat WORD") # 16 bit word format... or BYTE for 8 bit format - WORD recommended, see more comments below when the data is actually retrieved
        self.KsInfiniiVisionX.write(":WAVeform:BYTeorder LSBFirst") # Explicitly set this to avoid confusion - only applies to WORD FORMat
        self.KsInfiniiVisionX.write(":WAVeform:UNSigned 0") # Explicitly set this to avoid confusion
        self.ACQ_TYPE = str(self.KsInfiniiVisionX.query(":ACQuire:TYPE?")).strip("\n")
        if self.ACQ_TYPE == "AVER" or self.ACQ_TYPE == "HRES": # Don't need to check for both types of mnemonics like this: if ACQ_TYPE == "AVER" or ACQ_TYPE == "AVERage": because the scope ALWAYS returns the short form
            self.POINTS_MODE = "NORMal" # Use for Average and High Resoultion acquisition Types.
        else:
            self.POINTS_MODE = "RAW" # Use for Acq. Type NORMal or PEAK
        self.KsInfiniiVisionX.write(":WAVeform:SOURce CHANnel" + str(self.FIRST_CHANNEL_ON))
        self.KsInfiniiVisionX.write(":WAVeform:POINts MAX") # This command sets the points mode to MAX AND ensures that the maximum # of points to be transferred is set, though they must still be on screen

        self.KsInfiniiVisionX.write(":WAVeform:POINts:MODE " + str(self.POINTS_MODE))
        self.MAX_CURRENTLY_AVAILABLE_POINTS = int(self.KsInfiniiVisionX.query(":WAVeform:POINts?")) # This is the max number of points currently available - this is for on screen data only - Will not change channel to channel.
        if self.USER_REQUESTED_POINTS < 100:
            self.USER_REQUESTED_POINTS = 100
        if self.MAX_CURRENTLY_AVAILABLE_POINTS < 100:
            self.MAX_CURRENTLY_AVAILABLE_POINTS = 100

        if self.USER_REQUESTED_POINTS > self.MAX_CURRENTLY_AVAILABLE_POINTS or self.ACQ_TYPE == "PEAK":
            self.USER_REQUESTED_POINTS = self.MAX_CURRENTLY_AVAILABLE_POINTS
        self.KsInfiniiVisionX.write(":WAVeform:POINts " + str(self.USER_REQUESTED_POINTS))

        self.NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE = int(self.KsInfiniiVisionX.query(":WAVeform:POINts?"))

        Pre = self.KsInfiniiVisionX.query(":WAVeform:PREamble?").split(',') # This does need to be set to a channel that is on, but that is already done... e.g. Pre = self.KsInfiniiVisionX.query(":WAVeform:SOURce CHANnel" + str(FIRST_CHANNEL_ON) + ";PREamble?").split(',')
        self.X_INCrement = float(Pre[4]) # Time difference between data points; Could also be found with :WAVeform:XINCrement? after setting :WAVeform:SOURce
        self.X_ORIGin = float(Pre[5]) # Always the first data point in memory; Could also be found with :WAVeform:XORigin? after setting :WAVeform:SOURce
        self.X_REFerence = float(Pre[6]) # Specifies the data point associated with x-origin; The x-reference point is the first point displayed and XREFerence is always 0.; Could also be found with :WAVeform:XREFerence? after setting :WAVeform:SOURce
        del Pre
        self.DataTime = ((np.linspace(0,self.NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE-1,self.NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE)-self.X_REFerence)*self.X_INCrement)+self.X_ORIGin
        if self.ACQ_TYPE == "PEAK": # This means Peak Detect Acq. Type
            self.DataTime = np.repeat(self.DataTime,2)

        if self.ACQ_TYPE == "PEAK": # This means peak detect mode ### SEE IMPORTANT NOTE ABOUT PEAK DETECT MODE AT VERY END, specific to fast time scales
            self.Wav_Data = np.zeros([2*self.NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE,self.NUMBER_CHANNELS_ON])
        else: # For all other acquistion modes
            self.Wav_Data = np.zeros([self.NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE,self.NUMBER_CHANNELS_ON])
        WFORM = str(self.KsInfiniiVisionX.query(":WAVeform:FORMat?"))
        if WFORM == "BYTE":
            FORMAT_MULTIPLIER = 1
        else: #WFORM == "WORD"
            FORMAT_MULTIPLIER = 2

        if self.ACQ_TYPE == "PEAK":
            POINTS_MULTIPLIER = 2 # Recall that Peak Acq. Type basically doubles the number of points.
        else:
            POINTS_MULTIPLIER = 1

        self.TOTAL_BYTES_TO_XFER = POINTS_MULTIPLIER * self.NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE * FORMAT_MULTIPLIER + 11
        if self.TOTAL_BYTES_TO_XFER >= 400000:
            self.KsInfiniiVisionX.chunk_size = self.TOTAL_BYTES_TO_XFER

    def grab(self, ind):
        i = 0 # index of Wav_data, recall that python indices start at 0, so ch1 is index 0
        for channel_number in self.CHS_ON:
            self.Wav_Data[:,i] = np.array(self.KsInfiniiVisionX.query_binary_values(':WAVeform:SOURce CHANnel' + str(channel_number) + ';DATA?', "h", False)) # See also: https://PyVisa.readthedocs.io/en/stable/rvalues.html#reading-binary-values
            self.Wav_Data[:,i] = ((self.Wav_Data[:,i]-self.ANALOGVERTPRES[channel_number+7])*self.ANALOGVERTPRES[channel_number-1])+self.ANALOGVERTPRES[channel_number+3]
            i +=1

        if self.TOTAL_BYTES_TO_XFER >= 400000:
            self.KsInfiniiVisionX.chunk_size = 20480
#        if len(self.fnames) >= 1:
#            recent = self.fnames[-1]
#            pre = len(self.BASE_FILE_NAME)
#            suf = -4
#            try:
#                i = int(recent[pre:suf]) + 1
#            except:
#                raise TypeError
#        else:
#            i = 0
        filename = join(self.BASE_DIRECTORY, self.BASE_FILE_NAME + "{0}".format(ind) + ".npy")
        with open(filename, 'wb') as filehandle: # wb means open for writing in binary; can overwrite
            np.save(filehandle, np.insert(self.Wav_Data, 0, self.DataTime, axis=1))
        arr = np.insert(self.Wav_Data, 0, self.DataTime, axis=1)
        return arr

    def close(self):
        self.KsInfiniiVisionX.clear()
        self.KsInfiniiVisionX.close()


if __name__ == '__main__':
    d = join(dirname(getcwd()), "data")
    s = Scope(d)
    s.grab("test")  #  mm
