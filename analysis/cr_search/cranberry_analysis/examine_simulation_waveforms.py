#!/usr/bin/env python3
'''
A quick script to load in the waveforms that Andrew has created.
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint
import glob

import numpy
import scipy
import scipy.signal
import time

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.flipbook_reader import flipbookToDict, concatenateFlipbookToArray, concatenateFlipbookToDict
import  beacon.tools.get_plane_tracks as pt
from tools.airplane_traffic_loader import getDataFrames, getFileNamesFromTimestamps

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
plt.ion()


if __name__ == '__main__':
    plt.close('all')
    waveform_paths = glob.glob(os.path.join(os.environ['BEACON_ANALYSIS_DIR'] , 'analysis', 'sept2021-week1-analysis', 'andrew_sample_waveforms', '2-11-2022') + '/*.npy')

    for path in waveform_paths:
        plt.figure()
        with open(path, 'rb') as f:
            args = numpy.load(f, allow_pickle=True)
            channel0 = numpy.load(f)
            t = 2.0*numpy.arange(len(channel0))
            plt.plot(t, channel0)
            channel1 = numpy.load(f)
            plt.plot(t, channel1)
            channel2 = numpy.load(f)
            plt.plot(t, channel2)
            channel3 = numpy.load(f)
            plt.plot(t, channel3)
            channel4 = numpy.load(f)
            plt.plot(t, channel4)
            channel5 = numpy.load(f)
            plt.plot(t, channel5)
            channel6 = numpy.load(f)
            plt.plot(t, channel6)
            channel7 = numpy.load(f)
            plt.plot(t, channel7)


        data = numpy.load(path,allow_pickle=True)

    # wf_list = ['event000088.npy']