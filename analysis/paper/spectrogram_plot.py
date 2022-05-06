import sys
import os
import gc
import inspect
import h5py
import copy
from pprint import pprint
import textwrap
from multiprocessing import Process

import numpy
import scipy
import scipy.signal

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
# from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
# from beacon.tools.data_handler import createFile
from beacon.tools.spectrogram import getSpectData

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, FuncFormatter
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.dates as mdates
import time
from datetime import datetime
import pytz

datapath = os.environ['BEACON_DATA']


if __name__ == '__main__':
    plt.close('all')

    #runs =  numpy.arange(6000,6020)+2*20
    #runs = [6011, 6030, 6049]
    # run = int(sys.argv[1]) if len(sys.argv) > 1 else 5911 #Selects which run to examine
    run = int(6049)

    event_limit = 30000
    channels = numpy.array([0])

    reader, freqs, spectra_dbish_binned, time_range = getSpectData(datapath,run,event_limit,bin_size=10,trigger_type=1,group_fft=False, channels=channels)

    gc.collect()

    cmaps=[ 'coolwarm', 'Greys','viridis','plasma','inferno']#['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']


    time_offset = 0.5 #To remove the run startup events

    #Label Tuning
    label_fontsize = 18
    box_pad = 0.2
    rounding_size = 0.2


    plt.rc('xtick',labelsize=label_fontsize)
    plt.rc('ytick',labelsize=label_fontsize)



    for cmap in cmaps:
        for channel in channels:
            fig = plt.figure(figsize=(12,6))
            ax = plt.gca()
            #plt.title('Run %i, Channel %i'%(run,channel),fontsize=28)
            plt.imshow(spectra_dbish_binned['ch%i'%channel],extent = [time_range[0]-time_offset, time_range[1] - time_offset, min(freqs)/1e6,max(freqs)/1e6],aspect='auto',cmap=cmap)
            plt.xlim(0, time_range[1] - time_offset)
            plt.ylim(0,100)
            #plt.xlim(0,100)
            plt.ylabel('Frequency (MHz)',fontsize=20)
            plt.xlabel('Readout Time (min)',fontsize=20)
            cb = plt.colorbar()
            cb.set_label('dB (arb)',fontsize=20)

            #Label 48 MHz blip 
            arrow_head_xy = (26.05 + 0.15, 48.5 - 0.1)#10.88
            text_xy = (32,30)


            ann = ax.annotate("48 MHz Blip",
                  xy=arrow_head_xy, xycoords='data',
                  xytext=text_xy, textcoords='data',
                  size=label_fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=+0.2",
                                  fc="w"),
                  )

            #Label 42 MHz blip 
            arrow_head_xy = (20.17 + 0.1, 42.2 - 0.5)#10.88
            text_xy = (28,10)


            ann = ax.annotate("42 MHz Blip",
                  xy=arrow_head_xy, xycoords='data',
                  xytext=text_xy, textcoords='data',
                  size=label_fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=-0.2",
                                  fc="w"),
                  )

            #Label CW  
            arrow_head_xy = (23, 88.5)#10.88
            text_xy = (28,90)

            ann = ax.annotate("CW",
                  xy=arrow_head_xy, xycoords='data',
                  xytext=text_xy, textcoords='data',
                  size=label_fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=-0.1",
                                  fc="w"),
                  )

            arrow_head_xy = (23, 92)
            text_xy = (28,90)

            ann = ax.annotate("CW",
                  xy=arrow_head_xy, xycoords='data',
                  xytext=text_xy, textcoords='data',
                  size=label_fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=+0.1",
                                  fc="w"),
                  )


            # Add label of TV band

            tv_top = 60
            tv_bottom = 54
            tv_middle = (tv_top + tv_bottom)/2.0
            x = 28
            arrow_head_xy = (x,tv_middle)
            text_xy = (x+5,tv_middle)

            ann = ax.annotate("TV Band",
                  xy=arrow_head_xy, xycoords='data',
                  xytext=text_xy, textcoords='data',
                  size=label_fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(  arrowstyle="-[,widthB=0.55, lengthB=0.2, angleB=0",
                                    fc="w"),
                  )
            plt.tight_layout()
            fig.savefig('./figures/spectrogram/spectrogram_run%i_ch%i_%s.pdf'%(run, channel, cmap), dpi=300)



            




    #,cmap='RdGy'