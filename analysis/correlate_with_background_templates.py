#!/usr/bin/env python3
'''
Using a tempalte created using the match_77MHz script, this will cross correlate the signals in
a particular run with it and store the maximum correlation value.  This could be used to cut out
signals with high correlation values.  
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from objects.fftmath import TimeDelayCalculator
from tools.data_handler import createFile

import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import numpy
import sys
import math
import matplotlib
from scipy.fftpack import fft
import datetime as dt
import inspect
from ast import literal_eval
import astropy.units as apu
from astropy.coordinates import SkyCoord
import itertools
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


datapath = os.environ['BEACON_DATA']

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})



if __name__=="__main__":
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1701

    try:
        run = int(run)
        save = True
        plot = False
        reader = Reader(datapath,run)
        try:
            print(reader.status())
        except Exception as e:
            print('Status Tree not present.  Returning Error.')
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
        if filename is not None:
            with h5py.File(filename, 'a') as file:
                #eventids with rf trigger
                for event_type in [0,1]:
                    eventids = file['eventids'][...]
                    if numpy.size(eventids) != 0:
                        print('run = ',run)

                    dsets = list(file.keys()) #Existing datasets

                    correlation_values = numpy.zeros((file.attrs['N'],8))
                    if save == True:
                        if not numpy.isin('correlation_77MHz_type%i'%(event_type),dsets):
                            file.create_dataset('correlation_77MHz_type%i'%(event_type), (file.attrs['N'],8), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                        else:
                            print('correlation_77MHz_type%i group already exists in file %s'%(event_type,filename))

                    template = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR']+'templates/template_77MHz_type1.csv',delimiter=',')
                    template_std = numpy.std(template,axis=1)*len(reader.t()) #Normalization factor

                    for eventid in eventids:
                        if eventid%1000 == 0:
                            sys.stdout.write('\r%i/%i'%(eventid,len(eventids)-1))
                            sys.stdout.flush()
                        reader.setEntry(eventid)
                        for antenna in range(8):
                            signal = reader.wf(antenna)
                            std = numpy.std(signal)
                            c = scipy.signal.correlate(template[antenna],signal)/(std*template_std[antenna])
                            correlation_values[eventid,antenna] = numpy.max(c)

                    if plot == True:
                        plt.figure()
                        plt.hist(numpy.max(correlation_values,axis=1),bins=50)

                    if save == True:
                        file['correlation_77MHz_type%i'%(event_type)][...] = correlation_values
                file.close()
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)
    sys.exit(0)

