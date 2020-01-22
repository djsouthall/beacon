#!/usr/bin/env python3
'''
This is meant to calculate the how many similar events within a run that each event has.
It will also save these values as a percentage of the run for easier comparison accross runs.
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



font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})


def countSimilar(delays,similarity_atol=2.5,verbose=True):
    '''
    Given a set of delays this function will determine which ones are similar to eachother.
    Essentially each event will be given a similarity metric that describes how many other
    events are within a certain number of nanoseconds from it.  Because it considers all
    time delays per event, it should only gain similarity when things are from similar
    sources.

    Each row in delays should denote a single event with 6 columns representing the
    6 baseline time delays for a single polarization.

    Similarity_atol is applied in ns for each delay in the comparison.
    '''
    try:

        n_rows = delays.shape[0]
        

        similarity_count = numpy.zeros(n_rows) #zeros for each column (event)

        delays_rolled = delays.copy()

        for roll in numpy.arange(1,n_rows):
            if verbose == True:
                if roll%100 == 0:
                    sys.stdout.write('(%i/%i)\t\t\t\r'%(roll,n_rows))
                    sys.stdout.flush()
            delays_rolled = numpy.roll(delays_rolled,1,axis=0) #Comparing each event to the next event.
            comparison = numpy.isclose(delays,delays_rolled,atol=similarity_atol)
            if True:
                similarity_count += numpy.all(comparison,axis=1) #All time delays are within tolerance between the two events.
            else:
                similarity_count += numpy.sum(comparison,axis=1) >= 5 #5/6 time delays are within tolerance between the two events.

        return similarity_count
    except Exception as e:
        file.close()
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e

if __name__=="__main__":
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1701

    datapath = os.environ['BEACON_DATA']

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
                eventids = file['eventids'][...]
                if numpy.size(eventids) != 0:
                    print('run = ',run)
                dsets = list(file.keys()) #Existing datasets

                if not numpy.isin('time_delays',dsets):
                    print('time delays not present to perform similarity count in run %i'%run)
                    file.close()
                else:

                    if not numpy.isin('similarity_count',dsets):
                        file.create_group('similarity_count')
                    else:
                        print('similarity_count group already exists in file %s'%filename)

                    time_delays_dsets = list(file['time_delays'].keys())
                    similarity_count_dsets = list(file['similarity_count'].keys())

                    for tdset in time_delays_dsets:
                        if not numpy.isin(tdset,similarity_count_dsets):
                            file['similarity_count'].create_group(tdset)
                        else:
                            print('similarity_count["%s"] group already exists in file %s'%(tdset,filename))

                        tdsets = list(file['similarity_count'][tdset].keys())

                        print('Calculating similarity_count for %s'%tdset)
                        if plot == True:
                            plt.figure()

                        for pol in ['hpol','vpol']:
                            print(pol)
                            print('')
                            if save == True:
                                if not numpy.isin('%s_count'%(pol),tdsets):
                                    file['similarity_count'][tdset].create_dataset('%s_count'%(pol), (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                                else:
                                    print('Values in %s_count of %s will be overwritten by this analysis script.'%(pol,filename))

                                if not numpy.isin('%s_fraction'%(pol),tdsets):
                                    file['similarity_count'][tdset].create_dataset('%s_fraction'%(pol), (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                                else:
                                    print('Values in %s_fraction of %s will be overwritten by this analysis script.'%(pol,filename))

                            print(list(file['time_delays'][tdset].keys()))
                            try:
                                delays = numpy.vstack((file['time_delays'][tdset]['%s_t_%isubtract%i'%(pol,0,1)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%(pol,0,2)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%(pol,0,3)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%(pol,1,2)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%(pol,1,3)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%(pol,2,3)][...])).T
                            except Exception as e:
                                print('\nError in %s'%inspect.stack()[0][3])
                                print('In the past this error has indicated corrupt data.')
                                print(e)
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno)
                                sys.exit(1)
                            #import pdb; pdb.set_trace()
                            #continue
                            similarity_count = countSimilar(delays)
                            similarity_fraction = similarity_count/len(eventids)
                            if save == True:
                                file['similarity_count'][tdset]['%s_count'%(pol)][...] = similarity_count
                                file['similarity_count'][tdset]['%s_fraction'%(pol)][...] = similarity_fraction
                            if plot == True:
                                plt.hist(similarity_count,label=tdset + '\n' + pol,alpha=0.7)

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

