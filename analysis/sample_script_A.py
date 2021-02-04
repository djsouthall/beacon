'''
This script is intended to demonstrate loading waveforms with the Reader class, as well as some other basic
functionalities.  
'''

#General Imports
import numpy
import itertools
import os
import sys
import inspect

#BEACON Imports
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
import tools.get_plane_tracks as pt
from tools.fftmath import TimeDelayCalculator #This is a daughter class of FFTPrepper, and has all of the functions plus some.
from tools.correlator import Correlator #This is used for generating maps.  Be aware that it will assume the antenna positions are calibrated based upon the default_deploy defined in tools/info.py

#Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


#Settings
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()
datapath = os.environ['BEACON_DATA'] #must be appropriately defined in the environment variables.

'''
Commonly used run numbers:

Pulsing data from the 2019 deployment was taken in 3 seperate runs with different source locations.  This information
will be loaded in the main function below, but I will note upfront that the run numbers are:
Site 1/A (South East, ~500m): Run 1507
Site 2/B (East, Far ~1km): Run 1509
Site 3/C (North East, ~500m): Run 1511

The legitimacy of the data in runs varies depending on if certain antennas are still active, but one run that is commonly
used when testing code is run 1650.  This is a run that exists under "normal operation", and is chosen arbitrarily.
'''

'''
Script Structure:

Typically I will define functions outside of the __main__ portion, and then the __main__ portion is the actual
script portion that will run, usually then calling the functions.  Here I have one function defined with returns the
trigger IDs per event as an example.

This script is intended to demonstrate certain features/common use cases of the tools.  Individual sections
may be tabbed over behind a "if True" or "if False" statement.  This is simply intended to allow the user to
"turn off" certain portions of the code easily.  The code in that black will run if it is under "if True" and
will not run if it is under "if False".
'''

def loadTriggerTypes(reader):
    '''
    Will get a list of trigger types corresponding to all eventids for the given reader
    trigger_type:
    1 Software
    2 RF
    3 GPS

    This function is originally defined in the tools/data_handler.py script, but is copied here to demonstrate it's use.
    This function is one of the very few examples of needing to access the underlying ROOT framework to obtain info.
    Typically in normal analysis this function would've already been run and the event types are stored in the
    generated analysis script.  
    '''


    try:
        N = reader.head_tree.Draw("trigger_type","","goff") 
        trigger_type = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
        return trigger_type
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print('Error while trying to copy header elements to attrs.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e


if __name__ == '__main__':
    plt.close('all') #Uncomment this if you want figures to be closed before this is run (helps if running multiple times in a row to avoid plot congestion).
    


    '''
    Here we pick a run, create a reader, get the eventids, get which eventids correspond to which trigger type,
    then plot 1 event from each trigger type.
    '''
    if True:
        plot_N_per_type = 2 #The number of events to plot her trigger type.  Meant to demonstrate what looping over events might look like.

        #Get run and events you want to look at.
        run = 1650
        #Create a Reader object for the specific run. 
        reader = Reader(datapath,run)
        print('The run associated with this reader is:')
        print(reader.run)
        print('This run has %i events'%(reader.N()))
        eventids = numpy.arange(reader.N())
        trigger_type = loadTriggerTypes(reader)

        times = reader.t() #The times of a waveform in ns.  Not upsampled.

        for trig_type in [1,2,3]:
            print('Plotting %i eventids of trig type %i'%(plot_N_per_type,trig_type))
            trig_eventids = eventids[trigger_type == trig_type] #All eventids of this trig type
            trig_eventids = numpy.sort(numpy.random.choice(trig_eventids,2)) #Randomly choosing a subset and sorting for faster loading of events

            for eventid in trig_eventids:
                reader.setEntry(eventid) #Actually makes the wf function adress the correct event.


                fig = plt.figure()
                fig.canvas.set_window_title('Run %i event %i, Trig Type %i'%(run, eventid, trig_type))
                plt.title('Run %i event %i, Trig Type %i'%(run, eventid, trig_type))
                plt.subplot(2,1,1)
                plt.ylabel('Hpol Adu')
                plt.subplot(2,1,2)
                plt.ylabel('Vpol Adu')
                plt.xlabel('t (ns)')
                for channel in range(8):
                    plt.subplot(2,1,channel%2 + 1)

                    wf = reader.wf(channel)
                    plt.plot(times, wf,label='Ch %i'%channel)

                plt.subplot(2,1,1)
                plt.legend(loc = 'upper left')
                plt.subplot(2,1,2)
                plt.legend(loc = 'upper left')