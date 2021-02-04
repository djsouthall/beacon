'''
This script is intended to demonstrate some of the tools that can be used for analyzing BEACON data.
It is highly recommended that the user works on a CLI such as ipython, where they can run this script
from and stay in the python namespace, such that they can play with the reader and events once the
script is run.
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

This script is intended to demonstrate certain features/common use cases of the tools.  Individual sections
may be tabbed over behind a "if True" or "if False" statement.  This is simply intended to allow the user to
"turn off" certain portions of the code easily.  The code in that black will run if it is under "if True" and
will not run if it is under "if False".
'''


if __name__ == '__main__':
    plt.close('all') #Uncomment this if you want figures to be closed before this is run (helps if running multiple times in a row to avoid plot congestion).
    


    '''
    Loading pulsing events from site 1, plotting them with no filters applied, then plotting them with a different
    filter applied to each of the antennas.  Finally the time delays will be calculated and printed, with the signals
    being plotted and aligned based on the calculated time delays.
    '''
    if True:
        #Get run and events you want to look at.
        run = 1507
        eventids = info.loadPulserEventids()['run%i'%run]['hpol'] #produces a dictionary with all runs, and events seperated into polarization.  For simplicity I am specifying that I want only the eventids of hpol events from run1507. 

        #Below are a common set of input variables that are used across multiple tools.
        #As this script is set up, the filter parameters are only used on the second Time Delay Calculator.
        #This is just to demonstrate the difference.
        final_corr_length = 2**17

        crit_freq_low_pass_MHz = [90,75,75,75,75,75,75,75] #Filters here are attempting to correct for differences in signals from pulsers.
        low_pass_filter_order = [8,8,14,8,12,8,8,8]

        crit_freq_high_pass_MHz = None
        high_pass_filter_order = None

        apply_phase_response = True
        hilbert = False

        #Load antenna position information from the info.py script
        origin = info.loadAntennaZeroLocation()#Assuming default_deploy
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()#Assuming default_deploy

        #Create a Reader object for the specific run. 
        reader = Reader(datapath,run)
        print('The run associated with this reader is:')
        print(reader.run)
        print('This run has %i events'%(reader.N()))

        #Create a TimeDelayCalculator object for the specified run. Note that if the above parameters haven't been change
        tdc_raw = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None,plot_filters=False,apply_phase_response=False)
        #Plot raw event
        tdc_raw.plotEvent(eventids[0], channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None, additional_title_text=None)

        tdc_filtered = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,plot_filters=True,apply_phase_response=apply_phase_response)
        tdc_filtered.plotEvent(eventids[0], channels=[0,1,2,3,4,5,6,7], apply_filter=True, hilbert=False, sine_subtract=False, apply_tukey=None, additional_title_text='Filtered')
        tdc_filtered.plotEvent(eventids[0], channels=[0,1,2,3,4,5,6,7], apply_filter=True, hilbert=True, sine_subtract=False, apply_tukey=None, additional_title_text='Hilbert')


    '''
    Loading an event that is known to have CW, showing the event.  The loading the same event once a sine subtract
    object has been added, and plotting that.  Additionally a map will be generated for the event, also with and 
    without the CW being subtracted.
    '''
    if False:
        #Setting up for a run with known CW
        run = 1650
        eventid = 113479#89436#89436,21619,113479

        #Below are a common set of input variables that are used across multiple tools.
        final_corr_length = 2**17

        crit_freq_low_pass_MHz = None
        low_pass_filter_order = None

        crit_freq_high_pass_MHz = None
        high_pass_filter_order = None

        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.09
        sine_subtract_percent = 0.03

        apply_phase_response = True
        hilbert = False

        #Create a Reader object for the specific run. 
        reader = Reader(datapath,run)
        print('The run associated with this reader is:')
        print(reader.run)
        print('This run has %i events'%(reader.N()))

        #Create a TimeDelayCalculator object for the specified run. 
        tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,plot_filters=False,apply_phase_response=apply_phase_response)
        if sine_subtract:
            #Sine subtract objects are added to the calculator using the addSineSubtract function.  Multiple can be added if you wish to make more targeted searches.
            tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)


        tdc.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None, additional_title_text=None)
        tdc.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=True, apply_tukey=None, additional_title_text=None)

        #Create a Correlator object and produce maps for this event.

        cor = Correlator(reader,  upsample=final_corr_length, n_phi=720, n_theta=720,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=True, tukey=False, sine_subtract=sine_subtract)

        raw_result = cor.map(eventid, 'hpol', center_dir='E', plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=0,mollweide=True,circle_zenith=None,circle_az=None) #This one does not have CW subtracted.
        print('Note that the map should be intereactive, such that if you double click on a point it will produce waveforms aligned based on that directions time delays.')
        if sine_subtract:
            #Sine subtract objects are added to the calculator using the addSineSubtract function.  Multiple can be added if you wish to make more targeted searches.
            #Note that in this case the correlator class defines it's own internal FFTPrepper object, which is what loads the waveforms.  The sine subtract is being added to that (cor.prep).
            cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
            raw_result = cor.map(eventid, 'hpol', center_dir='E', plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=0,mollweide=True,circle_zenith=None,circle_az=None) #This one should have CW subtracted.






