'''
The code below is from Kaeli.  x = East.  This will give the locations of the antennas relative to antenna zero.
The location of the pulser can also be determined using this code. 
'''

import numpy
import scipy.spatial
import scipy.signal
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import pymap3d as pm
import itertools
pairs = list(itertools.combinations((0,1,2,3), 2))
plt.ion()


c = 2.99700e8 #m/s

if __name__ == '__main__':
    try:
        if len(sys.argv) == 2:
            run_label = 'run%i'%int(sys.argv[1])
        else:
            print('No run number given.  Defaulting to 793')
            run_label = 'run793'
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()
        pulser_location = info.loadPulserLocationsENU()[run_label] #ENU

        print('Pulser Physical Location:')
        print(pulser_location,'\n')
        
        
        
        labels = ['Physical','Hpol Phase Center','Vpol Phase Center']
        print_prefixs = {   'Physical':'expected_time_differences_hpol' ,
                            'Hpol Phase Center':'expected_time_differences_vpol' ,
                            'Vpol Phase Center':'max_time_differences'}
        for index, antennas in enumerate([antennas_physical,antennas_phase_hpol,antennas_phase_vpol]):
            #print('\nCalculating expected time delays from %s location'%labels[index])
            tof = {}
            dof = {}
            for antenna, location in antennas.items():
                distance = numpy.sqrt((pulser_location[0] - location[0])**2 + (pulser_location[1] - location[1])**2 + (pulser_location[2] - location[2])**2)
                time = (distance / c)*1e9 #ns
                tof[antenna] = time
                dof[antenna] = distance

            dt = []
            max_dt = []
            for pair in pairs:
                dt.append(tof[pair[0]] - tof[pair[1]]) #Convention of 0 - 1 to match the time delays in frequency_domain_time_delays.py
                max_dt.append(numpy.sign(tof[pair[0]] - tof[pair[1]])*(numpy.sqrt((antennas[pair[0]][0] - antennas[pair[1]][0])**2 + (antennas[pair[0]][1] - antennas[pair[1]][1])**2 + (antennas[pair[0]][2] - antennas[pair[1]][2])**2) / c)*1e9) #ns

            print(print_prefixs[labels[index]],' = ',list(zip(pairs,dt)))
            #print('\t',list(zip(pairs,max_dt)))


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






