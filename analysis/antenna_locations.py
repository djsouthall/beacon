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

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import pymap3d as pm
import itertools
pairs = list(itertools.combinations((0,1,2,3), 2))
plt.ion()

Antennas = {0:(0.0,0.0,0.0),1:(-6.039,-1.618,2.275),2:(-1.272,-10.362,1.282),3:(3.411,-11.897,-0.432)}
pulser_location = (37.5861,-118.2332,3779.52)#latitude,longitude,elevation
A0Location = (37.5893,-118.2381,3894.12)#latitude,longtidue,elevation
pulser_location = pm.geodetic2enu(pulser_location[0],pulser_location[1],pulser_location[2],A0Location[0],A0Location[1],A0Location[2])

c = 2.99700e8 #m/s



if __name__ == '__main__':
    try:
        print(pulser_location)
        tof = {}
        dof = {}
        for antenna, location in Antennas.items():
            distance = numpy.sqrt((pulser_location[0] - location[0])**2 + (pulser_location[1] - location[1])**2 + (pulser_location[2] - location[2])**2)
            time = (distance / c)*1e9 #ns
            tof[antenna] = time
            dof[antenna] = distance

        dt = []
        max_dt = []
        for pair in pairs:
            dt.append(tof[pair[0]] - tof[pair[1]]) #Convention of 0 - 1 to match the time delays in frequency_domain_time_delays.py
            max_dt.append(numpy.sign(tof[pair[0]] - tof[pair[1]])*(numpy.sqrt((Antennas[pair[0]][0] - Antennas[pair[1]][0])**2 + (Antennas[pair[0]][1] - Antennas[pair[1]][1])**2 + (Antennas[pair[0]][2] - Antennas[pair[1]][2])**2) / c)*1e9) #ns

        print(list(zip(pairs,dt)))
        print(list(zip(pairs,max_dt)))


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






