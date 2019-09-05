'''
This script use time delays measured by frequency_domain_time_delays to estimate the phase
center locations of the antennas.  
'''

import numpy
import itertools
import os
import sys
import csv
from iminuit import Minuit

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import objects.station as bc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

pairs = numpy.array(list(itertools.combinations((0,1,2,3), 2)))

Antennas = {0:(0.0,0.0,0.0),1:(-6.039,-1.618,2.275),2:(-1.272,-10.362,1.282),3:(3.411,-11.897,-0.432)}
antenna_locs = numpy.array([[0.0,0.0,0.0],[-6.039,-1.618,2.275],[-1.272,-10.362,1.282],[3.411,-11.897,-0.432]])
antenna_locs_3d = antenna_locs[pairs]
pulser_location = (37.5861,-118.2332,3779.52)#latitude,longitude,elevation
A0Location = (37.5893,-118.2381,3894.12)#latitude,longtidue,elevation


#12 variables
ant0_physical_x = 0.0
ant0_physical_y = 0.0
ant0_physical_z = 0.0

ant1_physical_x = -6.039
ant1_physical_y = -1.618
ant1_physical_z = 2.275

ant2_physical_x = -1.272
ant2_physical_y = -10.362
ant2_physical_z = 1.282

ant3_physical_x = 3.411
ant3_physical_y = -11.897
ant3_physical_z = -0.432



station = bc.Station('station_0',A0Location)
station.addSource('pulser_0',pulser_location)
pulser_enu = station.known_sources['pulser_0'].relative_enu[station.key]

for key, value in Antennas.items():
    station.addAntennaRelative('ant_%s'%str(key),value)

c = 2.99700e8 #m/s

measured_time_delays_hpol = [((0, 1), -13.522159868650306), ((0, 2), 19.163342519369962), ((0, 3), 30.86098740386865), ((1, 2), 32.763473610383244), ((1, 3), 44.17987278635087), ((2, 3), 11.588118967840956)]
measured_time_delays_errors_hpol = [((0, 1), 1.0), ((0, 2), 1.0), ((0, 3), 1.0), ((1, 2), 1.0), ((1, 3), 1.0), ((2, 3), 1.0)]

measured_time_delays_vpol = [((0, 1), -9.875403459432164), ((0, 2), 17.19470489079049), ((0, 3), 38.70998128227029), ((1, 2), 27.116830049415512), ((1, 3), 48.600856917725196), ((2, 3), 21.40575047482207)]
measured_time_delays_errors_vpol = [((0, 1), 1.0), ((0, 2), 1.0), ((0, 3), 1.0), ((1, 2), 1.0), ((1, 3), 1.0), ((2, 3), 1.0)]

mode = 'vpol'
if mode == 'hpol':
    measured_time_delays = measured_time_delays_hpol
    measured_time_delays_errors = measured_time_delays_errors_hpol
elif mode == 'vpol':
    measured_time_delays = measured_time_delays_vpol
    measured_time_delays_errors = measured_time_delays_errors_vpol

def f(ant0_x, ant0_y, ant0_z, ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z):
    '''
    To generalize, look into from_array_func Minuit initializer.  
    '''
    #Calculate distances from pulser to each antenna
    d0 = numpy.sqrt((pulser_enu[0] - ant0_x)**2 + (pulser_enu[1] - ant0_y)**2 + (pulser_enu[2] - ant0_z)**2 )
    d1 = numpy.sqrt((pulser_enu[0] - ant1_x)**2 + (pulser_enu[1] - ant1_y)**2 + (pulser_enu[2] - ant1_z)**2 )
    d2 = numpy.sqrt((pulser_enu[0] - ant2_x)**2 + (pulser_enu[1] - ant2_y)**2 + (pulser_enu[2] - ant2_z)**2 )
    d3 = numpy.sqrt((pulser_enu[0] - ant3_x)**2 + (pulser_enu[1] - ant3_y)**2 + (pulser_enu[2] - ant3_z)**2 )

    #calculate differences in distances travelled and convert to expected time delays for known pairs.
    #Ant 0 - Ant 1
    p0 = ((d0 - d1)/c)*1e9 #ns
    #Ant 0 - Ant 2
    p1 = ((d0 - d2)/c)*1e9 #ns
    #Ant 0 - Ant 3
    p2 = ((d0 - d3)/c)*1e9 #ns
    #Ant 1 - Ant 2
    p3 = ((d1 - d2)/c)*1e9 #ns
    #Ant 1 - Ant 3
    p4 = ((d1 - d3)/c)*1e9 #ns
    #Ant 2 - Ant 3
    p5 = ((d2 - d3)/c)*1e9 #ns
    
    chi_2 =     ((p0 - measured_time_delays[0][1])**2)/measured_time_delays_errors[0][1] + \
                ((p1 - measured_time_delays[1][1])**2)/measured_time_delays_errors[1][1] + \
                ((p2 - measured_time_delays[2][1])**2)/measured_time_delays_errors[2][1] + \
                ((p3 - measured_time_delays[3][1])**2)/measured_time_delays_errors[3][1] + \
                ((p4 - measured_time_delays[4][1])**2)/measured_time_delays_errors[4][1] + \
                ((p5 - measured_time_delays[5][1])**2)/measured_time_delays_errors[5][1]
    return chi_2


if __name__ == '__main__':
    try:
        print('Performing calculations for %s'%mode)
        initial_step = 0.1 #m
        m = Minuit(f,   ant0_x=ant0_physical_x, ant0_y=ant0_physical_y, ant0_z=ant0_physical_z, ant1_x=ant1_physical_x, ant1_y=ant1_physical_y, ant1_z=ant1_physical_z, ant2_x=ant2_physical_x, ant2_y=ant2_physical_y, ant2_z=ant2_physical_z, ant3_x=ant3_physical_x, ant3_y=ant3_physical_y, ant3_z=ant3_physical_z,\
                        error_ant0_x=initial_step, error_ant0_y=initial_step, error_ant0_z=initial_step, error_ant1_x=initial_step, error_ant1_y=initial_step, error_ant1_z=initial_step, error_ant2_x=initial_step, error_ant2_y=initial_step, error_ant2_z=initial_step, error_ant3_x=initial_step, error_ant3_y=initial_step, error_ant3_z=initial_step,\
                        errordef = 1.0)
        result = m.migrad()
        pprint(m.get_fmin())
        print(result)

        #12 variables
        ant0_phase_x = m.values['ant0_x']
        ant0_phase_y = m.values['ant0_y']
        ant0_phase_z = m.values['ant0_z']

        ant1_phase_x = m.values['ant1_x']
        ant1_phase_y = m.values['ant1_y']
        ant1_phase_z = m.values['ant1_z']

        ant2_phase_x = m.values['ant2_x']
        ant2_phase_y = m.values['ant2_y']
        ant2_phase_z = m.values['ant2_z']

        ant3_phase_x = m.values['ant3_x']
        ant3_phase_y = m.values['ant3_y']
        ant3_phase_z = m.values['ant3_z']

        phase_locs = numpy.array([[ant0_phase_x,ant0_phase_y,ant0_phase_z],[ant1_phase_x,ant1_phase_y,ant1_phase_z],[ant2_phase_x,ant2_phase_y,ant2_phase_z],[ant3_phase_x,ant3_phase_y,ant3_phase_z]])

        print('Antenna Locations: \n%s'%str(antenna_locs))
        print('Phase Locations: \n%s'%str(phase_locs))

        print('\nDifference (antenna_locs - phase_locs): \n%s'%str(antenna_locs - phase_locs))





    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






