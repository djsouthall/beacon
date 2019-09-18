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
import tools.info as info

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

A0Location = (37.5893,-118.2381,3894.12)#latitude,longtidue,elevation
pulser_locations = info.loadPulserLocationsENU() #ENU
day5_pulser_location = pulser_locations['run782']
day6_pulser_location = pulser_locations['run793']

pulser_location = (37.5861,-118.2332,3779.52)#latitude,longitude,elevation


#12 variables (x2 because of polarizations for output variables)
ant0_physical_x = Antennas[0][0]
ant0_physical_y = Antennas[0][1]
ant0_physical_z = Antennas[0][2]

ant1_physical_x = Antennas[1][0]
ant1_physical_y = Antennas[1][1]
ant1_physical_z = Antennas[1][2]

ant2_physical_x = Antennas[2][0]
ant2_physical_y = Antennas[2][1]
ant2_physical_z = Antennas[2][2]

ant3_physical_x = Antennas[3][0]
ant3_physical_y = Antennas[3][1]
ant3_physical_z = Antennas[3][2]

#Limits 
guess_range = 5.0 #Limit to how far from physical center the phase centers can be.  
ant0_physical_limits_x = (Antennas[0][0] - guess_range ,Antennas[0][0] + guess_range)
ant0_physical_limits_y = (Antennas[0][1] - guess_range ,Antennas[0][1] + guess_range)
ant0_physical_limits_z = (Antennas[0][2] - guess_range ,Antennas[0][2] + guess_range)

ant1_physical_limits_x = (Antennas[1][0] - guess_range ,Antennas[1][0] + guess_range)
ant1_physical_limits_y = (Antennas[1][1] - guess_range ,Antennas[1][1] + guess_range)
ant1_physical_limits_z = (Antennas[1][2] - guess_range ,Antennas[1][2] + guess_range)

ant2_physical_limits_x = (Antennas[2][0] - guess_range ,Antennas[2][0] + guess_range)
ant2_physical_limits_y = (Antennas[2][1] - guess_range ,Antennas[2][1] + guess_range)
ant2_physical_limits_z = (Antennas[2][2] - guess_range ,Antennas[2][2] + guess_range)

ant3_physical_limits_x = (Antennas[3][0] - guess_range ,Antennas[3][0] + guess_range)
ant3_physical_limits_y = (Antennas[3][1] - guess_range ,Antennas[3][1] + guess_range)
ant3_physical_limits_z = (Antennas[3][2] - guess_range ,Antennas[3][2] + guess_range)

c = 2.99700e8 #m/s

#Measureables (6 baselines x 2 sites x 2 polarizations = 24)

day5_measured_time_delays_hpol = [((0, 1), -14.224894169837777), ((0, 2), 19.877913870967042), ((0, 3), 33.41603125439723), ((1, 2), 34.01793900846452), ((1, 3), 47.58058297857065), ((2, 3), 13.49814216298322)]
day5_measured_time_delays_errors_hpol = [((0, 1), 0.06242523254593814), ((0, 2), 0.06551777511545404), ((0, 3), 0.3474837161367034), ((1, 2), 0.10155187104093898), ((1, 3), 0.3618096657854842), ((2, 3), 0.3472850258299439)]

day5_measured_time_delays_vpol = [((0, 1), -10.883704486434414), ((0, 2), 17.344005103996537), ((0, 3), 38.23462324810745), ((1, 2), 28.064767462355917), ((1, 3), 49.041921011622634), ((2, 3), 20.96146716322024)]
day5_measured_time_delays_errors_vpol = [((0, 1), 0.1194031290358806), ((0, 2), 0.048623147381227666), ((0, 3), 0.09075103566110058), ((1, 2), 0.07860361150323145), ((1, 3), 0.06454928284754104), ((2, 3), 0.0732093872343773)]


day6_measured_time_delays_hpol = [((0, 1), -13.592091429151054), ((0, 2), 19.16001027052159), ((0, 3), 30.75389605307086), ((1, 2), 32.70726977544983), ((1, 3), 44.207337841605415), ((2, 3), 11.299536705693228)]
day6_measured_time_delays_errors_hpol = [((0, 1), 0.09620101986130024), ((0, 2), 0.13301888847570267), ((0, 3), 0.39484710750876817), ((1, 2), 0.1262325763913089), ((1, 3), 0.360032335595451), ((2, 3), 0.33944838464119326)]

day6_measured_time_delays_vpol = [((0, 1), -9.907815800005917), ((0, 2), 17.246090416069613), ((0, 3), 38.69389075198866), ((1, 2), 27.113051415306572), ((1, 3), 48.574239548950025), ((2, 3), 21.462514816608664)]
day6_measured_time_delays_errors_vpol = [((0, 1), 0.09705042719954765), ((0, 2), 0.11589576468440983), ((0, 3), 0.10906867130376277), ((1, 2), 0.07376204794601858), ((1, 3), 0.079778528687297), ((2, 3), 0.06997973562561836)]


def f(ant0_x, ant0_y, ant0_z, ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z):
    '''
    To generalize, look into from_array_func Minuit initializer.  
    '''
    #Calculate distances from pulser to each antenna
    day5_d0 = numpy.sqrt((day5_pulser_location[0] - ant0_x)**2 + (day5_pulser_location[1] - ant0_y)**2 + (day5_pulser_location[2] - ant0_z)**2 )
    day5_d1 = numpy.sqrt((day5_pulser_location[0] - ant1_x)**2 + (day5_pulser_location[1] - ant1_y)**2 + (day5_pulser_location[2] - ant1_z)**2 )
    day5_d2 = numpy.sqrt((day5_pulser_location[0] - ant2_x)**2 + (day5_pulser_location[1] - ant2_y)**2 + (day5_pulser_location[2] - ant2_z)**2 )
    day5_d3 = numpy.sqrt((day5_pulser_location[0] - ant3_x)**2 + (day5_pulser_location[1] - ant3_y)**2 + (day5_pulser_location[2] - ant3_z)**2 )

    day6_d0 = numpy.sqrt((day6_pulser_location[0] - ant0_x)**2 + (day6_pulser_location[1] - ant0_y)**2 + (day6_pulser_location[2] - ant0_z)**2 )
    day6_d1 = numpy.sqrt((day6_pulser_location[0] - ant1_x)**2 + (day6_pulser_location[1] - ant1_y)**2 + (day6_pulser_location[2] - ant1_z)**2 )
    day6_d2 = numpy.sqrt((day6_pulser_location[0] - ant2_x)**2 + (day6_pulser_location[1] - ant2_y)**2 + (day6_pulser_location[2] - ant2_z)**2 )
    day6_d3 = numpy.sqrt((day6_pulser_location[0] - ant3_x)**2 + (day6_pulser_location[1] - ant3_y)**2 + (day6_pulser_location[2] - ant3_z)**2 )


    #calculate differences in distances travelled and convert to expected time delays for known pairs.
    #Ant 0 - Ant 1
    day5_p0 = ((day5_d0 - day5_d1)/c)*1.0e9 #ns
    day6_p0 = ((day6_d0 - day6_d1)/c)*1.0e9 #ns
    #Ant 0 - Ant 2
    day5_p1 = ((day5_d0 - day5_d2)/c)*1.0e9 #ns
    day6_p1 = ((day6_d0 - day6_d2)/c)*1.0e9 #ns
    #Ant 0 - Ant 3
    day5_p2 = ((day5_d0 - day5_d3)/c)*1.0e9 #ns
    day6_p2 = ((day6_d0 - day6_d3)/c)*1.0e9 #ns
    #Ant 1 - Ant 2
    day5_p3 = ((day5_d1 - day5_d2)/c)*1.0e9 #ns
    day6_p3 = ((day6_d1 - day6_d2)/c)*1.0e9 #ns
    #Ant 1 - Ant 3
    day5_p4 = ((day5_d1 - day5_d3)/c)*1.0e9 #ns
    day6_p4 = ((day6_d1 - day6_d3)/c)*1.0e9 #ns
    #Ant 2 - Ant 3
    day5_p5 = ((day5_d2 - day5_d3)/c)*1.0e9 #ns
    day6_p5 = ((day6_d2 - day6_d3)/c)*1.0e9 #ns
    
    chi_2 =     ((day5_p0 - day5_measured_time_delays[0][1])**2)/day5_measured_time_delays_errors[0][1]**2 + \
                ((day6_p0 - day6_measured_time_delays[0][1])**2)/day6_measured_time_delays_errors[0][1]**2 + \
                ((day5_p1 - day5_measured_time_delays[1][1])**2)/day5_measured_time_delays_errors[1][1]**2 + \
                ((day6_p1 - day6_measured_time_delays[1][1])**2)/day6_measured_time_delays_errors[1][1]**2 + \
                ((day5_p2 - day5_measured_time_delays[2][1])**2)/day5_measured_time_delays_errors[2][1]**2 + \
                ((day6_p2 - day6_measured_time_delays[2][1])**2)/day6_measured_time_delays_errors[2][1]**2 + \
                ((day5_p3 - day5_measured_time_delays[3][1])**2)/day5_measured_time_delays_errors[3][1]**2 + \
                ((day6_p3 - day6_measured_time_delays[3][1])**2)/day6_measured_time_delays_errors[3][1]**2 + \
                ((day5_p4 - day5_measured_time_delays[4][1])**2)/day5_measured_time_delays_errors[4][1]**2 + \
                ((day6_p4 - day6_measured_time_delays[4][1])**2)/day6_measured_time_delays_errors[4][1]**2 + \
                ((day5_p5 - day5_measured_time_delays[5][1])**2)/day5_measured_time_delays_errors[5][1]**2 + \
                ((day6_p5 - day6_measured_time_delays[5][1])**2)/day6_measured_time_delays_errors[5][1]**2
    return chi_2


if __name__ == '__main__':
    try:

        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to vpol')
                mode = 'vpol'

        print('Performing calculations for %s'%mode)
        if mode == 'hpol':
            day5_measured_time_delays = day5_measured_time_delays_hpol
            day6_measured_time_delays = day6_measured_time_delays_hpol

            day5_measured_time_delays_errors = day5_measured_time_delays_errors_hpol
            day6_measured_time_delays_errors = day6_measured_time_delays_errors_hpol

        elif mode == 'vpol':
            day5_measured_time_delays = day5_measured_time_delays_vpol
            day6_measured_time_delays = day6_measured_time_delays_vpol

            day5_measured_time_delays_errors = day5_measured_time_delays_errors_vpol
            day6_measured_time_delays_errors = day6_measured_time_delays_errors_vpol

        initial_step = 0.1 #m
        m = Minuit(     f,\
                        ant0_x=ant0_physical_x,\
                        ant0_y=ant0_physical_y,\
                        ant0_z=ant0_physical_z,\
                        ant1_x=ant1_physical_x,\
                        ant1_y=ant1_physical_y,\
                        ant1_z=ant1_physical_z,\
                        ant2_x=ant2_physical_x,\
                        ant2_y=ant2_physical_y,\
                        ant2_z=ant2_physical_z,\
                        ant3_x=ant3_physical_x,\
                        ant3_y=ant3_physical_y,\
                        ant3_z=ant3_physical_z,\
                        error_ant0_x=initial_step,\
                        error_ant0_y=initial_step,\
                        error_ant0_z=initial_step,\
                        error_ant1_x=initial_step,\
                        error_ant1_y=initial_step,\
                        error_ant1_z=initial_step,\
                        error_ant2_x=initial_step,\
                        error_ant2_y=initial_step,\
                        error_ant2_z=initial_step,\
                        error_ant3_x=initial_step,\
                        error_ant3_y=initial_step,\
                        error_ant3_z=initial_step,\
                        errordef = 1.0,\
                        limit_ant0_x=ant0_physical_limits_x,\
                        limit_ant0_y=ant0_physical_limits_y,\
                        limit_ant0_z=ant0_physical_limits_z,\
                        limit_ant1_x=ant1_physical_limits_x,\
                        limit_ant1_y=ant1_physical_limits_y,\
                        limit_ant1_z=ant1_physical_limits_z,\
                        limit_ant2_x=ant2_physical_limits_x,\
                        limit_ant2_y=ant2_physical_limits_y,\
                        limit_ant2_z=ant2_physical_limits_z,\
                        limit_ant3_x=ant3_physical_limits_x,\
                        limit_ant3_y=ant3_physical_limits_y,\
                        limit_ant3_z=ant3_physical_limits_z)
        result = m.migrad()
        m.hesse()
        m.minos()
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






