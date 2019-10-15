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

antennas_physical = info.loadAntennaLocationsENU(deploy_index=1)[0]
antenna_locs = numpy.array([antennas_physical[0],antennas_physical[1],antennas_physical[2],antennas_physical[3]])
antenna_locs_3d = antenna_locs[pairs]

A0Location = loadAntennaZeroLocation(deploy_index=1)#latitude,longtidue,elevation
pulser_locations = info.loadPulserLocationsENU() #ENU

site1_pulser_location = pulser_locations['run1507']
site2_pulser_location = pulser_locations['run1509']
site3_pulser_location = pulser_locations['run1511']

#12 variables (x2 because of polarizations for output variables)
ant0_physical_x = antennas_physical[0][0]
ant0_physical_y = antennas_physical[0][1]
ant0_physical_z = antennas_physical[0][2]

ant1_physical_x = antennas_physical[1][0]
ant1_physical_y = antennas_physical[1][1]
ant1_physical_z = antennas_physical[1][2]

ant2_physical_x = antennas_physical[2][0]
ant2_physical_y = antennas_physical[2][1]
ant2_physical_z = antennas_physical[2][2]

ant3_physical_x = antennas_physical[3][0]
ant3_physical_y = antennas_physical[3][1]
ant3_physical_z = antennas_physical[3][2]

#Limits 
guess_range = None #Limit to how far from physical center the phase centers can be.  

if guess_range is not None:
    ant0_physical_limits_x = (antennas_physical[0][0] - guess_range ,antennas_physical[0][0] + guess_range)
    ant0_physical_limits_y = (antennas_physical[0][1] - guess_range ,antennas_physical[0][1] + guess_range)
    ant0_physical_limits_z = (antennas_physical[0][2] - guess_range ,antennas_physical[0][2] + guess_range)

    ant1_physical_limits_x = (antennas_physical[1][0] - guess_range ,antennas_physical[1][0] + guess_range)
    ant1_physical_limits_y = (antennas_physical[1][1] - guess_range ,antennas_physical[1][1] + guess_range)
    ant1_physical_limits_z = (antennas_physical[1][2] - guess_range ,antennas_physical[1][2] + guess_range)

    ant2_physical_limits_x = (antennas_physical[2][0] - guess_range ,antennas_physical[2][0] + guess_range)
    ant2_physical_limits_y = (antennas_physical[2][1] - guess_range ,antennas_physical[2][1] + guess_range)
    ant2_physical_limits_z = (antennas_physical[2][2] - guess_range ,antennas_physical[2][2] + guess_range)

    ant3_physical_limits_x = (antennas_physical[3][0] - guess_range ,antennas_physical[3][0] + guess_range)
    ant3_physical_limits_y = (antennas_physical[3][1] - guess_range ,antennas_physical[3][1] + guess_range)
    ant3_physical_limits_z = (antennas_physical[3][2] - guess_range ,antennas_physical[3][2] + guess_range)
else:
    ant0_physical_limits_x = None
    ant0_physical_limits_y = None
    ant0_physical_limits_z = None

    ant1_physical_limits_x = None
    ant1_physical_limits_y = None
    ant1_physical_limits_z = None

    ant2_physical_limits_x = None
    ant2_physical_limits_y = None
    ant2_physical_limits_z = None

    ant3_physical_limits_x = None
    ant3_physical_limits_y = None
    ant3_physical_limits_z = None

c = 2.99700e8 #m/s

#Measureables (6 baselines x 3 sites x 2 polarizations = 36)

site1_measured_time_delays_hpol = [((0, 1), -14.224921871745725), ((0, 2), 19.876589950811745), ((0, 3), 33.40912298266971), ((1, 2), 34.019589228664664), ((1, 3), 47.57044139896039), ((2, 3), 13.48957876430875)]
site1_measured_time_delays_errors_hpol = [((0, 1), 0.06233491795311048), ((0, 2), 0.06382554973518859), ((0, 3), 0.33814236909528766), ((1, 2), 0.10214427694663512), ((1, 3), 0.3658870823486229), ((2, 3), 0.34880895137219037)]

site1_measured_time_delays_vpol = [((0, 1), -10.8811708733604), ((0, 2), 17.34400605219815), ((0, 3), 38.236602651330955), ((1, 2), 28.061713346561103), ((1, 3), 49.04098522640182), ((2, 3), 20.9601676263453)]
site1_measured_time_delays_errors_vpol = [((0, 1), 0.11638830591906603), ((0, 2), 0.048620026672220505), ((0, 3), 0.08857784505147924), ((1, 2), 0.07591263744700386), ((1, 3), 0.06274341696668266), ((2, 3), 0.07334332990065455)]

site2_measured_time_delays_hpol = [((0, 1), -13.58716769257573), ((0, 2), 19.158550981849476), ((0, 3), 30.77279240847105), ((1, 2), 32.69936857396089), ((1, 3), 44.18385816610001), ((2, 3), 11.27931686951937)]
site2_measured_time_delays_errors_hpol = [((0, 1), 0.09531186417637323), ((0, 2), 0.12724109382433263), ((0, 3), 0.3875413892537076), ((1, 2), 0.12470147484877567), ((1, 3), 0.36688889196595226), ((2, 3), 0.34983579961273653)]

site2_measured_time_delays_vpol = [((0, 1), -9.910919892043706), ((0, 2), 17.243676724603866), ((0, 3), 38.692527821047), ((1, 2), 27.114368394000184), ((1, 3), 48.56902487092354), ((2, 3), 21.457044552683175)]
site2_measured_time_delays_errors_vpol = [((0, 1), 0.09570601760961257), ((0, 2), 0.11429843724284253), ((0, 3), 0.10545687811227517), ((1, 2), 0.07421032869730562), ((1, 3), 0.07877429114919211), ((2, 3), 0.07282936925867901)]

def f(ant0_x, ant0_y, ant0_z, ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z):
    '''
    To generalize, look into from_array_func Minuit initializer.  
    '''
    #Calculate distances from pulser to each antenna
    site1_d0 = numpy.sqrt((site1_pulser_location[0] - ant0_x)**2 + (site1_pulser_location[1] - ant0_y)**2 + (site1_pulser_location[2] - ant0_z)**2 )
    site1_d1 = numpy.sqrt((site1_pulser_location[0] - ant1_x)**2 + (site1_pulser_location[1] - ant1_y)**2 + (site1_pulser_location[2] - ant1_z)**2 )
    site1_d2 = numpy.sqrt((site1_pulser_location[0] - ant2_x)**2 + (site1_pulser_location[1] - ant2_y)**2 + (site1_pulser_location[2] - ant2_z)**2 )
    site1_d3 = numpy.sqrt((site1_pulser_location[0] - ant3_x)**2 + (site1_pulser_location[1] - ant3_y)**2 + (site1_pulser_location[2] - ant3_z)**2 )

    site2_d0 = numpy.sqrt((site2_pulser_location[0] - ant0_x)**2 + (site2_pulser_location[1] - ant0_y)**2 + (site2_pulser_location[2] - ant0_z)**2 )
    site2_d1 = numpy.sqrt((site2_pulser_location[0] - ant1_x)**2 + (site2_pulser_location[1] - ant1_y)**2 + (site2_pulser_location[2] - ant1_z)**2 )
    site2_d2 = numpy.sqrt((site2_pulser_location[0] - ant2_x)**2 + (site2_pulser_location[1] - ant2_y)**2 + (site2_pulser_location[2] - ant2_z)**2 )
    site2_d3 = numpy.sqrt((site2_pulser_location[0] - ant3_x)**2 + (site2_pulser_location[1] - ant3_y)**2 + (site2_pulser_location[2] - ant3_z)**2 )


    #calculate differences in distances travelled and convert to expected time delays for known pairs.
    #Ant 0 - Ant 1
    site1_p0 = ((site1_d0 - site1_d1)/c)*1.0e9 #ns
    site2_p0 = ((site2_d0 - site2_d1)/c)*1.0e9 #ns
    #Ant 0 - Ant 2
    site1_p1 = ((site1_d0 - site1_d2)/c)*1.0e9 #ns
    site2_p1 = ((site2_d0 - site2_d2)/c)*1.0e9 #ns
    #Ant 0 - Ant 3
    site1_p2 = ((site1_d0 - site1_d3)/c)*1.0e9 #ns
    site2_p2 = ((site2_d0 - site2_d3)/c)*1.0e9 #ns
    #Ant 1 - Ant 2
    site1_p3 = ((site1_d1 - site1_d2)/c)*1.0e9 #ns
    site2_p3 = ((site2_d1 - site2_d2)/c)*1.0e9 #ns
    #Ant 1 - Ant 3
    site1_p4 = ((site1_d1 - site1_d3)/c)*1.0e9 #ns
    site2_p4 = ((site2_d1 - site2_d3)/c)*1.0e9 #ns
    #Ant 2 - Ant 3
    site1_p5 = ((site1_d2 - site1_d3)/c)*1.0e9 #ns
    site2_p5 = ((site2_d2 - site2_d3)/c)*1.0e9 #ns
    
    chi_2 =     ((site1_p0 - site1_measured_time_delays[0][1])**2)/site1_measured_time_delays_errors[0][1]**2 + \
                ((site2_p0 - site2_measured_time_delays[0][1])**2)/site2_measured_time_delays_errors[0][1]**2 + \
                ((site1_p1 - site1_measured_time_delays[1][1])**2)/site1_measured_time_delays_errors[1][1]**2 + \
                ((site2_p1 - site2_measured_time_delays[1][1])**2)/site2_measured_time_delays_errors[1][1]**2 + \
                ((site1_p2 - site1_measured_time_delays[2][1])**2)/site1_measured_time_delays_errors[2][1]**2 + \
                ((site2_p2 - site2_measured_time_delays[2][1])**2)/site2_measured_time_delays_errors[2][1]**2 + \
                ((site1_p3 - site1_measured_time_delays[3][1])**2)/site1_measured_time_delays_errors[3][1]**2 + \
                ((site2_p3 - site2_measured_time_delays[3][1])**2)/site2_measured_time_delays_errors[3][1]**2 + \
                ((site1_p4 - site1_measured_time_delays[4][1])**2)/site1_measured_time_delays_errors[4][1]**2 + \
                ((site2_p4 - site2_measured_time_delays[4][1])**2)/site2_measured_time_delays_errors[4][1]**2 + \
                ((site1_p5 - site1_measured_time_delays[5][1])**2)/site1_measured_time_delays_errors[5][1]**2 + \
                ((site2_p5 - site2_measured_time_delays[5][1])**2)/site2_measured_time_delays_errors[5][1]**2
    return chi_2


if __name__ == '__main__':
    try:

        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to vpol')
                mode = 'vpol'
        else:
            print('No mode given.  Defaulting to vpol')
            mode = 'vpol'

        print('Performing calculations for %s'%mode)
        if mode == 'hpol':
            site1_measured_time_delays = site1_measured_time_delays_hpol
            site2_measured_time_delays = site2_measured_time_delays_hpol

            site1_measured_time_delays_errors = site1_measured_time_delays_errors_hpol
            site2_measured_time_delays_errors = site2_measured_time_delays_errors_hpol

        elif mode == 'vpol':
            site1_measured_time_delays = site1_measured_time_delays_vpol
            site2_measured_time_delays = site2_measured_time_delays_vpol

            site1_measured_time_delays_errors = site1_measured_time_delays_errors_vpol
            site2_measured_time_delays_errors = site2_measured_time_delays_errors_vpol

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






