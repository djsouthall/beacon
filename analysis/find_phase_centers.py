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

A0Location = info.loadAntennaZeroLocation(deploy_index=1)#latitude,longtidue,elevation
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



site1_measured_time_delays_hpol = [((0, 1), -27.053132099394283), ((0, 2), 105.53573434852417), ((0, 3), 30.08809070857938), ((1, 2), 145.51448633239156), ((1, 3), 70.36538379164826), ((2, 3), -75.30671213626783)]
site1_measured_time_delays_errors_hpol = [((0, 1), 0.049940419317735546), ((0, 2), 0.04451524800887434), ((0, 3), 0.04436462118733028), ((1, 2), 0.02611109734500819), ((1, 3), 0.029957540691833113), ((2, 3), 0.024498957055263823)]

site1_measured_time_delays_vpol = [((0, 1), -38.00260225579561), ((0, 2), 101.75125074208155), ((0, 3), 42.84079069759988), ((1, 2), 139.7311771661454), ((1, 3), 80.813510732201), ((2, 3), -58.88512983235407)]
site1_measured_time_delays_errors_vpol = [((0, 1), 0.029672793772042835), ((0, 2), 0.031483581625543926), ((0, 3), 0.035289050598712395), ((1, 2), 0.03016916157628555), ((1, 3), 0.03048344816865985), ((2, 3), 0.03221659467466419)]

site2_measured_time_delays_hpol = [((0, 1), -82.1668079571178), ((0, 2), 40.52924990302058), ((0, 3), -44.79556385927719), ((1, 2), 109.76177884322139), ((1, 3), 37.43774146085787), ((2, 3), -85.16942184392973)]
site2_measured_time_delays_errors_hpol = [((0, 1), 0.05940492985642724), ((0, 2), 0.05401962848166091), ((0, 3), 0.053629739039319266), ((1, 2), 0.04827978940049101), ((1, 3), 0.06416465723791487), ((2, 3), 0.04647844461537252)]

site2_measured_time_delays_vpol = [((0, 1), -79.95072281742472), ((0, 2), 36.69241182459076), ((0, 3), -31.955840625431993), ((1, 2), 116.60680076925239), ((1, 3), 48.014757819207645), ((2, 3), -68.59702651834894)]
site2_measured_time_delays_errors_vpol = [((0, 1), 0.04322954912532788), ((0, 2), 0.042846388825191005), ((0, 3), 0.040935002479091195), ((1, 2), 0.04706496853995274), ((1, 3), 0.03970422560921766), ((2, 3), 0.03766728653913236)]

site3_measured_time_delays_hpol = [((0, 1), -81.46902043268355), ((0, 2), -130.58422457283476), ((0, 3), -177.12919403225192), ((1, 2), -49.264947052474426), ((1, 3), -82.45418396924151), ((2, 3), -33.39096291225481)]
site3_measured_time_delays_errors_hpol = [((0, 1), 0.043725920483796865), ((0, 2), 0.054900873054177665), ((0, 3), 0.058695059566109445), ((1, 2), 0.04774585569696324), ((1, 3), 0.03522800606179125), ((2, 3), 0.05701787042277225)]

site3_measured_time_delays_vpol = [((0, 1), -92.07269595912595), ((0, 2), -147.17878030502635), ((0, 3), -166.03925551194664), ((1, 2), -55.13493366555078), ((1, 3), -73.94166201041543), ((2, 3), -18.84385983782996)]
site3_measured_time_delays_errors_vpol = [((0, 1), 0.032237057691814675), ((0, 2), 0.030132778733763417), ((0, 3), 0.03268367008475346), ((1, 2), 0.030870262323053385), ((1, 3), 0.03090586129896359), ((2, 3), 0.03267690656774697)]


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

    site3_d0 = numpy.sqrt((site3_pulser_location[0] - ant0_x)**2 + (site3_pulser_location[1] - ant0_y)**2 + (site3_pulser_location[2] - ant0_z)**2 )
    site3_d1 = numpy.sqrt((site3_pulser_location[0] - ant1_x)**2 + (site3_pulser_location[1] - ant1_y)**2 + (site3_pulser_location[2] - ant1_z)**2 )
    site3_d2 = numpy.sqrt((site3_pulser_location[0] - ant2_x)**2 + (site3_pulser_location[1] - ant2_y)**2 + (site3_pulser_location[2] - ant2_z)**2 )
    site3_d3 = numpy.sqrt((site3_pulser_location[0] - ant3_x)**2 + (site3_pulser_location[1] - ant3_y)**2 + (site3_pulser_location[2] - ant3_z)**2 )

    #calculate differences in distances travelled and convert to expected time delays for known pairs.
    #Ant 0 - Ant 1
    site1_p0 = ((site1_d0 - site1_d1)/c)*1.0e9 #ns
    site2_p0 = ((site2_d0 - site2_d1)/c)*1.0e9 #ns
    site3_p0 = ((site3_d0 - site3_d1)/c)*1.0e9 #ns

    #Ant 0 - Ant 2
    site1_p1 = ((site1_d0 - site1_d2)/c)*1.0e9 #ns
    site2_p1 = ((site2_d0 - site2_d2)/c)*1.0e9 #ns
    site3_p1 = ((site3_d0 - site3_d2)/c)*1.0e9 #ns

    #Ant 0 - Ant 3
    site1_p2 = ((site1_d0 - site1_d3)/c)*1.0e9 #ns
    site2_p2 = ((site2_d0 - site2_d3)/c)*1.0e9 #ns
    site3_p2 = ((site3_d0 - site3_d3)/c)*1.0e9 #ns

    #Ant 1 - Ant 2
    site1_p3 = ((site1_d1 - site1_d2)/c)*1.0e9 #ns
    site2_p3 = ((site2_d1 - site2_d2)/c)*1.0e9 #ns
    site3_p3 = ((site3_d1 - site3_d2)/c)*1.0e9 #ns

    #Ant 1 - Ant 3
    site1_p4 = ((site1_d1 - site1_d3)/c)*1.0e9 #ns
    site2_p4 = ((site2_d1 - site2_d3)/c)*1.0e9 #ns
    site3_p4 = ((site3_d1 - site3_d3)/c)*1.0e9 #ns

    #Ant 2 - Ant 3
    site1_p5 = ((site1_d2 - site1_d3)/c)*1.0e9 #ns
    site2_p5 = ((site2_d2 - site2_d3)/c)*1.0e9 #ns
    site3_p5 = ((site3_d2 - site3_d3)/c)*1.0e9 #ns

    
    chi_2 =     ((site1_p0 - site1_measured_time_delays[0][1])**2)/site1_measured_time_delays_errors[0][1]**2 + \
                ((site2_p0 - site2_measured_time_delays[0][1])**2)/site2_measured_time_delays_errors[0][1]**2 + \
                ((site3_p0 - site3_measured_time_delays[0][1])**2)/site3_measured_time_delays_errors[0][1]**2 + \
                \
                ((site1_p1 - site1_measured_time_delays[1][1])**2)/site1_measured_time_delays_errors[1][1]**2 + \
                ((site2_p1 - site2_measured_time_delays[1][1])**2)/site2_measured_time_delays_errors[1][1]**2 + \
                ((site3_p1 - site3_measured_time_delays[1][1])**2)/site3_measured_time_delays_errors[1][1]**2 + \
                \
                ((site1_p2 - site1_measured_time_delays[2][1])**2)/site1_measured_time_delays_errors[2][1]**2 + \
                ((site2_p2 - site2_measured_time_delays[2][1])**2)/site2_measured_time_delays_errors[2][1]**2 + \
                ((site3_p2 - site3_measured_time_delays[2][1])**2)/site3_measured_time_delays_errors[2][1]**2 + \
                \
                ((site1_p3 - site1_measured_time_delays[3][1])**2)/site1_measured_time_delays_errors[3][1]**2 + \
                ((site2_p3 - site2_measured_time_delays[3][1])**2)/site2_measured_time_delays_errors[3][1]**2 + \
                ((site3_p3 - site3_measured_time_delays[3][1])**2)/site3_measured_time_delays_errors[3][1]**2 + \
                \
                ((site1_p4 - site1_measured_time_delays[4][1])**2)/site1_measured_time_delays_errors[4][1]**2 + \
                ((site2_p4 - site2_measured_time_delays[4][1])**2)/site2_measured_time_delays_errors[4][1]**2 + \
                ((site3_p4 - site3_measured_time_delays[4][1])**2)/site3_measured_time_delays_errors[4][1]**2 + \
                \
                ((site1_p5 - site1_measured_time_delays[5][1])**2)/site1_measured_time_delays_errors[5][1]**2 + \
                ((site2_p5 - site2_measured_time_delays[5][1])**2)/site2_measured_time_delays_errors[5][1]**2 + \
                ((site3_p5 - site3_measured_time_delays[5][1])**2)/site3_measured_time_delays_errors[5][1]**2
    
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
            site3_measured_time_delays = site3_measured_time_delays_hpol

            site1_measured_time_delays_errors = site1_measured_time_delays_errors_hpol
            site2_measured_time_delays_errors = site2_measured_time_delays_errors_hpol
            site3_measured_time_delays_errors = site3_measured_time_delays_errors_hpol

        elif mode == 'vpol':
            site1_measured_time_delays = site1_measured_time_delays_vpol
            site2_measured_time_delays = site2_measured_time_delays_vpol
            site3_measured_time_delays = site3_measured_time_delays_vpol

            site1_measured_time_delays_errors = site1_measured_time_delays_errors_vpol
            site2_measured_time_delays_errors = site2_measured_time_delays_errors_vpol
            site3_measured_time_delays_errors = site3_measured_time_delays_errors_vpol

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






