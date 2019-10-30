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
        cable_delays = info.loadCableDelays()[mode]

        pairs = numpy.array(list(itertools.combinations((0,1,2,3), 2)))

        antennas_physical = info.loadAntennaLocationsENU(deploy_index=1)[0]
        antenna_locs = numpy.array([antennas_physical[0],antennas_physical[1],antennas_physical[2],antennas_physical[3]])
        antenna_locs_3d = antenna_locs[pairs]

        A0Location = info.loadAntennaZeroLocation(deploy_index=1)#latitude,longtidue,elevation
        all_pulser_locations = info.loadPulserPhaseLocationsENU() #ENU

        site1_pulser_location = all_pulser_locations[mode]['run1507']
        site2_pulser_location = all_pulser_locations[mode]['run1509']
        site3_pulser_location = all_pulser_locations[mode]['run1511']

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
            
            puls1_x_limits          = (site1_pulser_location[0] - guess_range ,site1_pulser_location[0] + guess_range)
            puls1_y_limits          = (site1_pulser_location[1] - guess_range ,site1_pulser_location[1] + guess_range)
            puls1_z_limits          = (site1_pulser_location[2] - guess_range ,site1_pulser_location[2] + guess_range)

            puls2_x_limits          = (site2_pulser_location[0] - guess_range ,site2_pulser_location[0] + guess_range)
            puls2_y_limits          = (site2_pulser_location[1] - guess_range ,site2_pulser_location[1] + guess_range)
            puls2_z_limits          = (site2_pulser_location[2] - guess_range ,site2_pulser_location[2] + guess_range)

            puls3_x_limits          = (site3_pulser_location[0] - guess_range ,site3_pulser_location[0] + guess_range)
            puls3_y_limits          = (site3_pulser_location[1] - guess_range ,site3_pulser_location[1] + guess_range)
            puls3_z_limits          = (site3_pulser_location[2] - guess_range ,site3_pulser_location[2] + guess_range)


        else:
            #ant0_physical_limits_x = None 
            #ant0_physical_limits_y = None
            #ant0_physical_limits_z = None

            ant1_physical_limits_x = None#(None,0.0)
            ant1_physical_limits_y = None#None
            ant1_physical_limits_z = None#(10.0,None)

            ant2_physical_limits_x = None#None
            ant2_physical_limits_y = None#(None,0.0)
            ant2_physical_limits_z = None#None

            ant3_physical_limits_x = None#(None,0.0)
            ant3_physical_limits_y = None#(None,0.0)
            ant3_physical_limits_z = None#(0.0,None)

            puls1_x_limits          = (0.0, None)
            puls1_y_limits          = (None, 0.0)
            puls1_z_limits          = (None, -10.0)#(site1_pulser_location[2] - 300, site1_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)

            puls2_x_limits          = (0.0, None)
            puls2_y_limits          = (None, 0.0)
            puls2_z_limits          = (None, -10.0)#(site2_pulser_location[2] - 300, site2_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)

            puls3_x_limits          = (0.0, None)
            puls3_y_limits          = (0.0, None)
            puls3_z_limits          = (None, -10.0)#(None, 0.0)#(site3_pulser_location[2] - 300, site3_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)



        n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
        c = 299792458/n #m/s

        site1_measured_time_delays_hpol =  [((0, 1), -40.20651888606884), ((0, 2), 105.42486911650398), ((0, 3), 30.076727191600487), ((1, 2), 145.47506405667053), ((1, 3), 70.40830523439116), ((2, 3), -75.22308276818167)]
        site1_measured_time_delays_errors_hpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]
        
        site1_measured_time_delays_vpol =  [((0, 1), -38.04924843261725), ((0, 2), 101.73562399320996), ((0, 3), 36.36094981687252), ((1, 2), 139.78487242582722), ((1, 3), 74.47272782785069), ((2, 3), -65.37467417633744)]
        site1_measured_time_delays_errors_vpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]


        site2_measured_time_delays_hpol =  [((0, 1), -82.30008730167084), ((0, 2), 40.63430060469886), ((0, 3), -44.66647351085744), ((1, 2), 122.62181636325664), ((1, 3), 37.72738525374733), ((2, 3), -85.20700265262238)]
        site2_measured_time_delays_errors_hpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]
        
        site2_measured_time_delays_vpol =  [((0, 1), -79.95580072832284), ((0, 2), 36.75841347009681), ((0, 3), -38.41504264859608), ((1, 2), 116.62044273548572), ((1, 3), 41.603272388349374), ((2, 3), -75.1734561186929)]
        site2_measured_time_delays_errors_vpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]


        site3_measured_time_delays_hpol =  [((0, 1), -94.49037748308051), ((0, 2), -143.62662406045482), ((0, 3), -177.25932209942096), ((1, 2), -49.29253234893085), ((1, 3), -82.51888738184999), ((2, 3), -33.38264080447568)]
        site3_measured_time_delays_errors_hpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]
        
        site3_measured_time_delays_vpol =  [((0, 1), -92.02106229248727), ((0, 2), -147.1274253433212), ((0, 3), -172.50823464410232), ((1, 2), -55.10636305083391), ((1, 3), -80.39340088868113), ((2, 3), -25.380809300781134)]
        site3_measured_time_delays_errors_vpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]


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


        def f(ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z, puls1_x, puls1_y, puls1_z, puls2_x, puls2_y, puls2_z, puls3_x, puls3_y, puls3_z):
            #To generalize, look into from_array_func Minuit initializer.  
            try:
                #fixing the locations of antenna zero.
                ant0_x = 0.0
                ant0_y = 0.0
                ant0_z = 0.0

                #Calculate distances (already converted to ns) from pulser to each antenna

                site1_d0 = (numpy.sqrt((puls1_x - ant0_x)**2 + (puls1_y - ant0_y)**2 + (puls1_z - ant0_z)**2 )/c)*1.0e9 #ns
                site1_d1 = (numpy.sqrt((puls1_x - ant1_x)**2 + (puls1_y - ant1_y)**2 + (puls1_z - ant1_z)**2 )/c)*1.0e9 #ns
                site1_d2 = (numpy.sqrt((puls1_x - ant2_x)**2 + (puls1_y - ant2_y)**2 + (puls1_z - ant2_z)**2 )/c)*1.0e9 #ns
                site1_d3 = (numpy.sqrt((puls1_x - ant3_x)**2 + (puls1_y - ant3_y)**2 + (puls1_z - ant3_z)**2 )/c)*1.0e9 #ns

                site2_d0 = (numpy.sqrt((puls2_x - ant0_x)**2 + (puls2_y - ant0_y)**2 + (puls2_z - ant0_z)**2 )/c)*1.0e9 #ns
                site2_d1 = (numpy.sqrt((puls2_x - ant1_x)**2 + (puls2_y - ant1_y)**2 + (puls2_z - ant1_z)**2 )/c)*1.0e9 #ns
                site2_d2 = (numpy.sqrt((puls2_x - ant2_x)**2 + (puls2_y - ant2_y)**2 + (puls2_z - ant2_z)**2 )/c)*1.0e9 #ns
                site2_d3 = (numpy.sqrt((puls2_x - ant3_x)**2 + (puls2_y - ant3_y)**2 + (puls2_z - ant3_z)**2 )/c)*1.0e9 #ns

                site3_d0 = (numpy.sqrt((puls3_x - ant0_x)**2 + (puls3_y - ant0_y)**2 + (puls3_z - ant0_z)**2 )/c)*1.0e9 #ns
                site3_d1 = (numpy.sqrt((puls3_x - ant1_x)**2 + (puls3_y - ant1_y)**2 + (puls3_z - ant1_z)**2 )/c)*1.0e9 #ns
                site3_d2 = (numpy.sqrt((puls3_x - ant2_x)**2 + (puls3_y - ant2_y)**2 + (puls3_z - ant2_z)**2 )/c)*1.0e9 #ns
                site3_d3 = (numpy.sqrt((puls3_x - ant3_x)**2 + (puls3_y - ant3_y)**2 + (puls3_z - ant3_z)**2 )/c)*1.0e9 #ns



                #calculate differences in distances (already converted to ns) travelled and convert to expected time delays for known pairs.
                #Here I add cable delays, because I am looking at physical distance + delay to measurement to match the measured time differences.
                #Ant 0 - Ant 1
                site1_p0 = (site1_d0 + cable_delays[0]) - (site1_d1 + cable_delays[1])
                site2_p0 = (site2_d0 + cable_delays[0]) - (site2_d1 + cable_delays[1])
                site3_p0 = (site3_d0 + cable_delays[0]) - (site3_d1 + cable_delays[1])

                #Ant 0 - Ant 2
                site1_p1 = (site1_d0 + cable_delays[0]) - (site1_d2 + cable_delays[2])
                site2_p1 = (site2_d0 + cable_delays[0]) - (site2_d2 + cable_delays[2])
                site3_p1 = (site3_d0 + cable_delays[0]) - (site3_d2 + cable_delays[2])

                #Ant 0 - Ant 3
                site1_p2 = (site1_d0 + cable_delays[0]) - (site1_d3 + cable_delays[3])
                site2_p2 = (site2_d0 + cable_delays[0]) - (site2_d3 + cable_delays[3])
                site3_p2 = (site3_d0 + cable_delays[0]) - (site3_d3 + cable_delays[3])

                #Ant 1 - Ant 2
                site1_p3 = (site1_d1 + cable_delays[1]) - (site1_d2 + cable_delays[2])
                site2_p3 = (site2_d1 + cable_delays[1]) - (site2_d2 + cable_delays[2])
                site3_p3 = (site3_d1 + cable_delays[1]) - (site3_d2 + cable_delays[2])

                #Ant 1 - Ant 3
                site1_p4 = (site1_d1 + cable_delays[1]) - (site1_d3 + cable_delays[3])
                site2_p4 = (site2_d1 + cable_delays[1]) - (site2_d3 + cable_delays[3])
                site3_p4 = (site3_d1 + cable_delays[1]) - (site3_d3 + cable_delays[3])

                #Ant 2 - Ant 3
                site1_p5 = (site1_d2 + cable_delays[2]) - (site1_d3 + cable_delays[3])
                site2_p5 = (site2_d2 + cable_delays[2]) - (site2_d3 + cable_delays[3])
                site3_p5 = (site3_d2 + cable_delays[2]) - (site3_d3 + cable_delays[3])


                if False:
                    print('\n')
                    print('site1_p0 = %0.3f \t site1_measured_time_delays[0][1] = %0.3f \t diff = %0.3f'%(site1_p0, site1_measured_time_delays[0][1], site1_p0 - site1_measured_time_delays[0][1]))
                    print('site2_p0 = %0.3f \t site2_measured_time_delays[0][1] = %0.3f \t diff = %0.3f'%(site2_p0, site2_measured_time_delays[0][1], site2_p0 - site2_measured_time_delays[0][1]))
                    print('site3_p0 = %0.3f \t site3_measured_time_delays[0][1] = %0.3f \t diff = %0.3f'%(site3_p0, site3_measured_time_delays[0][1], site3_p0 - site3_measured_time_delays[0][1]))
                    print('site1_p1 = %0.3f \t site1_measured_time_delays[1][1] = %0.3f \t diff = %0.3f'%(site1_p1, site1_measured_time_delays[1][1], site1_p1 - site1_measured_time_delays[1][1]))
                    print('site2_p1 = %0.3f \t site2_measured_time_delays[1][1] = %0.3f \t diff = %0.3f'%(site2_p1, site2_measured_time_delays[1][1], site2_p1 - site2_measured_time_delays[1][1]))
                    print('site3_p1 = %0.3f \t site3_measured_time_delays[1][1] = %0.3f \t diff = %0.3f'%(site3_p1, site3_measured_time_delays[1][1], site3_p1 - site3_measured_time_delays[1][1]))
                    print('site1_p2 = %0.3f \t site1_measured_time_delays[2][1] = %0.3f \t diff = %0.3f'%(site1_p2, site1_measured_time_delays[2][1], site1_p2 - site1_measured_time_delays[2][1]))
                    print('site2_p2 = %0.3f \t site2_measured_time_delays[2][1] = %0.3f \t diff = %0.3f'%(site2_p2, site2_measured_time_delays[2][1], site2_p2 - site2_measured_time_delays[2][1]))
                    print('site3_p2 = %0.3f \t site3_measured_time_delays[2][1] = %0.3f \t diff = %0.3f'%(site3_p2, site3_measured_time_delays[2][1], site3_p2 - site3_measured_time_delays[2][1]))
                    print('site1_p3 = %0.3f \t site1_measured_time_delays[3][1] = %0.3f \t diff = %0.3f'%(site1_p3, site1_measured_time_delays[3][1], site1_p3 - site1_measured_time_delays[3][1]))
                    print('site2_p3 = %0.3f \t site2_measured_time_delays[3][1] = %0.3f \t diff = %0.3f'%(site2_p3, site2_measured_time_delays[3][1], site2_p3 - site2_measured_time_delays[3][1]))
                    print('site3_p3 = %0.3f \t site3_measured_time_delays[3][1] = %0.3f \t diff = %0.3f'%(site3_p3, site3_measured_time_delays[3][1], site3_p3 - site3_measured_time_delays[3][1]))
                    print('site1_p4 = %0.3f \t site1_measured_time_delays[4][1] = %0.3f \t diff = %0.3f'%(site1_p4, site1_measured_time_delays[4][1], site1_p4 - site1_measured_time_delays[4][1]))
                    print('site2_p4 = %0.3f \t site2_measured_time_delays[4][1] = %0.3f \t diff = %0.3f'%(site2_p4, site2_measured_time_delays[4][1], site2_p4 - site2_measured_time_delays[4][1]))
                    print('site3_p4 = %0.3f \t site3_measured_time_delays[4][1] = %0.3f \t diff = %0.3f'%(site3_p4, site3_measured_time_delays[4][1], site3_p4 - site3_measured_time_delays[4][1]))
                    print('site1_p5 = %0.3f \t site1_measured_time_delays[5][1] = %0.3f \t diff = %0.3f'%(site1_p5, site1_measured_time_delays[5][1], site1_p5 - site1_measured_time_delays[5][1]))
                    print('site2_p5 = %0.3f \t site2_measured_time_delays[5][1] = %0.3f \t diff = %0.3f'%(site2_p5, site2_measured_time_delays[5][1], site2_p5 - site2_measured_time_delays[5][1]))
                    print('site3_p5 = %0.3f \t site3_measured_time_delays[5][1] = %0.3f \t diff = %0.3f'%(site3_p5, site3_measured_time_delays[5][1], site3_p5 - site3_measured_time_delays[5][1]))

                w = lambda old, new : numpy.exp(old - new)**2
                measured_baselines = {'01':129*0.3048,
                                      '02':163*0.3048,
                                      '03':181*0.3048,
                                      '12':151*0.3048,
                                      '13':102*0.3048,
                                      '23':85 *0.3048}

                current_baselines = {   '01':numpy.sqrt((ant0_x - ant1_x)**2 + (ant0_y - ant1_y)**2 + (ant0_z - ant1_z)**2),\
                                        '02':numpy.sqrt((ant0_x - ant2_x)**2 + (ant0_y - ant2_y)**2 + (ant0_z - ant2_z)**2),\
                                        '03':numpy.sqrt((ant0_x - ant3_x)**2 + (ant0_y - ant3_y)**2 + (ant0_z - ant3_z)**2),\
                                        '12':numpy.sqrt((ant1_x - ant2_x)**2 + (ant1_y - ant2_y)**2 + (ant1_z - ant2_z)**2),\
                                        '13':numpy.sqrt((ant1_x - ant3_x)**2 + (ant1_y - ant3_y)**2 + (ant1_z - ant3_z)**2),\
                                        '23':numpy.sqrt((ant2_x - ant3_x)**2 + (ant2_y - ant3_y)**2 + (ant2_z - ant3_z)**2)}

                baseline_weights = {  '01':w(measured_baselines['01'], current_baselines['01']),
                                      '02':w(measured_baselines['02'], current_baselines['02']),
                                      '03':w(measured_baselines['03'], current_baselines['03']),
                                      '12':w(measured_baselines['12'], current_baselines['12']),
                                      '13':w(measured_baselines['13'], current_baselines['13']),
                                      '23':w(measured_baselines['23'], current_baselines['23'])}

                #These are the measured height differences in antennas based on the GPS used for pulsing.
                #I am not using their absolute values, but I hope their relative differences will be representative of
                #reality.  There is no measurements with the same gps for the array so i'll just weight pulsers. 

                measured_pulser_relative_heights = {'1-2':3763.1-3690.7,\
                                                   '2-3':3690.7-3806.25,\
                                                   '1-3':3763.1-3806.25}

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
                            ((site3_p5 - site3_measured_time_delays[5][1])**2)/site3_measured_time_delays_errors[5][1]**2 + \
                            \
                            ((current_baselines['01'] - measured_baselines['01'])**2)/baseline_weights['01'] + \
                            ((current_baselines['02'] - measured_baselines['02'])**2)/baseline_weights['02'] + \
                            ((current_baselines['03'] - measured_baselines['03'])**2)/baseline_weights['03'] + \
                            ((current_baselines['12'] - measured_baselines['12'])**2)/baseline_weights['12'] + \
                            ((current_baselines['13'] - measured_baselines['13'])**2)/baseline_weights['13'] + \
                            ((current_baselines['23'] - measured_baselines['23'])**2)/baseline_weights['23'] + \
                            \
                            ((puls1_z - puls2_z) - (measured_pulser_relative_heights['1-2']))**2.0 +\
                            ((puls2_z - puls3_z) - (measured_pulser_relative_heights['2-3']))**2.0 +\
                            ((puls1_z - puls3_z) - (measured_pulser_relative_heights['1-3']))**2.0

                return chi_2
            except Exception as e:
                print('Error in f')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


        initial_chi_2 = f(ant1_physical_x,ant1_physical_y,ant1_physical_z,ant2_physical_x,ant2_physical_y,ant2_physical_z,ant3_physical_x,ant3_physical_y,ant3_physical_z,site1_pulser_location[0],site1_pulser_location[1],site1_pulser_location[2],site2_pulser_location[0],site2_pulser_location[1],site2_pulser_location[2],site3_pulser_location[0],site3_pulser_location[1],site3_pulser_location[2])
        print('Initial Chi^2 is %0.3f'%(initial_chi_2))

        initial_step = 0.01 #m
        #-12 ft on pulser locations relative to antennas to account for additional mast elevation.
        m = Minuit(     f,\
                        ant1_x=ant1_physical_x,\
                        ant1_y=ant1_physical_y,\
                        ant1_z=ant1_physical_z,\
                        ant2_x=ant2_physical_x,\
                        ant2_y=ant2_physical_y,\
                        ant2_z=ant2_physical_z,\
                        ant3_x=ant3_physical_x,\
                        ant3_y=ant3_physical_y,\
                        ant3_z=ant3_physical_z,\
                        puls1_x=site1_pulser_location[0],\
                        puls1_y=site1_pulser_location[1],\
                        puls1_z=site1_pulser_location[2] - 12.0*0.3048,\
                        puls2_x=site2_pulser_location[0],\
                        puls2_y=site2_pulser_location[1],\
                        puls2_z=site2_pulser_location[2] - 12.0*0.3048,\
                        puls3_x=site3_pulser_location[0],\
                        puls3_y=site3_pulser_location[1],\
                        puls3_z=site3_pulser_location[2] - 12.0*0.3048,\
                        error_ant1_x=initial_step,\
                        error_ant1_y=initial_step,\
                        error_ant1_z=initial_step/2.0,\
                        error_ant2_x=initial_step,\
                        error_ant2_y=initial_step,\
                        error_ant2_z=initial_step/2.0,\
                        error_ant3_x=initial_step,\
                        error_ant3_y=initial_step,\
                        error_ant3_z=initial_step/2.0,\
                        error_puls1_x=initial_step,\
                        error_puls1_y=initial_step,\
                        error_puls1_z=initial_step,\
                        error_puls2_x=initial_step,\
                        error_puls2_y=initial_step,\
                        error_puls2_z=initial_step,\
                        error_puls3_x=initial_step,\
                        error_puls3_y=initial_step,\
                        error_puls3_z=initial_step,\
                        errordef = 1.0,\
                        limit_ant1_x=ant1_physical_limits_x,\
                        limit_ant1_y=ant1_physical_limits_y,\
                        limit_ant1_z=ant1_physical_limits_z,\
                        limit_ant2_x=ant2_physical_limits_x,\
                        limit_ant2_y=ant2_physical_limits_y,\
                        limit_ant2_z=ant2_physical_limits_z,\
                        limit_ant3_x=ant3_physical_limits_x,\
                        limit_ant3_y=ant3_physical_limits_y,\
                        limit_ant3_z=ant3_physical_limits_z,\
                        limit_puls1_x=puls1_x_limits,\
                        limit_puls1_y=puls1_y_limits,\
                        limit_puls1_z=puls1_z_limits,\
                        limit_puls2_x=puls2_x_limits,\
                        limit_puls2_y=puls2_y_limits,\
                        limit_puls2_z=puls2_z_limits,\
                        limit_puls3_x=puls3_x_limits,\
                        limit_puls3_y=puls3_y_limits,\
                        limit_puls3_z=puls3_z_limits)


        result = m.migrad()
        m.hesse()
        m.minos()
        pprint(m.get_fmin())
        print(result)

        #12 variables
        ant0_phase_x = 0.0#m.values['ant0_x']
        ant0_phase_y = 0.0#m.values['ant0_y']
        ant0_phase_z = 0.0#m.values['ant0_z']

        ant1_phase_x = m.values['ant1_x']
        ant1_phase_y = m.values['ant1_y']
        ant1_phase_z = m.values['ant1_z']

        ant2_phase_x = m.values['ant2_x']
        ant2_phase_y = m.values['ant2_y']
        ant2_phase_z = m.values['ant2_z']

        ant3_phase_x = m.values['ant3_x']
        ant3_phase_y = m.values['ant3_y']
        ant3_phase_z = m.values['ant3_z']

        puls1_phase_x = m.values['puls1_x']
        puls1_phase_y = m.values['puls1_y']
        puls1_phase_z = m.values['puls1_z']
        puls2_phase_x = m.values['puls2_x']
        puls2_phase_y = m.values['puls2_y']
        puls2_phase_z = m.values['puls2_z']
        puls3_phase_x = m.values['puls3_x']
        puls3_phase_y = m.values['puls3_y']
        puls3_phase_z = m.values['puls3_z']

        phase_locs = numpy.array([[ant0_phase_x,ant0_phase_y,ant0_phase_z],[ant1_phase_x,ant1_phase_y,ant1_phase_z],[ant2_phase_x,ant2_phase_y,ant2_phase_z],[ant3_phase_x,ant3_phase_y,ant3_phase_z]])

        print('Antenna Locations: \n%s'%str(antenna_locs))
        print('Phase Locations: \n%s'%str(phase_locs))

        print('\nDifference (antenna_locs - phase_locs): \n%s'%str(antenna_locs - phase_locs))

        print('\nSite 1 Physical Location: \n%s'%str((site1_pulser_location[0], site1_pulser_location[1], site1_pulser_location[2])))
        print('Site 1 Phase Location: \n%s'%str((puls1_phase_x, puls1_phase_y, puls1_phase_z)))

        print('\nSite 2 Physical Location: \n%s'%str((site2_pulser_location[0], site2_pulser_location[1], site2_pulser_location[2])))
        print('Site 2 Phase Location: \n%s'%str((puls2_phase_x, puls2_phase_y, puls2_phase_z)))

        print('\nSite 3 Physical Location: \n%s'%str((site3_pulser_location[0], site3_pulser_location[1], site3_pulser_location[2])))
        print('Site 3 Phase Location: \n%s'%str((puls3_phase_x, puls3_phase_y, puls3_phase_z)))

        measured_baselines = {'01':129*0.3048,
                              '02':163*0.3048,
                              '03':181*0.3048,
                              '12':151*0.3048,
                              '13':102*0.3048,
                              '23':85 *0.3048}
        baselines = {}
        print('Measured Baseline  -  Phase Baseline  =  ')
        for pair in pairs:
            #print('Measured Baseline = ', measured_baselines[str(min(pair))+str(max(pair))])
            baselines[str(min(pair))+str(max(pair))] = numpy.sqrt((phase_locs[min(pair)][0] - phase_locs[max(pair)][0])**2 + (phase_locs[min(pair)][1] - phase_locs[max(pair)][1])**2 + (phase_locs[min(pair)][2] - phase_locs[max(pair)][2])**2)
            #print('Phase Baseline = ', baselines[str(min(pair))+str(max(pair))])
            print('%0.3f  -  %0.3f  =  %0.3f'%(measured_baselines[str(min(pair))+str(max(pair))], baselines[str(min(pair))+str(max(pair))], measured_baselines[str(min(pair))+str(max(pair))]-baselines[str(min(pair))+str(max(pair))]))


        if True:
            colors = ['g','r','b','m']
            antennas_physical = info.loadAntennaLocationsENU(deploy_index=1)[0]

            fig = plt.figure()
            fig.canvas.set_window_title('Antenna Locations')
            ax = fig.add_subplot(111, projection='3d')

            for i, a in antennas_physical.items():
                ax.scatter(a[0], a[1], a[2], marker='o',color=colors[i],label='Physical %i'%i)

            for i, a in enumerate(phase_locs):
                ax.scatter(a[0], a[1], a[2], marker='*',color=colors[i],label='%s Phase Center %i'%(mode, i))

            pulser_locations = info.loadPulserLocationsENU()['physical']


            for site, key in enumerate(['run1507','run1509','run1511']):
                site += 1
                ax.scatter(pulser_locations[key][0], pulser_locations[key][1], pulser_locations[key][2], color='k', marker='o',label='Physical Pulser Site %i'%site)

            ax.plot([pulser_locations['run1507'][0],puls1_phase_x],[pulser_locations['run1507'][1],puls1_phase_y],[pulser_locations['run1507'][2],puls1_phase_z],linestyle='--')
            ax.scatter(puls1_phase_x, puls1_phase_y, puls1_phase_z, color='k', marker='*',label='Physical Pulser Site %i'%1)

            ax.plot([pulser_locations['run1509'][0],puls2_phase_x],[pulser_locations['run1509'][1],puls2_phase_y],[pulser_locations['run1509'][2],puls2_phase_z],linestyle='--')
            ax.scatter(puls2_phase_x, puls2_phase_y, puls2_phase_z, color='k', marker='*',label='Physical Pulser Site %i'%2)

            ax.plot([pulser_locations['run1511'][0],puls3_phase_x],[pulser_locations['run1511'][1],puls3_phase_y],[pulser_locations['run1511'][2],puls3_phase_z],linestyle='--')
            ax.scatter(puls3_phase_x, puls3_phase_y, puls3_phase_z, color='k', marker='*',label='Physical Pulser Site %i'%3)

            ax.set_xlabel('E (m)')
            ax.set_ylabel('N (m)')
            ax.set_zlabel('Relative Elevation (m)')
            plt.legend()


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






