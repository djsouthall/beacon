'''
Given the best measured time delays for each of the 6 pulsing sites from the July 2021 BEACON pulsing campaign, this
will attempt to minimize the antenna positions and cable delays.  
'''

#General Imports
import numpy
import itertools
import os
import sys
import csv
import scipy
import scipy.interpolate
import pymap3d as pm
from iminuit import Minuit
import inspect
import h5py

#Personal Imports
from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_handler import createFile, getTimes
import beacon.tools.station as bc
import beacon.tools.info as info
import beacon.tools.get_plane_tracks as pt
from beacon.tools.fftmath import TimeDelayCalculator
from beacon.tools.data_slicer import dataSlicerSingleRun
from beacon.tools.correlator import Correlator
import beacon.tools.config_reader as bcr


#Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment


#Settings
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()
datapath = os.environ['BEACON_DATA']

n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
c = 299792458/n #m/s


measured_time_delays = {}

measured_time_delays['d2sa'] = {}

measured_time_delays['d2sa']['hpol'] = {}
measured_time_delays['d2sa']['hpol']['delays_ns'] = numpy.array([-63.14130048,  99.38889723,  15.64368973, 162.3543261 ,  78.65923556, -83.85536426])
measured_time_delays['d2sa']['hpol']['sigma_ns'] = numpy.array([0.17644782, 0.1987261 , 0.1869557 , 0.17651783, 0.20839108, 0.16655057])

measured_time_delays['d2sa']['vpol'] = {}
measured_time_delays['d2sa']['vpol']['delays_ns'] = numpy.array([-70.02166451,  86.85031485,  11.30731964, 156.85079363,  81.30439472, -75.52956864])
measured_time_delays['d2sa']['vpol']['sigma_ns'] = numpy.array([0.11617623, 0.10046536, 0.14437393, 0.12835038, 0.13399128, 0.11643392])


measured_time_delays['d3sa'] = {}

measured_time_delays['d3sa']['hpol'] = {}
measured_time_delays['d3sa']['hpol']['delays_ns'] = numpy.array([-100.9643034 ,   39.12872542,  -52.79778144,  140.02931026,  48.10385574,  -92.04129368])
measured_time_delays['d3sa']['hpol']['sigma_ns'] = numpy.array([0.06557082, 0.07901542, 0.05177623, 0.07434778, 0.05871251, 0.06644926])

measured_time_delays['d3sa']['vpol'] = {}
measured_time_delays['d3sa']['vpol']['delays_ns'] = numpy.array([-107.24171021,   26.75577314,  -56.59551099,  134.310878  ,  50.92560667,  -83.4251287 ])
measured_time_delays['d3sa']['vpol']['sigma_ns'] = numpy.array([0.00387661, 0.11545836, 0.16632272, 0.14901272, 0.26207502, 0.18229291])


measured_time_delays['d3sb'] = {}

measured_time_delays['d3sb']['hpol'] = {}
measured_time_delays['d3sb']['hpol']['delays_ns'] = numpy.array([-118.24860437,    3.11211276,  -89.96367379,  121.1021402 ,  28.22563402,  -93.03585489])
measured_time_delays['d3sb']['hpol']['sigma_ns'] = numpy.array([0.07043019, 0.07144881, 0.11915923, 0.06628051, 0.11001272, 0.11761661])

measured_time_delays['d3sb']['vpol'] = {}
measured_time_delays['d3sb']['vpol']['delays_ns'] = numpy.array([-124.30123945,   -9.2209266 ,  -93.83340383,  115.03118086,  30.68261666,  -84.49031588])
measured_time_delays['d3sb']['vpol']['sigma_ns'] = numpy.array([0.19245565, 0.25414605, 0.31903936, 0.29155481, 0.37501811, 0.37506079])


measured_time_delays['d3sc'] = {}

measured_time_delays['d3sc']['hpol'] = {}
# measured_time_delays['d3sc']['hpol']['delays_ns'] = numpy.array([-124.50656734,  -12.67950929, -105.88945604,  111.74108185,  18.69346089,  -93.14504255])
# measured_time_delays['d3sc']['hpol']['sigma_ns'] = numpy.array([0.09842963, 0.10768558, 0.09280989, 0.07931319, 0.10520202, 0.11498814])
measured_time_delays['d3sc']['hpol']['delays_ns'] = numpy.array([-124.91391579,  -13.15003191, -105.93984956,  111.70737811, 19.02486885,  -92.80343433])
measured_time_delays['d3sc']['hpol']['sigma_ns'] = numpy.array([0.22920522, 0.19421038, 0.21978905, 0.14111465, 0.21970912, 0.24054242])

measured_time_delays['d3sc']['vpol'] = {}
measured_time_delays['d3sc']['vpol']['delays_ns'] = numpy.array([-131.23459943,  -25.33789367, -109.8755877 ,  105.97147853,  21.44267034,  -84.53585933])
measured_time_delays['d3sc']['vpol']['sigma_ns'] = numpy.array([0.09881008, 0.13022104, 0.13173773, 0.12295037, 0.13130819, 0.12364438])


measured_time_delays['d4sa'] = {}

measured_time_delays['d4sa']['hpol'] = {}
measured_time_delays['d4sa']['hpol']['delays_ns'] = numpy.array([-131.05787774,  -80.48935891, -156.11747805,   50.37710341, -25.22189312,  -75.68285714])
measured_time_delays['d4sa']['hpol']['sigma_ns'] = numpy.array([0.11962765, 0.13788958, 0.22954265, 0.13345461, 0.22534428, 0.28682674])

measured_time_delays['d4sa']['vpol'] = {}
measured_time_delays['d4sa']['vpol']['delays_ns'] = numpy.array([-137.63837021,  -93.07503106, -160.44372872,   44.55829189, -22.84918766,  -67.41484311])
measured_time_delays['d4sa']['vpol']['sigma_ns'] = numpy.array([0.19247366, 0.09339184, 0.22085245, 0.19228158, 0.23594743, 0.22635479])


measured_time_delays['d4sb'] = {}
# Hpol 4-6 was done using a windowed signal because it had a weird cross corr that seemed to be a cycle off of the expected delay. 
measured_time_delays['d4sb']['hpol'] = {}
measured_time_delays['d4sb']['hpol']['delays_ns'] = numpy.array([-108.48271003, -145.52936836, -182.42420984,  -37.07392944, -74.3088871 ,  -36.91059479])
measured_time_delays['d4sb']['hpol']['sigma_ns'] = numpy.array([0.14458919, 0.12773987, 0.23587384, 0.10636644, 0.28704927, 0.17559892])


measured_time_delays['d4sb']['vpol'] = {}
measured_time_delays['d4sb']['vpol']['delays_ns'] = numpy.array([-115.07169322, -157.99423434, -186.63177201,  -43.00198129, -71.51992424,  -28.60474262])
measured_time_delays['d4sb']['vpol']['sigma_ns'] = numpy.array([0.17114803, 0.16868515, 0.21001127, 0.23489705, 0.157785  , 0.3050704 ])

#The attenuations used to get the time delays
attenuations_dict = {'hpol':{           'd2sa' : [20],
                                        'd3sa' : [10],
                                        'd3sb' : [6],
                                        'd3sc' : [10],
                                        'd4sa' : [20],
                                        'd4sb' : [6]
                                    },
                             'vpol':{   'd2sa' : [10],
                                        'd3sa' : [6],
                                        'd3sb' : [20],
                                        'd3sc' : [10],
                                        'd4sa' : [10],
                                        'd4sb' : [6]
                                    }
                            }


class AntennaMinimizer:
    '''
    This is the minimizer to be used for a single polarization.  It is intended to readily encapsulate all of the
    necessary prepwork for a single pol, such that the 2 could be readily combined easily into a single minimization.

    This doesn't actually include the minimizer itself, but rather the meta data for the chi^2 function that will
    be minimized.  How that function is handled is still done externally.
    '''
    def __init__(self, pol, deploy_index, origin=None, use_sites=['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb'], 
                    random_offset_amount=0.05,
                    included_antennas_lumped=[0,1,2,3],
                    include_baselines=[0,1,2,3,4,5],
                    initial_step_x=0.1,
                    initial_step_y=0.1,
                    initial_step_z=0.1,
                    initial_step_cable_delay=3,
                    cable_delay_guess_range=30,
                    antenna_position_guess_range_x=0.75,
                    antenna_position_guess_range_y=0.75,
                    antenna_position_guess_range_z=0.75,
                    manual_offset_ant0_x=0,
                    manual_offset_ant0_y=0,
                    manual_offset_ant0_z=0,
                    manual_offset_ant1_x=0,
                    manual_offset_ant1_y=0,
                    manual_offset_ant1_z=0,
                    manual_offset_ant2_x=0,
                    manual_offset_ant2_y=0,
                    manual_offset_ant2_z=0,
                    manual_offset_ant3_x=0,
                    manual_offset_ant3_y=0,
                    manual_offset_ant3_z=0,
                    fix_ant0_x=True,
                    fix_ant0_y=True,
                    fix_ant0_z=True,
                    fix_ant1_x=False,
                    fix_ant1_y=False,
                    fix_ant1_z=False,
                    fix_ant2_x=False,
                    fix_ant2_y=False,
                    fix_ant2_z=False,
                    fix_ant3_x=False,
                    fix_ant3_y=False,
                    fix_ant3_z=False,
                    fix_cable_delay0=True,
                    fix_cable_delay1=False,
                    fix_cable_delay2=False,
                    fix_cable_delay3=False):
        try:
            self.included_antennas_lumped = included_antennas_lumped
            self.include_baselines = include_baselines
            self.included_antennas_channels = numpy.concatenate([[2*i,2*i+1] for i in self.included_antennas_lumped])
            self.deploy_index = deploy_index
            self.pol = pol
            self.use_sites = use_sites


            #Force antennas not to be included to be fixed.  
            if not(0 in self.included_antennas_lumped):
                fix_ant0_x = True
                fix_ant0_y = True
                fix_ant0_z = True
                fix_cable_delay0 = True

            if not(1 in self.included_antennas_lumped):
                fix_ant1_x = True
                fix_ant1_y = True
                fix_ant1_z = True
                fix_cable_delay1 = True
            if not(2 in self.included_antennas_lumped):
                fix_ant2_x = True
                fix_ant2_y = True
                fix_ant2_z = True
                fix_cable_delay2 = True
            if not(3 in self.included_antennas_lumped):
                fix_ant3_x = True
                fix_ant3_y = True
                fix_ant3_z = True
                fix_cable_delay3 = True

            #This math is to set the pairs to include in the calculation.  Typically it will be all of them, but if the option is enabled to remove some
            #from the calculation then this will allow for that to be done.
            self.pairs = numpy.array(list(itertools.combinations((0,1,2,3), 2)))
            self.pairs_cut = []
            for pair_index, pair in enumerate(numpy.array(list(itertools.combinations((0,1,2,3), 2)))):
                self.pairs_cut.append(numpy.logical_and(numpy.all(numpy.isin(numpy.array(pair),self.included_antennas_lumped)), pair_index in self.include_baselines)) #include_baselines Overwritten when antennas removed.
            self.include_baselines = numpy.where(self.pairs_cut)[0] #Effectively the same as the self.pairs_cut but index based for baselines.
            #include_baselines = numpy.arange(4)
            print('Including baseline self.pairs:')
            print(self.pairs[self.pairs_cut])

            #I think adding an absolute time offset for each antenna and letting that vary could be interesting.  It could be used to adjust the cable delays.
            self.cable_delays = info.loadCableDelays(deploy_index=self.deploy_index,return_raw=True)[self.pol]

            if origin is None:
                self.origin = info.loadAntennaZeroLocation(deploy_index=self.deploy_index)
            else:
                self.origin = origin

            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=self.deploy_index)

            pulser_info = PulserInfo()

            self.pulser_locations_ENU = {}

            for site in self.use_sites:        
                #Prepare correlators for future use on a per event basis
                source_latlonel = pulser_info.getPulserLatLonEl(site)
                
                # Prepare expected angle and arrival times
                self.pulser_locations_ENU[site] = pm.geodetic2enu(source_latlonel[0],source_latlonel[1],source_latlonel[2],self.origin[0],self.origin[1],self.origin[2])


            if True:
                if self.pol == 'hpol':
                    self.antennas_phase_start = antennas_phase_hpol
                else:
                    self.antennas_phase_start = antennas_phase_vpol
            else:
                print('WARNING, USING PHYSICAL LOCATIONS TO START')
                self.antennas_phase_start = antennas_physical

            if cable_delay_guess_range is not None:

                limit_cable_delay0 = (self.cable_delays[0] - cable_delay_guess_range , self.cable_delays[0] + cable_delay_guess_range)
                limit_cable_delay1 = (self.cable_delays[1] - cable_delay_guess_range , self.cable_delays[1] + cable_delay_guess_range)
                limit_cable_delay2 = (self.cable_delays[2] - cable_delay_guess_range , self.cable_delays[2] + cable_delay_guess_range)
                limit_cable_delay3 = (self.cable_delays[3] - cable_delay_guess_range , self.cable_delays[3] + cable_delay_guess_range)
            else:
                limit_cable_delay0 = None
                limit_cable_delay1 = None
                limit_cable_delay2 = None
                limit_cable_delay3 = None

            if antenna_position_guess_range_x is not None:
                ant0_physical_limits_x = (self.antennas_phase_start[0][0] + manual_offset_ant0_x - antenna_position_guess_range_x ,self.antennas_phase_start[0][0] + manual_offset_ant0_x + antenna_position_guess_range_x)
                ant1_physical_limits_x = (self.antennas_phase_start[1][0] + manual_offset_ant1_x - antenna_position_guess_range_x ,self.antennas_phase_start[1][0] + manual_offset_ant1_x + antenna_position_guess_range_x)
                ant2_physical_limits_x = (self.antennas_phase_start[2][0] + manual_offset_ant2_x - antenna_position_guess_range_x ,self.antennas_phase_start[2][0] + manual_offset_ant2_x + antenna_position_guess_range_x)
                ant3_physical_limits_x = (self.antennas_phase_start[3][0] + manual_offset_ant3_x - antenna_position_guess_range_x ,self.antennas_phase_start[3][0] + manual_offset_ant3_x + antenna_position_guess_range_x)
            else:
                ant0_physical_limits_x = None
                ant1_physical_limits_x = None
                ant2_physical_limits_x = None
                ant3_physical_limits_x = None

            if antenna_position_guess_range_y is not None:
                ant0_physical_limits_y = (self.antennas_phase_start[0][1] + manual_offset_ant0_y - antenna_position_guess_range_y ,self.antennas_phase_start[0][1] + manual_offset_ant0_y + antenna_position_guess_range_y)
                ant1_physical_limits_y = (self.antennas_phase_start[1][1] + manual_offset_ant1_y - antenna_position_guess_range_y ,self.antennas_phase_start[1][1] + manual_offset_ant1_y + antenna_position_guess_range_y)
                ant2_physical_limits_y = (self.antennas_phase_start[2][1] + manual_offset_ant2_y - antenna_position_guess_range_y ,self.antennas_phase_start[2][1] + manual_offset_ant2_y + antenna_position_guess_range_y)
                ant3_physical_limits_y = (self.antennas_phase_start[3][1] + manual_offset_ant3_y - antenna_position_guess_range_y ,self.antennas_phase_start[3][1] + manual_offset_ant3_y + antenna_position_guess_range_y)
            else:
                ant0_physical_limits_y = None
                ant1_physical_limits_y = None
                ant2_physical_limits_y = None
                ant3_physical_limits_y = None

            if antenna_position_guess_range_z is not None:
                ant0_physical_limits_z = (self.antennas_phase_start[0][2] + manual_offset_ant0_z - antenna_position_guess_range_z ,self.antennas_phase_start[0][2] + manual_offset_ant0_z + antenna_position_guess_range_z)
                ant1_physical_limits_z = (self.antennas_phase_start[1][2] + manual_offset_ant1_z - antenna_position_guess_range_z ,self.antennas_phase_start[1][2] + manual_offset_ant1_z + antenna_position_guess_range_z)
                ant2_physical_limits_z = (self.antennas_phase_start[2][2] + manual_offset_ant2_z - antenna_position_guess_range_z ,self.antennas_phase_start[2][2] + manual_offset_ant2_z + antenna_position_guess_range_z)
                ant3_physical_limits_z = (self.antennas_phase_start[3][2] + manual_offset_ant3_z - antenna_position_guess_range_z ,self.antennas_phase_start[3][2] + manual_offset_ant3_z + antenna_position_guess_range_z)
            else:
                ant0_physical_limits_z = None
                ant1_physical_limits_z = None
                ant2_physical_limits_z = None
                ant3_physical_limits_z = None

            if random_offset_amount > 0:
                print('RANDOMLY SHIFTING INPUT POSITIONS BY SPECIFIED AMOUNT:%0.2f m'%random_offset_amount)

            initial_ant0_x = manual_offset_ant0_x + self.antennas_phase_start[0][0] + float(not fix_ant0_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant0_y = manual_offset_ant0_y + self.antennas_phase_start[0][1] + float(not fix_ant0_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant0_z = manual_offset_ant0_z + self.antennas_phase_start[0][2] + float(not fix_ant0_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant1_x = manual_offset_ant1_x + self.antennas_phase_start[1][0] + float(not fix_ant1_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant1_y = manual_offset_ant1_y + self.antennas_phase_start[1][1] + float(not fix_ant1_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant1_z = manual_offset_ant1_z + self.antennas_phase_start[1][2] + float(not fix_ant1_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant2_x = manual_offset_ant2_x + self.antennas_phase_start[2][0] + float(not fix_ant2_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant2_y = manual_offset_ant2_y + self.antennas_phase_start[2][1] + float(not fix_ant2_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant2_z = manual_offset_ant2_z + self.antennas_phase_start[2][2] + float(not fix_ant2_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant3_x = manual_offset_ant3_x + self.antennas_phase_start[3][0] + float(not fix_ant3_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant3_y = manual_offset_ant3_y + self.antennas_phase_start[3][1] + float(not fix_ant3_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant3_z = manual_offset_ant3_z + self.antennas_phase_start[3][2] + float(not fix_ant3_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]

            self.initial_ant0_ENU = numpy.array([initial_ant0_x, initial_ant0_y, initial_ant0_z])
            self.initial_ant1_ENU = numpy.array([initial_ant1_x, initial_ant1_y, initial_ant1_z])
            self.initial_ant2_ENU = numpy.array([initial_ant2_x, initial_ant2_y, initial_ant2_z])
            self.initial_ant3_ENU = numpy.array([initial_ant3_x, initial_ant3_y, initial_ant3_z])

            if True:
                #Initial baselines just to be printed out
                initial_baselines = {   '01':numpy.sqrt((initial_ant0_x - initial_ant1_x)**2 + (initial_ant0_y - initial_ant1_y)**2 + (initial_ant0_z - initial_ant1_z)**2),\
                                        '02':numpy.sqrt((initial_ant0_x - initial_ant2_x)**2 + (initial_ant0_y - initial_ant2_y)**2 + (initial_ant0_z - initial_ant2_z)**2),\
                                        '03':numpy.sqrt((initial_ant0_x - initial_ant3_x)**2 + (initial_ant0_y - initial_ant3_y)**2 + (initial_ant0_z - initial_ant3_z)**2),\
                                        '12':numpy.sqrt((initial_ant1_x - initial_ant2_x)**2 + (initial_ant1_y - initial_ant2_y)**2 + (initial_ant1_z - initial_ant2_z)**2),\
                                        '13':numpy.sqrt((initial_ant1_x - initial_ant3_x)**2 + (initial_ant1_y - initial_ant3_y)**2 + (initial_ant1_z - initial_ant3_z)**2),\
                                        '23':numpy.sqrt((initial_ant2_x - initial_ant3_x)**2 + (initial_ant2_y - initial_ant3_y)**2 + (initial_ant2_z - initial_ant3_z)**2)}
                print(self.pol)
                print('The initial baselines (specified by deploy_index = %s) with random offsets and manual adjustsments in meters are:'%(str(info.returnDefaultDeploy())))
                print(initial_baselines)

                for key in self.use_sites:
                    d0 = numpy.sqrt((self.pulser_locations_ENU[key][0] - initial_ant0_x)**2 + (self.pulser_locations_ENU[key][1] - initial_ant0_y)**2 + (self.pulser_locations_ENU[key][2] - initial_ant0_z)**2 ) #m
                    print('Pulser %s is %0.2f m away from Antenna 0'%(key, d0))


            self.initial_ant0_x=initial_ant0_x
            self.initial_ant0_y=initial_ant0_y
            self.initial_ant0_z=initial_ant0_z
            self.initial_ant1_x=initial_ant1_x
            self.initial_ant1_y=initial_ant1_y
            self.initial_ant1_z=initial_ant1_z
            self.initial_ant2_x=initial_ant2_x
            self.initial_ant2_y=initial_ant2_y
            self.initial_ant2_z=initial_ant2_z
            self.initial_ant3_x=initial_ant3_x
            self.initial_ant3_y=initial_ant3_y
            self.initial_ant3_z=initial_ant3_z
            self.cable_delay0=self.cable_delays[0]
            self.cable_delay1=self.cable_delays[1]
            self.cable_delay2=self.cable_delays[2]
            self.cable_delay3=self.cable_delays[3]

            self.errors = {}
            self.limits = {}
            self.fixed = {}
            self.errors['ant0_x'] = initial_step_x
            self.errors['ant0_y'] = initial_step_y
            self.errors['ant0_z'] = initial_step_z
            self.errors['ant1_x'] = initial_step_x
            self.errors['ant1_y'] = initial_step_y
            self.errors['ant1_z'] = initial_step_z
            self.errors['ant2_x'] = initial_step_x
            self.errors['ant2_y'] = initial_step_y
            self.errors['ant2_z'] = initial_step_z
            self.errors['ant3_x'] = initial_step_x
            self.errors['ant3_y'] = initial_step_y
            self.errors['ant3_z'] = initial_step_z
            self.errors['cable_delay0'] = initial_step_cable_delay
            self.errors['cable_delay1'] = initial_step_cable_delay
            self.errors['cable_delay2'] = initial_step_cable_delay
            self.errors['cable_delay3'] = initial_step_cable_delay
            self.errordef = 1.0
            self.limits['ant0_x'] = ant0_physical_limits_x
            self.limits['ant0_y'] = ant0_physical_limits_y
            self.limits['ant0_z'] = ant0_physical_limits_z
            self.limits['ant1_x'] = ant1_physical_limits_x
            self.limits['ant1_y'] = ant1_physical_limits_y
            self.limits['ant1_z'] = ant1_physical_limits_z
            self.limits['ant2_x'] = ant2_physical_limits_x
            self.limits['ant2_y'] = ant2_physical_limits_y
            self.limits['ant2_z'] = ant2_physical_limits_z
            self.limits['ant3_x'] = ant3_physical_limits_x
            self.limits['ant3_y'] = ant3_physical_limits_y
            self.limits['ant3_z'] = ant3_physical_limits_z
            self.limits['cable_delay0'] = limit_cable_delay0
            self.limits['cable_delay1'] = limit_cable_delay1
            self.limits['cable_delay2'] = limit_cable_delay2
            self.limits['cable_delay3'] = limit_cable_delay3
            self.fixed['ant0_x'] = fix_ant0_x
            self.fixed['ant0_y'] = fix_ant0_y
            self.fixed['ant0_z'] = fix_ant0_z
            self.fixed['ant1_x'] = fix_ant1_x
            self.fixed['ant1_y'] = fix_ant1_y
            self.fixed['ant1_z'] = fix_ant1_z
            self.fixed['ant2_x'] = fix_ant2_x
            self.fixed['ant2_y'] = fix_ant2_y
            self.fixed['ant2_z'] = fix_ant2_z
            self.fixed['ant3_x'] = fix_ant3_x
            self.fixed['ant3_y'] = fix_ant3_y
            self.fixed['ant3_z'] = fix_ant3_z
            self.fixed['cable_delay0'] = fix_cable_delay0
            self.fixed['cable_delay1'] = fix_cable_delay1
            self.fixed['cable_delay2'] = fix_cable_delay2
            self.fixed['cable_delay3'] = fix_cable_delay3
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def rawChi2(self, ant0_x, ant0_y, ant0_z, ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z, cable_delay0, cable_delay1, cable_delay2, cable_delay3):
        '''
        This is a chi^2 that loops over locations from potential RFI, calculating expected time delays for those locations.  Then
        it will compares those to the calculated time delays for suspected corresponding events.  
        '''
        try:
            #Calculate distances (already converted to ns) from pulser to each antenna
            chi_2 = 0.0
            _cable_delays = [cable_delay0,cable_delay1,cable_delay2,cable_delay3]

            for key in self.use_sites:
                d0 = (numpy.sqrt((self.pulser_locations_ENU[key][0] - ant0_x)**2 + (self.pulser_locations_ENU[key][1] - ant0_y)**2 + (self.pulser_locations_ENU[key][2] - ant0_z)**2 )/c)*1.0e9 #ns
                d1 = (numpy.sqrt((self.pulser_locations_ENU[key][0] - ant1_x)**2 + (self.pulser_locations_ENU[key][1] - ant1_y)**2 + (self.pulser_locations_ENU[key][2] - ant1_z)**2 )/c)*1.0e9 #ns
                d2 = (numpy.sqrt((self.pulser_locations_ENU[key][0] - ant2_x)**2 + (self.pulser_locations_ENU[key][1] - ant2_y)**2 + (self.pulser_locations_ENU[key][2] - ant2_z)**2 )/c)*1.0e9 #ns
                d3 = (numpy.sqrt((self.pulser_locations_ENU[key][0] - ant3_x)**2 + (self.pulser_locations_ENU[key][1] - ant3_y)**2 + (self.pulser_locations_ENU[key][2] - ant3_z)**2 )/c)*1.0e9 #ns

                d = [d0,d1,d2,d3]

                for pair_index, pair in enumerate(self.pairs):
                    if self.pairs_cut[pair_index]:
                        geometric_time_delay = (d[pair[0]] + _cable_delays[pair[0]]) - (d[pair[1]] + _cable_delays[pair[1]])
                        vals = ((geometric_time_delay - measured_time_delays[key][self.pol]['delays_ns'][pair_index])**2)/(measured_time_delays[key][self.pol]['sigma_ns'][pair_index]**2)
                        chi_2 += numpy.sum(vals)

            return chi_2
        except Exception as e:
            print('Error in rawChi2')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


class CombinedAntennaMinimizer:
    '''
    Takes 2 AntennaMinimizer objects (of different polarizations) and defines a new chi^2 that combines their info.
    '''
    def __init__(self, am_hpol, am_vpol):
        self.am_hpol = am_hpol
        self.am_vpol = am_vpol

    def rawChi2(self, ant0_x_hpol, ant0_y_hpol, ant0_z_hpol, ant1_x_hpol, ant1_y_hpol, ant1_z_hpol, ant2_x_hpol, ant2_y_hpol, ant2_z_hpol, ant3_x_hpol, ant3_y_hpol, ant3_z_hpol, cable_delay0_hpol, cable_delay1_hpol, cable_delay2_hpol, cable_delay3_hpol, cable_delay0_vpol, cable_delay1_vpol, cable_delay2_vpol, cable_delay3_vpol):
        '''
        This will use the hpol variations to also move about the vpol antennas (such that they move with the same offset
        from the original value).  The combined chi^2 is returned.
        '''
        try:
            ant0_x_hpol_offset = ant0_x_hpol -  self.am_hpol.initial_ant0_x
            ant0_y_hpol_offset = ant0_y_hpol -  self.am_hpol.initial_ant0_y
            ant0_z_hpol_offset = ant0_z_hpol -  self.am_hpol.initial_ant0_z
            ant1_x_hpol_offset = ant1_x_hpol -  self.am_hpol.initial_ant1_x
            ant1_y_hpol_offset = ant1_y_hpol -  self.am_hpol.initial_ant1_y
            ant1_z_hpol_offset = ant1_z_hpol -  self.am_hpol.initial_ant1_z
            ant2_x_hpol_offset = ant2_x_hpol -  self.am_hpol.initial_ant2_x
            ant2_y_hpol_offset = ant2_y_hpol -  self.am_hpol.initial_ant2_y
            ant2_z_hpol_offset = ant2_z_hpol -  self.am_hpol.initial_ant2_z
            ant3_x_hpol_offset = ant3_x_hpol -  self.am_hpol.initial_ant3_x
            ant3_y_hpol_offset = ant3_y_hpol -  self.am_hpol.initial_ant3_y
            ant3_z_hpol_offset = ant3_z_hpol -  self.am_hpol.initial_ant3_z


            chi_2_hpol = self.am_hpol.rawChi2(  ant0_x_hpol_offset + self.am_hpol.initial_ant0_x,
                                                ant0_y_hpol_offset + self.am_hpol.initial_ant0_y,
                                                ant0_z_hpol_offset + self.am_hpol.initial_ant0_z,
                                                ant1_x_hpol_offset + self.am_hpol.initial_ant1_x,
                                                ant1_y_hpol_offset + self.am_hpol.initial_ant1_y,
                                                ant1_z_hpol_offset + self.am_hpol.initial_ant1_z,
                                                ant2_x_hpol_offset + self.am_hpol.initial_ant2_x,
                                                ant2_y_hpol_offset + self.am_hpol.initial_ant2_y,
                                                ant2_z_hpol_offset + self.am_hpol.initial_ant2_z,
                                                ant3_x_hpol_offset + self.am_hpol.initial_ant3_x,
                                                ant3_y_hpol_offset + self.am_hpol.initial_ant3_y,
                                                ant3_z_hpol_offset + self.am_hpol.initial_ant3_z,
                                                cable_delay0_hpol,
                                                cable_delay1_hpol,
                                                cable_delay2_hpol,
                                                cable_delay3_hpol)
            chi_2_vpol = self.am_vpol.rawChi2(  ant0_x_hpol_offset + self.am_vpol.initial_ant0_x,
                                                ant0_y_hpol_offset + self.am_vpol.initial_ant0_y,
                                                ant0_z_hpol_offset + self.am_vpol.initial_ant0_z,
                                                ant1_x_hpol_offset + self.am_vpol.initial_ant1_x,
                                                ant1_y_hpol_offset + self.am_vpol.initial_ant1_y,
                                                ant1_z_hpol_offset + self.am_vpol.initial_ant1_z,
                                                ant2_x_hpol_offset + self.am_vpol.initial_ant2_x,
                                                ant2_y_hpol_offset + self.am_vpol.initial_ant2_y,
                                                ant2_z_hpol_offset + self.am_vpol.initial_ant2_z,
                                                ant3_x_hpol_offset + self.am_vpol.initial_ant3_x,
                                                ant3_y_hpol_offset + self.am_vpol.initial_ant3_y,
                                                ant3_z_hpol_offset + self.am_vpol.initial_ant3_z,
                                                cable_delay0_vpol,
                                                cable_delay1_vpol,
                                                cable_delay2_vpol,
                                                cable_delay3_vpol)
            return chi_2_hpol + chi_2_vpol
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    try:
        plt.close('all')
        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol','both']:
                pol = str(sys.argv[1])
                print('POL SET TO %s'%pol)
            else:
                print('Given pol not in options.  Defaulting to hpol')
                pol = 'hpol'
        else:
            print('No pol given.  Defaulting to hpol')
            pol = 'hpol'

        if True:
            deploy_index = 'rtk-gps-day3-june22-2021.json'#'theodolite-day3-june22-2021_only_enu.json'#'rtk-gps-day3-june22-2021.json'
            if pol == 'hpol':
                use_sites = ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']
            elif pol == 'vpol':
                use_sites = ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']
            elif pol == 'both':
                use_sites = {}
                use_sites['hpol'] = ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']
                use_sites['vpol'] = ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']



        plot_histograms = False
        limit_events = 100
        plot_time_delays_on_maps = True
        plot_expected_direction = True
        
        iterate_sub_baselines = 6 #The lower this is the higher the time it will take to plot.  Does combinatoric subsets of baselines with this length. 

        #Filter settings
        final_corr_length = 2**16
        cor_upsample = final_corr_length
        apply_phase_response = True

        crit_freq_low_pass_MHz = 80
        low_pass_filter_order = 14

        crit_freq_high_pass_MHz = 20
        high_pass_filter_order = 4

        sine_subtract = False
        sine_subtract_min_freq_GHz = 0.02
        sine_subtract_max_freq_GHz = 0.15
        sine_subtract_percent = 0.01
        max_failed_iterations = 3

        plot_filters = False
        plot_multiple = False

        hilbert = False #Apply hilbert envelope to wf before correlating
        align_method = 0 

        shorten_signals = True
        shorten_thresh = 0.7
        shorten_delay = 10.0
        shorten_length = 1500.0
        shorten_keep_leading = 500.0

        waveform_index_range = (None,None)

        random_offset_amount = 0.05#m (every antenna will be stepped randomly by this amount.  Set to 0 if you don't want this. ), Note that this is applied to 
        included_antennas_lumped = [0,1,2,3]#[0,1,2,3] #If an antenna is not in this list then it will not be included in the chi^2 (regardless of if it is fixed or not)  Lumped here imlies that antenna 0 in this list means BOTH channels 0 and 1 (H and V of crossed dipole antenna 0).
        included_antennas_channels = numpy.concatenate([[2*i,2*i+1] for i in included_antennas_lumped])
        include_baselines = [0,1,2,3,4,5]#[1,3,5] #Basically sets the starting condition of which baselines to include, then the lumped channels and antennas will cut out further from that.  The above options of excluding antennas will override this to exclude baselines, but if both antennas are included but the baseline is not then it will not be included.  Overwritten when antennas removed.

        #Limits 
        initial_step_x = 0.1#2.0 #m
        initial_step_y = 0.1#2.0 #m
        initial_step_z = 0.1#0.75 #m
        initial_step_cable_delay = 3 #ns
        cable_delay_guess_range = None#30 #ns
        antenna_position_guess_range_x = None#0.75 #Limit to how far from input phase locations to limit the parameter space to
        antenna_position_guess_range_y = None#0.75 #Limit to how far from input phase locations to limit the parameter space to
        antenna_position_guess_range_z = None#0.75 #Limit to how far from input phase locations to limit the parameter space to

        #Manually shifting input of antenna 0 around so that I can find a fit that has all of its baselines visible for valley sources. 
        manual_offset_ant0_x = 0
        manual_offset_ant0_y = 0
        manual_offset_ant0_z = 0

        manual_offset_ant1_x = 0
        manual_offset_ant1_y = 0
        manual_offset_ant1_z = 0

        manual_offset_ant2_x = 0
        manual_offset_ant2_y = 0
        manual_offset_ant2_z = 0

        manual_offset_ant3_x = 0#2
        manual_offset_ant3_y = 0#0
        manual_offset_ant3_z = 0#2


        fix_ant0_x = True
        fix_ant0_y = True
        fix_ant0_z = True
        fix_ant1_x = False
        fix_ant1_y = False
        fix_ant1_z = False
        fix_ant2_x = False
        fix_ant2_y = False
        fix_ant2_z = False
        fix_ant3_x = False
        fix_ant3_y = False
        fix_ant3_z = False
        fix_cable_delay0 = True
        fix_cable_delay1 = False
        fix_cable_delay2 = False
        fix_cable_delay3 = False

        origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
        if pol != 'both':
            am = AntennaMinimizer(  pol, 
                                    deploy_index,
                                    origin=origin,
                                    use_sites=use_sites, 
                                    random_offset_amount=random_offset_amount,
                                    included_antennas_lumped=included_antennas_lumped,
                                    include_baselines=include_baselines,
                                    initial_step_x=initial_step_x,
                                    initial_step_y=initial_step_y,
                                    initial_step_z=initial_step_z,
                                    initial_step_cable_delay=initial_step_cable_delay,
                                    cable_delay_guess_range=cable_delay_guess_range,
                                    antenna_position_guess_range_x=antenna_position_guess_range_x,
                                    antenna_position_guess_range_y=antenna_position_guess_range_y,
                                    antenna_position_guess_range_z=antenna_position_guess_range_z,
                                    manual_offset_ant0_x=manual_offset_ant0_x,
                                    manual_offset_ant0_y=manual_offset_ant0_y,
                                    manual_offset_ant0_z=manual_offset_ant0_z,
                                    manual_offset_ant1_x=manual_offset_ant1_x,
                                    manual_offset_ant1_y=manual_offset_ant1_y,
                                    manual_offset_ant1_z=manual_offset_ant1_z,
                                    manual_offset_ant2_x=manual_offset_ant2_x,
                                    manual_offset_ant2_y=manual_offset_ant2_y,
                                    manual_offset_ant2_z=manual_offset_ant2_z,
                                    manual_offset_ant3_x=manual_offset_ant3_x,
                                    manual_offset_ant3_y=manual_offset_ant3_y,
                                    manual_offset_ant3_z=manual_offset_ant3_z,
                                    fix_ant0_x=fix_ant0_x,
                                    fix_ant0_y=fix_ant0_y,
                                    fix_ant0_z=fix_ant0_z,
                                    fix_ant1_x=fix_ant1_x,
                                    fix_ant1_y=fix_ant1_y,
                                    fix_ant1_z=fix_ant1_z,
                                    fix_ant2_x=fix_ant2_x,
                                    fix_ant2_y=fix_ant2_y,
                                    fix_ant2_z=fix_ant2_z,
                                    fix_ant3_x=fix_ant3_x,
                                    fix_ant3_y=fix_ant3_y,
                                    fix_ant3_z=fix_ant3_z,
                                    fix_cable_delay0=fix_cable_delay0,
                                    fix_cable_delay1=fix_cable_delay1,
                                    fix_cable_delay2=fix_cable_delay2,
                                    fix_cable_delay3=fix_cable_delay3)
            try:
                m = Minuit(     am.rawChi2,\
                                ant0_x=am.initial_ant0_x,\
                                ant0_y=am.initial_ant0_y,\
                                ant0_z=am.initial_ant0_z,\
                                ant1_x=am.initial_ant1_x,\
                                ant1_y=am.initial_ant1_y,\
                                ant1_z=am.initial_ant1_z,\
                                ant2_x=am.initial_ant2_x,\
                                ant2_y=am.initial_ant2_y,\
                                ant2_z=am.initial_ant2_z,\
                                ant3_x=am.initial_ant3_x,\
                                ant3_y=am.initial_ant3_y,\
                                ant3_z=am.initial_ant3_z,\
                                cable_delay0=am.cable_delays[0],\
                                cable_delay1=am.cable_delays[1],\
                                cable_delay2=am.cable_delays[2],\
                                cable_delay3=am.cable_delays[3],\
                                error_ant0_x=am.errors['ant0_x'],\
                                error_ant0_y=am.errors['ant0_y'],\
                                error_ant0_z=am.errors['ant0_z'],\
                                error_ant1_x=am.errors['ant1_x'],\
                                error_ant1_y=am.errors['ant1_y'],\
                                error_ant1_z=am.errors['ant1_z'],\
                                error_ant2_x=am.errors['ant2_x'],\
                                error_ant2_y=am.errors['ant2_y'],\
                                error_ant2_z=am.errors['ant2_z'],\
                                error_ant3_x=am.errors['ant3_x'],\
                                error_ant3_y=am.errors['ant3_y'],\
                                error_ant3_z=am.errors['ant3_z'],\
                                error_cable_delay0=am.errors['cable_delay0'],\
                                error_cable_delay1=am.errors['cable_delay1'],\
                                error_cable_delay2=am.errors['cable_delay2'],\
                                error_cable_delay3=am.errors['cable_delay3'],\
                                errordef = am.errordef,\
                                limit_ant0_x=am.limits['ant0_x'],\
                                limit_ant0_y=am.limits['ant0_y'],\
                                limit_ant0_z=am.limits['ant0_z'],\
                                limit_ant1_x=am.limits['ant1_x'],\
                                limit_ant1_y=am.limits['ant1_y'],\
                                limit_ant1_z=am.limits['ant1_z'],\
                                limit_ant2_x=am.limits['ant2_x'],\
                                limit_ant2_y=am.limits['ant2_y'],\
                                limit_ant2_z=am.limits['ant2_z'],\
                                limit_ant3_x=am.limits['ant3_x'],\
                                limit_ant3_y=am.limits['ant3_y'],\
                                limit_ant3_z=am.limits['ant3_z'],\
                                limit_cable_delay0=am.limits['cable_delay0'],\
                                limit_cable_delay1=am.limits['cable_delay1'],\
                                limit_cable_delay2=am.limits['cable_delay2'],\
                                limit_cable_delay3=am.limits['cable_delay3'],\
                                fix_ant0_x=am.fixed['ant0_x'],\
                                fix_ant0_y=am.fixed['ant0_y'],\
                                fix_ant0_z=am.fixed['ant0_z'],\
                                fix_ant1_x=am.fixed['ant1_x'],\
                                fix_ant1_y=am.fixed['ant1_y'],\
                                fix_ant1_z=am.fixed['ant1_z'],\
                                fix_ant2_x=am.fixed['ant2_x'],\
                                fix_ant2_y=am.fixed['ant2_y'],\
                                fix_ant2_z=am.fixed['ant2_z'],\
                                fix_ant3_x=am.fixed['ant3_x'],\
                                fix_ant3_y=am.fixed['ant3_y'],\
                                fix_ant3_z=am.fixed['ant3_z'],\
                                fix_cable_delay0=am.fixed['cable_delay0'],\
                                fix_cable_delay1=am.fixed['cable_delay1'],\
                                fix_cable_delay2=am.fixed['cable_delay2'],\
                                fix_cable_delay3=am.fixed['cable_delay3'])
                result = m.migrad(resume=False)
            except Exception as e:
                print(e)
                print('Attempting setup of iminuit again, but assuming newer version of imnuit.')
                m = Minuit(     am.rawChi2,\
                                ant0_x=am.initial_ant0_x,\
                                ant0_y=am.initial_ant0_y,\
                                ant0_z=am.initial_ant0_z,\
                                ant1_x=am.initial_ant1_x,\
                                ant1_y=am.initial_ant1_y,\
                                ant1_z=am.initial_ant1_z,\
                                ant2_x=am.initial_ant2_x,\
                                ant2_y=am.initial_ant2_y,\
                                ant2_z=am.initial_ant2_z,\
                                ant3_x=am.initial_ant3_x,\
                                ant3_y=am.initial_ant3_y,\
                                ant3_z=am.initial_ant3_z,\
                                cable_delay0=am.cable_delays[0],\
                                cable_delay1=am.cable_delays[1],\
                                cable_delay2=am.cable_delays[2],\
                                cable_delay3=am.cable_delays[3])

                m.errors['ant0_x'] = am.errors['ant0_x']
                m.errors['ant0_y'] = am.errors['ant0_y']
                m.errors['ant0_z'] = am.errors['ant0_z']
                m.errors['ant1_x'] = am.errors['ant1_x']
                m.errors['ant1_y'] = am.errors['ant1_y']
                m.errors['ant1_z'] = am.errors['ant1_z']
                m.errors['ant2_x'] = am.errors['ant2_x']
                m.errors['ant2_y'] = am.errors['ant2_y']
                m.errors['ant2_z'] = am.errors['ant2_z']
                m.errors['ant3_x'] = am.errors['ant3_x']
                m.errors['ant3_y'] = am.errors['ant3_y']
                m.errors['ant3_z'] = am.errors['ant3_z']
                m.errors['cable_delay0'] = am.errors['cable_delay0']
                m.errors['cable_delay1'] = am.errors['cable_delay1']
                m.errors['cable_delay2'] = am.errors['cable_delay2']
                m.errors['cable_delay3'] = am.errors['cable_delay3']
                m.errordef = am.errordef
                m.limits['ant0_x'] = am.limits['ant0_x']
                m.limits['ant0_y'] = am.limits['ant0_y']
                m.limits['ant0_z'] = am.limits['ant0_z']
                m.limits['ant1_x'] = am.limits['ant1_x']
                m.limits['ant1_y'] = am.limits['ant1_y']
                m.limits['ant1_z'] = am.limits['ant1_z']
                m.limits['ant2_x'] = am.limits['ant2_x']
                m.limits['ant2_y'] = am.limits['ant2_y']
                m.limits['ant2_z'] = am.limits['ant2_z']
                m.limits['ant3_x'] = am.limits['ant3_x']
                m.limits['ant3_y'] = am.limits['ant3_y']
                m.limits['ant3_z'] = am.limits['ant3_z']
                m.limits['cable_delay0'] = am.limits['cable_delay0']
                m.limits['cable_delay1'] = am.limits['cable_delay1']
                m.limits['cable_delay2'] = am.limits['cable_delay2']
                m.limits['cable_delay3'] = am.limits['cable_delay3']
                m.fixed['ant0_x'] = am.fixed['ant0_x']
                m.fixed['ant0_y'] = am.fixed['ant0_y']
                m.fixed['ant0_z'] = am.fixed['ant0_z']
                m.fixed['ant1_x'] = am.fixed['ant1_x']
                m.fixed['ant1_y'] = am.fixed['ant1_y']
                m.fixed['ant1_z'] = am.fixed['ant1_z']
                m.fixed['ant2_x'] = am.fixed['ant2_x']
                m.fixed['ant2_y'] = am.fixed['ant2_y']
                m.fixed['ant2_z'] = am.fixed['ant2_z']
                m.fixed['ant3_x'] = am.fixed['ant3_x']
                m.fixed['ant3_y'] = am.fixed['ant3_y']
                m.fixed['ant3_z'] = am.fixed['ant3_z']
                m.fixed['cable_delay0'] = am.fixed['cable_delay0']
                m.fixed['cable_delay1'] = am.fixed['cable_delay1']
                m.fixed['cable_delay2'] = am.fixed['cable_delay2']
                m.fixed['cable_delay3'] = am.fixed['cable_delay3']

                result = m.migrad()


            print(result)
            m.hesse()
            if True:
                m.minos()
                pprint(m.get_fmin())
            else:
                try:
                    m.minos()
                    pprint(m.get_fmin())
                except:
                    print('MINOS FAILED, NOT VALID SOLUTION.')
            print('\a')
        else:
            am_hpol = AntennaMinimizer( 'hpol', 
                                        deploy_index,
                                        origin=origin,
                                        use_sites=use_sites['hpol'], 
                                        random_offset_amount=random_offset_amount,
                                        included_antennas_lumped=included_antennas_lumped,
                                        include_baselines=include_baselines,
                                        initial_step_x=initial_step_x,
                                        initial_step_y=initial_step_y,
                                        initial_step_z=initial_step_z,
                                        initial_step_cable_delay=initial_step_cable_delay,
                                        cable_delay_guess_range=cable_delay_guess_range,
                                        antenna_position_guess_range_x=antenna_position_guess_range_x,
                                        antenna_position_guess_range_y=antenna_position_guess_range_y,
                                        antenna_position_guess_range_z=antenna_position_guess_range_z,
                                        manual_offset_ant0_x=manual_offset_ant0_x,
                                        manual_offset_ant0_y=manual_offset_ant0_y,
                                        manual_offset_ant0_z=manual_offset_ant0_z,
                                        manual_offset_ant1_x=manual_offset_ant1_x,
                                        manual_offset_ant1_y=manual_offset_ant1_y,
                                        manual_offset_ant1_z=manual_offset_ant1_z,
                                        manual_offset_ant2_x=manual_offset_ant2_x,
                                        manual_offset_ant2_y=manual_offset_ant2_y,
                                        manual_offset_ant2_z=manual_offset_ant2_z,
                                        manual_offset_ant3_x=manual_offset_ant3_x,
                                        manual_offset_ant3_y=manual_offset_ant3_y,
                                        manual_offset_ant3_z=manual_offset_ant3_z,
                                        fix_ant0_x=fix_ant0_x,
                                        fix_ant0_y=fix_ant0_y,
                                        fix_ant0_z=fix_ant0_z,
                                        fix_ant1_x=fix_ant1_x,
                                        fix_ant1_y=fix_ant1_y,
                                        fix_ant1_z=fix_ant1_z,
                                        fix_ant2_x=fix_ant2_x,
                                        fix_ant2_y=fix_ant2_y,
                                        fix_ant2_z=fix_ant2_z,
                                        fix_ant3_x=fix_ant3_x,
                                        fix_ant3_y=fix_ant3_y,
                                        fix_ant3_z=fix_ant3_z,
                                        fix_cable_delay0=fix_cable_delay0,
                                        fix_cable_delay1=fix_cable_delay1,
                                        fix_cable_delay2=fix_cable_delay2,
                                        fix_cable_delay3=fix_cable_delay3)
            am_vpol = AntennaMinimizer( 'vpol', 
                                        deploy_index,
                                        origin=origin,
                                        use_sites=use_sites['vpol'], 
                                        random_offset_amount=random_offset_amount,
                                        included_antennas_lumped=included_antennas_lumped,
                                        include_baselines=include_baselines,
                                        initial_step_x=initial_step_x,
                                        initial_step_y=initial_step_y,
                                        initial_step_z=initial_step_z,
                                        initial_step_cable_delay=initial_step_cable_delay,
                                        cable_delay_guess_range=cable_delay_guess_range,
                                        antenna_position_guess_range_x=antenna_position_guess_range_x,
                                        antenna_position_guess_range_y=antenna_position_guess_range_y,
                                        antenna_position_guess_range_z=antenna_position_guess_range_z,
                                        manual_offset_ant0_x=manual_offset_ant0_x,
                                        manual_offset_ant0_y=manual_offset_ant0_y,
                                        manual_offset_ant0_z=manual_offset_ant0_z,
                                        manual_offset_ant1_x=manual_offset_ant1_x,
                                        manual_offset_ant1_y=manual_offset_ant1_y,
                                        manual_offset_ant1_z=manual_offset_ant1_z,
                                        manual_offset_ant2_x=manual_offset_ant2_x,
                                        manual_offset_ant2_y=manual_offset_ant2_y,
                                        manual_offset_ant2_z=manual_offset_ant2_z,
                                        manual_offset_ant3_x=manual_offset_ant3_x,
                                        manual_offset_ant3_y=manual_offset_ant3_y,
                                        manual_offset_ant3_z=manual_offset_ant3_z,
                                        fix_ant0_x=fix_ant0_x,
                                        fix_ant0_y=fix_ant0_y,
                                        fix_ant0_z=fix_ant0_z,
                                        fix_ant1_x=fix_ant1_x,
                                        fix_ant1_y=fix_ant1_y,
                                        fix_ant1_z=fix_ant1_z,
                                        fix_ant2_x=fix_ant2_x,
                                        fix_ant2_y=fix_ant2_y,
                                        fix_ant2_z=fix_ant2_z,
                                        fix_ant3_x=fix_ant3_x,
                                        fix_ant3_y=fix_ant3_y,
                                        fix_ant3_z=fix_ant3_z,
                                        fix_cable_delay0=fix_cable_delay0,
                                        fix_cable_delay1=fix_cable_delay1,
                                        fix_cable_delay2=fix_cable_delay2,
                                        fix_cable_delay3=fix_cable_delay3)

            cm = CombinedAntennaMinimizer(am_hpol, am_vpol)
            try:
                m = Minuit(     cm.rawChi2,\
                                ant0_x_hpol=cm.am_hpol.initial_ant0_x,\
                                ant0_y_hpol=cm.am_hpol.initial_ant0_y,\
                                ant0_z_hpol=cm.am_hpol.initial_ant0_z,\
                                ant1_x_hpol=cm.am_hpol.initial_ant1_x,\
                                ant1_y_hpol=cm.am_hpol.initial_ant1_y,\
                                ant1_z_hpol=cm.am_hpol.initial_ant1_z,\
                                ant2_x_hpol=cm.am_hpol.initial_ant2_x,\
                                ant2_y_hpol=cm.am_hpol.initial_ant2_y,\
                                ant2_z_hpol=cm.am_hpol.initial_ant2_z,\
                                ant3_x_hpol=cm.am_hpol.initial_ant3_x,\
                                ant3_y_hpol=cm.am_hpol.initial_ant3_y,\
                                ant3_z_hpol=cm.am_hpol.initial_ant3_z,\
                                cable_delay0_hpol=cm.am_hpol.cable_delays[0],\
                                cable_delay1_hpol=cm.am_hpol.cable_delays[1],\
                                cable_delay2_hpol=cm.am_hpol.cable_delays[2],\
                                cable_delay3_hpol=cm.am_hpol.cable_delays[3],\
                                cable_delay0_vpol=cm.am_vpol.cable_delays[0],\
                                cable_delay1_vpol=cm.am_vpol.cable_delays[1],\
                                cable_delay2_vpol=cm.am_vpol.cable_delays[2],\
                                cable_delay3_vpol=cm.am_vpol.cable_delays[3],\
                                error_ant0_x_hpol=cm.am_hpol.errors['ant0_x'],\
                                error_ant0_y_hpol=cm.am_hpol.errors['ant0_y'],\
                                error_ant0_z_hpol=cm.am_hpol.errors['ant0_z'],\
                                error_ant1_x_hpol=cm.am_hpol.errors['ant1_x'],\
                                error_ant1_y_hpol=cm.am_hpol.errors['ant1_y'],\
                                error_ant1_z_hpol=cm.am_hpol.errors['ant1_z'],\
                                error_ant2_x_hpol=cm.am_hpol.errors['ant2_x'],\
                                error_ant2_y_hpol=cm.am_hpol.errors['ant2_y'],\
                                error_ant2_z_hpol=cm.am_hpol.errors['ant2_z'],\
                                error_ant3_x_hpol=cm.am_hpol.errors['ant3_x'],\
                                error_ant3_y_hpol=cm.am_hpol.errors['ant3_y'],\
                                error_ant3_z_hpol=cm.am_hpol.errors['ant3_z'],\
                                error_cable_delay0_hpol=cm.am_hpol.errors['cable_delay0'],\
                                error_cable_delay1_hpol=cm.am_hpol.errors['cable_delay1'],\
                                error_cable_delay2_hpol=cm.am_hpol.errors['cable_delay2'],\
                                error_cable_delay3_hpol=cm.am_hpol.errors['cable_delay3'],\
                                error_cable_delay0_vpol=cm.am_vpol.errors['cable_delay0'],\
                                error_cable_delay1_vpol=cm.am_vpol.errors['cable_delay1'],\
                                error_cable_delay2_vpol=cm.am_vpol.errors['cable_delay2'],\
                                error_cable_delay3_vpol=cm.am_vpol.errors['cable_delay3'],\
                                errordef = cm.am_hpol.errordef,\
                                limit_ant0_x_hpol=cm.am_hpol.limits['ant0_x'],\
                                limit_ant0_y_hpol=cm.am_hpol.limits['ant0_y'],\
                                limit_ant0_z_hpol=cm.am_hpol.limits['ant0_z'],\
                                limit_ant1_x_hpol=cm.am_hpol.limits['ant1_x'],\
                                limit_ant1_y_hpol=cm.am_hpol.limits['ant1_y'],\
                                limit_ant1_z_hpol=cm.am_hpol.limits['ant1_z'],\
                                limit_ant2_x_hpol=cm.am_hpol.limits['ant2_x'],\
                                limit_ant2_y_hpol=cm.am_hpol.limits['ant2_y'],\
                                limit_ant2_z_hpol=cm.am_hpol.limits['ant2_z'],\
                                limit_ant3_x_hpol=cm.am_hpol.limits['ant3_x'],\
                                limit_ant3_y_hpol=cm.am_hpol.limits['ant3_y'],\
                                limit_ant3_z_hpol=cm.am_hpol.limits['ant3_z'],\
                                limit_cable_delay0_hpol=cm.am_hpol.limits['cable_delay0'],\
                                limit_cable_delay1_hpol=cm.am_hpol.limits['cable_delay1'],\
                                limit_cable_delay2_hpol=cm.am_hpol.limits['cable_delay2'],\
                                limit_cable_delay3_hpol=cm.am_hpol.limits['cable_delay3'],\
                                limit_cable_delay0_vpol=cm.am_vpol.limits['cable_delay0'],\
                                limit_cable_delay1_vpol=cm.am_vpol.limits['cable_delay1'],\
                                limit_cable_delay2_vpol=cm.am_vpol.limits['cable_delay2'],\
                                limit_cable_delay3_vpol=cm.am_vpol.limits['cable_delay3'],\
                                fix_ant0_x_hpol=cm.am_hpol.fixed['ant0_x'],\
                                fix_ant0_y_hpol=cm.am_hpol.fixed['ant0_y'],\
                                fix_ant0_z_hpol=cm.am_hpol.fixed['ant0_z'],\
                                fix_ant1_x_hpol=cm.am_hpol.fixed['ant1_x'],\
                                fix_ant1_y_hpol=cm.am_hpol.fixed['ant1_y'],\
                                fix_ant1_z_hpol=cm.am_hpol.fixed['ant1_z'],\
                                fix_ant2_x_hpol=cm.am_hpol.fixed['ant2_x'],\
                                fix_ant2_y_hpol=cm.am_hpol.fixed['ant2_y'],\
                                fix_ant2_z_hpol=cm.am_hpol.fixed['ant2_z'],\
                                fix_ant3_x_hpol=cm.am_hpol.fixed['ant3_x'],\
                                fix_ant3_y_hpol=cm.am_hpol.fixed['ant3_y'],\
                                fix_ant3_z_hpol=cm.am_hpol.fixed['ant3_z'],\
                                fix_cable_delay0_hpol=cm.am_hpol.fixed['cable_delay0'],\
                                fix_cable_delay1_hpol=cm.am_hpol.fixed['cable_delay1'],\
                                fix_cable_delay2_hpol=cm.am_hpol.fixed['cable_delay2'],\
                                fix_cable_delay3_hpol=cm.am_hpol.fixed['cable_delay3'],\
                                fix_cable_delay0_vpol=cm.am_vpol.fixed['cable_delay0'],\
                                fix_cable_delay1_vpol=cm.am_vpol.fixed['cable_delay1'],\
                                fix_cable_delay2_vpol=cm.am_vpol.fixed['cable_delay2'],\
                                fix_cable_delay3_vpol=cm.am_vpol.fixed['cable_delay3'])
                result = m.migrad(resume=False)
            except Exception as e:
                print(e)
                print('Attempting setup of iminuit again, but assuming newer version of imnuit.')
                m = Minuit(     am.rawChi2,\
                                ant0_x_hpol=cm.am_hpol.initial_ant0_x,\
                                ant0_y_hpol=cm.am_hpol.initial_ant0_y,\
                                ant0_z_hpol=cm.am_hpol.initial_ant0_z,\
                                ant1_x_hpol=cm.am_hpol.initial_ant1_x,\
                                ant1_y_hpol=cm.am_hpol.initial_ant1_y,\
                                ant1_z_hpol=cm.am_hpol.initial_ant1_z,\
                                ant2_x_hpol=cm.am_hpol.initial_ant2_x,\
                                ant2_y_hpol=cm.am_hpol.initial_ant2_y,\
                                ant2_z_hpol=cm.am_hpol.initial_ant2_z,\
                                ant3_x_hpol=cm.am_hpol.initial_ant3_x,\
                                ant3_y_hpol=cm.am_hpol.initial_ant3_y,\
                                ant3_z_hpol=cm.am_hpol.initial_ant3_z,\
                                cable_delay0_hpol=cm.am_hpol.cable_delays[0],\
                                cable_delay1_hpol=cm.am_hpol.cable_delays[1],\
                                cable_delay2_hpol=cm.am_hpol.cable_delays[2],\
                                cable_delay3_hpol=cm.am_hpol.cable_delays[3],\
                                cable_delay0_vpol=cm.am_vpol.cable_delays[0],\
                                cable_delay1_vpol=cm.am_vpol.cable_delays[1],\
                                cable_delay2_vpol=cm.am_vpol.cable_delays[2],\
                                cable_delay3_vpol=cm.am_vpol.cable_delays[3])

                m.errors['ant0_x_hpol'] = cm.am_hpol.errors['ant0_x']
                m.errors['ant0_y_hpol'] = cm.am_hpol.errors['ant0_y']
                m.errors['ant0_z_hpol'] = cm.am_hpol.errors['ant0_z']
                m.errors['ant1_x_hpol'] = cm.am_hpol.errors['ant1_x']
                m.errors['ant1_y_hpol'] = cm.am_hpol.errors['ant1_y']
                m.errors['ant1_z_hpol'] = cm.am_hpol.errors['ant1_z']
                m.errors['ant2_x_hpol'] = cm.am_hpol.errors['ant2_x']
                m.errors['ant2_y_hpol'] = cm.am_hpol.errors['ant2_y']
                m.errors['ant2_z_hpol'] = cm.am_hpol.errors['ant2_z']
                m.errors['ant3_x_hpol'] = cm.am_hpol.errors['ant3_x']
                m.errors['ant3_y_hpol'] = cm.am_hpol.errors['ant3_y']
                m.errors['ant3_z_hpol'] = cm.am_hpol.errors['ant3_z']
                m.errors['cable_delay0_hpol'] = cm.am_hpol.errors['cable_delay0']
                m.errors['cable_delay1_hpol'] = cm.am_hpol.errors['cable_delay1']
                m.errors['cable_delay2_hpol'] = cm.am_hpol.errors['cable_delay2']
                m.errors['cable_delay3_hpol'] = cm.am_hpol.errors['cable_delay3']
                m.errors['cable_delay0_vpol'] = cm.am_vpol.errors['cable_delay0']
                m.errors['cable_delay1_vpol'] = cm.am_vpol.errors['cable_delay1']
                m.errors['cable_delay2_vpol'] = cm.am_vpol.errors['cable_delay2']
                m.errors['cable_delay3_vpol'] = cm.am_vpol.errors['cable_delay3']
                m.errordef = cm.am_hpol.errordef
                m.limits['ant0_x_hpol'] = cm.am_hpol.limits['ant0_x']
                m.limits['ant0_y_hpol'] = cm.am_hpol.limits['ant0_y']
                m.limits['ant0_z_hpol'] = cm.am_hpol.limits['ant0_z']
                m.limits['ant1_x_hpol'] = cm.am_hpol.limits['ant1_x']
                m.limits['ant1_y_hpol'] = cm.am_hpol.limits['ant1_y']
                m.limits['ant1_z_hpol'] = cm.am_hpol.limits['ant1_z']
                m.limits['ant2_x_hpol'] = cm.am_hpol.limits['ant2_x']
                m.limits['ant2_y_hpol'] = cm.am_hpol.limits['ant2_y']
                m.limits['ant2_z_hpol'] = cm.am_hpol.limits['ant2_z']
                m.limits['ant3_x_hpol'] = cm.am_hpol.limits['ant3_x']
                m.limits['ant3_y_hpol'] = cm.am_hpol.limits['ant3_y']
                m.limits['ant3_z_hpol'] = cm.am_hpol.limits['ant3_z']
                m.limits['cable_delay0_hpol'] = cm.am_hpol.limits['cable_delay0']
                m.limits['cable_delay1_hpol'] = cm.am_hpol.limits['cable_delay1']
                m.limits['cable_delay2_hpol'] = cm.am_hpol.limits['cable_delay2']
                m.limits['cable_delay3_hpol'] = cm.am_hpol.limits['cable_delay3']
                m.limits['cable_delay0_vpol'] = cm.am_vpol.limits['cable_delay0']
                m.limits['cable_delay1_vpol'] = cm.am_vpol.limits['cable_delay1']
                m.limits['cable_delay2_vpol'] = cm.am_vpol.limits['cable_delay2']
                m.limits['cable_delay3_vpol'] = cm.am_vpol.limits['cable_delay3']
                m.fixed['ant0_x_hpol'] = cm.am_hpol.fixed['ant0_x']
                m.fixed['ant0_y_hpol'] = cm.am_hpol.fixed['ant0_y']
                m.fixed['ant0_z_hpol'] = cm.am_hpol.fixed['ant0_z']
                m.fixed['ant1_x_hpol'] = cm.am_hpol.fixed['ant1_x']
                m.fixed['ant1_y_hpol'] = cm.am_hpol.fixed['ant1_y']
                m.fixed['ant1_z_hpol'] = cm.am_hpol.fixed['ant1_z']
                m.fixed['ant2_x_hpol'] = cm.am_hpol.fixed['ant2_x']
                m.fixed['ant2_y_hpol'] = cm.am_hpol.fixed['ant2_y']
                m.fixed['ant2_z_hpol'] = cm.am_hpol.fixed['ant2_z']
                m.fixed['ant3_x_hpol'] = cm.am_hpol.fixed['ant3_x']
                m.fixed['ant3_y_hpol'] = cm.am_hpol.fixed['ant3_y']
                m.fixed['ant3_z_hpol'] = cm.am_hpol.fixed['ant3_z']
                m.fixed['cable_delay0_hpol'] = cm.am_hpol.fixed['cable_delay0']
                m.fixed['cable_delay1_hpol'] = cm.am_hpol.fixed['cable_delay1']
                m.fixed['cable_delay2_hpol'] = cm.am_hpol.fixed['cable_delay2']
                m.fixed['cable_delay3_hpol'] = cm.am_hpol.fixed['cable_delay3']
                m.fixed['cable_delay0_vpol'] = cm.am_vpol.fixed['cable_delay0']
                m.fixed['cable_delay1_vpol'] = cm.am_vpol.fixed['cable_delay1']
                m.fixed['cable_delay2_vpol'] = cm.am_vpol.fixed['cable_delay2']
                m.fixed['cable_delay3_vpol'] = cm.am_vpol.fixed['cable_delay3']

                result = m.migrad()


            print(result)
            m.hesse()
            if True:
                m.minos()
                pprint(m.get_fmin())
            else:
                try:
                    m.minos()
                    pprint(m.get_fmin())
                except:
                    print('MINOS FAILED, NOT VALID SOLUTION.')
            print('\a')

        # # ##########
        # # Plot Setup
        # # ##########
        if pol != 'both':
            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Initial Positions')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            if 0 in am.included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant0_x, am.initial_ant0_y, am.initial_ant0_z,c='r',alpha=0.5,label='Initial Ant0')
            if 1 in am.included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant1_x, am.initial_ant1_y, am.initial_ant1_z,c='g',alpha=0.5,label='Initial Ant1')
            if 2 in am.included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant2_x, am.initial_ant2_y, am.initial_ant2_z,c='b',alpha=0.5,label='Initial Ant2')
            if 3 in am.included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant3_x, am.initial_ant3_y, am.initial_ant3_z,c='m',alpha=0.5,label='Initial Ant3')

            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            
            chi2_ax.dist = 10
            plt.legend()
            
            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Both')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            if 0 in included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant0_x, am.initial_ant0_y, am.initial_ant0_z,c='r',alpha=0.5,label='Initial Ant0')
            if 1 in included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant1_x, am.initial_ant1_y, am.initial_ant1_z,c='g',alpha=0.5,label='Initial Ant1')
            if 2 in included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant2_x, am.initial_ant2_y, am.initial_ant2_z,c='b',alpha=0.5,label='Initial Ant2')
            if 3 in included_antennas_lumped:
                chi2_ax.scatter(am.initial_ant3_x, am.initial_ant3_y, am.initial_ant3_z,c='m',alpha=0.5,label='Initial Ant3')

            for antenna in range(4):
                fig = plt.figure()
                fig.canvas.set_window_title('Ant %i chi^2'%antenna)
                for index, key in enumerate(['ant%i_x'%antenna,'ant%i_y'%antenna,'ant%i_z'%antenna]):
                    plt.subplot(1,3,index + 1)
                    m.draw_profile(key)

            if cable_delay_guess_range is not None:
                fig = plt.figure()
                fig.canvas.set_window_title('Cable Delays')
                for antenna in range(4):
                    plt.subplot(2,2,antenna + 1)
                    m.draw_profile('cable_delay%i'%antenna)

            #12 variables
            ant0_phase_x = m.values['ant0_x']
            ant0_phase_y = m.values['ant0_y']
            ant0_phase_z = m.values['ant0_z']
            ant0_cable_delay = m.values['cable_delay0']

            ant1_phase_x = m.values['ant1_x']
            ant1_phase_y = m.values['ant1_y']
            ant1_phase_z = m.values['ant1_z']
            ant1_cable_delay = m.values['cable_delay1']

            ant2_phase_x = m.values['ant2_x']
            ant2_phase_y = m.values['ant2_y']
            ant2_phase_z = m.values['ant2_z']
            ant2_cable_delay = m.values['cable_delay2']

            ant3_phase_x = m.values['ant3_x']
            ant3_phase_y = m.values['ant3_y']
            ant3_phase_z = m.values['ant3_z']
            ant3_cable_delay = m.values['cable_delay3']

            ant0_ENU = numpy.array([ant0_phase_x, ant0_phase_y, ant0_phase_z])
            ant1_ENU = numpy.array([ant1_phase_x, ant1_phase_y, ant1_phase_z])
            ant2_ENU = numpy.array([ant2_phase_x, ant2_phase_y, ant2_phase_z])
            ant3_ENU = numpy.array([ant3_phase_x, ant3_phase_y, ant3_phase_z])
            resulting_cable_delays = numpy.array([ant0_cable_delay,ant1_cable_delay,ant2_cable_delay,ant3_cable_delay])

            output_antennas_phase = {0:ant0_ENU, 1:ant1_ENU, 2:ant2_ENU, 3:ant3_ENU}

            chi2_ax.plot([am.initial_ant0_x , ant0_phase_x], [am.initial_ant0_y , ant0_phase_y], [am.initial_ant0_z , ant0_phase_z],c='r',alpha=0.5,linestyle='--')
            chi2_ax.plot([am.initial_ant1_x , ant1_phase_x], [am.initial_ant1_y , ant1_phase_y], [am.initial_ant1_z , ant1_phase_z],c='g',alpha=0.5,linestyle='--')
            chi2_ax.plot([am.initial_ant2_x , ant2_phase_x], [am.initial_ant2_y , ant2_phase_y], [am.initial_ant2_z , ant2_phase_z],c='b',alpha=0.5,linestyle='--')
            chi2_ax.plot([am.initial_ant3_x , ant3_phase_x], [am.initial_ant3_y , ant3_phase_y], [am.initial_ant3_z , ant3_phase_z],c='m',alpha=0.5,linestyle='--')

            chi2_ax.scatter(ant0_phase_x, ant0_phase_y, ant0_phase_z,marker='*',c='r',alpha=0.5,label='Final Ant0')
            chi2_ax.scatter(ant1_phase_x, ant1_phase_y, ant1_phase_z,marker='*',c='g',alpha=0.5,label='Final Ant1')
            chi2_ax.scatter(ant2_phase_x, ant2_phase_y, ant2_phase_z,marker='*',c='b',alpha=0.5,label='Final Ant2')
            chi2_ax.scatter(ant3_phase_x, ant3_phase_y, ant3_phase_z,marker='*',c='m',alpha=0.5,label='Final Ant3')
            
            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            chi2_ax.dist = 10
            plt.legend()



            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Final Positions')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            if 0 in included_antennas_lumped:
                chi2_ax.scatter(ant0_phase_x, ant0_phase_y, ant0_phase_z,marker='*',c='r',alpha=0.5,label='Final Ant0')
            if 1 in included_antennas_lumped:
                chi2_ax.scatter(ant1_phase_x, ant1_phase_y, ant1_phase_z,marker='*',c='g',alpha=0.5,label='Final Ant1')
            if 2 in included_antennas_lumped:
                chi2_ax.scatter(ant2_phase_x, ant2_phase_y, ant2_phase_z,marker='*',c='b',alpha=0.5,label='Final Ant2')
            if 3 in included_antennas_lumped:
                chi2_ax.scatter(ant3_phase_x, ant3_phase_y, ant3_phase_z,marker='*',c='m',alpha=0.5,label='Final Ant3')

            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            chi2_ax.dist = 10
            plt.legend()

            #Plot Pulser Events
            pulser_info = PulserInfo()
            for key in am.use_sites:
                #Calculate old and new geometries
                #Distance needed when calling correlator, as it uses that distance.
                original_pulser_ENU = numpy.array([am.pulser_locations_ENU[key][0] , am.pulser_locations_ENU[key][1] , am.pulser_locations_ENU[key][2]])
                original_distance_m = numpy.linalg.norm(original_pulser_ENU)
                original_zenith_deg = numpy.rad2deg(numpy.arccos(original_pulser_ENU[2]/original_distance_m))
                original_elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(original_pulser_ENU[2]/original_distance_m))
                original_azimuth_deg = numpy.rad2deg(numpy.arctan2(original_pulser_ENU[1],original_pulser_ENU[0]))

                pulser_ENU_new = numpy.array([am.pulser_locations_ENU[key][0] - ant0_ENU[0] , am.pulser_locations_ENU[key][1] - ant0_ENU[1] , am.pulser_locations_ENU[key][2] - ant0_ENU[2]])
                distance_m = numpy.linalg.norm(pulser_ENU_new)
                zenith_deg = numpy.rad2deg(numpy.arccos(pulser_ENU_new[2]/distance_m))
                elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(pulser_ENU_new[2]/distance_m))
                azimuth_deg = numpy.rad2deg(numpy.arctan2(pulser_ENU_new[1],pulser_ENU_new[0]))

                map_resolution = 0.25 #degrees
                range_phi_deg = (-90, 90)
                range_theta_deg = (80,120)
                n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                

                known_pulser_ids = info.load2021PulserEventids()[key][pol]
                known_pulser_ids = known_pulser_ids[numpy.isin(known_pulser_ids['attenuation_dB'], attenuations_dict[pol][key])]
                reference_event  = pulser_info.getPulserReferenceEvent(key, pol)
                if True:
                    event_info = reference_event
                else:
                    event_info = numpy.random.choice(known_pulser_ids)

                reader = Reader(datapath,int(event_info['run']))
                
                cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)


                if pol == 'hpol':
                    cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,am.initial_ant0_ENU,am.initial_ant1_ENU,am.initial_ant2_ENU,am.initial_ant3_ENU,cor.A0_vpol,cor.A1_vpol,cor.A2_vpol,cor.A3_vpol,verbose=False)
                elif pol == 'vpol':
                    cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,cor.A0_hpol,cor.A1_hpol,cor.A2_hpol,cor.A3_hpol,am.initial_ant0_ENU,am.initial_ant1_ENU,am.initial_ant2_ENU,am.initial_ant3_ENU,verbose=False)


                adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                if pol == 'hpol':
                    adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,adjusted_cor.A0_vpol,adjusted_cor.A1_vpol,adjusted_cor.A2_vpol,adjusted_cor.A3_vpol,verbose=False)
                    adjusted_cor.overwriteCableDelays(m.values['cable_delay0'], adjusted_cor.cable_delays[1], m.values['cable_delay1'], adjusted_cor.cable_delays[3], m.values['cable_delay2'], adjusted_cor.cable_delays[5], m.values['cable_delay3'], adjusted_cor.cable_delays[7])
                else:
                    adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,adjusted_cor.A0_hpol,adjusted_cor.A1_hpol,adjusted_cor.A2_hpol,adjusted_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False)
                    adjusted_cor.overwriteCableDelays(adjusted_cor.cable_delays[0], m.values['cable_delay0'], adjusted_cor.cable_delays[2], m.values['cable_delay1'], adjusted_cor.cable_delays[4], m.values['cable_delay2'], adjusted_cor.cable_delays[6], m.values['cable_delay3'])                    

                if plot_expected_direction == False:
                    zenith_deg = None
                    azimuth_deg = None

                if plot_time_delays_on_maps:
                    if False:
                        #Good for troubleshooting if a cycle slipped.
                        cycle_slip_estimate_ns = 15
                        n_cycles = 1
                        td_dict = {pol:{'[0, 1]' :  measured_time_delays[key][pol]['delays_ns'][0] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[0, 2]' : measured_time_delays[key][pol]['delays_ns'][1] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[0, 3]' : measured_time_delays[key][pol]['delays_ns'][2] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[1, 2]' : measured_time_delays[key][pol]['delays_ns'][3] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[1, 3]' : measured_time_delays[key][pol]['delays_ns'][4] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[2, 3]' : measured_time_delays[key][pol]['delays_ns'][5] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns}}
                    else:
                        td_dict = {pol:{'[0, 1]' :  [measured_time_delays[key][pol]['delays_ns'][0]], '[0, 2]' : [measured_time_delays[key][pol]['delays_ns'][1]], '[0, 3]' : [measured_time_delays[key][pol]['delays_ns'][2]], '[1, 2]' : [measured_time_delays[key][pol]['delays_ns'][3]], '[1, 3]' : [measured_time_delays[key][pol]['delays_ns'][4]], '[2, 3]' : [measured_time_delays[key][pol]['delays_ns'][5]]}}
                else:
                    td_dict = {}


                #mean_corr_values, fig, ax = cor.map(int(event_info['eventid']), pol, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                adjusted_mean_corr_values, adjusted_fig, adjusted_ax = adjusted_cor.map(int(event_info['eventid']), pol, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                adjusted_fig.set_size_inches(16, 9)
                plt.sca(adjusted_ax)
                plt.tight_layout()
                adjusted_fig.savefig('./%s.png'%key,dpi=90)

                if plot_histograms:
                    map_resolution = 0.1 #degrees
                    range_phi_deg=(azimuth_deg - 10, azimuth_deg + 10)
                    range_theta_deg=(zenith_deg - 10,zenith_deg + 10)
                    n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                    n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                                    
                    cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                    adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    if pol == 'hpol':
                        adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,adjusted_cor.A0_vpol,adjusted_cor.A1_vpol,adjusted_cor.A2_vpol,adjusted_cor.A3_vpol,verbose=False)
                        adjusted_cor.overwriteCableDelays(m.values['cable_delay0'], adjusted_cor.cable_delays[1], m.values['cable_delay1'], adjusted_cor.cable_delays[3], m.values['cable_delay2'], adjusted_cor.cable_delays[5], m.values['cable_delay3'], adjusted_cor.cable_delays[7])
                    else:
                        adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,adjusted_cor.A0_hpol,adjusted_cor.A1_hpol,adjusted_cor.A2_hpol,adjusted_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False)
                        adjusted_cor.overwriteCableDelays(adjusted_cor.cable_delays[0], m.values['cable_delay0'], adjusted_cor.cable_delays[2], m.values['cable_delay1'], adjusted_cor.cable_delays[4], m.values['cable_delay2'], adjusted_cor.cable_delays[6], m.values['cable_delay3'])                    

                    
                    run_cut = known_pulser_ids['run'] == reader.run #Make sure all eventids in same run
                    hist = adjusted_cor.histMapPeak(numpy.sort(numpy.random.choice(known_pulser_ids[run_cut],min(limit_events,len(known_pulser_ids[run_cut]))))['eventid'], pol, plot_map=True, hilbert=False, max_method=0, use_weight=False, mollweide=False, center_dir='E', radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90],circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='Hist ' + key, include_baselines=include_baselines,iterate_sub_baselines=iterate_sub_baselines)

            # Finalized Output 
            print('Estimated degrees of freedom: %i'%sum([not v for k, v in m.fixed.items()]))
            print('Estimated input measured values: %i'%(len(am.include_baselines)*len(am.use_sites) + len(am.include_baselines)*len(am.use_sites)))


            print('\n')
            print('STARTING CONDITION INPUT VALUES HERE')
            print('\n')
            print('')
            print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(pol, am.initial_ant0_x,am.initial_ant0_y,am.initial_ant0_z ,  am.initial_ant1_x,am.initial_ant1_y,am.initial_ant1_z,  am.initial_ant2_x,am.initial_ant2_y,am.initial_ant2_z,  am.initial_ant3_x,am.initial_ant3_y,am.initial_ant3_z))
            print('')
            print('cable_delays_%s = numpy.array([%f,%f,%f,%f])'%(pol,am.cable_delays[0],am.cable_delays[1],am.cable_delays[2],am.cable_delays[3]))



            print('\n')
            print(result)
            print('\n')
            print('Copy-Paste Prints:\n------------')
            print('')
            print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(pol, m.values['ant0_x'],m.values['ant0_y'],m.values['ant0_z'] ,  m.values['ant1_x'],m.values['ant1_y'],m.values['ant1_z'],  m.values['ant2_x'],m.values['ant2_y'],m.values['ant2_z'],  m.values['ant3_x'],m.values['ant3_y'],m.values['ant3_z']))
            print('antennas_phase_%s_hesse = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(pol, m.errors['ant0_x'],m.errors['ant0_y'],m.errors['ant0_z'] ,  m.errors['ant1_x'],m.errors['ant1_y'],m.errors['ant1_z'],  m.errors['ant2_x'],m.errors['ant2_y'],m.errors['ant2_z'],  m.errors['ant3_x'],m.errors['ant3_y'],m.errors['ant3_z']))
            print('')
            print('cable_delays_%s = numpy.array([%f,%f,%f,%f])'%(pol,m.values['cable_delay0'],m.values['cable_delay1'],m.values['cable_delay2'],m.values['cable_delay3']))
            print('cable_delays_%s_hesse = numpy.array([%f,%f,%f,%f])'%(pol,m.errors['cable_delay0'],m.errors['cable_delay1'],m.errors['cable_delay2'],m.errors['cable_delay3']))

            print('Code completed.')
            print('\a')

            if True:
                #This code is intended to save the output configuration produced by this script. 
                initial_deploy_index = str(info.returnDefaultDeploy())
                initial_origin, initial_antennas_physical, initial_antennas_phase_hpol, initial_antennas_phase_vpol, initial_cable_delays, initial_description = bcr.configReader(initial_deploy_index,return_description=True)

                output_origin = initial_origin
                output_antennas_physical = initial_antennas_physical
                if pol == 'hpol':
                    output_antennas_phase_hpol = output_antennas_phase
                    output_antennas_phase_vpol = initial_antennas_phase_vpol
                else:
                    output_antennas_phase_hpol = initial_antennas_phase_hpol
                    output_antennas_phase_vpol = output_antennas_phase

                output_cable_delays = initial_cable_delays
                output_cable_delays[pol] = resulting_cable_delays
                output_description = 'Automatically generated description for a calibration starting from deploy_index: %s.  This config has updated %s values based on a calibration that was performed.  Initial description: %s'%(initial_deploy_index, pol, initial_description)

                if len(os.path.split(initial_deploy_index)) == 2:
                    json_path = initial_deploy_index
                else:
                    json_path = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'config','automatically_generated_config_0.json')
                
                with open('./antenna_position_minimization.py', "r") as this_file:
                    #read whole file to a string
                    script_string = this_file.read()

                bcr.configWriter(json_path, output_origin, output_antennas_physical, output_antennas_phase_hpol, output_antennas_phase_vpol, output_cable_delays, description=output_description,update_latlonel=True,force_write=True, additional_text=script_string) #does not overwrite.

        else:
            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Initial Positions')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            if 0 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant0_x, cm.am_hpol.initial_ant0_y, cm.am_hpol.initial_ant0_z,c='r',alpha=0.5, marker='$H$',label='Hpol Initial Ant0')
            if 1 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant1_x, cm.am_hpol.initial_ant1_y, cm.am_hpol.initial_ant1_z,c='g',alpha=0.5, marker='$H$',label='Hpol Initial Ant1')
            if 2 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant2_x, cm.am_hpol.initial_ant2_y, cm.am_hpol.initial_ant2_z,c='b',alpha=0.5, marker='$H$',label='Hpol Initial Ant2')
            if 3 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant3_x, cm.am_hpol.initial_ant3_y, cm.am_hpol.initial_ant3_z,c='m',alpha=0.5, marker='$H$',label='Hpol Initial Ant3')

            if 0 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant0_x, cm.am_vpol.initial_ant0_y, cm.am_vpol.initial_ant0_z,c='r',alpha=0.5, marker='$V$',label='Vpol Initial Ant0')
            if 1 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant1_x, cm.am_vpol.initial_ant1_y, cm.am_vpol.initial_ant1_z,c='g',alpha=0.5, marker='$V$',label='Vpol Initial Ant1')
            if 2 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant2_x, cm.am_vpol.initial_ant2_y, cm.am_vpol.initial_ant2_z,c='b',alpha=0.5, marker='$V$',label='Vpol Initial Ant2')
            if 3 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant3_x, cm.am_vpol.initial_ant3_y, cm.am_vpol.initial_ant3_z,c='m',alpha=0.5, marker='$V$',label='Vpol Initial Ant3')

            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            
            chi2_ax.dist = 10
            plt.legend()

            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Both')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            if 0 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant0_x, cm.am_hpol.initial_ant0_y, cm.am_hpol.initial_ant0_z,c='r',alpha=0.5, marker='$H_i$',label='Hpol Initial Ant0')
            if 1 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant1_x, cm.am_hpol.initial_ant1_y, cm.am_hpol.initial_ant1_z,c='g',alpha=0.5, marker='$H_i$',label='Hpol Initial Ant1')
            if 2 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant2_x, cm.am_hpol.initial_ant2_y, cm.am_hpol.initial_ant2_z,c='b',alpha=0.5, marker='$H_i$',label='Hpol Initial Ant2')
            if 3 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_hpol.initial_ant3_x, cm.am_hpol.initial_ant3_y, cm.am_hpol.initial_ant3_z,c='m',alpha=0.5, marker='$H_i$',label='Hpol Initial Ant3')

            if 0 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant0_x, cm.am_vpol.initial_ant0_y, cm.am_vpol.initial_ant0_z,c='r',alpha=0.5, marker='$V_i$',label='Vpol Initial Ant0')
            if 1 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant1_x, cm.am_vpol.initial_ant1_y, cm.am_vpol.initial_ant1_z,c='g',alpha=0.5, marker='$V_i$',label='Vpol Initial Ant1')
            if 2 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant2_x, cm.am_vpol.initial_ant2_y, cm.am_vpol.initial_ant2_z,c='b',alpha=0.5, marker='$V_i$',label='Vpol Initial Ant2')
            if 3 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(cm.am_vpol.initial_ant3_x, cm.am_vpol.initial_ant3_y, cm.am_vpol.initial_ant3_z,c='m',alpha=0.5, marker='$V_i$',label='Vpol Initial Ant3')

            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            
            chi2_ax.dist = 10
            plt.legend()

            for antenna in range(4):
                fig = plt.figure()
                fig.canvas.set_window_title('Ant %i chi^2'%antenna)
                for index, key in enumerate(['ant%i_x_hpol'%antenna,'ant%i_y_hpol'%antenna,'ant%i_z_hpol'%antenna]):
                    plt.subplot(1,3,index + 1)
                    m.draw_profile(key)

            if cable_delay_guess_range is not None:
                fig = plt.figure()
                fig.canvas.set_window_title('HPol Cable Delays')
                for antenna in range(4):
                    plt.subplot(2,2,antenna + 1)
                    m.draw_profile('cable_delay%i_hpol'%antenna)
                fig = plt.figure()
                fig.canvas.set_window_title('VPol Cable Delays')
                for antenna in range(4):
                    plt.subplot(2,2,antenna + 1)
                    m.draw_profile('cable_delay%i_vpol'%antenna)

            #12 variables
            ant0_phase_x_hpol = m.values['ant0_x_hpol']
            ant0_phase_x_vpol = cm.am_vpol.initial_ant0_x + ( ant0_phase_x_hpol - cm.am_hpol.initial_ant0_x )
            ant0_phase_y_hpol = m.values['ant0_y_hpol']
            ant0_phase_y_vpol = cm.am_vpol.initial_ant0_y + ( ant0_phase_y_hpol - cm.am_hpol.initial_ant0_y )
            ant0_phase_z_hpol = m.values['ant0_z_hpol']
            ant0_phase_z_vpol = cm.am_vpol.initial_ant0_z + ( ant0_phase_z_hpol - cm.am_hpol.initial_ant0_z )
            ant0_cable_delay_hpol = m.values['cable_delay0_hpol']
            ant0_cable_delay_vpol = m.values['cable_delay0_vpol']

            ant1_phase_x_hpol = m.values['ant1_x_hpol']
            ant1_phase_x_vpol = cm.am_vpol.initial_ant1_x + ( ant1_phase_x_hpol - cm.am_hpol.initial_ant1_x )
            ant1_phase_y_hpol = m.values['ant1_y_hpol']
            ant1_phase_y_vpol = cm.am_vpol.initial_ant1_y + ( ant1_phase_y_hpol - cm.am_hpol.initial_ant1_y )
            ant1_phase_z_hpol = m.values['ant1_z_hpol']
            ant1_phase_z_vpol = cm.am_vpol.initial_ant1_z + ( ant1_phase_z_hpol - cm.am_hpol.initial_ant1_z )
            ant1_cable_delay_hpol = m.values['cable_delay1_hpol']
            ant1_cable_delay_vpol = m.values['cable_delay1_vpol']

            ant2_phase_x_hpol = m.values['ant2_x_hpol']
            ant2_phase_x_vpol = cm.am_vpol.initial_ant2_x + ( ant2_phase_x_hpol - cm.am_hpol.initial_ant2_x )
            ant2_phase_y_hpol = m.values['ant2_y_hpol']
            ant2_phase_y_vpol = cm.am_vpol.initial_ant2_y + ( ant2_phase_y_hpol - cm.am_hpol.initial_ant2_y )
            ant2_phase_z_hpol = m.values['ant2_z_hpol']
            ant2_phase_z_vpol = cm.am_vpol.initial_ant2_z + ( ant2_phase_z_hpol - cm.am_hpol.initial_ant2_z )
            ant2_cable_delay_hpol = m.values['cable_delay2_hpol']
            ant2_cable_delay_vpol = m.values['cable_delay2_vpol']

            ant3_phase_x_hpol = m.values['ant3_x_hpol']
            ant3_phase_x_vpol = cm.am_vpol.initial_ant3_x + ( ant3_phase_x_hpol - cm.am_hpol.initial_ant3_x )
            ant3_phase_y_hpol = m.values['ant3_y_hpol']
            ant3_phase_y_vpol = cm.am_vpol.initial_ant3_y + ( ant3_phase_y_hpol - cm.am_hpol.initial_ant3_y )
            ant3_phase_z_hpol = m.values['ant3_z_hpol']
            ant3_phase_z_vpol = cm.am_vpol.initial_ant3_z + ( ant3_phase_z_hpol - cm.am_hpol.initial_ant3_z )
            ant3_cable_delay_hpol = m.values['cable_delay3_hpol']
            ant3_cable_delay_vpol = m.values['cable_delay3_vpol']

            ant0_ENU_hpol = numpy.array([ant0_phase_x_hpol, ant0_phase_y_hpol, ant0_phase_z_hpol])
            ant1_ENU_hpol = numpy.array([ant1_phase_x_hpol, ant1_phase_y_hpol, ant1_phase_z_hpol])
            ant2_ENU_hpol = numpy.array([ant2_phase_x_hpol, ant2_phase_y_hpol, ant2_phase_z_hpol])
            ant3_ENU_hpol = numpy.array([ant3_phase_x_hpol, ant3_phase_y_hpol, ant3_phase_z_hpol])
            ant0_ENU_vpol = numpy.array([ant0_phase_x_vpol, ant0_phase_y_vpol, ant0_phase_z_vpol])
            ant1_ENU_vpol = numpy.array([ant1_phase_x_vpol, ant1_phase_y_vpol, ant1_phase_z_vpol])
            ant2_ENU_vpol = numpy.array([ant2_phase_x_vpol, ant2_phase_y_vpol, ant2_phase_z_vpol])
            ant3_ENU_vpol = numpy.array([ant3_phase_x_vpol, ant3_phase_y_vpol, ant3_phase_z_vpol])
            resulting_cable_delays_hpol = numpy.array([ant0_cable_delay_hpol,ant1_cable_delay_hpol,ant2_cable_delay_hpol,ant3_cable_delay_hpol])
            resulting_cable_delays_vpol = numpy.array([ant0_cable_delay_vpol,ant1_cable_delay_vpol,ant2_cable_delay_vpol,ant3_cable_delay_vpol])

            output_antennas_phase_hpol = {0:ant0_ENU_hpol, 1:ant1_ENU_hpol, 2:ant2_ENU_hpol, 3:ant3_ENU_hpol}
            output_antennas_phase_vpol = {0:ant0_ENU_vpol, 1:ant1_ENU_vpol, 2:ant2_ENU_vpol, 3:ant3_ENU_vpol}

            chi2_ax.plot([cm.am_hpol.initial_ant0_x , ant0_phase_x_hpol], [cm.am_hpol.initial_ant0_y , ant0_phase_y_hpol], [cm.am_hpol.initial_ant0_z , ant0_phase_z_hpol],c='r',alpha=0.5,linestyle='--')
            chi2_ax.plot([cm.am_hpol.initial_ant1_x , ant1_phase_x_hpol], [cm.am_hpol.initial_ant1_y , ant1_phase_y_hpol], [cm.am_hpol.initial_ant1_z , ant1_phase_z_hpol],c='g',alpha=0.5,linestyle='--')
            chi2_ax.plot([cm.am_hpol.initial_ant2_x , ant2_phase_x_hpol], [cm.am_hpol.initial_ant2_y , ant2_phase_y_hpol], [cm.am_hpol.initial_ant2_z , ant2_phase_z_hpol],c='b',alpha=0.5,linestyle='--')
            chi2_ax.plot([cm.am_hpol.initial_ant3_x , ant3_phase_x_hpol], [cm.am_hpol.initial_ant3_y , ant3_phase_y_hpol], [cm.am_hpol.initial_ant3_z , ant3_phase_z_hpol],c='m',alpha=0.5,linestyle='--')

            chi2_ax.scatter(ant0_phase_x_hpol, ant0_phase_y_hpol, ant0_phase_z_hpol,c='r',alpha=0.5,label='Hpol Final Ant0', marker='$H_f$')
            chi2_ax.scatter(ant1_phase_x_hpol, ant1_phase_y_hpol, ant1_phase_z_hpol,c='g',alpha=0.5,label='Hpol Final Ant1', marker='$H_f$')
            chi2_ax.scatter(ant2_phase_x_hpol, ant2_phase_y_hpol, ant2_phase_z_hpol,c='b',alpha=0.5,label='Hpol Final Ant2', marker='$H_f$')
            chi2_ax.scatter(ant3_phase_x_hpol, ant3_phase_y_hpol, ant3_phase_z_hpol,c='m',alpha=0.5,label='Hpol Final Ant3', marker='$H_f$')

            chi2_ax.plot([cm.am_vpol.initial_ant0_x , ant0_phase_x_vpol], [cm.am_vpol.initial_ant0_y , ant0_phase_y_vpol], [cm.am_vpol.initial_ant0_z , ant0_phase_z_vpol],c='r',alpha=0.5,linestyle='--')
            chi2_ax.plot([cm.am_vpol.initial_ant1_x , ant1_phase_x_vpol], [cm.am_vpol.initial_ant1_y , ant1_phase_y_vpol], [cm.am_vpol.initial_ant1_z , ant1_phase_z_vpol],c='g',alpha=0.5,linestyle='--')
            chi2_ax.plot([cm.am_vpol.initial_ant2_x , ant2_phase_x_vpol], [cm.am_vpol.initial_ant2_y , ant2_phase_y_vpol], [cm.am_vpol.initial_ant2_z , ant2_phase_z_vpol],c='b',alpha=0.5,linestyle='--')
            chi2_ax.plot([cm.am_vpol.initial_ant3_x , ant3_phase_x_vpol], [cm.am_vpol.initial_ant3_y , ant3_phase_y_vpol], [cm.am_vpol.initial_ant3_z , ant3_phase_z_vpol],c='m',alpha=0.5,linestyle='--')

            chi2_ax.scatter(ant0_phase_x_vpol, ant0_phase_y_vpol, ant0_phase_z_vpol,c='r',alpha=0.5,label='Vpol Final Ant0', marker='$H_f$')
            chi2_ax.scatter(ant1_phase_x_vpol, ant1_phase_y_vpol, ant1_phase_z_vpol,c='g',alpha=0.5,label='Vpol Final Ant1', marker='$H_f$')
            chi2_ax.scatter(ant2_phase_x_vpol, ant2_phase_y_vpol, ant2_phase_z_vpol,c='b',alpha=0.5,label='Vpol Final Ant2', marker='$H_f$')
            chi2_ax.scatter(ant3_phase_x_vpol, ant3_phase_y_vpol, ant3_phase_z_vpol,c='m',alpha=0.5,label='Vpol Final Ant3', marker='$H_f$')
            
            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            chi2_ax.dist = 10
            plt.legend()



            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Final Positions')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            if 0 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(ant0_phase_x_hpol, ant0_phase_y_hpol, ant0_phase_z_hpol,marker='$H$',c='r',alpha=0.5,label='Hpol Final Ant0')
            if 1 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(ant1_phase_x_hpol, ant1_phase_y_hpol, ant1_phase_z_hpol,marker='$H$',c='g',alpha=0.5,label='Hpol Final Ant1')
            if 2 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(ant2_phase_x_hpol, ant2_phase_y_hpol, ant2_phase_z_hpol,marker='$H$',c='b',alpha=0.5,label='Hpol Final Ant2')
            if 3 in cm.am_hpol.included_antennas_lumped:
                chi2_ax.scatter(ant3_phase_x_hpol, ant3_phase_y_hpol, ant3_phase_z_hpol,marker='$H$',c='m',alpha=0.5,label='Hpol Final Ant3')
            if 0 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(ant0_phase_x_vpol, ant0_phase_y_vpol, ant0_phase_z_vpol,marker='$V$',c='r',alpha=0.5,label='Vpol Final Ant0')
            if 1 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(ant1_phase_x_vpol, ant1_phase_y_vpol, ant1_phase_z_vpol,marker='$V$',c='g',alpha=0.5,label='Vpol Final Ant1')
            if 2 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(ant2_phase_x_vpol, ant2_phase_y_vpol, ant2_phase_z_vpol,marker='$V$',c='b',alpha=0.5,label='Vpol Final Ant2')
            if 3 in cm.am_vpol.included_antennas_lumped:
                chi2_ax.scatter(ant3_phase_x_vpol, ant3_phase_y_vpol, ant3_phase_z_vpol,marker='$V$',c='m',alpha=0.5,label='Vpol Final Ant3')

            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            chi2_ax.dist = 10
            plt.legend()

            #Plot Pulser Events
            pulser_info = PulserInfo()
            for _pol in ['hpol','vpol']:
                for key in ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']:
                    #Calculate old and new geometries
                    #Distance needed when calling correlator, as it uses that distance.
                    if _pol == 'hpol':
                        original_pulser_ENU = numpy.array([cm.am_hpol.pulser_locations_ENU[key][0] , cm.am_hpol.pulser_locations_ENU[key][1] , cm.am_hpol.pulser_locations_ENU[key][2]])
                    else:
                        original_pulser_ENU = numpy.array([cm.am_vpol.pulser_locations_ENU[key][0] , cm.am_vpol.pulser_locations_ENU[key][1] , cm.am_vpol.pulser_locations_ENU[key][2]])

                    original_distance_m = numpy.linalg.norm(original_pulser_ENU)
                    original_zenith_deg = numpy.rad2deg(numpy.arccos(original_pulser_ENU[2]/original_distance_m))
                    original_elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(original_pulser_ENU[2]/original_distance_m))
                    original_azimuth_deg = numpy.rad2deg(numpy.arctan2(original_pulser_ENU[1],original_pulser_ENU[0]))

                    if _pol == 'hpol':
                        pulser_ENU_new = numpy.array([cm.am_hpol.pulser_locations_ENU[key][0] - ant0_ENU_hpol[0] , cm.am_hpol.pulser_locations_ENU[key][1] - ant0_ENU_hpol[1] , cm.am_hpol.pulser_locations_ENU[key][2] - ant0_ENU_hpol[2]])
                    else:
                        pulser_ENU_new = numpy.array([cm.am_vpol.pulser_locations_ENU[key][0] - ant0_ENU_vpol[0] , cm.am_vpol.pulser_locations_ENU[key][1] - ant0_ENU_vpol[1] , cm.am_vpol.pulser_locations_ENU[key][2] - ant0_ENU_vpol[2]])
                    distance_m = numpy.linalg.norm(pulser_ENU_new)
                    zenith_deg = numpy.rad2deg(numpy.arccos(pulser_ENU_new[2]/distance_m))
                    elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(pulser_ENU_new[2]/distance_m))
                    azimuth_deg = numpy.rad2deg(numpy.arctan2(pulser_ENU_new[1],pulser_ENU_new[0]))

                    map_resolution = 0.25 #degrees
                    range_phi_deg = (-90, 90)
                    range_theta_deg = (80,120)
                    n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                    n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                    

                    known_pulser_ids = info.load2021PulserEventids()[key][_pol]
                    known_pulser_ids = known_pulser_ids[numpy.isin(known_pulser_ids['attenuation_dB'], attenuations_dict[_pol][key])]
                    reference_event  = pulser_info.getPulserReferenceEvent(key, _pol)
                    if True:
                        event_info = reference_event
                    else:
                        event_info = numpy.random.choice(known_pulser_ids)

                    reader = Reader(datapath,int(event_info['run']))
                    
                    cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,cm.am_hpol.initial_ant0_ENU,cm.am_hpol.initial_ant1_ENU,cm.am_hpol.initial_ant2_ENU,cm.am_hpol.initial_ant3_ENU,cm.am_vpol.initial_ant0_ENU,cm.am_vpol.initial_ant1_ENU,cm.am_vpol.initial_ant2_ENU,cm.am_vpol.initial_ant3_ENU,verbose=False)
                   
                    adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                    adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU_hpol,ant1_ENU_hpol,ant2_ENU_hpol,ant3_ENU_hpol,ant0_ENU_vpol,ant1_ENU_vpol,ant2_ENU_vpol,ant3_ENU_vpol,verbose=False)
                    adjusted_cor.overwriteCableDelays(m.values['cable_delay0_hpol'], m.values['cable_delay0_vpol'], m.values['cable_delay1_hpol'], m.values['cable_delay1_vpol'], m.values['cable_delay2_hpol'], m.values['cable_delay2_vpol'], m.values['cable_delay3_hpol'], m.values['cable_delay3_vpol'])

                    if plot_expected_direction == False:
                        zenith_deg = None
                        azimuth_deg = None

                    if plot_time_delays_on_maps:
                        td_dict = {}
                        if False:
                            #Good for troubleshooting if a cycle slipped.
                            cycle_slip_estimate_ns = 15
                            n_cycles = 1
                            td_dict[_pol] = {'[0, 1]' :  measured_time_delays[key][_pol]['delays_ns'][0] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[0, 2]' : measured_time_delays[key][_pol]['delays_ns'][1] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[0, 3]' : measured_time_delays[key][_pol]['delays_ns'][2] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[1, 2]' : measured_time_delays[key][_pol]['delays_ns'][3] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[1, 3]' : measured_time_delays[key][_pol]['delays_ns'][4] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns, '[2, 3]' : measured_time_delays[key][_pol]['delays_ns'][5] + numpy.arange(-n_cycles, n_cycles + 1)*cycle_slip_estimate_ns}
                        else:
                            td_dict[_pol] = {'[0, 1]' :  [measured_time_delays[key][_pol]['delays_ns'][0]], '[0, 2]' : [measured_time_delays[key][_pol]['delays_ns'][1]], '[0, 3]' : [measured_time_delays[key][_pol]['delays_ns'][2]], '[1, 2]' : [measured_time_delays[key][_pol]['delays_ns'][3]], '[1, 3]' : [measured_time_delays[key][_pol]['delays_ns'][4]], '[2, 3]' : [measured_time_delays[key][_pol]['delays_ns'][5]]}
                    else:
                        td_dict = {}


                    #mean_corr_values, fig, ax = cor.map(int(event_info['eventid']), pol, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                    adjusted_mean_corr_values, adjusted_fig, adjusted_ax = adjusted_cor.map(int(event_info['eventid']), _pol, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                    adjusted_fig.set_size_inches(16, 9)
                    plt.sca(adjusted_ax)
                    plt.tight_layout()
                    adjusted_fig.savefig('./%s.png'%key,dpi=90)

                    if plot_histograms:
                        map_resolution = 0.1 #degrees
                        range_phi_deg=(azimuth_deg - 10, azimuth_deg + 10)
                        range_theta_deg=(zenith_deg - 10,zenith_deg + 10)
                        n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                        n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                                        
                        cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                        cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,cm.am_hpol.initial_ant0_ENU,cm.am_hpol.initial_ant1_ENU,cm.am_hpol.initial_ant2_ENU,cm.am_hpol.initial_ant3_ENU,cm.am_vpol.initial_ant0_ENU,cm.am_vpol.initial_ant1_ENU,cm.am_vpol.initial_ant2_ENU,cm.am_vpol.initial_ant3_ENU,verbose=False)

                        adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                        adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU_hpol,ant1_ENU_hpol,ant2_ENU_hpol,ant3_ENU_hpol,ant0_ENU_vpol,ant1_ENU_vpol,ant2_ENU_vpol,ant3_ENU_vpol,verbose=False)
                        adjusted_cor.overwriteCableDelays(m.values['cable_delay0_hpol'], m.values['cable_delay0_vpol'], m.values['cable_delay1_hpol'], m.values['cable_delay1_vpol'], m.values['cable_delay2_hpol'], m.values['cable_delay2_vpol'], m.values['cable_delay3_hpol'], m.values['cable_delay3_vpol'])
                        
                        run_cut = known_pulser_ids['run'] == reader.run #Make sure all eventids in same run
                        hist = adjusted_cor.histMapPeak(numpy.sort(numpy.random.choice(known_pulser_ids[run_cut],min(limit_events,len(known_pulser_ids[run_cut]))))['eventid'], _pol, plot_map=True, hilbert=False, max_method=0, use_weight=False, mollweide=False, center_dir='E', radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90],circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='Hist ' + key, include_baselines=include_baselines,iterate_sub_baselines=iterate_sub_baselines)

            # Finalized Output 

            print('Estimated degrees of freedom: %i'%sum([not v for k, v in m.fixed.items()]))
            print('Estimated input measured values: %i'%(len(cm.am_hpol.include_baselines)*len(cm.am_hpol.use_sites) + len(cm.am_hpol.include_baselines)*len(cm.am_hpol.use_sites) + len(cm.am_vpol.include_baselines)*len(cm.am_vpol.use_sites) + len(cm.am_vpol.include_baselines)*len(cm.am_vpol.use_sites)))


            print('\n')
            print('STARTING CONDITION INPUT VALUES HERE')
            print('\n')
            print('')

            print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%('hpol', cm.am_hpol.initial_ant0_x,cm.am_hpol.initial_ant0_y,cm.am_hpol.initial_ant0_z ,  cm.am_hpol.initial_ant1_x,cm.am_hpol.initial_ant1_y,cm.am_hpol.initial_ant1_z,  cm.am_hpol.initial_ant2_x,cm.am_hpol.initial_ant2_y,cm.am_hpol.initial_ant2_z,  cm.am_hpol.initial_ant3_x,cm.am_hpol.initial_ant3_y,cm.am_hpol.initial_ant3_z))
            print('')
            print('cable_delays_%s = numpy.array([%f,%f,%f,%f])'%('hpol',cm.am_hpol.cable_delays[0],cm.am_hpol.cable_delays[1],cm.am_hpol.cable_delays[2],cm.am_hpol.cable_delays[3]))

            print('\n')
            print(result)
            print('\n')
            print('Copy-Paste Prints:\n------------')
            print('')

            print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%('hpol', ant0_ENU_hpol[0] , ant0_ENU_hpol[1] , ant0_ENU_hpol[2] , ant1_ENU_hpol[0] , ant1_ENU_hpol[1] , ant1_ENU_hpol[2],  ant2_ENU_hpol[0] , ant2_ENU_hpol[1] , ant2_ENU_hpol[2],  ant3_ENU_hpol[0] , ant3_ENU_hpol[1] , ant3_ENU_hpol[2]))
            print('antennas_phase_%s_hesse = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%('hpol', m.errors['ant0_x_hpol'],m.errors['ant0_y_hpol'],m.errors['ant0_z_hpol'] ,  m.errors['ant1_x_hpol'],m.errors['ant1_y_hpol'],m.errors['ant1_z_hpol'],  m.errors['ant2_x_hpol'],m.errors['ant2_y_hpol'],m.errors['ant2_z_hpol'],  m.errors['ant3_x_hpol'],m.errors['ant3_y_hpol'],m.errors['ant3_z_hpol']))
            print('')
            print('cable_delays_%s = numpy.array([%f,%f,%f,%f])'%('hpol',m.values['cable_delay0_hpol'],m.values['cable_delay1_hpol'],m.values['cable_delay2_hpol'],m.values['cable_delay3_hpol']))
            print('cable_delays_%s_hesse = numpy.array([%f,%f,%f,%f])'%('hpol',m.errors['cable_delay0_hpol'],m.errors['cable_delay1_hpol'],m.errors['cable_delay2_hpol'],m.errors['cable_delay3_hpol']))



            print('\n')
            print('STARTING CONDITION INPUT VALUES HERE')
            print('\n')
            print('')

            print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%('vpol', cm.am_vpol.initial_ant0_x,cm.am_vpol.initial_ant0_y,cm.am_vpol.initial_ant0_z ,  cm.am_vpol.initial_ant1_x,cm.am_vpol.initial_ant1_y,cm.am_vpol.initial_ant1_z,  cm.am_vpol.initial_ant2_x,cm.am_vpol.initial_ant2_y,cm.am_vpol.initial_ant2_z,  cm.am_vpol.initial_ant3_x,cm.am_vpol.initial_ant3_y,cm.am_vpol.initial_ant3_z))
            print('')
            print('cable_delays_%s = numpy.array([%f,%f,%f,%f])'%('vpol',cm.am_vpol.cable_delays[0],cm.am_vpol.cable_delays[1],cm.am_vpol.cable_delays[2],cm.am_vpol.cable_delays[3]))

            print('\n')
            print(result)
            print('\n')
            print('Copy-Paste Prints:\n------------')
            print('')

            print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%('vpol', ant0_ENU_vpol[0] , ant0_ENU_vpol[1] , ant0_ENU_vpol[2] , ant1_ENU_vpol[0] , ant1_ENU_vpol[1] , ant1_ENU_vpol[2],  ant2_ENU_vpol[0] , ant2_ENU_vpol[1] , ant2_ENU_vpol[2],  ant3_ENU_vpol[0] , ant3_ENU_vpol[1] , ant3_ENU_vpol[2]))
            print('antennas_phase_%s_hesse = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%('vpol', m.errors['ant0_x_hpol'],m.errors['ant0_y_hpol'],m.errors['ant0_z_hpol'] ,  m.errors['ant1_x_hpol'],m.errors['ant1_y_hpol'],m.errors['ant1_z_hpol'],  m.errors['ant2_x_hpol'],m.errors['ant2_y_hpol'],m.errors['ant2_z_hpol'],  m.errors['ant3_x_hpol'],m.errors['ant3_y_hpol'],m.errors['ant3_z_hpol'])) #Hpol errors because vpol don't move independently
            print('')
            print('cable_delays_%s = numpy.array([%f,%f,%f,%f])'%('vpol',m.values['cable_delay0_vpol'],m.values['cable_delay1_vpol'],m.values['cable_delay2_vpol'],m.values['cable_delay3_vpol']))
            print('cable_delays_%s_hesse = numpy.array([%f,%f,%f,%f])'%('vpol',m.errors['cable_delay0_vpol'],m.errors['cable_delay1_vpol'],m.errors['cable_delay2_vpol'],m.errors['cable_delay3_vpol']))


            print('Code completed.')
            print('\a')

            if True:
                #This code is intended to save the output configuration produced by this script. 
                initial_deploy_index = str(info.returnDefaultDeploy())
                initial_origin, initial_antennas_physical, initial_antennas_phase_hpol, initial_antennas_phase_vpol, initial_cable_delays, initial_description = bcr.configReader(initial_deploy_index,return_description=True)

                output_origin = initial_origin
                output_antennas_physical = initial_antennas_physical
                output_antennas_phase_hpol = output_antennas_phase_hpol
                output_antennas_phase_vpol = output_antennas_phase_vpol

                output_cable_delays = {}
                output_cable_delays['hpol'] = resulting_cable_delays_hpol
                output_cable_delays['vpol'] = resulting_cable_delays_vpol
                output_description = 'Automatically generated description for a calibration starting from deploy_index: %s.  This config has updated %s values based on a calibration that was performed.  Initial description: %s'%(initial_deploy_index, pol, initial_description)

                if len(os.path.split(initial_deploy_index)) == 2:
                    json_path = initial_deploy_index
                else:
                    json_path = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'config','automatically_generated_config_0.json')
                
                with open('./antenna_position_minimization.py', "r") as this_file:
                    #read whole file to a string
                    script_string = this_file.read()

                bcr.configWriter(json_path, output_origin, output_antennas_physical, output_antennas_phase_hpol, output_antennas_phase_vpol, output_cable_delays, description=output_description,update_latlonel=True,force_write=True, additional_text=script_string) #does not overwrite.






            
    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






