'''
This script assumes the antenna positions have already roughly been set/determined
using find_phase_centers.py  .  It will assume these as starting points of the antennas
and then vary all 4 of their locations to match measured time delays from planes. 
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

#Personal Imports
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import objects.station as bc
import tools.info as info
from tools.correlator import Correlator
import tools.get_plane_tracks as pt
from objects.fftmath import TimeDelayCalculator

#Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


#Settings
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()
datapath = os.environ['BEACON_DATA']

n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
c = 299792458/n #m/s

class PlanePoly:
    def __init__(self, t, enu, order=7,plot=False):
        '''
        Given times (array of numbers) and enu data (tuple of 3 arrays of same length as t),
        this will create 3 polynomial fit functions to each of the ENU coordinates.  These
        can then be used to calculate ENU coordinates of the planes at interpolated values
        in time.
        '''
        try:
            self.order = order
            self.time_offset = min(t)
            self.fit_params = []
            self.poly_funcs = []
            self.range = (min(t),max(t))

            if plot == True:
                plane_norm_fig = plt.figure()
                plane_norm_ax = plt.gca()
            for i in range(3):
                index = numpy.sort(numpy.unique(enu[i],return_index=True)[1]) #Indices of values to be used in fit.
                self.fit_params.append(numpy.polyfit(t[index] - self.time_offset, enu[i][index],self.order)) #Need to shift t to make fit work
                self.poly_funcs.append(numpy.poly1d(self.fit_params[i]))
                if plot == True:
                    plane_norm_ax.plot(t,self.poly_funcs[i](t - self.time_offset)/1000.0,label='Fit %s Coord order %i'%(['E','N','U'][i],order))
                    plane_norm_ax.scatter(t[index],enu[i][index]/1000.0,label='Data for %s'%(['E','N','U'][i]),alpha=0.3)
            if plot == True:
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.xlabel('Timestamp')
                plt.ylabel('Distance from Ant0 Along Axis (km)')

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def poly(self,t):
        '''
        Will return the resulting values of ENU for the given t.
        '''
        try:
            return numpy.vstack((self.poly_funcs[0](t-self.time_offset),self.poly_funcs[1](t-self.time_offset),self.poly_funcs[2](t-self.time_offset))).T
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)




if __name__ == '__main__':
    try:
        plot_planes = False
        plot_interps = False

        final_corr_length = 2**18
        crit_freq_low_pass_MHz = None#60 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = None#5

        crit_freq_high_pass_MHz = None#60
        high_pass_filter_order = None#6

        waveform_index_range = (None,None)#(150,400)

        apply_phase_response = False
        hilbert = False

        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to hpol')
                mode = 'hpol'
        else:
            print('No mode given.  Defaulting to hpol')
            mode = 'hpol'

        print('Loading known plane locations.')
        known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks()
        origin = info.loadAntennaZeroLocation(deploy_index = 1)
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()
        if mode == 'hpol':
            antennas_phase_start = antennas_phase_hpol
        else:
            antennas_phase_start = antennas_phase_vpol

        print('Loading in cable delays.')
        cable_delays = info.loadCableDelays()[mode]

        if plot_planes == True:
            plane_fig = plt.figure()
            plane_fig.canvas.set_window_title('3D Plane Tracks')
            plane_ax = plane_fig.add_subplot(111, projection='3d')
            plane_ax.scatter(0,0,0,label='Antenna 0',c='k')

        plane_polys = {}
        interpolated_plane_locations = {}
        measured_plane_times_delays = {}
        measured_plane_times_delays_weights = {}
        for key in list(calibrated_trigtime.keys()):
            enu = pm.geodetic2enu(output_tracks[key]['lat'],output_tracks[key]['lon'],output_tracks[key]['alt'],origin[0],origin[1],origin[2])

            plane_polys[key] = PlanePoly(output_tracks[key]['timestamps'],enu,plot=plot_interps)

            interpolated_plane_locations[key] = plane_polys[key].poly(calibrated_trigtime[key])
            
            if plot_planes == True:
                plane_ax.plot(enu[0]/1000.0,enu[1]/1000.0,enu[2]/1000.0,label=key + ' : ' + known_planes[key]['known_flight'])
                plane_ax.scatter(interpolated_plane_locations[key][:,0]/1000.0,interpolated_plane_locations[key][:,1]/1000.0,interpolated_plane_locations[key][:,2]/1000.0,label=key + ' : ' + known_planes[key]['known_flight'])

            # enu_az = numpy.rad2deg(numpy.arctan2(interpolated_plane_locations[key][:,1],interpolated_plane_locations[key][:,0]))
            # enu_elevation_angle = numpy.rad2deg(numpy.arctan2(interpolated_plane_locations[key][:,2],numpy.sqrt(interpolated_plane_locations[key][:,0]**2+interpolated_plane_locations[key][:,1]**2)))
            run = int(key.split('-')[0])
            reader = Reader(datapath,run)
            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
            eventids = known_planes[key]['eventids'][:,1]
            time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=0,hilbert=hilbert)

            

            pair_cut = numpy.array([pair in known_planes[key]['baselines'][mode] for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ]) #Checks which pairs are worth looping over.

            if mode == 'hpol':
                measured_plane_times_delays[key] = time_shifts[0:6,:][pair_cut].T
                measured_plane_times_delays_weights[key] = corrs[0:6,:][pair_cut].T #might need something better than this. 
            else:
                measured_plane_times_delays[key] = time_shifts[6:12,:][pair_cut].T
                measured_plane_times_delays_weights[key] = corrs[6:12,:][pair_cut].T #might need something better than this. 



        if plot_planes == True:
            plt.legend(loc='upper right')
            plane_ax.set_xlabel('East (km)',linespacing=10)
            plane_ax.set_ylabel('North (km)',linespacing=10)
            plane_ax.set_zlabel('Up (km)',linespacing=10)
            plane_ax.dist = 10


        ##########
        # Define Chi^2
        ##########

        def rawChi2(ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z):
            '''
            This is a chi^2 that loops over locations from planes, calculating expected time delays for those locations.  Then
            it will compares those to the calculated time delays for known plane events.  
            '''
            try:
                #fixing the locations of antenna zero.
                ant0_x = 0.0
                ant0_y = 0.0
                ant0_z = 0.0

                #Calculate distances (already converted to ns) from pulser to each antenna


                chi_2 = 0.0


                for plane in list(known_planes.keys()):
                    d0 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant0_x)**2 + (interpolated_plane_locations[key][:,1] - ant0_y)**2 + (interpolated_plane_locations[key][:,2] - ant0_z)**2 )/c)*1.0e9 #ns
                    d1 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant1_x)**2 + (interpolated_plane_locations[key][:,1] - ant1_y)**2 + (interpolated_plane_locations[key][:,2] - ant1_z)**2 )/c)*1.0e9 #ns
                    d2 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant2_x)**2 + (interpolated_plane_locations[key][:,1] - ant2_y)**2 + (interpolated_plane_locations[key][:,2] - ant2_z)**2 )/c)*1.0e9 #ns
                    d3 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant3_x)**2 + (interpolated_plane_locations[key][:,1] - ant3_y)**2 + (interpolated_plane_locations[key][:,2] - ant3_z)**2 )/c)*1.0e9 #ns

                    d = [d0,d1,d2,d3]

                    for pair_index, pair in enumerate(known_planes[plane]['baselines'][mode]):
                        geometric_time_delay = (d[pair[0]] + cable_delays[0]) - (d[pair[1]] + cable_delays[1])
                        #Right now these seem reversed from what I would expect based on the plot?  In time I mean. 

                        import pdb; pdb.set_trace()
                        #NOTE: Right now this will weight planes with more events higher.  That might be what I want, it might not be.  Might want to divide by number of events per plane so each plane has equal overall weighting
                        #Could just average instead of sum here?
                        chi_2 += sum(((geometric_time_delay - measured_plane_times_delays[key][pair_index])**2)/measured_plane_times_delays_weights[key][pair_index]**2)
                 
                return chi_2
            except Exception as e:
                print('Error in rawChi2')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


        initial_step = 1.0 #m
        #-12 ft on pulser locations relative to antennas to account for additional mast elevation.
        
        
        
        m = Minuit(     rawChi2,\
                        ant1_x=antennas_phase_start[1][0],\
                        ant1_y=antennas_phase_start[1][1],\
                        ant1_z=antennas_phase_start[1][2],\
                        ant2_x=antennas_phase_start[2][0],\
                        ant2_y=antennas_phase_start[2][1],\
                        ant2_z=antennas_phase_start[2][2],\
                        ant3_x=antennas_phase_start[3][0],\
                        ant3_y=antennas_phase_start[3][1],\
                        ant3_z=antennas_phase_start[3][2],\
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
                        limit_ant1_x=None,\
                        limit_ant1_y=None,\
                        limit_ant1_z=None,\
                        limit_ant2_x=None,\
                        limit_ant2_y=None,\
                        limit_ant2_z=None,\
                        limit_ant3_x=None,\
                        limit_ant3_y=None,\
                        limit_ant3_z=None,\
                        fix_ant1_x=False,\
                        fix_ant1_y=False,\
                        fix_ant1_z=False,\
                        fix_ant2_x=False,\
                        fix_ant2_y=False,\
                        fix_ant2_z=False,\
                        fix_ant3_x=False,\
                        fix_ant3_y=False,\
                        fix_ant3_z=False)


        result = m.migrad(resume=False)

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


        '''


        if True:
            print('\n\nATTEMPTING SECOND CHI^2 WITH FIRST AS INPUT\n')
            initial_chi_2 = rawChi2(m.values['ant1_x'], m.values['ant1_y'], m.values['ant1_z'], m.values['ant2_x'], m.values['ant2_y'], m.values['ant2_z'], m.values['ant3_x'], m.values['ant3_y'], m.values['ant3_z'], m.values['puls1_x'], m.values['puls1_y'], m.values['puls1_z'], m.values['puls2_x'], m.values['puls2_y'], m.values['puls2_z'], m.values['puls3_x'], m.values['puls3_y'], m.values['puls3_z'],m.values['cable_delay_ant0'], m.values['cable_delay_ant1'], m.values['cable_delay_ant2'], m.values['cable_delay_ant3'])
            print('Initial Chi^2 is %0.3f\n'%(initial_chi_2))
            initial_step = 10.0

            m = Minuit(     rawChi2,\
                            ant1_x=m.values['ant1_x'],\
                            ant1_y=m.values['ant1_y'],\
                            ant1_z=m.values['ant1_z'],\
                            ant2_x=m.values['ant2_x'],\
                            ant2_y=m.values['ant2_y'],\
                            ant2_z=m.values['ant2_z'],\
                            ant3_x=m.values['ant3_x'],\
                            ant3_y=m.values['ant3_y'],\
                            ant3_z=m.values['ant3_z'],\
                            puls1_x=m.values['puls1_x'],\
                            puls1_y=m.values['puls1_y'],\
                            puls1_z=m.values['puls1_z'],\
                            puls2_x=m.values['puls2_x'],\
                            puls2_y=m.values['puls2_y'],\
                            puls2_z=m.values['puls2_z'],\
                            puls3_x=m.values['puls3_x'],\
                            puls3_y=m.values['puls3_y'],\
                            puls3_z=m.values['puls3_z'],\
                            cable_delay_ant0=m.values['cable_delay_ant0'],\
                            cable_delay_ant1=m.values['cable_delay_ant1'],\
                            cable_delay_ant2=m.values['cable_delay_ant2'],\
                            cable_delay_ant3=m.values['cable_delay_ant3'],\
                            error_ant1_x=initial_step,\
                            error_ant1_y=initial_step,\
                            error_ant1_z=initial_step,\
                            error_ant2_x=initial_step,\
                            error_ant2_y=initial_step,\
                            error_ant2_z=initial_step,\
                            error_ant3_x=initial_step,\
                            error_ant3_y=initial_step,\
                            error_ant3_z=initial_step,\
                            error_puls1_x=initial_step,\
                            error_puls1_y=initial_step,\
                            error_puls1_z=initial_step,\
                            error_puls2_x=initial_step,\
                            error_puls2_y=initial_step,\
                            error_puls2_z=initial_step,\
                            error_puls3_x=initial_step,\
                            error_puls3_y=initial_step,\
                            error_puls3_z=initial_step,\
                            error_cable_delay_ant0=1.0,\
                            error_cable_delay_ant1=1.0,\
                            error_cable_delay_ant2=1.0,\
                            error_cable_delay_ant3=1.0,\
                            errordef = 1.0,\
                            limit_ant1_x=(None,None),\
                            limit_ant1_y=(None,None),\
                            limit_ant1_z=(None,None),\
                            limit_ant2_x=(None,None),\
                            limit_ant2_y=(None,None),\
                            limit_ant2_z=(None,None),\
                            limit_ant3_x=(None,None),\
                            limit_ant3_y=(None,None),\
                            limit_ant3_z=(None,None),\
                            limit_puls1_x=(None,None),\
                            limit_puls1_y=(None,None),\
                            limit_puls1_z=(None,None),\
                            limit_puls2_x=(None,None),\
                            limit_puls2_y=(None,None),\
                            limit_puls2_z=(None,None),\
                            limit_puls3_x=(None,None),\
                            limit_puls3_y=(None,None),\
                            limit_puls3_z=(None,None),
                            limit_cable_delay_ant0=(None,None),\
                            limit_cable_delay_ant1=(None,None),\
                            limit_cable_delay_ant2=(None,None),\
                            limit_cable_delay_ant3=(None,None),\
                            fix_ant1_x=False,\
                            fix_ant1_y=False,\
                            fix_ant1_z=False,\
                            fix_ant2_x=False,\
                            fix_ant2_y=False,\
                            fix_ant2_z=False,\
                            fix_ant3_x=False,\
                            fix_ant3_y=False,\
                            fix_ant3_z=False,\
                            fix_puls1_x=False,\
                            fix_puls1_y=False,\
                            fix_puls1_z=False,\
                            fix_puls2_x=False,\
                            fix_puls2_y=False,\
                            fix_puls2_z=False,\
                            fix_puls3_x=False,\
                            fix_puls3_y=False,\
                            fix_puls3_z=False,\
                            fix_cable_delay_ant0=True,\
                            fix_cable_delay_ant1=True,\
                            fix_cable_delay_ant2=True,\
                            fix_cable_delay_ant3=True)


            m.tol = m.tol/100
            result = m.migrad(resume=False)

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

            cable_delay_ant0 = m.values['cable_delay_ant0']
            print('New cable delay, ant 0', cable_delay_ant0)
            print('cable_delays[0] - cable_delay_ant0 = ',cable_delays[0] - cable_delay_ant0)
            cable_delay_ant1 = m.values['cable_delay_ant1']
            print('New cable delay, ant 1', cable_delay_ant1)
            print('cable_delays[1] - cable_delay_ant1 = ',cable_delays[1] - cable_delay_ant1)
            cable_delay_ant2 = m.values['cable_delay_ant2']
            print('New cable delay, ant 2', cable_delay_ant2)
            print('cable_delays[2] - cable_delay_ant2 = ',cable_delays[2] - cable_delay_ant2)
            cable_delay_ant3 = m.values['cable_delay_ant3']
            print('New cable delay, ant 3', cable_delay_ant3)
            print('cable_delays[3] - cable_delay_ant3 = ',cable_delays[3] - cable_delay_ant3)


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


        print('Copy-Paste Prints:\n------------')
        print('')
        print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, phase_locs[0][0],phase_locs[0][1],phase_locs[0][2],  phase_locs[1][0],phase_locs[1][1],phase_locs[1][2],  phase_locs[2][0],phase_locs[2][1],phase_locs[2][2],  phase_locs[3][0],phase_locs[3][1],phase_locs[3][2]))
        print('antennas_phase_%s_hesse = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, 0.0 ,0.0 ,0.0 ,  m.errors['ant1_x'],m.errors['ant1_y'],m.errors['ant1_z'],  m.errors['ant2_x'],m.errors['ant2_y'],m.errors['ant2_z'],  m.errors['ant3_x'],m.errors['ant3_y'],m.errors['ant3_z']))
        print('')
        print('pulser_locations_ENU[\'%s\'] = {\'run1507\':[%f, %f, %f], \'run1509\':[%f, %f, %f], \'run1511\':[%f, %f, %f]}'%(mode,m.values['puls1_x'], m.values['puls1_y'], m.values['puls1_z'], m.values['puls2_x'], m.values['puls2_y'], m.values['puls2_z'], m.values['puls3_x'], m.values['puls3_y'], m.values['puls3_z']))
        print('pulser_locations_ENU[\'%s_hesse_error\'] = {\'run1507\':[%f, %f, %f], \'run1509\':[%f, %f, %f], \'run1511\':[%f, %f, %f]}'%(mode,m.errors['puls1_x'], m.errors['puls1_y'], m.errors['puls1_z'], m.errors['puls2_x'], m.errors['puls2_y'], m.errors['puls2_z'], m.errors['puls3_x'], m.errors['puls3_y'], m.errors['puls3_z']))
        print('')
        print('cable delays\n\'%s\' : numpy.array(['%mode,m.values['cable_delay_ant0'],',',m.values['cable_delay_ant1'],',',m.values['cable_delay_ant2'],',',m.values['cable_delay_ant3'],'])')


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

            pulser_locations = info.loadPulserPhaseLocationsENU()['physical']


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

        '''


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






