#!/usr/bin/env python3
'''
This is a simple script designed to get an intitial impression of the data in-order to determine an appropriate
analysis trajectory. 
'''

import sys
import os
import inspect
import h5py
import copy

import numpy
import scipy
import scipy.signal
import pymap3d as pm

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import loadTriggerTypes
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.line_of_sight import circleSource
import beacon.tools.info as info
from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
import time


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
processed_datapath = os.environ['BEACON_PROCESSED_DATA']

from scipy.optimize import curve_fit
def gaus(x,a,x0,sigma, normalize=False):
    out = a*numpy.exp(-(x-x0)**2.0/(2.0*(sigma**2.0)))
    if normalize == False:
        return out
    else:
        return out/(numpy.sum(out)*numpy.diff(x)[0])

def bivariateGaus(xy ,x0,sigma_x, y0,sigma_y, rho, scale_factor=1.0, return_2d=False):
    '''
    This calculation is already normalized as a pdf?  So I need to reverse normalize when fitting I guess?
    '''
    x, y = xy

    mu = numpy.array([sigma_x,sigma_y])
    sigma = numpy.array([[sigma_x**2, rho*sigma_x*sigma_y],[rho*sigma_x*sigma_y,sigma_y**2]])

    z = 1.0/(2*numpy.pi*sigma_x*sigma_y*numpy.sqrt(1-rho**2)) * numpy.exp( -1.0 / ( 2 * ( 1 - rho**2 ) ) * ( ((x - x0)/sigma_x)**2 - 2*rho*((x - x0)/sigma_x)*((y - y0)/sigma_y) + ((y - y0)/sigma_y)**2 ) )

    if return_2d:
        '''
        Useful when using in curve_fit, which expects a 1d output.
        '''
        return z*scale_factor
    else:
        '''
        Useful for plotting, which expects a 2d output.
        '''
        return z.ravel()*scale_factor



def fitGaus(x, _counts, _ax, add_label=False, center=False, normalize=False, debug=False, **kwargs):
    try:
        p0 = [numpy.max(_counts),x[numpy.argmax(_counts)],2*numpy.abs(x[numpy.argmax(_counts[_counts != 0])] - x[numpy.argmin(_counts[_counts != 0])])]
        if debug == True:
            import pdb; pdb.set_trace()
        popt, pcov = curve_fit(gaus,x,_counts,p0=p0) #Only want to use normalize when plotting not fitting.
        popt[2] = abs(popt[2]) #I want positive sigma.

        plot_x = numpy.linspace(min(x),max(x),100*len(x))


        if center == False:
            if add_label == True:
                _ax.plot(plot_x,gaus(plot_x,*popt),label='Center = %0.4f,  Sigma = %0.4f deg'%(popt[1],popt[2]), **kwargs)
                _ax.axvline(popt[1], **kwargs)
            else:
                _ax.plot(plot_x,gaus(plot_x,*popt), **kwargs)
        else:
            if add_label == True:
                _ax.plot(plot_x - popt[1],gaus(plot_x,*popt, normalize=normalize),label='Center = %0.4f,  Sigma = %0.4f deg'%(popt[1],popt[2]), **kwargs)
            else:
                _ax.plot(plot_x - popt[1],gaus(plot_x,*popt, normalize=normalize), **kwargs)

        return popt


    except Exception as e:
        print('Failed to fit histogram')
        print(e)
        print('Trying to add info without fit.')

'''
'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v',\
'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',
'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish','theta_best_h','theta_best_v','elevation_best_h','elevation_best_v','phi_best_h','phi_best_v',\
'calibrated_trigtime','triggered_beams','beam_power','hpol_peak_to_sidelobe','vpol_peak_to_sidelobe','hpol_max_possible_map_value','vpol_max_possible_map_value',\
'map_max_time_delay_0subtract1_h','map_max_time_delay_0subtract2_h','map_max_time_delay_0subtract3_h',\
'map_max_time_delay_1subtract2_h','map_max_time_delay_1subtract3_h','map_max_time_delay_2subtract3_h',\
'map_max_time_delay_0subtract1_v','map_max_time_delay_0subtract2_v','map_max_time_delay_0subtract3_v',\
'map_max_time_delay_1subtract2_v','map_max_time_delay_1subtract3_v','map_max_time_delay_2subtract3_v'
'''


def mahalanobisDistanceThreshold(p):
    '''
    This calculates the threshold value t to achieve a particular probability p.
    '''
    return numpy.sqrt(-2*numpy.log(1-p))


def covarianceMatrixEllipse(sigma):
    '''
    This follows the guide found at https://cookierobotics.com/007/

    Given a covariance matrix of form [[a,b],[c,d]] this will determine the values of lambda_1 and lambda_2
    where sqrt(lambda_1) is the semi-major axis of the ellipse and sqrt(lambda_2) is the semiminor axis.  It will also
    calculate theta_rad, which is the counter-clockwise offset from the positive x-axis in which the semi-major axis
    is rotated.  
    '''
    a = sigma[0][0]
    b = sigma[0][1]
    c = sigma[1][1]
    
    lambda_1 = (a + c)/2.0 + numpy.sqrt(((a - c)/2)**2 + b**2)
    lambda_2 = (a + c)/2.0 - numpy.sqrt(((a - c)/2)**2 + b**2)

    if b == 0 and a >= c:
        theta_rad = 0
    elif b == 0 and a < c:
        theta_rad = numpy.pi/2
    else:
        theta_rad = numpy.arctan2(lambda_1 - a , b)

    return lambda_1, lambda_2, theta_rad

def parametricCovarianceEllipse(sigma, mean, confidence_interval_value, n=10000):
    '''
    This calculate the parameterized vertices of an ellipse by passing sigma to covarianceMatrixEllipse.  
    '''
    scale_factor = mahalanobisDistanceThreshold(confidence_interval_value)
    lambda_1, lambda_2, theta_rad = covarianceMatrixEllipse(sigma)
    t = numpy.linspace(0,2*numpy.pi, n)
    x = numpy.sqrt(lambda_1) * numpy.cos(theta_rad) * numpy.cos(t) - numpy.sqrt(lambda_2) * numpy.sin(theta_rad) * numpy.sin(t)
    y = numpy.sqrt(lambda_1) * numpy.sin(theta_rad) * numpy.cos(t) - numpy.sqrt(lambda_2) * numpy.cos(theta_rad) * numpy.sin(t)

    #unsure if this is the correct way to apply the scale factor yet.  Maybe under the sqrt above?
    x = x*scale_factor + mean[0]
    y = y*scale_factor + mean[1]

    return numpy.vstack((x,y)).T



def contourArea(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return abs(a)

known_pulser_ids = info.load2021PulserEventids()
known_pulser_runs = numpy.unique(numpy.concatenate([numpy.append(numpy.unique(known_pulser_ids[site]['hpol']['run']),numpy.unique(known_pulser_ids[site]['vpol']['run'])) for site in list(known_pulser_ids.keys())]))
pulser_run_sites = {}
for site in list(known_pulser_ids.keys()):
    for run in numpy.unique(numpy.append(numpy.unique(known_pulser_ids[site]['hpol']['run']),numpy.unique(known_pulser_ids[site]['vpol']['run']))):
        pulser_run_sites[run] = site

attenuations_dict = {        'hpol':{   'd2sa' : [20],
                                        'd3sa' : [10],
                                        'd3sb' : [6],
                                        'd3sc' : [20],
                                        'd4sa' : [20],
                                        'd4sb' : [6]
                                    },
                             'vpol':{   'd2sa' : [10],
                                        'd3sa' : [6],
                                        'd3sb' : [20],
                                        'd3sc' : [20],
                                        'd4sa' : [10],
                                        'd4sb' : [6]
                                    }
                            }

if __name__ == '__main__':

    fontsize = 18

    plt.close('all')
    cmap = 'cool'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

    print("Preparing dataSlicer")

    confidence_integral_value = 0.9

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -90
    max_phi     = 90
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)
    run_modes = ['rfi']#['rfi', 'pulsers']#['pulsers']#['rfi', 'pulsers']

    cut_range = 2.5

    for mode in run_modes:
        print('Generating %s plots:'%mode)
        if mode == 'rfi':
            # runs = numpy.arange(5733,5736)# numpy.arange(5733,5974,dtype=int)#numpy.arange(5733,5736)
            runs = numpy.arange(5733,5974,dtype=int)#numpy.arange(5733,5736)

            roi_dict = {}
            roi_dict['RFI Source 0'] = {'phi_best_h':[-9,-7.0],      'elevation_best_h':[-7.2,1.8],  'phi_best_v':[-9,-7.0],'elevation_best_v':[-7.2,1.8],'std_h':[1,10], 'std_v':[1,5]}
            roi_dict['RFI Source 1'] = {'phi_best_h':[-7.5,-3.7] ,   'elevation_best_h':[-6.3,-3.7], 'p2p_v':[0,40],'impulsivity_v':[0,0.35]}
            roi_dict['RFI Source 2'] = {'phi_best_h':[-3,-1.6] ,     'elevation_best_h':[-7.2,-4],   'cr_template_search_v':[0.25,0.5],'impulsivity_v':[0.2,1]}
            roi_dict['RFI Source 3'] = {'phi_best_h':[-1.7,0.0] ,    'elevation_best_h':[-10,-2.25], 'mean_max_corr_h':[0.65,1.1],'mean_max_corr_v':[0.775,1.1],'impulsivity_v':[0.4,0.6]}
            roi_dict['RFI Source 4'] = {'phi_best_h':[17,18.5] ,     'elevation_best_h':[-6,-2.0],   'std_v':[0,4],'mean_max_corr_v':[0.3,1.0]}
            roi_dict['RFI Source 5'] = {'phi_best_h':[22.35,23.6] ,  'elevation_best_h':[-5,-0.5],   'p2p_v':[0,40]}
            roi_dict['RFI Source 6'] = {'phi_best_h':[40.75,42] ,    'elevation_best_h':[-5,-0.5],   'std_v':[2.5,8],'std_h':[4,9.5], 'cr_template_search_v':[0.775,1.0],'vpol_peak_to_sidelobe':[2.05,3.5]}
            trigger_types = [2]
            # pols = ['hpol']
            pols = ['hpol']#,'choice']
        else:
            origin = info.loadAntennaZeroLocation()
            pulser_info = PulserInfo()
            direction_dict = {}
            for site in ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']:
                source_latlonel = pulser_info.getPulserLatLonEl(site)
                # Prepare expected angle and arrival times
                enu = pm.geodetic2enu(source_latlonel[0],source_latlonel[1],source_latlonel[2],origin[0],origin[1],origin[2])
                source_distance_m = numpy.sqrt(enu[0]**2 + enu[1]**2 + enu[2]**2)
                azimuth_deg = numpy.rad2deg(numpy.arctan2(enu[1],enu[0]))
                zenith_deg = numpy.rad2deg(numpy.arccos(enu[2]/source_distance_m))
                direction_dict[site] = {'azimuth_deg':azimuth_deg,'zenith_deg':zenith_deg,'elevation_deg':90.0-zenith_deg,'distance_m':source_distance_m}


                print(site)
                print(source_latlonel[0],source_latlonel[1],source_latlonel[2])
                print(azimuth_deg , 90.0 - zenith_deg, source_distance_m)


            runs = numpy.array([5630, 5631, 5632, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5656, 5657, 5659, 5660], dtype=int)
            roi_dict = {}
            pm_range = 6
            global_azimuth_offset = 1.5
            roi_dict['d2sa'] = {'phi_best_h':[  direction_dict['d2sa']['azimuth_deg'] + global_azimuth_offset - pm_range, direction_dict['d2sa']['azimuth_deg'] + global_azimuth_offset + pm_range,] }
            roi_dict['d3sa'] = {'phi_best_h':[  direction_dict['d3sa']['azimuth_deg'] + global_azimuth_offset - pm_range, direction_dict['d3sa']['azimuth_deg'] + global_azimuth_offset + pm_range,] }
            roi_dict['d3sb'] = {'phi_best_h':[  direction_dict['d3sb']['azimuth_deg'] + global_azimuth_offset - pm_range, direction_dict['d3sb']['azimuth_deg'] + global_azimuth_offset + pm_range,] }
            roi_dict['d3sc'] = {'phi_best_h':[  direction_dict['d3sc']['azimuth_deg'] + global_azimuth_offset - pm_range, direction_dict['d3sc']['azimuth_deg'] + global_azimuth_offset + pm_range,] }
            roi_dict['d4sa'] = {'phi_best_h':[  direction_dict['d4sa']['azimuth_deg'] + global_azimuth_offset - pm_range, direction_dict['d4sa']['azimuth_deg'] + global_azimuth_offset + pm_range,] }
            roi_dict['d4sb'] = {'phi_best_h':[  direction_dict['d4sb']['azimuth_deg'] + global_azimuth_offset - pm_range, direction_dict['d4sb']['azimuth_deg'] + global_azimuth_offset + pm_range,] }

            trigger_types = [3]
            pols = ['hpol']#['hpol','vpol']

        ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, remove_incomplete_runs=True, curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                        n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta))


        for key, item in roi_dict.items():
            ds.addROI(key, item)

        ds.conference_mode = True

        #ds.addROI('RFI Source 1',{'elevation_best_h':[-7.2,1.8],'phi_best_h':[-9,-7.0],'elevation_best_v':[-7.2,1.8],'phi_best_v':[-9,-7.0],'std_h':[1,10], 'std_v':[1,5]})
        save_fig_dir = './figures/resolution_plots/' #Set to None for interactive mode
        fullsize_fig_dims = (20,10)
        halfsize_fig_dims = (10,10)

        if False:
            plotter = plt.semilogy
            logscale = True
        elif True:
            plotter = plt.scatter
            logscale = False
        else:
            plotter = plt.plot
            logscale = False

        # The python terminal may need to be restarted if you want to change this after running once.
        if save_fig_dir is not None:
            plt.ioff()
        else:
            plt.ion()
        
        for pol in pols:
            if pol == 'hpol':
                key_x, key_y = 'phi_best_h','elevation_best_h'
            elif pol == 'choice':
                key_x, key_y = 'phi_best_choice','elevation_best_choice'
            else:
                key_x, key_y = 'phi_best_v','elevation_best_v'

            if mode == 'rfi':
                eventids_dict = None
            else:
                eventids_dict = {}
                for site in list(roi_dict.keys()):
                    all_event_info = known_pulser_ids[site][pol]

                    for run in numpy.unique(all_event_info['run']):
                        if run not in list(eventids_dict.keys()):
                            eventids_dict[run] = all_event_info[all_event_info['run'] == run]['eventid']
                        else:
                            eventids_dict[run] = numpy.append(eventids_dict[run],all_event_info[all_event_info['run'] == run]['eventid'])


            do = list(ds.roi.keys())
            # do = ['RFI Source 3']
            for roi_index, roi_key in enumerate(list(ds.roi.keys())):
                if roi_key not in do:
                    continue
                try:
                    #Make new desired gridspec'd layout
                    fig_2d = plt.figure(figsize=(20,7))
                    # fig_2d.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

                    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
                    ax1 = fig_2d.add_subplot(gs[:, :-1])
                    ax2 = fig_2d.add_subplot(gs[0, 1])
                    ax3 = fig_2d.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)

                    plt.sca(ax1)
                    fig_2d, ax1 = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True, eventids_dict=eventids_dict, fig=fig_2d, ax=ax1)
                    
                    ax1.set_ylim((-25, 10))
                    ax1.set_xlim((-90, 90))

                    ax1.set_ylabel('Elevation (deg)', fontsize=fontsize)
                    ax1.set_xlabel('Azimuth (From East = 0 deg, North = 90 deg)', fontsize=fontsize)

                    ax1.grid(which='both', axis='both')
                    ax1.minorticks_on()
                    # ax1.grid(b=True, which='major', color='k', linestyle='-')
                    ax1.grid(b=True, which='major', color='tab:gray', linestyle='--',alpha=0.5)
                    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                    if True:
                        roi_eventids_dict = ds.getCutsFromROI(roi_key,load=False,save=False,verbose=False)
                        if eventids_dict is not None:
                            roi_eventids_dict = ds.returnCommonEvents(roi_eventids_dict, eventids_dict)
                    else:
                        roi_eventids_dict = eventids_dict

                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    print('Testing plot for calculating %s and %s'%(key_x,key_y))

                    #Add events for single ROI, counts plotted manually later.
                    _fig, _ax, counts = ds.plotROI2dHist(key_x, key_y, eventids_dict=roi_eventids_dict, return_counts=True, cmap='coolwarm', include_roi=False, fig=None, ax =None)
                    plt.close(_fig)

                    
                    # Bins must be called after plotROI2dHist
                    current_bin_edges_x = ds.current_bin_edges_x
                    current_bin_centers_x = (current_bin_edges_x[:-1] + current_bin_edges_x[1:])/2
                    current_label_x = ds.current_label_x

                    current_bin_edges_y = ds.current_bin_edges_y
                    current_bin_centers_y = (current_bin_edges_y[:-1] + current_bin_edges_y[1:])/2
                    current_label_y  = ds.current_label_y

                    max_x_column = numpy.argmax(numpy.max(counts,axis=0))
                    max_x_value = current_bin_centers_x[max_x_column]
                    #numpy.isin(numpy.max(counts), counts[:,max_x_column])

                    max_y_row = numpy.argmax(numpy.max(counts,axis=1))
                    max_y_value = current_bin_centers_y[max_y_row]
                    
                    '''
                    ATTEMPTING A 2D GAUSSIAN FIT
                    '''

                    #Will want to switch to interpolated grid for actual final plotting, but this is fine for testing.
                    x, y = ds.current_bin_centers_mesh_x, ds.current_bin_centers_mesh_y #Used for calculating, but pcolormesh expects bin edges.

                    x_guess = numpy.mean(ds.roi[roi_key]['phi_best_h'])
                    y_guess = numpy.mean(ds.roi[roi_key]['elevation_best_h'])

                    fit_guess_params =  [x_guess, 0.1, y_guess, 0.2, 0.0]#x0,sigma_x, y0,sigma_y, rho 

                    xy_range_cut = numpy.where(numpy.logical_and(numpy.abs(x - max_x_value) < cut_range , numpy.abs(y - max_y_value) < cut_range).ravel())[0]

                    popt, pcov = curve_fit(bivariateGaus,(x.ravel()[xy_range_cut], y.ravel()[xy_range_cut]),counts.ravel()[xy_range_cut] / (numpy.sum(counts.ravel()[xy_range_cut])*numpy.diff(x,axis=1)[0][0]*numpy.diff(y,axis=0)[0][0]),p0=fit_guess_params) #Only want to use normalize when plotting not fitting.
                    
                    x0          = popt[0]
                    sigma_x     = popt[1]
                    y0          = popt[2]
                    sigma_y     = popt[3]
                    rho         = popt[4]
                    mean = numpy.array([x0,y0])
                    sigma = numpy.array([[sigma_x**2, rho*sigma_x*sigma_y],[rho*sigma_x*sigma_y,sigma_y**2]])

                    ellipse_vertices = parametricCovarianceEllipse(sigma, mean, confidence_integral_value, n=1000)
                    ellipse_path = matplotlib.path.Path(ellipse_vertices)
                    ellipse_area = contourArea(ellipse_path.vertices) #square degrees
                    

                    print('Comparing Initial and Fit Values')
                    
                    print('x0 :         initial %0.4f, \tfit  %0.4f, \tdiff  %0.6f'%(fit_guess_params[0], x0,       x0 - fit_guess_params[0]))
                    print('sigma_x :    initial %0.4f, \tfit  %0.4f, \tdiff  %0.6f'%(fit_guess_params[1], sigma_x,  sigma_x - fit_guess_params[1]))
                    print('y0 :         initial %0.4f, \tfit  %0.4f, \tdiff  %0.6f'%(fit_guess_params[2], y0,       y0 - fit_guess_params[2]))
                    print('sigma_y :    initial %0.4f, \tfit  %0.4f, \tdiff  %0.6f'%(fit_guess_params[3], sigma_y,  sigma_y - fit_guess_params[3]))
                    print('rho :        initial %0.4f, \tfit  %0.4f, \tdiff  %0.6f'%(fit_guess_params[4], rho,      rho - fit_guess_params[4]))
                    print(roi_key + r' 90% Confidence = ' + '%0.4f deg^2'%(ellipse_area))


                    #popt[2] = abs(popt[2]) #I want positive sigma.

                    plot_x_edges = ds.current_bin_edges_x[numpy.abs(ds.current_bin_edges_x - max_x_value) < cut_range]
                    plot_x_edges =  numpy.linspace(min(plot_x_edges), max(plot_x_edges), 500)
                    plot_x_centers = (plot_x_edges[:-1] + plot_x_edges[1:]) / 2
                    
                    plot_y_edges = ds.current_bin_edges_y[numpy.abs(ds.current_bin_edges_y - max_y_value) < cut_range]
                    plot_y_edges =  numpy.linspace(min(plot_y_edges), max(plot_y_edges), 200)
                    plot_y_centers = (plot_y_edges[:-1] + plot_y_edges[1:]) / 2

                    plot_x_edges_mesh, plot_y_edges_mesh = numpy.meshgrid(plot_x_edges, plot_y_edges)
                    plot_x_centers_mesh, plot_y_centers_mesh = numpy.meshgrid(plot_x_centers, plot_y_centers)


                    initial_z = bivariateGaus( (x, y) ,fit_guess_params[0], fit_guess_params[1], fit_guess_params[2], fit_guess_params[3], fit_guess_params[4], scale_factor = (numpy.sum(counts)*numpy.diff(x,axis=1)[0][0]*numpy.diff(y,axis=0)[0][0]), return_2d=True)
                    
                    scale_factor = (numpy.sum(counts)*numpy.diff(x,axis=1)[0][0]*numpy.diff(y,axis=0)[0][0])
                    fit_z = bivariateGaus( (plot_x_centers_mesh, plot_y_centers_mesh) ,popt[0], popt[1], popt[2], popt[3], popt[4], scale_factor = scale_factor, return_2d=True)

                    plt.sca(ax2)
                    im = ax2.pcolormesh(ds.current_bin_edges_mesh_x, ds.current_bin_edges_mesh_y, counts,norm=colors.LogNorm(vmin=0.5, vmax=counts.max()),cmap='coolwarm')
                    # y_modifier = 0.6
                    # x_modifier = 0.2
                    # ax2.set_ylim(y0 - y_modifier*cut_range, y0 + y_modifier*cut_range + 0.5)
                    # ax2.set_xlim(x0 - x_modifier*cut_range, x0 + x_modifier*cut_range)
                    ax2.set_ylim(y0 - 2.5, y0 + 3.5)
                    ax2.set_xlim(x0 - 1.5, x0 + 1.5)


                    try:
                        cbar = plt.colorbar(im, ax=ax2)
                        cbar.set_label('Counts', fontsize=fontsize)
                    except Exception as e:
                        print('Error in colorbar, often caused by no events.')
                        print(e)

                    # plt.xlabel('Azimuth (deg)', fontsize=fontsize)
                    ax2.set_ylabel('Elevation (deg)', fontsize=fontsize)

                    ax2.grid(which='both', axis='both')
                    ax2.minorticks_on()
                    ax2.grid(b=True, which='major', color='k', linestyle='-')
                    ax2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    plt.sca(ax3)
                    im = ax3.pcolormesh(plot_x_edges_mesh, plot_y_edges_mesh, fit_z,norm=colors.LogNorm(vmin=0.5, vmax=fit_z.max()),cmap='coolwarm', label='2D Gaussian Fit')
                    # ax3.plot(ellipse_vertices[:,0],ellipse_vertices[:,1], color=ds.roi_colors[roi_index],label='%0.2f'%(confidence_integral_value*100) + r'% PDF Area = ' + '%0.3f deg^2\nrho = %0.3f'%(ellipse_area,rho))
                    ax3.plot(ellipse_vertices[:,0],ellipse_vertices[:,1], color=ds.roi_colors[roi_index],label='%i'%(confidence_integral_value*100) + r'%' + 'Fit PDF Area\n= %0.2f deg^2'%(ellipse_area))

                    if mode == 'pulsers':
                        ax2.axvline(direction_dict[roi_key]['azimuth_deg'], linewidth=2, color='fuchsia', label='Expected Dir: %0.2f, %0.2f'%(direction_dict[roi_key]['azimuth_deg'], direction_dict[roi_key]['elevation_deg']))
                        ax2.axhline(direction_dict[roi_key]['elevation_deg'], linewidth=2, color='fuchsia')

                        ax2.axvline(popt[0], linewidth=2, color='lime', label='Fit Dir: %0.2f, %0.2f'%(popt[0], popt[2]))
                        ax2.axhline(popt[2], linewidth=2, color='lime')

                        ax3.axvline(direction_dict[roi_key]['azimuth_deg'], linewidth=2, color='fuchsia')
                        ax3.axhline(direction_dict[roi_key]['elevation_deg'], linewidth=2, color='fuchsia')

                        ax3.axvline(popt[0], linewidth=2, color='lime')
                        ax3.axhline(popt[2], linewidth=2, color='lime')


                        plt.sca(ax3)
                        print('Comparing EXPECTED and Fit Values')
                        print('Az :         initial %0.4f, \tfit  %0.4f, \tdiff  %0.6f'%(direction_dict[roi_key]['azimuth_deg'], popt[0],       popt[0] - direction_dict[roi_key]['azimuth_deg']))
                        print('El :         initial %0.4f, \tfit  %0.4f, \tdiff  %0.6f'%(direction_dict[roi_key]['elevation_deg'], popt[2],       popt[2] - direction_dict[roi_key]['elevation_deg']))
                    
                    ax2.legend(title='Counts from\n%s'%roi_key, loc='upper right', fontsize=fontsize-4)
                    ax3.legend(loc = 'upper right', fontsize=fontsize-4)

                    try:
                        cbar = plt.colorbar(im, ax=ax3)
                        cbar.set_label('Counts', fontsize=fontsize)
                    except Exception as e:
                        print('Error in colorbar, often caused by no events.')
                        print(e)

                    ax3.set_xlabel('Azimuth (deg)', fontsize=fontsize)
                    ax3.set_ylabel('Elevation (deg)', fontsize=fontsize)

                    ax3.grid(which='both', axis='both')
                    ax3.minorticks_on()
                    ax3.grid(b=True, which='major', color='k', linestyle='-')
                    ax3.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                    ann = ax1.annotate(roi_key,
                      xy=(x0, y0-5*sigma_y), xycoords='data',
                      xytext=(x0, y0-5*sigma_y-4), textcoords='data',
                      size=fontsize-4, va="center", ha="center",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="-|>",
                                      fc="w"),
                      )


                    #fig_2d.set_tight_layout(True)
                    if save_fig_dir is not None:
                        # fig_2d.set_size_inches(halfsize_fig_dims)
                        fig_2d.subplots_adjust(top=0.93)
                        fig_2d.savefig(os.path.join(save_fig_dir,'%s_spot_%s.pdf'%(roi_key.replace(' ','').lower() , pol)) , dpi=300,transparent=True)

                    if True:
                        # Sanity check
                        #Generate 10000 events using the fit gaussian

                        out = numpy.random.multivariate_normal(mean, sigma, size=100000)
                        in_contour = numpy.array([ellipse_path.contains_point(p) for p in out])
                        percent_in_contour = numpy.sum(in_contour)/len(in_contour)
                        print('Under a MC-based sanity check the percentage of randomly generated events within the given %0.2f CI is %0.4f'%(confidence_integral_value*100, percent_in_contour*100))
                except Exception as e:
                    print('Error in plotting')
                    print('\nError in %s'%inspect.stack()[0][3])
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

