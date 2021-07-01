'''
This is a script plot the current parameters for the range of runs given, and save them to an output directory in the
given path.  
'''
import os
import sys
import inspect
import warnings
import datetime
import numpy
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_slicer import dataSlicer
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']

if False:
    1
    # def returnTriggerThresholds(reader, expected_max_beam=19, plot=False):
    #   '''
    #   Given a reader object this will extract the trigger thresholds for each beam.

    #   If expected_max_beam = None then this will attempt to access beams until the reader returns an error.
    #   '''
    #   failed = False
    #   beam_index = 0
    #   while failed == False:
    #     try:
    #       N = reader.status_tree.Draw("trigger_thresholds[%i]:Entry$"%beam_index,"","goff")

    #       if beam_index == 0:
    #         thresholds = numpy.frombuffer(reader.status_tree.GetV1(), numpy.dtype('float64'), N)
    #         eventids = numpy.frombuffer(reader.status_tree.GetV2(), numpy.dtype('float64'), N).astype(int)
    #       else:
    #         thresholds = numpy.vstack((thresholds,numpy.frombuffer(reader.status_tree.GetV1(), numpy.dtype('float64'), N)))
    #       if beam_index is not None:
    #         if beam_index == expected_max_beam:
    #           failed=True
    #       beam_index += 1
    #     except:
    #       failed = True

    #   if plot:
    #     plt.figure()
    #     plt.title('Trigger Thresholds')
    #     for beam_index, t in enumerate(thresholds):
    #       plt.plot(eventids, t, label='Beam %i'%beam_index)
    #     plt.xlabel('EntryId / eventid')
    #     plt.ylabel('Power Sum (arb)')

    #   return thresholds

    # def returnBeamScalers(reader, expected_max_beam=19, plot=False):
    #   '''
    #   Given a reader object this will extract the beam scalers for each beam as they are presented on monutau.

    #   If expected_max_beam = None then this will attempt to access beams until the reader returns an error.
    #   '''
    #   try:
    #     failed = False
    #     beam_index = 0
    #     while failed == False:
    #       try:
    #         N = reader.status_tree.Draw("beam_scalers[0][%i]/10.:Entry$"%beam_index,"","goff")

    #         if beam_index == 0:
    #           beam_scalers = numpy.frombuffer(reader.status_tree.GetV1(), numpy.dtype('float64'), N)
    #           eventids = numpy.frombuffer(reader.status_tree.GetV2(), numpy.dtype('float64'), N).astype(int)
    #         else:
    #           beam_scalers = numpy.vstack((beam_scalers,numpy.frombuffer(reader.status_tree.GetV1(), numpy.dtype('float64'), N)))
    #         if beam_index is not None:
    #           if beam_index == expected_max_beam:
    #             failed=True
    #         beam_index += 1
    #       except:
    #         failed = True

    #     if plot:
    #       plt.figure()
    #       plt.title('Beam Scalers')
    #       for beam_index, t in enumerate(beam_scalers):
    #         plt.plot(eventids, t, label='Beam %i'%beam_index)
    #       plt.xlabel('EntryId / eventid')
    #       plt.ylabel('Hz')

    #     return beam_scalers
    #   except Exception as e:
    #     print('\nError in %s'%inspect.stack()[0][3])
    #     print(e)
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)


    # def returnGlobalScalers(reader, plot=False):
    #   '''
    #   This will return global scalers global_scalers[0], global_scalers[1]/10, global_scalers[2]/10 as they are
    #   read in and presented on monutau.
    #   '''
    #   try:
    #     N = reader.status_tree.Draw("global_scalers[0]:global_scalers[1]/10.:global_scalers[2]/10.:trigger_thresholds[%i]:Entry$"%beam_index,"","goff")

    #     global_scalers_0 = numpy.frombuffer(reader.status_tree.GetV1(), numpy.dtype('float64'), N)
    #     global_scalers_1 = numpy.frombuffer(reader.status_tree.GetV2(), numpy.dtype('float64'), N) 
    #     global_scalers_2 = numpy.frombuffer(reader.status_tree.GetV3(), numpy.dtype('float64'), N)
    #     eventids = numpy.frombuffer(reader.status_tree.GetV4(), numpy.dtype('float64'), N).astype(int)

    #     if plot:
    #       plt.figure()
    #       plt.plot(eventids, global_scalers_0, label = 'Fast') # Labels taken from Monutau.
    #       plt.plot(eventids, global_scalers_0, label = 'Slow Gated') # Labels taken from Monutau.
    #       plt.plot(eventids, global_scalers_0, label = 'Slow') # Labels taken from Monutau.
    #       plt.ylabel('Hz')
    #       plt.xlabel('EntryId / eventid')
    #       plt.legend()

    #     return global_scalers_0, global_scalers_1, global_scalers_2, eventids
    #   except Exception as e:
    #     print('\nError in %s'%inspect.stack()[0][3])
    #     print(e)
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)

    # def returnTriggerInfo(reader):
    #   '''
    #   This will return global scalers global_scalers[0], global_scalers[1]/10, global_scalers[2]/10 as they are
    #   read in and presented on monutau.
    #   '''
    #   try:
    #     N = reader.head_tree.Draw("triggered_beams:beam_power:Entry$","","goff")

    #     # The below attempts return nan or inf.  Unsure why. Using a workaround loop below.  
    #     # triggered_beams = numpy.log2(numpy.frombuffer(reader.status_tree.GetV1(), numpy.dtype('float64'), N).astype(int))
    #     # beam_power = numpy.frombuffer(reader.status_tree.GetV2(), numpy.dtype('float64'), N).astype(int)
    #     eventids = numpy.frombuffer(reader.status_tree.GetV3(), numpy.dtype('float64'), N).astype(int)

    #     triggered_beams = numpy.zeros(reader.N())
    #     beam_power = numpy.zeros(reader.N())
    #     for eventid in range(reader.N()):
    #       reader.setEntry(eventid)
    #       triggered_beams[eventid] = int(reader.header().triggered_beams)
    #       beam_power[eventid] = int(reader.header().beam_power)

    #     return numpy.log2(triggered_beams).astype(int), beam_power.astype(int), eventids
    #   except Exception as e:
    #     print('\nError in %s'%inspect.stack()[0][3])
    #     print(e)
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)


if __name__=="__main__":
  outpath_made = False
  if False:
    outpath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'figures', 'beam_rate_plots_' + str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-'))
    matplotlib.use('Agg')
    os.mkdir(outpath)
    outpath_made = True
  else:
    outpath = None
    plt.ion()
  plt.close('all')


  #Main Control Parameters
  runs = numpy.arange(1643,1645)#numpy.arange(1643,1729)
  #runs = runs[runs != 1663]
  figsize = (16,9)
  dpi = 108*4

  # Other Parameters
  time_delays_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
  impulsivity_dset_key = time_delays_dset_key
  map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_30-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_180-scope_allsky'

  trigger_types = [2]

  n_phi = int(map_direction_dset_key.split('-n_phi_')[-1].split('-')[0])
  range_phi_deg = (float(map_direction_dset_key.split('-min_phi_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_phi_')[-1].split('-')[0].replace('neg','-')))
  n_theta = int(map_direction_dset_key.split('-n_theta_')[-1].split('-')[0])
  range_theta_deg = (float(map_direction_dset_key.split('-min_theta_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_theta_')[-1].split('-')[0].replace('neg','-')))

  lognorm = True
  cmap = 'binary'#'YlOrRd'#'binary'#'coolwarm'

  try:

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, remove_incomplete_runs=True,\
        curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
        cr_template_n_bins_h=1000,cr_template_n_bins_v=1000,\
        impulsivity_n_bins_h=1000,impulsivity_n_bins_v=1000,\
        time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-200,max_time_delays_val=200,\
        std_n_bins_h=200,std_n_bins_v=200,max_std_val=None,\
        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=None,\
        n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg)

    plot_param_pairs = [['phi_best_h', 'elevation_best_h'], ['phi_best_h', 'triggered_beams'], ['elevation_best_h', 'triggered_beams']]

    triggered_beams = ds.concatenateParamDict(ds.getDataFromParam(ds.getEventidsFromTriggerType(),'triggered_beams'))
    beams, n_trig_per_beam = numpy.unique(triggered_beams, return_counts=True)
    if False:
        max_beam_output = scipy.stats.mode(ds.concatenateParamDict(ds.getDataFromParam(ds.getEventidsFromTriggerType(),'triggered_beams')))
        max_beam = max_beam_output[0][0]
        max_beam_counts = max_beam_output[0][0]
        ds.addROI('beam %i'%max_beam, {'triggered_beams': [max_beam - 0.5, max_beam + 0.5]})
    elif True:
        selected_beam = 1
        ds.addROI('beam %i'%selected_beam, {'triggered_beams': [selected_beam - 0.5, selected_beam + 0.5]})
    else:
        for beam in range(20):
            if n_trig_per_beam[beams == beam] > 0:
                ds.addROI('beam %i'%beam, {'triggered_beams': [beam - 0.5, beam + 0.5]})

    for key_x, key_y in plot_param_pairs:
        print('Generating %s plot'%(key_x + ' vs ' + key_y))
        # if key_x == 'phi_best_h' and key_y == 'elevation_best_h':
        #     fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=True)
        #     fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=False)
        # else:
        #     fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm)
        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm)
    # fig.set_size_inches(figsize[0], figsize[1])
    # plt.tight_layout()
    # fig.savefig(os.path.join(outpath,key_x + '-vs-' + key_y + '.png'),dpi=dpi)


    reader = ds.data_slicers[0].reader
    # reader.header().Dump()
    thresholds = reader.returnTriggerThresholds(expected_max_beam = 19, plot=True)
    beam_scalers = reader.returnBeamScalers(expected_max_beam = 19, plot=True)

    if True:
        triggered_beams, beam_power, eventids = reader.returnTriggerInfo()
        eventid = 60123
        print(reader.run, eventid)
        print('triggered_beams[%i] = '%eventid + str(triggered_beams[eventid]))
        print('beam_power[%i] = '%eventid + str(beam_power[eventid]))

    current_bin_edges, label = ds.data_slicers[0].getSingleParamPlotBins('triggered_beams', ds.getEventidsFromTriggerType(), verbose=False)
    plt.figure()
    plt.hist(triggered_beams, bins = current_bin_edges)
    plt.xlabel('Beam Trigger')
  except Exception as e:
    print(e)
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


