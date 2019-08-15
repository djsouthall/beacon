import numpy
import scipy.spatial
import os
import sys

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.

import matplotlib.pyplot as plt
from pprint import pprint
plt.ion()

def pulserRuns():
    '''
    Returns
    -------
    pulser_runs : numpy.ndarray of ints
        This is the list of known pulser runs as determined by the matching_times.py script.
    '''
    pulser_runs = numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) 
    return pulser_runs

def getTimes(reader):
    '''
    This pulls timing information for each event from the reader object.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.

    Returns
    -------
    times : numpy.ndarray of floats
        The raw_approx_trigger_time values for each event from the Tree.
    subtimes : numpy.ndarray of floats
        The raw_approx_trigger_time_nsecs values for each event from the Tree. 
    trigtimes : numpy.ndarray of floats
        The trig_time values for each event from the Tree.
    '''
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time","trigger_type==2","goff") 
    #ROOT.gSystem.ProcessEvents()
    subtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
    times = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
    trigtimes = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)     
    return times, subtimes, trigtimes



def pulserLocator(reader, nearest_neighbor=10, scale_subtimes=10.0, scale_times=1.0, percent_cut=0.005, plot=True, verbose=True):
    '''
    This uses a nearest neighbour calculation to select the cal pulser events from underneath the
    noise.  Adjust the parameters until it works for you specific run.  

    This obviously will only work for runs with a cal pulser in them.  It will check if the
    selected run is one of the known cal pulser runs, and exit if it is not. 

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.
    nearest_neighbor : int
        This is the the 'order' of the nearest neighbour for which the distances will be calculated.
        Events passing a threshold for closest (nearest_neighbor)th distance will be assumed to be
        in the cal pulser event.  Adjust until works.  (Default is 10).
    scale_subtimes : float
        The two time scales of this run are initially scaled to range from 0 to 1.  Adjusting this
        will make distances in the subtimes direction larger.  Effectively this means that larger 
        this is the less the nearest neighbor favours vertical lines.
    scale_times : 
        The two time scales of this run are initially scaled to range from 0 to 1.  Adjusting this
        will make distances in the times direction larger.  Effectively this means that larger this 
        is the less the nearest neighbor favours horizontal lines.
    percent_cut : float
        The events will be sorted based on nearest neighbor distances.  This sets the cut on that 
        sorted list, selecting the shortest percent_cut percentage of events.  Should be between
        0 and 1.  (Default is 0.005).
    plot : bool
        Enables plotting.

    Returns
    -------
    times : numpy.ndarray of floats
        The raw_approx_trigger_time values for each event from the Tree.
    subtimes : numpy.ndarray of floats
        The raw_approx_trigger_time_nsecs values for each event from the Tree. 
    trigtimes : numpy.ndarray of floats
        The trig_time values for each event from the Tree.
    eventids : numpy.ndarray of ints
        The event ids that the algorithm believes to be part of the most intense pulser strip
        in the time-subtime plot. 
    '''
    try:
        run = reader.run
        if run not in pulserRuns():
            print('WARNING:  The selected run is not in the known list of pulser runs.')

        times, subtimes, trigtimes = getTimes(reader)

        times_scaled       =    scale_times *    (times - min(times))    /    (max(times) - min(times))
        subtimes_scaled    = scale_subtimes * (subtimes - min(subtimes)) / (max(subtimes) - min(subtimes))
        points = numpy.vstack((times_scaled,subtimes_scaled)).T
        tree = scipy.spatial.KDTree(points)
        distance, index = tree.query(points,k=nearest_neighbor)
        distance = distance[:,nearest_neighbor-1]

        eventids = numpy.argsort(distance)[0:int(percent_cut*len(distance))]
        if plot == True:
            fig = plt.figure()
            plt.scatter(times,subtimes,c=distance,marker=',',s=(72./fig.dpi)**2)
            plt.ylabel('Sub times')
            plt.xlabel('Times')
            plt.colorbar()

            fig = plt.figure()
            plt.scatter(times,subtimes,c='b',marker=',',s=(72./fig.dpi)**2)
            plt.scatter(times[eventids],subtimes[eventids],c='r',marker=',',s=(72./fig.dpi)**2)
            plt.ylabel('Sub times')
            plt.xlabel('Times')

            fig = plt.figure()
            plt.scatter(trigtimes,subtimes,c='b',marker=',',s=(72./fig.dpi)**2)
            plt.scatter(trigtimes[eventids],subtimes[eventids],c='r',marker=',',s=(72./fig.dpi)**2)
            plt.ylabel('Sub Times')
            plt.xlabel('Trig Times')

        return times, subtimes, trigtimes, eventids
    except Exception as e:
        print('Error in pulserLocator.')
        print(e)
        return 0


def getClockCorrection(reader, nearest_neighbor=10, scale_subtimes=10.0, scale_times=1.0, slope_bound=0.001, percent_cut=0.001, nominal_clock_rate=31.25e6, lower_rate_bound=31.2e6, upper_rate_bound=31.3e6, plot=True, verbose=True):
    '''
    This will attempt fit a pulser line within the selected run with a line, and then adjust the 
    clock rate to obtain the clock rate that results in a consistent cal pulser line.

    This obviously will only work for runs with a cal pulser in them.  It will check if the
    selected run is one of the known cal pulser runs, and exit if it is not. 

    This uses the pulserLocator function to identify the points in the pulser.

    With the identified points, a linear fit is made and adjusted using the bisection method until
    the clock rate makes the line flat.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.
    nearest_neighbor : int
        This is the the 'order' of the nearest neighbour for which the distances will be calculated.
        Events passing a threshold for closest (nearest_neighbor)th distance will be assumed to be
        in the cal pulser event.  Adjust until works.  (Default is 10).
    scale_subtimes : float
        The two time scales of this run are initially scaled to range from 0 to 1.  Adjusting this
        will make distances in the subtimes direction larger.  Effectively this means that larger 
        this is the less the nearest neighbor favours vertical lines.
    scale_times : 
        The two time scales of this run are initially scaled to range from 0 to 1.  Adjusting this
        will make distances in the times direction larger.  Effectively this means that larger this 
        is the less the nearest neighbor favours horizontal lines.
    percent_cut : float
        The events will be sorted based on nearest neighbor distances.  This sets the cut on that 
        sorted list, selecting the shortest percent_cut percentage of events.  Should be between
        0 and 1.  (Default is 0.005).
    slope_bound : float
        This sets the slope below which the bisection method will terminate.
    plot : bool
        Enables plotting.
    nominal_clock_rate : float
        This is the expected clockrate given in Hz.
    lower_rate_bound : float
        This is the lower bound of the bisection method (which adjusts the clock rate).
    upper_rate_bound : float
        This is the upper bound of the bisection method (which adjusts the clock rate).

    Returns
    -------
    clock_rate : float
        The final adjusted clock rate in Hz.
    times : numpy.ndarray of floats
        The raw_approx_trigger_time values for each event from the Tree.
    subtimes : numpy.ndarray of floats
        The raw_approx_trigger_time_nsecs values for each event from the Tree. 
    trigtimes : numpy.ndarray of floats
        The trig_time values for each event from the Tree.  Note these are NOT adjusted by the new
        clock rate. 
    eventids : numpy.ndarray of ints
        The event ids that the algorithm believes to be part of the most intense pulser strip
        in the time-subtime plot. 
    '''
    try:
        run = reader.run
        times, subtimes, trigtimes, eventids = pulserLocator(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, percent_cut=percent_cut, plot=plot, verbose=verbose)

        clock_rate = nominal_clock_rate
        slope = 1000000.0
        clock_rate_upper = upper_rate_bound
        clock_rate_lower = lower_rate_bound
        pulser_x = times[eventids]
        max_iter = 1000
        i = 0
        roll_offset = 0
        if verbose == True:
            print('Attempting bisection method.')
        while abs(slope) > slope_bound and i <= max_iter:
            if verbose == True:
                print(clock_rate)
            i += 1
            pulser_y = (trigtimes[eventids] + roll_offset)%clock_rate
            max_diff = numpy.max(numpy.diff(pulser_y))
            if max_diff > (max((trigtimes + roll_offset)%clock_rate) - min((trigtimes + roll_offset)%clock_rate))/2.0: #In case the points get split
                roll_offset += (max((trigtimes + roll_offset)%clock_rate) - min((trigtimes + roll_offset)%clock_rate))/2.0
                pulser_y = (trigtimes[eventids] + roll_offset)%clock_rate
            coeff = numpy.polyfit(pulser_x,pulser_y,1)
            slope = coeff[0]

            if slope > 0:
                clock_rate_lower = clock_rate
                clock_rate = (clock_rate_upper + clock_rate)/2.0 #Not an even average because it expects the nominal clock rate 
            else:
                clock_rate_upper = clock_rate
                clock_rate = (clock_rate_lower + clock_rate)/2.0 #Not an even average because it expects the nominal clock rate 

        if i == max_iter:
            print('Bisection clock correction timed out.  Returning on run %i'%run)
            return 0
        else:
            adjusted_trigtimes = trigtimes%clock_rate
            coeff = numpy.polyfit(times[eventids],adjusted_trigtimes[eventids],1)
            p = numpy.poly1d(coeff)
            x = numpy.arange(min(times[eventids]),max(times[eventids]),100)
            y = p(x)
            poly_label = '$y = ' + str(['%0.9g x^%i'%(coefficient,order_index) for order_index,coefficient in enumerate(coeff[::-1])]).replace("'","").replace(', ',' + ').replace('[','').replace(']','')+ '$' #Puts the polynomial into a str name.

            if plot == True:
                fig = plt.figure()
                plt.scatter(times,adjusted_trigtimes,c='b',marker=',',s=(72./fig.dpi)**2)
                plt.scatter(times[eventids],adjusted_trigtimes[eventids],c='r',marker=',',s=(72./fig.dpi)**2)
                plt.ylabel('Trig times')
                plt.xlabel('Times')

                plt.plot(x,y,label='Linear Fit ' + poly_label, c = 'y')
                plt.legend()
                plt.title('Run: %i\tAdjusted Clock Rate = %f MHz'%(run,clock_rate/1e6))


            print('Run: %i\tAdjusted Clock Rate = %f MHz'%(run,clock_rate/1e6))
            return clock_rate, times, subtimes, trigtimes, eventids

    except Exception as e:
        print('Error in getClockCorrection.  Skipping run %i'%run)
        print(e)
        return 0

#Below is known good percentages to identify pulser points.
percent_cuts = {'run792':0.0011,
                'run793':0.003}

if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    nearest_neighbor = 10 #Adjust until works.
    scale_subtimes = 10.0 #The larger this is the the less the nearest neighbor favors vertical lines.
    scale_times = 1.0  #The larger this is the the less the nearest neighbor favors horizontal lines.
    slope_bound = 1.0e-9
    percent_cut = 0.003
    nominal_clock_rate = 31.25e6
    lower_rate_bound = 31.2e6 #Don't make the bounds too large or the bisection method will overshoot and roll over.
    upper_rate_bound = 31.3e6 #Don't make the bounds too large or the bisection method will overshoot and roll over.
    plot = True
    verbose = False

    all_adjusted_clock_rates = []
    good_runs = numpy.ones_like(runs,dtype=bool)
    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        try:
            clock_rate, times, subtimes, trigtimes, eventids = getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cuts['run%i'%run], nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
            all_adjusted_clock_rates.append(clock_rate)
        except Exception as e:
            print('Error in main clock correction loop.')
            print(e)
            good_runs[run_index] = False

    all_adjusted_clock_rates = numpy.array(all_adjusted_clock_rates)
    print('Adjusted clockrates are:')
    for i in all_adjusted_clock_rates:
        print(i)
    plt.figure()
    cut = ~numpy.logical_or(all_adjusted_clock_rates == lower_rate_bound,all_adjusted_clock_rates == upper_rate_bound)
    print('Mean adjusted clock rate = %f'%numpy.mean(all_adjusted_clock_rates[cut]))
    plt.plot(runs[good_runs][cut],all_adjusted_clock_rates[cut]/1e6)
    plt.axhline(nominal_clock_rate/1e6,c='r',linestyle='--',label='Nominal Value, %f MHz'%(nominal_clock_rate/1e6))
    plt.ylabel('Adjusted Clock Rate (MHz)')
    plt.xlabel('Run Number')
    plt.legend()
    '''
    This was a plot for Kaeli's poster.
    fig = plt.figure()
    adjusted_trigtimes = trigtimes%clock_rate
    adjusted_trigtimes = adjusted_trigtimes/(max(adjusted_trigtimes))
    plt.scatter(times-times[0],adjusted_trigtimes,c='b',marker=',',s=(2*72./fig.dpi)**2)
    #plt.scatter(times[eventids],adjusted_trigtimes[eventids],c='r',marker=',',s=(72./fig.dpi)**2)
    plt.ylabel('Sub-Second Trigger Time',fontsize=16)
    plt.xlabel('Seconds from Start of Run',fontsize=16)

    plt.xlim((7300,8400))
    plt.ylim((0,0.4))
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)

    #plt.legend()
    #plt.title('Run: %i - Adjusted Clock Rate = %f MHz'%(run,clock_rate/1e6),fontsize=20)
    '''