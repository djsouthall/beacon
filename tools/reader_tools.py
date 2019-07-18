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


def getTimes(reader):
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time","trigger_type==2","goff") 
    #ROOT.gSystem.ProcessEvents()
    subtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
    times = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
    trigtimes = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)     
    return times, subtimes,trigtimes


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([792])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    nearest_neighbor = 10 #Adjust until works.
    scale_subtimes = 10.0 #The larger this is the the less the nearest neighbor favors vertical lines.
    scale_times = 1.0  #The larger this is the the less the nearest neighbor favors horizontal lines.
    slope_bound = 0.001
    percent_cut = 0.001
    plot = True
    nominal_clock_rate = 31.25e6
    lower_rate_bound = 31.2e6
    upper_rate_bound = 31.3e6

    all_adjusted_clock_rates = []
    good_runs = numpy.ones_like(runs,dtype=bool)
    for run_index, run in enumerate(runs):
        try:
            reader = Reader(datapath,run)

            times, subtimes, trigtimes = getTimes(reader)

            times_scaled       =    scale_times *    (times - min(times))    /    (max(times) - min(times))
            subtimes_scaled    = scale_subtimes * (subtimes - min(subtimes)) / (max(subtimes) - min(subtimes))
            points = numpy.vstack((times_scaled,subtimes_scaled)).T
            tree = scipy.spatial.KDTree(points)
            distance,index = tree.query(points,k=nearest_neighbor)
            distance = distance[:,nearest_neighbor-1]

            cut1 = numpy.argsort(distance)[0:int(percent_cut*len(distance))]
            if plot == True:
                fig = plt.figure()
                plt.scatter(times,subtimes,c=distance,marker=',',s=(72./fig.dpi)**2)
                plt.ylabel('Sub times')
                plt.xlabel('Times')
                plt.colorbar()

                fig = plt.figure()
                plt.scatter(times,subtimes,c='b',marker=',',s=(72./fig.dpi)**2)
                plt.scatter(times[cut1],subtimes[cut1],c='r',marker=',',s=(72./fig.dpi)**2)
                plt.ylabel('Sub times')
                plt.xlabel('Times')

            clock_rate = nominal_clock_rate
            slope = 1000000.0
            clock_rate_upper = upper_rate_bound
            clock_rate_lower = lower_rate_bound
            pulser_x = times[cut1]
            max_iter = 1000
            i = 0
            roll_offset = 0

            while abs(slope) > slope_bound and i <= max_iter:
                #print(clock_rate)
                i += 1
                pulser_y = (trigtimes[cut1] + roll_offset)%clock_rate
                max_diff = numpy.max(numpy.diff(pulser_y))
                if max_diff > (max((trigtimes + roll_offset)%clock_rate) - min((trigtimes + roll_offset)%clock_rate))/2.0: #In case the points get split
                    roll_offset += (max((trigtimes + roll_offset)%clock_rate) - min((trigtimes + roll_offset)%clock_rate))/2.0
                    pulser_y = (trigtimes[cut1] + roll_offset)%clock_rate
                coeff = numpy.polyfit(pulser_x,pulser_y,1)
                slope = coeff[0]

                if slope > 0:
                    clock_rate_lower = clock_rate
                    clock_rate = (clock_rate_upper + clock_rate)/2.0
                else:
                    clock_rate_upper = clock_rate
                    clock_rate = (clock_rate_lower + clock_rate)/2.0

            if i == max_iter:
                print('Bisection clock correction timed out.  Skipping run %i'%run)
                good_runs[run_index] = False
            else:
                adjusted_trigtimes = trigtimes%clock_rate
                coeff = numpy.polyfit(times[cut1],adjusted_trigtimes[cut1],1)
                p = numpy.poly1d(coeff)
                x = numpy.arange(min(times[cut1]),max(times[cut1]),100)
                y = p(x)
                poly_label = '$y = ' + str(['%0.9g x^%i'%(coefficient,order_index) for order_index,coefficient in enumerate(coeff[::-1])]).replace("'","").replace(', ',' + ').replace('[','').replace(']','')+ '$' #Puts the polynomial into a str name.

                if plot == True:
                    fig = plt.figure()
                    plt.scatter(times,adjusted_trigtimes,c='b',marker=',',s=(72./fig.dpi)**2)
                    plt.scatter(times[cut1],adjusted_trigtimes[cut1],c='r',marker=',',s=(72./fig.dpi)**2)
                    plt.ylabel('Sub times')
                    plt.xlabel('Times')

                    plt.plot(x,y,label='Linear Fit ' + poly_label, c = 'y')
                    plt.legend()
                    plt.title('Run: %i\tAdjusted Clock Rate = %f MHz'%(run,clock_rate/1e6))


                print('Run: %i\tAdjusted Clock Rate = %f MHz'%(run,clock_rate/1e6))
                all_adjusted_clock_rates.append(clock_rate)
        except Exception as e:
            print('Error.  Skipping run %i'%run)
            good_runs[run_index] = False

    all_adjusted_clock_rates = numpy.array(all_adjusted_clock_rates)
    print('Mean adjusted clock rate = %f'%numpy.mean(all_adjusted_clock_rates))

    plt.figure()
    cut = ~numpy.logical_or(all_adjusted_clock_rates == lower_rate_bound,all_adjusted_clock_rates == upper_rate_bound)
    print('Mean adjusted clock rate = %f'%numpy.mean(all_adjusted_clock_rates[cut]))
    plt.plot(runs[good_runs][cut],all_adjusted_clock_rates[cut]/1e6)
    plt.axhline(nominal_clock_rate/1e6,c='r',linestyle='--',label='Nominal Value, %f MHz'%(nominal_clock_rate/1e6))
    plt.ylabel('Adjusted Clock Rate (MHz)')
    plt.xlabel('Run Number')
    plt.legend()