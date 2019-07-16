import numpy
import os
import sys

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import analysis.tools.interpret #Must be imported before matplotlib or else plots don't load.

import matplotlib.pyplot as plt
from pprint import pprint
plt.ion()


def getTimes(reader):
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time","trigger_type==2","goff") 
    #ROOT.gSystem.ProcessEvents()
    subtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N) 
    times = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
    return times, subtimes


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = 792 #Selects which run to examine
    reader = Reader(datapath,run)

    times, subtimes = getTimes(reader)
    fig = plt.figure()
    plt.scatter(times,subtimes,marker=',',s=(72./fig.dpi)**2)