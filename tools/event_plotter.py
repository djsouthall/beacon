
import numpy
import os
import sys

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.

import matplotlib.pyplot as plt
from pprint import pprint
plt.ion()


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = 792 #Selects which run to examine
    eventid = None#numpy.array([1,2,3]) #If None then a random event id is selected. Can be array of eventids as well.
    reader = Reader(datapath,run)
    verbose = True

    # this is a random event
    if type(eventid) == None:
        eventid = numpy.array([numpy.random.randint(reader.N())])
    elif type(eventid) == int:
        eventid = numpy.array([eventid])
    elif type(eventid) == list:
        eventid = numpy.array(eventid)
    elif type(eventid) == numpy.ndarray:
        pass
    else:
        print('event id not set in valid way, setting to random')
        eventid = numpy.array([numpy.random.randint(reader.N())])


    for eid in eventid:
        reader.setEntry(eid) 

        ## dump the headers and status, just to show they're there
        if verbose == True:
            print('\nReader:')
            pprint(tools.interpret.getReaderDict(reader))
            print('\nHeader:')
            pprint(tools.interpret.getHeaderDict(reader))
            print('\nStatus:')
            pprint(tools.interpret.getStatusDict(reader))

        reader.header().Dump(); 
        reader.status().Dump(); 
        #print reader.N() 

        # plot all waveforms
        plt.figure()
        for i in range(8): 
            if i == 0:
                ax = plt.subplot(2,4,i+1); 
                plt.plot(reader.t(), reader.wf(i))
            else:
                plt.subplot(2,4,i+1,sharex = ax, sharey =ax ); 
                plt.plot(reader.t(), reader.wf(i))
            if i in [0,4]:
                plt.ylabel('V (adu)')
            if i > 3:
                plt.xlabel('t (arb)')
        plt.suptitle('Run %i, Event %i'%(run,eid))
