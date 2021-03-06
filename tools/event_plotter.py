
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
    printCredit()
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = 1509 #Selects which run to examine
    eventids = numpy.array([2401])#numpy.array([90652,90674,90718,90766,90792,91019,91310])


    for eventid in eventids:
        #eventid = None#numpy.array([1,2,3]) #If None then a random event id is selected. Can be array of eventids as well.
        
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
        
        eventid = [2401]
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
            for i in range(4): 
                if i == 0:
                    ax = plt.subplot(4,1,i+1); 
                    plt.plot(reader.t()-5200, reader.wf(2*i))
                else:
                    plt.subplot(4,1,i+1,sharex = ax, sharey =ax ); 
                    plt.plot(reader.t()-5200, reader.wf(2*i))
                if i in [0,1,2,3]:
                    plt.ylabel('V (adu)')
                if i == 3:
                    plt.xlabel('t (ns)')
                plt.suptitle('Run %i, Event %i, Hpol'%(run,eid))
                plt.xlim(0,600)
                plt.minorticks_on()
                plt.grid(b=True,which='major', color='k', linestyle='-')
                plt.grid(b=True,which='minor', color='tab:gray',linestyle='--', alpha=0.5)
