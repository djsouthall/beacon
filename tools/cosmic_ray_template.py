'''
The purpose of this script is to provide the code used to generate CR signals.  Eventually this should entail actual CR 
models/physics, however as an initial starting place a bipolar delta function convolved with the
system response will be used. 
'''

import sys
import os
import inspect

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info

import numpy
import matplotlib.pyplot as plt

datapath = os.environ['BEACON_DATA']

plt.ion()

class CosmicRayGenerator():
    """
    Given a particular model in init, this will produce CR's of specific Energies.

    The models are intended to be given in Electric field such that they can be convolved with the system response and 
    scaled to produce adu signals.

    This functionality is anticipated to be filled out in the future.  Early versions    will simply produce a bipolar 
    signal, and convolve it with the system response.  The bipolar signal will be scaled to have visible magnitude above
    normal noise levels. 

    Parameters
    ----------
    model : str
        This is the model you wish to use.  Currently the options include:
        'bi-delta' : A bi polar delta function convolved with the system response.
    """

    def __init__(self, model='bi-delta'):
        try:
            self.accepted_model_list = ['bi-delta']
            if not type(model) == str:
                print('ERROR')
                print('Model parameter incorrectly given.  Must be a string, not a %s'%str(type(model)))
                return
            elif model not in self.accepted_model_list:
                print('ERROR')
                print('Given model [%s] no in list of allowable models:\n'%model)
                print(self.accepted_model_list)
                return
            else:
                self.model = model


            #One-time preparation required for each model can be performed or called below.
            if self.model == 'bi-delta':
                self.system_response = numpy.zeros(1000)#filler.  This is where I should load the sytem response.

                '''
                phase_response.py is the only response data I remember.  I need to refresh myself the meaning of this data
                and if it is the system impulse response I want.  Additionally do we have any measure of antenna response?
                ''' 

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        

    def eFieldGenerator(self,t_ns,t_offset=0):
        '''
        For a given set of time data this will produce a signal for the set model.

        Parameters
        ----------
        t_ns : array of float
            This should be the time series of the electric field to be output.  Expected in ns.
        t_offset : float
            This is the time offset within the given time serious the signal should start.  What "start" means will be 
            model dependant likely.  Descriptions of meanings for each model are:
            'bi-delta' : The initial rise time value will occur at the time step closest to this offset (rounded up).
        '''
        if self.model == 'bi-delta':
            #Generate "Electric field" portion of bipolar signal (signal before response)
            extent_ns = 5.0#ns How long the entire bipolar signal will take to return to no signal (roughly).
            half_extent_index = int(numpy.ceil(extent_ns/(2*(t_ns[1] - t_ns[0])))) #The number of indices corresponding to half of the extent.  I.e. the extent in time (indices) of each pol of the bipolar signal.
            print(half_extent_index)
            efield = numpy.zeros_like(t_ns)
            start_index = numpy.where(t_ns >= t_offset)[0][0]
            efield[start_index:start_index+half_extent_index] = 1.0
            efield[start_index + half_extent_index:start_index+2*half_extent_index] = -1.0

            #Use impulse response to get signal.

        else:
            print('Model not yet programmed in function: eFieldGenerator')

        return efield


if __name__ == '__main__':
    try:
        plt.close('all')
        #Get timing info from real BEACON data for testing.

        run = 1509
        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        eventid = known_pulser_ids['run%i'%run]['hpol'][0]
        reader = Reader(datapath,run)
        reader.setEntry(eventid)
        test_t = reader.t()
        test_pulser_adu = reader.wf(0)




        #Creating test signal
        cr_gen = CosmicRayGenerator(model='bi-delta')

        test_e = cr_gen.eFieldGenerator(test_t,t_offset=500.0)


        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(test_t,test_pulser_adu,label='Pulser Signal')
        plt.ylabel('E (adu)')
        plt.xlabel('t (ns)')

        plt.legend()
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        plt.subplot(2,1,2)
        plt.plot(test_t,test_e,label='Test CR Signal')
        plt.scatter(test_t,test_e,c='r')
        plt.ylabel('E (adu)')
        plt.xlabel('t (ns)')

        plt.legend()
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
