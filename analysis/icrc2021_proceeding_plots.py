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
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.config_reader import configSchematicPlotter
import matplotlib
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']


if __name__=="__main__":
    outpath_made = False
    if True:
        outpath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'figures', 'icrc2021_plots_' + str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-'))
        matplotlib.use('Agg')
        os.mkdir(outpath)
        outpath_made = True
    else:
        outpath = None
        plt.ion()

    plt.close('all')

    en_figsize = 2.5*numpy.array((2.95,2.5))#(16,16)
    eu_figsize = 2.5*numpy.array((2.95,2.5))#(16,9)

    dpi = 108*4
    extension = '.pdf'




    #Schematic Figure:
    if True:
        '''
        This is the figure of the array layout.  
        '''
        json_path = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'config','rtk-gps-day3-june22-2021.json')
        figs, axs, names = configSchematicPlotter(deploy_index = json_path,en_figsize=en_figsize, eu_figsize=eu_figsize, antenna_scale_factor=5)
        
        if outpath_made:
            for index, fig in enumerate(figs):
                plt.tight_layout()
                fig.savefig(os.path.join(outpath,names[index] + extension),dpi=dpi, bbox_inches = 'tight',pad_inches = 0)

        




