import numpy
import scipy.spatial
import scipy.signal
import os
import sys
import glob
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
plt.ion()

def readerFieldFox(csv_file,header=17):
    '''
    Reads a csv file created by the FieldFox, removing the header and ignoring the last line.

    Paramters
    ---------
    csv_file : str
        The path to a particular csv file produced by the field fox.

    Returns
    -------
    x : numpy.ndarray of floats
        An array containing the 'x' values of the data (typically frequencies given in Hz).
    y : numpy.ndarray of floats
        An array containing the 'y' values of the data (typically Log Magnitude/Linear/VSWR/Phase).
    '''
    csv_reader = csv.reader(open(csv_file),delimiter=',')
    for n in range(header):
        next(csv_reader)
    x = []
    y = []
    for row in csv_reader:
        if len(row) == 2:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return numpy.array(x), numpy.array(y)


if __name__ == '__main__':
    plt.close('all')
    in_dir = '/home/dsouthall/Projects/Beacon/beacon/data/ff'
    infiles = glob.glob(in_dir + '/*')

    plt.figure()

    for infile in infiles:
        if 'LOG' in infile:
            freqs, mag = readerFieldFox(infile)
            plt.plot(freqs/1e6,mag,label = infile.split('/')[-1].replace('.csv',''))
    plt.legend(fontsize=16)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('dB',fontsize=16)
    plt.xlabel('Freqs (MHz)',fontsize=16)
