'''
This is is meant to take in and intrerpet data taken on the roof of ERC.
'''
import numpy
import os
import sys
from pytz import timezone,utc
from datetime import datetime
from pprint import pprint
import glob
import scipy
import scipy.signal

sys.path.append(os.environ['BEACON_INSTALL_DIR']) 
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.interpret import getReaderDict, getHeaderDict, getStatusDict #Must be imported before matplotlib or else plots don't load.
from tools.read_files import readRigol

import matplotlib.pyplot as plt
plt.ion()


def skyNoiseDominance(freqs,complex_z, lna_temp,z_0 = 50.0,ground_temp = 0.0, ime = None):
    '''
    Calculates the Sky Noise Dominance (SND) as defined by the first source below using 
    the Impedance Mismatch Efficiency (IME) (also defined in the first source), and the 
    sky noise model given in the second source below. 

    Sources
    -------
    Source 1
        Paper : "A wide-band, active antenna system for long wavelength radio astronomy"
        Authors : Brian Hicks, agini Paravastu-Dalal, ... etc.
        url : https://arxiv.org/abs/1210.0506
    Source 2
        Paper : "Design and Evaluation of an Active Antenna for a 29–47 MHz Radio Telescope Array"
        Authors : S.W. Ellingson et al.
        url : http://www.phys.unm.edu/~lwa/memos/memo/lwa0060.pdf

    Parameters
    ----------
    freqs : numpy.ndarray of floats
        The frequencies for corresponding to the points used to calculated the SND.
    complex_z : numpy.ndarray of complex floats
        The complex impedances used to calculate SND.  If ime is not None, then these are not used
        and the ime input is used instead.
    lna_temp : float
        The temperature of the LNA in the SND calculation.
    z_0 : float, optional
        The input impedance, typically 50 Ohms.
    ground_temp : flaot, optional
        The ground noise temperature.  This is added to the sky noise.  Should be given in K.
    ime : numpy.ndarray of floats
        This is the impedance mismatch efficiency.  Typically this is calculated internall from complex_z,
        but if given as a kwarg then that step is bipassed and it tries to use this ime.  The freqs and ime 
        shoud have the same dimensions.
    '''
    if type(lna_temp) is not numpy.ndarray:
        lna_temp = numpy.array(lna_temp)

    temp_A = skyTemperature(freqs) + ground_temp
    if ime is None:
        ime = impedanceMismatchEfficiency(complex_z, z_0 = z_0)
    lna_temp = numpy.tile(lna_temp,(len(freqs),1)).T
    numerator = numpy.tile(numpy.multiply(temp_A, ime),(len(lna_temp),1)) #This is T_A * IME repeated for each lna temp
    snd = 10.0*numpy.log10(numpy.divide(numerator,lna_temp))
    if numpy.shape(snd)[0] == 1:
        return snd[0]
    else:
        return snd



config_legend_day1 = {  'cfg1':'Low Pass 1, Full, EW, Power On',
                        'cfg2':'Low Pass 1, Full, EW, Power On',
                        'cfg3':'Low Pass 1, Full, NS, Power On',
                        'cfg4':'Low Pass 1, Full, NS, Power Off',
                        'cfg5':'Low Pass 1, Full, EW, Power Off',
                        'cfg6':'Low Pass 1, Stripped, EW, Power On',
                        'cfg7':'Low Pass 1, Stripped, NS, Power On',
                        'cfg8':'Low Pass 1, Stripped, NS, Power Off',
                        'cfg9':'Low Pass 1, Wrapped, EW, Power On',
                        'cfg10':'Low Pass 1, Wrapped, EW, Power On',
                        'cfg11':'Low Pass 1, Wrapped, NS, Power On',
                        'cfg12':'Low Pass 1, Wrapped, NS, Power Off',
                        'cfg13':'Low Pass 1, Wrapped, EW, Power Off'}

config_legend_day2 = {  'cfg1':'Low Pass 1, Full, EW, Power On',
                        'cfg2':'Low Pass 1, Full, NS, Power On',
                        'cfg3':'Low Pass 1, Full, EW, Power Off',
                        'cfg4':'Low Pass 1, Full, NS, Power Off',
                        'cfg5':'Low Pass 1, Stripped, EW, Power On',
                        'cfg6':'Low Pass 1, Stripped, NS, Power On',
                        'cfg7':'Low Pass 1, Stripped, EW, Power Off',
                        'cfg8':'Low Pass 1, Stripped, NS, Power Off',
                        'cfg9':'Low Pass 1, Wrapped, EW, Power On',
                        'cfg10':'Low Pass 1, Wrapped, NS, Power On',
                        'cfg11':'Low Pass 1, Wrapped, EW, Power Off',
                        'cfg12':'Low Pass 1, Wrapped, NS, Power Off',
                        'cfg13':'Low Pass 1, Stripped, Anechoic, Power On',
                        'cfg14':'Low Pass 1, Stripped, Anechoic, Power Off',
                        'cfg15':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg16':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg17':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg18':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg19':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg20':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg21':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg22':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg23':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg24':'Low Pass 1, Wrapped, Anechoic, Power On',
                        'cfg25':'Low Pass 1, Wrapped, Anechoic, Power On'}

config_legend_day3 = {  'cfg1':'Low Pass 1, Full, NS, Power On',
                        'cfg2':'Low Pass 1, Full, EW, Power On',
                        'cfg3':'Low Pass 1, Stripped, Anechoic, Power On',
                        'cfg4':'Low Pass 1, Stripped, Anechoic, Power On'}

config_legend_day4 = {  'cfg1':'No Low Pass, Stripped, Anechoic, Power On',
                        'cfg2':'No Low Pass, Stripped, Anechoic, Power On',
                        'cfg3':'No Low Pass, Full, NS, Power On',
                        'cfg4':'No Low Pass, Full, NS, Power On',
                        'cfg5':'No Low Pass, Full, EW, Power On',
                        'cfg6':'No Low Pass, Full, EW, Power On',
                        'cfg7':'Low Pass 2 + Ext, Full, NS, Power On',
                        'cfg8':'Low Pass 2 + Ext, Full, NS, Power On',
                        'cfg9':'Low Pass 2 + Ext, Full, EW, Power On',
                        'cfg10':'Low Pass 2 + Ext, Full, EW, Power On',
                        'cfg11':'Low Pass 2 + Ext, Stripped, Anechoic, Power On',
                        'cfg12':'Low Pass 2 + Ext, Stripped, Anechoic, Power On'}

config_legend_day5 = {  'cfg1':'Aug21 - No On Board Filters, 40 dB - 10 dB, Full, NS, Power On',
                        'cfg2':'Aug21 - No On Board Filters, 40 dB - 10 dB, Full, NS, Power On',
                        'cfg3':'Aug21 - No On Board Filters, 40 dB - 10 dB, Full, EW, Power On',
                        'cfg4':'Aug21 - No On Board Filters, 40 dB - 10 dB, Full, EW, Power On',
                        'cfg5':'Aug21 - No On Board Filters, 40 dB - 10 dB, Stripped, Anechoic',
                        'cfg6':'Aug21 - No On Board Filters, 40 dB, Stripped, Anechoic'}


#Low pass 1 = old wrong low pass (Eric says he messed up the calculation)
#No low pass = when L4,L5, and C5 were removed
#Low pass 2 = a low pass that was added using the parts we had at the time: L4,L5=100nH, C5=82pF
# + Ext = means that an external low pass filter is added similar to the second stage in the DAQ.  This part is: SLP-90+ from mini-circuits

def averageRIGOL(infiles,mode='log'):
    '''
    This will load in the files listed in infiles and average them using the specified mode.
    '''
    if numpy.size(infiles) > 1:
        headers = []
        for average_index , infile in enumerate(infiles):
            _freqs, _antenna_dB, _header = readRigol(infile)
            headers.append(_header)
            if average_index == 0:
                freqs = _freqs
                antenna_dB = _antenna_dB
            else:
                antenna_dB = numpy.vstack((antenna_dB, _antenna_dB))

        if mode == 'log':
            antenna_dB = 10*numpy.log10(numpy.mean(10**(antenna_dB/10),axis=0))
        elif mode == 'linear':
            antenna_dB = numpy.mean(antenna_dB,axis=0)
        else:
            print('Selected mode not expected.  Using log mode.')
            antenna_dB = 10*numpy.log10(numpy.mean(10**(antenna_dB/10),axis=0))
    else:
        if type(infiles) == list or type(infiles) == numpy.ndarray:
            infile = infiles[0]
        else:
            infile = infiles
        freqs, antenna_dB, headers = readRigol(infile)
    return freqs, antenna_dB, headers

if __name__ == '__main__':
    infiles = numpy.array(glob.glob(os.environ['BEACON_ANALYSIS_DIR'] + 'data/aug21/*cfg*.csv'))
    config_legend = config_legend_day5
    infiles = numpy.array(infiles)[numpy.argsort([int(i.split('cfg')[-1].replace('.csv','')) for i in infiles])]
    plot_title_root = '5 Minute Sweep with 10 KHz BWR'
    roof_cfgs =  [[1,2],[3,4],[5],[6]]#[[3,4],[5,6],[1,2]] #any multi element portion will be averaged
    lna_index = 2
    #Preparation.  Getting file names for averaging later.
    roof_cfgs_files = []
    roof_cfgs_labels = []
    for i in roof_cfgs:
        _infiles = []
        if numpy.size(i) > 1:
            for j in i:
                _infiles.append(infiles[numpy.array([int(infile.split('cfg')[-1].replace('.csv','')) for infile in infiles]) == j][0])
            roof_cfgs_labels.append(config_legend['cfg'+ str(j)])
        else:
            if type(i) == list or type(i) == numpy.ndarray:
                _infiles.append(infiles[numpy.array([int(infile.split('cfg')[-1].replace('.csv','')) for infile in infiles]) == i[0]][0])
                roof_cfgs_labels.append(config_legend['cfg'+ str(i[0])])
            else:
                _infiles.append(infiles[numpy.array([int(infile.split('cfg')[-1].replace('.csv','')) for infile in infiles]) == i][0])
                roof_cfgs_labels.append(config_legend['cfg'+ str(i)])
        roof_cfgs_files.append(_infiles)

    #PLOTS

    #Main spectra
    plt.figure()
    plt.ylabel('dBm',fontsize=16)
    plt.xlabel('Frequency (MHz)',fontsize=16)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylim([-100,10])
    
    for index, _infiles in enumerate(roof_cfgs_files):
        freqs, antenna_dB, headers = averageRIGOL(_infiles,mode='log')
        plt.plot(freqs/1.0e6,antenna_dB,label=roof_cfgs_labels[index])

    plt.title(plot_title_root,fontsize=20)
    plt.legend(fontsize=16, loc = 'upper left')


    #lna subtracted spectra
    freqs, lna_antenna_dB, lna_headers = averageRIGOL(roof_cfgs_files[lna_index],mode='log')
    plt.figure()
    plt.ylabel('dB',fontsize=16)
    plt.xlabel('Frequency (MHz)',fontsize=16)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    for index, _infiles in enumerate(roof_cfgs_files):
        if index == lna_index:
            continue
        freqs, antenna_dB, headers = averageRIGOL(_infiles,mode='log')
        plt.plot(freqs/1.0e6,antenna_dB - lna_antenna_dB,label=roof_cfgs_labels[index])

    plt.title(plot_title_root + ' lna noise subtracted',fontsize=20)
    plt.legend(fontsize=16, loc = 'upper left')

    #########
    #Subtracted Plots
    #########
    '''
    plt.figure()
    plt.ylabel('Power Above LNA (dBm)',fontsize=16)
    plt.xlabel('Frequency (MHz)',fontsize=16)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    #plt.ylim([-80,10])
    roof_cfgs = [1,2]
    headers = []
    lna_freqs, lna_antenna_dB, lna_header = readRigol(infiles[numpy.array([int(infile.split('cfg')[-1].replace('.csv','')) for infile in infiles]) == 4][0])
    

    for i in roof_cfgs:
        infile = infiles[numpy.array([int(infile.split('cfg')[-1].replace('.csv','')) for infile in infiles]) == i][0]

        freqs, antenna_dB, header = readRigol(infile)
        headers.append(header)
        plt.plot(freqs/1.0e6,antenna_dB - lna_antenna_dB,label=config_legend['cfg%i'%i] + ' - ' + config_legend['cfg%i'%4])

    plt.title('5 Minute Sweep with 10 KHz BWR',fontsize=20)
    plt.legend(fontsize=16, loc = 'upper left')
    '''

    '''
    config_legend = config_legend_day2

    #Mean LNA Spectrum
    lna_cfgs = numpy.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    lna_dB = []
    for infile in infiles:
        cfg = infile.split('/')[-1].replace('.csv','')
        if int(cfg.split('cfg')[-1]) in lna_cfgs:
            print(cfg)
            lna_dB.append(readRigol(infile)[1])
    
    lna_dB = numpy.array(lna_dB)
    lna_dB = 10*numpy.log10(numpy.mean(10**(numpy.array(lna_dB)/10),axis=0))

    #Roof Anteenna
    roof_cfgs = [1,2]
    for i in roof_cfgs:
        infile = infiles[numpy.array([int(infile.split('cfg')[-1].replace('.csv','')) for infile in infiles]) == i][0]

        freqs, antenna_dB, header = readRigol(infile)

        plt.figure()
        plt.ylabel('dBm',fontsize=16)
        plt.xlabel('Frequency (MHz)',fontsize=16)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.plot(freqs/1.0e6,lna_dB,label=config_legend['cfg15'] + ', Average of 10 Spectra')
        plt.plot(freqs/1.0e6,antenna_dB,label=config_legend['cfg%i'%i])
        plt.legend(fontsize=16, loc = 'upper left')
        plt.ylim([-80,0])

        plt.figure()
        plt.ylabel('dBm',fontsize=16)
        plt.xlabel('Frequency (MHz)',fontsize=16)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.plot(freqs/1.0e6,antenna_dB - lna_dB,label=config_legend['cfg%i'%i] + ' SUBTRACT ' + config_legend['cfg15']+ ', Average of 10 Spectra')
        plt.legend(fontsize=16)

    '''
    '''
    day = 2

    if day == 1:
        infiles = glob.glob(os.environ['BEACON_ANALYSIS_DIR'] + 'data/roof/*.csv')
        infiles = 
        interest = ['cfg2','cfg6','cfg10','cfg12']
        config_legend = config_legend_day1
    elif day == 2:
        infiles = glob.glob(os.environ['BEACON_ANALYSIS_DIR'] + 'data/roof2/*.csv')
        infiles = 
        interest = ['cfg1','cfg9']
        config_legend = config_legend_day2


    data = []
    plt.figure()
    for infile in infiles:
        cfg = infile.split('/')[-1].replace('.csv','')
        if cfg in interest:
            freqs, dB, header = readRigol(infile)
            data.append(dB)
            #peaks, _ =scipy.signal.find_peaks(dB)
            plt.plot(freqs/1e6,dB,label=config_legend[cfg])
            #plt.scatter(freqs[peaks], dB[peaks], 'x', )
            plt.legend()

    data = numpy.array(data)
    dB_diff = data[0,:] - data[1,:]
    plt.ylabel('dBm')
    plt.xlabel('Frequency (MHz)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


    plt.figure()
    plt.ylabel('dBm')
    plt.xlabel('Frequency (MHz)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.plot(freqs/1.0e6,dB_diff,label=config_legend[cfg])
    '''