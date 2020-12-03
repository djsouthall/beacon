'''
This module contains functions helpful for loading csv data from the field fox and 
converting various unit types.  It also contains some helpful plotting functions.

Parts of this use the pySmithPlot package that was downloaded from the git:
https://github.com/vMeijin/pySmithPlot.git

It should be located in the main Beacon git folder.
'''
import numpy
import pylab
import glob
import csv
import sys
import pylab
import scipy
import scipy.interpolate
sys.path.append('/home/dsouthall/Beacon')
#from pySmithPlot.smithplot import SmithAxes
import beacon.tools.constants as constants
pylab.ion()

# CALCULATION FUNCTIONS

def readerFieldFox(csv_file,header=17,delimiter=','):
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
    csv_reader = csv.reader(open(csv_file),delimiter=delimiter)
    for n in range(header):
        next(csv_reader)
    x = []
    y = []
    for row in csv_reader:
        if len(row) == 2:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return numpy.array(x), numpy.array(y)

def logMagToLin(vals):
    '''
    Converts the Log Magnitude units from FieldFox to the Linear units for FieldFox.
    The linear units are 'Voltage' units (i.e. NOT power units), as is indicated by the
    factor of 20.0 in the calculation.

    Paramaters
    ----------
    vals : numpy.ndarray of floats
        The values to be converted from Log Magnitude to Linear values.

    Returns
    -------
    vals : numpy.ndarray of floats
        The ouput values converted into Linear values.
    '''
    return 10.0**(vals/20.0)

def linToLogMag(vals):
    '''
    Converts the Log Magnitude units from FieldFox to the Linear units for FieldFox.
    The linear units are 'Voltage' units (i.e. NOT power units), as is indicated by the
    factor of 20.0 in the calculation.

    Paramaters
    ----------
    vals : numpy.ndarray of floats
        The input values to be converted into Log Magnitude values.

    Returns
    -------
    vals : numpy.ndarray of floats
        The converted values in Log Magnitude units.
    '''
    return 20.0*numpy.log10(vals)

def magPhaseToReIm(mag,phase):
    '''
    Converts Magnitude of and phase measurement to Re and Imag.  Expects phase to be
    in degrees.  Assumes magnitude is given in linear units.
    
    Paramaters
    ----------
    mag : numpy.ndarray of floats
        The (linear) magnitude of the values.
    phase : numpy.ndarray of floats
        The phase of the values given in degrees.

    Returns
    -------
    Re : numpy.ndarray of floats
        The real components of the values.
    Im : numpy.ndarray of floats
        The imaginary components of the values.  Note that this is the coefficient
        beside j, and should be given/return as floats not imaginary. I.e. 
        y = Re + 1j*Im
    '''
    Re = mag*numpy.cos(numpy.deg2rad(phase))
    Im = mag*numpy.sin(numpy.deg2rad(phase))

    return Re, Im

def reImToMagPhase(Re,Im):
    '''
    Converts the Real and Imaginary portions of a complex number to the magnitude and
    phase components (given in degrees). 

    Paramaters
    ----------
    Re : numpy.ndarray of floats
        The real components of the values.
    Im : numpy.ndarray of floats
        The imaginary components of the values.  Note that this is the coefficient
        beside j, and should be given/return as floats not imaginary. I.e. 
        y = Re + 1j*Im

    Returns
    -------
    mag : numpy.ndarray of floats
        The (linear) magnitude of the values.
    phase : numpy.ndarray of floats
        The phase of the values given in degrees.
    '''
    mag = numpy.sqrt(Re**2.0 + Im**2.0)
    phase = numpy.arctan2(Im/Real)
    return mag, numpy.deg2rad(phase)

def linToVSWR(mag):
    '''
    Converts the linear magnitude of the S11 parameter to VSWR.

    Parameters
    ----------
    mag : numpy.ndarray of floats
        The linear magnitudes of the S11 parameter.

    Returns
    -------
    VSWR : numpy.ndarray of floats
        The calculated VSWR values.
    '''
    VSWR = numpy.divide(1.0 + numpy.fabs(mag),1.0 - numpy.fabs(mag) )
    return VSWR


def linToComplexZ( complexS11, z_0 = 50.0 ):
    '''
    Converts Magnitude of and phase measurement to Re and Imag.  Expects phase to be
    in degrees.  Assumes magnitude is given in linear units.
    
    Paramaters
    ----------
    complexS11 : numpy.ndarray of complex floats
        The complex linear S11 paramter. 
    z_0 : float, optional
        The input impedance, typically 50 ohms.

    Returns
    -------
    z : numpy.ndarray of complex floats
        The impedance.  Given in Ohms.
    '''

    z = z_0 * numpy.divide( 1.0 + complexS11 , 1.0 - complexS11 )
    return z
def logToComplexZ(logmag, unwrapped_phase, z_0 = 50.0 ):
    '''
    Converts the log magnitude S11 to the reflection coefficient.
    Converts linear complex s11 of and phase measurement to Re and Imag.  Expects phase to be
    in degrees.  Assumes magnitude is given in linear units. 
    
    Paramaters
    ----------
    logmag : numpy.ndarray of floats
        The values to be converted from Log Magnitude to Linear values.
    unwrapped_phase : numpy.ndarray of floats
        The unwrapped phase.
    z_0 : float
        The characteristic impedance you are matching to.

    Returns
    -------
    z : numpy.ndarray of complex floats
        The complex impedance.
    '''

    lin = logMagToLin(logmag)
    re, im = magPhaseToReIm(lin,unwrapped_phase)
    complexS11 = re + 1.0j*im
    z = linToComplexZ(complexS11,z_0=z_0)
    return z

def logMagToReflectionCoefficient( logmag, unwrapped_phase, z=None, z_0 = 50.0 ):
    '''
    Converts the log magnitude S11 to the reflection coefficient.
    Converts linear complex s11 of and phase measurement to Re and Imag.  Expects phase to be
    in degrees.  Assumes magnitude is given in linear units. 
    
    Paramaters
    ----------
    logmag : numpy.ndarray of floats
        The values to be converted from Log Magnitude to Linear values.
    unwrapped_phase : numpy.ndarray of floats
        The unwrapped phase.
    z_0 : float
        The characteristic impedance you are matching to.

    Returns
    -------
    gamma : numpy.ndarray of complex floats
        The reflection coefficient.
    '''
    if z is None:
        z = logToComplexZ(logmag, unwrapped_phase, z_0 = 50.0 )
    gamma = numpy.divide(z - z_0, z + z_0)
    return gamma

def skyIntensity(freqs, plot = False, fig = None, ax = None):
    '''
    This attempts to calculate the I_nu from equation 1 in the source below. 

    Sources
    -------
        Paper : "Design and Evaluation of an Active Antenna for a 29–47 MHz Radio Telescope Array"
        Authors : S.W. Ellingson et al.
        url : http://www.phys.unm.edu/~lwa/memos/memo/lwa0060.pdf

    Parameters
    ----------
    freqs : numpy.ndarra of floats
        The frequencies for which to calculate the Galactic Noise Temperature.  Should be given in Hz.
    '''
    freqs_MHz = freqs/1.0e6 #For use in parameterization
    tau = 5.0*(freqs_MHz**-2.1)
    I_g = 2.48e-20 #W m^-2 Hz^-1 sr^-1 
    I_eg = 1.06e-20 #W m^-2 Hz^-1 sr^-1 
    I_nu = numpy.multiply(I_g * freqs_MHz**(-0.52), numpy.divide( - numpy.expm1(-tau) , tau) ) + numpy.multiply( I_eg*freqs_MHz**(-0.80), numpy.expm1(-tau) + 1.0) #W m^-2 Hz^-1 sr^-1 

    if plot == True:
        if fig is None:
            fig = pylab.figure()
        if ax is None:
            ax = fig.gca()
        ax.loglog(freqs_MHz, I_nu)
        pylab.ylabel('Intensity [W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$ ')
        pylab.xlabel('Freq [MHz]')
        pylab.grid(b=True, which='major', color='k', linestyle='-')
        pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    return I_nu

def skyTemperature(freqs, plot = False, fig = None):
    '''
    This attempts to calculate the T_sky from equation 1 in the source below. 

    Sources
    -------
        Paper : "Design and Evaluation of an Active Antenna for a 29–47 MHz Radio Telescope Array"
        Authors : S.W. Ellingson et al.
        url : http://www.phys.unm.edu/~lwa/memos/memo/lwa0060.pdf

    Parameters
    ----------
    freqs : numpy.ndarray of floats
        The frequencies for which to calculate the Galactic Noise Temperature.  Should be given in Hz.
    plot : bool
        Enables plotting
    fig : matplotlib.pyplot.figure
        The figure to plot on.  If no given the it is created.

    Returns
    -------
    T_sky : numpy.ndarray of floats
        The sky noise temperature calculated for each frequency.
    '''
    if type(freqs) is list:
        freqs = numpy.array(freqs)
    freqs_GHz = freqs/1.0e9 #for compatibility with c in m/ns

    if plot == True:
        if fig is None:
            fig = pylab.figure()
        ax = pylab.subplot(2,1,1)
    else:
        fig = None
        ax = None
    I_nu = skyIntensity(freqs,plot=plot,fig=fig,ax=ax) #W m^-2 Hz^-1 sr^-1 
    k = constants.boltzmann # J K^-1 = kg m^2 s^-2 K^-1 
    c = constants.speed_light # m ns^-1

    T_sky = (1.0/(2.0*k)) * I_nu * numpy.divide(c**2.0,freqs_GHz**2) # K (I checked units and the only thing that doesn't cancel is the sr^-1)

    if plot == True:
        ax = pylab.subplot(2,1,2)
        ax.loglog(freqs/1.0e6, T_sky)
        pylab.ylabel('T$_\mathrm{sky}$ [K]')
        pylab.xlabel('Freq [MHz]')
        pylab.grid(b=True, which='major', color='k', linestyle='-')
        pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    return T_sky


def impedanceMismatchEfficiency(complex_z, z_0 = 50.0):
    '''
    Calcualtes the Impedance Mismatch Efficiency (IME) for the input impedances.

    Parameters
    ----------
    complex_z : numpy.ndarray of complex floats
        The complex imedance used to calculate the IME.
    z_0 : float, optional
        The input impedance, typically 50 ohms.

    Returns
    -------
    ime : numpy.ndarray of floats
        The IME calculated for each impedance.
    '''
    ratio = numpy.divide(complex_z - z_0,complex_z + z_0)
    ime = 1.0 - numpy.real(ratio * numpy.conj( ratio ) )
    #ime = 1.0 - numpy.absolute( ratio )
    return ime

def skyNoiseDominance(freqs,complex_z, lna_temp, z_0 = 50.0,ground_temp = 0.0, ime = None):
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

# PLOTTING FUNCTIONS


def plotLogMag(csv,fig=None,freq_range = None,expect_linear=False,**kwargs):
    '''
    Plots the log magnitude from a csv (assuming the csv contains log magnitude data unless expect_linear == True).
    If fig is given then it will plot on that figure with the predefined axis/title/etc.  If label
    is given as one of the optional kwargs then the plot will be given this label, otherwise it will 
    pull one from csv.  Will always plot in MHz, so ensure that if fig is given that it accounts for this.

    Parameters
    ---------
    csv : str
        The path/filename of the csv to be plotted.  It should lead to a csv file that was made
        using the field fox in the Log Magnitude setting.
    fig : matlotlib.pyplot.figure(), optional
        The input figure, to be plotted on.  If not given then a figure will be created.
    freq_range : tuple of floats
        The range for which to plot the frequencies.  (min_freq,max_freq), given in MHz.
    expect_linear : bool
        If true then the csv is expected to contain linear magnitudes.
    **kwargs
        Should contain only additional plotting kwargs, such as label.

    Returns
    -------
    fig : matlotlib.pyplot.figure()
        The plotted figure object.
    '''
    if fig is None:
        new_plot = True
        fig = pylab.figure()
    else:
        new_plot = False
    if numpy.isin('label',list(kwargs.keys())) == False:
        kwargs['label'] = 'LogMag For ' +csv.split('/')[-1].replace('.csv','')
    if expect_linear:
        freqs, lin = readerFieldFox(csv)
        LM = linToLogMag(lin)
    else:
        freqs, LM = readerFieldFox(csv)
    if freq_range is not None:
        cut = numpy.logical_and(freqs/1e6 > freq_range[0], freqs/1e6 < freq_range[1])
    else:
        cut = numpy.ones_like(freqs,dtype=bool)

    pylab.plot(freqs[cut]/1e6,LM[cut],**kwargs)

    if new_plot == True:
        pylab.legend()
        pylab.xlabel('Frequency (MHz)')
        pylab.ylabel('Log Mag (dB)')

    return fig

def plotPhase(csv,fig=None,freq_range = None,**kwargs):
    '''
    Plots the phase from a csv (assuming the csv contains phase data).
    If fig is given then it will plot on that figure with the predefined axis/title/etc.  If label
    is given as one of the optional kwargs then the plot will be given this label, otherwise it will 
    pull one from csv.  Will always plot in MHz, so ensure that if fig is given that it accounts for this.  
    Phase will be plotted in degrees (as is set by the field fox).

    Parameters
    ---------
    csv : str
        The path/filename of the csv to be plotted.  It should lead to a csv file that was made
        using the field fox in the phase setting.
    fig : matlotlib.pyplot.figure(), optional
        The input figure, to be plotted on.  If not given then a figure will be created.
    freq_range : tuple of floats
        The range for which to plot the frequencies.  (min_freq,max_freq), given in MHz.
    **kwargs
        Should contain only additional plotting kwargs, such as label.

    Returns
    -------
    fig : matlotlib.pyplot.figure()
        The plotted figure object.
    '''
    if fig is None:
        new_plot = True
        fig = pylab.figure()
    else:
        new_plot = False

    if numpy.isin('label',list(kwargs.keys())) == False:
        kwargs['label'] = 'Phase For ' +csv.split('/')[-1].replace('.csv','')

    freqs, phase = readerFieldFox(csv)
    if freq_range is not None:
        cut = numpy.logical_and(freqs/1e6 > freq_range[0], freqs/1e6 < freq_range[1])
    else:
        cut = numpy.ones_like(freqs,dtype=bool)
    pylab.plot(freqs[cut]/1e6,phase[cut],**kwargs)

    if new_plot == True:
        pylab.legend()
        pylab.xlabel('Frequency (MHz)')
        pylab.ylabel('Phase (deg)')

    return fig

def plotVSWR(csv,fig=None,freq_range = None,expect_linear =  False,**kwargs):
    '''
    Plots the VSWR from a csv (assuming the csv contains log magnitude data).
    If fig is given then it will plot on that figure with the predefined axis/title/etc.  If label
    is given as one of the optional kwargs then the plot will be given this label, otherwise it will pull one from csv.  Will
    always plot in MHz, so ensure that if fig is given that it accounts for this.

    Parameters
    ---------
    csv : str
        The path/filename of the csv to be plotted.  It should lead to a csv file that was made
        using the field fox in the Log Magnitude setting.
    fig : matlotlib.pyplot.figure(), optional
        The input figure, to be plotted on.  If not given then a figure will be created.
    freq_range : tuple of floats
        The range for which to plot the frequencies.  (min_freq,max_freq), given in MHz.
    expect_linear : bool
        If true then the csv is expected to contain linear magnitudes.
    **kwargs
        Should contain only additional plotting kwargs, such as label.
    Returns
    -------
    fig : matlotlib.pyplot.figure()
        The plotted figure object.
    '''
    if fig is None:
        new_plot = True
        fig = pylab.figure()
    else:
        new_plot = False
    if numpy.isin('label',list(kwargs.keys())) == False:
        kwargs['label'] = 'VSWR For ' + csv.split('/')[-1].replace('.csv','')
    
    if expect_linear:
        freqs, lin = readerFieldFox(csv)
        VSWR = linToVSWR(lin)
    else:
        freqs, LM = readerFieldFox(csv)
        VSWR = linToVSWR(logMagToLin(LM))
    
    if freq_range is not None:
        cut = numpy.logical_and(freqs/1e6 > freq_range[0], freqs/1e6 < freq_range[1])
    else:
        cut = numpy.ones_like(freqs,dtype=bool)

    pylab.plot(freqs[cut]/1e6,VSWR[cut],**kwargs)

    if new_plot == True:
        pylab.legend()
        pylab.xlabel('Frequency (MHz)')
        pylab.ylabel('VSWR')

    return fig



if __name__ == '__main__':
    print('This module contains functions helpful for working with field fox data.')
    #pylab.close('all')

    '''
    freqs = numpy.arange(2.0e6,350.0e6,1000) #Hz

    fig = pylab.figure()
    T_sky = skyTemperature(freqs,plot=True,fig = fig)
    test_x = numpy.array([29e6,47e6])
    test_y = skyTemperature(test_x) 
    
    ax = fig.gca()
    ax.scatter(test_x[0]/1.0e6,test_y[0], c='r',label='T$_\mathrm{sky}$(%0.2f MHz) = %0.2f'%(test_x[0]/1.0e6, test_y[0]))
    ax.scatter(test_x[1]/1.0e6,test_y[1], c='g',label='T$_\mathrm{sky}$(%0.2f MHz) = %0.2f'%(test_x[1]/1.0e6, test_y[1]))
    pylab.legend()
    print(list(zip(test_x,test_y)))

    fig = pylab.figure()
    ime_file = '/home/dsouthall/Projects/Beacon/pyBeaconKit/DipoleTesting/dataFiles/FromPapers/IME-From-1210-05016.csv'
    freqs,ime = readerFieldFox(ime_file,header=0)
    snd_file = '/home/dsouthall/Projects/Beacon/pyBeaconKit/DipoleTesting/dataFiles/FromPapers/SND-From-1210-05016.csv'
    freqs_snd,snd_data = readerFieldFox(snd_file,header=0)
    ReZ_file = '/home/dsouthall/Projects/Beacon/pyBeaconKit/DipoleTesting/dataFiles/FromPapers/ReZ-From-1210-05016.csv'
    freqs_ReZ,ReZ_data = readerFieldFox(ReZ_file,header=0)
    ImZ_file = '/home/dsouthall/Projects/Beacon/pyBeaconKit/DipoleTesting/dataFiles/FromPapers/ImZ-From-1210-05016.csv'
    freqs_ImZ,ImZ_data = readerFieldFox(ImZ_file,header=0)
    
    freqs_interp = numpy.linspace(numpy.max(numpy.append(numpy.min(freqs_ReZ),numpy.min(freqs_ImZ))),numpy.min(numpy.append(numpy.max(freqs_ReZ),numpy.max(freqs_ImZ))),numpy.max([len(freqs_ReZ),len(freqs_ImZ)]))
    interp_ReZ = scipy.interpolate.interp1d(freqs_ReZ,ReZ_data)(freqs_interp) 
    interp_ImZ = scipy.interpolate.interp1d(freqs_ImZ,ImZ_data)(freqs_interp) 
    
    complex_z = interp_ReZ + 1.0j*interp_ImZ
    ime_calc = impedanceMismatchEfficiency(complex_z, z_0 = 100.0)
    T_lna = numpy.array([200.0,255.0,275.0,300.0])
    snd = skyNoiseDominance(freqs,numpy.array([]), T_lna,z_0 = 50.0,ground_temp = 0.0, ime = ime)


    pylab.subplot(3,1,1)
    pylab.suptitle('Attempts at Recreating Plots From ar$\chi$iv 1210.0506')
    pylab.plot(freqs_interp/1e6,interp_ReZ,label='Re(Z) as Shown in ar$\chi$iv 1210.0506')
    pylab.plot(freqs_interp/1e6,interp_ImZ,label='Im(Z) as Shown in ar$\chi$iv 1210.0506')
    pylab.ylabel('Z (Ohms)')
    pylab.xlabel('Frequency (MHz)')
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    pylab.legend(loc='upper left')

    pylab.subplot(3,1,2)
    pylab.plot(freqs/1e6,ime,label='IME as Shown in ar$\chi$iv 1210.0506')
    pylab.plot(freqs_interp/1e6,ime_calc,label='Calculated from Impedances in ar$\chi$iv 1210.0506')

    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    pylab.ylabel('IME')
    pylab.xlabel('Frequency (MHz)')
    pylab.legend()

    pylab.subplot(3,1,3)
    pylab.plot(freqs_snd/1e6,snd_data,c='k',linewidth=0.5,alpha=0.5,label='SND from ar$\chi$iv 1210.0506')
    pylab.scatter(freqs_snd/1e6,snd_data,c='k',s=1,label='SND from ar$\chi$iv 1210.0506')
    for index,row in enumerate(snd):
        pylab.plot(freqs/1e6,row,label='T$_{LNA}$ = %0.2f K'%T_lna[index])
    pylab.ylabel('SND (dB)')
    pylab.ylim(0,15)
    pylab.minorticks_on()
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    pylab.xlabel('Frequency (MHz)')
    pylab.legend(loc='lower right')


    
    '''