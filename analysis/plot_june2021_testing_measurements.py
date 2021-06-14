#!/usr/bin/env python3
'''
This is intended to plot s21 and s11 data from measurements made on 6/8/2021.  
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import inspect
import glob
import datetime

import beacon.tools.field_fox as ff
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import lines


params = {'legend.fontsize': 10,
          'figure.figsize': (14, 6.5),
         'axes.labelsize': 16,
         'axes.titlesize':16,
         'xtick.labelsize':16,
         'ytick.labelsize':16}
plt.rcParams.update(params)

def gainFromS21(likelike_logmag, likeunlike_logmag, distance_m,header=17):
    '''
    Calculates the gain from the S21 following the source:
    https://www.pasternack.com/t-calculator-fspl.aspx

    This assumes the two antennas have the same gain. 
    '''
    try:
        freqs_Hz, logmag_vals = ff.readerFieldFox(likelike_logmag,header=header)
        fspl = numpy.abs(logmag_vals)
        p1 = 20*numpy.log10(distance_m)
        p2 = 20*numpy.log10(freqs_Hz)
        p3 = 20*numpy.log10(4*numpy.pi/(299792458))
        gain_like = (p1 + p2 + p3 - fspl)/2.0

        freqs_Hz, logmag_vals = ff.readerFieldFox(likeunlike_logmag,header=header)
        fspl = numpy.abs(logmag_vals)
        p1 = 20*numpy.log10(distance_m)
        p2 = 20*numpy.log10(freqs_Hz)
        p3 = 20*numpy.log10(4*numpy.pi/(299792458))
        gain_unlike = p1 + p2 + p3 - fspl - gain_like

        #import pdb; pdb.set_trace()

        return freqs_Hz, gain_like, gain_unlike
    except Exception as e:
            print(e)

def gainFromS21LikeLike(filename_logmag, distance_m,header=17):
    '''
    Calculates the gain from the S21 following the source:
    https://www.pasternack.com/t-calculator-fspl.aspx

    This assumes the two antennas have the same gain. 
    '''
    try:
        freqs_Hz, logmag_vals = ff.readerFieldFox(filename_logmag,header=header)
        fspl = numpy.abs(logmag_vals)
        p1 = 20*numpy.log10(distance_m)
        p2 = 20*numpy.log10(freqs_Hz)
        p3 = 20*numpy.log10(4*numpy.pi/(299792458))
        gain = (p1 + p2 + p3 - fspl)/2.0
        #import pdb; pdb.set_trace()

        return freqs_Hz, gain,
    except Exception as e:
            print(e)

def plotGainLikeLike(distance_m,datapath,figsize=(16,9),dpi=108*4,outpath=None):
    '''
    This will plot the gain assuming that both ends of the s21 measurement are similar antennas.
    '''
    infiles = numpy.array(glob.glob(os.path.join(datapath , '*txtx*s21*logmag*.csv')))

    plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_linestyles = numpy.array(list(lines.lineStyles.keys()))
    linestyles = numpy.array(['-', '--',':', '-.', ' ', ''], dtype='<U4')#all_linestyles[~numpy.isin(all_linestyles,['None'])]

    #PLOT PREPPING
    alpha = 0.8
    thickness = 4
    #PLOT Gain
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.ylabel('dBi')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    for infile in infiles:
        try:
            label = os.path.split(infile)[-1].replace('.csv','').replace('_phase','').replace('_logmag','').replace('HOURGLASS','HOURGLASS ').replace('_',' ').title().replace('Pf',' pF').replace('Nh',' nH').replace('Ohm',' $\\Omega$').replace('Noshield','No Shield')
            label = 'Hpol Gain for BEACON Tx Antenna'
            freqs, gain = gainFromS21LikeLike(infile,distance_m,header=17)
            
            plot_cut_ul = 250            
            plot_cut = freqs/1e6 < plot_cut_ul

            linestyle = '-'

            ax.plot(freqs[plot_cut]/1e6, gain[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)#,color=color)


        except Exception as e:
            print(e)

    plt.axhline(2.0,linewidth=thickness,linestyle='--',c=(112/256,173/256,71/256),label='2 dBi Line')
    ax.legend()
    ax.set_xlim([0,plot_cut_ul])

    if outpath is not None:
        fig.set_size_inches(figsize[0], figsize[1])
        plt.tight_layout()
        fig.savefig(os.path.join(outpath,'gain_likelike.png'),dpi=dpi)


def plotS21LikeUnlike(distance_m,datapath,header=17,figsize=(16,9),dpi=108*4,outpath=None):
    '''
    This will plot the gain assuming that both ends of the s21 measurement are similar antennas.
    '''
    infiles_tx = numpy.array(glob.glob(os.path.join(datapath , '*txtx*s21*logmag*.csv')))
    infiles_rx = numpy.array(glob.glob(os.path.join(datapath , '*rxtx*s21*logmag*.csv')))
    try:
        infiles_rx = infiles_rx[numpy.argsort([int(os.path.split(i)[-1].split('_')[3].replace('ant','')) for i in infiles_rx])]
    except Exception as e:
        print(e)
        



    plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_linestyles = numpy.array(list(lines.lineStyles.keys()))
    linestyles = numpy.array(['-', '--',':', '-.', ' ', ''], dtype='<U4')#all_linestyles[~numpy.isin(all_linestyles,['None'])]

    #PLOT PREPPING
    alpha = 0.8
    thickness = 4
    #PLOT Gain
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.title('S21')
    plt.ylabel('dB')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    infile_tx = infiles_tx[0]
    label = ''

    for infile_index, infile_rx in enumerate(infiles_rx):
        try:
            

            linestyle = '-'

            if infile_index == 0:
                freqs_Hz, s21_tx_logmag_vals = ff.readerFieldFox(infile_tx,header=header)
                plot_cut_ul = 250            
                plot_cut = freqs_Hz/1e6 < plot_cut_ul
                label = os.path.split(infile_tx)[-1].replace('.csv','').replace('_', ' ').replace(' logmag', '').replace('beacon ','').replace(' s21','').replace('txtx','tx -> tx')
                ax.plot(freqs_Hz[plot_cut]/1e6, s21_tx_logmag_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)

            freqs_Hz, s21_rx_logmag_vals = ff.readerFieldFox(infile_rx,header=header)
            plot_cut_ul = 250            
            plot_cut = freqs_Hz/1e6 < plot_cut_ul

            label = os.path.split(infile_rx)[-1].replace('.csv','').replace('_', ' ').replace(' logmag', '').replace('beacon ','').replace(' s21','').replace('rxtx','tx -> rx')
            ax.plot(freqs_Hz[plot_cut]/1e6, s21_rx_logmag_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)


        except Exception as e:
            print(e)

    ax.legend()
    ax.set_xlim([0,plot_cut_ul])

    if outpath is not None:
        fig.set_size_inches(figsize[0], figsize[1])
        plt.tight_layout()
        fig.savefig(os.path.join(outpath,'s21_likelike.png'),dpi=dpi)

def plotS11(distance_m,header=17,figsize=(16,9),dpi=108*4,outpath=None):
    '''
    This will plot the gain assuming that both ends of the s21 measurement are similar antennas.
    '''
    datapath1 = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon_s21_june6_2021')
    #infiles = numpy.append(numpy.array(glob.glob(os.path.join(datapath1 , '*s11*logmag*.csv'))),numpy.array(glob.glob(os.path.join(datapath1 , '*s22*logmag*.csv'))))
    infiles = numpy.array(glob.glob(os.path.join(datapath1 , '*s11*logmag*.csv')))
    
    if True:
        datapath2 = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon_tx_s11_june7_2021')
        new_infiles = numpy.array(glob.glob(os.path.join(datapath2 , '*s11*logmag*.csv')))
        new_infiles = new_infiles[numpy.argsort([int(i.split('bowtie')[-1][0]) for i in new_infiles])]

        infiles = numpy.append(infiles, new_infiles)

    if True:
        datapath3 = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon_s21_june11_2021')
        new_infiles = numpy.append(numpy.array(glob.glob(os.path.join(datapath3 , '*s11*logmag*.csv'))),numpy.array(glob.glob(os.path.join(datapath3 , '*s22*logmag*.csv'))))
        infiles = numpy.append(infiles, new_infiles)


    plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_linestyles = numpy.array(list(lines.lineStyles.keys()))
    linestyles = numpy.array(['-', '--',':', '-.', ' ', ''], dtype='<U4')#all_linestyles[~numpy.isin(all_linestyles,['None'])]

    #PLOT PREPPING
    alpha = 0.8
    thickness = 4
    #S11 plot
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.ylabel('dB')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    

    plt.title('Comparing BEACON Tx Antennas')
    for infile in infiles:

        label = os.path.split(infile)[-1].replace('.csv','').replace('_', ' ')
        if 'tx1' in label:
            label = 'beacon s11 dipole 0'
        label = label.replace(' logmag','').replace('bowtie','bowtie ')

        try:
            freqs_Hz, logmag_vals = ff.readerFieldFox(infile,header=header)
            
            plot_cut_ul = 250            
            plot_cut = freqs_Hz/1e6 < plot_cut_ul

            linestyle = '-'

            ax.plot(freqs_Hz[plot_cut]/1e6, logmag_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)


        except Exception as e:
            print(e)

    ax.legend()
    ax.set_xlim([0,plot_cut_ul])

    if outpath is not None:
        fig.set_size_inches(figsize[0], figsize[1])
        plt.tight_layout()
        fig.savefig(os.path.join(outpath,'all_s11.png'),dpi=dpi)


    #Highlighted S11 plot
    for index in range(len(infiles)):
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        plt.ylabel('dB')
        plt.xlabel('MHz')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        
        plt.title('Comparing BEACON Tx Antennas')
        for i, infile in enumerate(infiles):
            if i == index:
                label = os.path.split(infile)[-1].replace('.csv','').replace('_', ' ')
                if 'tx1' in label:
                    label = 'beacon s11 dipole 0'
                label = label.replace(' logmag','').replace('bowtie','bowtie ')
            try:
                freqs_Hz, logmag_vals = ff.readerFieldFox(infile,header=header)
                
                plot_cut_ul = 250            
                plot_cut = freqs_Hz/1e6 < plot_cut_ul

                linestyle = '-'

                if i == index:
                    ax.plot(freqs_Hz[plot_cut]/1e6, logmag_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle, color = 'r')
                else:
                    ax.plot(freqs_Hz[plot_cut]/1e6, logmag_vals[plot_cut],linewidth=thickness/2,alpha=alpha*.66,linestyle=linestyle, color = 'k')


            except Exception as e:
                print(e)

        ax.legend()
        ax.set_xlim([0,plot_cut_ul])
        if outpath is not None:
            fig.set_size_inches(figsize[0], figsize[1])
            plt.tight_layout()
            fig.savefig(os.path.join(outpath,label.replace(' ','_') + '.png'),dpi=dpi)



def plotBoardS21(header=17,figsize=(16,9),dpi=108*4,outpath=None):
    '''
    This will plot the gain assuming that both ends of the s21 measurement are similar antennas.
    '''
    datapath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon2021_board_s21','boards')
    infiles = numpy.array(glob.glob(os.path.join(datapath , '*s21*logmag*.csv')))
    infiles = infiles[numpy.argsort([int(os.path.split(i)[-1].split('_')[2].replace('b','')) for i in infiles])]

    plot_cut_ul = 250            

    plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_linestyles = numpy.array(list(lines.lineStyles.keys()))
    linestyles = numpy.array(['-', '--',':', '-.', ' ', ''], dtype='<U4')#all_linestyles[~numpy.isin(all_linestyles,['None'])]

    #PLOT PREPPING
    alpha = 0.8
    thickness = 4

    fig = plt.figure()
    primary_ax = plt.subplot(2,1,1)
    plt.ylabel('S21 (dB)')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    for infile_index, infile in enumerate(infiles):
        try:
            freqs_Hz, logmag_vals = ff.readerFieldFox(infile,header=header)
            if infile_index == 0:
                all_log_vals = logmag_vals
            else:
                all_log_vals = numpy.vstack((all_log_vals,logmag_vals))
            
            plot_cut = freqs_Hz/1e6 < plot_cut_ul

            linestyle = '-'
            label = os.path.split(infile)[-1].replace('.csv','').replace('_', ' ')

            primary_ax.plot(freqs_Hz[plot_cut]/1e6, logmag_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)
        except Exception as e:
            print(e)

    primary_ax.legend(loc = 'upper right')
    primary_ax.set_xlim([0,plot_cut_ul])
    primary_ax.set_ylim([10,50])

    ax = plt.subplot(2,1,2,sharex=primary_ax)
    plt.ylabel('S21 (degrees)')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    for infile_index, infile in enumerate(infiles):
        try:
            infile = infile.replace('logmag','phase')
            freqs_Hz, phase_vals = ff.readerFieldFox(infile,header=header)
            if infile_index == 0:
                all_phase_vals = phase_vals
            else:
                all_phase_vals = numpy.vstack((all_phase_vals,phase_vals))

            
            plot_cut = freqs_Hz/1e6 < plot_cut_ul

            linestyle = '-'
            label = os.path.split(infile)[-1].replace('.csv','').replace('_', ' ')

            ax.plot(freqs_Hz[plot_cut]/1e6, phase_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)
        except Exception as e:
            print(e)

    ax.legend(loc = 'upper right')
    ax.set_xlim([0,plot_cut_ul])

    if outpath is not None:
        fig.set_size_inches(figsize[0], figsize[1])
        plt.tight_layout()
        fig.savefig(os.path.join(outpath,'boards_s21.png'),dpi=dpi)


    #Residuals
    fig = plt.figure()
    ax = plt.subplot(2,1,1,sharex=primary_ax)
    plt.ylabel('S21 (dB)\nResiduals')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    mean_log_vals = numpy.mean(all_log_vals,axis=0)
    mean_phase_vals = numpy.mean(all_phase_vals,axis=0)

    for infile_index, infile in enumerate(infiles):
        try:
            freqs_Hz, logmag_vals = ff.readerFieldFox(infile,header=header)
            
            plot_cut = freqs_Hz/1e6 < plot_cut_ul

            linestyle = '-'
            label = os.path.split(infile)[-1].replace('.csv','').replace('_', ' ')

            ax.plot(freqs_Hz[plot_cut]/1e6, logmag_vals[plot_cut] - mean_log_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)
        except Exception as e:
            print(e)

    ax.legend(loc = 'upper right')
    ax.set_xlim([0,plot_cut_ul])

    ax = plt.subplot(2,1,2,sharex=primary_ax)
    plt.ylabel('S21 (degrees)\nResiduals')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    for infile_index, infile in enumerate(infiles):
        try:
            infile = infile.replace('logmag','phase')
            freqs_Hz, phase_vals = ff.readerFieldFox(infile,header=header)
            
            plot_cut = freqs_Hz/1e6 < plot_cut_ul

            linestyle = '-'
            label = os.path.split(infile)[-1].replace('.csv','').replace('_', ' ')

            ax.plot(freqs_Hz[plot_cut]/1e6, phase_vals[plot_cut] - mean_phase_vals[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)
        except Exception as e:
            print(e)

    ax.legend(loc = 'upper right')
    ax.set_xlim([0,plot_cut_ul])

    if outpath is not None:
        fig.set_size_inches(figsize[0], figsize[1])
        plt.tight_layout()
        fig.savefig(os.path.join(outpath,'boards_s21_residuals.png'),dpi=dpi)

def plotGainLikeUnlike(distance_m,datapath,figsize=(16,9),dpi=108*4,outpath=None):
    '''
    This will plot the gain assuming that both ends of the s21 measurement are similar antennas.
    '''
    infiles_tx = numpy.array(glob.glob(os.path.join(datapath , '*txtx*s21*logmag*.csv')))
    infiles_rx = numpy.array(glob.glob(os.path.join(datapath , '*rxtx*s21*logmag*.csv')))

    try:
        infiles_rx = infiles_rx[numpy.argsort([int(os.path.split(i)[-1].split('_')[3].replace('ant','')) for i in infiles_rx])]
    except Exception as e:
        print(e)




    plot_cut_ul = 250            

    plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_linestyles = numpy.array(list(lines.lineStyles.keys()))
    linestyles = numpy.array(['-', '--',':', '-.', ' ', ''], dtype='<U4')#all_linestyles[~numpy.isin(all_linestyles,['None'])]

    #PLOT PREPPING
    alpha = 0.8
    thickness = 4
    #PLOT Gain
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.title('Gain')
    plt.ylabel('dBi')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    
    infile_tx = infiles_tx[0]

    for infile_index, infile_rx in enumerate(infiles_rx):
        label = ''

        try:
            freqs, gain_tx, gain_rx = gainFromS21(infile_tx,infile_rx,distance_m,header=17)

            plot_cut = freqs/1e6 < plot_cut_ul

            linestyle = '-'


            if infile_index == 0:
                label = os.path.split(infile_tx)[-1].replace('.csv','').replace('_', ' ').replace(' logmag', '').replace('beacon ','').replace(' s21','').replace('txtx','tx -> tx')
                ax.plot(freqs[plot_cut]/1e6, gain_tx[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)
            
            label = os.path.split(infile_rx)[-1].replace('.csv','').replace('_', ' ').replace(' logmag', '').replace('beacon ','').replace(' s21','').replace('rxtx','tx -> rx')
            ax.plot(freqs[plot_cut]/1e6, gain_rx[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    ax.axvspan(30,80,color='y',alpha=0.5)

    plt.axhline(2.0,linewidth=thickness,linestyle='--',c=(112/256,173/256,71/256),label='2 dBi Line')
    ax.legend()
    ax.set_xlim([0,plot_cut_ul])

    if outpath is not None:
        fig.set_size_inches(figsize[0], figsize[1])
        plt.tight_layout()
        fig.savefig(os.path.join(outpath,'gain_likeunlike.png'),dpi=dpi)


if __name__ == '__main__':
    if True:
        outpath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'figures', 's11_testing_' + str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-'))
        matplotlib.use('Agg')
        os.mkdir(outpath)
    else:
        outpath = None
        plt.ion()

    plt.close('all')
    datapath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon_s21_june6_2021')
    distance_m = 0.3048*50
    # plotGainLikeLike(distance_m,datapath,outpath=outpath)
    # plotGainLikeUnlike(distance_m,datapath,outpath=outpath)
    # plotS21LikeUnlike(distance_m,datapath,outpath=outpath)
    # plotS11(distance_m,header=17,outpath=outpath)
    # plotBoardS21(outpath=outpath)


    # datapath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon_s21_june10_2021')
    # plotGainLikeUnlike(distance_m,datapath,outpath=outpath)
    # plotS21LikeUnlike(distance_m,datapath,outpath=outpath)

    datapath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon_s21_june11_2021')
    plotGainLikeUnlike(distance_m,datapath,outpath=outpath)
    plotS21LikeUnlike(distance_m,datapath,outpath=outpath)
    plotS11(distance_m,header=17,outpath=outpath)