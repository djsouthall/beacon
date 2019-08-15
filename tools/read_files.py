'''
This module will contains functions helpful for working with reading in misc files (such as those from
simulatons).
'''
import numpy
import pylab
import glob
import csv
import sys

def readTxt(infile, header=0, delimeter='\t'):
    '''
    This is really just a wrapper of an inbuilt function.
    '''
    file = open(infile,'r')
    lines = file.readlines()
    vals = []
    for index, line in enumerate(lines):
        if index > header - 1:
            line = line.replace('\n','').split(delimeter)
            vals.append( numpy.array(line,dtype=float) )
    vals = numpy.array(vals)
    return vals

def readRigol(infile):
    '''
    This reads in csv files from the Rigol spectrum analyzer, returns the
    data and a dictionary of the  header information.
    '''
    file = open(infile,'r')
    lines = file.readlines()
    vals = []
    header = {}
    header_lines = 32
    for index, line in enumerate(lines):
        line = line.lower().replace(' ','_').split(',')
        if index < header_lines - 1:
            header[line[0]] = float(line[1])
        elif index == header_lines - 1:
            freqs = numpy.zeros(int(header['number_of_points']))
            dB = numpy.zeros(int(header['number_of_points']))
            data_index = 0
        else:
            freqs[data_index] = float(line[0])
            dB[data_index] = float(line[1])
            data_index += 1
    return freqs, dB, header


if __name__ == '__main__':
    print('Some functions for reading in files.')
    infile = '/home/dsouthall/Projects/Beacon/pyBeaconKit/DipoleTesting/dataFiles/Simulations/BEAC_S11_SIM_A_IMAG.txt'
    freqs, dB, header = readRigol(infile)
    
    

