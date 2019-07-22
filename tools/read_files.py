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



if __name__ == '__main__':
    print('Some functions for reading in files.')
    infile = '/home/dsouthall/Projects/Beacon/pyBeaconKit/DipoleTesting/dataFiles/Simulations/BEAC_S11_SIM_A_IMAG.txt'
    vals = readTxt(infile,header=2,delimeter='\t')
    
    

