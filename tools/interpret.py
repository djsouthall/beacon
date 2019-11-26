#!/usr/bin/env python3
'''
This contains functions that are helpful for interpreting the beaconroot classes in
a more pythonic way.  
'''

import numpy
import os
import sys

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
import examples.beacon_data_reader #Must be imported before matplotlib or else plots don't load.
import glob
from pprint import pprint



def getHeaderKeys(reader=None):
    '''
    Get the keys for the header portion of reader files.  If reader
    is None then this will just initiate its own reader just to get the class
    type to pull the keys.  This assumes that the beacon data is stored
    at the location specified by the environment variable $BEACON_DATA.

    Note that this also assumes particular formatting for the keys (i.e.
    that the ones we care about are the ones that don't start with a
    capital letter or an underscore).

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        The reader object created using the beaconroot example script.

    Returns
    -------
    out_dict : numpy.ndarray of str
        The keys corresponding to the information from the header of the reader.
    '''
    try:
        if reader == None:
            reader = Reader(os.environ['BEACON_DATA'],int(os.path.basename(glob.glob(os.environ['BEACON_DATA']+'/*')[0]).split('run')[-1]) ) #Just picks the first listed run in the default data directory. 
        raw_keys = numpy.array(list(type(reader.header()).__dict__.keys()))
        parsed_keys = []
        for key in raw_keys:
            if '__' in key:
                continue
            if key[0].isupper():
                continue
            parsed_keys.append(key)
        return numpy.array(parsed_keys)
    except Exception as e:
        print('Error trying to get keys:')
        print(e)
        return

def getStatusKeys(reader=None):
    '''
    Get the keys for the status portion of reader files.  If reader
    is None then this will just initiate its own reader just to get the class
    type to pull the keys.  This assumes that the beacon data is stored
    at the location specified by the environment variable $BEACON_DATA.

    Note that this also assumes particular formatting for the keys (i.e.
    that the ones we care about are the ones that don't start with a
    capital letter or an underscore).

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        The reader object created using the beaconroot example script.

    Returns
    -------
    out_dict : numpy.ndarray of str
        The keys corresponding to the information from the status of the reader.
    '''
    try:
        if reader == None:
            reader = examples.beacon_data_reader.Reader(os.environ['BEACON_DATA'],int(os.path.basename(glob.glob(os.environ['BEACON_DATA']+'/*')[0]).split('run')[-1]) ) #Just picks the first listed run in the default data directory. 
        raw_keys = numpy.array(list(type(reader.status()).__dict__.keys()))
        parsed_keys = []
        for key in raw_keys:
            if '__' in key:
                continue
            if key[0].isupper():
                continue
            parsed_keys.append(key)
        return numpy.array(parsed_keys)
    except Exception as e:
        print('Error trying to get keys:')
        print(e)
        return

def getHeaderDict(reader):
    '''
    Get the header from the reader object and returns it in a dict.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        The reader object created using the beaconroot example script.

    Returns
    -------
    out_dict : dict
        The dict containing the information from the header of the reader.  Note
        that some entries to this dict may be malformed because they are not stored
        in proper way in the header file.  Eventually exceptions should be added
        for these keys if it is deemed reasonable. 
    '''
    out_dict = {}
    keys = getHeaderKeys(reader)
    for key in keys:
        out_dict[key] = getattr(reader.header(),key)
    return out_dict

def getStatusDict(reader):
    '''
    Get the status from the reader object and returns it in a dict.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        The reader object created using the beaconroot example script.

    Returns
    -------
    out_dict : dict
        The dict containing the information from the status of the reader.  
    '''
    out_dict = {}
    keys = getStatusKeys(reader)
    for key in keys:
        out_dict[key] = getattr(reader.status(),key)
    return out_dict

def getReaderDict(reader):
    '''
    Get the attrs from the reader object and returns it in a dict.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        The reader object created using the beaconroot example script.

    Returns
    -------
    out_dict : dict
        The dict containing the information from the attrs of the reader.  
    '''
    out_dict = {}
    raw_keys = numpy.array(list(reader.__dict__.keys()))
    parsed_keys = []
    for key in raw_keys:
        if '__' in key:
            continue
        parsed_keys.append(key)
    for key in parsed_keys:
        out_dict[key] = getattr(reader,key)
    return out_dict

def getEventIds(reader, trigger_type=None):
    '''
    Will get a list of eventids for the given reader, but only those matching the trigger
    types supplied.  If trigger_type  == None then all eventids will be returned. 
    trigger_type:
    1 Software
    2 RF
    3 GPS
    '''
    if trigger_type == None:
        trigger_type = numpy.array([1,2,3])
    elif type(trigger_type) == int:
        trigger_type = numpy.array([trigger_type])

    eventids = []
    for trig in trigger_type:
        N = reader.head_tree.Draw("Entry$","trigger_type==%i"%trig,"goff") 
        eventids.append(numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int))

    eventids = numpy.sort(eventids)
    return eventids


if __name__ == '__main__':
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = 367 #Selects which run to examine
    reader = examples.beacon_data_reader.Reader(datapath,run)

    print('\nReader:')
    d = getReaderDict(reader)
    pprint(d)
    print('\nHeader:')
    h = getHeaderDict(reader)
    pprint(h)
    print('\nStatus:')
    s = getStatusDict(reader)
    pprint(s)

