#!/usr/bin/env python3
'''
This contains a function to help read in flipbooks of event images into a dictionary. 
'''
import os
import sys
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import numpy
import copy


def flipbookToDictRecursive(path):
    '''
    '''
    event_dtype = numpy.dtype([('run','i'),('eventid','i')])
    out_dict = {}
    out_dict['events'] = numpy.array([], dtype=event_dtype)
    for key in os.listdir(path):
        if '.txt' in key:
            out_dict[key.replace('.txt','')] = str(open(os.path.join(path, key),'r').read())
        elif os.path.isdir(os.path.join(path, key)):
            _path = os.path.join(path, key)
            out_dict[key] = flipbookToDictRecursive(_path)
            out_dict[key]['events'] = out_dict[key]['events'][numpy.argsort(out_dict[key]['events'])]
        elif key[0] == 'r' and len(key.split('e')) == 2 and '.png' in key:
            run     = int(key.split('e')[0].replace('r',''))
            eventid = int(key.split('e')[1].replace('.png',''))
            out_dict['events'] = numpy.append(out_dict['events'], numpy.array((run, eventid), dtype=event_dtype))

    out_dict['events'] = out_dict['events'][numpy.lexsort((out_dict['events']['eventid'],out_dict['events']['run']))]
    out_dict['eventids_dict'] = {}
    for run in numpy.unique(out_dict['events']['run']):
        out_dict['eventids_dict'][run] = out_dict['events']['eventid'][out_dict['events']['run'] == run]
    return copy.deepcopy(out_dict)

def flipbookToDict(path):
    '''
    '''
    out_dict = flipbookToDictRecursive(path)
    out_dict['unsorted'] = {}
    out_dict['unsorted']['events'] = out_dict['events']
    del out_dict['events']
    return copy.deepcopy(out_dict)

def concatenateFlipbookToArray(flipbook):
    '''
    takes an existing event flipbook and takes the event arrays from each dub directory and puts them into a single 
    array.
    '''
    out_array = None
    for key in list(flipbook.keys()):
        if key == 'events':
            if out_array is None:
                out_array = flipbook['events']
            else:
                out_array = numpy.append(out_array, flipbook['events'])
        else:
            if type(flipbook[key]) is dict:
                _out_array = concatenateFlipbookToArray(flipbook[key])
                if _out_array is not None:
                    if out_array is None:
                        out_array = _out_array
                    else:
                        out_array = numpy.append(out_array, _out_array)
    if out_array is not None:
        out_array = numpy.sort(numpy.unique(out_array),order=('run','eventid'))
    return out_array

def concatenateFlipbookToDict(flipbook):
    '''
    Does what concatenateFlipbookToArray does but to a eventids_dict.
    '''
    sorted_array = concatenateFlipbookToArray(flipbook)
    out_dict = {}
    for run in numpy.unique(sorted_array['run']):
        out_dict[run] = sorted_array[sorted_array['run'] == run]['eventid']
    return copy.deepcopy(out_dict)





if __name__ == "__main__":
    path = '/home/dsouthall/scratch-midway2/event_flipbook_1642725413'#os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'tools', )

    out_dict = flipbookToDict(path)