#!/usr/bin/env python3
'''
This contains a function to help read in flipbooks of event images into a dictionary. 
'''
import os
import sys
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import numpy
import copy

def fieldsView(array, fields):
    return array.getfield(numpy.dtype(
        {name: array.dtype.fields[name] for name in fields}
    ))

def flipbookToDictRecursive(path, ignore_runs=[]):
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
            out_dict[key] = flipbookToDictRecursive(_path, ignore_runs=ignore_runs)
            out_dict[key]['events'] = out_dict[key]['events'][numpy.argsort(out_dict[key]['events'])]
        elif key[0] == 'r' and len(key.split('e')) == 2 and '.png' in key:
            run     = int(key.split('e')[0].replace('r',''))
            eventid = int(key.split('e')[1].replace('.png',''))
            out_dict['events'] = numpy.append(out_dict['events'], numpy.array((run, eventid), dtype=event_dtype))

    out_dict['events'] = out_dict['events'][numpy.lexsort((out_dict['events']['eventid'],out_dict['events']['run']))]
    out_dict['events'] = out_dict['events'][~numpy.isin(out_dict['events']['run'],ignore_runs)]
    out_dict['eventids_dict'] = {}
    for run in numpy.unique(out_dict['events']['run']):
        if run in ignore_runs:
            continue
        else:
            out_dict['eventids_dict'][run] = out_dict['events']['eventid'][out_dict['events']['run'] == run]
    return copy.deepcopy(out_dict)

def flipbookToDict(path, ignore_runs=[]):
    '''
    '''
    out_dict = flipbookToDictRecursive(path, ignore_runs=ignore_runs)
    out_dict['unsorted'] = {}
    out_dict['unsorted']['events'] = out_dict['events']
    del out_dict['events']
    return copy.deepcopy(out_dict)

def concatenateEventDictToArray(eventids_dict, ignore_runs=[]):
    '''
    This replicates behaviour of concatenateFlipbookToArray for dictionaries that are already in the eventids_dict
    format. of {run:[eventids],run:[eventids]}.
    '''
    event_dtype = numpy.dtype([('run','i'),('eventid','i'), ('key', '<U16')])
    out_array = numpy.array([],dtype=event_dtype)
    for run, eventids in eventids_dict.items():
        if run in ignore_runs:
            continue
        _out_array = numpy.zeros(len(eventids),dtype=event_dtype)
        _out_array['run'] = run
        _out_array['eventid'] = eventids
        _out_array['key'] = 'unsorted'
        out_array = numpy.append(out_array,_out_array)
    return numpy.sort(numpy.unique(out_array),order=('run','eventid'))


def concatenateFlipbookToArray(flipbook, ignore_runs=[], parent_key=''):
    '''
    takes an existing event flipbook and takes the event arrays from each dub directory and puts them into a single 
    array.
    '''
    try:
        out_array = None
        for key in list(flipbook.keys()):
            if key == 'events':
                if out_array is None:
                    out_array = flipbook['events']
                else:
                    out_array = numpy.append(out_array, flipbook['events'])
            else:
                if type(flipbook[key]) is dict:
                    _out_array = concatenateFlipbookToArray(flipbook[key], ignore_runs=ignore_runs, parent_key=key)
                    if _out_array is not None:
                        if 'key' not in _out_array.dtype.names:
                            _out_array = numpy.lib.recfunctions.rec_append_fields(_out_array, 'key', numpy.array([key]*len(_out_array), dtype='<U16')) #hopefully adds the key of the deepest folder this event exists in. 

                        if out_array is None:
                            out_array = _out_array
                        else:
                            try:
                                if len(out_array) == 0:
                                    out_array = _out_array
                                else:
                                    if out_array.dtype != _out_array.dtype:
                                        out_array = numpy.lib.recfunctions.rec_append_fields(out_array, 'key', numpy.array([parent_key]*len(out_array), dtype='<U16')) #hopefully adds the key of the deepest folder this event exists in. 
                                    out_array = numpy.append(out_array, _out_array)
                            except Exception as e:
                                print(e)
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno)
                                import pdb; pdb.set_trace()
        if out_array is not None:
            out_array = numpy.sort(numpy.unique(out_array),order=('run','eventid'))
            if 'key' in out_array.dtype.names:
                # Handle the scenario where an event is in multiple folders, and give it a key with multiple values.
                concatenated_out_array = numpy.array([],dtype=out_array.dtype)
                for run in numpy.unique(out_array['run']):
                    r = out_array[out_array['run'] == run]
                    for eventid in numpy.unique(r['eventid']):
                        e = r[r['eventid'] == eventid]
                        for k_index, k in enumerate(e['key']):
                            if k_index == 0:
                                key = k
                            else:
                                key = key + '+' + k
                        concatenated_out_array = numpy.append(concatenated_out_array, numpy.array([(run,eventid, key)],dtype=concatenated_out_array.dtype))
                out_array = concatenated_out_array[~numpy.isin(concatenated_out_array['run'],ignore_runs)]
            else:
                out_array = out_array[~numpy.isin(out_array['run'],ignore_runs)]

        return out_array
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        import pdb; pdb.set_trace()



def concatenateFlipbookToDict(flipbook, ignore_runs=[]):
    '''
    Does what concatenateFlipbookToArray does but to a eventids_dict.
    '''
    if str in [type(k) for k in flipbook.keys()]:
        sorted_array = concatenateFlipbookToArray(flipbook, ignore_runs=ignore_runs)
    elif numpy.all([numpy.issubdtype(k, numpy.integer) for k in flipbook.keys()]):
        print('Assuming passed "flipbook" as eventids_dict instead of flipbook')
        sorted_array = concatenateEventDictToArray(flipbook, ignore_runs=ignore_runs)
    else:
        sorted_array = concatenateFlipbookToArray(flipbook, ignore_runs=ignore_runs)
    out_dict = {}
    for run in numpy.unique(sorted_array['run']):
        out_dict[run] = numpy.unique(sorted_array[sorted_array['run'] == run]['eventid'])
    return copy.deepcopy(out_dict)

if __name__ == "__main__":
    path = '/home/dsouthall/scratch-midway2/event_flipbook_1642725413'#os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'tools', )

    out_dict = flipbookToDict(path)