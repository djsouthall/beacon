'''
This script was basically written for the specific use case of running scripts on
thinlinc that output event values to the clipboard.  Then wanting to open those
events in monutau on my windows machine.  It is a nice use case but I figured
it would save me time, and it was an interesting thing to do. 
'''

import os
import sys
import webbrowser
import numpy
import ast
import win32clipboard

def openMonutau(entries):
    '''
    Will open a chrome window with monutau pages for each event.
    Must be run from directory with chrome.lnk file in it. 
    expects things to be in form of [[run, eventid],[run, eventid]]
    '''
    urls = ['https://users.rcc.uchicago.edu/~cozzyd/monutau/#event&run=%i&entry=%i'%(run,eid) for run, eid in entries]

    #open new instance of chrome
    if len(urls) > 0:
        os.system('chrome.lnk')

        for url_index, url in enumerate(urls):
            if url_index == 0:
                #New instance opens tab, use that
                webbrowser.open(url, new=0, autoraise=True)
            else:
                #Create new tab.
                webbrowser.open(url)

def parseEventidsStr(eid_str):
    '''
    Parses strings of the format:
    array([[  1661,  34206],
           [  1661, 35542],
           [  1661, 36609]], dtype=uint32)
    '''
    entries = numpy.array(ast.literal_eval(eid_str.replace('array(','').replace(', dtype=uint32)','')))
    return entries


possible_plane_1 = '''
array([[  1661,  34206],
       [  1661, 35542],
       [  1661, 36609]], dtype=uint32)
'''

possible_plane_2 = '''
array([[  1651,  49],
       [  1651, 6817],
       [  1651, 12761]], dtype=uint32)
'''

possible_plane_3 = '''
array([[  1662,  58427],
       [  1662, 58647]], dtype=uint32)
'''




#Defining as block string that is parsed for quick copy and paste
eventids_str = '''
array([[ 1773, 14413],
       [ 1773, 14540],
       [ 1773, 14590]], dtype=uint32)
'''

if __name__=="__main__":
    try:
        # get clipboard data
        print('Attempting to open monutau for events in clipboard')
        win32clipboard.OpenClipboard()
        data = win32clipboard.GetClipboardData().replace('"','')
        win32clipboard.CloseClipboard()
        print(data)
        openMonutau(parseEventidsStr(data))
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        openMonutau(parseEventidsStr(eventids_str))


    
