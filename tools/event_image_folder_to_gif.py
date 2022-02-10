#!/usr/bin/env python3
#Execute in a folder to turn all images into a single gif.  This is currently designed to only pull images with the
#expected format rNUMBEReNUMBER.png

import sys
import os
from PIL import Image
from beacon.tools.flipbook_reader import flipbookToDict, concatenateFlipbookToArray
from time import time
import glob

if __name__ == '__main__':
    outpath = './' #Should get reset
    if len(sys.argv) >= 2:
        if len(sys.argv) >= 3:
            filepaths = []
            for i, arg in enumerate(sys.argv):
                if i == 0:
                    continue
                elif i == len(sys.argv) - 1:
                    outpath = arg
                else:
                    if os.path.isdir(arg):
                        filepaths.append(arg)
        else:
            filepaths = [str(sys.argv[1])]
    else:
        filepaths = ['./']

    for filepath in filepaths:
        print('Attempting to make gif from folder:\n',filepath)
        event_array = concatenateFlipbookToArray(flipbookToDict(filepath))

        gif_name = os.path.join(outpath, os.path.split(os.path.realpath(filepath))[-1] + '_%i.gif'%time())

        # Create the frames
        frames = []
        imgs = [os.path.join(filepath,'r%ie%i.png'%(event_array['run'][i], event_array['eventid'][i])) for i in range(len(event_array))]

        size_ratio = 0.25

        for i in imgs:
            new_frame = Image.open(i)
            width = int(size_ratio*new_frame.size[0])
            height = int(size_ratio*new_frame.size[1])
            new_frame = new_frame.resize((width, height), Image.ANTIALIAS)
            frames.append(new_frame)
         
        # Save into a GIF file that loops forever
        frames[0].save(gif_name, format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=200, loop=0,
                       interlace=False,
                       optimize=True)


        