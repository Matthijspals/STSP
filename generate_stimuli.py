import matplotlib.pyplot as plt   
from PIL import Image
from random import randint
import psychopy.visual
import psychopy.event
import numpy as np
#Create png images using Psychopy

#4.385 cycles

pixels=128 #width and height of images

diameter=pixels 
spatialfr=4.385/pixels
angles=np.arange(-90,91,1)
phases=np.arange(0,1,1)
win = psychopy.visual.Window(
    size=[diameter, diameter],
    units="pix",
    fullscr=False,
    color=(0,0,0)
)

grating = psychopy.visual.GratingStim(
    win=win,
    units="pix",
    size=[diameter, diameter]
)

grating.sf = spatialfr

grating.mask = "circle"

grating.contrast = 0.5
#generate stimuli
for angle in angles:
    for phase in phases:
        name="stim"+str(angle)+"_"+str(round(phase,1))+".png"
        grating.ori = angle
        grating.phase = randint(0, 9)*0.1 #phase
        grating.draw()
        win.flip()
        win.getMovieFrame()
        win.saveMovieFrames(name)
    
grating.contrast = 1 

#generate ping
grating = psychopy.visual.RadialStim(win=win, mask='circle', size=[diameter, diameter])
grating.setAngularCycles(0)
grating.sf = spatialfr
name="stim999"+".png"
grating.draw()
win.flip()
win.getMovieFrame()
win.saveMovieFrames(name)
#angles.append(999)


    
win.close()