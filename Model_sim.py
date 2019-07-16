import numpy as np
from PIL import Image
import matplotlib.pyplot as plt   
from nengo.dists import Uniform
import nengo
import math
from stp_ocl_implementation import *
import os, inspect
from nengo_extras.vision import Gabor, Mask
from random import randint

#CODE TO RUN SIMULATIONS WITH EXAMPLE MODEL

#SIMULATION CONTROL
full_sim=True #set if you want to run both the  cued and uncued model
runs=2         #amount of runs (note that each run contains 14 trials, one trial for each possible orientation difference between the memory item and the probe)
N=1500         #amount of neurons in memory and sensory layer


#set this if you are using nengo OCL
platform = cl.get_platforms()[0]   #select platform, should be 0
device=platform.get_devices()[2]   #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3)
context=cl.Context([device])


#LOAD INPUT STIMULI (images created using the psychopy package)
#(Stimuli should be in a subfolder named 'Stimuli') 

#width and height of images
diameter=col=row=128 

#load grating stimuli
angles=np.arange(-90,90,1)  #rotation
phases=np.arange(0,1,0.1)   #phase
imagearr=np.zeros((0,diameter**2))
for phase in phases:
    for angle in angles:
        name="Stimuli/stim"+str(angle)+"_"+str(round(phase,1))+".png"
        img=Image.open(name)
        img=np.array(img.convert('L'))
        imagearr=np.vstack((imagearr,img.ravel())) 
 
#also load the  bull's eye 'impulse stimulus'  
name="Stimuli/stim999.png"
img=Image.open(name)
img=np.array(img.convert('L'))
imagearr=np.vstack((imagearr,img.ravel())) 

#normalize to be between -1 and 1
imagearr=imagearr/255
imagearr=2 * imagearr - 1


#INPUT FUNCTIONS

#set default input
stim = 0
probe = 0 
cued = True #cued module is reactivated at t=1.35 s
modelseed=0
#np array to keep track of the input during the simulation runs
initialangle_c=np.zeros(runs*14)
initialangle_uc=np.zeros(runs*14)
angle_index=0

#input stimuli
def input_func3(t):
    t=t-.05
    if t > 0 and t < .25:
        return imagearr[stim,:]/100             #memory item
    elif t > 2.15 and t < 2.25:
        return imagearr[1800,:]/50              #'impulse' at twice the contrast
    elif t > 2.65 and t < 2.90:
        return imagearr[probe,:]/100            #probe
    else:
        return np.zeros(128*128) 

#reactivate memory ensemble with nonspecific signal        
def reactivate_func(t):
    if t>1.400 and t<1.42 and cued==True:
        return np.ones(N)*0.0200
    else:
        return np.zeros(N)


#Create matrix of sine and cosine values associated with the stimuli
#so that we can later specify a transform from stimuli to rotation        
F=np.arange(-90,90,1)
Fa=F
for i in range(0,9):
    Fa=np.append(Fa,F)
Frad=(Fa/90)*math.pi
Sin=np.sin(Frad)
Cos=np.cos(Frad)
answers=np.vstack((Sin,Cos))


#SIMULATION
for run in range(0, runs):
    print(run)
    
    #create new gabor filters every 96 runs to simulate a new participant
    if (run%96==0):
        #generate gabor filters    
        gabors = Gabor().generate(N, (col/3, row/3))#.reshape(N, -1)
        gabors = Mask((col, row)).populate(gabors, flatten=True).reshape(N, -1)
        gabors=gabors/abs(max(np.amax(gabors),abs(np.amin(gabors))))
        #array to use for SVD
        x=np.vstack((imagearr,gabors))    
        modelseed=modelseed+1    
        #Do SVD    
        U, S, V = np.linalg.svd(x.T)
        print("SVD done")
        bases = 24

        #Use result of SVD to create encoders
        e = np.dot(gabors, U[:,:bases]) 
        compressed_im=np.dot(imagearr[:1800,:]/100, U[:,:bases])
    
    #STSP MODEL
    with nengo.Network(seed=modelseed) as model:
   
        #input nodes   
        inputNode=nengo.Node(input_func3)     
        reactivate=nengo.Node(reactivate_func)  
        
        #sensory and memory ensemble
        sensory = nengo.Ensemble(N, bases, encoders=e, intercepts=Uniform(0.01, .1),radius=1)
        memory = nengo.Ensemble(N, bases,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1)
     
        #input connection
        nengo.Connection(inputNode,sensory,transform=U[:,:bases].T)
        nengo.Connection(reactivate,memory.neurons)
        
        #connect sensory to memory
        nengo.Connection(sensory, memory, transform=.1)
       
        #learning connection (memory to memory)
        nengo.Connection(memory, memory,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))
        
        #decision represents sin, cosine of theta of both sensory and memory ensemble
        decision = nengo.Ensemble(n_neurons=2000, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1))
        
        #connect memory and sensory to decision
        nengo.Connection(memory, decision[2:],eval_points=compressed_im,function=answers.T)
        nengo.Connection(sensory, decision[:2],eval_points=compressed_im,function=answers.T)
    
        #create eval points so that we can go from sine and cosine of theta in sensory and memory layer
        #to the difference in theta between the two
        samples=10000
        sinAcosA=nengo.dists.UniformHypersphere(surface=True).sample(samples,2)
        thetaA=np.arctan2(sinAcosA[:,0],sinAcosA[:,1])
        thetaDiff=(90*np.random.random(samples)-45)/180*np.pi
        thetaB=thetaA+thetaDiff
        sinBcosB=np.vstack((np.sin(thetaB),np.cos(thetaB)))
        scale=np.random.random(samples)*0.9+0.1
        sinBcosB=sinBcosB*scale
        ep=np.hstack((sinAcosA,sinBcosB.T))
        
        #continuous variant of arctan(a,b)-arctan(c,d)
        def arctan_func(v):
            yA, xA, yB, xB=v
            z = np.arctan2(yA, xA) - np.arctan2(yB, xB)
            pos_ans = [z, z+2*np.pi, z-2*np.pi]
            i = np.argmin(np.abs(pos_ans))
            return pos_ans[i]*90/math.pi
        
           
        #output_ens represents the difference in theta decoded from the sensory and memory ensembles
        output_ens = nengo.Ensemble(n_neurons=2000,  dimensions=1,radius=45)
        #connect decision to output_ens
        nengo.Connection(decision, output_ens, eval_points=ep, scale_eval_points=False, function=arctan_func)

        #probes
        p_dtheta=nengo.Probe(output_ens, synapse=0.01)
        p_mem=nengo.Probe(memory, synapse=0.01)
        p_sen=nengo.Probe(sensory, synapse=0.01)
         

        nengo_gui_on = __name__ == 'builtins' #python3
    # run in python + save
    if not(nengo_gui_on):

        cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/data/' #store output in data subfolder
    
        #use StpOCLsimulator to make use of the Nengo OCL implementation of STSP
        sim = StpOCLsimulator(network=model, seed=run, context=context,progress_bar=False) #set progress bar to false so nohup doesnt create a gigantic output file
        
        #orientation differences between probe and memory item
        probelist=[-42, -33, -25, -18, -12, -7, -3, 3, 7, 12, 18, 25, 33, 42]
        
        #normalise stimuli to be between 0 and 180 degrees orientation
        def norm_p(p):
            if p<0:
                return 180+p
            if p>180:
                return p-180
            else:
                return p
     
        #Calculate normalised cosine similarity and avoid divide by 0 errors
        def cosine_sim(a,b):
            out=np.zeros(a.shape[0])
            for i in range(0,  a.shape[0]):
                if abs(np.linalg.norm(a[i])) > 0.05:
                    out[i]=np.dot(a[i], b)/(np.linalg.norm(a[i])*np.linalg.norm(b))
            return out
        
        #run a trial with each possible orientation difference
        for anglediff in probelist:
        
         #---------
         #cued module  
            
  
            #set probe and stim
            stim=randint(0, 179)
            probe=stim+anglediff
            probe=norm_p(probe)
            #random phase
            or_stim=stim
            stim=stim+(180*randint(0, 9))
            probe=probe+(180*randint(0, 9))
            
            #run sim cued
            cued=True
            #store orientation
            initialangle_c[angle_index]=stim
            #run simulation
            sim.run(3)
            #store output
            np.savetxt(cur_path+"82_Diff_Theta_%i_run_%i.csv" % (anglediff,(run)), sim.data[p_dtheta][2500:2999,:], delimiter=",")
            
            #reset simulator, clean probes thoroughly
            sim.reset()
            for probe2 in sim.model.probes:
                del sim._probe_outputs[probe2][:]
            del sim.data
            sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
            
            #store cosine simularity on trial with a orientation difference of 42 degrees between memory item and probe
            if (anglediff==42 and full_sim==True):
                diffs=[0,3,7,12,18,25,33,42]
                for dif in diffs:
                   stim_temp=norm_p(or_stim+dif)+(180*randint(0, 9))
                   cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[stim_temp,:]/100, U[:,:bases]))
                   np.savetxt(cur_path+"82_cs_sen_cued_stim_%i_run_%i.csv" % (dif,run), cs, delimiter=",")
                
                cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[1800,:]/100, U[:,:bases]))
                np.savetxt(cur_path+"82_cs_sen_cued_stim_999_run_%i.csv" % (run), cs, delimiter=",")
                 
         
                #---------
                #uncued module
                sim.reset()
                for probe2 in sim.model.probes:
                    del sim._probe_outputs[probe2][:]
                del sim.data
                sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
                
                stim=randint(0, 179)
                probe=stim+anglediff
                probe=norm_p(probe)
                #random phase
                or_stim=stim
                stim=stim+(180*randint(0, 9))
                probe=probe+(180*randint(0, 9))
                #run sim unqued
                cued=False
                #initialangle_uc[angle_index]=stim
                sim.run(3)
                
                #store cosine simularity
                diffs=[0,3,7,12,18,25,33,42]
                for dif in diffs:
                   stim_temp=norm_p(or_stim+dif)+(180*randint(0, 9))
                   cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[stim_temp,:]/100, U[:,:bases]))
                   np.savetxt(cur_path+"82_cs_sen_uncued_stim_%i_run_%i.csv" % (dif,run), cs, delimiter=",")
                
                cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[1800,:]/100, U[:,:bases]))
                np.savetxt(cur_path+"82_cs_sen_uncued_stim_999_run_%i.csv" % (run), cs, delimiter=",")
                
                
                #reset simulator, clean probes thoroughly 
                sim.reset()
                for probe2 in sim.model.probes:
                    del sim._probe_outputs[probe2][:]
                del sim.data
                sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
                

            angle_index=angle_index+1
            
np.savetxt(cur_path+"82_initial_angles_cued_%i_runs.csv" % (run), initialangle_c,delimiter=",")
 

          
    
    
