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
full_sim=False #set if you want to run both the  cued and uncued model
runs=2880        #amount of runs (note that each run contains 14 trials, one trial for each possible orientation difference between the memory item and the probe)
Ns=1000        #amount of neurons in memory and sensory layer
Nm=1500
Nc=1500
Nd=1000
sim_no="95"      #simulation number (used in the names of the outputfiles)

#set this if you are using nengo OCL
platform = cl.get_platforms()[0]   #select platform, should be 0
device=platform.get_devices()[0]   #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3)
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

#t 250 ms memory items, 800 ms fixation, 20 ms reacitvation, 1080 ms fixation, 100 ms impulse, 400 ms fixation, and 250 ms probe.


def input_func_L(t):
    #t=t-.05
    if t > 0 and t < 0.25:
        return imagearr[memory_item_L,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[1800,:]/50
    elif t > 2.65 and t < 2.90:
        return imagearr[probe,:]/100
    else:
        return np.zeros(128*128) 

    
#reactivate memory ensemble with nonspecific signal        
def reactivate_func(t):
    if t>1.050 and t<1.070:
        return np.ones(Nm)*0.0200
    else:
        return np.zeros(Nm)


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
for run in range(2496, runs):
    print(run)
    
    #create new gabor filters every 96 runs to simulate a new participant
    if (run%96==0):
        gabors_L = Gabor().generate(Ns, (col/3, row/3))#.reshape(N, -1)
        gabors_L = Mask((col, row)).populate(gabors_L, flatten=True).reshape(Ns, -1)
        gabors_L=gabors_L/abs(max(np.amax(gabors_L),abs(np.amin(gabors_L))))
 
        #gabors_R = Gabor().generate(Ns, (col/3, row/3))#.reshape(N, -1)
        #gabors_R = Mask((col, row)).populate(gabors_R, flatten=True).reshape(Ns, -1)
        #gabors_R=gabors_R/abs(max(np.amax(gabors_R),abs(np.amin(gabors_R))))
        #array to use for SVD
        x_L=np.vstack((imagearr,gabors_L))    
        #x_R=np.vstack((imagearr,gabors_R))    
      

        #Do SVD    
        U_L, S_L, V_L = np.linalg.svd(x_L.T)
        #U_R, S_R, V_R = np.linalg.svd(x_R.T)
        print("SVD done")
        bases = 24

        #Use result of SVD to create encoders
        bases = 24
        #Use result of SVD to create encoders
        e_L = np.dot(gabors_L, U_L[:,:bases]) 
        compressed_im_L=np.dot(imagearr[:1800,:]/100, U_L[:,:bases])
        #e_R = np.dot(gabors_R, U_R[:,:bases]) 
        #compressed_im_R=np.dot(imagearr[:1800,:]/100, U_R[:,:bases])
    
    #STSP MODEL
    with nengo.Network(seed=modelseed) as model:
   
        #input nodes   
        inputNode_L=nengo.Node(input_func_L)     
        #inputNode_R=nengo.Node(input_func_R)
        
        reactivate=nengo.Node(reactivate_func)
        
        #sensory and memory ensemble
        sensory_L = nengo.Ensemble(Ns, bases, encoders=e_L, intercepts=Uniform(0.01, .1),radius=1)
        memory_L = nengo.Ensemble(Nm, bases,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1)
        
        #sensory_R = nengo.Ensemble(Ns, bases, encoders=e_R, intercepts=Uniform(0.01, .1),radius=1)
        #memory_R = nengo.Ensemble(Nm, bases,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1)
     
        #input connection
        nengo.Connection(inputNode_L,sensory_L,transform=U_L[:,:bases].T)
        #nengo.Connection(inputNode_R,sensory_R,transform=U_R[:,:bases].T)
        
        nengo.Connection(reactivate,memory_L.neurons)
        
        #connect sensory to memory
        nengo.Connection(sensory_L, memory_L, transform=.1)
        #nengo.Connection(sensory_R, memory_R, transform=.1)
       
        #learning connection (memory to memory)
        nengo.Connection(memory_L, memory_L,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))
        #nengo.Connection(memory_R, memory_R,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))
        
        #comparison represents sin, cosine of theta of both sensory and memory ensemble
        comparison_L = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1))
        #comparison_R = nengo.Ensemble(Nd, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1))
        
        #connect memory and sensory to comparison
        nengo.Connection(memory_L, comparison_L[2:],eval_points=compressed_im_L,function=answers.T)
        nengo.Connection(sensory_L, comparison_L[:2],eval_points=compressed_im_L,function=answers.T)
        #nengo.Connection(memory_R, comparison_R[2:],eval_points=compressed_im_R,function=answers.T)
        #nengo.Connection(sensory_R, comparison_R[:2],eval_points=compressed_im_R,function=answers.T)
    
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
        decision = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45)
        #connect decision to output_ens
        nengo.Connection(comparison_L, decision, eval_points=ep, scale_eval_points=False, function=arctan_func)
        #nengo.Connection(comparison_R, decision, eval_points=ep, scale_eval_points=False, function=arctan_func)
         
        p_dec=nengo.Probe(decision, synapse=0.01)
        
        
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
 
            
  
            #set probe and stim
            memory_item_L=randint(0, 179)
           # memory_item_R=randint(0, 179)
            probe=memory_item_L+anglediff
            probe=norm_p(probe)
            #random phase
            or_memory_item_L=memory_item_L
           # or_memory_item_R=memory_item_R
            memory_item_L=memory_item_L+(180*randint(0, 9))
           # memory_item_R=memory_item_R+(180*randint(0, 9))
            probe=probe+(180*randint(0, 9))
            
          
            #store orientation
            initialangle_c[angle_index]=or_memory_item_L
           # initialangle_uc[angle_index]=or_memory_item_R
            
            #run simulation
            sim.run(3)
            #store output
            np.savetxt(cur_path+sim_no+"_Diff_Theta_%i_run_%i.csv" % (anglediff,(run)), sim.data[p_dec][2500:2999,:], delimiter=",")
            
            #reset simulator, clean probes thoroughly
            sim.reset()
            for probe2 in sim.model.probes:
                del sim._probe_outputs[probe2][:]
            del sim.data
            sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
            


            angle_index=angle_index+1
            
        np.savetxt(cur_path+sim_no+"_initial_angles_cued_%i_runs.csv" % (runs+2), initialangle_c,delimiter=",")
       # np.savetxt(cur_path+sim_no+"_initial_angles_uncued_%i_runs.csv" % (runs), initialangle_uc,delimiter=",")

          
    
    