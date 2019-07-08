import numpy as np
from PIL import Image
import matplotlib.pyplot as plt   
from nengo.dists import Uniform
import nengo
import math
from stp_ocl_implementation import *
import os, inspect
from nengo_extras.vision import Gabor, Mask
#import random
from random import randint


#set this if you are using nengo OCL, otherwise comment out
platform = cl.get_platforms()[0]   #select platform, should be 0
device=platform.get_devices()[2]   #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3)
context=cl.Context([device])
    
#loads images in imagearr (images created using the psychopy package)
N=2000
diameter=col=row=128  #width and height of images
angles=np.arange(-90,90,1)  #angle of the grating, 999 is the bullseye 'ping'
#angles=np.append(angles, 999)
phases=np.arange(0,1,0.1)
imagearr=np.zeros((0,diameter**2))

for phase in phases:
    for angle in angles:
        name="Stimuli/stim"+str(angle)+"_"+str(round(phase,1))+".png"
        img=Image.open(name)
        img=np.array(img.convert('L'))
        imagearr=np.vstack((imagearr,img.ravel())) 
        
name="Stimuli/stim999.png"
img=Image.open(name)
img=np.array(img.convert('L'))
imagearr=np.vstack((imagearr,img.ravel())) 
#normalize to be between -1 and 1
imagearr=imagearr/255
imagearr=2 * imagearr - 1
#set default input
stim = 0
probe = 0 
cued = True
#input functions
full_sim=False
def input_func(t):
    t=t-.05
    if t > 0 and t < .25:
        return imagearr[stim,:]/5
    elif t > 1.15 and t < 1.25:
        return imagearr[181,:]/5
    elif t > 1.65 and t < 1.90:
        return imagearr[probe,:]/5
    else:
        return np.zeros(128*128) 
    
    
def input_func2(t):
    t=t-.05
    if t>0 and t<0.25:
        return imagearr[stim,:]/5
    if t>1.45 and t<1.65:
        return imagearr[probe,:]/5
    else:
        return np.zeros(128*128)    
    
def input_func3(t):
    t=t-.05
    if t > 0 and t < .25:
        return imagearr[stim,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[1800,:]/50
    elif t > 2.65 and t < 2.90:
        return imagearr[probe,:]/100
    else:
        return np.zeros(128*128) 
        
def reactivate_func(t):
    if t>1.400 and t<1.42 and cued==True:
        return np.ones(N)*0.0200
    else:
        return np.zeros(N)
        
F=np.arange(-90,90,1)
Fa=F
for i in range(0,9):
    Fa=np.append(Fa,F)

Frad=(Fa/90)*math.pi
Sin=np.sin(Frad)
Cos=np.cos(Frad)
answers=np.vstack((Sin,Cos))
modelseed=0
runs=2881
initialangle_c=np.zeros(runs*14)
initialangle_uc=np.zeros(runs*14)
angle_index=0
for run in range(2304, 2880):
    print(run)
    
    #create new gabor filters every 5 runs
    if (run%96==0):
        #seed=run #random.randint(0,1000)
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
    
    #STSP network
    with nengo.Network(seed=modelseed) as model:
   
            
        inputNode=nengo.Node(input_func3)     
        reactivate=nengo.Node(reactivate_func)  
        
        #vision (sensory) and memory
        vision = nengo.Ensemble(N, bases, encoders=e, intercepts=Uniform(0.01, .1),radius=1)
        memory = nengo.Ensemble(N, bases,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1)
     
        #input connection
        nengo.Connection(inputNode,vision,transform=U[:,:bases].T)
        nengo.Connection(reactivate,memory.neurons)
        #learning connections
        nengo.Connection(vision, memory, transform=.1)
        nengo.Connection(memory, memory,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))
        
        samples=10000
        sinAcosA=nengo.dists.UniformHypersphere(surface=True).sample(samples,2)
        thetaA=np.arctan2(sinAcosA[:,0],sinAcosA[:,1])
        thetaDiff=(90*np.random.random(samples)-45)/180*np.pi
        thetaB=thetaA+thetaDiff
        sinBcosB=np.vstack((np.sin(thetaB),np.cos(thetaB)))
        scale=np.random.random(samples)*0.9+0.1
        sinBcosB=sinBcosB*scale
        ep=np.hstack((sinAcosA,sinBcosB.T))

        #need both sin, cosine of vision and memory
        allens = nengo.Ensemble(n_neurons=2000, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1))
      
        #theta
        def decision_func(v):
            x1,y1,x2,y2=v
            if np.linalg.norm(v)<0.01:
                return 0
            else:
                theta1=np.arctan2(x1,y1)
                theta2=np.arctan2(x2,y2)
                return (theta2-theta1)*90/math.pi
      
       
        def arctan_func(v):
            yA, xA, yB, xB=v
            z = np.arctan2(yA, xA) - np.arctan2(yB, xB)
            pos_ans = [z, z+2*np.pi, z-2*np.pi]
            i = np.argmin(np.abs(pos_ans))
            return pos_ans[i]*90/math.pi
        
           
        #connect sin and cosine to allens
        nengo.Connection(memory, allens[2:],eval_points=compressed_im,function=answers.T)
        nengo.Connection(vision, allens[:2],eval_points=compressed_im,function=answers.T)
    
        #difference in orientation
        dtheta = nengo.Ensemble(n_neurons=2000,  dimensions=1,radius=45)#,intercepts=Uniform(0.01,.1))
         nengo.Connection(allens, dtheta, eval_points=ep, scale_eval_points=False, function=arctan_func)

        #probe dtheta
        p_dtheta=nengo.Probe(dtheta, synapse=0.01)
        p_mem=nengo.Probe(memory, synapse=0.01)
        p_sen=nengo.Probe(vision, synapse=0.01)
       # p_allens=nengo.Probe(allens, synapse=0.01)
        
      #  p_theta=nengo.Probe(theta,synapse=0.01)
      #  p_dtheta2=nengo.Probe(dtheta2,synapse=0.01)
       

       # run in python + save
        
        
        
        nengo_gui_on = __name__ == 'builtins' #python3

    if not(nengo_gui_on):

        cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/data/' # script path
    
        sim = StpOCLsimulator(network=model, seed=run, context=context,progress_bar=False) #set progress bar to false so nohup doesnt create a gigantic output file
        probelist=[-42, -33, -25, -18, -12, -7, -3, 3, 7, 12, 18, 25, 33, 42]
        #probelist=[42]
        
        def norm_p(p):
            if p<0:
                return 180+p
            if p>180:
                return p-180
            else:
                return p
     
        def cosine_sim(a,b):
            out=np.zeros(a.shape[0])
            for i in range(0,  a.shape[0]):
                if abs(np.linalg.norm(a[i])) > 0.05:
                    out[i]=np.dot(a[i], b)/(np.linalg.norm(a[i])*np.linalg.norm(b))
            return out
        
        for anglediff in probelist:
        
         #---------
         #cued module  
            #reset simulator, clean probes thoroughly
  
            #set probe and stim
            stim=randint(0, 179)
            probe=stim+anglediff
            probe=norm_p(probe)
            #random phase
            or_stim=stim
            stim=stim+(180*randint(0, 9))
            probe=probe+(180*randint(0, 9))
            #run sim qued
            cued=True
            initialangle_c[angle_index]=stim
          
            sim.run(3)
            np.savetxt(cur_path+"81_Diff_Theta_%i_run_%i.csv" % (anglediff,(run)), sim.data[p_dtheta][2500:2999,:], delimiter=",")
            sim.reset()
            for probe2 in sim.model.probes:
                del sim._probe_outputs[probe2][:]
            del sim.data
            sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
                
            if (anglediff==42 and full_sim==True):
                diffs=[0,3,7,12,18,25,33,42]
                for dif in diffs:
                   stim_temp=norm_p(or_stim+dif)+(180*randint(0, 9))
                   cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[stim_temp,:]/100, U[:,:bases]))
                   np.savetxt(cur_path+"79_cs_sen_cued_stim_%i_run_%i.csv" % (dif,run), cs, delimiter=",")
                
                cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[1800,:]/100, U[:,:bases]))
                np.savetxt(cur_path+"79_cs_sen_cued_stim_999_run_%i.csv" % (run), cs, delimiter=",")
                 
         
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
              
                diffs=[0,3,7,12,18,25,33,42]
                for dif in diffs:
                   stim_temp=norm_p(or_stim+dif)+(180*randint(0, 9))
                   cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[stim_temp,:]/100, U[:,:bases]))
                   np.savetxt(cur_path+"79_cs_sen_uncued_stim_%i_run_%i.csv" % (dif,run), cs, delimiter=",")
                
                cs=cosine_sim(sim.data[p_sen],np.dot(imagearr[1800,:]/100, U[:,:bases]))
                np.savetxt(cur_path+"79_cs_sen_uncued_stim_999_run_%i.csv" % (run), cs, delimiter=",")
                 
                sim.reset()
                for probe2 in sim.model.probes:
                    del sim._probe_outputs[probe2][:]
                del sim.data
                sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
                

            angle_index=angle_index+1
            
np.savetxt(cur_path+"80_initial_angles_cued_%i_runs.csv" % (run), initialangle_c,delimiter=",")
#np.savetxt(cur_path+"76_initial_angles_uncued_%i_runs.csv" % (run), initialangle_uc,delimiter=",")
        

          
    
    
