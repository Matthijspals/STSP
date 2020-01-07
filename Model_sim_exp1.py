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
import nengo.spa as spa
import os.path

#SIMULATION CONTROL for GUI
uncued=False #set if you want to run both the cued and uncued model
load_gabors_svd=True #set to false if you want to generate new ones
store_representations = False #store representations of model runs (for Fig 3 & 4)
store_decisions = False #store decision ensemble (for Fig 5 & 6)
store_spikes_and_resources = False #store spikes, calcium etc. (Fig 3 & 4)

#specify here which sim you want to run if you do not use the nengo GUI
#1 = simulation to generate Fig 3 & 4
#2 = simulation to generate Fig 5 & 6
sim_to_run = 1
sim_no="1"      #simulation number (used in the names of the outputfiles)


#set this if you are using nengo OCL
platform = cl.get_platforms()[0]   #select platform, should be 0
device=platform.get_devices()[1]   #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3)
context=cl.Context([device])



#MODEL PARAMETERS
D = 24  #dimensions of representations
Ns = 1000 #number of neurons in sensory layer
Nm = 1500 #number of neurons in memory layer
Nc = 1500 #number of neurons in comparison
Nd = 1000 #number of neurons in decision


#LOAD INPUT STIMULI (images created using the psychopy package)
#(Stimuli should be in a subfolder named 'Stimuli') 

#width and height of images
diameter=col=row=128 

#load grating stimuli
angles=np.arange(-90,90,1)  #rotation
phases=np.arange(0,1,0.1)   #phase

try:
    imagearr = np.load('Stimuli/all_stims.npy') #load stims if previously generated
except FileNotFoundError: #or generate
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
    
    #imagearr is a (1801, 16384) np array containing all stimuli + the impulse
    np.save('Stimuli/all_stims.npy',imagearr)



#INPUT FUNCTIONS

#set default input
memory_item_cued = 0
probe_cued = 0 
memory_item_uncued = 0
probe_uncued = 0 

#input stimuli
#250 ms memory items | 0-250
#800 ms fixation | 250-1050 
#20 ms reactivation | 1050-1070
#1080 ms fixation | 1070-2150
#100 ms impulse | 2150-2250
#400 ms fixation | 2250-2650
#250 ms probe | 2650-2900
def input_func_cued(t):
    if t > 0 and t < 0.25:
        return imagearr[memory_item_cued,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 2.65 and t < 2.90:
        return imagearr[probe_cued,:]/100
    else:
        return np.zeros(128*128) #blank screen

def input_func_uncued(t):
    if t > 0 and t < 0.25:
        return imagearr[memory_item_uncued,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 2.65 and t < 2.90:
        return imagearr[probe_uncued,:]/100
    else:
        return np.zeros(128*128) #blank screen

#reactivate memory cued ensemble with nonspecific signal        
def reactivate_func(t):
    if t>1.050 and t<1.070:
        return np.ones(Nm)*0.0200
    else:
        return np.zeros(Nm)

#Create matrix of sine and cosine values associated with the stimuli
#so that we can later specify a transform from stimuli to rotation        
Fa = np.tile(angles,phases.size) #want to do this for each phase
Frad = (Fa/90) * math.pi #make radians
Sin = np.sin(Frad)
Cos = np.cos(Frad)
sincos = np.vstack((Sin,Cos)) #sincos

#Create eval points so that we can go from sine and cosine of theta in sensory and memory layer
#to the difference in theta between the two
samples = 10000
sinAcosA = nengo.dists.UniformHypersphere(surface=True).sample(samples,2)
thetaA = np.arctan2(sinAcosA[:,0],sinAcosA[:,1])
thetaDiff = (90*np.random.random(samples)-45)/180*np.pi
thetaB = thetaA + thetaDiff

sinBcosB = np.vstack((np.sin(thetaB),np.cos(thetaB)))
scale = np.random.random(samples)*0.9+0.1
sinBcosB = sinBcosB * scale
ep = np.hstack((sinAcosA,sinBcosB.T))


#continuous variant of arctan(a,b)-arctan(c,d)
def arctan_func(v):
    yA, xA, yB, xB = v
    z = np.arctan2(yA, xA) - np.arctan2(yB, xB)
    pos_ans = [z, z+2*np.pi, z-2*np.pi]
    i = np.argmin(np.abs(pos_ans))
    return pos_ans[i]*90/math.pi



#MODEL

#gabor generation for a particular model-participant
def generate_gabors():

    global e_cued
    global U_cued
    global compressed_im_cued

    global e_uncued
    global U_uncued
    global compressed_im_uncued

    #to speed things up, load previously generated ones
    if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_cued.npz'):
        gabors_svd_cued = np.load('Stimuli/gabors_svd_cued.npz') #load stims if previously generated
        e_cued = gabors_svd_cued['e_cued']
        U_cued = gabors_svd_cued['U_cued']
        compressed_im_cued = gabors_svd_cued['compressed_im_cued']
        print("SVD cued loaded")

    else: #or generate and save

        #cued module
        #for each neuron in the sensory layer, generate a Gabor of 1/3 of the image size
        gabors_cued = Gabor().generate(Ns, (col/3, row/3)) 
        #put gabors on image and make them the same shape as the stimuli
        gabors_cued = Mask((col, row)).populate(gabors_cued, flatten=True).reshape(Ns, -1)
        #normalize
        gabors_cued=gabors_cued/abs(max(np.amax(gabors_cued),abs(np.amin(gabors_cued))))
        #gabors are added to imagearr for SVD
        x_cued=np.vstack((imagearr,gabors_cued))    

        #SVD  
        print("SVD cued started...")
        U_cued, S_cued, V_cued = np.linalg.svd(x_cued.T)
        print("SVD cued done")

        #Use result of SVD to create encoders
        e_cued = np.dot(gabors_cued, U_cued[:,:D]) #encoders
        compressed_im_cued = np.dot(imagearr[:1800,:]/100, U_cued[:,:D]) #D-dimensional vector reps of the images
        compressed_im_cued = np.vstack((compressed_im_cued, np.dot(imagearr[-1,:]/50, U_cued[:,:D])))

        np.savez('Stimuli/gabors_svd_cued.npz', e_cued=e_cued, U_cued=U_cued, compressed_im_cued=compressed_im_cued)

    #same for uncued module
    if uncued:

        if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_uncued.npz'):
            gabors_svd_uncued = np.load('Stimuli/gabors_svd_uncued.npz') #load stims if previously generated
            e_uncued = gabors_svd_uncued['e_uncued']
            U_uncued = gabors_svd_uncued['U_uncued']
            compressed_im_uncued = gabors_svd_uncued['compressed_im_uncued']
            print("SVD uncued loaded")
        else:
            gabors_uncued = Gabor().generate(Ns, (col/3, row/3))#.reshape(N, -1)
            gabors_uncued = Mask((col, row)).populate(gabors_uncued, flatten=True).reshape(Ns, -1)
            gabors_uncued=gabors_uncued/abs(max(np.amax(gabors_uncued),abs(np.amin(gabors_uncued))))
            x_uncued=np.vstack((imagearr,gabors_uncued))    

            print("SVD uncued started...")
            U_uncued, S_uncued, V_uncued = np.linalg.svd(x_uncued.T)
            print("SVD uncued done")
            e_uncued = np.dot(gabors_uncued, U_uncued[:,:D]) 
            compressed_im_uncued=np.dot(imagearr[:1800,:]/100, U_uncued[:,:D])
            compressed_im_uncued = np.vstack((compressed_im_uncued, np.dot(imagearr[-1,:]/50, U_uncued[:,:D])))
            
            np.savez('Stimuli/gabors_svd_uncued.npz', e_uncued=e_uncued, U_uncued=U_uncued, compressed_im_uncued=compressed_im_uncued)


nengo_gui_on = __name__ == 'builtins' #python3

def create_model(seed=None):

    global model
    
    #create vocabulary to show representations in gui
    if nengo_gui_on:
        vocab_angles = spa.Vocabulary(D)
        for name in [0, 3, 7, 12, 18, 25, 33, 42]:
            #vocab_angles.add('D' + str(name), np.linalg.norm(compressed_im_cued[name+90])) #take mean across phases
            v = compressed_im_cued[name+90]
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v /= nrm
            vocab_angles.add('D' + str(name), v) #take mean across phases

        v = np.dot(imagearr[-1,:]/50, U_cued[:,:D])
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        vocab_angles.add('Impulse', v)
    
    #model = nengo.Network(seed=seed)
    model = spa.SPA(seed=seed)
    with model:

        #input nodes
        inputNode_cued=nengo.Node(input_func_cued,label='input_cued')     
        reactivate=nengo.Node(reactivate_func,label='reactivate') 
    
        #sensory ensemble
        sensory_cued = nengo.Ensemble(Ns, D, encoders=e_cued, intercepts=Uniform(0.01, .1),radius=1,label='sensory_cued')
        nengo.Connection(inputNode_cued,sensory_cued,transform=U_cued[:,:D].T)
        
        #memory ensemble
        memory_cued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1,label='memory_cued') 
        nengo.Connection(reactivate,memory_cued.neurons) #potential reactivation
        nengo.Connection(sensory_cued, memory_cued, transform=.1) #.1)
        
        #recurrent STSP connection
        nengo.Connection(memory_cued, memory_cued,transform=1, learning_rule_type=STP(), solver=nengo.solvers.LstsqL2(weights=True))

        #comparison represents sin, cosine of theta of both sensory and memory ensemble
        comparison_cued = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_cued') 
        nengo.Connection(sensory_cued, comparison_cued[:2],eval_points=compressed_im_cued[0:-1],function=sincos.T)
        nengo.Connection(memory_cued, comparison_cued[2:],eval_points=compressed_im_cued[0:-1],function=sincos.T)
       
        #decision represents the difference in theta decoded from the sensory and memory ensembles
        decision_cued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,label='decision_cued') 
        nengo.Connection(comparison_cued, decision_cued, eval_points=ep, scale_eval_points=False, function=arctan_func)

        #same for uncued
        if uncued:
            inputNode_uncued=nengo.Node(input_func_uncued,label='input_uncued')

            sensory_uncued = nengo.Ensemble(Ns, D, encoders=e_uncued, intercepts=Uniform(0.01, .1),radius=1,label='sensory_uncued')
            nengo.Connection(inputNode_uncued,sensory_uncued,transform=U_uncued[:,:D].T)
            
            memory_uncued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1,label='memory_uncued')
            nengo.Connection(sensory_uncued, memory_uncued, transform=.1)
   
            nengo.Connection(memory_uncued, memory_uncued,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))
    
            comparison_uncued = nengo.Ensemble(Nd, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_uncued')
    
            nengo.Connection(memory_uncued, comparison_uncued[2:],eval_points=compressed_im_uncued[0:-1],function=sincos.T)
            nengo.Connection(sensory_uncued, comparison_uncued[:2],eval_points=compressed_im_uncued[0:-1],function=sincos.T)
            
            decision_uncued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,label='decision_uncued') 
            nengo.Connection(comparison_uncued, decision_uncued, eval_points=ep, scale_eval_points=False, function=arctan_func)
     
        #decode for gui
        if nengo_gui_on:
            model.sensory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='sensory_decode')
            for ens in model.sensory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(sensory_cued, model.sensory_decode.input,synapse=None)
     
            model.memory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='memory_decode')
            for ens in model.memory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(memory_cued, model.memory_decode.input,synapse=None)
            
        #probes
        if not(nengo_gui_on):
            if store_representations: #sim 1 trials 1-100
                #p_dtheta_cued=nengo.Probe(decision_cued, synapse=0.01)
                model.p_mem_cued=nengo.Probe(memory_cued, synapse=0.01)
                #p_sen_cued=nengo.Probe(sensory_cued, synapse=0.01)
           
                if uncued:
                    model.p_mem_uncued=nengo.Probe(memory_uncued, synapse=0.01)
                
            if store_spikes_and_resources: #sim 1 trial 1
                model.p_spikes_mem_cued=nengo.Probe(memory_cued.neurons, 'spikes')
                model.p_res_cued=nengo.Probe(memory_cued.neurons, 'resources')
                model.p_cal_cued=nengo.Probe(memory_cued.neurons, 'calcium')
    
                if uncued:
                    model.p_spikes_mem_uncued=nengo.Probe(memory_uncued.neurons, 'spikes')
                    model.p_res_uncued=nengo.Probe(memory_uncued.neurons, 'resources')
                    model.p_cal_uncued=nengo.Probe(memory_uncued.neurons, 'calcium')
            
            if store_decisions: #sim 2
                model.p_dec_cued=nengo.Probe(decision_cued, synapse=0.01)

#PLOTTING CODE
from nengo.utils.matplotlib import rasterplot
from matplotlib import style
from plotnine import *
theme = theme_classic()
plt.style.use('default')

def plot_sim_1(sp_c,sp_u,res_c,res_u,cal_c,cal_u, mem_cued, mem_uncued):

    #FIGURE 31
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)
    
        fig, axes, = plt.subplots(2,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        #spikes, calcium, resources Cued
        ax1=axes[0,0]
        ax1.set_title("Cued Module")
        ax1.set_ylabel('# cell', color='black')
        ax1.set_yticks(np.arange(0,Nm,500))
        ax1.tick_params('y')#, colors='black')
        rasterplot(sim.trange(), sp_c,ax1,colors=['black']*sp_c.shape[0])
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_xlim(0,3)
        ax2 = ax1.twinx()
        ax2.plot(t, res_c, "#00bfc4",linewidth=2)
        ax2.plot(t, cal_c, "#e38900",linewidth=2)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        ax2.set_ylim(0,1.1)

        #spikes, calcium, resources Uncued
        ax3=axes[0,1]
        ax3.set_title("Uncued Module")
        rasterplot(sim.trange(), sp_u,ax3,colors=['black']*sp_u.shape[0])
        ax3.set_xticklabels([])
        ax3.set_xticks([])
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        ax3.set_xlim(0,3)
        ax4 = ax3.twinx()
        ax4.plot(t, res_u, "#00bfc4",linewidth=2)
        ax4.plot(t, cal_u, "#e38900",linewidth=2)
        ax4.set_ylabel('synaptic variables', color="black",size=11)
        ax4.tick_params('y', labelcolor='#333333',labelsize=9,color='#333333')
        ax4.set_ylim(0,1.1)

        #representations cued
        plot_mc=axes[1,0]
        plot_mc.plot(sim.trange(),(mem_cued));

        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(0.0,3.45,0.5))
        plot_mc.set_xticklabels(np.arange(0,3500,500).tolist())
        plot_mc.set_xlabel('time (ms)')
        plot_mc.set_xlim(0,3)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7","#ff66a8", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        #representations uncued
        plot_mu=axes[1,1]

        plot_mu.plot(sim.trange(),(mem_uncued));
        plot_mu.set_xticks(np.arange(0.0,3.45,0.5))
        plot_mu.set_xticklabels(np.arange(0,3500,500).tolist())
        plot_mu.set_xlabel('time (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(0,3)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","3°","7°","12°","18°","25°","33°","42°", "Impulse"], title="Stimulus", bbox_to_anchor=(1.02, -0.25, .30, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)


        fig.set_size_inches(11, 5)
        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        theme.apply(plt.gcf().axes[2])
        theme.apply(plt.gcf().axes[3])
        plt.savefig('Figure_3.eps', format='eps', dpi=1000)
        plt.show()
    
    
    #FIGURE 32
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)
    
        fig, axes, = plt.subplots(1,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)
   
        plot_mc=axes[0]
        plot_mc.set_title("Cued Module")
        plot_mc.plot(sim.trange(),(mem_cued));
        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(2.15,2.35,0.05))
        plot_mc.set_xticklabels(np.arange(0,250,50).tolist())
        plot_mc.set_xlabel('time after onset impulse (ms)')
        plot_mc.set_xlim(2.15,2.3)
        plot_mc.set_ylim(0,0.9)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7","#ff66a8", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        plot_mu=axes[1]
        plot_mu.set_title("Uncued Module")
        plot_mu.plot(sim.trange(),(mem_uncued));
        plot_mu.set_xticks(np.arange(2.15,2.35,0.05))
        plot_mu.set_xticklabels(np.arange(0,250,50).tolist())
        plot_mu.set_xlabel('time after onset impulse (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(2.15,2.30)
        plot_mu.set_ylim(0,0.9)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","3°","7°","12°","18°","25°","33°","42°", "Impulse"], title="Stimulus", bbox_to_anchor=(0.85, 0.25, .55, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(6, 4)

        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        plt.savefig('Figure_4.eps', format='eps', dpi=1000)
        plt.show()    
    
    
    

#SIMULATION
#note that this is split for running a single trial in the nengo gui, and a full simulation

        
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
                
                

if nengo_gui_on:
    generate_gabors() #generate gabors
    create_model(seed=0) #build model

    memory_item_cued = 0 + 90
    probe_cued = 42 + 90 
    memory_item_uncued = 0 + 90
    probe_uncued = 42 + 90


else: #no gui
    
    #path
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/data/' #store output in data subfolder
    
    #simulation 1: recreate fig 3 & 4, 100 trials for both cued and uncued with 0 and 42 degree memory items and probes
    if sim_to_run == 1:
    
        print('Running simulation 1')
        print('')
        
        load_gabors_svd = False #no need to randomize this
        
        ntrials = 100
        store_representations = True
        store_decisions = False
        uncued = True


        #store results        
        templates=np.array([90,93,97,102,108,115,123,132])
        mem_cued = np.zeros((3000,len(templates)+1)) #keep cosine sim for 9 items (templates + impulse)
        mem_uncued = np.zeros((3000,len(templates)+1))
        
        #first, run 100 trials to get average cosine sim
        for run in range(ntrials):
        
            print('Run ' + str(run+1))

            #stimuli
            phase = 180*randint(0, 9)
            memory_item_cued = 0 + 90 + phase
            probe_cued = 42 + 90 + phase
            memory_item_uncued = memory_item_cued
            probe_uncued = probe_cued

            #create new gabor filters every 10 trials
            if run % 10 == 0:
                generate_gabors()
        
            create_model(seed=run)
            sim = StpOCLsimulator(network=model, seed=run, context=context,progress_bar=False)

            #run simulation
            sim.run(3)

            #reset simulator, clean probes thoroughly
            #print(sim.data[model.p_mem_cued].shape)
            #calc cosine sim with templates
            temp_phase = list(templates + phase) + [1800]
            
            for cnt, templ in enumerate(temp_phase):
                mem_cued[:,cnt] += cosine_sim(sim.data[model.p_mem_cued][:,:,],compressed_im_cued[templ,:])
                mem_uncued[:,cnt] += cosine_sim(sim.data[model.p_mem_uncued][:,:,],compressed_im_uncued[templ,:])

            sim.reset()
            for probe2 in sim.model.probes:
                del sim._probe_outputs[probe2][:]
            del sim.data
            sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
            
        
        #average
        mem_cued /= ntrials
        mem_uncued /= ntrials

        #second, run 1 trial to get calcium and spikes
        store_spikes_and_resources = True
        store_representations = False
        create_model(seed=0) #recreate model to change probes
        sim = StpOCLsimulator(network=model, seed=0, context=context,progress_bar=False)

        print('Run ' + str(ntrials+1))
        sim.run(3)

        #store spikes and calcium
        sp_c = sim.data[model.p_spikes_mem_cued]
        res_c=np.mean(sim.data[model.p_res_cued][:,:,],1) #take mean over neurons
        cal_c=np.mean(sim.data[model.p_cal_cued][:,:,],1) #take mean over neurons

        sp_u=sim.data[model.p_spikes_mem_uncued]
        res_u=np.mean(sim.data[model.p_res_uncued][:,:,],1)
        cal_u=np.mean(sim.data[model.p_cal_uncued][:,:,],1)

        #plot
        plot_sim_1(sp_c,sp_u,res_c,res_u,cal_c,cal_u, mem_cued, mem_uncued)
        

    #simulation 2: collect data for fig 5 & 6. 1344 trials for 30 subjects
    if sim_to_run == 2:
    
        load_gabors_svd = False #set to false for real simulation

        n_subj = 30
        trials_per_subj = 1344
        store_representations = False 
        store_decisions = True 
        uncued = False

        #np array to keep track of the input during the simulation runs
        initialangle_c = np.zeros(n_subj*trials_per_subj) #cued
        angle_index=0
        
        #orientation differences between probe and memory item for each run
        probelist=[-42, -33, -25, -18, -12, -7, -3, 3, 7, 12, 18, 25, 33, 42]

        for subj in range(n_subj):

            #create new gabor filters and model for each new participant
            generate_gabors()
            create_model(seed=subj)

            #use StpOCLsimulator to make use of the Nengo OCL implementation of STSP
            sim = StpOCLsimulator(network=model, seed=subj, context=context,progress_bar=False)

            #trials come in sets of 14, which we call a run (all possible orientation differences between memory and probe),
            runs = int(trials_per_subj / 14)   

            for run in range(runs):
     
                #run a trial with each possible orientation difference
                for cnt_in_run, anglediff in enumerate(probelist):
  
                    print('Subject ' + str(subj+1) + '/' + str(n_subj) + '; Trial ' + str(run*14 + cnt_in_run + 1) + '/' + str(trials_per_subj))

                    #set probe and stim
                    memory_item_cued=randint(0, 179) #random memory
                    probe_cued=memory_item_cued+anglediff #probe based on that
                    probe_cued=norm_p(probe_cued) #normalise probe

                    #random phase
                    or_memory_item_cued=memory_item_cued #original
                    memory_item_cued=memory_item_cued+(180*randint(0, 9))
                    probe_cued=probe_cued+(180*randint(0, 9))
            
                    #store orientation
                    initialangle_c[angle_index]=or_memory_item_cued
              
                    #run simulation
                    sim.run(3)
                
                    #store output
                    np.savetxt(cur_path+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i.csv" % (anglediff, subj+1, run*14+cnt_in_run+1), sim.data[model.p_dec_cued][2500:2999,:], delimiter=",")
            
                    #reset simulator, clean probes thoroughly
                    sim.reset()
                    for probe2 in sim.model.probes:
                        del sim._probe_outputs[probe2][:]
                    del sim.data
                    sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
            
                    angle_index=angle_index+1
            
        np.savetxt(cur_path+sim_no+"_initial_angles_cued.csv", initialangle_c,delimiter=",")
