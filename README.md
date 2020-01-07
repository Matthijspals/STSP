# STSP
A Nengo Implementation of Short-term Synaptic Plasticity (STSP) as proposed by Mongillo, Barak and Tsodyks (2008)
## How to use it
In order to import the necessary classes/functions use: 

```from stp_ocl_implementation import *```

In order to use Spiking leaky integrate-and-fire implementing STSP, specify the neuron type of an ensemble as follows:

```neuron_type=stpLIF()```

And specify the following learning rule for outgoing connections:

```learning_rule_type=STP()```

To use the OCL implementation use the following simulator:

```StpOCLsimulator()```

The following additional probes can be used:

```nengo.Probe(ensemble.neurons, 'calcium')```
```nengo.Probe(ensemble.neurons, 'resources')  ```   

## Example models/simulations
The implementation of STSP was used to create a functional spiking neuron model of working memory: https://www.biorxiv.org/content/10.1101/823559v1. Using this mechanism, the model is able to maintain information in activity-silent states. This model was then used to simulate three working memory tasks (the Model_sim_exp.py files), earlier performed by human participants (Wolff et al. 2017). Both the model's behavior as well as its neural representations are in agreement with the human data. 

## Theoretical background
Synaptic efficiency is based on two parameters: the amount of available resources to the presynaptic neuron (x, normalised to be between 0 and 1) and the fraction of resources used each time a neuron fires (u), reflecting the residual presynaptic calcium level.

For all LIF neurons to which we want to apply STSP, every simulation time step u and x are calculated according to equation 2.1 and 2.2, respectively. When a neuron fires, its resources x are decreased by u x, mimicking neurotransmitter depletion. At the same time, its calcium level u is increased, mimicking calcium influx into the presynaptic terminal. Both u and x relax back to baseline with time constants 𝜏_𝐷 (0.2s) and 𝜏_𝐹 (1.5s), respectively. The mechanisms are described by:

𝑑𝑥/𝑑𝑡= (1−𝑥)/𝜏_𝐷 − 𝑢 𝑥 𝛿(𝑡−𝑡_𝑠𝑝) (2.1)

𝑑𝑢/𝑑𝑡= (𝑈−𝑢)/𝜏_𝐹 − 𝑈 (1−𝑢) 𝛿(𝑡−𝑡_𝑠𝑝) (2.2) 

Where x represents the available resources, u represents the residual calcium level and U its baseline level, 𝜏_𝐹 is the facilitating time constant and 𝜏_𝐷 the depressing time constant, 𝛿 represents the Dirac delta function, t the simulation time and t_sp the time of a presynaptic spike. 

Outgoing connection weights of neurons implementing STSP are determined by both their initial connection weight and their current synaptic efficiency. Initial connections weights are calculated by the NEF, while synaptic efficiency is set to the product of the current value of u and x of the presynaptic neuron, normalised by their baseline value (equation 2.3). This results in a system where after a neuron fires its outgoing connections will be depressed on the time scale of 𝜏_𝐷 and facilitated on the timescale of 𝜏_𝐹.

𝑑𝑤_𝑖𝑗/𝑑𝑡= (c 𝑢)/𝐶 𝑤_0𝑖𝑗 (2.3)

Where 𝑤_𝑖𝑗 represents the connection weight between neuron i and j and 𝑤_0𝑖𝑗 the initial connection weight between neuron i and j.
