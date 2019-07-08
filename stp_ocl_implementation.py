#imports
from __future__ import division

import logging
import warnings

import numpy as np
import nengo

from nengo.exceptions import SimulationError, ValidationError, BuildError
import nengo.learning_rules
from nengo.builder.operator import Reset #,DotInc, ElementwiseInc, Copy
from nengo.builder import Operator, Builder, Signal
from nengo.utils.neurons import settled_firingrate
from nengo.neurons import AdaptiveLIFRate, LIF
from nengo.config import SupportDefaultsMixin
from nengo.params import (Default, IntParam, FrozenObject, NumberParam,
                          Parameter, Unconfigurable)
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.compat import is_iterable, is_string, itervalues, range
from nengo.learning_rules import *
from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.node import Node
from nengo.builder.learning_rules import *
from nengo.dists import Uniform
from nengo.processes import Piecewise
import nengo_spa as spa

from nengo_ocl import Simulator
from nengo_ocl.utils import as_ascii #, indent, round_up
from mako.template import Template
import pyopencl as cl
from nengo_ocl.plan import Plan
from nengo_ocl.clra_nonlinearities import _plan_template

from collections import OrderedDict

import pyopencl as cl
from mako.template import Template
import nengo.dists as nengod
from nengo.utils.compat import is_number, itervalues, range

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.plan import Plan
from nengo_ocl.utils import as_ascii, indent, round_up


#create new neuron type stpLIF with resources (x) and calcium (u)

class stpLIF(LIF):
    probeable = ('spikes', 'resources', 'voltage', 'refractory_time', 'calcium')
    
    tau_x = NumberParam('tau_x', low=0, low_open=True)
    tau_u = NumberParam('tau_u', low=0, low_open=True)

    def __init__(self, tau_x=0.2, tau_u=1.5, **lif_args):
        super(stpLIF, self).__init__(**lif_args)
        self.tau_x = tau_x
        self.tau_u = tau_u

    @property
    def _argreprs(self):
        args = super(LIFRate, self)._argreprs
        if self.tau_x != 0.2:
            args.append("tau_n=%s" % self.tau_n)
        if self.tau_u != 1.5:
            args.append("tau_n=%s" % self.tau_n)
        return args

    def step_math(self, dt, J, output, voltage, ref, resources, calcium):
        """Implement the u and x parameters """
        x = resources
        u = calcium
        LIF.step_math(self, dt, J, output, voltage, ref)
        
        #calculate u and x
        dx=dt * ( (1-x)/self.tau_x - u*x*output )
        du=dt * ( (0.2-u)/self.tau_u + 0.2*(1-u)*output )
        
        x += dx
        u += du
        


#add builder for stpLIF

from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
@Builder.register(stpLIF)
def build_stpLIF(model, stplif, neurons):
    
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['resources'] = Signal(
        np.ones(neurons.size_in), name="%s.resources" % neurons)
    model.sig[neurons]['calcium'] = Signal(
        np.full(neurons.size_in, 0.2), name="%s.calcium" % neurons)
    model.add_op(SimNeurons(neurons=stplif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['refractory_time'],
                                    model.sig[neurons]['resources'],
                                    model.sig[neurons]['calcium']]))



#create new learning rule to model short term plasticity (only works if pre-ensemble has neuron type StpLIF)

class STP(LearningRuleType):
    """STP learning rule.

    Modifies connection weights according to the calcium and resources of the presynapse


    Parameters
    ----------
    learning_rate : float, optional (Default: 1)
        A scalar indicating the rate at which weights will be adjusted (exponential).

    Attributes
    ----------
    learning_rate : float
    """

    modifies = 'weights'
    probeable = ('delta', 'calcium', 'resources')

    learning_rate = NumberParam(
        'learning_rate', low=0, readonly=True, default=1)

    def __init__(self, learning_rate=Default):
        super(STP, self).__init__(learning_rate, size_in=0)

    @property
    def _argdefaults(self):
        return (('learning_rate', STP.learning_rate.default))



#builders for STP
class SimSTP(Operator):
    r"""Calculate connection weight change according to the STP rule.

    Implements the STP learning rule of the form:

    .. math:: omega_{ij} = .....

    where

    * :math:`\omega_{ij}` is the connection weight between the two neurons.

    Parameters
    ----------
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, weights]``
    4. updates ``[delta]``
    """

    def __init__(self, calcium, resources, weights, delta,
                 learning_rate, tag=None):
        super(SimSTP, self).__init__(tag=tag)
        self.learning_rate = learning_rate
       # self.init_weights=init_weights
        self.sets = []
        self.incs = []
        self.reads = [weights, calcium, resources]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def weights(self):
        return self.reads[0]
        
    @property
    def calcium(self):
        return self.reads[1]
    
    @property
    def resources(self):
        return self.reads[2]
    
    
  
   
    def _descstr(self):
        return '%s' % (self.delta)       

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        delta = signals[self.delta]
        alpha = self.learning_rate #* dt
        init_weights = self.weights.initial_value
        calcium = signals[self.calcium]
        resources = signals[self.resources]
        def step_simstp():
            # perform update
                delta[...] = ((calcium * resources)/0.2) * init_weights - weights
            
        return step_simstp
    
@Builder.register(STP)
def build_stp(model, stp, rule):
    """Builds a `.STP` object into a model.

   

    Parameters
    ----------
    model : Model
        The model to build into.
    stp : STP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Stp` instance.
    """

    conn = rule.connection
    calcium = model.sig[get_pre_ens(conn).neurons]['calcium']
    resources = model.sig[get_pre_ens(conn).neurons]['resources']
  

    model.add_op(SimSTP(calcium,
                        resources,
                        model.sig[conn]['weights'],
                        model.sig[rule]['delta'],
                        learning_rate=stp.learning_rate))

    # expose these for probes
    model.sig[rule]['calcium'] = calcium
    model.sig[rule]['resources'] = resources


#----- Nengo OCL implementation of STP and StpLIF ------
#-------------------------------------------------------


def plan_stp(queue, calcium, resources, weights, delta, alpha, init_weights, tag=None):
    assert (len(calcium) == len(resources) == len(weights) == len(delta) ==
            alpha.size == len(init_weights))
    N = len(calcium)

    for arr in (calcium, resources):  # vectors
        assert (arr.shape1s == 1).all()
    for arr in (delta, weights,init_weights):  # matrices
        assert (arr.stride1s == 1).all()

    #assert (resources.shape0s == weights.shape0s).all()
    #assert (calcium.shape0s == weights.shape1s).all()
    assert (weights.shape0s == delta.shape0s).all()
    assert (weights.shape1s == delta.shape1s).all()
    assert (weights.shape0s == init_weights.shape0s).all()
    assert (weights.shape1s == init_weights.shape1s).all()

    assert (calcium.ctype == resources.ctype == weights.ctype == delta.ctype ==
            alpha.ctype == init_weights.ctype)

    text = """
    __kernel void stp(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *calcium_stride0s,
        __global const int *calcium_starts,
        __global const ${type} *calcium_data,
        __global const int *resources_stride0s,
        __global const int *resources_starts,
        __global const ${type} *resources_data,
        __global const int *weights_stride0s,
        __global const int *weights_starts,
        __global const ${type} *weights_data,
        __global const int *delta_stride0s,
        __global const int *delta_starts,
        __global ${type} *delta_data,
        __global const ${type} *alphas,
        __global const int *init_weights_stride0s,
        __global const int *init_weights_starts,
        __global const ${type} *init_weights_data
        
    )
    {
        const int ij = get_global_id(0);
        const int k = get_global_id(1);
        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        const int i = ij / shape1;
        const int j = ij % shape1;
        __global ${type} *delta = delta_data + delta_starts[k];
        const ${type} calcium = calcium_data[calcium_starts[k] + i*calcium_stride0s[k]];
        const ${type} resources = resources_data[resources_starts[k] + i*resources_stride0s[k]];
        const ${type} weight = weights_data[
            weights_starts[k] + i*weights_stride0s[k]+j];
        const ${type} alpha = alphas[k];
        const ${type} init_weights = init_weights_data[init_weights_starts[k] + i*init_weights_stride0s[k]+j];
        if (i < shape0) {
            delta[i*delta_stride0s[k] + j] =
               ((calcium*resources/0.2)*init_weights)-weight;
        }
    }
    """

    textconf = dict(type=calcium.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        delta.cl_shape0s, delta.cl_shape1s,
        calcium.cl_stride0s, calcium.cl_starts, calcium.cl_buf,
        resources.cl_stride0s, resources.cl_starts, resources.cl_buf,
        weights.cl_stride0s, weights.cl_starts, weights.cl_buf,
        delta.cl_stride0s, delta.cl_starts, delta.cl_buf,
        alpha, init_weights.cl_stride0s, init_weights.cl_starts, init_weights.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().stp
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (delta.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_stp", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 6 * delta.sizes.sum()
    plan.bw_per_call = (calcium.nbytes + resources.nbytes + weights.nbytes +
                        delta.nbytes + alpha.nbytes + init_weights.nbytes)
    return plan



def plan_stplif(queue, dt, J, V, W, outS, ref, tau, amp, U, X, tau_u, tau_x, upsample=1, **kwargs):
    assert J.ctype == 'float'
    for x in [V, W, outS, U, X]:
        assert x.ctype == J.ctype

    inputs = dict(J=J, V=V, W=W, X=X, U=U)
    outputs = dict(outV=V, outW=W, outS=outS, outX=X, outU=U)
    parameters = dict(tau=tau, ref=ref, amp=amp,tau_x=tau_x, tau_u=tau_u)

    dt = float(dt)
    textconf = dict(
        type=J.ctype, dt=dt, upsample=upsample,
        dtu=dt/upsample, dtu_inv=upsample/dt, dt_inv=1/dt)
    decs = """
        char spiked;
        ${type} dV;
        const ${type} V_threshold = 1;
        const ${type} dtu = ${dtu}, dtu_inv = ${dtu_inv}, dt_inv = ${dt_inv};
        ${type} delta_t;
        const ${type} dt = ${dt};
        """
    # TODO: could precompute -expm1(-dtu / tau)
    text = """
        spiked = 0;
% for ii in range(upsample):
        W -= dtu;
        delta_t = (W > dtu) ? 0 : (W < 0) ? dtu : dtu - W;
        dV = -expm1(-delta_t / tau) * (J - V);
        V += dV;
        if (V > V_threshold) {
            const ${type} t_spike = dtu + tau * log1p(
                -(V - V_threshold) / (J - V_threshold));
            W = ref + t_spike;
            V = 0;
            spiked = 1;
        }else if (V < 0) {
            V = 0;
        }        

% endfor
        outV = V;
        outW = W;
        outS = (spiked) ? amp*dt_inv : 0;
        outX = X+ dt* ((1-X)/tau_x - U*X*outS);
        outU = U+ dt* ((0.2-U)/tau_u + 0.2*(1-U)*outS) ;
        """
    decs = as_ascii(Template(decs, output_encoding='ascii').render(**textconf))
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    cl_name = "cl_stplif"
    return _plan_template(
        queue, cl_name, text, declares=decs,
        inputs=inputs, outputs=outputs, parameters=parameters, **kwargs)


class StpOCLsimulator(Simulator):

    def _plan_stpLIF(self, ops):
        if not all(op.neurons.min_voltage == 0 for op in ops):
            raise NotImplementedError("LIF min voltage")
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        W = self.all_data[[self.sidx[op.states[1]] for op in ops]]
        X = self.all_data[[self.sidx[op.states[2]] for op in ops]]
        U = self.all_data[[self.sidx[op.states[3]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau_x = self.RaggedArray([op.neurons.tau_x * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        tau_u = self.RaggedArray([op.neurons.tau_u * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        
        return [plan_stplif(self.queue, dt, J, V, W, S, ref, tau, amp, U, X, tau_u, tau_x)]
    

    def plan_SimSTP(self, ops):
        calcium = self.all_data[[self.sidx[op.calcium] for op in ops]]
        resources = self.all_data[[self.sidx[op.resources] for op in ops]]
        weights = self.all_data[[self.sidx[op.weights] for op in ops]]
        #init_weights=self.all_data[[self.sidx[op.init_weights] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        test= self.all_data[[self.sidx[op.weights] for op in ops]]
        init_weights = self.RaggedArray([op.weights.initial_value for op in ops], dtype=calcium.dtype)
        
        return [plan_stp(self.queue, calcium, resources, weights, delta, alpha, init_weights)]

