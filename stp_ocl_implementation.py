#imports
from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.neurons import LIF, LIFRate
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import *
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam)
from nengo.utils.compat import is_iterable, is_string, itervalues, range
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo_ocl import Simulator
from nengo_ocl.utils import as_ascii #, indent, round_up
from mako.template import Template
import pyopencl as cl
from nengo_ocl.plan import Plan
from nengo_ocl.clra_nonlinearities import _plan_template
from collections import OrderedDict
import nengo.dists as nengod
from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo.dists import Uniform


#create new neuron type stpLIF with resources (x) and calcium (u)

class stpLIF(LIF):
    probeable = ('spikes', 'resources', 'voltage', 'refractory_time', 'calcium')
    
    tau_x = NumberParam('tau_x', low=0, low_open=True)
    tau_u = NumberParam('tau_u', low=0, low_open=True)
    U = NumberParam('U', low=0, low_open=True)

    def __init__(self, tau_x=0.2, tau_u=1.5, U=0.2, **lif_args):
        super(stpLIF, self).__init__(**lif_args)
        self.tau_x = tau_x
        self.tau_u = tau_u
        self.U = U

    @property
    def _argreprs(self):
        args = super(LIFRate, self)._argreprs
        if self.tau_x != 0.2:
            args.append("tau_x=%s" % self.tau_x)
        if self.tau_u != 1.5:
            args.append("tau_u=%s" % self.tau_u)
        if self.U!= 0.2:
            args.append("U=%s" % self.U)
        return args

    def step_math(self, dt, J, output, voltage, ref, resources, calcium):
        """Implement the u and x parameters """
        x = resources
        u = calcium
        LIF.step_math(self, dt, J, output, voltage, ref)
        
        #calculate u and x
        dx=dt * ( (1-x)/self.tau_x - u*x*output )
        du=dt * ( (self.U-u)/self.tau_u + self.U*(1-u)*output )
        
        x += dx
        u += du
        


#add builder for stpLIF

@Builder.register(stpLIF)
def build_stpLIF(model, stplif, neurons):
    
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['resources'] = Signal(
        np.ones(neurons.size_in), name="%s.resources" % neurons)
    model.sig[neurons]['calcium'] = Signal(
        np.full(neurons.size_in, stplif.U), name="%s.calcium" % neurons)
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
    Modifies connection weights according to the calcium and resources of the neuron presynaptic
    """
    modifies = 'weights'
    probeable = ('delta', 'calcium', 'resources')

    def __init__(self):
        super(STP, self).__init__(size_in=0)




#builders for STP
class SimSTP(Operator):
    r"""Calculate connection weight change according to the STP rule.
    Implements the STP learning rule of the form:
    .. math:: omega_{ij} = ((u_i * x_i) / U_i) * omega_{ij-initial}
    where
    * :math:`\omega_{ij}` is the connection weight between the two neurons.
    * :math:`u_i` is the calcium level of the presynaptic neuron.
    * :math:`x_i` is the resources level of the presynaptic neuron.
    * :math:`U_i` is the baseline calcium level of the presynaptic neuron.
    * :math:`\omega_{ij-initial}` is the initial connection weight between the two neurons.
    Parameters
    ----------
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta ((u_i * x_i) / U_i) * initial_omega_{ij} - omega_{ij}`.
    calcium : Signal
        The calcium level of the presynaptic neuron, :math:`u_i`.
    resources : Signal
        The resources level of the presynaptic neuron, :math:`x_i`.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    calcium : Signal
        The calcium level of the presynaptic neuron, :math:`u_i`.
    resources : Signal
        The resources level of the presynaptic neuron, :math:`x_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[weights, calcium, resources]``
    4. updates ``[delta]``
    """

    def __init__(self, calcium, resources, weights, delta,
                 tag=None):
        super(SimSTP, self).__init__(tag=tag)
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
        init_weights = self.weights.initial_value
        calcium = signals[self.calcium]
        resources = signals[self.resources]
        U=self.calcium.initial_value
        def step_simstp():
            # perform update
                delta[...] = ((calcium * resources)/U) * init_weights - weights
            
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
    more than once with the same `.STP` instance.
    """

    conn = rule.connection
    calcium = model.sig[get_pre_ens(conn).neurons]['calcium']
    resources = model.sig[get_pre_ens(conn).neurons]['resources']

    model.add_op(SimSTP(calcium,
                        resources,
                        model.sig[conn]['weights'],
                        model.sig[rule]['delta'],
                        ))

    # expose these for probes
    model.sig[rule]['calcium'] = calcium
    model.sig[rule]['resources'] = resources


#----- Nengo OCL implementation of STP and StpLIF ------
#-------------------------------------------------------


def plan_stp(queue, calcium, resources, weights, delta, init_weights, init_calcium, tag=None):
    assert (len(calcium) == len(resources) == len(weights) == len(delta) ==
           len(init_weights) == init_calcium.size)
    N = len(calcium)

    for arr in (calcium, resources):  # vectors
        assert (arr.shape1s == 1).all()
    for arr in (delta, weights, init_weights):  # matrices
        assert (arr.stride1s == 1).all()

    #assert (resources.shape0s == weights.shape0s).all()
    #assert (calcium.shape0s == weights.shape1s).all()
    assert (weights.shape0s == delta.shape0s).all()
    assert (weights.shape1s == delta.shape1s).all()
    assert (weights.shape0s == init_weights.shape0s).all()
    assert (weights.shape1s == init_weights.shape1s).all()

    assert (calcium.ctype == resources.ctype == weights.ctype == delta.ctype ==
            init_weights.ctype == init_calcium.ctype)

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
        __global const int *init_weights_stride0s,
        __global const int *init_weights_starts,
        __global const ${type} *init_weights_data,
        __global const ${type} *init_calciums
        
        
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
        const ${type} init_weights = init_weights_data[init_weights_starts[k] + i*init_weights_stride0s[k]+j];
        const ${type} init_calcium = init_calciums[k];
    
        if (i < shape0) {
            delta[i*delta_stride0s[k] + j] =
               ((calcium*resources/init_calcium)*init_weights)-weight;
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
        init_weights.cl_stride0s, init_weights.cl_starts, init_weights.cl_buf, 
        init_calcium,
    )
    _fn = cl.Program(queue.context, text).build().stp
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (delta.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_stp", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 6 * delta.sizes.sum()
    plan.bw_per_call = (calcium.nbytes + resources.nbytes + weights.nbytes +
                        delta.nbytes + init_weights.nbytes + init_calcium.nbytes)
    return plan



def plan_stplif(queue, dt, J, V, W, outS, ref, tau, amp, u, x, tau_u, tau_x, U, upsample=1, **kwargs):
    assert J.ctype == 'float'
    for x in [V, W, outS, u, x]:
        assert x.ctype == J.ctype

    inputs = dict(J=J, V=V, W=W, x=x, u=u)
    outputs = dict(outV=V, outW=W, outS=outS, outx=x, outu=u )
    parameters = dict(tau=tau, ref=ref, amp=amp, tau_x=tau_x, tau_u=tau_u, U=U)

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
        outx = x+ dt* ((1-x)/tau_x - u*x*outS);
        outu = u+ dt* ((U-u)/tau_u + U*(1-u)*outS) ;
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
        x = self.all_data[[self.sidx[op.states[2]] for op in ops]]
        u = self.all_data[[self.sidx[op.states[3]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau_x = self.RaggedArray([op.neurons.tau_x * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        tau_u = self.RaggedArray([op.neurons.tau_u * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        U = self.RaggedArray([op.neurons.U * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        
        return [plan_stplif(self.queue, dt, J, V, W, S, ref, tau, amp, u, x, tau_u, tau_x, U)]
    

    def plan_SimSTP(self, ops):
        calcium = self.all_data[[self.sidx[op.calcium] for op in ops]]
        resources = self.all_data[[self.sidx[op.resources] for op in ops]]
        weights = self.all_data[[self.sidx[op.weights] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        init_weights = self.RaggedArray([op.weights.initial_value for op in ops], dtype=calcium.dtype)
        init_calcium = self.Array([op.calcium.initial_value[0] for op in ops])
        
        return [plan_stp(self.queue, calcium, resources, weights, delta, init_weights, init_calcium)]