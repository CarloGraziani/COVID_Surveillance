#!/usr/bin/python3

# Lightweight base class for ODE dynamical systems.
#
# Carlo Graziani, ANL

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

#######################################################################
#######################################################################
#######################################################################
class ODE_model(object):
    """
    Support for ODE dynamical models.

    Constructor Args:

      params (`Tensor`[...,:]): Model parameters. Leftmost indices denote
        chains.

    Note that if a subclass is intended to represent an epidemiological
    model, it should overwrite the 'infection_rate' method, which returns
    the infection rate -- not counting recoveries and fatalities -- for
    the current state.
    """

    Ndim = 0  # Each derived class needs to set this as appropriate

#######################################################################
    def __init__(self, params):
        """
        Constructor
        """

        self.params = params

#######################################################################
    def RHS(self, time, state):
        """
        Overwrite me, derived classes. I should return the RHS of the ODE
        at the current state.
        """
        return 0


#######################################################################
    def infection_rate(self, state, axis):
        """
        Should return the current infection rate, not counting recoveries
        and fatalities.  Calling the base-class version of this method is
        a NotImplementedError. 'axis' should be the axis of 'state' along
        which the state components are defined, the other axes corresponding
        to times and chains.
        """

        raise NotImplementedError("Method 'infection_rate' undefined.")

#######################################################################
#######################################################################
#######################################################################
class SIR(ODE_model):
    """
    SIR epidemiological model

    Constructor Args:

      params (`Tensor`[...,:]) Model parameters.  Leftmost indices denote
        chains.
          params[...,0]: R0, "Reproduction  number"
          params[...,1]: mu, "Steady-state birth rate"
          params[...,2]: nu, "recovery rate"

    Ndim is 2, since recovered fraction is 1-s-i.
    """

    Ndim = 2

#######################################################################
    def RHS(self, time, state):
        """
        RHS of SIR ODE.

        Args:

          state (`Tensor`[...,2]): State vector.  Leftmost indices denote
          chains.
            state[...,0]: Infected fraction
            state[...,1]: Susceptible fraction
        """

        R0 = self.params[...,0]
        mu = self.params[...,1]
        nu = self.params[...,2]
        i = state[...,0]
        s = state[...,1]
        ret0 = (nu+mu) * (R0*s - 1) * i
        ret1 = mu - (nu+mu)*R0*i*s - mu*s
        ret = tf.stack((ret0, ret1), axis=-1)
        return ret

#######################################################################
        def infection_rate(self, state, axis):
            """
            Return infection rate, not counting recoveries and fatalities.

            Args:
              state (`Tensor`[...,2,...]): State vector.  Leftmost indices denote
              chains, rightmost indices are times of evaluation, axis 'axis'
              corresponds to state components.

              axis (int): axis along which the state components are defined.

            Returns:
              infection rate
            """

            R0 = self.params[...,0]
            mu = self.params[...,1]
            nu = self.params[...,2]
            i = tf.gather(state, 0, axis=axis)
            s = tf.gather(state, 1, axis=axis)

            ir = (nu+mu)*R0*i*s
            return ir
    
#######################################################################
#######################################################################
#######################################################################
class ViralDynamics(ODE_model):
    """
    Simple viral dynamics model

    Constructor Args:

      params (`Tensor`[...,:]) Model parameters.  Leftmost indices denote
        chains.
          params[...,0]: beta, "rate at which virus infects host cells"
          params[...,1]: c, "production rate of free virions"
          params[...,2]: alpha, "per capita attrition rate of infected cells"
          params[...,3]: gamma, "per capita attrition rate of virions"
          params[...,4]: mu, "natural death rate of cells"

    Ndim is 3
    
    """

    Ndim = 3
    

#######################################################################
    def RHS(self, time, state):
        """
        RHS of viral dynamics ODE.

        Args:

          state (`Tensor`[...,2]): State vector.  Leftmost indices denote
          chains.
            state[...,0]: Number of uninfected cells
            state[...,1]: Number of infected cells
            state[...,2]: Number of free virus particles
        """

        beta = self.params[...,0]
        c = self.params[...,1]
        alpha = self.params[...,2]
        gamma = self.params[...,3]
        mu = self.params[...,4]
        v = state[...,0]
        x = state[...,1]
        y = state[...,2]
        
        ret0 = c*y -gamma*v -beta*v*x
        ret1 = mu*(1-x) -beta*v*x
        ret2 = beta*v*x -alpha*y

        ret = tf.stack((ret0, ret1, ret2), axis=-1)
        return ret

#######################################################################
        def virus_concentration(self, state, axis):
            """
            Return virus concentration

            Args:
              state (`Tensor`[...,2,...]): State vector.  Leftmost indices denote
              chains, rightmost indices are times of evaluation, axis 'axis'
              corresponds to state components.

              axis (int): axis along which the state components are defined.

            Returns:
              infection rate
            """

           # beta = self.params[...,0]
           # c = self.params[...,1]
           # alpha = self.params[...,2]
           # gamma = self.params[...,3]
           # mu = self.params[...,4]
           # x = tf.gather(state, 0, axis=axis)
           # y = tf.gather(state, 1, axis=axis)
            v = tf.gather(state, 0, axis=axis)

            return v