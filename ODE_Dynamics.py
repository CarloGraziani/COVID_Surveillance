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