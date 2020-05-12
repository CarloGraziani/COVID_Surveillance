#!/usr/bin/python3

# Likelihood function support for epidemic model parameter estimation.
#
# Carlo Graziani, ANL

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

#######################################################################
#######################################################################
#######################################################################

class loglik(object):
    """
    Log likelihood

    Constructor Args:

      test_data (`Tensor`[:,3]): 
        test_data[:,0] -- day number
        test_data[:,1] -- number of tests administered
        test_data[:,2] -- number of positive test results
    
      vdyn_ode_fn: (callable): Returns RHS Tensor of an ODE describing viral
        kinetics. This should be a subclass of ODE_Dynamics. The 0-th entry is 
        the time-derivative of viral load.

      positive_fn (callable): Function of current viral load representing the
        probability of a positive test result. Arguments are (load, parameters).

      symptom_fn (callable): Function of current viral load representing the
        probability of exhibiting symptoms. Arguments are (load, parameters).

      prob_s_ibar (float): Probability of symptoms and no infection

      prob_fp (float): False postive probability

      Epi_Model (callable): Epidemic model (ODE RHS, subclass of ODE_Dynamics) 
        to be called. The first component is "Infected fraction" i.e. an "SIR"
        model would be an "ISR" model.  Default is SIR().

      Duration (float): Disease duration, used as look-back time
        for viral dynamics integration.

      Epi_cadence (float): Time intervals for epidemic model computation.

      Vir_cadence (float): Time intervals for viral dynamics computation.

    The __call__() returns the log-likelihood.
    """

#######################################################################
    def __init__(self, test_data, vdyn_ode_fn, positive_fn, 
                 symptom_fn, prob_s_ibar, prob_fp=0.0, Epi_Model=None,
                 duration=24.0, Epi_cadence=0.5, Vir_cadence=0.0625):
        """
        Constructor
        """

        self.test_data = test_data
        self.vdyn_ode_fn = vdyn_ode_fn
        self.positive_fn = positive_fn
        self.symptom_fn = symptom_fn
        self.prob_s_ibar = prob_s_ibar
        self.prob_fp = prob_fp
        if Epi_Model is None:
            self.Epi_Model = SIR
        else:
            self.Epi_Model = Epi_Model

        self.duration = duration

        if Epi_cadence < Vir_cadence:
            raise ValueError("Epi_cadence should be longer than Vir_cadence")
        self.Epi_cadence = Epi_cadence
        self.Vir_cadence = Vir_cadence
        

#######################################################################
    def __call__(self, epipar, vpar, pospar, sympar):
        """
        Log likelihood computation.

        Args:

          epipar (`Tensor`[...,np_epi]): parameters to local epidemic model.
            Left-most indices denote chains.

          vpar (`Tensor`[...,np_vdyn]): parameters to the viral dynamics model.

          pospar (`Tensor`[...,np_pos]): parameters to positive-test function.
            Left-most indices denote chains.

          sympar (`Tensor`[...,np_symp]): parameters to symptom function.
            Left-most indices denote chains.

        Returns: Log-likelihood (`Tensor`[...]): Log-likelihood indexed over chains.
        """

        pp = self._pos_prob(epipar, vpar, pospars, sympar)
        # The leading indices of pp are chains, the rightmost index is the data index.

        ll = tf.log(pp) * self.data[:,2] + tf.log(1-pp) * (self.data[:,1] - self.data[:,2])
        ll = tf.reduce_sum(ll, axis=-1)

        return ll

#######################################################################
    def _pos_prob(self, epipar, vpar, pospar, sympar):
        """
        Probability of a positive test conditioned on exhibiting symptoms

        Args:

          locpar (`Tensor`[...,np_epi]): parameters to local epidemic model.
            Left-most indices denote chains.

          vpar (`Tensor`[...,np_vdyn]): parameters to the viral dynamics model.

          pospar (`Tensor`[...,np_pos]): parameters to positive-test function.
            Left-most indices denote chains.

          sympar (`Tensor`[...,np_symp]): parameters to symptom function.
            Left-most indices denote chains.

        Returns: Probability (`Tensor`[...,:]).  The rightmost is the data
          index, other indices are chains.

        """

        self._epidemic(epipar)
        self._vdyn(vpar)
        ig0, ig1 = self._prob_integrals(pospar, sympar)

        p_given_si = ig1 / ig0
        i_given_s = ig0 /(ig0 + self.prob_s_ibar)
        p_given_ibar_s = self.prob_fp
        ibar_given_s = 1.0 - i_given_s

        p_given_s = p_given_si * i_given_s + p_given_ibar_s * ibar_given_s

        return p_given_s


#######################################################################
    def _epidemic(self, epipar):
        """
        Integrate an epidemic model. The ODE model has dimension D_Epi. 

        Args:

          epipar (`Tensor`[...,:]): Model parameters. The last D_Epi 
            parameters in the rightmost index of epipar are the initial state.
            The other indices correspond to chains.

        Returns: Nothing
        Creates:

          self.estates (`Tensor`[...,:]): The epidemic states. Final index
            denotes times, other indices correspond to chains.
          
          self.etimes (`Tensor`[:]): Times at which states were computed.

        Times of integration are every (self.Epi_cadence * 1 day) starting at 
        (-self.duration - 2*self.Epi_cadence) days relative to start of data, and 
        ending at +2*self.Epi_cadence days relative to end of data.
        """

        self.em = self.Epi_Model(epipar)  # Need this in _prob_integrals()
        D_Epi = self.em.Ndim
        initial_state = epipar[...,-D_epi:]
        self.initial_time = self.test_data[0,0] - self.duration - 2.0*self.Epi_cadence
        st1 = initial_time
        st2 = self.test_data[-1,0] + 2.0*self.Epi_cadence
        self.etimes = tf.constant(np.arange(st1, st2, step=self.Epi_cadence, 
                                            dtype=np.float32))

        DP = tfp.math.ode.DormandPrince()
        results = DP.solve(self.em.RHS, initial_time, initial_state, 
                           solution_times=self.etimes)

        self.estates = results.states
        # But this has shape self.etimes.shape[0] + epipar.shape[:-1] + [D_Epi].
        # We want shape epipar.shape[:-1] + [D_Epi] + self.etimes.shape[0].
        ls = len(self.estates.shape)
        p = (np.arange(ls) + 1) % ls
        self.estates = tf.transpose(self.estates, perm=p)

#######################################################################
    def _vdyn(vpar):
        """
        Integrate the virus dynamics. The ODE model has dimension D_Vir. 

        Args:

          vpar (`Tensor`[...,:]): Model parameters. The last D_Vir 
            parameters in the rightmost index of vpar are the initial state.
            The other indices correspond to chains.

        Returns: Nothing
        Creates:

          self.vload (`Tensor`[...,:]): The viral loads. Final index
            denotes times, other indices correspond to chains.
          
          self.vtimes (`Tensor`[:]): Times at which states were computed.

        Times of integration are every (self.Vir_cadence * 1 day) starting at 
        zero, and ending at self.duration days.
        """

        vm = vdyn_ode_fn(vpar)
        D_Vir = vm.Ndim
        initial_state = vpar[...,-D_Vir:]
        st1 = 0.0
        st2 = self.duration
        self.vtimes = tf.constant(np.arange(st1, st2, step=self.Vir_cadence, 
                                            dtype=np.float32))

        DP = tfp.math.ode.DormandPrince()
        results = DP.solve(vm.RHS, initial_time, self.vdyn_init, 
                           solution_times=self.vtimes)

        self.vload = results.states[...,0]
        # But this has shape self.vtimes.shape[0] + vpar.shape[:-1].
        # We want vpar.shape[:-1] + self.vtimes.shape[0]
        r = len(self.vload.shape)
        p = (np.arange(r) + 1) % r
        self.vload = tf.transpose(self.vload, perm=p)

#######################################################################
    def _prob_integrals(self, pospar, sympar):
        """
        Compute the time quadratures required for likelihood

        Args:

          pospar(`Tensor`[...,np_pos]): parameters to positive-test function.
            Left-most indices denote chains.

          sympar (`Tensor`[...,np_symp]): parameters to symptom function.
            Left-most indices denote chains.

        Returns:
            (ig0, ig1)
            The required quadratures. The last index is the data index.

            ig0[...,:]: Prob(i,s)
            ig1[...,:]: Prob(i,s,p)

        """

        # array of t - \tau
        tmtau = tf.expand_dims(self.test_data[:,0], 1) \
                -  tf.expand_dims(self.vtimes, 0)

        iv = cubic_interpolation(tmtau, self.etimes[0], self.Epi_cadence,
                                 self.estates)

        ir = self.em.infection_rate(iv, axis=-3) # Infection rate 
         # Shape: chain_shape + self.test_data.shape[0] + self.vtimes.shape

        integrand_1 = ir * self.symptom_fn(self.vload, sympar)
        integrand_2 = ir * self.symptom_fn(self.vload, sympar) \
                         * self.positive_fn(self.vload, pospar)

        ig_0 = tf.reduce_sum(integrand_1, axis=-1) * self.Vir_cadence
        ig_1 = tf.reduce_sum(integrand_2, axis=-1) * self.Vir_cadence

        return ig_0, ig_1

#######################################################################
#######################################################################
#######################################################################

rtarr = np.array([[-0.5,0.5,1.5],
                  [-1.5,0.5,1.5],
                  [-1.5,-0.5,1.5],
                  [-1.5,-0.5,0.5]], dtype=np.float32)
rtarr = tf.constant(rtarr)

prodarr = np.array([-6.0, 2.0, -2.0, 6.0], dtype=np.float32)
prodarr = tf.constant(1/prodarr)

def cubic_interpolation(t, t0, dt, fvals):
    """
    Produce a cubic interpolation of the vector function whose samples spaced
    by dt starting at t0 are in the array fvals, using the Lagrange
    interpolation formula.

    Args:

      t (`Tensor`[:]): times of desired interpolation
      t0 (float): time corresponding to fvals[...,0]
      dt (float): cadence of equally-spaced times
      fvals (`Tensor`[...,:,:]): Function samples.  Left-most indices correspond
        to chains. Second-to-last index is over vector (i.e. state) component
        Right-most index is over samples.

    Returns:
      interpolant (`Tensor`[fvals.shape[:-1] + t.shape ]): Interpolant values.
    """
                                                        # Shapes:
    ts = t.shape ; fs = fvals.shape
    nt = (t-t0)/dt                                      # ts
 # to guarantee we have data for cubic
    assert(not tf.reduce_any(nt < 1) and 
           not tf.reduce_any(nt > fs[-1]-2))

    i0 = tf.expand_dims(tf.cast(nt, tf.int64) - 1, -1)  # ts + [1]
    indices = i0 + tf.constant(np.arange(4))            # ts + [4]
    ftrain = tf.gather(fvals, indices, axis=-1)         # fs[:-1] + ts + [4]

    tt = tf.reshape(nt%1 - 0.5, ts + [1,1])             # ts + [1,1]
    res = tf.reduce_prod((tt-rtarr), axis=-1)           # ts + [4]
    res = res * prodarr                                 # ts + [4]
    res = res * ftrain                                  # fs[:-1] + ts + [4]
    res = tf.reduce_sum(res, axis=-1)                   # fs[:-1] + ts

    return res
