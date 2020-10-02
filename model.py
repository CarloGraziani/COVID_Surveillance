#!/usr/bin/python3

# Likelihood function support for epidemic model parameter estimation.
#
# Carlo Graziani, Marieme Ngom, Enaksi Saha, ANL

import tensorflow as tf
import tensorflow_probability as tfp
#import tensorflow_graphics as tfg
import numpy as np
import simulation as sim

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
      
      phi_s
      
      psi_s

    The __call__() returns the log-likelihood.
    """

#######################################################################
    def __init__(self, test_data, vdyn_ode_fn, positive_fn, 
                 symptom_fn, prob_s_ibar, prob_fp=0.0, Epi_Model=None,
                 duration=8.0, Epi_cadence=0.5, Vir_cadence=0.0625, phi_s = 0.55, psi_s = 0.1):
        """
        Constructor
        """

        self.test_data = test_data
        self.vdyn_ode_fn = vdyn_ode_fn
        self.positive_fn = positive_fn
        self.symptom_fn = symptom_fn
        self.prob_s_ibar = prob_s_ibar
        self.prob_fp = prob_fp
        self.phi_s = phi_s
        self.psi_s = psi_s
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
    def __call__(self, test_data, epipar, vpar, pospar, sympar):
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

        pp = self._pos_prob(epipar, vpar, pospar, sympar)
        # The leading indices of pp are chains, the rightmost index is the data index.
        
        #self.test_data = test_data ## mng needs to look into why she needs to redefine this and add it to the variables of the function
	
        N_xt = test_data[:,1]  #number of RT-PCR tests performed at epoch t at location x
        C_xt = test_data[:,2] #number of positive confirmed results from tests
        ll =  tf.keras.backend.log(pp) * C_xt + tf.keras.backend.log(1-pp) * (N_xt - C_xt)
        ll = tf.reduce_sum(ll, axis=-1)

        return ll, pp

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
        ig_0, suscep = self._prob_integrals(pospar, sympar)
        den = self.phi_s * (1.0 - suscep) + self.psi_s * (suscep)
#        p_given_si = ig1 / ig0
#        i_given_s = ig0 /(ig0 + self.prob_s_ibar * (1-ig2))
#        p_given_ibar_s = self.prob_fp
#        ibar_given_s = 1.0 - i_given_s


        p_given_s = ig_0 / (den * self.Vir_cadence) #p_given_si * i_given_s + p_given_ibar_s * ibar_given_s
        
    
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
        D_epi = self.em.Ndim
        initial_state = epipar[...,-D_epi:]
        self.initial_time = self.test_data[0,0] - self.duration - 2.*self.Epi_cadence #mng took 2_epi_cadence
        st1 = self.initial_time
        print('initial time')
        print(st1)
        st2 = self.test_data[-1,0] + 2.*self.Epi_cadence #mng took 2_epi_cadence
        print('final time')
        print(st2)
        self.etimes = tf.constant(np.arange(st1, st2, step=self.Epi_cadence,
                                            dtype=np.float32))
            
        DP = tfp.math.ode.DormandPrince()
        results = DP.solve(self.em.RHS, self.initial_time, initial_state,solution_times=self.etimes)
                                            
        self.estates = results.states
        
#                                            # But this has shape self.etimes.shape[0] + epipar.shape[:-1] + [D_Epi].
                                            # We want shape epipar.shape[:-1] + [D_Epi] + self.etimes.shape[0].
        ls = len(self.estates.shape)
        p = (np.arange(ls) + 1) % ls
        self.estates = tf.transpose(self.estates, perm=p)



#######################################################################
    def _vdyn(self,vpar):
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

        vm = self.vdyn_ode_fn(vpar)
        D_Vir = vm.Ndim
        initial_state = vpar[...,-D_Vir:]
        st1 = 0.0
        vdyn_initial_time = st1
        st2 = self.duration
        self.vtimes = tf.constant(np.arange(st1, st2, step=self.Vir_cadence,
                                            dtype=np.float32))
#

        DP = tfp.math.ode.DormandPrince()
        results = DP.solve(vm.RHS, vdyn_initial_time, initial_state,
                           solution_times=self.vtimes)  #mng replaced vdyn_init by initial_states and self.initial_times

        self.vload = results.states[...,0]  
        # But this has shape self.vtimes.shape[0] + vpar.shape[:-1].
        # We want vpar.shape[:-1] + self.vtimes.shape[0]
        r = len(self.vload.shape)
        p = (np.arange(r) + 1) % r
        self.vload = sim.testing_distribution(tf.transpose(self.vload, perm=p))

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

               #iv = self.ir_lookback(self.estates, self.duration)   #
        iv =  cubic_interpolation(tmtau, self.etimes[0], self.Epi_cadence,self.estates)
        suscep = iv[:,1,:,:]
        

        ir = self.em.infection_rate(iv, axis=-3) # Infection rate  #
         # Shape: chain_shape + self.test_data.shape[0] + self.vtimes.shape
        
        N_days = self.test_data[-1,0] - self.test_data[0,0] 
        N_days = tf.dtypes.cast(N_days, tf.int32)
        ir_current = ir[:,0,:]
        integrand_0 = ir_current * self.vload
        #
        integrand_0= tf.reshape(integrand_0 , [1,integrand_0.shape[0], integrand_0.shape[1]])
        
        suscep_t = suscep[0, :, 0]
        
        
       
        

        for index_day in range(N_days):

            ir_current = ir[:,index_day +1,:]
            
            integrand_0_new = ir_current * self.vload
            integrand_0_new = tf.reshape(integrand_0_new , [1,integrand_0_new.shape[0], integrand_0_new.shape[1]])
            integrand_0 = tf.concat([integrand_0, integrand_0_new], axis = 0)
        
  
        #print(self.vload)
        
        

        

        ###Need to compute expectations
        
        ig_0 = tf.reduce_sum(integrand_0, axis=-1) * self.Vir_cadence * self.phi_s
        suscep_t = tf.reshape(suscep_t, [ig_0.shape[0], ig_0.shape[1]])
        
        
        
        ##computing expectations
        #        ig_0 = tf.reduce_mean(ig_0, axis = 1)

        
        return ig_0, suscep_t #######################################################################
#######################################################################
#######################################################################
######## Function to recast infection rate into right format without the interpolation wheb\n epi_cadence=vir_cadence=1
    def ir_lookback(self, estates, look_back_time):
        if self.Epi_cadence != 1. or self.Vir_cadence != 1.:
            raise ValueError("Epi_cadence and than Vir_cadence should both be 1. ")
        lb_time =  tf.dtypes.cast(look_back_time, tf.int32)
        N_days = self.test_data[-1,0] - self.test_data[0,0]
        N_days = tf.dtypes.cast(N_days, tf.int32)

        new_estates = estates[:, :, lb_time +1 :1:-1]

        for i in range(N_days ):
            new_estates_current = estates[:,:, i + 1 + lb_time + 1 : i + 1 + 1:-1]
            new_estates = tf.concat([new_estates, new_estates_current], axis = 0)
        
        new_estates = tf.transpose(new_estates, perm = [1, 0 , 2])
        new_estates = tf.reshape(new_estates, [1, new_estates.shape[0],new_estates.shape[1], new_estates.shape[2]])
    
    
        return (new_estates)

rtarr = np.array([[-0.5,0.5,1.5],
                  [-1.5,0.5,1.5],
                  [-1.5,-0.5,1.5],
                  [-1.5,-0.5,0.5]], dtype=np.float32)
rtarr = tf.constant(rtarr)

prodarr = np.array([-6.0, 2.0, -2.0, 6.0], dtype=np.float32)
prodarr = tf.constant(1/prodarr)





def cubic_interpolation(t, t0, dt, fvals):
    """
#    Produce a cubic interpolation of the vector function whose samples spaced
#    by dt starting at t0 are in the array fvals, using the Lagrange
#    interpolation formula.
#
#    Args:
#
#      t (`Tensor`[:]): times of desired interpolation
#      t0 (float): time corresponding to fvals[...,0]
#      dt (float): cadence of equally-spaced times
#      fvals (`Tensor`[...,:,:]): Function samples.  Left-most indices correspond
#        to chains. Second-to-last index is over vector (i.e. state) component
#        Right-most index is over samples.
#
#    Returns:
#      interpolant (`Tensor`[fvals.shape[:-1] + t.shape ]): Interpolant values.
    """


                                                        # Shapes:
    ts = t.shape ; fs = fvals.shape
    nt = (t-t0)/dt                                      # ts
 # to guarantee we have data for cubic
    assert(not tf.reduce_any(nt < 1) and
          not tf.reduce_any(nt > fs[-1]-2))

    i0 = tf.expand_dims(tf.cast(nt, tf.int64) - 1, -1)  # ts + [1]
    indices = i0 + tf.constant(np.arange(4)) - 1           # ts + [4] ##mng added -1 to test
##
    ftrain = tf.gather(fvals, indices, axis=-1)         # fs[:-1] + ts + [4]
##
    tt = tf.reshape(nt%1 - 0.5, ts + [1,1])             # ts + [1,1]
    res = tf.reduce_prod((tt-rtarr), axis=-1)           # ts + [4]
    res = res * prodarr                                 # ts + [4]
    res = res * ftrain                                  # fs[:-1] + ts + [4]
    res = tf.reduce_sum(res, axis=-1)                   # fs[:-1] + ts
#
    return res

