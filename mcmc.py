import functools
import sys
import math
#import scipy.io
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp

import Test_Likelihood as tl
import ODE_Dynamics as od
import Positive_Symptom_fn as fn

from functools import partial

import scipy.stats as ss

import simulation as sim
import model as mdl

tfd = tfp.distributions
tfb = tfp.bijectors

n_sample = 1000
n_burnin = 5000
n_adaptation = 4000

vload = sim.sample_viral_load(mu_b = 12,sigma_b = 1)
prob_s_i0 = 0.55; prob_s_ibar0 = 0.1
p_threshold0 = 170306.4 * 1E-05
s_threshold0 = sim.get_symptom_threshold(vload)
start_day0 = 10
full_data = sim.simulate_epidemic(vload, pop_size = 10000, start_day = start_day0,
                             prob_s_i = prob_s_i0, prob_s_ibar = prob_s_ibar0, 
                             v_threshold = p_threshold0)
start_day = 30
end_day = 40
sliced_data = full_data[start_day:end_day,:]
#print(sliced_data)

# Arrange data in workable format
days = [float(i) for i in list(range(len(sliced_data)))]
tests = sliced_data[:,0]
positives = sliced_data[:,1]
test_data = np.column_stack((days, tests, positives))
test_data = tf.cast(test_data, dtype = tf.float32)
#print(test_data)

vdyn_ode_fn = od.ViralDynamics
positive_fn = fn.proba_pos_sym(p_threshold0).positive_fn
symptom_fn = fn.proba_pos_sym(p_threshold0).symptom_fn
prob_s_ibar = prob_s_ibar0
sample_size = 1000
k = 1
index = 1
mu_b, sigma_b = 12, 1
beta = np.random.normal(mu_b, sigma_b, 1)   #"rate at which virus infects host cells"
L = 0.0025/beta

V0 = np.random.normal(1E3, 1E2, 1)
X0 = 1E6
Y0 = V0

par=np.array([[L,0.01,beta*1E-7,0.5,20.0,10.0,V0,X0,Y0]])

init_state=(np.array([[V0,X0,Y0]], dtype=np.float32))

while index <= sample_size - 1:
    beta = np.random.normal(mu_b, sigma_b, 1)   #"rate at which virus infects host cells"
    L = 0.0025/beta
    
    V0 = np.random.normal(1E3, 1E2, 1)
    X0 = 1E6
    Y0 = V0
    
    par_new=np.array([[L,0.01,beta*1E-7,0.5,20.0,10.0,V0,X0,Y0]])
    par = np.concatenate((par, par_new), axis = 0)
    
    init_state_new=(np.array([[V0,X0,Y0]], dtype=np.float32))
    init_state = np.concatenate((init_state, init_state_new), 0)

    index +=1
        

vpar = tf.constant(par, dtype=tf.float32)
pospar = par
sympar = par
#print(par.shape)

# Joint prior on nu, i, s
# nu ~ Beta(2,8), so that mode = 0.1
# i, s, (1-i-s) ~ Dirichlet(0.1,0.1,0.1)

joint_prior = tfd.JointDistributionNamed(dict(
    recov_rate = tfd.Beta(2,8),
    epi_state = tfd.Dirichlet(
    concentration = np.ones(3, np.float32)/10),
))

# Joint log prior
def joint_log_prior(R0, nu, epi_state):
    epi_state = np.array(epi_state)
    return joint_prior.log_prob(
      recov_rate = nu, epi_state = epi_state)- math.log(R0)

# Joint log-likelihood
# Prior on R0: 1/R0

def joint_log_prob(test_data, R0, nu, epi_state):
    epi_state = np.array(epi_state)
    i = epi_state[...,0]
    s = epi_state[...,1]
    epipar = tf.constant(np.array([[R0, 5.0E-08, nu, i, s]], dtype=np.float32))

    loglike = mdl.loglik(test_data, vdyn_ode_fn, positive_fn, symptom_fn, prob_s_ibar = 0.1, prob_fp=0.0, Epi_Model=od.SIR,
                 duration= start_day0, Epi_cadence = 0.5, Vir_cadence = 0.0675, phi_s = 0.55, psi_s = 0.1)
    ll,pp = loglike.__call__(test_data, epipar, vpar, pospar, sympar)
    epi_state = [i,s,1.0-i-s]
    ll = ll + joint_log_prior(R0, nu, epi_state).numpy()
    return sum(ll)

#joint_log_prob(test_data, R0 = 1.8, nu = 0.1, epi_state = [0.01,0.9, 0.09])

# Compute log_lik 
# Impute things we don't want to sample, eg. observations

unnormalized_posterior_log_prob = partial(joint_log_prob, test_data)

R0 = 1.8
nu0 = 0.1
i0 = 0.02237592; s0 = 0.9492399; r0 = 0.02838415
initial_state = [tf.convert_to_tensor(R0, dtype = tf.float32), tf.convert_to_tensor(nu0,dtype = tf.float32), tf.convert_to_tensor([i0, s0, r0], dtype = tf.float32)]

# Transform all parameters so that parameter spaces are unconstrained
# Every parameter needs a bijector, might be 'Identity'

unconstraining_bijectors = [
    tfb.Exp(),
    tfb.Sigmoid(),
    tfb.SoftmaxCentered()  
]

#@tf.function(autograph=False)
def sample():
    return tfp.mcmc.sample_chain(
        num_results = n_sample,
        num_burnin_steps = n_burnin,
        current_state = initial_state,
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn = unnormalized_posterior_log_prob,
                     step_size = 0.01,
                     num_leapfrog_steps = 2),
                bijector = unconstraining_bijectors),
             num_adaptation_steps = n_adaptation),
        trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

[R0, nu, epi_state], is_accepted = sample()

# Compute posterior means of Epidemic parameters

acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32)).numpy()
mean_R0 = tf.reduce_mean(R0, axis=0).numpy()
mean_nu = tf.reduce_mean(nu, axis=0).numpy()
i_s = np.array(epi_state)
i = i_s[...,0]
s = i_s[...,1]
r = i_s[...,2]
mean_i = tf.reduce_mean(i, axis=0).numpy()
mean_s = tf.reduce_mean(s, axis=0).numpy()
mean_r = tf.reduce_mean(r, axis=0).numpy()

results = np.column_stack((R0.numpy(),nu.numpy(),i,s,r))
headings = ['R0', 'nu', 'Infected', 'Susceptible', 'Recovered']
results = np.vstack([headings, results])
    
with open("mcmc_draws_40_bijector_exp.txt", "w") as txt_file:
    for line in results:
        txt_file.write(" ".join(line) + "\n")