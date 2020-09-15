#!/usr/bin/python3

# Lightweight base class for ODE dynamical systems.
#
# Enakshi Saha, University of Chicago

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import sys
sys.path.append("..")
import ODE_Dynamics as od
import math
from random import sample

#######################################################################
#######################################################################
#######################################################################

def sample_viral_load(mu_b = 12, sigma_b = 1, duration = 160):
    # Store 1000 viral load curves
    sample_size = 1000
    k = 1
    index = 1
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

    vm = od.ViralDynamics(vpar)
    D_Vir = vm.Ndim
    initial_state = vpar[...,-D_Vir:]
    st1 = 0.0
    vdyn_initial_time = st1
    st2 = duration
    vtimes = tf.constant(np.arange(st1, st2, step = 1, 
                                            dtype=np.float32))

    DP = tfp.math.ode.DormandPrince()
    results = DP.solve(vm.RHS, vdyn_initial_time, initial_state,
                           solution_times = vtimes)

    vload = results.states[...,0]  
    # But this has shape self.vtimes.shape[0] + vpar.shape[:-1].
    # We want vpar.shape[:-1] + self.vtimes.shape[0]
    r = len(vload.shape)
    p = (np.arange(r) + 1) % r
    vload = tf.transpose(vload, perm = p)
    
    return vload

def get_symptom_threshold(vload):
    # Compute empirical distribution of viral load peak days
    max_vload = []
    for id in range(len(vload)):
        v = vload[id,:].numpy()
        maxload = max(v)
        max_vload.append(maxload)
        
    s_threshold = np.percentile(max_vload, 40)
        
    return s_threshold

def testing_distribution(vload):
    # Compute empirical distribution of viral load peak days

    maxpos = []
    for id in range(len(vload)):
        v = vload[id,:].numpy()
        pos = np.argmax(v)
        maxpos.append(pos)
    from collections import Counter
    v_dist = Counter(maxpos)
    vmax_prob = []
    for x in range(vload.shape[1]):
        vmax_prob.append(v_dist[x]/len(vload))
        
    return vmax_prob

    

def simulate_epidemic(vload, start_day = 10, duration = 160, pop_size = 10000, prob_s_i = 0.6, prob_s_ibar = 0.05, prob_fp = 0, v_threshold = 170306.4 * 1E-05):
    
    #symp_threshold = get_symptom_threshold(vload)
    sample_size = len(vload)
    vmax_prob = testing_distribution(vload)
    peak_day = np.argmax(vmax_prob)
    n_tests = []; n_positives = []; n_new_infections = [] ; n_true_negatives = []; n_false_positives = [];

    # Set population size and duration of the epidemic
    pop_id = [id for id in range(pop_size)]

    # Set parameters for epidemic model
    R0 = 1.8
    mu = 5.0E-08
    nu = 0.1
    par=tf.constant(np.array([[R0, mu, nu]], dtype = np.float32))
    
    # Get epidemic model
    mod = od.SIR(par)

    # At the start of the epidemic most people are succeptible
    init_state=tf.constant(np.array([[0.001,0.999]], dtype=np.float32))

    # Generate time stamps for duration of epidemic
    init_time=tf.constant(0)
    num = int(duration)
    soln_times=tf.constant(np.linspace(0, duration, num, dtype=np.int32))

    # Get I, S, R values through duration of the epidemic
    dp = tfp.math.ode.DormandPrince()
    results = dp.solve(mod.RHS, init_time, init_state, solution_times=soln_times)
    t = results.times
    i = results.states[:,0,0] ; s = results.states[:,0,1] ; r = 1.0 - i - s;

    # Before epidemic everybody is succeptible
    I = []
    S = pop_id
    R = []

    # Record id of individuals in I, S, R compartments at time = 0

    time = 0
    n_i = int(pop_size * i.numpy()[time])
    n_s = int(pop_size * s.numpy()[time])
    n_r = int(pop_size * r.numpy()[time])
    
    I = sample(pop_id, n_i)
    S = [id for id in S if id not in I]
    R = []

    n_new_infections.append(len(I))

    # Record time stamp of infection for each infected individual

    I_T = [0] * len(I)

    # Record people who tested positive

    positives = []
    
    time = 1
    while time < start_day:
    
        # New births
        n_b = math.floor(0.5 + mu * len(S))
        pop_size += n_b
    
        if n_b > 0:
            id_first = pop_size + 1
            id_last = id_first + n_b -1
            new_id = [id for id in range(id_first, id_last, 1)]
            S.append(new_id)
    
        # Removal by death
        s1 = math.floor(0.5 + mu * len(S))
        if s1 > 0:
            S1 = sample(S, s1)
            S = [id for id in S if id not in S1]

        i1 = math.floor(0.5 + mu * len(I))
        if i1 > 0:
            I1 = sample(I, i1)
            for id in I1:
                ind = I.index(id)
                # print(ind, len(I_T), len(I))
                del I_T[ind]
                del I[ind]
        
        r1 = math.floor(0.5 + mu * len(R))
        if r1 > 0:
            R1 = sample(R, r1)
            R = [id for id in R if id not in R1]

        pop_size = pop_size - s1 - i1 -r1
    
        n_i0 = int(pop_size * i.numpy()[time-1])
        n_s0 = int(pop_size * s.numpy()[time-1])
        n_r0 = pop_size - n_i0 - n_s0
    
        n_i = int(pop_size * i.numpy()[time])
        n_s = int(pop_size * s.numpy()[time])
        n_r = pop_size - n_i - n_s
    
        # New numbers of I, S, R at time
        dn_s = n_s0 - n_s; dn_r = n_r - n_r0
    
        n_new_infections.append(dn_s)
    
        # Update I, S, R compartments
        if dn_s <= len(S):
            I0 = sample(S, dn_s)
            S = [id for id in S if id not in I0]
            I = I + I0
            I_T = I_T + [time] * len(I0)
        else:
            I = I + S
            S = []
    
        if dn_r <= len(I):
            #R0 = sample(I, dn_r)
            #recov_time = np.random.exponential(scale= 1/nu, size = len(I))
            R0 = [id for id in I if I_T[I.index(id)] >= peak_day]
            if dn_r <= len(R0):
                R0 = sample(R0, dn_r)
            else:
                R0 = R0
                #break
            for id in R0:
                ind = I.index(id)
                del I_T[ind]
                del I[ind]
            R = R + R0
        else:
            R = R + I
            I = []
            I_T = []
            break
        time += 1

    while time < duration:
    
        # New births
        n_b = math.floor(0.5 + mu * len(S))
        pop_size += n_b
    
        if n_b > 0:
            id_first = pop_size + 1
            id_last = id_first + n_b -1
            new_id = [id for id in range(id_first, id_last, 1)]
            S.append(new_id)
    
        # Removal by death
        s1 = math.floor(0.5 + mu * len(S))
        if s1 > 0:
            S1 = sample(S, s1)
            S = [id for id in S if id not in S1]

        i1 = math.floor(mu * len(I))
        if i1 > 0:
            I1 = sample(I, i1)
            for id in I1:
                ind = I.index(id)
                del I_T[ind]
                del I[ind]
        
        r1 = math.floor(0.5 + mu * len(R))
        if r1 > 0:
            R1 = sample(R, r1)
            R = [id for id in R if id not in R1]

        pop_size = pop_size - s1 - i1 -r1
    
        n_i0 = int(pop_size * i.numpy()[time-1])
        n_s0 = int(pop_size * s.numpy()[time-1])
        n_r0 = pop_size - n_i0 - n_s0
    
        n_i = int(pop_size * i.numpy()[time])
        n_s = int(pop_size * s.numpy()[time])
        n_r = pop_size - n_i - n_s
    
        # New numbers of I, S, R at time
        dn_s = n_s0 - n_s; dn_r = n_r - n_r0
    
        n_new_infections.append(dn_s)
    
        # Update I, S, R compartments
        if dn_s <= len(S):
            I0 = sample(S, dn_s)
            S = [id for id in S if id not in I0]
            I = I + I0
            I_T = I_T + [time] * len(I0)
        else:
            I = I + S
            S = []
    
        if dn_r <= len(I):
            #R0 = sample(I, dn_r)
            #recov_time = np.random.exponential(scale= 1/nu, size = len(I))
            #R0 = [id for id in I if int(time - I_T[I.index(id)]) >= recov_time[I.index(id)]]
            R0 = [id for id in I if I_T[I.index(id)] >= peak_day]
            if dn_r <= len(R0):
                R0 = sample(R0, dn_r)
            else:
                R0 = R0
                break
            for id in R0:
                ind = I.index(id)
                del I_T[ind]
                del I[ind]
            R = R + R0
        else:
            R = R + I
            I = []
            I_T = []
            break
    
        # Choose infected individuals who are symptomatic
        smp_i = []; I_T_smp = []
        #for id in range(len(I)):
         #   if int(np.random.binomial(size = 1, n = 1, p = prob_s_i))== 1:
          #      smp_i.append(I[id])
           #     I_T_smp.append(I_T[id])
              
        for id in range(len(I)):
            tau = int(time - I_T[id])
            random_id = sample(range(sample_size), 1)[0]
            # v_tau = vload[random_id, tau].numpy()
            v_max = np.argmax(vload[random_id, :].numpy())
            if tau == v_max:
                 if int(np.random.binomial(size = 1, n = 1, p = prob_s_i))== 1:
                    smp_i.append(I[id])
                    I_T_smp.append(I_T[id])
    
        # Choose healthy individuals who are symptomatic
        S_R = S + R
        smp_ibar = [id for id in S_R if int(np.random.binomial(size = 1, n = 1, p = prob_s_ibar))== 1]
    
        # All symptomatic individual
        smp = smp_i + smp_ibar
        T_smp = I_T_smp + [-10000] * len(smp_ibar)
    
        # Only those individuals are tested who have never been tested positive
        tested = [id for id in smp if id not in positives]
        n_tests.append(len(tested))
    
        # Find viral load of infected individuals being tested
    
        # Determine which individuals are tested positive
    
        # Positives among infected
        pos1 = []
    
        # Find viral load of all infected individuals being tested
        smp_i_tested = [id for id in smp_i if id in tested]
        for id in range(len(smp_i_tested)):
            tau = int(time - I_T_smp[id])
            random_id = sample(range(sample_size), 1)[0]
            v_tau = vload[random_id, tau].numpy()
            if v_tau > v_threshold:
                pos1.append(id)
    
        # Positives among uninfected
        pos2 = [id for id in smp_ibar if int(np.random.binomial(1, prob_fp, 1)) == 1]
        n_true_negatives.append(len(tested) - len(smp_i_tested))
        n_false_positives.append(len(pos2))
    
        # All positive tests
        pos = pos1 + pos2
        n_positives.append(len(pos))
    
        # Update historical set of all positive individuals
        positives = positives + pos
    
        #print(time)
        time += 1
        print(time)
    
    results = np.column_stack((n_tests,n_positives))
    simulation_results = np.column_stack((n_tests,n_positives,n_new_infections[start_day:duration],n_true_negatives, n_false_positives))
    headings = ['Tests', 'Positives', 'New Infections', 'True Negatives', 'False Positives']
    simulation_results = np.vstack([headings, simulation_results])
    
    with open("simulation_data.txt", "w") as txt_file:
        for line in simulation_results:
            txt_file.write(" ".join(line) + "\n")

    return results
    