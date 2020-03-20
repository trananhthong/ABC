import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from constants import M_0, S_SQ_0, N, Sim_per_batch, Batch_num, agents, chunk_size


# Prior s^2 ~ Scaled-Inv-Chi-sqr(v,s^2)
# If X ~ Scaled-Inv-Chi-sqr(v,s^2) then X ~ Inv-Gamma(v/2,(vs^2)/2)

def prior(v, s_sq, n):
    alpha = v / 2
    beta = v * s_sq / 2
    return invgamma(alpha, scale = beta, size = n)


# Sampling

def sampling(args):
    sample_mean, sample_size, repeats, i = args
    simulations = []
    variances = scaled_inversed_chi_square(repeats)
    for variance in variances:
        mean = normal(sample_mean, np.sqrt(variance/sample_size), 1)[0]
        y = normal(mean, np.sqrt(variance), sample_size)
        theta = np.array([mean, variance])
        simulations.append((theta, y))

    np.save('simulations/simulations_' + str(i) + '.npy', np.array(simulations), allow_pickle=True)



# Distributions for sampling

def scaled_inversed_chi_square(repeats):
    return invgamma.rvs(8, scale = 4, size = repeats)

def normal(mean, var, repeats):
    return norm.rvs(M_0, np.sqrt(var), size = repeats)

def simulation_run():
    start = time.time()
    data = np.load('data.npy', allow_pickle = True)
    mean_data = np.mean(data)

    batches_args = [(mean_data, N, Sim_per_batch, i) for i in np.arange(1, Batch_num + 1)]

    with Pool(processes=agents) as pool:
        pool.map(sampling, batches_args, chunk_size)

    dur = time.time() - start
    print('Simulation time: ' + str(dur))