import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from constants import M_0, S_SQ_0, N, agents, chunk_size, Alpha, Beta


# Prior s^2 ~ Scaled-Inv-Chi-sqr(v,s^2)
# If X ~ Scaled-Inv-Chi-sqr(v,s^2) then X ~ Inv-Gamma(v/2,(vs^2)/2)

def prior(v, s_sq, n):
    alpha = v / 2
    beta = v * s_sq / 2
    return invgamma(alpha, scale = beta, size = n)


# Sampling

def sampling(args):
    sample_mean, sample_size, repeats = args
    simulations = []
    variances = scaled_inversed_chi_square(repeats)
    for variance in variances:
        mean = normal(sample_mean, np.sqrt(variance/sample_size), 1)[0]
        y = normal(mean, np.sqrt(variance), sample_size)
        theta = np.array([mean, variance])
        simulations.append((theta, y))

    return np.array(simulations)



# Distributions for sampling

def scaled_inversed_chi_square(repeats):
    return invgamma.rvs(Alpha, scale = Beta, size = repeats)

def normal(mean, var, repeats):
    return norm.rvs(M_0, np.sqrt(var), size = repeats)

def simulation_run(data):
    start = time.time()
    mean_data = np.mean(data)

    batches_args = [(mean_data, N, chunk_size) for i in np.arange(1, agents + 1)]

    with Pool(processes=agents) as pool:
        results = pool.map(sampling, batches_args, chunk_size)

    simulations = np.concatenate([result for result in results])

    dur = time.time() - start
    print('Simulation time: ' + str(dur))

    return simulations