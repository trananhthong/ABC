import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N, Batch_num


# Prior s^2 ~ Scaled-Inv-Chi-sqr(v,s^2)
# If X ~ Scaled-Inv-Chi-sqr(v,s^2) then X ~ Inv-Gamma(v/2,(vs^2)/2)

def prior(v, s_sq, n):
    alpha = v / 2
    beta = v * s_sq / 2
    return invgamma(alpha, scale = beta, size = n)


# Sampling

def sampling(sample_mean, sample_size, repeats):
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
    return invgamma.rvs(N/2, scale = N/2*S_SQ_0, size = repeats)

def normal(mean, var, repeats):
    return norm.rvs(M_0, np.sqrt(var), size = repeats)

if __name__ == "__main__":
    start = time.process_time()
    data = np.load('data.npy', allow_pickle = True)
    mean_data = np.mean(data)

    for i in range(1, Batch_num + 1):
        start_i = time.process_time()
        simulations = sampling(mean_data, N, 10000)
        np.save('simulations/simulations_' + str(i) + '.npy', simulations, allow_pickle=True)
        dur_i = time.process_time() - start_i
        print('Batch ' + str(i) + ' completed in ' + str(dur_i))

    dur = time.process_time() - start
    print('Simulation time: ' + str(dur))