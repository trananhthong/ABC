import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N, Batch_num


DEBUG = False
M_0 = 0
N = 10000

# Prior s^2 ~ Scaled-Inv-Chi-sqr(v,s^2)
# If X ~ Scaled-Inv-Chi-sqr(v,s^2) then X ~ Inv-Gamma(v/2,(vs^2)/2)

def prior(v, s_sq, n):
    alpha = v / 2
    beta = v * s_sq / 2
    return invgamma(alpha, scale = beta, size = n)

# Generate true posterior. See Gelman's BDA Chapter 3.3.

def true_posterior(data, m_0, s_sq_0, v_0, k_0, n):
    m_data = np.mean(data)
    n_data = len(data)
    sum_sq_diff_data = np.sum([(x - m_data) ** 2 for x in data])

    k_n = k_0 + n_data
    v_n = v_0 + n_data
    v_n_times_s_sq_n = v_0 * s_sq_0 + sum_sq_diff_data + k_0 * n_data / k_n * (m_data - m_0) ** 2

    alpha = v_n / 2
    beta = v_n_times_s_sq_n / 2

    x = np.linspace(invgamma.ppf(1 / n, alpha, scale=beta), invgamma.ppf(1 - 1 / n, alpha, scale=beta), n)
    y = invgamma.pdf(x, alpha, scale=beta)

    return alpha, beta, x, y


# Kernels

def uniform_kernel(u, h):
    result = 1 / (2 * h) if np.abs(u / h) < 1 else 0
    return result


def triangle_kernel(u, h):
    result = (1 - np.abs(u / h)) if np.abs(u / h) < 1 else 0
    return result


# Distance measures euclidean (weighted and non weighted), mahalanobis

def euclidean_d(S, s_obs):
    return np.array([euclidean(s, s_obs) for s in S])

# def euclidean_d_(S, s_obs):
#     return np.array([mahalanobis(s, s_obs, np.identity(S.shape[1])) for s in S])

# Standardized Euclidean distance using np.mahalanobis
def s_euclidean_d(S, s_obs):
    #w = [1/np.var(s) for s in S.T]
    w = np.diag(np.diag(np.linalg.inv(np.cov(S.T))))
    return np.array([mahalanobis(s, s_obs, w) for s in S])

# Standardized Euclidean distance using np.seuclidean, which is surprisingly slower
def w_euclidean_d_(S, s_obs):
    w = [np.var(s) for s in S.T]
    return np.array([seuclidean(s, s_obs, w) for s in S])


def mahalanobis_d(S, s_obs):
    sigma = np.cov(S.T)
    sigma_inv = np.linalg.inv(sigma)
    return np.array([mahalanobis(s, s_obs, sigma_inv) for s in S])


# Sampling

def sampling(prior, likelihood, sample_size, repeats):
    simulations = []
    thetas = prior(repeats)
    for theta in thetas:
        y = likelihood(theta, sample_size)
        simulations.append((theta, y))
    return simulations


# Distributions for sampling

def scaled_inversed_chi_square(repeats):
    return invgamma.rvs(500, scale = 500, size = repeats)

def normal(var, repeats):
    return norm.rvs(M_0, var, repeats)

if __name__ == "__main__":
    start = time.process_time()
    for i in range(1, Batch_num + 1):
        start_i = time.process_time()
        simulations = sampling(scaled_inversed_chi_square, normal, N, 10000)
        np.save('simulations_' + str(i) + '.npy', simulations, allow_pickle=True)
        dur_i = time.process_time() - start_i
        print('Batch ' + str(i) + ' completed in ' + str(dur_i))
    dur = time.process_time() - start
    print('Simulation time: ' + str(dur))