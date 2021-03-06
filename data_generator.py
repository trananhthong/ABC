import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import time
from constants import M_0, S_SQ_0, N


# Generate data

def generate_data(M_0, S_SQ_0, N):
    data = norm.rvs(M_0, np.sqrt(S_SQ_0), N)

    return data


# Generate true posterior. See Gelman's BDA Chapter 3.3.

def true_posterior(data, m_0, s_sq_0, v_0, k_0, n):
    m_data = np.mean(data)
    n_data = len(data)
    sum_sq_diff_data = np.sum([(x - m_data) ** 2 for x in data])

    k_n = k_0 + n_data
    v_n = v_0 + n_data
    m_n = k_0 * m_0 / k_n + n_data * m_data / k_n
    v_n_times_s_sq_n = v_0 * s_sq_0 + sum_sq_diff_data + k_0 * n_data / k_n * (m_data - m_0) ** 2

    alpha = v_n / 2
    beta = v_n_times_s_sq_n / 2

    x = np.linspace(invgamma.ppf(1 / n, alpha, scale=beta), invgamma.ppf(1 - 1 / n, alpha, scale=beta), n*10)
    y = invgamma.pdf(x, alpha, scale=beta)
    
    return alpha, beta, m_n, k_n, x, y


def data_generator_run():
    data = generate_data(M_0, S_SQ_0, N)

    alpha, beta, m, k, x, y = true_posterior(data, M_0 * 1.1, S_SQ_0 * 1.5, 1, 1, N)
    posterior_var = invgamma.rvs(alpha, scale = beta, size = N)
    posterior_mean = np.array([norm.rvs(m, scale = np.sqrt(var/k), size = 1)[0] for var in posterior_var])
    true_posterior_sample = np.dstack((posterior_mean, posterior_var))[0]
    true_posterior_var_pdf = np.dstack((x,y))[0]
    
    return data, true_posterior_sample, true_posterior_var_pdf
