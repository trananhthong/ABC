import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N


# Generate data

def generate_data(M_0, S_SQ_0, N):
    data = norm.rvs(M_0, S_SQ_0, N)

    return data


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


if __name__ == "__main__":
    data = generate_data(M_0, S_SQ_0, N)
    np.save('data.npy', data, allow_pickle=True)

    alpha, beta, x, y = true_posterior(data, M_0, 2,1,1, 10000)
    posterior_pdf = np.dstack((x,y))[0]
    posterior_sample = invgamma.rvs(alpha, scale = beta, size = 10000)
    np.save('true_posterior_pdf.npy', posterior_pdf, allow_pickle=True)
    np.save('true_posterior_sample.npy', posterior_sample, allow_pickle=True)