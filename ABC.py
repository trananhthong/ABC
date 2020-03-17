import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N, Batch_num


# Kernels

def uniform_kernel(u, h):
    result = 1 / (2 * h) if np.abs(u / h) < 1 else 0
    return result


def triangle_kernel(u, h):
    result = (1 - np.abs(u / h)) if np.abs(u / h) < 1 else 0
    return result


# Distance measures euclidean (weighted and non weighted), mahalanobis

# Euclidean distance
def euclidean_d(S, s_obs):
    return np.array([euclidean(s, s_obs) for s in S])

# Standardized Euclidean distance using np.mahalanobis
def s_euclidean_d(S, s_obs):
    #w = [1/np.var(s) for s in S.T]
    w = np.diag(np.diag(np.linalg.inv(np.cov(S.T))))
    return np.array([mahalanobis(s, s_obs, w) for s in S])

# Weighted Euclidean distance 
def w_euclidean_d(S, s_obs, w):
    return np.array([euclidean(s, s_obs, w) for s in S])


def mahalanobis_d(S, s_obs):
    sigma = np.cov(S.T)
    sigma_inv = np.linalg.inv(sigma)
    return np.array([mahalanobis(s, s_obs, sigma_inv) for s in S])


# Distance and posterior calculation

def ABC():
    distance_measures = {'euclidean': euclidean_d, 's_euclidean': s_euclidean_d, 'mahalanobis': mahalanobis_d}
    statistics_sets = ['mean_variance', 'quantiles', 'min_max', 'mixed']

    for statistics_set in statistics_sets:
        data_parameter_estimate = np.load('parameter_estimates/data_' + statistics_set + '_estimate.npy', allow_pickle = True)
        sample_estimates = np.load('parameter_estimates/' + statistics_set + '_estimates.npy', allow_pickle = True)
        thetas = [(est[0], est[1]) for est in sample_estimates]
        parameter_estimates = np.array([np.array([est[2], est[3]]) for est in sample_estimates])

        for k,f in distance_measures.items():
            start_i = time.process_time()
            distance_est = f(parameter_estimates, data_parameter_estimate).reshape(-1,1)
            distances = np.hstack((thetas, distance_est))
            np.save('distances/' + statistics_set + '_' + k + '_distances.npy', distances, allow_pickle = True)

            # Set h
            h = np.quantile(distance_est, 0.03, interpolation='higher')
            posterior = []

            for mean, variance, distance in distances:
                if triangle_kernel(distance, h) >= np.random.rand():
                    posterior.append((mean, variance))

            print(len(posterior))

            np.save('ABC_posteriors/' + statistics_set + '_' + k + '_posterior.npy', np.array(posterior), allow_pickle = True)

            dur_i = time.process_time() - start_i
            print(statistics_set + ' ' + k + ' distance and posterior calculation completed in ' + str(dur_i))



if __name__ == '__main__':

    start1 = time.process_time()
    ABC()
    dur1 = time.process_time() - start1
    print('All distances calculated in ' + str(dur1))




