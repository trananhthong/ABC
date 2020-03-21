import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from constants import M_0, S_SQ_0, N, Batch_size, Batch_num, Run_num, Cut_off, distance_agents, distance_chunk_size
from data_generator import data_generator_run
from simulation import simulation_run
from statistic_generator import statistic_generator_run
from parameter_regression import parameter_regression_run


# Kernels

def uniform_kernel(u, h):
    result = 1 / (2 * h) if np.abs(u / h) < 1 else 0
    return result


def triangle_kernel(u, h):
    result = (1 - np.abs(u / h)) if np.abs(u / h) < 1 else 0
    return result


# Distance measures euclidean (weighted and non weighted), mahalanobis

# Euclidean distance
def eucl(args):
    s, s_obs = args
    return euclidean(s, s_obs)

def euclidean_d(S, s_obs):
    args = [(s, s_obs) for s in S]
    with Pool(processes=distance_agents) as pool:
        results = pool.map(eucl, args, distance_chunk_size)
    return np.array(results)

# Standardized Euclidean distance using np.mahalanobis
def s_eucl(args):
    s, s_obs, w = args
    return mahalanobis(s, s_obs, w)

def s_euclidean_d(S, s_obs):
    w = np.diag(np.diag(np.linalg.inv(np.cov(S.T))))
    args = [(s, s_obs, w) for s in S]
    with Pool(processes=distance_agents) as pool:
        results = pool.map(s_eucl, args, distance_chunk_size)
    return np.array(results)

# Weighted Euclidean distance 
def w_euclidean_d(S, s_obs, w):
    return np.array([euclidean(s, s_obs, w) for s in S])

def maha(args):
    s, s_obs, sigma_inv = args
    return mahalanobis(s, s_obs, sigma_inv)

def mahalanobis_d(S, s_obs):
    sigma = np.cov(S.T)
    sigma_inv = np.linalg.inv(sigma)
    args = [(s, s_obs, sigma_inv) for s in S]
    with Pool(processes=distance_agents) as pool:
        results = pool.map(maha, args, distance_chunk_size)
    return np.array(results)



# ABC

def ABC(distance_dict, acceptance_rate_dict, cut_off, runs):
    for i in range(runs):
        print('RUN ' + str(i + 1) + '\n')
        start = time.time()
        print('\nGenerating data and true posterior...')
        data_generator_run()
        print('\nGenerating simulation...')
        simulation_run()
        print('\nGenerating statistics...')
        statistic_generator_run()
        print('\nDoing parameter regression...')
        parameter_regression_run()

        distance_measures = {'euclidean': euclidean_d, 's_euclidean': s_euclidean_d, 'mahalanobis': mahalanobis_d}
        statistics_sets = ['mean_variance', 'quantiles', 'min_max', 'mixed']
        t = np.load('true_posterior_sample.npy', allow_pickle = True)
        true_posterior_var = np.array([var for mean, var in t])
        del t

        # Summary statistics constructed by linear regression
        print('\nComputing ABC posteriors and Wasserstein distances...')

        for statistics_set in statistics_sets:
            a = np.load('parameter_estimates/data_' + statistics_set + '_estimate.npy', allow_pickle = True)
            data_parameter_estimate = a.copy()
            del a
            a = np.load('parameter_estimates/' + statistics_set + '_estimates.npy', allow_pickle = True)
            sample_estimates = a.copy()
            del a
            thetas = [(est[0], est[1]) for est in sample_estimates]
            parameter_estimates = np.array([np.array([est[2], est[3]]) for est in sample_estimates])

            a = np.load('statistics/data_' + statistics_set + '.npy', allow_pickle = True)
            data_statistics = a.copy()
            del a
            a = np.load('statistics/' + statistics_set + '.npy', allow_pickle = True)
            sample_statistics = np.array([col2 for col1,col2 in a])
            del a


            for k,f in distance_measures.items():
                
                # Linear regression distance and posterior
                start_i = time.time()
                lr_distance_est = f(parameter_estimates, data_parameter_estimate).reshape(-1,1)
                lr_distances = np.hstack((thetas, lr_distance_est))
                np.save('distances/' + statistics_set + '_' + k + '_lr_distances.npy', lr_distances, allow_pickle = True)


                # Set h
                h = np.quantile(lr_distance_est, cut_off, interpolation='higher')
                lr_posterior = []

                for mean, variance, lr_distance in lr_distances:
                    if triangle_kernel(lr_distance, h) >= np.random.rand():
                        lr_posterior.append((mean, variance))

                lr_posterior_var = np.array([row[1] for row in lr_posterior])
                lr_w_d = wasserstein_distance(lr_posterior_var, true_posterior_var)
                distance_dict[statistics_set + '_' + k + '_linear_regression_posterior_distance'].append(lr_w_d)
                lr_a_r = len(lr_posterior)/(Batch_num * Batch_size)
                acceptance_rate_dict[statistics_set + '_' + k + '_linear_regression_acceptance_rate'].append(lr_a_r)


                np.save('ABC_posteriors/' + statistics_set + '_' + k + '_linear_regression_posterior.npy', np.array(lr_posterior), allow_pickle = True)

                dur_i = time.time() - start_i
                print('\n' + statistics_set + ' ' + k + ' distance and posterior with linear regression calculation completed in ' + str(dur_i))
                print('Wasserstein distance to true posterior: ' + str(lr_w_d))
                print('Accepted: ' + str(len(lr_posterior)/(Batch_num * Batch_size / 100)) + '%')


                # Raw statistics distance and posterior
                start_i = time.time()
                distance_est = f(sample_statistics, data_statistics).reshape(-1,1)
                distances = np.hstack((thetas, distance_est))
                np.save('distances/' + statistics_set + '_' + k + '_distances.npy', distances, allow_pickle = True)


                # Set h
                h = np.quantile(distance_est, cut_off, interpolation='higher')
                posterior = []

                for mean, variance, distance in distances:
                    if triangle_kernel(distance, h) >= np.random.rand():
                        posterior.append((mean, variance))

                posterior_var = np.array([var for mean, var in posterior])
                w_d = wasserstein_distance(posterior_var, true_posterior_var)
                distance_dict[statistics_set + '_' + k + '_posterior_distance'].append(w_d)
                a_r = len(posterior)/(Batch_num * Batch_size)
                acceptance_rate_dict[statistics_set + '_' + k + '_acceptance_rate'].append(a_r)

                np.save('ABC_posteriors/' + statistics_set + '_' + k + '_posterior.npy', np.array(posterior), allow_pickle = True)

                dur_i = time.time() - start_i
                print('\n' + statistics_set + ' ' + k + ' distance and posterior calculation completed in ' + str(dur_i))
                print('Wasserstein distance to true posterior: ' + str(w_d))
                print('Accepted: ' + str(len(posterior)/(Batch_num * Batch_size / 100)) + '%')

        dur = time.time() - start
        print('\nRun ' + str(i+1) + ' completed in: ' + str(dur) + '\n\n\n')





if __name__ == '__main__':
    distance_measures = ['euclidean', 's_euclidean', 'mahalanobis']
    statistics_sets = ['mean_variance', 'quantiles', 'min_max', 'mixed']
    distance_results = {}
    acceptance_rate_results = {}

    for statistics_set in statistics_sets:
        for distance_measure in distance_measures:
            distance_results[statistics_set + '_' + distance_measure + '_linear_regression_posterior_distance'] = []
            distance_results[statistics_set + '_' + distance_measure + '_posterior_distance'] = []
            acceptance_rate_results[statistics_set + '_' + distance_measure + '_linear_regression_acceptance_rate'] = []
            acceptance_rate_results[statistics_set + '_' + distance_measure + '_acceptance_rate'] = []



    start1 = time.time()
    ABC(distance_results, acceptance_rate_results, Cut_off, Run_num)
    np.save('wasserstein_distance_results.npy', distance_results, allow_pickle = True)
    np.save('acceptance_rate_results.npy', acceptance_rate_results, allow_pickle = True)
    dur1 = time.time() - start1
    print('All ' + str(Run_num) + ' ABC run completed in: ' + str(dur1))

