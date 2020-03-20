import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N, Sim_per_batch, Batch_num, Run_num, Cut_off
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
        true_posterior = np.load('true_posterior_sample.npy', allow_pickle = True)
        true_posterior_var = np.array([var for mean, var in true_posterior])

        # Summary statistics constructed by linear regression
        print('\nComputing ABC posteriors and Wasserstein distances...')

        for statistics_set in statistics_sets:
            data_parameter_estimate = np.load('parameter_estimates/data_' + statistics_set + '_estimate.npy', allow_pickle = True)
            sample_estimates = np.load('parameter_estimates/' + statistics_set + '_estimates.npy', allow_pickle = True)
            thetas = [(est[0], est[1]) for est in sample_estimates]
            parameter_estimates = np.array([np.array([est[2], est[3]]) for est in sample_estimates])

            data_statistics = np.load('statistics/data_' + statistics_set + '.npy', allow_pickle = True)
            sample_statistics = np.load('statistics/' + statistics_set + '.npy', allow_pickle = True)
            sample_statistics = np.array([col2 for col1,col2 in sample_statistics])


            for k,f in distance_measures.items():
                
                # Linear regression distance and posterior
                start_i = time.process_time()
                lr_distance_est = f(parameter_estimates, data_parameter_estimate).reshape(-1,1)
                lr_distances = np.hstack((thetas, lr_distance_est))
                np.save('distances/' + statistics_set + '_' + k + '_lr_distances.npy', lr_distances, allow_pickle = True)


                # Set h
                h = np.quantile(lr_distance_est, cut_off, interpolation='higher')
                lr_posterior = []

                for mean, variance, lr_distance in lr_distances:
                    if triangle_kernel(lr_distance, h) >= np.random.rand():
                        lr_posterior.append((mean, variance))

                posterior_var = np.array([var for mean, var in lr_posterior])
                w_d = wasserstein_distance(posterior_var, true_posterior_var)
                distance_dict[statistics_set + '_' + k + '_linear_regression_posterior_distance'].append(w_d)
                a_r = len(lr_posterior)/(Batch_num * N)
                acceptance_rate_dict[statistics_set + '_' + k + '_linear_regression_acceptance_rate'].append(a_r)



                np.save('ABC_posteriors/' + statistics_set + '_' + k + '_linear_regression_posterior.npy', np.array(lr_posterior), allow_pickle = True)

                dur_i = time.process_time() - start_i
                print('\n' + statistics_set + ' ' + k + ' distance and posterior with linear regression calculation completed in ' + str(dur_i))
                print('Wasserstein distance to true posterior: ' + str(w_d))
                print('Accepted: ' + str(len(lr_posterior)/(Batch_num * Sim_per_batch / 100)) + '%')


                # Raw statistics distance and posterior
                start_i = time.process_time()
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
                a_r = len(posterior)/(Batch_num * N)
                acceptance_rate_dict[statistics_set + '_' + k + '_acceptance_rate'].append(a_r)

                np.save('ABC_posteriors/' + statistics_set + '_' + k + '_posterior.npy', np.array(posterior), allow_pickle = True)

                dur_i = time.process_time() - start_i
                print('\n' + statistics_set + ' ' + k + ' distance and posterior calculation completed in ' + str(dur_i))
                print('Wasserstein distance to true posterior: ' + str(w_d))
                print('Accepted: ' + str(len(posterior)/(Batch_num * Sim_per_batch / 100)) + '%')

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

