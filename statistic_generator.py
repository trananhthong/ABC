import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N, Batch_num


# Generator for summary statistics

def summary_statistics(statistic_choices):
    summaries = np.array([])

    for i in range(1, Batch_num + 1):
        start_i = time.process_time()
        batch = np.load('simulations/simulations_' + str(i) + '.npy', allow_pickle=True)
        batch_summaries = []
        for theta, y in batch:
            statistics = statistic_choices(y)
            batch_summaries.append((theta, statistics))
        summaries = np.append(summaries, batch_summaries)
        dur_i = time.process_time() - start_i
        print('Batch ' + str(i) + ' processed in ' + str(dur_i))

    return summaries


# Statistics choices

def mean_variance(data):
    return np.array([np.mean(data), np.var(data)])


def quantiles(data):
    q = []

    for i in range(1, 20):
        q.append(np.quantile(data, 0.05 * i, interpolation='midpoint'))

    return np.array(q)


def min_max(data):
    return np.array([np.min(data), np.max(data)])


def mixed(data):
    return np.append(np.array([np.mean(data), np.var(data), np.min(data), np.max(data)]), np.array([np.quantile(data, 0.1 * i, interpolation='midpoint') for i in range(1,10)]))



if __name__ == '__main__':
    data = np.load('Code/data.npy', allow_pickle = True)

    start_1 = time.process_time()
    mean_var_stats = summary_statistics(mean_variance)
    np.save('statistics/mean_variance' + '.npy', mean_var_stats, allow_pickle=True)
    np.save('statistics/data_mean_variance' + '.npy', mean_variance(data), allow_pickle=True)
    dur_1 = time.process_time() - start_1
    print('Mean-Variance process completed in ' + str(dur_1))

    start_2 = time.process_time()
    quantiles_stats = summary_statistics(quantiles)
    np.save('statistics/quantiles' + '.npy', quantiles_stats, allow_pickle=True)
    np.save('statistics/data_quantiles' + '.npy', quantiles(data), allow_pickle=True)
    dur_2 = time.process_time() - start_2
    print('Quantiles process completed in ' + str(dur_2))
    
    start_3 = time.process_time()
    min_max_stat = summary_statistics(min_max)
    np.save('statistics/min_max' + '.npy', min_max_stat, allow_pickle=True)
    np.save('statistics/data_min_max' + '.npy', min_max(data), allow_pickle=True)
    dur_3 = time.process_time() - start_3
    print('Min-Max process completed in ' + str(dur_3))

    start_4 = time.process_time()
    min_max_stat = summary_statistics(mixed)
    np.save('statistics/mixed' + '.npy', mixed_stat, allow_pickle=True)
    np.save('statistics/data_mixed' + '.npy', mixed(data), allow_pickle=True)
    dur_4 = time.process_time() - start_4
    print('Mixed process completed in ' + str(dur_4))

    dur = time.process_time() - start_1
    print('All completed in: ' + str(dur))
