import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import time
from multiprocessing import Pool
from constants import M_0, S_SQ_0, N, agents, chunk_size, Batch_num, Batch_size


# Generator for summary statistics

def summary_statistics_par(args):
    Y, f = args

    statistics = np.array([f(y) for y in Y])

    return statistics

def summary_statistics(Y, f):
    statistics = np.array([f(y) for y in Y])

    return statistics


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



def statistic_generator_run_par(data, simulations):
    start = time.time()
    choices = {'mean_variance': mean_variance, 'quantiles': quantiles, 'min_max': min_max, 'mixed': mixed}
    statistics = {}
    data_statistics = {}

    for k,f in choices.items():
        start_i = time.time()
        print('Starting ' + k + ' computation...')
        
        y_batch = np.array_split(np.array([y for theta, y in simulations]), Batch_size)
        args = [(y,f) for y in y_batch]
        thetas = np.array([theta for theta, y in simulations])
        
        with Pool(processes=agents) as pool:
            results = pool.map(summary_statistics_par, args, chunk_size)

        stats = np.vstack([result for result in results])
        stats = np.hstack((thetas, stats))
        statistics[k] = stats
        data_statistics[k] = f(data)
        dur_i = time.time() - start_i
        print(k + ' computation completed in: ' + str(dur_i) + '\n')

    dur = time.time() - start
    print('Statistics generating time: ' + str(dur))

    return statistics, data_statistics


def statistic_generator_run(data, simulations):
    start = time.time()
    choices = {'mean_variance': mean_variance, 'quantiles': quantiles, 'min_max': min_max, 'mixed': mixed}
    statistics = {}
    data_statistics = {}

    for k,f in choices.items():
        start_i = time.time()
        # print('Starting ' + k + ' computation...')
        
        y_batch = np.array_split(np.array([y for theta, y in simulations]), Batch_size)
        thetas = np.array([theta for theta, y in simulations])
        results = []

        for y in y_batch:
            results.append(summary_statistics(y, f))

        stats = np.vstack([result for result in results])
        stats = np.hstack((thetas, stats))
        statistics[k] = stats
        data_statistics[k] = f(data)
        dur_i = time.time() - start_i
        # print(k + ' computation completed in: ' + str(dur_i) + '\n')

    dur = time.time() - start
    # print('Statistics generating time: ' + str(dur))

    return statistics, data_statistics