import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from constants import M_0, S_SQ_0, N, Batch_num, agents, chunk_size


# Generator for summary statistics

def batch_process(args):
    f, i = args
    batch_summaries = []
    batch = np.load('simulations/simulations_' + str(i) + '.npy', allow_pickle=True)

    for theta, y in batch:
        statistics = np.array(f(y))
        row = (theta, statistics)
        batch_summaries.append(row)

    return batch_summaries

def summary_statistics(statistic_choices):
    summaries = []
    batch_n = [(statistic_choices, n)  for n in np.arange(1, Batch_num + 1)]

    pool = Pool(processes=agents)
    results = pool.map(batch_process, batch_n, chunk_size)

    for result in results:
        summaries = summaries + result

    return np.array(summaries)


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



def statistic_generator_run():
    start = time.time()
    data = np.load('data.npy', allow_pickle = True)
    choices = {'mean_variance': mean_variance, 'quantiles': quantiles, 'min_max': min_max, 'mixed': mixed}

    for k,f in choices.items():
        start_i = time.time()
        print('Starting ' + k + ' computation...')
        stats = summary_statistics(f)
        np.save('statistics/'+ k + '.npy', stats, allow_pickle=True)
        np.save('statistics/data_' + k + '.npy', f(data), allow_pickle=True)
        dur_i = time.time() - start_i
        print(k + ' computation completed in: ' + str(dur_i) + '\n')

    dur = time.time() - start
    print('Statistics generating time: ' + str(dur))
