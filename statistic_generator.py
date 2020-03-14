import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N, Batch_num

# Linear Regression for summary statistics

def summary_statistics(statistic_choices):
    summaries = np.array([])

    for i in range(1, Batch_num + 1):
        start_i = time.process_time()
        batch = np.load('simulations_' + str(i) + '.npy', allow_pickle=True)
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

    return q


def min_max(data):
    return np.array([np.min(data), np.max(data)])

