import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from constants import M_0, S_SQ_0, N, Batch_num


def parameter_estimate(theta_statistics, data_statistics):
	thetas = np.array([theta for theta, statistics in theta_statistics])
	stats = np.array([statistics for theta, statistics in theta_statistics])

	reg = LinearRegression().fit(stats, thetas)
	# print(reg.score(stats, thetas))
	# print(reg.coef_)
	results = reg.predict(stats)
	# print(results[:10])
	sample_estimates = np.hstack((thetas, results))
	data_estimate = reg.predict([data_statistics])[0]

	return sample_estimates, data_estimate


def parameter_regression_run():
	start = time.process_time()

	statistics_sets = ['mean_variance', 'quantiles', 'min_max', 'mixed']

	for statistics_set in statistics_sets:
		start_i = time.process_time()
		simulations_statistics = np.load('statistics/' + statistics_set + '.npy', allow_pickle=True)
		data_statistics = np.load('statistics/data_' + statistics_set + '.npy', allow_pickle=True)
		sample_estimates, data_estimate = parameter_estimate(simulations_statistics, data_statistics)
		np.save('parameter_estimates/' + statistics_set + '_estimates.npy', sample_estimates, allow_pickle = True)
		np.save('parameter_estimates/data_' + statistics_set + '_estimate.npy', data_estimate, allow_pickle = True)
		dur_i = time.process_time() - start_i
		print(statistics_set + ' parameter estimation completed in: ' + str(dur_i))

	dur = time.process_time() - start
	print('Parameter estimation time: ' + str(dur))