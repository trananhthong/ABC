import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import time
from sklearn.linear_model import LinearRegression
from constants import M_0, S_SQ_0, N


def parameter_estimate(simulated_statistics, data_statistics):
	reg = LinearRegression().fit(stats, thetas)
	results = reg.predict(stats)
	data_estimate = reg.predict([data_statistics])[0]

	return sample_estimates, data_estimate


def parameter_regression_run(stats, data_stats):
	start = time.time()

	simulation_theta_hat = {}
	data_theta_hat = {}

	statistics_sets = ['mean_variance', 'quantiles', 'min_max', 'mixed']

	for statistics_set in statistics_sets:
		start_i = time.time()

		statistics = stats[statistics_set]
		thetas = np.array([row[:2] for row in statistics])
		simulated_statistics = np.array([row[2:] for row in statistics])
		data_statistics = data_stats[statistics_set]

		reg = LinearRegression().fit(simulated_statistics, thetas)
		sample_estimates = reg.predict(simulated_statistics)
		sample_estimates = np.hstack((thetas, sample_estimates))
		simulation_theta_hat[statistics_set] = sample_estimates
		data_theta_hat[statistics_set] = reg.predict([data_statistics])[0]

		dur_i = time.time() - start_i
		print(statistics_set + ' parameter estimation completed in: ' + str(dur_i))

	dur = time.time() - start
	print('Parameter estimation time: ' + str(dur))

	return simulation_theta_hat, data_theta_hat