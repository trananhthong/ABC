import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from constants import M_0, S_SQ_0, N, Batch_num


def parameter_estimate(theta_statistics):
	thetas = np.array([theta for theta, statistics in theta_statistics])
	stats = np.array([statistics for theta, statistics in theta_statistics])

	reg = LinearRegression().fit(thetas, statistics)
	theta_hats = reg.predict(stats)

	return np.dstack((thetas, theta_hats))




if __name__ == "__main__":
    start = time.process_time()
    for i in range(1, Batch_num + 1):
        start_i = time.process_time()
        simulations = sampling(scaled_inversed_chi_square, normal, N, 10000)
        np.save('simulations/simulations_' + str(i) + '.npy', simulations, allow_pickle=True)
        dur_i = time.process_time() - start_i
        print('Batch ' + str(i) + ' completed in ' + str(dur_i))
    dur = time.process_time() - start
    print('Simulation time: ' + str(dur))