import numpy as np
from scipy.stats import norm, invgamma, wasserstein_distance
from scipy.spatial.distance import euclidean, seuclidean, mahalanobis
import matplotlib.pyplot as plt
import time
from constants import M_0, S_SQ_0, N



def generate_data(M_0, S_SQ_0, N):
    data = norm.rvs(M_0, S_SQ_0, N)
    np.savetxt('data.csv', data, delimiter=",")


if __name__ == "__main__":

    generate_data(M_0, S_SQ_0, N)