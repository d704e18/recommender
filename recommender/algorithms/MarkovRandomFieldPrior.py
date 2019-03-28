import numpy as np
import pandas as pd
from algorithms.MRFKNN import KNN


class MarkovRandomFieldPrior:
    """
    Implementation of the paper: N-dimensional Markov random filed prior for cold-start recommendation
    Can be found at https://www.sciencedirect.com/science/article/pii/S0925231216000825
    """

    users_latentProfile = None
    items_latentProfile = None
    knn_euclideans_users = None
    knn_euclideans_items = None
    knn_indexes_users = None
    knn_indexes_items = None
    _observed_ratings_users = None
    _observed_ratings_items = None
    _observed_ratings = None

    def MarkovRandomFieldPrior(self):
        print("This is amazing")


    def fit(self, observed_ratings, user_features: np.ndarray, item_features: np.ndarray, hyper_parameters, n_iter):
        alpha, d, k = hyper_parameters

        N = user_features.shape[0]
        M = item_features.shape[0]
        self.users_latentProfile = np.random.rand(N, d)
        self.items_latentProfile = np.random.rand(M, d)
        self._observed_ratings = observed_ratings

        self.knn_euclideans_users = np.array([[2, 3],
                                              [1, 2],
                                              [3, 4],
                                              [4, 3],
                                              [4, 3],
                                              [4, 3],
                                              [4, 3]])

        for i in range(0, n_iter):
            for n in range(0, N):
                self.users_latentProfile[n] = self.compute_A(n, alpha, k, "user") - 1
            for m in range(0, M):
                self.items_latentProfile[m] = self.compute_A(m, alpha, k, "items") - 1

    def compute_A(self, i, alpha, k, with_respect_to="user"):

        if with_respect_to == "user":
            current_latents = self.users_latentProfile[i]
            euclideans = self.knn_euclideans_users[i]
            latents = self.items_latentProfile[self._observed_ratings_users[i]]
        else:
            current_latents = self.items_latentProfile[i]
            euclideans = self.knn_euclideans_items[i]
            latents = self.users_latentProfile[self._observed_ratings_items[i]]

        return np.sum(np.dot(latents, np.transpose(latents))) + alpha*current_latents*(1+np.divide(np.sum(euclideans)**2, k**2))

    # def compute_B(self):


    def _compute_S(self, i, with_respect_to):

        if with_respect_to == "user":
            idx = self.knn_indexes_users[i]
            knn_latents = self.users_latentProfile[idx]
            euclideans = self.knn_euclideans_users[i]
        else:
            idx = self.knn_indexes_items[i]
            knn_latents = self.items_latentProfile[idx]
            euclideans = self.knn_euclideans_items[i]

        return np.sum(knn_latents*euclideans)
