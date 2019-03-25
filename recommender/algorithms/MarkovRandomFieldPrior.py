import numpy as np
import pandas as pd




class MarkovRandomFieldPrior:
    """
    Implementation of the paper: N-dimensional Markov random filed prior for cold-start recommendation
    Can be found at https://www.sciencedirect.com/science/article/pii/S0925231216000825
    """

    users_latentProfile = None
    items_latentProfile = None
    _observed_ratings = None

    def MarkovRandomFieldPrior(self):
        print("This is amazing")


    def fit(self, observed_ratings: np.ndarray, user_features: np.ndarray, item_features: np.ndarray, hyper_parameters, n_iter):
        alpha, d, k = hyper_parameters

        N = user_features.shape[0]
        M = item_features.shape[0]
        self.users_latentProfile = np.random.rand(N, d)
        self.items_latentProfile = np.random.rand(M, d)
        self._observed_ratings = observed_ratings

        for i in range(0, n_iter):
            for n in range(0, N):
                print("turn down for what")
            for m in range(0, M):
                print("turn up for what")

    def _compute_U(self):

    def _compute_I(self):


    def compute_A(self, i, observed_ratings, latent_profile, knn_euclideans, alpha, k):
        item_idx = observed_ratings[i]
        latents = latent_profile[item_idx]

        np.sum(latents*np.transpose(latents))

        alpha(1+np.divide(np.sum(knn_euclideans[i])**2, k**2)) # TODO: somehow put I in between alpha and ()


    def _compute_S(self, i, knn_euclideans, knn_x_indexes, x_latent_profile):
        idx = knn_x_indexes[i]
        knn_us = x_latent_profile[idx]
        euclideans = knn_euclideans[i]

        return np.sum(knn_us*euclideans)

if __name__ == "__main__":


    print("Yay Done")
