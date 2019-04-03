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
    knn_euclidians_users = None
    knn_euclidians_items = None
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

        self.knn_euclidians_users = np.array([[2, 3],
                                              [1, 2],
                                              [3, 4],
                                              [4, 3],
                                              [4, 3],
                                              [4, 3],
                                              [4, 3]])

        for i in range(0, n_iter):
            for n in range(0, N):
                self.users_latentProfile[n] = self._compute_A(n, alpha, k, "user") - 1
            for m in range(0, M):
                self.items_latentProfile[m] = self._compute_A(m, alpha, k, "items") - 1

    def _compute_A(self, i, alpha, k, with_respect_to="user"):

        if with_respect_to == "user":
            current_latents = self.users_latentProfile[i]
            knn_idxs = np.argwhere(self.knn_indexes_users == i)
            euclidians = self.knn_euclidians_users[knn_idxs] # euclidians where i is a neighbour of p
            latents = self.items_latentProfile[self._observed_ratings_users[i]]
        else:
            current_latents = self.items_latentProfile[i]
            knn_idxs = np.argwhere(self.knn_indexes_items == i)
            euclidians = self.knn_euclidians_items[knn_idxs] # euclidians where i is a neighbour of p
            latents = self.users_latentProfile[self._observed_ratings_items[i]]

        return np.sum(np.dot(latents, np.transpose(latents))) + alpha*current_latents*(1+np.divide(np.sum(euclidians)**2, k**2))

    def compute_B(self, i, alpha, k, with_respect_to="user"):

        if with_respect_to == "user":
            i_latents = self.users_latentProfile[i]
            other_latents = self.items_latentProfile[self._observed_ratings_users[i]]
            ratings = self._observed_ratings_users[i]
            

            IneP_idx = np.argwhere(self.knn_indexes_users == i)  # where i is a neighbour of p
            IneP_euclidians = self.knn_euclidians_users[IneP_idx]
            IneP_latents = self.users_latentProfile[IneP_idx] #TODO this probably wont worke since idx is 2-d
            
            PneI_idx = self.knn_indexes_users[i]  # where p is a neighbour of i
            PneI_euclidians = self.knn_euclidians_users[i]
            PneI_latents = self.users_latentProfile[PneI_idx]
        else:
            latents = self.users_latentProfile[self._observed_ratings_items[i]]
            ratings = self._observed_ratings_items[i]

        b_1 = np.sum(np.dot(other_latents, ratings))
        b_2 = alpha*np.divide(np.sum(PneI_euclidians*PneI_latents)+np.sum(IneP_euclidians*IneP_latents), k)
        b_3 = alpha*np.divide(np.sum(current_latents), k**2)




        if with_respect_to == "user":
            current_latents = self.users_latentProfile[i]
            knn_idxs = np.argwhere(self.knn_indexes_users == i)
            euclidians = self.knn_euclidians_users[knn_idxs] # euclidians where i is a neighbour of p
            knn_latents = self.users_latentProfile[knn_idxs]

            latents = self.items_latentProfile[self._observed_ratings_users[i]]
            ratings = self._observed_ratings_users[i]
        else:
            current_latents = self.items_latentProfile[i]
            knn_idxs = np.argwhere(self.knn_indexes_users == i)
            euclidians = self.knn_euclidians_items[knn_idxs] # euclidians where i is a neighbour of p
            knn_latents = self.items_latentProfile[knn_idxs]

            latents = self.users_latentProfile[self._observed_ratings_items[i]]
            ratings = self._observed_ratings_items[i]

        return b_1+b_2-b_3




    def _compute_S(self, i, with_respect_to):

        if with_respect_to == "user":
            idx = self.knn_indexes_users[i]
            knn_latents = self.users_latentProfile[idx]
            euclidians = self.knn_euclidians_users[i]
        else:
            idx = self.knn_indexes_items[i]
            knn_latents = self.items_latentProfile[idx]
            euclidians = self.knn_euclidians_items[i]

        return np.sum(euclidians*knn_latents)
