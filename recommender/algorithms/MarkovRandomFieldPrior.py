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


    def fit(self, observed_ratings, user_features, item_features, hyper_parameters, n_iter):
        alpha, d, k = hyper_parameters

        N = user_features.shape[0]
        M = item_features.shape[0]
        self.users_latentProfile = np.random.rand(N, d)
        self.items_latentProfile = np.random.rand(M, d)
        self._observed_ratings_users, self._observed_ratings_items, self._observed_ratings = observed_ratings

        print("item knn")
        _, self.knn_indexes_items, self.knn_euclidians_items = self.computeKNN(item_features.drop('movie_id', axis=1), True, k)
        print("user knn")
        _, self.knn_indexes_users, self.knn_euclidians_users = self.computeKNN(user_features, True, k)
        
        self.IneP_users = {i: np.argwhere(self.knn_indexes_users == i) for i in range(N)}
        self.IneP_items = {i: np.argwhere(self.knn_indexes_items == i) for i in range(M)}



        for i in range(0, n_iter):
            for n in range(0, N):
                self.users_latentProfile[n] = self._compute_A(n, alpha, k, "user") \
                                              - 1 * self._compute_B(n, alpha, k, "user")
            print("Iter: {} User latents updated".format(i))

            for m in range(0, M):
                self.items_latentProfile[m] = self._compute_A(m, alpha, k, "items") \
                                              - 1 * self._compute_B(m, alpha, k, "items")
            print("Iter: {} item latents updated".format(i))

    def computeKNN(self, features, categories, k):
        knn_computer = KNN(features, categories)
        return knn_computer.get_knn(k, n_jobs=1)


    def _compute_A(self, i, alpha, k, with_respect_to="user"):

        if with_respect_to == "user":
            current_latents = self.users_latentProfile[i]
            knn_idxs = self.IneP_users[i]
            euclidians = self.knn_euclidians_users[knn_idxs] # euclidians where i is a neighbour of p
            latents = self.items_latentProfile[self._observed_ratings_users[i]]
        else:
            current_latents = self.items_latentProfile[i]
            knn_idxs = self.IneP_items[i]
            euclidians = self.knn_euclidians_items[knn_idxs] # euclidians where i is a neighbour of p
            latents = self.users_latentProfile[self._observed_ratings_items[i]]

        res = np.sum(np.dot(latents, np.transpose(latents))) + alpha*current_latents*(1+np.divide(np.sum(euclidians)**2, k**2))

        return res

    def _compute_B(self, i, alpha, k, with_respect_to="user"):

        if with_respect_to == "user":
            this_latents = self.users_latentProfile[i]  # latents for user i
            other_latents = self.items_latentProfile[self._observed_ratings_users[i]]  #  latents for items rated by user i
            ratings = self._observed_ratings.loc[self._observed_ratings['user_id'] == i, 'rating']  # ratings

            IneP_idx = self.IneP_users[i]  # where i is a neighbour of p
            IneP_euclidians = self.knn_euclidians_users[IneP_idx].reshape(-1,1)  # euclid's where i is a neighbour of p
            IneP_latents = self.users_latentProfile[IneP_idx[0]] # latents where i is a neighbour of p
            
            PneI_idx = self.knn_indexes_users[i]  # where p is a neighbour of i
            PneI_euclidians = self.knn_euclidians_users[i].reshape(-1,1)  # euclid's where p is a neighbour of i
            PneI_latents = self.users_latentProfile[PneI_idx]  # latents where p is a neighbour of i
        else:
            this_latents = self.items_latentProfile[i]  # latents for user i
            other_latents = self.users_latentProfile[self._observed_ratings_items[i]]  # latents for items rated by user i
            ratings = self._observed_ratings.loc[self._observed_ratings['movie_id'] == i, 'rating']  # ratings

            IneP_idx = self.IneP_items[i]  # where i is a neighbour of p
            IneP_euclidians = self.knn_euclidians_items[IneP_idx].reshape(-1, 1)  # euclid's where i is a neighbour of p
            IneP_latents = self.items_latentProfile[IneP_idx[0]]  # latents where i is a neighbour of p

            PneI_idx = self.knn_indexes_items[i]  # where p is a neighbour of i
            PneI_euclidians = self.knn_euclidians_items[i].reshape(-1, 1)  # euclid's where p is a neighbour of i
            PneI_latents = self.items_latentProfile[PneI_idx]  # latents where p is a neighbour of i

        b_1 = np.sum(other_latents*ratings.values.reshape(-1, 1))
        b_2 = alpha*np.divide(np.sum(PneI_euclidians*PneI_latents)+np.sum(IneP_euclidians*IneP_latents), k)
        b_3 = alpha*np.divide(np.sum(this_latents), k**2)

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
