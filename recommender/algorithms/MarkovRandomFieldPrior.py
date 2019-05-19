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
    IneP_users = None
    IneP_items = None

    def MarkovRandomFieldPrior(self):
        print("This is amazing")


    def fit(self, observed_ratings, user_features, item_features, hyper_parameters, n_iter, knn_precomputed_path=None):
        alpha, d, k = hyper_parameters
        self._observed_ratings_users, self._observed_ratings_items, self._observed_ratings = observed_ratings


        N = user_features.shape[0]
        M = item_features.shape[0]
        M_rated_items = len(self._observed_ratings_items)
        self.users_latentProfile = np.random.rand(N, d)
        self.items_latentProfile = np.random.rand(M, d)

        if knn_precomputed_path is not None:
            print("Loading pre-computed knn")
            self.knn_indexes_items = np.load(knn_precomputed_path+"/knn_indexes_items.npy")
            self.knn_indexes_users = np.load(knn_precomputed_path+"/knn_indexes_users.npy")
            self.knn_euclidians_items = np.load(knn_precomputed_path+"/knn_euclidians_items.npy")
            self.knn_euclidians_users = np.load(knn_precomputed_path+"/knn_euclidians_users.npy")
        else:
            print("computing item knn")
            _, self.knn_indexes_items, self.knn_euclidians_items = self.computeKNN(item_features.drop('movie_id', axis=1), True, k)
            print("computing user knn")
            _, self.knn_indexes_users, self.knn_euclidians_users = self.computeKNN(user_features.drop(['zip_code', 'user_id'], axis=1), True, k)

        
        self.IneP_users = {i: np.where(self.knn_indexes_users == i) for i in range(N)}
        self.IneP_items = {i: np.where(self.knn_indexes_items == i) for i in range(M)}



        for i in range(0, n_iter):
            for n in range(0, N):
                a_user = self._compute_A(n, alpha, k, "user")
                b_user = 1 * self._compute_B(n, alpha, k, "user")
                self.users_latentProfile[n] = a_user - b_user
            print("Iter: {} User latents updated".format(i))

            for m in range(0, M_rated_items):
                a_item = self._compute_A(m, alpha, k, "items")
                b_item = 1 * self._compute_B(m, alpha, k, "items")
                self.items_latentProfile[m] = a_item - b_item
            print("Iter: {} item latents updated".format(i))

        return self.users_latentProfile, self.items_latentProfile

    def computeKNN(self, features, categories, k):
        knn_computer = KNN(features, categories)
        return knn_computer.get_knn(k, n_jobs=2)


    def _compute_A(self, i, alpha, k, with_respect_to="user"):

        if with_respect_to == "user":
            current_latents = self.users_latentProfile[i]
            n_IneP = self.IneP_users[i][0].size  # Amount of times i is a neighbour
            latents = self.items_latentProfile[self._observed_ratings_users[i]]

            knn_idxs = self.IneP_users[i]
            euclidians = self.knn_euclidians_users[knn_idxs]  # euclidians where i is a neighbour of p
        else:
            current_latents = self.items_latentProfile[i]
            n_IneP = self.IneP_items[i][0].size  # Amount of times i is a neighbour
            latents = self.users_latentProfile[self._observed_ratings_items[i]]

            knn_idxs = self.IneP_items[i]
            euclidians = self.knn_euclidians_items[knn_idxs]  # euclidians where i is a neighbour of p

        res = np.sum(np.dot(np.transpose(latents), latents), axis=0) + alpha*current_latents*(1+np.divide(n_IneP**2, k**2))

        return res

    def _compute_B(self, i, alpha, k, with_respect_to="user"):

        if with_respect_to == "user":
            this_latents = self.users_latentProfile[i]  # latents for user i
            other_latents = self.items_latentProfile[self._observed_ratings_users[i]]  #  latents for items rated by user i
            ratings = self._observed_ratings.loc[self._observed_ratings['user_id'] == i, 'rating']  # ratings

            IneP_idx = self.IneP_users[i]  # where i is a neighbour of p
            IneP_latents = self.users_latentProfile[IneP_idx[0]] # latents where i is a neighbour of p
            b3_in = np.array([np.sum(self.users_latentProfile[self.IneP_users[k][0]], axis=0) for k in IneP_idx[0]])
            
            PneI_idx = self.knn_indexes_users[i]  # where p is a neighbour of i
            PneI_latents = self.users_latentProfile[PneI_idx]  # latents where p is a neighbour of i
        else:
            this_latents = self.items_latentProfile[i]  # latents for user i
            other_latents = self.users_latentProfile[self._observed_ratings_items[i]]  # latents for items rated by user i
            ratings = self._observed_ratings.loc[self._observed_ratings['movie_id'] == i, 'rating']  # ratings

            IneP_idx = self.IneP_items[i]  # where i is a neighbour of p
            IneP_latents = self.items_latentProfile[IneP_idx[0]]  # latents where i is a neighbour of p
            b3_in = np.array([np.sum(self.items_latentProfile[self.IneP_items[k][0]], axis=0) for k in IneP_idx[0]])

            PneI_idx = self.knn_indexes_items[i]  # where p is a neighbour of i
            PneI_latents = self.items_latentProfile[PneI_idx]  # latents where p is a neighbour of i

        b_1 = np.sum(other_latents*ratings.values.reshape(-1, 1), axis=0)
        b_2 = alpha*np.divide(np.sum(PneI_latents, axis=0)+np.sum(IneP_latents, axis=0), k)
        b3_in = b3_in[b3_in.all(axis=1)]-this_latents if b3_in.size > 0 else [0]
        b_3 = alpha*np.divide(np.sum(b3_in, axis=0), k**2)

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
