import datetime
import os

import numpy as np

import recommender
from recommender.algorithms.common import Recommender


class MatrixFactorization(Recommender):  # TODO: it needs to be stochastic, or at least kinda
    _P = None
    _Q = None

    def __init__(self, dataset, train_size=0.8, n_latent_factors=3, steps=1000, alpha=0.0002, beta=0.02):
        Recommender.__init__(self, dataset, train_size)
        self.n_latent_factors = n_latent_factors
        self.steps = steps
        self.alpha = alpha
        self.beta = beta

    def fit(self, ratings=None, n_latent_factors=None, steps=None, alpha=None, beta=None):
        if ratings is None:
            ratings = self.dataset.ratings
        if n_latent_factors is None:
            n_latent_factors = self.n_latent_factors
        if steps is None:
            steps = self.steps
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        self.n_latent_factors = n_latent_factors
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        n_users = ratings['user_id'].unique().max()
        n_items = ratings['movie_id'].unique().max()
        P = np.random.rand(n_users, n_latent_factors)
        Q = np.random.rand(n_latent_factors, n_items)

        print('Started fitting of model at time: {0}...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        for step in range(0, steps):
            reg_error = 0
            breaker = []
            for row in ratings.itertuples(index=False, name='row'):  # ['user_id', 'movie_id', 'rating', 'timestamp']
                user = row.user_id - 1
                item = row.movie_id - 1
                rating = row.rating

                p_u = P[user]
                q_i = Q[:, item]
                error = rating - np.dot(p_u, q_i)
                P[user] = p_u + alpha * (2 * error * q_i - beta * p_u)
                Q[:, item] = q_i + alpha * (2 * error * p_u - beta * q_i)

                if step % 100 == 0:
                    reg_error += \
                        np.square(error) + \
                        beta * (np.square(np.linalg.norm(p_u) + np.square(np.linalg.norm(q_i))))
                breaker.append(error)

            if step % 100 == 0:
                print('done with step: {0} at time {1}. Error: {2}'.format(step, datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'), reg_error))
            else:
                print('done with step: {0} at time {1}'.format(step, datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')))
            if np.mean(breaker) < 0.001:
                print('BREAK')
                break
        print('Done fitting model...')
        self._P = P
        self._Q = Q
        return P, Q

    def save_model(self, dataset):
        dir_name = (f'mfnonvec_{dataset}_ts{self.train_size}_'
                    f'nlf{self.n_latent_factors}_s{self.steps}_a{self.alpha}_b{self.beta}')
        os.makedirs(os.path.join(recommender.BASE_DIR, dir_name))
        self._P.tofile(os.path.join(recommender.BASE_DIR, dir_name, 'mf_P.bin'))
        self._Q.tofile(os.path.join(recommender.BASE_DIR, dir_name, 'mf_Q.bin'))

    def load_model(self, path):
        self._P = np.fromfile(os.path.join(path, 'mf_P.bin'))
        self._Q = np.fromfile(os.path.join(path, 'mf_Q.bin'))

    def _compute_prediction(self, P, Q):
        return np.dot(P, Q)

    def _compute_error(self, ratings, pred):
        return ratings-pred

    def top_n(self, n, user=None):
        rankings = self._P[user-1] * self._Q
        recommendations = sorted(range(len(rankings)), key=lambda k: rankings[k])
        recommendations = [(x + 1, None) for x in recommendations]
        return recommendations[:n]
