import numpy as np
import pandas as pd

from .common import Recommender


class MostPopular2(Recommender):

    Most_Popular = None

    def __init__(self, dataset, train_size=0.8, alpha=0):
        Recommender.__init__(self, dataset, train_size)
        self.alpha = alpha

    def fit(self, ratings):

        r_hat = np.sum(ratings['rating'])/ratings.shape[0]  # Average rating

        # Construct movie rated by user collection
        grouped_movies = ratings.groupby('movie_id')
        movie_users = {}
        movie_ids = ratings['movie_id'].unique()

        for m_id in movie_ids:
            movie_users[m_id] = list(grouped_movies.get_group(m_id)['user_id'])

        mp_movies = {}
        for key, movie_n in movie_users.items():
            n_i = len(movie_n)
            ratings_movie_n = ratings.loc[ratings['movie_id'] == key, 'rating']
            r_i_hat = 1/n_i*np.sum(ratings_movie_n)
            mp_movies[key] = (r_i_hat * n_i - r_hat * self.alpha) / (n_i + self.alpha)

        res = pd.DataFrame.from_dict(mp_movies, orient='index', columns=['mp_rating'])
        res = res.sort_values(by='mp_rating', ascending=False)
        self.Most_Popular = res
        return res

    def top_n(self, n, user=None):
        if self.Most_Popular is None:
            raise EnvironmentError("A model has not yet been fitted, please call fit() with eligible parameters")

        return self.Most_Popular[:n].index.values

