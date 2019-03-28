from algorithms.MarkovRandomFieldPrior import MarkovRandomFieldPrior

import numpy as np
import pandas as pd
import config
config.set_settings('LOCAL')
from dataset.movielens import MovieLensDS


def transform_ratings(pure_ratings: pd.DataFrame):
    user_movies = {}

    grouped_users = pure_ratings.groupby('user_id')
    user_keys = pure_ratings['user_id'].unique()
    for key in user_keys:
        user_movies[key] = (list(grouped_users.get_group(key)['movie_id']))

    movie_users = {}
    grouped_movies = pure_ratings.groupby('movie_id')
    movie_keys = pure_ratings['movie_id'].unique()
    for key in movie_keys:
        movie_users[key] = (list(grouped_movies.get_group(key)['user_id']))

    print(pure_ratings.head())

    return user_movies, movie_users, pure_ratings.drop('timestamp', axis=1)


if __name__ == "__main__":
    generator = MovieLensDS()
    user_ratings = transform_ratings(generator.ratings)

    hyperParams = (2, 5, 2)

    u_features = np.array([[5, 1],
                           [2, 3],
                           [4, 0],
                           [4, 1]])
    i_features = np.array([[8, 2],
                           [2, 2],
                           [4, 4],
                           [4, 0]])

    mk = MarkovRandomFieldPrior()
    # mk.fit(observed_ratings, u_features, i_features, hyperParams, 10)

    print("Yay Done")
