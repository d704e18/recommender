from algorithms.MarkovRandomFieldPrior import MarkovRandomFieldPrior

import numpy as np
import pandas as pd
import config
config.set_settings('LOCAL')
from dataset.movielens import MovieLensDS


def transform_ratings(pure_ratings: pd.DataFrame):
    user_movies = {}

    movie_users = {}
    grouped_movies = pure_ratings.groupby('movie_id')
    movie_keys = pure_ratings['movie_id'].unique()

    grouped_users = pure_ratings.groupby('user_id')
    user_keys = pure_ratings['user_id'].unique()

    for key in user_keys:
        user_movies[key-1] = np.where(np.isin(movie_keys, grouped_users.get_group(key)['movie_id']))

    for i, key in enumerate(movie_keys):
        movie_users[i] = (np.array(grouped_movies.get_group(key)['user_id'])-1).tolist()

    print(pure_ratings.head())

    pure_ratings['user_id'] = pure_ratings['user_id'] - 1
    pure_ratings['movie_id'] = np.where(np.isin(movie_keys, pure_ratings['movie_id']))

    return user_movies, movie_users, pure_ratings.drop('timestamp', axis=1)


def one_hot_user(df):
    one_hot_gender = pd.get_dummies(df['gender'])
    one_hot_gender.drop('M', inplace=True, axis=1)
    one_hot_occupation = pd.get_dummies(df['occupation'])


    return pd.concat([one_hot_gender, one_hot_occupation], axis=1)

if __name__ == "__main__":
    generator = MovieLensDS()
    ratings = transform_ratings(generator.ratings)

    hyperParams = (2, 5, 2)

    generator.items.drop(generator.items.iloc[:, 33:], inplace=True, axis=1)
    generator.items.drop(generator.items.iloc[:, 1:14], inplace=True, axis=1)
    wat = generator.items.columns.values




    print(generator.users.head())
    print(generator.users.shape)
    print(generator.items.head())
    print(generator.items.shape)

    u_features = one_hot_user(generator.users)
    i_features = generator.items

    print(u_features.head())
    print(u_features.shape)

    mk = MarkovRandomFieldPrior()
    mk.fit(ratings, u_features, i_features, hyperParams, 10)

    print("Yay Done")
