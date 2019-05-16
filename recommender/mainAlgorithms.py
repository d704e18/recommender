from algorithms.MarkovRandomFieldPrior import MarkovRandomFieldPrior

import numpy as np
import pandas as pd
import os
import config
from time import time
config.set_settings('LOCAL')
from dataset.movielens import MovieLensDS


def transform_ratings(pure_ratings: pd.DataFrame, item_features, user_features):

    # First, drop items not in both sets
    pure_ratings = pure_ratings[pure_ratings['movie_id'].isin(item_features['movie_id'])]
    item_features = item_features[item_features['movie_id'].isin(pure_ratings['movie_id'])]

    # Also 0 index user_id
    pure_ratings['user_id'] = pure_ratings['user_id']-1

    # Convert to 0 indexed items
    item_id_conversion = item_features['movie_id']
    pure_ratings['movie_id'].replace(list(item_id_conversion), list(np.arange(len(item_id_conversion))), inplace=True)

    # Construct movie rated by user collection and user rated movie collection
    grouped_movies = pure_ratings.groupby('movie_id')
    grouped_users = pure_ratings.groupby('user_id')


    user_movies = {}
    for key in pure_ratings['user_id'].unique():
        user_movies[key] = list(grouped_users.get_group(key)['movie_id'])


    movie_users = {}
    for key in pure_ratings['movie_id'].unique():
        movie_users[key] = list(grouped_movies.get_group(key)['user_id'])

    print(pure_ratings.head())


    return user_movies, movie_users, pure_ratings.drop('timestamp', axis=1)


def one_hot_user(df):
    one_hot_gender = pd.get_dummies(df['gender'])
    one_hot_gender.drop('M', inplace=True, axis=1)
    one_hot_occupation = pd.get_dummies(df['occupation'])


    return pd.concat([one_hot_gender, one_hot_occupation], axis=1)

if __name__ == "__main__":
    start = time()
    generator = MovieLensDS()
    end = time()

    print("Time: {}".format(end-start))


    hyperParams = (2, 5, 2)

    generator.items.drop(generator.items.iloc[:, 33:], inplace=True, axis=1)
    generator.items.drop(generator.items.iloc[:, 1:14], inplace=True, axis=1)
    generator.users.drop('gender_M', inplace=True, axis=1)

    ratings = transform_ratings(generator.ratings, generator.items, generator.users)
    print(generator.users.head())
    print(generator.users.shape)
    print(generator.items.head())
    print(generator.items.shape)


    item_features = generator.items
    user_features = generator.users

    mk = MarkovRandomFieldPrior()
    mk.fit(ratings, user_features, item_features, hyperParams, 10, os.getcwd()+"/data")

    print("Yay Done")
