import pandas as pd
import numpy as np


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