import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from dataset.data_utils import transform_ratings


class Recommender(object):
    def __init__(self, dataset, train_size=0.8):
        self.dataset = dataset
        self.train_size = train_size
        self.user_movies, self.movie_user, self.transRatings = transform_ratings(dataset.ratings, dataset.items, dataset.users)

    def tfidf_cosine_similarities(self, df, columns=('title', 'original_title', 'summary')):
        """Compute the similarity between items by columns containing text."""
        texts = df[list(columns)].apply(lambda x: ' '.join(x), axis=1)
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        tfidf = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf, dense_output=True)

    def euclidean_distances_similarities(self, df, columns=('runtime',)):
        """Compute the similarity between items by specified columns containing continuous data."""
        features = df[list(columns)]
        features = features.fillna(0)
        distance_matrix = pairwise_distances(features)
        normalized_dist_matrix = normalize(distance_matrix, norm='max')
        return 1 - normalized_dist_matrix

    def jaccard_similarities(self, df, columns=(), n_jobs=8):
        features = df
        if columns:
            features = features[list(columns)]
        jaccard_sim_matrix = pairwise_distances(features.to_numpy(), metric='jaccard', n_jobs=n_jobs)
        return 1 - jaccard_sim_matrix

    def combine_similarity_matrices(self, sim_matrices, weights=None):
        """Combine similarity matrices into one similarity matrix."""
        return np.average(sim_matrices, axis=0, weights=weights)

    def top_n(self, n, user=None):
        return NotImplementedError('This method must be implemented in subclasses')

    def user_average_ratings(self, dataframe):
        user_average_ratings = pd.DataFrame(data=None, columns=['user_id', 'rating'])

        for id in dataframe.user_id.unique():
            elements = dataframe[dataframe.user_id == id].rating
            size = elements.shape[0]
            ratings_total = elements.values.sum()
            user_average_ratings = user_average_ratings.append(
                pd.Series([id, (ratings_total / size)], index=user_average_ratings.columns), ignore_index=True)

        return user_average_ratings

    def mae(self, rating_df=None):
        ratings = self.dataset.df_test
        ae = 0
        for rating in ratings.itertuples():
            ae += abs((rating.rating) - self.get_prediction(rating.user_id, rating.movie_id))
        return ae/ratings.shape[0]

    def rmse(self, rating_df=None):
        ratings = self.dataset.df_test
        se = 0
        for rating in ratings.itertuples():
            se += (rating.rating - self.get_prediction(rating.user_id, rating.movie_id))**2

        return math.sqrt(se/ratings.shape[0])

    def mean_average_precision(self, rating_df=None):
        if rating_df is None:
            rating_df = self.dataset.ratings
        list_of_ap = []
        average_rating_df = self.user_average_ratings(rating_df)
        for user in rating_df.user_id.unique():
            average_rating = average_rating_df.rating.loc[average_rating_df['user_id'] == user].iloc[0]

            good_recommendations = rating_df.loc[rating_df.user_id == user]
            good_recommendations = good_recommendations[good_recommendations.rating >= average_rating].movie_id

            #top_n_recommendations = self.top_n(n=10, user=user)

            top_n_recommendations = [x[0] for x in self.top_n(n=10, user=user)]

            print(top_n_recommendations)

            total_recs = 0
            correct_recs = 0
            scorelist = []
            for rec in top_n_recommendations:
                total_recs += 1
                if rec in good_recommendations:
                    scorelist.append(1/total_recs)
                    correct_recs += 1

            if correct_recs != 0:
                score = sum(scorelist)/correct_recs
            else:
                score = 0

            list_of_ap.append(score)

        return np.mean(list_of_ap)

    def normalized_dcg_at_k(self, k=10):
        def discounted_cumulative_gain(scores):
            return scores[0] + sum(sc / math.log2(ind) for sc, ind in zip(scores[1:], range(2, len(scores)+1)))
        ratings = self.dataset.ratings
        ndcgs = []
        for user in ratings.user_id.unique():
            top_n_predictions = [x[0] for x in self.top_n(n=k, user=user)]
            users_ratings = ratings.loc[ratings.user_id == user]
            top_n_ratings = users_ratings.nlargest(k, 'rating').rating.tolist()
            relevances = [0.] * k
            for i, x in enumerate(top_n_predictions):
                relevances[i] = users_ratings.loc[x == users_ratings.movie_id].iloc[0]['rating']\
                    if x in users_ratings.movie_id.unique() else 0
            idcg = discounted_cumulative_gain(top_n_ratings)
            ndcgs.append(discounted_cumulative_gain(relevances) / idcg)
        return np.mean(ndcgs)

    def dcg(self, pred_ids, actual_ids):

        act_mask = np.isin(actual_ids, pred_ids)

        gain = 2**act_mask-1
        discounts = np.log2(np.arange(len(pred_ids))+2)

        return np.sum(gain/discounts)

    def ndcg(self, k=10):
        



    def save_model(self, dataset):
        return NotImplementedError('Implement in subclasses.')

    def load_model(self, path):
        return NotImplementedError('Implement in subclasses.')
