import numpy as np
import random
import os
import sys
import pandas as pd
from dataset.movielens import MovieLensDS
from dataset.data_utils import transform_ratings

from algorithms.mostPopular2 import MostPopular2




class Evaluator:
    
    def __init__(self, dataset):
        self.user_movies, self.user_ratings, self.movie_user, self.movie_ratings, self.ratings\
            = transform_ratings(dataset.ratings, dataset.items)


    def evaluate_cold_start_user(self, algos, k_folds):

        folder_user = self._createKFolder(self.ratings, k_folds)

        #  Evaluate user cold start
        for i in range(0, k_folds):
            test, train = folder_user(self.ratings, i)

            for name, algo in algos.items:
                algo.fit(train)
                preds = algo.predict_n(10)
                self.eval_ndcg(preds, )


    def _createKFolder(self, ratings, folds, user=True):

        if user:
            ids = np.array(ratings['user_id'].unique())
            n = ids.shape[0]/folds
            np.random.shuffle(ids)

            test_ids = []
            for k in range(0, folds):
                test_ids = ids[n*k:n*(k+1)]
                test_ids.append(test_ids)
        else:
            ids = ratings['movie_id'].unique()
            n = ids.shape[0]/folds
            np.random.shuffle(ids)

            test_ids = []
            for k in range(0, folds):
                test_ids = ids[n*k:n*(k+1)]
                test_ids.append(test_ids)


        def get_Fold(ratings, n):
            if n < 0 or n > folds:
                raise ValueError("This function was created with folds {}-{}, but was requested fold {}."
                                 "Please provide an eligible n".format(0, folds-1, n))

            fold_ids = test_ids[n]

            if user:
                idx_nonzero = np.nonzero(np.isin(ratings['user_id'], fold_ids))
            else:
                idx_nonzero = np.nonzero(np.isin(ratings['movie_id'], fold_ids))

            test = ratings.loc[idx_nonzero]
            train = ratings.loc[idx_nonzero == 0]

            return test, train

        return get_Fold



    def eval_ndcg(self, prediction_ids, test_data):
        return "wat"



    def _dcg(self, pred_ids, actual_ids):

        act_mask = np.isin(actual_ids, pred_ids) # 0'es or 1'es based on whether or not prediction ids is in actual ids

        gain = 2**act_mask-1
        discounts = np.log2(np.arange(len(pred_ids))+2)

        return np.sum(gain/discounts)

    def ndcg(self, prediction_ids, target_ids):
        return self._dcg(prediction_ids, target_ids)/self._dcg(target_ids, target_ids)


    def relevance(self, k):
        print("aw")
