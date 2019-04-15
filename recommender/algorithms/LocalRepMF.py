import pandas as pd
import numpy as np
import sys
from .DivRank import DivRank as dr

class LocalRepMF(object):
    """
    Implementation of the paper: Local Representative-Based Matrix Factorization for Cold-Start Recommendation
    Url: https://dl.acm.org/citation.cfm?id=3108148
    """

    def fit(self,
            ratings : pd.DataFrame,
            l1 : int, # number of global questions
            l2 : int, # number of local questions
            alpha = 0.05, # regularisation param
            beta = 0.05, # regularisation param
            epochs = 1000,
            boundry = 4, # decides when a user likes or dislikes (rating >= boundry)
            k_prime = 20):
        # brug lokale variable til at holde state, i.e. V, U1, U2 osv.

        # randomize V
        # generate candidate item set Icand subset af I
        # start for loop:
            # Brug algoritme 1 til at lave G
            # for hver g i G lær U2g og Tg ved at optimere eq. 9
            # optimer V ved at bruge eq. 7
        raise NotImplementedError()

def growTree(ratings : pd.DataFrame,
             V : np.ndarray,
             users : list,
             cand_items : list,
             split_value : int,
             depth : int,
             max_depth : int,
             user_groups = []): # alg. 1
    best_item = None
    best_loss = sys.maxsize
    for item in cand_items:
        # evaluate eq. 11
        ratings_on_item = ratings.where([ratings[1] == item])
        users_like = ratings.where([ratings_on_item[2] >= split_value])[0]
        users_dislike = ratings.where([ratings_on_item[2] < split_value])[0]

        #evaluate_eq11(users_like)


        loss = 0 # TODO: fix
        if loss > best_loss:
            best_item = item

    ratings_on_item = ratings.where([ratings[1] == best_item])
    users_like = ratings.where([ratings_on_item[2] >= split_value])[0]
    users_dislike = ratings.where([ratings_on_item[2] < split_value])[0]

    # Not a leaf node
    if users_like != None and users_dislike != None:
        if depth < max_depth: # TODO: fix max_depth
            cand_items.remove(best_item) # I cand - i*
            growTree(users_like, cand_items, split_value, depth + 1, max_depth, user_groups)
            growTree(users_dislike, cand_items, split_value, depth + 1, max_depth, user_groups)
            cand_items.append(best_item) # I cand + i*

    # Leaf node
    return user_groups.append(users)

def evaluate_eq11(R : np.ndarray,
                  B : np.ndarray,
                  T : np.ndarray,
                  V : np.ndarray,
                  alpha : int):
    # using frobenius norm
    return np.square(np.linalg.norm(R - np.matmul(np.matmul(B, T), V))) + alpha * np.square(np.linalg.norm(T)) # eq. 11



def optimizeTransformationMatrices(self): # eq. 9
    # use Maxvol to select local representatives (U2g)
    # Learn Tg
    raise NotImplementedError()

def optimizeGlobalItemRepMatrix(self): # eq. 9
    #
    raise NotImplementedError()

def generate_candidate_items(ratings : pd.DataFrame, boundry : int):
    ratingsMatrix = to_matrix(ratings)
    # constructing colike network
    colike_network = colike_itemitem_network(ratingsMatrix, boundry)

    # employing DivRank on the network
    return dr.rank(colike_network)

def colike_itemitem_network(ratingsMatrix : np.ndarray, boundry : int):
    _, n_items = ratingsMatrix.shape
    matrix = np.zeros(shape=(n_items, n_items))

    for item1 in range(0, n_items):
        for item2 in range(0, n_items):
            value = 0
            if item1 != item2:
                value = len(ratingsMatrix[np.where((ratingsMatrix[:,item1] >= boundry) * (ratingsMatrix[:,item2] >= boundry))])

            matrix[item1, item2] = value

    return normalize_matrix(matrix)

def normalize_matrix(matrix : np.ndarray):
    rows, cols = matrix.shape
    result = np.zeros(shape=(rows, cols))

    for row in range(0, rows):
        row_sum = sum(matrix[row])
        for col in range(0, cols):
            if row_sum > 0:
                result[row, col] = matrix[row, col] / row_sum

    return result

def to_matrix(ratings : pd.DataFrame):
    n_users = ratings[0].max()
    n_items = ratings[1].max()

    matrix = np.zeros(shape=(n_users, n_items))

    for row in ratings.itertuples(index=False):
        user = row[0] - 1
        item = row[1] - 1
        rating = row[2]

        matrix[user, item] = rating

    return matrix



