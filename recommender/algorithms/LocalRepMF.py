import pandas as pd
import numpy as np
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

    def growTree(self): # alg. 1
        raise NotImplementedError()

    def optimizeTransformationMatrices(self): # eq. 9
        # use Maxvol to select local representatives (U2g)
        # Learn Tg
        raise NotImplementedError()

    def optimizeGlobalItemRepMatrix(self): # eq. 9
        #
        raise NotImplementedError()

def generate_candidate_items(ratings : pd.DataFrame, boundry : int):
    ratingsMatrix = to_matrix(ratings)
    colike_network = colike_itemitem_network(ratingsMatrix, boundry)

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



