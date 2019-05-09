import datetime
import os
import pickle

import pandas as pd
import numpy as np
import sys
import scipy.linalg
from .DivRank import DivRank as dr

class LocalRepMF(object):
    """
    Implementation of the paper: Local Representative-Based Matrix Factorization for Cold-Start Recommendation
    Url: https://dl.acm.org/citation.cfm?id=3108148
    """

    def fit(self,
            ratingsMatrix : np.ndarray,
            number_of_cand_items : int,
            l1 : int, # number of global questions
            l2 : int, # number of local questions
            alpha = 0.01, # regularisation param
            beta = 0.01, # regularisation param
            epochs = 1000,
            boundry = 4, # decides when a user likes or dislikes (rating >= boundry)
            k_prime = 20) :

        n_users, n_items = ratingsMatrix.shape

        # randomize V
        print(f"Randomizing V with shape {k_prime} x {n_items}")
        V = np.random.rand(k_prime, n_items)

        # Generate candidate item set
        print("Generating candidate item set")
        cand_items = [item for item in
                      generate_candidate_set(ratingsMatrix).keys()][:number_of_cand_items]

        # Generating users
        users = []
        for user in range(0, n_users):
            users.append(user)

        for step in range (0, epochs):
            print(f'Starting iteration: {step}')
            print(f'Time: {datetime.datetime.now()}')
            # Generate global questions
            groups, global_questions = growTree(
                ratingsMatrix=ratingsMatrix,
                V=V,
                users=users,
                cand_items=cand_items,
                split_value=4,
                depth=0,
                max_depth=l1,
                alpha=alpha
            )
            print(f'Done with step 1')
            print(f'Time: {datetime.datetime.now()}')

            # Generate local questions and learn
            # transformation matrices for each group
            local_questions, list_of_Ts = optimizeTransformationMatrices(
                ratingsMatrix=ratingsMatrix,
                V=V,
                groups=groups,
                global_questions=global_questions,
                local_questions_count=l2,
                alpha=0.01)
            print(f'Done with step 2')
            print(f'Time: {datetime.datetime.now()}')

            # Optimize V with equation 7
            V = optimizeGlobalItemRepMatrix(
                ratingsMatrix=ratingsMatrix,
                groups=groups,
                global_questions=global_questions,
                local_questions=local_questions,
                list_of_transformation_matrices=list_of_Ts,
                n_users=n_users,
                beta = beta
            )
            print(f'Done with step 3 and thus iteration: {step}')
            print(f'Time: {datetime.datetime.now()}')

        return global_questions, local_questions, list_of_Ts, V

def growTree(ratingsMatrix : np.ndarray,
             V : np.ndarray,
             users : list,
             cand_items : list,  # assumes sorted based on divrank
             split_value : int,
             depth : int,
             max_depth : int,  # l1 questions in paper
             alpha = 0.01,
             already_asked_items: list = [], # items used as global questions
             groups = [],
             questions = []):

    # leaf node - i.e. do not ask more questions
    if depth >= max_depth:
        print(f"forming group: {users}")
        groups.append(users)
        questions.append(already_asked_items)

    best_item = None
    best_loss = sys.maxsize

    # todo: maybe change way which global questions are asked
    # could be maxvol or random approach

    # evaluate the best item from candidate set
    for item in cand_items:
        # split users into like and dislike
        likes = []
        dislikes = []
        for user in users:
            if ratingsMatrix[user, item] >= split_value:
                likes.append(user)
            else:
                dislikes.append(user)

        # if all users likes or dislikes the item, select a new question
        # i.e. no knowledge on that item
        if not likes or not dislikes:
            continue

        # generating ratings matrices for users who liked and disliked item
        ratingsMatrix_like = ratingsMatrix[likes]
        ratingsMatrix_dislike =  ratingsMatrix[dislikes]

        # generating B = [U1, e]
        u1_like = ratingsMatrix[likes,:][:, already_asked_items]
        e_users_like = np.ones(shape=(len(likes), 1))
        B_like = np.hstack((u1_like, e_users_like))

        u1_dislike = ratingsMatrix[dislikes,:][:, already_asked_items]
        e_users_dislike = np.ones(shape=(len(dislikes), 1))
        B_dislike = np.hstack((u1_dislike, e_users_dislike))

        # generating T by solving sylvester equation (XT + TY = Z)
        X_like = B_like.T @ B_like
        Y_like = alpha * np.linalg.inv(V @ V.T)
        Z_like = B_like.T @ ratingsMatrix_like @ V.T @ np.linalg.inv(V @ V.T)
        T_like = scipy.linalg.solve_sylvester(X_like, Y_like, Z_like)

        X_dislike = B_dislike.T @ B_dislike
        Y_dislike = alpha * np.linalg.inv(V @ V.T)
        Z_dislike = B_dislike.T @ ratingsMatrix_dislike @ V.T @ np.linalg.inv(V @ V.T)
        T_dislike = scipy.linalg.solve_sylvester(X_dislike, Y_dislike, Z_dislike)

        # Computing loss (||(rating - prediction)||^2 + alpha * ||T||^2
        loss_like = evaluate_eq11(ratingsMatrix_like, B_like, T_like, V, alpha)
        loss_dislike = evaluate_eq11(ratingsMatrix_dislike, B_dislike, T_dislike, V, alpha)
        loss = loss_like + loss_dislike

        if loss < best_loss:
            best_item = item
            users_like = likes
            users_dislike = dislikes

    # add the best item to asked questions
    global_questions = already_asked_items.copy()
    global_questions.append(best_item)

    # todo: something with total loss decreases

    # Not a leaf node
    if depth < max_depth:
        cand_items.remove(best_item) # I cand - i*

        # likes
        groups_likes, items_likes = growTree(
            ratingsMatrix = ratingsMatrix,
            V = V,
            users = users_like,
            cand_items = cand_items,
            split_value = split_value,
            depth = depth + 1,
            max_depth = max_depth,
            already_asked_items=global_questions)

        # dislikes
        groups_dislikes, items_dislikes = growTree(
            ratingsMatrix = ratingsMatrix,
            V=V,
            users=users_dislike,
            cand_items=cand_items,
            split_value=split_value,
            depth=depth + 1,
            max_depth=max_depth,
            already_asked_items=global_questions)

        cand_items.append(best_item) # I cand + i*

    return groups, questions

def evaluate_eq11(R : np.ndarray,
                  B : np.ndarray,
                  T : np.ndarray,
                  V : np.ndarray,
                  alpha : int):
    # using frobenius norm
    return np.linalg.norm(R - (B @ T @ V)) + alpha * np.linalg.norm(T) # eq. 11


def optimizeTransformationMatrices(
        ratingsMatrix : np.ndarray,
        V : np.ndarray,
        groups : list,
        global_questions : list,
        local_questions_count : int,
        alpha : int = 0.01): # eq. 9

    list_of_Ts = []
    list_of_local_questions = []

    for group, questions in zip(groups, global_questions):  # group is a list of userIds
        count_of_ratings = (ratingsMatrix[group] > 0).sum(axis=0)
        # sort number of ratings descending and take the first k items
        top_k_rated_items = np.argsort(-count_of_ratings)[:local_questions_count]

        # TODO: select items with maxvol instead
        # Generate ratings matrix for users in group on the selected items
        U2g = ratingsMatrix[group,:][:,top_k_rated_items]

        # generating Bg = [U1g, U2g, e]
        U1g = ratingsMatrix[group,:][:,questions]
        Bg = np.hstack((U1g, U2g, np.ones(shape=(len(group), 1))))

        # Calculate Tg with sylvester equation (XT + TY = Z)
        X = Bg.T @ Bg
        Y = alpha * np.linalg.inv(V @ V.T)
        Rg = ratingsMatrix[group]
        Z = Bg.T @ Rg @ V.T @ np.linalg.inv(V @ V.T)

        Tg = scipy.linalg.solve_sylvester(X, Y, Z)

        # add Tg to list of Ts
        list_of_Ts.append(Tg)
        # add local questions to list
        list_of_local_questions.append(top_k_rated_items)

    return list_of_local_questions, list_of_Ts

def optimizeGlobalItemRepMatrix(
        ratingsMatrix : np.ndarray,
        groups : list,
        global_questions : list,
        local_questions : list,
        list_of_transformation_matrices : list,
        n_users : int,
        beta : int = 0.01
    ):
    _, k_prime = list_of_transformation_matrices[0].shape
    S = np.zeros(shape=(n_users, k_prime))
    # Fill up S with S_i = [a1, a2, 1] @ T_g
    for userId in range(0, n_users):
        group = belongs_to_group(userId, groups)
        global_questions_for_user = global_questions[group]
        local_questions_for_user = local_questions[group]
        T_g = list_of_transformation_matrices[group]

        # answers = [a1, a2, 1]
        userRatings = ratingsMatrix[userId]
        a1 = userRatings[global_questions_for_user]
        a2 = userRatings[local_questions_for_user]
        answers = np.hstack((a1, a2, 1))

        S[userId] = answers @ T_g

    # Identity matrix of size k' x k'
    I = np.identity(k_prime)

    V = np.linalg.inv(S.T @ S + beta * I) @ S.T @ ratingsMatrix
    return V

def belongs_to_group(userId : int, groups : list):
    for g in groups:
        if userId in g:
            return groups.index(g)

    # Did not find userId in groups
    print(f'Did not find userId: {userId} in groups')
    return None

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
            value = 1
            if item1 != item2:
                value += len(ratingsMatrix[np.where((ratingsMatrix[:,item1] >= boundry) * (ratingsMatrix[:,item2] >= boundry))])

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

def load_divrank(filepath):
    pickle_file = open(filepath, "rb")
    return pickle.load(pickle_file)

def create_and_save_divrank(ratingsMatrix : np.ndarray):
    dirname = os.path.dirname(__file__)
    path_to_ranks = os.path.join(dirname, "../data/divrank.pkl")
    # represent network as matrix
    network = colike_itemitem_network(ratingsMatrix, 4)

    ranks = dr.rank(network)
    # save to file
    file = open(path_to_ranks, "wb")
    pickle.dump(ranks, file)
    file.close

    return ranks

def generate_candidate_set(ratingsMatrix):
    dirname = os.path.dirname(__file__)
    path_to_ranks = os.path.join(dirname, "../data/divrank.pkl")
    ranks_exist = os.path.isfile(path_to_ranks)
    if not ranks_exist:
        ranks = create_and_save_divrank(ratingsMatrix)
    else:
        ranks = load_divrank(path_to_ranks)

    return ranks



