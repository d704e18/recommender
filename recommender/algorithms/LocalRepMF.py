import datetime
import os
import pickle
import random
from collections import OrderedDict
import math

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
            test_matrix : np.ndarray,
            number_of_cand_items : int,
            divranks,
            l1 : int, # number of global questions
            l2 : int, # number of local questions
            alpha = 0.01, # regularisation param
            beta = 0.01, # regularisation param
            epochs = 100,
            boundry = 4, # decides when a user likes or dislikes (rating >= boundry)
            k_prime = 20) :

        n_users, n_items = ratingsMatrix.shape

        # randomize V
        print(f"Randomizing V with shape {k_prime} x {n_items}")
        V = np.random.rand(k_prime, n_items)

        # Generate candidate item set
        print("Generating candidate item set")
        # sort candidate set based on divranks
        sorted_cand_items = OrderedDict(sorted(
                divranks.items(),
                key=lambda kv: kv[1],
                reverse=True))

        cand_items = [item for item in sorted_cand_items][:number_of_cand_items]
        # Generating users
        users = []
        for user in range(0, n_users):
            # User must have rated items
            if ratingsMatrix[user].sum() > 0:
                users.append(user)

        list_of_losses = []
        list_of_rmse = []
        list_of_mae = []
        list_of_p1 = []
        list_of_p5 = []
        list_of_p10 = []
        list_of_ndcg10 = []
        for step in range (0, epochs):
            print(f'Starting iteration: {step}')
            #region Generate global questions
            groups, global_questions = growTree(
                ratingsMatrix=ratingsMatrix,
                V=V,
                users=users,
                cand_items=cand_items,
                split_value=4,
                depth=0,
                max_depth=l1,
                alpha=alpha,
                groups=[],
                already_asked_items=[],
                questions=[]
            )
            #endregion

            #region Generate local questions and learn transformation matrices for each group
            local_questions, list_of_Ts = optimizeTransformationMatrices(
                ratingsMatrix=ratingsMatrix,
                V=V,
                groups=groups,
                global_questions=global_questions,
                local_questions_count=l2,
                alpha=0.01)
            #endregion

            #region Optimize V with equation 7
            V_first = V.copy()
            V = optimizeGlobalItemRepMatrix(
                ratingsMatrix=ratingsMatrix,
                groups=groups,
                global_questions=global_questions,
                local_questions=local_questions,
                list_of_transformation_matrices=list_of_Ts,
                n_users=n_users,
                beta = beta
            )
            #endregion

            loss = 0
            for group, gq, lq, t in zip(groups, global_questions, local_questions, list_of_Ts):
                R = ratingsMatrix[group]

                ng = len(group)
                # create B
                U1 = R[:,gq]
                U2 = R[:,lq]
                e = np.ones(shape=(ng, 1))
                B = np.hstack((U1, U2, e))

                predictions = B @ t @ V
                loss += np.linalg.norm(R - predictions) + alpha * np.linalg.norm(t) + beta * np.linalg.norm(V)

            list_of_losses.append(loss)
            print(f'Loss: {loss}')

            rmse, mae, p1, p5, p10, ndcg10 = test_model(test_matrix=test_matrix,
                       global_questions=global_questions,
                       local_questions=local_questions,
                       list_of_Ts=list_of_Ts,
                       V=V)
            list_of_rmse.append(rmse)
            list_of_mae.append(mae)
            list_of_p1.append(p1)
            list_of_p5.append(p5)
            list_of_p10.append(p10)
            list_of_ndcg10.append(ndcg10)

            # Store best model
            if not list_of_losses:
                best_global_questions = global_questions
                best_local_questions = local_questions
                best_list_of_Ts = list_of_Ts
                best_V = V
            elif loss < list_of_losses[step - 1]:
                best_global_questions = global_questions
                best_local_questions = local_questions
                best_list_of_Ts = list_of_Ts
                best_V = V

        return best_global_questions, \
               best_local_questions, \
               best_list_of_Ts, \
               best_V, list_of_losses, list_of_rmse, \
               list_of_mae, list_of_p1, list_of_p5, list_of_p10


def growTree(ratingsMatrix : np.ndarray,
             V : np.ndarray,
             users : list,
             cand_items : list,  # assumes sorted based on divrank
             split_value : int,
             depth : int,
             max_depth : int,  # l1 questions in paper
             groups : list,
             already_asked_items : list,  # items used as global questions
             questions : list,
             alpha = 0.01):

    # leaf node - i.e. do not ask more questions
    if depth >= max_depth:
        groups.append(users)
        questions.append(already_asked_items)

    else:
        best_item = None
        best_loss = sys.maxsize

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

            #region Stuff for eq. 11
            # generating ratings matrices for users who liked and disliked item
            ratingsMatrix_like = ratingsMatrix[likes]
            ratingsMatrix_dislike =  ratingsMatrix[dislikes]

            # generating B = [U1, e]
            u1_like = ratingsMatrix[likes,:][:, already_asked_items + [item]]
            e_users_like = np.ones(shape=(len(likes), 1))
            B_like = np.hstack((u1_like, e_users_like))

            u1_dislike = ratingsMatrix[dislikes,:][:, already_asked_items + [item]]
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
            #endregion

            # Computing loss (||(rating - prediction)||^2 + alpha * ||T||^2
            loss_like = evaluate_eq11(ratingsMatrix_like, B_like, T_like, V, alpha)
            loss_dislike = evaluate_eq11(ratingsMatrix_dislike, B_dislike, T_dislike, V, alpha)
            loss = loss_like + loss_dislike

            if loss < best_loss:
                best_loss = loss
                best_item = item
                users_like = likes
                users_dislike = dislikes

        # add the best item to asked questions
        global_questions = already_asked_items.copy()
        global_questions.append(best_item)

        # region Splitting
        # Not a leaf node
        cand_items.remove(best_item) # I cand - i*

        print(f'Splitting on item: {best_item}')

        # likes
        growTree(
            ratingsMatrix = ratingsMatrix,
            V = V,
            users = users_like,
            cand_items = cand_items,
            split_value = split_value,
            depth = depth + 1,
            max_depth = max_depth,
            groups=groups,
            questions=questions,
            already_asked_items=global_questions)

        # dislikes
        growTree(
            ratingsMatrix = ratingsMatrix,
            V=V,
            users=users_dislike,
            cand_items=cand_items,
            split_value=split_value,
            depth=depth + 1,
            max_depth=max_depth,
            groups=groups,
            questions=questions,
            already_asked_items=global_questions)

        cand_items.append(best_item) # I cand + i*
        #endregion


    return groups, questions

def evaluate_eq11(R : np.ndarray,
                  B : np.ndarray,
                  T : np.ndarray,
                  V : np.ndarray,
                  alpha : int):
    predictions = B @ T @ V
    # using frobenius norm
    return np.linalg.norm(R[R != 0] - predictions[R != 0]) + alpha * np.linalg.norm(T) # eq. 11


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
        if group is None:
            continue
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

def to_matrix(ratings : pd.DataFrame, n_users : int, n_items : int):
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

def create_and_save_divrank(filename, ratingsMatrix : np.ndarray, boundry : int):
    dirname = os.path.dirname(__file__)
    path_to_ranks = os.path.join(dirname, f'../data/{filename}.pkl')
    # represent network as matrix
    network = colike_itemitem_network(ratingsMatrix, boundry)

    ranks = dr.rank(network)
    # save to file
    file = open(path_to_ranks, "wb")
    pickle.dump(ranks, file)
    file.close

    return ranks

def generate_candidate_set(filename, ratingsMatrix, boundry : int):
    dirname = os.path.dirname(__file__)
    path_to_ranks = os.path.join(dirname, f'../data/{filename}.pkl')
    ranks_exist = os.path.isfile(path_to_ranks)
    if not ranks_exist:
        ranks = create_and_save_divrank(filename, ratingsMatrix, boundry)
    else:
        ranks = load_divrank(path_to_ranks)

    return ranks


def test_user_belongs_to_group(ratings, userId, global_questions, question_number = 0, number_of_dislikes = 0):
    n_groups = len(global_questions)

    if n_groups == 1:
        return number_of_dislikes
    else:
        question = global_questions[0][question_number]
        answer = ratings[question]
        split_question_idx = int(n_groups / 2)

        if answer >= 4:
            return test_user_belongs_to_group(
                ratings, userId, global_questions[:split_question_idx],
                question_number + 1, number_of_dislikes)
        else:
            return test_user_belongs_to_group(
                ratings, userId, global_questions[split_question_idx:],
                question_number + 1, number_of_dislikes + split_question_idx)

def test_model(test_matrix : np.ndarray,
               global_questions : list,
               local_questions : list,
               list_of_Ts : list,
               V: np.ndarray):
    root_mean_squared_error = 0
    mean_abs_error = 0
    precision_1 = 0
    precision_5 = 0
    precision_10 = 0
    ndcg_at_10 = 0

    n_users, _ = test_matrix.shape
    # making list of test-user idx
    users = []
    for test_user in range(n_users):
        if sum(test_matrix[test_user] > 0):
            users.append(test_user)
    # preparing dicts with users
    users_dict : dict = {}
    for idx in range(len(global_questions)):
        users_dict[idx] = []

    # adding test users to right groups
    for test_user in users:
        user_ratings = test_matrix[test_user]
        group_id = test_user_belongs_to_group(user_ratings, test_user,
                                              global_questions)
        users_dict[group_id] += [test_user]

    for group, gq, lq, t in zip(users_dict.values(), global_questions, local_questions, list_of_Ts):
        R = test_matrix[group]

        ng = len(group)
        # create B
        U1 = R[:, gq]
        U2 = R[:, lq]
        e = np.ones(shape=(ng, 1))
        B = np.hstack((U1, U2, e))

        predictions = B @ t @ V
        root_mean_squared_error += rmse(predictions, R)
        mean_abs_error += mae(predictions, R)
        precision_1 += precision_at(1, predictions, R)
        precision_5 += precision_at(5, predictions, R)
        precision_10 += precision_at(10, predictions, R)
        ndcg_at_10 += ndcg_at_k(10, predictions, R)

    n_groups = np.sum([bool(x) for x in users_dict.values()])

    return root_mean_squared_error/n_groups, \
           mean_abs_error/n_groups, \
           precision_1/n_groups, \
           precision_5/n_groups, \
           precision_10/n_groups, \
           ndcg_at_10/n_groups

def precision_at(k : int,
                 predictions,
                 actuals,
                 relevant_boundry : int = 4):
    n_users, _ = actuals.shape

    if n_users == 0:
        return 0

    res = 0
    for user in range(n_users):
        sorted_predictions_idx = predictions[user].argsort()[::-1]
        top_k_predictions_idx = sorted_predictions_idx[:k]

        actual_row = actuals[user]
        tp = np.sum(actual_row[top_k_predictions_idx] >= relevant_boundry)

        res += tp / k

    return res/n_users

def rmse(predictions, actuals):
    n = len(predictions[actuals != 0])
    if n == 0:
        return 0
    error = np.square(predictions[actuals != 0] - actuals[actuals != 0])
    return np.sqrt(np.sum(error)/n)

def mae(predictions, actuals):
    n = len(predictions[actuals != 0])
    if n == 0:
        return 0
    error = np.abs(predictions[actuals != 0] - actuals[actuals != 0])
    return (np.sum(error) / n)

def dcg_at_k(k : int, relevant_boundry : int, predictions, actuals):
    n_users, _ = actuals.shape

    res = 0
    for user in range(n_users):
        sorted_predictions_idx = predictions[user].argsort()[::-1]
        top_k_predictions_idx = sorted_predictions_idx[:k]

        actual_row = actuals[user]
        user_res = 0
        for item, index in zip(top_k_predictions_idx, list(range(1, k + 1))):
            rel_i = 1 if actual_row[item] >= relevant_boundry else 0

            user_res += rel_i / math.log(index + 1, 2)

        res += user_res

    return res / n_users

def idcg_at_k(k : int, relevant_boundry : int, actuals):
    n_users, _ = actuals.shape

    res = 0
    for user in range(n_users):
        sorted_actual_row = np.sort(actuals[user])[::-1]

        user_res = 0
        for actual_value, index in zip(sorted_actual_row, list(range(1, k + 1))):
            rel_i = 1 if actual_value >= relevant_boundry else 0

            user_res += rel_i / math.log(index + 1, 2)

        res += user_res

    return res / n_users

def ndcg_at_k(k : int, predictions, actuals, relevant_boundry : int = 4):
    n_users, _ = actuals.shape

    if n_users == 0:
        return 0

    return dcg_at_k(k, relevant_boundry, predictions, actuals) / \
           idcg_at_k(k, relevant_boundry, actuals)





