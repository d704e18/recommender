from algorithms.LocalRepMF import *
import os
import matplotlib.pyplot as plt

def generate_cold_start_data(path_to_data : str, path_to_new_data : str, splits : int):
    data = pd.read_csv(path_to_data, delimiter="\t", header=None)  # first 2 users

    users = data[0].unique()
    n_users = len(users)
    n_test_users_per_split = round(n_users * 1/splits)
    random.shuffle(users)

    for split in range(splits):
        start_split_idx = n_test_users_per_split * split
        end_split_idx = n_test_users_per_split * (split + 1)
        if end_split_idx > n_users:
            end_split_idx = n_users
        test_users = users[start_split_idx:end_split_idx]

        # finding data for users in test and train
        test_data = data.loc[data[0].isin(test_users)]
        train_data = data.loc[~data[0].isin(test_users)]

        # saving to csv file
        path_to_train = os.path.join(path_to_new_data, f'train{split}.data')
        train_file = open(path_to_train, 'wb')
        train_data.to_csv(path_to_train, index=None, header=False)
        train_file.close()
        path_to_test = os.path.join(path_to_new_data, f'test{split}.data')
        test_file = open(path_to_test, 'wb')
        test_data.to_csv(path_to_test, index=None, header=False)
        test_file.close()


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    # path_to_data = os.path.join(dirname, 'ml-100k/u.data')
    path_to_new_data = os.path.join(dirname, 'ml-100k')


    # region Train
    # for split in range(5):
    #     train_path = os.path.join(path_to_new_data, f'train{split}.data')
    #     train_data = pd.read_csv(train_path, delimiter=',', header=None)
    #     test_path = os.path.join(path_to_new_data, f'test{split}.data')
    #     test_data = pd.read_csv(test_path, delimiter=',', header=None)
    #
    #     n_users = max(train_data[0].max(), test_data[0].max())
    #     n_items = max(train_data[1].max(), test_data[1].max())
    #
    #     train_matrix = to_matrix(train_data, n_users, n_items)
    #     test_matrix = to_matrix(test_data, n_users, n_items)
    #
    #     divranks = generate_candidate_set(f'divrank_{split}', train_matrix, 4)
    #     LRMF = LocalRepMF()
    #     U1, U2, Ts, V, losses, rmses, maes, p1s, p5s, p10s = LRMF.fit(
    #          divranks=divranks,
    #          ratingsMatrix=train_matrix,
    #          test_matrix=test_matrix,
    #          number_of_cand_items=50,
    #          l1=2,
    #          l2=2,
    #          epochs = 50,
    #          boundry = 4,  # decides when a user likes or dislikes (rating >= boundry)
    #          k_prime = 20)
    #
    #     with open(f'data/u1_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(U1, fp)
    #     with open(f'data/u2_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(U2, fp)
    #     with open(f'data/Ts_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(Ts, fp)
    #     with open(f'data/V_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(V, fp)
    #     with open(f'data/losses_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(losses, fp)
    #     with open(f'data/RMSE_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(rmses, fp)
    #     with open(f'data/MAE_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(maes, fp)
    #     with open(f'data/P1_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(p1s, fp)
    #     with open(f'data/P5_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(p5s, fp)
    #     with open(f'data/P10_{split}.txt', "wb") as fp:  # Pickling
    #         pickle.dump(p10s, fp)
    # endregion

    print("DONATELLO")

    # region Plot of test
    list_of_mae = np.array([])
    list_of_rmse = np.array([])
    list_of_p1 = np.array([])
    list_of_p5 = np.array([])
    list_of_p10 = np.array([])
    list_of_losses = np.array([])
    summed_ndcg = 0
    summed_rmse = 0
    summed_mae = 0
    summed_p1 = 0
    summed_p5 = 0
    summed_p10 = 0
    for split in range(5):
        # region Train metrics
        with open(f"data/losses_{split}.txt", "rb") as fp:  # Unpickling
            if list_of_losses.size == 0:
                list_of_losses = np.array(pickle.load(fp))
            else:
                list_of_losses = np.add(list_of_losses, pickle.load(fp))
        with open(f"data/MAE_{split}.txt", "rb") as fp:  # Unpickling
            if list_of_mae.size == 0:
                list_of_mae = np.array(pickle.load(fp))
            else:
                list_of_mae = np.add(list_of_mae, pickle.load(fp))
        with open(f"data/RMSE_{split}.txt", "rb") as fp:  # Unpickling
            if list_of_rmse.size == 0:
                list_of_rmse = np.array(pickle.load(fp))
            else:
                list_of_rmse = np.add(list_of_rmse, pickle.load(fp))
        with open(f"data/P1_{split}.txt", "rb") as fp:  # Unpickling
            if list_of_p1.size == 0:
                list_of_p1 = np.array(pickle.load(fp))
            else:
                list_of_p1 = np.add(list_of_p1, pickle.load(fp))
        with open(f"data/P5_{split}.txt", "rb") as fp:  # Unpickling
            if list_of_p5.size == 0:
                list_of_p5 = np.array(pickle.load(fp))
            else:
                list_of_p5 = np.add(list_of_p5, pickle.load(fp))
        with open(f"data/P10_{split}.txt", "rb") as fp:  # Unpickling
            if list_of_p10.size == 0:
                list_of_p10 = np.array(pickle.load(fp))
            else:
                list_of_p10 = np.add(list_of_p10, pickle.load(fp))
        # endregion

        with open(f"data/V_{split}.txt", "rb") as fp:
            V = pickle.load(fp)
        with open(f"data/Ts_{split}.txt", "rb") as fp:
            Ts = pickle.load(fp)
        with open(f"data/U1_{split}.txt", "rb") as fp:
            U1 = pickle.load(fp)
        with open(f"data/U2_{split}.txt", "rb") as fp:
            U2 = pickle.load(fp)

        train_data = pd.read_csv(f"ml-100k/train{split}.data", delimiter=',', header=None)
        test_data = pd.read_csv(f"ml-100k/test{split}.data", delimiter=',', header=None)

        n_users = max(train_data[0].max(), test_data[0].max())
        n_items = max(train_data[1].max(), test_data[1].max())
        test_matrix = to_matrix(test_data, n_users, n_items)

        rmse, mae, p1, p5, p10, ndcg10 = test_model(test_matrix, U1, U2, Ts, V)
        summed_ndcg += ndcg10
        summed_mae += mae
        summed_rmse += rmse
        summed_p1 += p1
        summed_p5 += p5
        summed_p10 += p10

    list_of_mae = list_of_mae / 5
    list_of_rmse = list_of_rmse / 5
    list_of_p1 = list_of_p1 / 5
    list_of_p5 = list_of_p5 / 5
    list_of_p10 = list_of_p10 / 5
    list_of_losses = list_of_losses / 5
    print(f'ndcg@10: {summed_ndcg / 5}')
    print(f'p1: {summed_p1 / 5}')
    print(f'p5: {summed_p5 / 5}')
    print(f'p10: {summed_p10 / 5}')
    print(f'RMSE: {summed_rmse / 5}')
    print(f'MAE: {summed_mae / 5}')

    epochs_list = np.array(range(50))
    plt.plot(epochs_list, list_of_losses, 'b--')

    #plt.plot(epochs_list, list_of_mae, 'r--')
    p1, p2, p3 = plt.plot(epochs_list, list_of_p1, 'b--',
            epochs_list, list_of_p5, 'y--',
            epochs_list, list_of_p10, 'g--')
    plt.axis([0, 50, 0, 1])
    plt.xlabel("epochs")
    plt.ylabel("precision")
    plt.legend((p1, p2, p3), ('P@1', 'P@5', 'P@10'))
    plt.show()
    #endregion


