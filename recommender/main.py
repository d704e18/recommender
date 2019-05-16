from algorithms.DivRank import DivRank
from algorithms.LocalRepMF import *
import pickle
import os

def generate_cold_start_data(path_to_data : str, path_to_new_data : str, split_percentage : float):
    data = pd.read_csv(path_to_data, delimiter="\t", header=None)  # first 2 users

    users = data[0].unique()
    n_users = len(users)
    n_test_users = round(n_users * split_percentage)
    n_train_users = n_users - n_test_users
    random.shuffle(users)

    test_users = users[:n_test_users]
    train_users = users[-n_train_users:]

    test_data = data.loc[data[0].isin(test_users)]
    train_data = data.loc[data[0].isin(train_users)]

    path_to_train = os.path.join(path_to_new_data, 'train.data')
    train_file = open(path_to_train, 'wb')
    train_data.to_csv(path_to_train, index=None, header=False)
    train_file.close()
    path_to_test = os.path.join(path_to_new_data, 'test.data')
    test_file = open(path_to_test, 'wb')
    test_data.to_csv(path_to_test, index=None, header=False)
    test_file.close()


if __name__ == '__main__':
        dirname = os.path.dirname(__file__)
        path_to_data = os.path.join(dirname, 'ml-100k/u.data')
        path_to_new_data = os.path.join(dirname, 'ml-100k')
        generate_cold_start_data(path_to_data, path_to_new_data, 0.3)

        train_path = os.path.join(path_to_new_data, 'train.data')
        train_data = pd.read_csv(train_path, delimiter=',', header=None)
        test_path = os.path.join(path_to_new_data, 'test.data')
        test_data = pd.read_csv(test_path, delimiter=',', header=None)

        n_users = max(train_data[0].max(), test_data[0].max())
        n_items = max(train_data[1].max(), test_data[1].max())

        train_matrix = to_matrix(train_data, n_users, n_items)
        test_matrix = to_matrix(test_data, n_users, n_items)

        LRMF = LocalRepMF()
        U1, U2, Ts, V, losses = LRMF.fit(
            ratingsMatrix=train_matrix,
            number_of_cand_items=50,
            l1=2,
            l2=2,
            epochs = 100,
            boundry = 4,  # decides when a user likes or dislikes (rating >= boundry)
            k_prime = 20)

        print("DONATELLO")
